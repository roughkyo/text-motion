"""원본 샘플을 모든 프레임 단위로 분해해 텍스트·도형·배경 변화를 측정한다.

산출물
- <id>-frames.csv : 프레임별 수치(전경 영역, 위치·크기, 선명도, 전경/배경 색, 변화량)
- <id>-events.json : 컷·전환·정착 후보 구간
- sheets/frames-NNN.png : 원본 프레임률 그대로의 1프레임 단위 연속 시트.
  글자·도형이 변하는 프레임은 모두 싣고, 변화 없는 정지 구간만 한 칸으로 묶어 범위를 표시한다.

수치는 판독 보조다. 글자·도형의 의미와 질감은 시트를 직접 보고 기록한다.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SKILL = Path(__file__).resolve().parents[1]
W, H = 640, 360


def probe(path):
    out = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                              'stream=r_frame_rate,nb_frames,width,height:format=duration',
                                              '-of', 'json', str(path)]))
    num, den = map(int, out['streams'][0]['r_frame_rate'].split('/'))
    return num / den, out


def frames(path):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-an', '-vf', f'scale={W}:{H}:flags=area',
                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE)
    size = W * H * 3
    try:
        while True:
            data = process.stdout.read(size)
            if len(data) < size:
                break
            yield np.frombuffer(data, np.uint8).reshape(H, W, 3)
    finally:
        process.kill()
        process.wait()


def measure(rgb, previous, previous_mask):
    f = rgb.astype(np.float32)
    luma = f @ np.array([.299, .587, .114], np.float32)
    border = np.concatenate([f[:18].reshape(-1, 3), f[-18:].reshape(-1, 3), f[:, :24].reshape(-1, 3), f[:, -24:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # 작은 반경의 국소 대비로 날카로운 글자·도형 획만 잡는다. 연기·그라데이션처럼 부드러운 배경은 빠진다.
    # 큰 글자는 외곽선 위주로 잡히므로 면적이 아니라 위치·크기·변화 판정용으로 쓴다.
    local = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(5)), np.float32)
    contrast = np.abs(f - local).max(axis=2)
    mask = contrast > 45
    area = float(mask.mean())
    # 블러 상태로 서서히 나타나는 글자는 획 경계가 없어 마스크에 안 잡힌다. 중앙 영역 밝기로 따로 감지한다.
    center = float(luma[H // 4:H * 3 // 4, W // 6:W * 5 // 6].mean())
    row = {'luma': float(luma.mean()), 'center_luma': center, 'bg_r': float(bg[0]), 'bg_g': float(bg[1]), 'bg_b': float(bg[2]),
           'fg_area': area, 'diff': float(np.abs(f - previous).mean()) if previous is not None else 0.0,
           'mask_change': 1.0, 'fg_diff': 255.0}
    if previous is not None:
        # 배경(연기 등)이 계속 흘러도 글자·도형이 멈췄는지 판정하도록 전경 영역만 비교한다.
        row['mask_change'] = float((mask ^ previous_mask).mean())
        # 경계 1픽셀 흔들림을 제외하려고 두 프레임 모두 전경인 화소의 색 변화만 잰다. 모양 변화는 mask_change가 맡는다.
        both = mask & previous_mask
        row['fg_diff'] = float(np.abs(f - previous)[both].mean()) if both.any() else 0.0
    if area > .0004:
        ys, xs = np.nonzero(mask)
        x0, x1 = np.percentile(xs, [1, 99])
        y0, y1 = np.percentile(ys, [1, 99])
        fg = f[mask]
        hsv = np.asarray(Image.fromarray(rgb).convert('HSV'), np.float32)[mask]
        lap = np.abs(4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:])
        inside = mask[1:-1, 1:-1]
        # 글자 줄 수: 전경 행 투영에서 연속 띠의 수
        rows = mask.mean(axis=1) > .004
        bands = int(np.sum(rows[1:] & ~rows[:-1]) + rows[0])
        row.update(x0=x0 / W, x1=x1 / W, y0=y0 / H, y1=y1 / H, cx=float(xs.mean() / W), cy=float(ys.mean() / H),
                   fg_r=float(np.median(fg[:, 0])), fg_g=float(np.median(fg[:, 1])), fg_b=float(np.median(fg[:, 2])),
                   fg_sat=float(np.median(hsv[:, 1])), sharp=float(lap[inside].mean()) if inside.any() else 0.0,
                   bands=bands)
    return row, f, mask


def events(rows, fps):
    diff = np.array([r['diff'] for r in rows])
    area = np.array([r['fg_area'] for r in rows])
    luma = np.array([r['luma'] for r in rows])
    width = np.array([r.get('x1', 0) - r.get('x0', 0) for r in rows])
    result = []
    for i in range(1, len(rows)):
        reasons = []
        if diff[i] > max(18, 4 * np.median(diff[max(0, i - 15):i + 1]) + 4):
            reasons.append('cut_or_flash')
        if abs(luma[i] - luma[i - 1]) > 40:
            reasons.append('bg_polarity')
        if area[i - 1] < .0004 <= area[i]:
            reasons.append('fg_appear')
        if area[i - 1] >= .0004 > area[i]:
            reasons.append('fg_clear')
        if width[i - 1] > 0 and width[i] > 0 and abs(width[i] - width[i - 1]) > .18:
            reasons.append('layout_jump')
        if reasons:
            result.append({'frame': i, 'time': round(i / fps, 4), 'reasons': reasons})
    # 같은 전환을 이루는 인접 이벤트는 묶는다.
    merged = []
    for event in result:
        if merged and event['frame'] - merged[-1]['frame_end'] <= 2:
            merged[-1]['frame_end'] = event['frame']
            merged[-1]['reasons'] = sorted(set(merged[-1]['reasons']) | set(event['reasons']))
        else:
            merged.append({'frame': event['frame'], 'frame_end': event['frame'], 'time': event['time'], 'reasons': event['reasons']})
    # 정착 구간: 변화량이 작게 유지되는 연속 프레임
    still, start = [], None
    for i, value in enumerate(diff):
        quiet = value < 1.2 and area[i] >= .0004
        if quiet and start is None:
            start = i
        if (not quiet or i == len(diff) - 1) and start is not None:
            if i - start >= max(3, round(fps * .15)):
                still.append({'start_frame': start, 'end_frame': i, 'start': round(start / fps, 4), 'end': round(i / fps, 4)})
            start = None
    return merged, still


def sheet(images, labels, columns, path, cell=(213, 120)):
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 12) if Path('C:/Windows/Fonts/malgun.ttf').exists() else ImageFont.load_default()
    rows = (len(images) + columns - 1) // columns
    board = Image.new('RGB', (columns * cell[0], rows * (cell[1] + 16)), (24, 24, 30))
    draw = ImageDraw.Draw(board)
    for k, (im, label) in enumerate(zip(images, labels)):
        x, y = (k % columns) * cell[0], (k // columns) * (cell[1] + 16)
        board.paste(Image.fromarray(im).resize(cell, Image.Resampling.BILINEAR), (x, y + 16))
        draw.text((x + 3, y + 1), label, font=font, fill=(235, 235, 245))
    board.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, help='예: 11971627')
    parser.add_argument('--data-dir', required=True, help='CSV·JSON 저장 폴더')
    parser.add_argument('--sheet-dir', required=True, help='접촉 시트 저장 폴더')
    args = parser.parse_args()
    source = SKILL / 'assets/motion_graphic_sample' / f'{args.sample}.mp4'
    fps, meta = probe(source)
    data_dir, sheet_dir = Path(args.data_dir), Path(args.sheet_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows, keep, previous, previous_mask = [], [], None, None
    for index, rgb in enumerate(frames(source)):
        row, previous, previous_mask = measure(rgb, previous, previous_mask)
        row = {'frame': index, 'time': round(index / fps, 4), **row}
        rows.append(row)
        keep.append(rgb)
    fields = ['frame', 'time', 'luma', 'center_luma', 'diff', 'mask_change', 'fg_diff', 'bg_r', 'bg_g', 'bg_b', 'fg_area', 'x0', 'x1', 'y0', 'y1', 'cx', 'cy',
              'fg_r', 'fg_g', 'fg_b', 'fg_sat', 'sharp', 'bands']
    with (data_dir / f'{args.sample}-frames.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    found, settled = events(rows, fps)
    (data_dir / f'{args.sample}-events.json').write_text(json.dumps(
        {'sample': args.sample, 'fps': fps, 'frames': len(rows), 'source_meta': meta, 'analysis_size': [W, H],
         'events': found, 'settled': settled}, ensure_ascii=False, indent=2), encoding='utf-8')
    # 1프레임 단위 시트. 정지 판정: 전경 마스크 변화·전경 화소 변화·전체 밝기 변화가 모두 작을 때.
    still = [i > 0 and rows[i]['mask_change'] < .0022 and rows[i]['fg_diff'] < 6 and abs(rows[i]['luma'] - rows[i - 1]['luma']) < 2
             and abs(rows[i]['center_luma'] - rows[i - 1]['center_luma']) < .35
             for i in range(len(rows))]
    tiles, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and still[j + 1]:
            j += 1
        tiles.append((i, f'f{i} {i / fps:.3f}s'))
        if j - i >= 2:
            tiles.append((j, f'정지 f{i + 1}~f{j} ({(j - i) / fps:.2f}s)'))
        elif j > i:
            tiles.extend((k, f'f{k} {k / fps:.3f}s') for k in range(i + 1, j + 1))
        i = j + 1
    per = 36
    for page in range(0, len(tiles), per):
        chunk = tiles[page:page + per]
        sheet([keep[k] for k, _ in chunk], [label for _, label in chunk], 6,
              sheet_dir / f'frames-{page // per + 1:03d}.png', cell=(320, 180))
    print(json.dumps({'sample': args.sample, 'fps': round(fps, 3), 'frames': len(rows), 'events': len(found),
                      'tiles': len(tiles), 'frame_sheets': (len(tiles) + per - 1) // per}, ensure_ascii=False))


if __name__ == '__main__':
    main()
