"""지정 구간을 원본 해상도로 분해해 요소 단위 변화를 측정한다(분석 2·3단계).

- 2단계: 글자 영역을 원본 해상도로 잘라 모든 프레임을 정지 묶음 없이 펼친 시트
- 3단계: 프레임별 수치 곡선(글자 높이·폭·중심, 위/아래 색, 선명도, 열별 선명도 분포)

사용 예: --sample 11971627 --start 90 --end 119 --label active
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SKILL = Path(__file__).resolve().parents[1]


def probe(path):
    out = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                              'stream=width,height,r_frame_rate', '-of', 'json', str(path)]))
    stream = out['streams'][0]
    num, den = map(int, stream['r_frame_rate'].split('/'))
    return stream['width'], stream['height'], num / den


def decode(path, start, end, fps, width, height):
    # 원본 해상도 그대로, 프레임 번호 기준으로 정확히 자른다.
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-an', '-vf',
                                f'select=between(n\\,{start}\\,{end})', '-fps_mode', 'passthrough', '-f', 'rawvideo',
                                '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE)
    size = width * height * 3
    frames = []
    while True:
        data = process.stdout.read(size)
        if len(data) < size:
            break
        frames.append(np.frombuffer(data, np.uint8).reshape(height, width, 3))
    process.wait()
    return frames


def text_mask(rgb, scale):
    f = rgb.astype(np.float32)
    local = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(max(2, 5 * scale))), np.float32)
    return np.abs(f - local).max(axis=2) > 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sheet-dir', required=True)
    parser.add_argument('--crop', help='고정 크롭 x0,y0,x1,y1 (원본 비율 0~1). 없으면 구간 전체 글자 영역 합집합')
    args = parser.parse_args()
    source = SKILL / 'assets/motion_graphic_sample' / f'{args.sample}.mp4'
    width, height, fps = probe(source)
    frames = decode(source, args.start, args.end, fps, width, height)
    scale = width / 640
    margin_y = int(height * .08)  # 상하 상시 레이어(로고·하단 문단) 제외
    masks = [text_mask(rgb, scale) for rgb in frames]
    for m in masks:
        m[:margin_y] = False
        m[-int(height * .1):] = False
    if args.crop:
        x0, y0, x1, y1 = (float(v) for v in args.crop.split(','))
        box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    else:
        union = np.zeros_like(masks[0])
        for m in masks:
            union |= m
        ys, xs = np.nonzero(union)
        if len(xs) == 0:
            box = (0, 0, width, height)
        else:
            pad = int(height * .04)
            box = (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)), int(min(width, xs.max() + pad)), int(min(height, ys.max() + pad)))
    rows = []
    for k, (rgb, m) in enumerate(zip(frames, masks)):
        index = args.start + k
        row = {'frame': index, 'time': round(index / fps, 4)}
        f = rgb.astype(np.float32)
        luma = f @ np.array([.299, .587, .114], np.float32)
        if m.any():
            ys, xs = np.nonzero(m)
            y0, y1 = np.percentile(ys, [1, 99])
            x0, x1 = np.percentile(xs, [1, 99])
            # 색은 경계가 아니라 배경과 충분히 다른 글자 내부 화소에서 잰다.
            border_bg = np.median(np.concatenate([f[margin_y:margin_y + 10].reshape(-1, 3), f[-int(height * .1) - 10:-int(height * .1)].reshape(-1, 3)]), axis=0)
            inside = np.zeros((height, width), bool)
            inside[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
            interior = inside & (np.abs(f - border_bg).max(axis=2) > 60)
            rows_idx = np.arange(height)[:, None]
            top = f[interior & (rows_idx < (y0 + (y1 - y0) * .3))]
            bottom = f[interior & (rows_idx > (y0 + (y1 - y0) * .7))]
            row['fill_ratio'] = round(float(interior.sum()) / max(1, inside.sum()), 4)
            lap = np.abs(4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:])
            inner = m[1:-1, 1:-1]
            # 열별 선명도: 글자 영역을 10칸으로 나눠 블러 스윕 진행 위치를 본다.
            columns = []
            for c in range(10):
                a, b = int(x0 + (x1 - x0) * c / 10), int(x0 + (x1 - x0) * (c + 1) / 10)
                band = inner[:, max(0, a - 1):max(1, b - 1)]
                columns.append(round(float(lap[:, max(0, a - 1):max(1, b - 1)][band].mean()), 1) if band.any() else 0)
            row.update(text_h=round((y1 - y0) / height, 4), text_w=round((x1 - x0) / width, 4),
                       cx=round(float(xs.mean()) / width, 4), cy=round(float(ys.mean()) / height, 4),
                       y_top=round(y0 / height, 4), y_bottom=round(y1 / height, 4),
                       top_rgb='#%02x%02x%02x' % tuple(int(v) for v in np.median(top, axis=0)) if len(top) else '',
                       bottom_rgb='#%02x%02x%02x' % tuple(int(v) for v in np.median(bottom, axis=0)) if len(bottom) else '',
                       sharp=round(float(lap[inner].mean()), 2), column_sharp=columns)
        border = np.concatenate([f[margin_y:margin_y + 10].reshape(-1, 3), f[-int(height * .1) - 10:-int(height * .1)].reshape(-1, 3)])
        row['bg_rgb'] = '#%02x%02x%02x' % tuple(int(v) for v in np.median(border, axis=0))
        rows.append(row)
    data_dir, sheet_dir = Path(args.data_dir), Path(args.sheet_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.sample}-detail-{args.label}-f{args.start}-{args.end}'
    (data_dir / f'{name}.json').write_text(json.dumps({'sample': args.sample, 'fps': fps, 'source_size': [width, height],
                                                       'crop_box': box, 'frames': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    # 원본 해상도 크롭을 그대로(최대 폭 600px로만 맞춤) 모든 프레임 펼친다.
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    cell_w = min(600, crop_w)
    cell_h = max(1, round(crop_h * cell_w / crop_w))
    columns = 3 if cell_w > 400 else 4
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 13) if Path('C:/Windows/Fonts/malgun.ttf').exists() else ImageFont.load_default()
    per = columns * max(1, 1800 // (cell_h + 18))
    for page in range(0, len(frames), per):
        chunk = list(range(page, min(len(frames), page + per)))
        board_rows = (len(chunk) + columns - 1) // columns
        board = Image.new('RGB', (columns * cell_w, board_rows * (cell_h + 18)), (24, 24, 30))
        draw = ImageDraw.Draw(board)
        for n, k in enumerate(chunk):
            x, y = (n % columns) * cell_w, (n // columns) * (cell_h + 18)
            crop = Image.fromarray(frames[k]).crop(box).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            board.paste(crop, (x, y + 18))
            r = rows[k]
            draw.text((x + 3, y + 2), f"f{r['frame']} h{r.get('text_h', '-')} w{r.get('text_w', '-')} s{r.get('sharp', '-')}",
                      font=font, fill=(235, 235, 245))
        board.save(sheet_dir / f'{name}-{page // per + 1:02d}.png')
    print(json.dumps({'name': name, 'frames': len(frames), 'crop_box': box, 'sheets': (len(frames) + per - 1) // per}, ensure_ascii=False))


if __name__ == '__main__':
    main()
