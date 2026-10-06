"""같은 문구를 10개 샘플로 렌더링해 정착 화면이 서로 구분되는지 검사한다.

등장 0.2초에만 차이가 있고 유지 구간이 같으면 사용자는 어떤 샘플을 골라도 같은 결과로 본다.
검사용 산출물이며 납품 영상 대신 쓰지 않는다.
"""
import argparse
import importlib.util
import itertools
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SKILL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kinetic', SKILL / 'scripts/2026-10-05-kinetic-engine.py')
kinetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kinetic)
TEXT = 'AI 수업\n\n학생이 직접 질문을 만든다\n\n탐구\n\n데이터로 근거를 찾는다\n\n함께 성장'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='대비표 PNG를 저장할 기존 폴더')
    parser.add_argument('--threshold', type=float, default=5.0, help='샘플 쌍의 최소 평균 화소 차이(0~255)')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_dir():
        raise SystemExit('출력 폴더가 없습니다.')
    if not shutil.which('ffmpeg'):
        raise SystemExit('FFmpeg가 필요합니다. SKILL.md의 런타임 준비를 먼저 하세요.')
    width, height = 480, 270
    settled, rows = {}, []
    for preset in kinetic.PROFILES:
        scenes, duration = kinetic.make_timeline(TEXT, preset, None, 'tight')
        project = {'scenes': scenes, 'duration': duration, 'preset': preset}
        kinetic.bind_rhythm(project, {'events': []})
        scene = scenes[1]
        times = [scene['start'] + scene['duration'] * f for f in (.06, .15, .3)] + [scene['start'] + scene['duration'] * .6]
        images = [kinetic.project_frame(project, t, width, height) for t in times]
        settled[preset] = np.asarray(images[-1]).astype(float)
        rows.append(np.hstack([np.asarray(im) for im in images]))
    sheet = output / 'sample-signature-sheet.png'
    if sheet.exists():
        raise SystemExit(f'기존 파일을 덮어쓰지 않습니다: {sheet}')
    Image.fromarray(np.vstack(rows)).save(sheet)
    pairs = sorted((float(np.abs(settled[a] - settled[b]).mean()), a, b) for a, b in itertools.combinations(settled, 2))
    print('행 순서:', ', '.join(f'{kinetic.PROFILES[p][0]}({p})' for p in settled))
    print('가장 비슷한 정착 화면 5쌍:')
    for value, a, b in pairs[:5]:
        print(f'  {value:6.2f}  {a} / {b}')
    print('대비표:', sheet)
    failed = [pair for pair in pairs if pair[0] < args.threshold]
    if failed:
        raise SystemExit(f'구분 부족 {len(failed)}쌍: 정착 화면이 너무 비슷합니다. scripts/samples의 해당 샘플 모듈을 확인하세요.')
    print('통과: 모든 샘플 쌍의 정착 화면이 기준 이상 다릅니다. 미적 유사성 판단은 대비표를 직접 보고 한다.')


if __name__ == '__main__':
    main()
