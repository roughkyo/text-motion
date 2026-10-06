"""시간·공간 필터로 원본 문구를 제거한 연속 연기 배경. 원본은 수정하지 않는다."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def clean(plate):
    # 중앙 띠만 합성하면 어두운 박스처럼 보인다. 전체에 같은 처리를 적용해 경계를 없앤다.
    return np.clip(plate, 0, 255).astype(np.uint8).tobytes()


def main():
    for identifier, color in [('60608725','cool'),('60806086','purple')]:
        output = ASSETS/f'2026-10-05-smoke-{color}-continuous-clean-v4.mp4'
        if output.exists():
            raise FileExistsError(output)
        source = ASSETS/'motion_graphic_sample'/(identifier+'.mp4')
        # 중앙을 선형 그라데이션으로 덮으면 세로 얼룩이 생긴다. 주변 시간의 실제 연기 질감을 사용한다.
        raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(source),'-an','-vf','scale=640:360,fps=4','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        sampled = np.frombuffer(raw, dtype=np.uint8).reshape(-1,360,640,3)
        plates = []
        for index in range(len(sampled)):
            window = sampled[max(0,index-12):min(len(sampled),index+13)]
            plate = Image.fromarray(np.percentile(window, 12, axis=0).astype(np.uint8))
            # 끝부분처럼 같은 문구가 오래 남는 구간의 잔재도 제거한다. 중앙 질감은 부드러워진다.
            plate = plate.filter(ImageFilter.MinFilter(17)).filter(ImageFilter.GaussianBlur(4))
            plates.append(np.asarray(plate).astype(np.float32))
        decoder = subprocess.Popen(['ffmpeg','-v','error','-i',str(source),'-an','-vf','scale=640:360,fps=30','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
        encoder = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',str(output)],stdin=subprocess.PIPE)
        frame_index = 0
        try:
            while True:
                data = decoder.stdout.read(640*360*3)
                if not data:
                    break
                if len(data) != 640*360*3:
                    raise ValueError('불완전한 원본 프레임')
                phase = frame_index/30*4
                a = min(int(phase), len(plates)-1)
                b = min(a+1, len(plates)-1)
                q = phase-int(phase)
                plate = plates[a]*(1-q)+plates[b]*q
                encoder.stdin.write(clean(plate))
                frame_index += 1
            encoder.stdin.close()
            assert decoder.wait()==0 and encoder.wait()==0
        finally:
            for process in (decoder,encoder):
                if process.poll() is None:
                    process.kill()
                    process.wait()
        print(output.name,flush=True)


if __name__ == '__main__':
    main()
