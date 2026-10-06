"""60944035: 넓은 초굵은 글자의 흰 채움/남색 외곽선 1프레임 교대, 슬릿 전환, 줌 버스트 (30fps).

근거: references/analysis/60944035-spec.md, 원본 해상도 크롭 f18–40, f50–66, f722–734, f849–862.
장면마다 '프레임별 상태 토큰' 목록을 원본 프레임 순서 그대로 옮겼다.
  F 흰 채움 / O 남색 외곽선 / G1~G6 회색 단계(1 밝음) / SD 위→아래 슬릿(위가 채움) / SU 아래→위 슬릿(아래가 채움)
  D 외곽선이 어두워지며 소멸 / B 빈 화면 / Z 거대 채움(줌 버스트) / z 작은 외곽선(줌 버스트)
같은 문자가 연속되면 연속 구간으로 진행도를 계산한다(SD×5 → 0.2, 0.4, … 1.0).
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

E = None
FPS = 30.0
WHITE = (255, 255, 255)
NAVY = (52, 40, 232)            # 외곽선 측정(#2a1bd8~#3428e8)
GRAYS = {1: (205, 205, 210), 2: (150, 150, 160), 3: (98, 96, 120), 4: (66, 62, 96), 5: (44, 40, 70), 6: (26, 24, 44)}
STRETCH = 1.35                 # 아주 넓은 서체 근사: Pretendard ExtraBold 가로 135%


def T(spec):
    """'O*8 SD*5 F*3' 같은 표기를 프레임별 토큰 목록으로 편다."""
    out = []
    for part in spec.split():
        token, _, count = part.partition('*')
        out.extend([token] * int(count or 1))
    return out


# 장면 템플릿: enter/exit 토큰은 원본 프레임 그대로, hold는 유지 상태, scale은 등장 구간 배율 키프레임
TEMPLATES = [
    {'n': 1, 'frames': 25, 'kind': 'word', 'name': 'INTRODUCING 거대 외곽선 스트로브 급수축',
     'enter': T('O F O F O F G1 G2 G3 G4 G5 O F'), 'scale': [(0, 2.3), (5, 1.9), (9, 1.25), (13, 1.08)],
     'hold': 'F', 'drift': (-.05, 0), 'exit': []},
    {'n': 2, 'frames': 68, 'kind': 'line', 'name': 'BY 외곽선 → 위아래 슬릿 → 외곽선 유지 → 스트로브',
     'enter': T('O*8 SD*5 F*4 SU*4'), 'scale': [(0, 1.12), (8, 1.04), (12, 1.0)],
     'hold': 'O', 'exit': T('F O F O O F O F O F D*3')},
    {'n': 3, 'frames': 26, 'kind': 'line', 'name': 'DO MORE 외곽선 왼쪽 이동 → 스트로브 정착',
     'enter': T('O*8 F O F O F'), 'scale': [(0, 1.0)], 'dx_enter': .025,
     'hold': 'F', 'drift': (0, -.012), 'exit': []},
    {'n': 4, 'frames': 27, 'kind': 'line', 'name': 'THAN 외곽선 → 아래위 슬릿 채움 → 위아래 슬릿 외곽선',
     'enter': T('O*6 SU*4'), 'scale': [(0, 1.0)],
     'hold': 'F', 'exit': T('SD*4 O*4 D*2', )},
    {'n': 5, 'frames': 89, 'kind': 'line', 'name': 'EXPECTED 하드컷 채움 → 천천히 확대 → 밝기 사이클',
     'enter': T('F'), 'scale': [(0, 1.0)],
     'hold': 'F', 'drift': (.06, -.01), 'exit': T('O F O F G1 G2 G3 G2 G1 F O*3 D*3')},
    {'n': 6, 'frames': 20, 'kind': 'word', 'name': 'TEXT 짙은 회색 → 흰색 밝아지며 수축',
     'enter': T('G6 G6 G5 G5 G4 G4 G3 G3 G2 G1 F'), 'scale': [(0, 1.8), (6, 1.25), (10, 1.0)],
     'hold': 'F', 'exit': T('O*5 D*3')},
    {'n': 7, 'frames': 20, 'kind': 'word', 'name': 'ANIMATION 화면보다 넓은 회색 → 흰 수축',
     'enter': T('G6 G6 G5 G5 G4 G4 G3 G3 G2 G1 F'), 'scale': [(0, 2.1), (6, 1.3), (10, 1.0)],
     'hold': 'F', 'exit': T('O*5 D*3')},
    {'n': 8, 'frames': 62, 'kind': 'line', 'name': 'PACK 회색 → 흰 수축 → 외곽선 유지',
     'enter': T('G6 G5 G5 G4 G3 G3 G2 G2 G1 F F'), 'scale': [(0, 1.6), (6, 1.15), (10, 1.0)],
     'hold': 'O', 'exit': T('F O F O F D D')},
    {'n': 9, 'frames': 143, 'kind': 'line', 'name': 'GOD IS LOVE 아주 작은 외곽선 확대 → 슬릿 → 스트로브',
     'enter': T('O*30 SU*3 F*8 SD*4 O*6 F O F O F O F'), 'scale': [(0, .22), (30, 1.0)],
     'hold': 'F', 'exit': T('F O F O F O O D*3')},
    {'n': 10, 'frames': 139, 'kind': 'line', 'name': 'DISCIPLINE 아주 작은 흰 글자 확대 → 슬릿 외곽선',
     'enter': T('F*26 SD*3 O*14 F O F O F O F'), 'scale': [(0, .12), (26, 1.0)],
     'hold': 'O', 'exit': T('SU*10 F*4 SD*7')},
    {'n': 11, 'frames': 89, 'kind': 'word', 'name': 'WAKE UP 연속 확대 → 거대 채움/작은 외곽선 줌 버스트',
     'enter': T('F*20 z Z z Z Z z F'), 'scale': [(0, .1), (20, 1.0)],
     'hold': 'F_flicker', 'exit': T('O*2 D*4')},
    {'n': 12, 'frames': 37, 'kind': 'line', 'name': 'CHOOSE 남색 빛 띠 안에서 커짐 → 잔상 남기며 아래로',
     'enter': T('BAR*9'), 'scale': [(0, .45), (9, 1.0)],
     'hold': 'F', 'drift': (0, 0), 'dy_hold': .025, 'exit': T('ECHO_DOWN*6')},
    {'n': 13, 'frames': 37, 'kind': 'line', 'name': 'GROWTH 위에서 내려오며 이전 잔상을 밀어냄',
     'enter': T('ECHO_IN*7'), 'scale': [(0, 1.0)],
     'hold': 'F', 'drift': (.05, 0), 'exit': T('O*4 D*3')},
    {'n': 14, 'frames': 64, 'kind': 'line', 'name': 'EVERY DAY 하드컷 채움 → 천천히 확대 → 스트로브',
     'enter': T('F'), 'scale': [(0, 1.0)],
     'hold': 'F', 'drift': (.06, 0), 'exit': T('F O F O O F O O D*3')},
    {'n': 15, 'frames': 128, 'kind': 'two', 'name': '두 줄 서로 반대로 채움/외곽선 교대',
     'enter': T('SU*5'), 'scale': [(0, 1.0)],
     'hold': 'SWAP', 'exit': T('SWAPFAST*14 SD*5')},
    {'n': 16, 'frames': 125, 'kind': 'line', 'name': 'MOTION 작은 외곽선 확대 → 슬릿 → 스트로브 → 밝기 사이클',
     'enter': T('O*10 SU*6 F*6 SD*5 O*10 F O F O F O F O F O'), 'scale': [(0, .3), (10, 1.0)],
     'hold': 'F', 'exit': T('F O F G4 G3 G2 G1 G2 F O F O D*2')},
    {'n': 17, 'frames': 60, 'kind': 'end', 'name': 'Thank you 회색/흰/빈 3단 깜빡임(일반 폭)',
     'enter': T('G3 F B G2 B F G3 F F B F G1 F F'), 'scale': [(0, 1.0)],
     'hold': 'F', 'exit': []},
]
LAST = TEMPLATES[-1]


def fit(template, words, chars):
    kind = template['kind']
    if kind == 'word':
        return chars <= 8
    if kind == 'two':
        return len(words) >= 3 and chars >= 9
    if kind == 'end':
        return False
    return True


def assign(paragraphs):
    # INTRODUCING 오프닝(1번)은 첫 문구에만 쓴다.
    if not paragraphs:
        return []
    first = TEMPLATES[0] if fit(TEMPLATES[0], E.split_words(paragraphs[0]), len(paragraphs[0].replace(' ', ''))) else TEMPLATES[1]
    pool = TEMPLATES[1:-1]
    if first in pool:
        i = pool.index(first)
        pool = pool[i + 1:] + pool[:i + 1]
    rest = E.assign_in_order(pool, paragraphs[1:], fit=fit, last=LAST) if len(paragraphs) > 1 else []
    return [first] + rest


def minimum(template, text):
    chars = len(text.replace(' ', ''))
    read = max(.3, chars / 16)
    return (len(template['enter']) + len(template['exit'])) / FPS + read


@lru_cache(maxsize=256)
def masks(text, size, wide=True):
    """(채움 마스크, 외곽선 마스크). 넓은 서체는 가로로 늘린다."""
    f = E.font(size, 'ExtraBold')
    ascent, descent = f.getmetrics()
    w = int(f.getlength(text)) + size // 2
    h = ascent + descent + size // 3
    fill = Image.new('L', (w, h))
    ImageDraw.Draw(fill).text((size // 4, size // 6), text, font=f, fill=255)
    stroke = max(2, round(size * .045))
    grown = Image.new('L', (w, h))
    ImageDraw.Draw(grown).text((size // 4, size // 6), text, font=f, fill=255, stroke_width=stroke // 2 + 1, stroke_fill=255)
    k = max(3, (stroke // 2) * 2 + 1)
    inner = fill.filter(ImageFilter.MinFilter(k))
    outline = ImageChops.subtract(grown, inner)
    if wide:
        nw = int(w * STRETCH)
        fill, outline = fill.resize((nw, h), Image.Resampling.BICUBIC), outline.resize((nw, h), Image.Resampling.BICUBIC)
    box = fill.getbbox() or (0, 0, w, h)
    pad = stroke + 2
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(fill.width, box[2] + pad), min(fill.height, box[3] + pad))
    return fill.crop(box), outline.crop(box)


def colored(mask, color, alpha=1.0):
    im = Image.new('RGBA', mask.size, (*color, 0))
    im.putalpha(mask if alpha >= .999 else mask.point(lambda a: round(a * alpha)))
    return im


def slit(fill_m, outline_m, q, down):
    """슬릿: 경계 위(down) 또는 아래(up)가 흰 채움, 나머지 외곽선. 경계에 얇은 흰 선."""
    w, h = fill_m.size
    y = int(h * q) if down else int(h * (1 - q))
    top = Image.new('L', (w, h))
    ImageDraw.Draw(top).rectangle((0, 0, w, y), fill=255)
    bottom = ImageChops.invert(top)
    filled_region, outline_region = (top, bottom) if down else (bottom, top)
    out = Image.new('RGBA', (w, h))
    out.alpha_composite(colored(ImageChops.multiply(outline_m, outline_region), NAVY))
    out.alpha_composite(colored(ImageChops.multiply(fill_m, filled_region), WHITE))
    line = Image.new('L', (w, h))
    ImageDraw.Draw(line).rectangle((0, y - max(1, h // 60), w, y), fill=255)
    out.alpha_composite(colored(ImageChops.multiply(line, ImageChops.lighter(fill_m, outline_m)), WHITE))
    return out


def token_runs(tokens):
    """각 프레임 토큰의 (같은 토큰 연속 구간 안 순번, 구간 길이)."""
    info, k = [], 0
    while k < len(tokens):
        j = k
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[k]:
            j += 1
        n = j - k + 1
        info.extend((i, n) for i in range(n))
        k = j + 1
    return info


def scale_at(keys, f):
    if f <= keys[0][0]:
        return keys[0][1]
    for (f0, s0), (f1, s1) in zip(keys, keys[1:]):
        if f0 <= f <= f1:
            q = (f - f0) / max(1e-6, f1 - f0)
            return s0 + (s1 - s0) * (1 - (1 - q) ** 2.2)
    return keys[-1][1]


def strobe_hold(f, seed):
    """유지 중 불규칙 깜빡임(WAKE UP 장면): 7~13프레임마다 외곽선 1프레임."""
    rng = np.random.default_rng(seed)
    marks = np.cumsum(rng.integers(7, 14, 40))
    return 'O' if int(f) in set(marks.tolist()) else 'F'


def sprite_for(token, fill_m, outline_m, progress):
    if token == 'F' or token == 'Z' or token == 'BAR':
        return colored(fill_m, WHITE)
    if token in ('O', 'z'):
        return colored(outline_m, NAVY)
    if token.startswith('G'):
        return colored(fill_m, GRAYS[int(token[1:])])
    if token == 'D':
        return colored(outline_m, NAVY, 1 - progress * .85)
    if token == 'SD':
        return slit(fill_m, outline_m, progress, True)
    if token == 'SU':
        return slit(fill_m, outline_m, progress, False)
    if token == 'B':
        return None
    return colored(fill_m, WHITE)


def background(width, height, frame_index):
    """검정 + 수평 디지털 노이즈 조각 + 가끔 1프레임 수평 번쩍임."""
    im = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(im)
    rng = np.random.default_rng(1000 + frame_index)
    s = width / 640
    for _ in range(int(rng.integers(14, 34))):
        y = height * rng.uniform(.18, .82)
        x = width * rng.uniform(.05, .95)
        length = width * rng.choice([.004, .01, .02, .04, .07])
        g = int(rng.integers(40, 120))
        draw.rectangle((x, y, x + length, y + max(1, s * rng.choice([1, 1, 2]))), fill=(g, g, g + 6, 255))
    if rng.random() < .07:
        y = height * rng.uniform(.3, .7)
        draw.rectangle((0, y, width, y + max(1, s)), fill=(235, 235, 240, 255))
    return im


def overlay(im, info):
    """모서리 상시 레이어: 좌상단 주제, 우상단 작은 원, 하단 양쪽 작은 글자."""
    w, h = im.size
    draw = ImageDraw.Draw(im)
    small = E.font(h * .016, 'Bold')
    gray = (150, 150, 158, 255)
    draw.text((w * .03, h * .045), info.get('label', ''), font=small, fill=gray)
    r = h * .012
    draw.ellipse((w * .965 - r, h * .055 - r, w * .965 + r, h * .055 + r), outline=gray, width=max(1, int(h / 540)))
    tiny = E.font(h * .012, 'Regular')
    footer = info.get('footer', '')
    if footer:
        draw.text((w * .03, h * .93), footer[:40], font=tiny, fill=(110, 110, 118, 255))
    draw.text((w * .86, h * .93), info.get('right', 'TEXT MOTION'), font=tiny, fill=(110, 110, 118, 255))


def settled_size(text, width, height, lines=1, wide=True):
    """원본 실측: 정착 캡 높이 약 0.075H(라틴 대문자), 폭은 0.82W 이하. 한글은 글자 높이가 커서 0.09H 근처가 된다."""
    size = int(height * (.105 if wide else .12))
    fill_m, _ = masks(text, size, wide)
    if fill_m.width > width * .82:
        size = max(8, int(size * width * .82 / fill_m.width))
    return size


def render(scene, local, width, height, info):
    tpl = scene['template']
    fps = FPS
    fe = max(1, int(round(scene['duration'] * fps)))
    f = int(local * fps)
    enter, exit_ = tpl['enter'], tpl['exit']
    le, lx = len(enter), len(exit_)
    # 시간이 모자라면(원본보다 짧은 장면) 등장 토큰을 앞부분부터 비율로 압축한다.
    if le + lx > fe:
        ratio = fe / (le + lx)
        le_c, lx_c = max(1, int(le * ratio)), max(0, fe - int(le * ratio))
    else:
        le_c, lx_c = le, lx
    if f < le_c:
        idx = int(f * le / le_c)
        token = enter[idx]
        run_i, run_n = token_runs(enter)[idx]
        tf = idx
    elif lx_c and f >= fe - lx_c:
        idx = min(lx - 1, int((f - (fe - lx_c)) * lx / lx_c))
        token = exit_[idx]
        run_i, run_n = token_runs(exit_)[idx]
        tf = le
    else:
        token, run_i, run_n, tf = tpl['hold'], 0, 1, le
    progress = (run_i + 1) / run_n
    frame_index = int(round(info.get('t', local) * fps))
    image = background(width, height, frame_index)
    text = scene['text']
    lines = scene.get('lines') or None
    if tpl['kind'] == 'two':
        words = E.split_words(text)
        lines = lines if lines and len(lines) == 2 else [' '.join(x) for x in E.balanced_lines(words)]
    elif not lines:
        words = E.split_words(text)
        short = len(text) <= (24 if tpl['kind'] == 'end' else 12)
        lines = [text] if short or tpl['kind'] == 'word' else [' '.join(x) for x in E.balanced_lines(words)]
    wide = tpl['kind'] != 'end'
    size = min(settled_size(ln, width, height, len(lines), wide) for ln in lines)
    scale = scale_at(tpl['scale'], tf)
    hold_t = max(0, f - le_c) / fps
    if tpl.get('drift') and f >= le_c:
        scale *= 1 + tpl['drift'][0] * hold_t
    cx = width / 2 + width * tpl.get('dx_enter', 0) * (1 - min(1, f / max(1, le_c)))
    if tpl.get('drift') and f >= le_c:
        cx += width * tpl['drift'][1] * hold_t
    cy = height / 2 + height * tpl.get('dy_hold', 0) * min(1, hold_t / 1.0)
    layer = Image.new('RGBA', (width, height))
    line_h = None
    for row, ln in enumerate(lines):
        fill_m, outline_m = masks(ln, size, wide)
        line_h = line_h or fill_m.height * .92
        y = cy + (row - (len(lines) - 1) / 2) * line_h * scale
        tok = token
        if token == 'SWAP':
            phase = hold_t / max(.1, (fe - le_c - lx_c) / fps)
            first = phase < .35
            if .35 <= phase < .6:
                first = (int(f / 3) % 2 == 0)
            tok = ('F' if (row == 0) == first else 'O')
        elif token == 'SWAPFAST':
            tok = 'F' if (row + int(f / 2)) % 2 == 0 else 'O'
        elif tpl['kind'] == 'two' and token == 'SU' and row == 1:
            tok = 'O'
        elif token == 'F_flicker':
            tok = strobe_hold(f - le_c, scene['index'])
        s = scale
        if tok == 'Z':
            s = scale * (2.45 + .15 * (f % 2))
        elif tok == 'z':
            s = scale * .62
        if tok == 'BAR':
            bar_h = height * .085
            q = min(1, f / max(1, le_c))
            bar = Image.new('RGBA', (width, height))
            ImageDraw.Draw(bar).rectangle((0, cy - bar_h / 2, width, cy + bar_h / 2), fill=(40, 30, 200, int(200 * (1 - q))))
            layer.alpha_composite(bar.filter(ImageFilter.GaussianBlur(height * .01)))
        if tok in ('ECHO_DOWN', 'ECHO_IN'):
            q = progress
            prev = info.get('previous')
            if tok == 'ECHO_DOWN':
                # 아래로 내려가며 외곽선 복제 2개를 위에 남긴다
                for k, dy in enumerate((-.10, -.05)):
                    E.paste_scaled(layer, colored(outline_m, NAVY, .9 - .3 * k), cx, y + height * dy * (1 - q * .3), s)
                E.paste_scaled(layer, colored(fill_m, WHITE), cx, y + height * .06 * q, s)
            else:
                if prev is not None:
                    pf, po = masks(prev['text'] if len(prev['text']) <= 12 else prev['text'][:12], size, True)
                    for k, dy in enumerate((.06, .11)):
                        E.paste_scaled(layer, colored(po, NAVY, (1 - q) * (.9 - .3 * k)), cx, y + height * (dy + .08 * q), s)
                E.paste_scaled(layer, colored(outline_m, NAVY, .7 * (1 - q)), cx, y - height * .05, s)
                E.paste_scaled(layer, colored(fill_m, WHITE), cx, y - height * .07 * (1 - q), s)
            continue
        sprite = sprite_for(tok, fill_m, outline_m, progress)
        if sprite is not None:
            E.paste_scaled(layer, sprite, cx, y, s)
    image.alpha_composite(layer)
    overlay(image, info)
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('60944035', 'outline_scan', FPS, TEMPLATES, assign, minimum, render)
