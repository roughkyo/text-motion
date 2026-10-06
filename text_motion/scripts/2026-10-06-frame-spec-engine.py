"""kinetic-5-frame-spec: 원본 샘플의 프레임 단위 사양(references/analysis/*-spec.md)을 장면 템플릿으로 옮긴 엔진.

원칙
- 샘플 = 원본 장면 순서 그대로의 템플릿 목록. 사용자 문구를 이 순서에 하나씩 대응한다(대표 효과 몇 개 순환 금지).
- 등장 속도·지연은 원본 프레임 값을 초로 환산해 그대로 쓰고, 목표 길이는 유지 시간으로만 맞춘다.
- 수치 근거: references/analysis/detail/*.json (원본 해상도 프레임별 측정).
"""
import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

VERSION = 'kinetic-5-frame-spec'
SKILL = Path(__file__).resolve().parents[1]
FONT_DIR = SKILL / 'assets/fonts'


# ---------------------------------------------------------------- 기본 도구

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease_out(x, power=3):
    return 1 - (1 - clamp(x)) ** power


def ease_in(x, power=2):
    return clamp(x) ** power


def mix(a, b, q):
    q = clamp(q)
    return tuple(round(a[i] + (b[i] - a[i]) * q) for i in range(len(a)))


def hex_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=256)
def font(size, weight='Bold'):
    size = max(4, int(round(size)))
    for path in (FONT_DIR / f'Pretendard-{weight}.otf', FONT_DIR / 'Pretendard-Bold.otf'):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    for path in ('C:/Windows/Fonts/malgunbd.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
                 '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc', '/System/Library/Fonts/AppleSDGothicNeo.ttc'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise ValueError('한국어 서체가 필요합니다: assets/fonts의 Pretendard 또는 맑은 고딕/Noto Sans CJK.')


def glyph_mask(text, size, weight='Bold', tracking=0.0):
    """글자별 위치를 함께 돌려주는 알파 마스크. tracking은 em 단위 추가 자간."""
    f = font(size, weight)
    ascent, descent = f.getmetrics()
    advances = [f.getlength(ch) + (tracking * size if k < len(text) - 1 else 0) for k, ch in enumerate(text)]
    width = max(1, math.ceil(sum(advances))) + 4
    height = ascent + descent + 4
    mask = Image.new('L', (width, height))
    draw = ImageDraw.Draw(mask)
    x, positions = 2, []
    for ch, adv in zip(text, advances):
        draw.text((x, 2), ch, font=f, fill=255)
        positions.append((x, f.getlength(ch)))
        x += adv
    return mask, positions, ascent


def fill_image(mask, kind, colors, phase=0.0):
    """마스크 크기의 채움 이미지. kind: flat / vgrad(위→아래) / hgrad(왼→오)."""
    w, h = mask.size
    if kind == 'flat':
        im = Image.new('RGBA', (w, h), (*colors[0], 255))
    else:
        steps = np.linspace(0, 1, h if kind == 'vgrad' else w)[:, None]
        a, b = np.array(colors[0], float), np.array(colors[1], float)
        ramp = (a + (b - a) * steps).clip(0, 255).astype(np.uint8)
        if kind == 'vgrad':
            arr = np.repeat(ramp[:, None, :], w, axis=1)
        else:
            arr = np.repeat(ramp[None, :, :], h, axis=0)
        im = Image.fromarray(np.dstack([arr, np.full((h, w), 255, np.uint8)]), 'RGBA')
    im.putalpha(mask)
    return im


def bloom(layer, radius, strength, color=None):
    """소프트 글로우. 1/4 해상도에서 흐려 비용을 줄인다."""
    w, h = layer.size
    alpha = layer.getchannel('A').resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.BILINEAR)
    alpha = alpha.filter(ImageFilter.GaussianBlur(max(.5, radius / 4))).resize((w, h), Image.Resampling.BILINEAR)
    glow = Image.new('RGBA', (w, h), (*(color or (255, 255, 255)), 0))
    glow.putalpha(alpha.point(lambda a: round(a * strength)))
    out = Image.new('RGBA', (w, h))
    out.alpha_composite(glow)
    out.alpha_composite(layer)
    return out


def paste_scaled(dst, im, cx, cy, scale=1.0, alpha=1.0, blur=0.0, sx=None):
    """중심 기준 배치. sx가 있으면 가로만 별도 배율(3D 회전 근사)."""
    if alpha <= 0.003 or scale <= 0:
        return
    sx = scale if sx is None else sx
    w, h = max(1, round(im.width * sx)), max(1, round(im.height * scale))
    if (w, h) != im.size:
        im = im.resize((w, h), Image.Resampling.BICUBIC)
    if blur > .3:
        pad = int(blur * 3)
        canvas = Image.new('RGBA', (w + pad * 2, h + pad * 2))
        canvas.alpha_composite(im, (pad, pad))
        im = canvas.filter(ImageFilter.GaussianBlur(blur))
    if alpha < .999:
        im = im.copy()
        im.putalpha(im.getchannel('A').point(lambda a: round(a * alpha)))
    dst.alpha_composite(im, (round(cx - im.width / 2), round(cy - im.height / 2)))


# ---------------------------------------------------------------- 배경

def radial(width, height, cx, cy, rx, ry, inner, outer, power=1.6):
    small_w, small_h = 160, 90
    yy, xx = np.mgrid[0:small_h, 0:small_w]
    d = np.hypot((xx / small_w - cx) / rx, (yy / small_h - cy) / ry)
    q = (np.clip(d, 0, 1) ** power)[..., None]
    arr = (np.array(inner, float) * (1 - q) + np.array(outer, float) * q).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((width, height), Image.Resampling.BICUBIC).convert('RGBA')


def background(kind, width, height, t=0.0, progress=0.0):
    """원본 11971627에서 측정한 배경 종류. progress는 장면 내 진행도(광원 세기 변화 등)."""
    if kind == 'black':
        return Image.new('RGBA', (width, height), (0, 0, 0, 255))
    if kind == 'white':
        return Image.new('RGBA', (width, height), (255, 255, 255, 255))
    if kind == 'darkgray':
        return Image.new('RGBA', (width, height), (12, 12, 13, 255))
    if kind == 'purple':
        return Image.new('RGBA', (width, height), (103, 102, 245, 255))
    if kind == 'navy_glow':
        # 남색 바탕 + 하단 중앙 보라 광원
        return radial(width, height, .5, 1.1, .62, .55, (58, 52, 160), (4, 3, 10), 1.15)
    if kind == 'navy_glow_rising':
        # Flow 장면: 하단 광원이 점점 강해짐
        q = .45 + .55 * progress
        return radial(width, height, .5, 1.08, .55, .42 + .2 * progress, (int(40 + 60 * q), int(36 + 58 * q), int(120 + 110 * q)), (6, 5, 16), 1.15)
    if kind == 'lavender':
        # 밝은 연보라 방사 + 하단 보라 광
        base = radial(width, height, .5, .45, .72, .62, (240, 239, 251), (142, 140, 196), 1.25)
        glow = radial(width, height, .5, 1.05, .25, .3, (127, 120, 255), (244, 243, 252), 1.5)
        return ImageChops.multiply(base, glow)
    if kind == 'diagonal_purple':
        small = np.zeros((90, 160, 3))
        yy, xx = np.mgrid[0:90, 0:160]
        q = np.clip(xx / 160 * 1.05 + (1 - yy / 90) * .35 - .45, 0, 1)[..., None] ** 1.3
        small = np.array((5, 5, 14), float) * (1 - q) + np.array((107, 99, 245), float) * q
        return Image.fromarray(small.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC).convert('RGBA')
    if kind == 'purple_dark_ellipse':
        base = Image.new('RGBA', (width, height), (108, 106, 246, 255))
        dark = radial(width, height, .5, .78, .32, .22, (10, 8, 30), (108, 106, 246), 1.2)
        return dark
    if kind in ('smoke_cool', 'smoke_purple'):
        return smoke_frame(t, width, height, kind.split('_')[1])
    raise ValueError('알 수 없는 배경: ' + kind)


@lru_cache(maxsize=4)
def smoke_chunk(chunk, color='cool'):
    # 원본 60608725(cool)·60806086(purple)의 연기 진행을 그대로 쓴다(글자를 지운 파생 영상). 2초 단위 디코딩 캐시.
    path = SKILL / f'assets/2026-10-05-smoke-{color}-continuous-clean-v4.mp4'
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(chunk * 2), '-i', str(path), '-t', '2',
                                   '-vf', 'scale=640:360,fps=30', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    return np.frombuffer(raw, np.uint8).reshape(-1, 360, 640, 3)


def smoke_frame(source_time, width, height, color='cool'):
    source_time = clamp(source_time, 0, 22.9 if color == 'cool' else 24.9)
    chunk = int(source_time // 2)
    frames = smoke_chunk(chunk, color)
    index = min(len(frames) - 1, int((source_time - chunk * 2) * 30))
    return Image.fromarray(frames[index]).resize((width, height), Image.Resampling.BILINEAR).convert('RGBA')


# ---------------------------------------------------------------- 3D 회전 4각 별

@lru_cache(maxsize=8)
def star_sprite(size):
    """보라→분홍 그라데이션 4각 별 + 흰 중심 광. 원본 11971627 f300~ 근거."""
    s = max(8, int(size))
    im = Image.new('RGBA', (s * 2, s * 2))
    draw = ImageDraw.Draw(im)
    c = s
    pts = []
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        r = s * (.95 if k % 2 == 0 else .2)
        pts.append((c + math.cos(a) * r, c + math.sin(a) * r))
    mask = Image.new('L', im.size)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    grad = fill_image(mask, 'vgrad', [(232, 120, 255), (110, 92, 255)])
    core = Image.new('RGBA', im.size)
    ImageDraw.Draw(core).ellipse((c - s * .18, c - s * .18, c + s * .18, c + s * .18), fill=(255, 255, 255, 230))
    grad.alpha_composite(core.filter(ImageFilter.GaussianBlur(s * .08)))
    return bloom(grad, s * .5, .55, (210, 120, 255))


def draw_star(dst, cx, cy, size, t, spin_hz=1.1, grow=1.0):
    """세로축 회전: 가로 배율 |cos|, 최소 0.08(옆면 세로 막대)."""
    sprite = star_sprite(round(size))
    sx = max(.08, abs(math.cos(t * math.tau * spin_hz))) * grow
    paste_scaled(dst, sprite, cx, cy, grow, sx=sx)


# ---------------------------------------------------------------- 글자 배치

def split_words(text):
    return [w for w in re.split(r'\s+', text.strip()) if w]


@lru_cache(maxsize=2048)
def word_mask(word, size, weight, tracking=0.0):
    return glyph_mask(word, size, weight, tracking)


def line_layout(words, size, weight, tracking=0.0):
    """단어별 마스크·상대 x·폭과 전체 폭. 띄어쓰기는 0.26em."""
    space = size * .26
    items, x = [], 0.0
    for word in words:
        mask, positions, ascent = word_mask(word, int(size), weight, tracking)
        items.append({'word': word, 'mask': mask, 'positions': positions, 'x': x, 'w': mask.width - 4, 'ascent': ascent})
        x += mask.width - 4 + space
    return items, max(0.0, x - space)


def fit_size(words, size, weight, max_width, tracking=0.0):
    _, total = line_layout(words, int(size), weight, tracking)
    return size if total <= max_width else size * max_width / max(1, total)


# ---------------------------------------------------------------- 11971627 템플릿

FPS_1 = 30000 / 1001
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
PURPLE = (102, 97, 246)          # f4 측정 #6661f6
PURPLE_DIM = (28, 27, 69)        # f1 측정 #1c1b45

# kind: 'line'(문장) / 'word'(한 단어) / 'flip'(첫 단어 크게 → 나머지 추가) / 'two'(2줄)
TEMPLATES_11971627 = [
    {'n': 1, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'exit': 'push_left'},
    {'n': 2, 'frames': 30, 'bg': 'navy_glow', 'kind': 'line', 'enter': 'type', 'fill': 'type_lavender', 'size': .135, 'per_char': 2.0},
    {'n': 3, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'glow': .35, 'exit': 'push_up'},
    {'n': 4, 'frames': 30, 'bg': 'black', 'kind': 'word', 'enter': 'rise_letters', 'fill': 'metal_dark', 'exit': 'shrink'},
    {'n': 5, 'frames': 30, 'bg': 'lavender', 'kind': 'word', 'enter': 'big', 'fill': 'navy_purple', 'blur_in': 6, 'exit': 'shrink'},
    {'n': 6, 'frames': 30, 'bg': 'black', 'kind': 'word', 'enter': 'big', 'fill': 'lavender', 'bloom': .7, 'exit': 'shrink'},
    {'n': 7, 'frames': 30, 'bg': 'white', 'kind': 'two', 'enter': 'big_lines', 'fill': 'black', 'start_scale': 1.35, 'size': .13, 'exit': 'shrink'},
    {'n': 9, 'frames': 25, 'bg': 'navy_glow', 'kind': 'word', 'enter': 'type', 'fill': 'lavender_flat', 'glow': .55, 'size': .2, 'per_char': 2.3, 'first_instant': False},
    {'n': 10, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'exit': 'push_left'},
    {'n': 11, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'star': 'tail'},
    {'n': 12, 'frames': 65, 'bg': 'white', 'kind': 'line', 'enter': 'type_scroll'},
    {'n': 13, 'frames': 40, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'star': 'intro_head'},
    {'n': 14, 'frames': 45, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'exit': 'remove_left'},
    {'n': 15, 'frames': 20, 'bg': 'darkgray', 'kind': 'word', 'enter': 'big', 'fill': 'white', 'size': .14, 'start_scale': 1.0, 'grow': .08},
    {'n': 16, 'frames': 30, 'bg': 'navy_glow_rising', 'kind': 'word', 'enter': 'big', 'fill': 'lavender', 'bloom': .6, 'exit': 'rise_grow'},
    {'n': 17, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'rise': True, 'exit': 'push_up'},
    {'n': 18, 'frames': 75, 'bg': 'white', 'kind': 'flip', 'enter': 'flip'},
    {'n': 19, 'frames': 30, 'bg': 'navy_glow', 'kind': 'word', 'enter': 'type', 'fill': 'white', 'glow': .45, 'size': .17, 'star': 'dot', 'per_char': 1.3, 'first_instant': False},
    {'n': 20, 'frames': 30, 'bg': 'white', 'kind': 'line', 'enter': 'type', 'fill': 'accent_word', 'per_char': 1.0},
    {'n': 21, 'frames': 30, 'bg': 'black', 'kind': 'word', 'enter': 'blur_sweep', 'fill': 'lavender'},
    {'n': 22, 'frames': 30, 'bg': 'diagonal_purple', 'kind': 'word', 'enter': 'blur_sweep', 'fill': 'lavender_light'},
    {'n': 23, 'frames': 31, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'star': 'head'},
    {'n': 24, 'frames': 29, 'bg': 'black', 'kind': 'line', 'enter': 'words', 'glow': .3, 'exit': 'push_up'},
    {'n': 25, 'frames': 30, 'bg': 'purple', 'kind': 'word', 'enter': 'rise_letters', 'fill': 'metal_navy', 'exit': 'shrink'},
    {'n': 26, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'big', 'fill': 'white', 'size': .12, 'start_scale': 1.12, 'exit': 'shrink'},
    {'n': 27, 'frames': 30, 'bg': 'purple_dark_ellipse', 'kind': 'word', 'enter': 'big', 'fill': 'lavender', 'bloom': .5, 'exit': 'rise_grow'},
    {'n': 28, 'frames': 30, 'bg': 'black', 'kind': 'two', 'enter': 'words', 'rise': True, 'exit': 'push_up'},
    {'n': 29, 'frames': 30, 'bg': 'navy_glow', 'kind': 'line', 'enter': 'words', 'glow': .35, 'exit': 'push_up'},
    {'n': 30, 'frames': 30, 'bg': 'white', 'kind': 'line', 'enter': 'big', 'fill': 'black', 'size': .1, 'start_scale': 1.04},
    {'n': 31, 'frames': 30, 'bg': 'black', 'kind': 'line', 'enter': 'big', 'fill': 'white', 'size': .1, 'start_scale': 1.0},
    {'n': 32, 'frames': 30, 'bg': 'black', 'kind': 'word', 'enter': 'big', 'fill': 'white', 'size': .12, 'start_scale': 1.0},
]
LIGHT_BG = {'white', 'lavender'}


def star_outline(points_per_edge=18):
    """안쪽으로 휜 네 변을 가진 4각 별 외곽선(반지름 1 기준, f308~f317 4K 크롭 근거)."""
    tips = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    out = []
    for i in range(4):
        x0, y0 = tips[i]
        x1, y1 = tips[(i + 1) % 4]
        # 조절점을 중심 쪽으로 당겨 오목한 곡선 변을 만든다.
        cxp, cyp = (x0 + x1) * .3, (y0 + y1) * .3
        for k in range(points_per_edge):
            t = k / points_per_edge
            out.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cxp + t * t * x1,
                        (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cyp + t * t * y1))
    return np.array(out, float)


STAR_OUTLINE = star_outline()
# 앞면 대각 그라데이션(왼쪽 위 → 오른쪽 아래): 노랑 → 연분홍 → 자홍 → 보라 → 청보라
STAR_STOPS = [(0.0, (252, 232, 104)), (.14, (255, 186, 214)), (.32, (236, 88, 238)), (.62, (176, 84, 246)), (1.0, (98, 104, 246))]
STAR_SIDE = np.array((98, 100, 238), float)


def gradient_lookup(t):
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (t0, c0), (t1, c1) in zip(STAR_STOPS, STAR_STOPS[1:]):
        m = (t >= t0) & (t <= t1)
        q = ((t - t0) / (t1 - t0))[m][:, None]
        out[m] = np.array(c0) * (1 - q) + np.array(c1) * q
    return out


def rotation(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


def star_3d(dst, cx, cy, size, spin, alpha=1.0):
    """두께 있는 3D 압출 별(11971627 f300~f329 근거).
    앞면: 대각 그라데이션 + 회전에 따라 미끄러지는 흰 반사 띠, 옆면: 청보라 단색 + 조명 명암, 원근 투영, 보라 글로우."""
    if size < 2 or alpha <= 0:
        return
    ss = 2  # 2배 슈퍼샘플
    R = size * ss
    depth = .16 * R
    ay = spin * math.tau
    ax = .62 * math.sin(spin * math.tau * .5 + .6) + .15
    az = .12 * math.sin(spin * math.tau * .25)
    rot = rotation(ax, ay, az)
    n = len(STAR_OUTLINE)
    local_front = np.column_stack([STAR_OUTLINE * R, np.full(n, depth / 2)])
    local_back = np.column_stack([STAR_OUTLINE * R, np.full(n, -depth / 2)])
    pad = int(R * 1.5)
    size_px = pad * 2

    def project(points):
        q = points @ rot.T
        persp = 1 / (1 - q[:, 2] / (R * 7))
        return np.column_stack([pad + q[:, 0] * persp, pad + q[:, 1] * persp]), q[:, 2]

    front, zf = project(local_front)
    back, zb = project(local_back)
    canvas = np.zeros((size_px, size_px, 4), np.float32)

    def paint(poly, color_fn):
        pts = np.array(poly)
        x0, y0 = np.maximum(0, np.floor(pts.min(axis=0)).astype(int) - 1)
        x1, y1 = np.minimum(size_px, np.ceil(pts.max(axis=0)).astype(int) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        mask = Image.new('L', (x1 - x0, y1 - y0))
        ImageDraw.Draw(mask).polygon([(px - x0, py - y0) for px, py in pts], fill=255)
        m = np.asarray(mask, np.float32)[..., None] / 255
        if m.max() == 0:
            return
        rgb = color_fn((x0, y0, x1, y1))
        sub = canvas[y0:y1, x0:x1]
        sub[..., :3] = sub[..., :3] * (1 - m) + rgb * m
        sub[..., 3:] = sub[..., 3:] * (1 - m) + m

    light = np.array([-.45, -.65, .62]); light /= np.linalg.norm(light)
    view = np.array([0, 0, 1.0])
    half = (light + view); half /= np.linalg.norm(half)
    # 옆면: 먼 것부터
    sides = []
    for i in range(n):
        j = (i + 1) % n
        e = local_front[j] - local_front[i]
        normal_local = np.array([e[1], -e[0], 0.0])
        normal_local /= max(1e-9, np.linalg.norm(normal_local))
        nr = rot @ normal_local
        if nr[2] <= 0:
            continue
        quad = [front[i], front[j], back[j], back[i]]
        sides.append(((zf[i] + zf[j] + zb[i] + zb[j]) / 4, quad, nr))
    for _, quad, nr in sorted(sides, key=lambda x: x[0]):
        shade = .55 + .55 * max(0, float(nr @ light))
        spec = max(0, float(nr @ half)) ** 24
        color = np.clip(STAR_SIDE * shade + 255 * spec * .85, 0, 255)
        paint(quad, lambda box, c=color: c)
    # 앞면 또는 뒷면(보이는 쪽)
    face_normal = rot @ np.array([0, 0, 1.0])
    showing_front = face_normal[2] > 0
    poly = front if showing_front else back
    origin = (front if showing_front else back).mean(axis=0)
    basis = rot[:2, :2]  # 로컬 x,y → 화면(원근 무시 근사)
    inv = np.linalg.pinv(basis)
    def face_color(box):
        x0, y0, x1, y1 = box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rel = np.stack([xx - origin[0], yy - origin[1]], axis=-1).reshape(-1, 2) @ inv.T
        u, v = rel[:, 0] / R, rel[:, 1] / R
        if not showing_front:
            u = -u
        t = (u - v) * .5 * .9 + .5
        rgb = gradient_lookup(t)
        # 흰 반사 띠: 왼쪽 위→아래로 비스듬히, 회전량에 따라 가로로 미끄러짐
        band_pos = -.15 + .55 * math.sin(ay + .4)
        d = (u * .93 + v * .37) - band_pos
        w = np.exp(-(d / .085) ** 2) * (.6 + .4 * max(0, float(face_normal @ light)))
        rgb = rgb * (1 - w[:, None] * .92) + 255 * w[:, None] * .92
        lam = .92 + .12 * max(0, float(face_normal @ light))
        return np.clip(rgb * lam, 0, 255).reshape(y1 - y0, x1 - x0, 3)

    paint(poly, face_color)
    canvas[..., 3] *= 255
    im = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), 'RGBA')
    im = im.resize((size_px // ss, size_px // ss), Image.Resampling.LANCZOS)
    # 글로우가 잘려 사각 자국이 남지 않도록 여백을 넉넉히 둔다.
    margin = int(size * 1.6)
    roomy = Image.new('RGBA', (im.width + margin * 2, im.height + margin * 2))
    roomy.alpha_composite(im, (margin, margin))
    im = bloom(roomy, size * 1.1, .7, (196, 64, 255))
    paste_scaled(dst, im, cx, cy, 1.0, alpha=alpha)


def spin_amount(t, events):
    """단어가 붙을 때마다 약 0.45초 동안 한 바퀴 텀블링(4K 크롭 f304~f320), 사이에는 ±20° 흔들림."""
    total = .055 * math.sin(math.tau * .45 * t)
    for te in events:
        if t >= te:
            total += ease_out((t - te) / .45, 2.2)
    return total


def scene_base(template):
    return BLACK if template['bg'] in LIGHT_BG else WHITE


def color_fill(mask, color):
    return fill_image(mask, 'flat', [color])


def text_fill(mask, template, f, fe):
    """장면별 글자 질감. f: 장면 프레임, fe: 장면 총 프레임."""
    kind = template.get('fill', 'base')
    if kind == 'type_lavender':
        q = clamp(f / 25)
        return fill_image(mask, 'vgrad', [mix(PURPLE, (250, 250, 254), q), mix(PURPLE, (227, 225, 252), q)])
    if kind == 'lavender_flat':
        return fill_image(mask, 'vgrad', [(203, 201, 253), (160, 156, 250)])
    if kind == 'navy_purple':
        q = clamp(f / 15)
        return fill_image(mask, 'vgrad', [mix((14, 13, 34), (54, 50, 120), q), mix((48, 45, 110), (94, 88, 230), q)])
    if kind == 'lavender':
        q = clamp(f / max(1, fe))
        return fill_image(mask, 'vgrad', [mix((138, 135, 236), (183, 181, 249), q), mix((165, 162, 242), (179, 176, 249), q)])
    if kind == 'lavender_light':
        return fill_image(mask, 'vgrad', [(205, 203, 255), (150, 145, 250)])
    if kind == 'metal_dark':
        return fill_image(mask, 'vgrad', [mix((101, 101, 101), WHITE, f / 10), mix((78, 78, 78), WHITE, (f - 3) / 24)])
    if kind == 'metal_light':
        return fill_image(mask, 'vgrad', [BLACK, mix((190, 190, 190), (20, 20, 20), (f - 3) / 24)])
    if kind == 'metal_navy':
        return fill_image(mask, 'vgrad', [(10, 9, 36), mix((72, 68, 205), (14, 13, 44), (f - 3) / 24)])
    if kind == 'black':
        return color_fill(mask, BLACK)
    if kind == 'white':
        return color_fill(mask, WHITE)
    return color_fill(mask, scene_base(template))


def exit_offsets(template, pe, width, height):
    """하드컷 직전 8프레임의 가속 이동(측정: 왼쪽 0.03W / 위 0.045H / 수축 18%)."""
    kind = template.get('exit')
    q = ease_in(pe, 3)
    if kind == 'push_left':
        return -.03 * width * q, 0, 1.0
    if kind == 'push_up':
        return 0, -.045 * height * q, 1.0
    if kind == 'shrink':
        return 0, 0, 1 - .18 * q
    if kind == 'rise_grow':
        return 0, -.05 * height * ease_in(pe, 2), 1 + .08 * ease_in(pe, 2)
    return 0, 0, 1.0


def place_layer(dst, layer, dx, dy, scale):
    if abs(scale - 1) < .002 and abs(dx) < .5 and abs(dy) < .5:
        dst.alpha_composite(layer)
        return
    paste_scaled(dst, layer, dst.width / 2 + dx, dst.height / 2 + dy, scale)


def balanced_lines(words):
    """글자 수가 가장 고른 두 줄 분할."""
    best, cut = None, 1
    for k in range(1, len(words)):
        a, b = len(''.join(words[:k])), len(''.join(words[k:]))
        if best is None or abs(a - b) < best:
            best, cut = abs(a - b), k
    return [words[:cut], words[cut:]]


def render_words(template, words, f, fe, width, height, size, layer, user_lines=None, remove=None):
    """단어별 페이드인: 3프레임 간격, 최신 단어 보라 → 다음 단어 등장 시 기본색(f0~f29 측정)."""
    base = scene_base(template)
    light = template['bg'] in LIGHT_BG
    start_remove = fe - 4 - 3 * max(1, len(words) - 1)
    if template.get('exit') == 'remove_left' and len(words) > 1 and f >= start_remove:
        # 왼쪽 단어부터 3프레임마다 하나씩 사라지고 남은 문장이 다시 가운데로 모인다(f468~f479)
        removed = min(len(words) - 1, int((f - start_remove) / 3) + 1)
        q = ease_out(((f - start_remove) % 3) / 2.5, 3) if removed < len(words) - 1 or f - start_remove < 3 * removed else 1
        rest = words[removed:]
        items, total = line_layout(rest, int(size), 'SemiBold')
        _, prev_total = line_layout(words[removed - 1:], int(size), 'SemiBold')
        _, first_w = line_layout(words[removed - 1:removed], int(size), 'SemiBold')
        start_x = width / 2 - prev_total / 2 + first_w + size * .26
        target_x = width / 2 - total / 2
        x0 = start_x + (target_x - start_x) * q
        for item in items:
            paste_scaled(layer, color_fill(item['mask'], base), x0 + item['x'] + item['w'] / 2, height * .507)
        return []
    lines = [words]
    if template.get('kind') == 'two' and len(words) >= 2:
        lines = [split_words(x) for x in user_lines] if user_lines and len(user_lines) == 2 else balanced_lines(words)
    star_head = template.get('star') in ('head', 'intro_head')
    order = 0
    events = []
    line_h = size * 1.08
    top = height * .507 - (len(lines) - 1) * line_h / 2
    star_pos = None
    last_end = None
    intro = 12 if template.get('star') == 'intro_head' else 0
    for row, line in enumerate(lines):
        items, total = line_layout(line, int(size), 'SemiBold')
        extra = size * 1.1 if star_head else 0
        x0 = width / 2 - (total + extra) / 2 + extra
        y = top + row * line_h
        for item in items:
            a = 1 + intro + 3 * order
            events.append(a / FPS_1)
            order += 1
            if f < a:
                continue
            p = clamp((f - a) / 3)
            nxt = 1 + intro + 3 * order
            is_last = order == len(words)
            if (not is_last and f >= nxt) or (is_last and f >= a + 10):
                color = base
            else:
                start = (220, 219, 250) if light else (PURPLE_DIM if order == 1 else (110, 110, 118))
                color = mix(start, PURPLE, p * 1.4)
            alpha = ease_out(p * 1.5, 2)
            dy = (1 - ease_out(p * .8, 3)) * .035 * height if template.get('rise') else 0
            sprite = color_fill(item['mask'], color)
            paste_scaled(layer, sprite, x0 + item['x'] + item['w'] / 2, y + dy, 1.0, alpha=alpha)
            last_end = (x0 + item['x'] + item['w'], y)
        if row == 0 and star_head:
            star_pos = (x0 - size * .62, y)
    star = template.get('star')
    t = f / FPS_1
    if star == 'tail' and last_end:
        star_3d(layer, last_end[0] + size * .78, last_end[1], size * .6, spin_amount(t, events))
    if star in ('head', 'intro_head') and star_pos:
        if star == 'intro_head' and f < 14:
            # 중앙 점에서 회전하며 커졌다가(f396~404) 머리 위치로 접혀 이동
            grow = ease_out(f / 9, 3)
            move = ease_out((f - 9) / 5, 3) if f >= 9 else 0
            cx = width / 2 + (star_pos[0] - width / 2) * move
            star_3d(layer, cx, star_pos[1], size * (.05 + .75 * grow - .3 * move), spin_amount(t * 2.2, []))
        else:
            star_3d(layer, star_pos[0], star_pos[1], size * .55, spin_amount(t, events))
    return events


def render_type(template, text, f, fe, width, height, size, layer, anchor_right=None, typing_start=0, per_char=None):
    """커서 타이핑. 첫 단어는 즉시 보이고 다음 글자부터 장면별 속도로 친다(f30·f635·f665 측정)."""
    count = len(text)
    per_char = per_char or template.get('per_char', 2.0)
    head = len(text.split(' ')[0]) + 1 if template.get('first_instant', True) and ' ' in text else 1
    shown = int(clamp(head + (f - typing_start) / per_char, 0, count)) if f >= typing_start else 0
    prefix = text[:shown]
    weight = 'SemiBold'
    if template.get('fill') == 'accent_word':
        words = prefix.split(' ')
        items, total = line_layout([w for w in words if w], int(size), weight)
    else:
        items = None
        mask, _, _ = glyph_mask(prefix, int(size), weight) if prefix else (Image.new('L', (1, 1)), [], 0)
        total = mask.width - 4 if prefix else 0
    if anchor_right is not None:
        x0 = anchor_right - total
    else:
        x0 = width / 2 - total / 2
    y = height * .507
    if items is not None:
        done = shown >= count
        for k, item in enumerate(items):
            last = k == len(items) - 1 and not (done and f > typing_start + count * per_char + 6)
            color = PURPLE if last else scene_base(template)
            paste_scaled(layer, color_fill(item['mask'], color), x0 + item['x'] + item['w'] / 2, y)
    elif prefix:
        sprite = text_fill(mask, template, f, fe)
        paste_scaled(layer, sprite, x0 + total / 2 + 2, y)
    typing = shown < count
    if typing or int(f // 4) % 2 == 0:
        cursor_color = PURPLE if template.get('fill') in ('type_lavender', 'accent_word') and typing else scene_base(template)
        cx = x0 + total + size * .1
        ImageDraw.Draw(layer).rectangle((cx, y - size * .52, cx + max(2, size * .07), y + size * .42), fill=(*cursor_color, 255))
    return x0, total, y


def render_rise_letters(template, word, f, fe, width, height, size, layer):
    """글자별 아래→위 솟음(f90~f119 측정: 1.2프레임 간격, 4프레임 상승, 기준선 아래는 가림)."""
    mask, positions, ascent = glyph_mask(word, int(size), 'Bold', -.02)
    sprite = text_fill(mask, template, f, fe)
    word_layer = Image.new('RGBA', sprite.size)
    for k, (x, adv) in enumerate(positions):
        a = 1 + 1.2 * k
        if f < a:
            continue
        p = clamp((f - a) / 4)
        dy = round((1 - ease_out(p, 3)) * ascent * .55)
        char = sprite.crop((int(x), 0, int(x + adv + 2), sprite.height))
        alpha = ease_out(clamp((f - a) / 3), 2)
        if alpha < 1:
            char.putalpha(char.getchannel('A').point(lambda v: round(v * alpha)))
        word_layer.alpha_composite(char, (int(x), dy))
    # 기준선 아래로 내려간 부분은 마스크로 가린다.
    clip = ascent + int(size * .03)
    if clip < word_layer.height:
        ImageDraw.Draw(word_layer).rectangle((0, clip, word_layer.width, word_layer.height), fill=(0, 0, 0, 0))
    drift = .037 * height * (f / max(1, fe))   # 유지 중 전체가 천천히 가라앉음
    paste_scaled(layer, word_layer, width / 2, height * .49 + drift)


def render_blur_sweep(template, word, f, fe, width, height, size, layer):
    """왼쪽→오른쪽 블러 스윕 등장·퇴장(f695~f754)."""
    mask, positions, ascent = glyph_mask(word, int(size), 'Bold', -.02)
    sprite = text_fill(mask, template, f, fe)
    x0 = width / 2 - (mask.width - 4) / 2
    for k, (x, adv) in enumerate(positions):
        a = .5 * k
        if f < a:
            continue
        blur = .04 * height * (1 - ease_out((f - a) / 4, 2))
        e = fe - 7 + 1.2 * k
        if f > e:
            blur += .05 * height * ease_out((f - e) / 3, 2)
        alpha = .3 + .7 * ease_out((f - a) / 3, 2)
        char = sprite.crop((int(x), 0, int(x + adv + 2), sprite.height))
        paste_scaled(layer, char, x0 + x + adv / 2, height * .5, 1.0, alpha=alpha, blur=blur * .5)


def scale_curve(template, f, fe):
    """빠른 안착(10프레임) → 느린 수축 → 컷 직전 가속(Online·Ready 측정)."""
    start = template.get('start_scale', 1.25)
    s = 1 + (start - 1) * (1 - ease_out(f / 10, 3))
    s *= 1 - .04 * (f / max(1, fe))
    if template.get('grow'):
        s *= 1 + template['grow'] * (f / max(1, fe))
    return s


def overlay(dst, info, light):
    """상시 작은 글자 레이어(원본의 로고·하단 문단 질감). 사용자 주제로 대체한다."""
    w, h = dst.size
    gray = (150, 150, 158, 255) if light else (92, 92, 100, 255)
    draw = ImageDraw.Draw(dst)
    small = font(h * .013, 'Bold')
    draw.text((w * .021, h * .03), info.get('label', ''), font=small, fill=gray)
    tiny = font(h * .0085, 'Regular')
    line = info.get('footer', '')
    if line:
        tw = draw.textlength(line, font=tiny)
        draw.text((w / 2 - tw / 2, h * .925), line, font=tiny, fill=(gray[0], gray[1], gray[2], 150))


def render_scene_11971627(scene, local, width, height, info):
    template = scene['template']
    fe = scene['duration'] * FPS_1
    f = local * FPS_1
    pe = clamp((f - (fe - 8)) / 8)
    words = split_words(scene['text'])
    bg_kind = template['bg']
    enter = template['enter']
    # 장면 중간 배경 전환(12: 흰→검정 스크롤, 18: 흰→검정 반전)
    if enter == 'type_scroll' and f >= fe * .38:
        bg_kind = 'black'
    if enter == 'flip' and f >= min(30, fe * .4):
        bg_kind = 'black'
    image = background(bg_kind, width, height, local, f / max(1, fe))
    layer = Image.new('RGBA', (width, height))
    unit = height
    max_w = width * .8
    if enter == 'words':
        size = fit_size(words, unit * template.get('size', .105), 'SemiBold', max_w if template.get('kind') != 'two' else max_w * 1.7)
        render_words(template, words, f, fe, width, height, size, layer, scene.get('lines'))
    elif enter == 'type':
        size = fit_size(words, unit * template.get('size', .105), 'SemiBold', max_w)
        render_type(template, ' '.join(words), f, fe, width, height, size, layer)
        done_at = len(' '.join(words)) * template.get('per_char', 2.0)
        if template.get('star') == 'dot' and f >= done_at * .6:
            # 원본은 i의 점을 별로 바꾼다. 한글에는 점이 없으므로 마지막 글자 오른쪽 위에 둔다.
            mask, _, _ = glyph_mask(' '.join(words), int(size), 'SemiBold')
            x_end = width / 2 + (mask.width - 4) / 2
            star_3d(layer, x_end - size * .05, height * .507 - size * .62, size * .24, spin_amount(local, [done_at * .6 / FPS_1]))
    elif enter == 'type_scroll':
        text = ' '.join(words)
        split = fe * .38
        if f < split:
            size = fit_size(words, unit * .085, 'SemiBold', max_w)
            render_type({**template, 'bg': 'white', 'fill': 'black'}, text, f, fe, width, height, size, layer, per_char=1.5)
        else:
            # 큰 글자로 바뀌고 커서(오른쪽 끝) 고정, 문장이 왼쪽으로 밀려남(f355~f394).
            # 원본은 문장 앞부분이 화면 밖으로 나가지만 한글 문장은 잘리면 읽히지 않으므로(사용자 검토 2026-10-06)
            # 완성 문장 전체가 화면 폭 86% 안에 들어오는 크기로 제한하고, 오른쪽 끝을 완성 문장의 오른쪽 끝에 고정한다.
            head = len(words[0]) + 1 if len(words) > 1 else 1
            typed_a = min(len(text), head + int(split / 1.5))
            size = fit_size(words, unit * .19, 'SemiBold', width * .86)
            full, _, _ = glyph_mask(text, int(size), 'SemiBold')
            right = width / 2 + (full.width - 4) / 2
            render_type({**template, 'bg': 'black', 'fill': 'white', 'first_instant': False}, text, f, fe, width, height, size, layer,
                        anchor_right=right, typing_start=split - (typed_a - 1) * 2.4, per_char=2.4)
    elif enter == 'rise_letters':
        word = ' '.join(words)
        size = fit_size([word], unit * .34, 'Bold', width * .72)
        render_rise_letters(template, word, f, fe, width, height, size, layer)
    elif enter == 'blur_sweep':
        word = ' '.join(words)
        size = fit_size([word], unit * .3, 'Bold', width * .72)
        render_blur_sweep(template, word, f, fe, width, height, size, layer)
    elif enter in ('big', 'big_lines'):
        lines = [' '.join(words)]
        if enter == 'big_lines' and len(words) >= 2:
            user = scene.get('lines')
            lines = user if user and len(user) == 2 else [' '.join(x) for x in balanced_lines(words)]
        base_size = unit * template.get('size', .3 if template['kind'] == 'word' else .105)
        size = min(fit_size([ln], base_size, 'Bold', width * .74) for ln in lines)
        word_layer = Image.new('RGBA', (width, height))
        for row, ln in enumerate(lines):
            mask, _, _ = glyph_mask(ln, int(size), 'Bold', -.015)
            sprite = text_fill(mask, template, f, fe)
            y = height * .5 + (row - (len(lines) - 1) / 2) * size * 1.08
            blur = template.get('blur_in', 0) * max(0, 1 - f / 4) * height / 1080
            paste_scaled(word_layer, sprite, width / 2, y, 1.0, blur=blur)
        if template.get('bloom'):
            word_layer = bloom(word_layer, height * .045, template['bloom'], (170, 165, 255))
        s = scale_curve(template, f, fe)
        place_layer(layer, word_layer, 0, 0, s)
    elif enter == 'flip':
        first, rest = (words[0], words[1:]) if words else ('', [])
        flip_at = min(30, fe * .4)
        if f < flip_at:
            size = fit_size([first], unit * .3, 'Bold', width * .72)
            render_rise_letters({**template, 'fill': 'metal_light'}, first, f, flip_at, width, height, size, layer)
        else:
            g = f - flip_at
            if not rest or g < 15:
                size = unit * .13 * (1 - .05 * clamp(g / 15))
                mask, _, _ = glyph_mask(first, int(size), 'SemiBold')
                paste_scaled(layer, color_fill(mask, WHITE), width / 2, height * .507)
            else:
                size = fit_size(words, unit * .105, 'SemiBold', max_w)
                sub = {**template, 'bg': 'black', 'kind': 'line'}
                # 첫 단어는 이미 보이는 상태로 두고 나머지 단어만 3프레임 간격으로 붙인다.
                render_words(sub, words, g - 15 + 4, fe - flip_at - 15, width, height, size, layer)
    if template.get('glow'):
        layer = bloom(layer, height * .03, template['glow'], (235, 232, 255))
    dx, dy, s = exit_offsets(template, pe, width, height)
    if enter == 'words' and template.get('exit') == 'remove_left':
        pass
    place_layer(image, layer, dx, dy, s)
    overlay(image, info, bg_kind in LIGHT_BG)
    return image.convert('RGB')


# ---------------------------------------------------------------- 60608725 템플릿(25fps)

FPS_5 = 25.0
SOFT_WHITE = (244, 245, 248)
TEAL, YELLOW, CYAN, PINK = (62, 224, 176), (232, 232, 58), (92, 224, 232), (255, 122, 138)
MAGENTA, GREEN = (208, 32, 224), (48, 226, 140)

TEMPLATES_60608725 = [
    {'n': 1, 'frames': 53, 'enter': 'hero_type', 'accent': TEAL, 'hero_scale': 2.6, 'exit': 'word_fade_left', 'delay': 0},
    {'n': 2, 'frames': 55, 'enter': 'word_fade', 'accent': YELLOW, 'exit': 'backspace', 'underline': (1, (110, 200, 60)), 'delay': 7, 'del_frames': 10, 'del_ease': 2},
    {'n': 3, 'frames': 58, 'enter': 'gradient_focus', 'gradient': ((150, 64, 255), (240, 96, 205)), 'exit': 'delete_left', 'delay': 0, 'del_rate': 1.8, 'del_ease': 2},
    {'n': 4, 'frames': 57, 'enter': 'rise_words', 'exit': 'tracking_out', 'delay': 0},
    {'n': 5, 'frames': 39, 'enter': 'type', 'pass_wave': CYAN, 'exit': 'backspace', 'delay': 6, 'del_rate': 1.7},
    {'n': 6, 'frames': 36, 'enter': 'type', 'pass_wave': PINK, 'exit': 'backspace', 'delay': 3, 'del_rate': 1.7},
    {'n': 7, 'frames': 50, 'enter': 'cursor_type', 'exit': 'fade_slide_left', 'delay': 1},
    {'n': 8, 'frames': 63, 'enter': 'hero_word', 'accent': MAGENTA, 'hero_scale': 2.5, 'exit': 'delete_left', 'delay': 4, 'del_rate': 2.3, 'del_ease': 1},
    {'n': 9, 'frames': 62, 'enter': 'blur_focus', 'hero_scale': 2.0, 'exit': 'squeeze_blur', 'underline': (-1, (150, 70, 230)), 'delay': 2},
    {'n': 10, 'frames': 101, 'enter': 'green_type_big', 'accent': GREEN, 'hero_scale': 2.2, 'exit': None},
]


@lru_cache(maxsize=4096)
def char_mask(ch, size, weight='Bold'):
    f = font(size, weight)
    ascent, descent = f.getmetrics()
    w = max(1, math.ceil(f.getlength(ch))) + 6
    m = Image.new('L', (w, ascent + descent + 6))
    ImageDraw.Draw(m).text((3, 3), ch, font=f, fill=255)
    return m, f.getlength(ch)


def compose_chars(layer, text, size, states, cx, cy, recenter=True, weight='Bold', collapse=0.0):
    """글자별 상태(visible/alpha/color/dx/dy/scale/blur/space/fill)로 한 줄을 그린다.
    recenter: 보이는 글자 범위의 중심을 cx에 맞춘다(타이핑 중 재중앙정렬, 60608725 전 구간).
    collapse: 0~1, 글자들을 줄 오른쪽 끝으로 겹쳐 모은다(f466~f473 가로 압축 퇴장).
    반환: 보이는 범위(left, right)와 글자별 (번호, 중심 x, 폭)."""
    advances = [char_mask(ch, int(size), weight)[1] for ch in text]
    xs, x = [], 0.0
    for k, adv in enumerate(advances):
        xs.append(x)
        x += adv + states[k].get('space', 0) * size
    visible = [k for k, st in enumerate(states) if st.get('visible', True) and text[k] != ' ']
    if not visible:
        return None
    lo = xs[visible[0]]
    hi = xs[visible[-1]] + advances[visible[-1]]
    offset = cx - (lo + hi) / 2 if recenter else cx - x / 2
    right_abs = offset + hi
    placed = []
    for k in visible:
        st = states[k]
        mask, adv = char_mask(text[k], int(size), weight)
        fill = st.get('fill')
        sprite = fill_image(mask, fill[0], fill[1]) if fill else fill_image(mask, 'flat', [st.get('color', SOFT_WHITE)])
        scale = st.get('scale', 1.0)
        px = offset + xs[k] + adv / 2
        # 글자 배율은 문장 중심을 기준으로 위치까지 함께 키운다(대형 첫 단어 수축).
        px = cx + (px - cx) * st.get('pos_scale', 1.0) + st.get('dx', 0)
        if collapse > 0:
            px = right_abs + (px - right_abs) * (1 - .85 * collapse)
        if st.get('rot'):
            sprite = sprite.rotate(st['rot'], Image.Resampling.BICUBIC, expand=True)
        paste_scaled(layer, sprite, px, cy + st.get('dy', 0), scale, alpha=st.get('alpha', 1.0), blur=st.get('blur', 0),
                     sx=scale * st.get('sx', 1.0))
        placed.append((k, px, adv * scale * st.get('sx', 1.0)))
    return {'left': offset + lo, 'right': right_abs, 'chars': placed}


def char_x(extent, idx):
    """글자 번호(실수)를 현재 화면 x로 바꾼다. 보이지 않는 범위는 평균 폭으로 연장."""
    chars = extent['chars']
    if not chars:
        return extent['left']
    avg = sum(w for _, _, w in chars) / len(chars)
    if idx <= chars[0][0]:
        return chars[0][1] - chars[0][2] / 2 + (idx - chars[0][0]) * avg
    for (k0, x0, w0), (k1, x1, w1) in zip(chars, chars[1:]):
        if k0 <= idx <= k1:
            q = (idx - k0) / max(1e-6, k1 - k0)
            return (x0 - w0 / 2) + ((x1 - w1 / 2) - (x0 - w0 / 2)) * q
    k, x, w = chars[-1]
    return x - w / 2 + (idx - k) * avg


def tint_band(layer, extent, cy, size, accent, mode, idx, soft):
    """글자 화소를 가로 위치에 따라 강조색에 섞는다. idx·soft는 글자 수 단위.
    mode 'trail': idx 오른쪽이 강조색, 경계는 soft 글자 폭만큼 부드럽다(흰색이 왼→오로 강조색을 밀어냄, f75~f82).
    mode 'band' : idx 중심의 부드러운 띠(청록 f237~f246, 분홍 f274~f281)."""
    if not extent or not extent['chars']:
        return
    avg = sum(w for _, _, w in extent['chars']) / len(extent['chars'])
    x0, x1 = max(0, int(extent['left'] - size)), min(layer.width, int(extent['right'] + size))
    y0, y1 = max(0, int(cy - size * 1.6)), min(layer.height, int(cy + size * 1.6))
    if x1 <= x0 or y1 <= y0:
        return
    center = char_x(extent, idx)
    width = max(1.0, soft * avg)
    xs = np.arange(x0, x1, dtype=np.float32)
    if mode == 'trail':
        w = np.clip((xs - center) / width * .5 + .5, 0, 1)
        w = w * w * (3 - 2 * w)
    else:
        w = np.exp(-((xs - center) / width) ** 2)
    region = np.asarray(layer.crop((x0, y0, x1, y1))).astype(np.float32)
    w = w[None, :, None]
    region[..., :3] = region[..., :3] * (1 - w) + np.array(accent, np.float32) * w
    layer.paste(Image.fromarray(region.clip(0, 255).astype(np.uint8), 'RGBA'), (x0, y0))


def tapered_stroke(layer, x0, x1, y, thick, colors, sag=0.0, alpha=1.0):
    """끝이 가는 붓 획 밑줄(가로 그라데이션, 살짝 휨). f100~f108 녹→노랑, f461~f473 자홍→보라."""
    if x1 - x0 < 1.5 or alpha <= 0:
        return
    n = 32
    top, bottom = [], []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        h = thick * (math.sin(math.pi * t) ** .6) / 2
        yc = y - sag * math.sin(math.pi * t)
        top.append((x, yc - h))
        bottom.append((x, yc + h))
    pad = int(thick * 2 + sag * 2 + 4)
    w, h = int(x1 - x0) + pad * 2, int(thick + sag) * 2 + pad * 2
    mask = Image.new('L', (w, h))
    ox, oy = x0 - pad, y - h / 2
    ImageDraw.Draw(mask).polygon([(px - ox, py - oy) for px, py in top + bottom[::-1]], fill=round(255 * alpha))
    stroke = fill_image(mask, 'hgrad', colors)
    layer.alpha_composite(stroke, (int(ox), int(oy)))


def word_index(text):
    """글자별 단어 번호(공백은 -1)."""
    out, w = [], 0
    for k, ch in enumerate(text):
        if ch == ' ':
            out.append(-1)
            if k and text[k - 1] != ' ':
                w += 1
        else:
            out.append(w)
    return out


def wave_color(accent, k, n, f, start, duration=6):
    """왼쪽→오른쪽으로 강조색이 흰색으로 바뀌는 파도(f20~f25, f72~f80 측정)."""
    w = start + duration * (k / max(1, n - 1))
    return mix(accent, SOFT_WHITE, (f - w) / 2)


def enter_states(template, text, f, fe):
    """등장 단계의 글자 상태. 반환: (states, 등장 완료 프레임)."""
    n = len(text)
    widx = word_index(text)
    words = max(widx) + 1 if n else 0
    first_len = next((k for k, ch in enumerate(text) if ch == ' '), n)
    states = [{'visible': False} for _ in text]
    enter = template['enter']
    accent = template.get('accent', SOFT_WHITE)
    if enter == 'hero_type':
        # 첫 단어 1.5프레임/글자 크게 → 6프레임 수축 → 나머지 1글자/프레임, 강조색 → 흰색 파도
        hero_end = 1 + 1.5 * (first_len - 1)
        shrink_end = hero_end + 6
        hero = template.get('hero_scale', 2.6)
        scale = 1 + (hero - 1) * (1 - ease_out((f - hero_end) / 6, 3)) if f > hero_end else hero
        rest_start = shrink_end + 1
        done = rest_start + (n - first_len) / 1.4
        wave_start = done + 1
        for k in range(n):
            a = 1 + 1.5 * k if k < first_len else rest_start + (k - first_len) / 1.4
            if f < a:
                continue
            p = clamp((f - a) / 2)
            color = mix((150, 150, 150), SOFT_WHITE, p) if k >= first_len else SOFT_WHITE
            states[k] = {'visible': True, 'alpha': .35 + .65 * p, 'blur': (1 - p) * 3 * scale, 'color': color,
                         'scale': scale, 'pos_scale': scale}
        tint = None
        if f >= rest_start:
            tail = -3 + (n + 6) * ease_in(clamp((f - wave_start) / 7), 1.3)
            tint = (accent, 'trail', tail, 3.5)
        return states, wave_start + 8, tint
    if enter == 'hero_word':
        # 자홍 대형 첫 단어 한 번에(약한 블러) → 5프레임 수축 → 나머지 1글자/프레임 자홍 → 흰 파도
        hero = template.get('hero_scale', 2.5)
        scale = 1 + (hero - 1) * (1 - ease_out((f - 1) / 5, 3))
        rest_start = 8
        done = rest_start + (n - first_len) / 1.5
        for k in range(n):
            a = 0 if k < first_len else rest_start + (k - first_len) / 1.5
            if f < a:
                continue
            p = clamp((f - a) / 2)
            states[k] = {'visible': True, 'alpha': .4 + .6 * p, 'blur': (1 - clamp(f / 4)) * 3 if k < first_len else (1 - p) * 2,
                         'color': SOFT_WHITE, 'scale': scale if k < first_len else 1.0, 'pos_scale': scale if k < first_len else 1.0}
        tail = -3 + (n + 6) * ease_in(clamp((f - done - 3) / 8), 1.3)
        return states, done + 12, (accent, 'trail', tail, 3.5)
    if enter == 'word_fade':
        # 원본 f61~f80(장면 시작 f54 기준 +7): 첫 단어 2프레임 → 이후 2.2글자/프레임으로 문장 40%까지 →
        # 3프레임 멈춤 → 1.3글자/프레임으로 끝까지. 새 글자는 회색 → 흰색(2프레임).
        # 측정: W f61 / What f62 / do f63 / yo f64 / you f65 → 3프레임 멈춤 → w f69 → 4프레임 멈춤 → 1.6글자/프레임
        burst_end = max(first_len + 1, round(n * .42))
        burst_end = next((k for k in range(burst_end, n) if text[k] == ' '), burst_end)
        resume = 1 + (burst_end - first_len) / 2.4 + 3
        times = []
        for k in range(n):
            if k < first_len:
                times.append(1.0 * k / max(1, first_len - 1))
            elif k < burst_end:
                times.append(1 + (k - first_len) / 2.4)
            elif k <= burst_end + 1:
                times.append(resume)
            else:
                times.append(resume + 4 + (k - burst_end - 2) / 1.6)
        done = times[-1] if times else 0
        for k in range(n):
            if widx[k] < 0 or f < times[k]:
                continue
            p = clamp((f - times[k]) / 2)
            states[k] = {'visible': True, 'alpha': .25 + .75 * p, 'color': mix((150, 150, 150), SOFT_WHITE, p)}
        # 노랑: 멈춤 뒤 타이핑 재개 2프레임 후(f71) 보이는 문장 전체가 노랑 →
        # 4프레임 뒤(f75)부터 흰색 꼬리가 왼→오로 가속하며 밀어내 f82에 문장 끝을 지난다.
        tint = None
        if f >= resume + 2:
            tail = -3 + (n + 6) * ease_in(clamp((f - resume - 6) / 7), 1.4)
            tint = (accent, 'trail', tail, 3.5)
        return states, done + 12, tint
    if enter in ('type', 'cursor_type'):
        per = 1.0 if enter == 'type' else 1.3
        done = per * (n - 1) + 1
        for k in range(n):
            a = per * k
            if f < a:
                continue
            p = clamp((f - a) / 1.5)
            states[k] = {'visible': True, 'alpha': .5 + .5 * p, 'color': SOFT_WHITE}
        wave = template.get('pass_wave')
        tint = None
        if wave and done + 1 <= f <= done + 13:
            # 완성 후 강조색 띠가 왼→오로 지나간다(청록 f237~f246 / 분홍 f274~f281)
            tint = (wave, 'band', -3 + (n + 6) * clamp((f - done - 1) / 10), 3.0)
        return states, done + (12 if wave else 4), tint
    if enter == 'gradient_focus':
        # 첫 단어: 보라→분홍 가로 그라데이션, 2.2배·강한 블러 → 14프레임에 선명·수축. 다음 단어 회보라 페이드 → 흰색
        hero = 2.2
        scale = 1 + (hero - 1) * (1 - ease_out(f / 14, 3))
        grad = template['gradient']
        for k in range(n):
            w = widx[k]
            if w < 0:
                continue
            a = 0 if w == 0 else 15 + 18 * (w - 1)
            if f < a:
                continue
            p = clamp((f - a) / 6)
            if w == 0:
                q = clamp((f - 17) / 7)
                fill = ('hgrad', [mix(grad[0], SOFT_WHITE, q), mix(grad[1], SOFT_WHITE, q)])
                states[k] = {'visible': True, 'alpha': ease_out(f / 3), 'blur': 8 * (1 - ease_out(f / 8, 2)), 'fill': fill,
                             'scale': scale, 'pos_scale': scale}
            else:
                states[k] = {'visible': True, 'alpha': .3 + .7 * p, 'blur': (1 - p) * 4, 'color': mix((160, 120, 205), SOFT_WHITE, (f - a - 3) / 5)}
        return states, 15 + 18 * max(0, words - 1) + 6, None
    if enter == 'rise_words':
        # 단어별 아래(+0.05H)·작은 크기에서 떠오름, 2프레임 간격(f167~f188)
        for k in range(n):
            w = widx[k]
            if w < 0:
                continue
            a = 0 if w == 0 else 8 + 2 * (w - 1)
            if f < a:
                continue
            p = ease_out((f - a) / (8 if w == 0 else 5), 3)
            states[k] = {'visible': True, 'alpha': p, 'dy': (1 - p) * (.05 if w == 0 else .025), 'scale': .8 + .2 * p}
        return states, 8 + 2 * max(0, words - 1) + 6, None
    if enter == 'blur_focus':
        # 첫 단어 2배·강한 블러 → 14프레임 선명, 이어 단어별 블러 페이드인, 전체 수축(f412~f437)
        hero = template.get('hero_scale', 2.0)
        first_end = 12
        scale = 1 + (hero - 1) * (1 - ease_out((f - first_end) / 10, 3)) if f > first_end else hero
        for k in range(n):
            w = widx[k]
            if w < 0:
                continue
            a = 0 if w == 0 else first_end + 1 + 2.5 * (w - 1)
            if f < a:
                continue
            dur = 14 if w == 0 else 5
            p = ease_out((f - a) / dur, 2)
            states[k] = {'visible': True, 'alpha': .2 + .8 * p, 'blur': (1 - p) * 9, 'scale': scale, 'pos_scale': scale}
        return states, first_end + 2 + 3 * max(0, words - 1) + 10, None
    if enter == 'green_type_big':
        # 녹색 큰 글자 2.3프레임/글자 타이핑(재중앙정렬) → 녹→청록→흰 → 10프레임 수축(f474~f514)
        hero = template.get('hero_scale', 2.2)
        per = 2.3
        done = per * (n - 1) + 2
        color_end = done + 9
        scale = hero if f < color_end else 1 + (hero - 1) * (1 - ease_out((f - color_end) / 10, 2))
        for k in range(n):
            a = per * k
            if f < a:
                continue
            p = clamp((f - a) / 2)
            q = clamp((f - done) / 9)
            color = mix(GREEN, CYAN, q * 2) if q < .5 else mix(CYAN, SOFT_WHITE, (q - .5) * 2)
            states[k] = {'visible': True, 'alpha': .4 + .6 * p, 'color': color, 'scale': scale, 'pos_scale': scale}
        return states, color_end + 11, None
    raise ValueError('알 수 없는 등장: ' + enter)


def exit_length(template, text):
    n = len(text)
    words = max(word_index(text)) + 1 if n else 0
    kind = template.get('exit')
    if kind in ('backspace', 'delete_left'):
        return template.get('del_frames') or n / template.get('del_rate', 2.0) + 1
    if kind == 'word_fade_left':
        return 5 * words
    if kind == 'fade_slide_left':
        return 2 * words + 1
    if kind == 'tracking_out':
        return 2.5 * words + 7
    if kind == 'squeeze_blur':
        return 8
    return 0


def apply_exit(template, text, states, g):
    """퇴장 단계(g: 퇴장 시작 후 프레임)."""
    kind = template.get('exit')
    n = len(text)
    widx = word_index(text)
    if kind == 'backspace':
        # 끝에서부터 1글자/프레임 삭제, 남은 글자 재중앙정렬(f98~f108)
        keep = n - int(n * ease_in(g / exit_length(template, text), template.get('del_ease', 2)))
        for k in range(n):
            if k >= keep:
                states[k]['visible'] = False
    elif kind == 'delete_left':
        gone = int(n * ease_in(g / exit_length(template, text), template.get('del_ease', 2)))
        for k in range(min(n, gone)):
            states[k]['visible'] = False
    elif kind in ('word_fade_left', 'fade_slide_left'):
        for k in range(n):
            w = widx[k]
            if w < 0:
                continue
            step = 5 if kind == 'word_fade_left' else 2
            q = clamp((g - step * w) / (4 if kind == 'word_fade_left' else 3))
            states[k]['alpha'] = states[k].get('alpha', 1) * (1 - q)
    elif kind == 'tracking_out':
        # 단어별 자간 벌어짐 + 축소 + 페이드, 왼쪽 단어부터 2프레임 간격(f210~f223)
        for k in range(n):
            w = widx[k]
            if w < 0:
                continue
            q = ease_out((g - 2.5 * w) / 9, 2)
            states[k]['space'] = .5 * q
            states[k]['alpha'] = states[k].get('alpha', 1) * (1 - q)
            states[k]['scale'] = states[k].get('scale', 1) * (1 - .3 * q)
    elif kind == 'squeeze_blur':
        q = clamp(g / 8)
        fade = (g - 2.5) / 5.5 * 1.15
        for k, st in enumerate(states):
            u = k / max(1, n - 1)
            st['sx'] = 1 - .45 * q
            st['blur'] = st.get('blur', 0) + 5 * ease_out(q, 2)
            st['alpha'] = st.get('alpha', 1) * (1 - clamp((fade - u) / .25))
    return states


def render_scene_60608725(scene, local, width, height, info):
    template = scene['template']
    text = ' '.join(split_words(scene['text']))
    fe = scene['duration'] * FPS_5 - template.get('delay', 0)
    f = local * FPS_5 - template.get('delay', 0)
    size = height * .083
    # 긴 문장은 폭 0.62W 안으로 줄인다.
    total = sum(char_mask(ch, int(size), 'Bold')[1] for ch in text)
    if total > width * .66:
        size *= width * .66 / total
    image = background('smoke_cool', width, height, scene['source_time'] + local * scene.get('source_rate', 1.0))
    layer = Image.new('RGBA', (width, height))
    states, _, tint = enter_states(template, text, f, fe)
    ex = exit_length(template, text)
    g = f - (fe - 1 - ex) if ex else -1
    if g >= 0:
        states = apply_exit(template, text, states, g)
    cx, cy = width / 2, height * .494
    if template.get('exit') == 'fade_slide_left' and g >= 0:
        cx -= width * .08 * ease_in(g / ex, 2)
    for st in states:
        if 'dy' in st:
            st['dy'] = st['dy'] * height
    recenter = template.get('exit') not in ('word_fade_left', 'tracking_out', 'fade_slide_left') or g < 0
    collapse = ease_out(clamp((g - 3) / 4.5), 2) if template.get('exit') == 'squeeze_blur' and g >= 3 else 0.0
    extent = compose_chars(layer, text, size, states, cx, cy, recenter=recenter, collapse=collapse)
    if tint and extent:
        tint_band(layer, extent, cy, size, *tint)
    # 커서(7) / 밑줄(2·9)
    if template['enter'] == 'cursor_type':
        shown = [k for k, st in enumerate(states) if st.get('visible', True) and st.get('alpha', 1) > .05]
        if shown:
            adv = [char_mask(ch, int(size), 'Bold')[1] for ch in text]
            vis_w = sum(adv[:shown[-1] + 1])
            x_end = cx + vis_w / 2 + size * .25
            if int(f // 6) % 2 == 0 or len(shown) < len(text):
                ImageDraw.Draw(layer).rectangle((x_end, cy - size * .48, x_end + max(2, size * .06), cy + size * .42), fill=(*SOFT_WHITE, 255))
    if template.get('underline') and extent:
        idx, _ = template['underline']
        if idx >= 0 and g >= 2:
            # 2번(f100~f108): 백스페이스 2프레임 뒤 문장 중간(글자 52% 지점) 아래에 짧은 녹→노랑 획이 생기고,
            # 지워지는 속도를 따라 왼쪽 첫 글자 쪽으로 이동하며 줄어 마지막 프레임에 점이 된다.
            q = clamp((g - 2) / max(1, ex - 2))
            center = len(text) * .52 * (1 - q ** 1.3)
            half = .3 + 2.2 * min(1, (g - 2) / 2) * (1 if q < .5 else 1 - (q - .5) / .5 * .9)
            a, b = char_x(extent, center - half), char_x(extent, center + half)
            tapered_stroke(layer, a, b, cy + size * .64, size * .045, [(96, 196, 64), (214, 228, 88)], sag=size * .02)
        elif idx < 0 and g >= -5:
            # 9번(f461~f473): '!' 아래 점에서 시작해 5프레임 동안 왼쪽으로 자라 마지막 단어+앞 글자까지 덮고,
            # 압축 퇴장 중에는 길이를 유지한 채 글자들과 함께 오른쪽 끝에 남는다.
            grow = ease_out((g + 5) / 5, 2)
            last_k = len(text) - 1
            right = char_x(extent, last_k + .7)
            span = char_x(extent, last_k + 1) - char_x(extent, max(0, last_k - 5.5)) if collapse == 0 else size * 1.6
            tapered_stroke(layer, right - span * grow, right, cy + size * .66, size * .045,
                           [(138, 82, 246), (214, 72, 236)], sag=size * .04)
    image.alpha_composite(layer)
    return image.convert('RGB')


# ---------------------------------------------------------------- 샘플 모듈 공통 도구(kit)

import os as _os

FONT_DIRS = [Path('C:/Windows/Fonts'), Path(_os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/Windows/Fonts',
             Path('/usr/share/fonts'), Path('/Library/Fonts'), Path.home() / 'Library/Fonts']


@lru_cache(maxsize=64)
def find_font_file(names):
    """설치된 서체 파일 중 이름이 맞는 첫 파일. 서체는 배포 패키지에 넣지 않고 대상 컴퓨터에서 찾는다."""
    for name in names:
        for folder in FONT_DIRS:
            if folder.is_dir():
                hit = folder / name
                if hit.exists():
                    return str(hit)
    return None


@lru_cache(maxsize=256)
def styled_font(style, size):
    """샘플별 서체 계열. 없으면 번들 Pretendard로 대체한다.
    rounded: 둥근 초굵은(배민 주아·카페24 써라운드) / light: 가는 그로테스크 / 그 외 Pretendard 굵기 이름."""
    size = max(4, int(round(size)))
    choices = {'rounded': ('BMJUA_ttf.ttf', 'Cafe24Ssurround.ttf', 'Cafe24Ohsquare.ttf'),
               'rounded_light': ('Cafe24SsurroundAir.ttf', 'Cafe24Ohsquareair.ttf')}
    if style in choices:
        path = find_font_file(choices[style])
        if path:
            return ImageFont.truetype(path, size)
        return font(size, 'ExtraBold' if style == 'rounded' else 'Regular')
    return font(size, style)


@lru_cache(maxsize=4096)
def text_mask(text, size, style='Bold', stretch=1.0, tracking=0.0):
    """여백을 잘라낸 글자 마스크와 (기준선 위 높이). stretch는 가로 배율, tracking은 em 단위 자간."""
    f = styled_font(style, int(size))
    ascent, descent = f.getmetrics()
    if tracking:
        advances = [f.getlength(ch) + tracking * size for ch in text]
        w = int(sum(advances)) + int(size)
    else:
        w = int(f.getlength(text)) + int(size)
    h = ascent + descent + int(size * .5)
    m = Image.new('L', (max(1, w), max(1, h)))
    d = ImageDraw.Draw(m)
    if tracking:
        x = size * .5
        for ch, adv in zip(text, advances):
            d.text((x, size * .25), ch, font=f, fill=255)
            x += adv
    else:
        d.text((size * .5, size * .25), text, font=f, fill=255)
    if stretch != 1.0:
        m = m.resize((max(1, int(m.width * stretch)), m.height), Image.Resampling.BICUBIC)
    box = m.getbbox() or (0, 0, 1, 1)
    pad = max(2, int(size * .06))
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(m.width, box[2] + pad), min(m.height, box[3] + pad))
    return m.crop(box)


def place(dst, sprite, cx, cy, scale=1.0, sx=None, rot=0.0, alpha=1.0, blur=0.0, motion=None):
    """paste_scaled + 회전(도) + 이동 방향 모션블러(motion=(dx, dy) 픽셀)."""
    if sprite is None or alpha <= .003 or scale <= .002:
        return
    sx = scale if sx is None else sx
    w, h = max(1, round(sprite.width * sx)), max(1, round(sprite.height * scale))
    im = sprite.resize((w, h), Image.Resampling.BICUBIC) if (w, h) != sprite.size else sprite
    if rot:
        im = im.rotate(rot, Image.Resampling.BICUBIC, expand=True)
    if motion and (abs(motion[0]) + abs(motion[1])) > 2:
        steps = int(min(12, max(3, (abs(motion[0]) + abs(motion[1])) / 6)))
        pad_x, pad_y = int(abs(motion[0])) + 2, int(abs(motion[1])) + 2
        trail = Image.new('RGBA', (im.width + pad_x * 2, im.height + pad_y * 2))
        for k in range(steps):
            q = k / (steps - 1)
            piece = im.copy()
            piece.putalpha(piece.getchannel('A').point(lambda a, q=q: round(a * (.25 + .75 * q) / steps * 2.2)))
            trail.alpha_composite(piece, (pad_x + int(-motion[0] * (1 - q)), pad_y + int(-motion[1] * (1 - q))))
        im = trail
    paste_scaled(dst, im, cx, cy, 1.0, alpha=alpha, blur=blur)


def tinted(mask, color, alpha=1.0):
    im = Image.new('RGBA', mask.size, (*color[:3], 0))
    im.putalpha(mask if alpha >= .999 else mask.point(lambda a: round(a * alpha)))
    return im


@lru_cache(maxsize=1024)
def extruded(text, size, style, face_top, face_bottom, side, depth=.12, stretch=1.0):
    """3D 압출 글자: 아래·오른쪽으로 두께, 앞면은 세로 그라데이션(57032891 근거)."""
    m = text_mask(text, size, style, stretch)
    d = max(2, int(size * depth))
    w, h = m.width + d + 2, m.height + d + 2
    out = Image.new('RGBA', (w, h))
    for k in range(d, 0, -1):
        shade = tuple(int(c * (.55 + .45 * (1 - k / d))) for c in side)
        out.alpha_composite(tinted(m, shade), (int(k * .55), k))
    out.alpha_composite(fill_image(m, 'vgrad', [face_top, face_bottom]), (0, 0))
    return out


def grain(im, amount, seed):
    """필름 그레인(3픽셀 단위 가우시안 노이즈)."""
    if amount <= 0:
        return im
    w, h = im.size
    noise = np.random.default_rng(seed).normal(0, amount, (h // 3 + 1, w // 3 + 1))
    noise = np.repeat(np.repeat(noise, 3, axis=0), 3, axis=1)[:h, :w, None]
    rgb = np.asarray(im.convert('RGB'), np.float32) + noise
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert('RGBA')


def stable_rng(*keys):
    """실행마다 같은 결과(결정적)를 주는 난수. 문자열 hash는 실행마다 달라지므로 직접 계산한다."""
    acc = 2166136261
    for key in keys:
        for ch in str(key):
            acc = ((acc ^ ord(ch)) * 16777619) % (2 ** 32)
    return np.random.default_rng(acc)


def block_layout(lines, size, styles, line_gap=1.0, stretch=1.0, tracking=0.0, unit='word'):
    """여러 줄 블록을 가운데 정렬로 배치한다. 반환: 항목 목록(줄·순번·문자열·마스크·중심 x,y 상대 좌표)과 블록 높이.
    unit='word'는 단어 단위, 'char'는 글자 단위 항목."""
    items, y = [], 0.0
    heights = []
    rows = []
    for row, line in enumerate(lines):
        style = styles[row % len(styles)]
        f = styled_font(style, int(size))
        space = f.getlength(' ') + tracking * size
        parts = list(line.replace(' ', '')) if unit == 'char' else split_words(line)
        if unit == 'char':
            # 글자 단위는 공백 위치를 유지한다.
            parts = list(line)
        widths = [f.getlength(p) * stretch + (tracking * size if unit == 'char' else 0) for p in parts]
        total = sum(widths) + (space * (len(parts) - 1) if unit == 'word' else 0)
        x = -total / 2
        row_items = []
        for k, (p, w) in enumerate(zip(parts, widths)):
            if p.strip():
                row_items.append({'row': row, 'k': k, 'text': p, 'style': style, 'cx': x + w / 2, 'w': w})
            x += w + (space if unit == 'word' else 0)
        rows.append(row_items)
        heights.append(size * line_gap)
    total_h = sum(heights)
    y = -total_h / 2
    for row_items, hgt in zip(rows, heights):
        for it in row_items:
            it['cy'] = y + hgt / 2
            items.append(it)
        y += hgt
    for n, it in enumerate(items):
        it['n'] = n
    return items, total_h


# ---------------------------------------------------------------- 샘플 등록부

import importlib.util as _importlib_util


class Sample:
    """샘플 하나의 공통 인터페이스. 각 샘플 모듈(scripts/samples/s<id>.py)은 같은 이름의 함수를 제공한다."""

    def __init__(self, identifier, preset, fps, templates, assign, minimum, render, hit=None, smoke=None):
        self.id, self.preset, self.fps, self.templates = identifier, preset, fps, templates
        self.assign, self.minimum, self.render = assign, minimum, render
        self.hit = hit or (lambda scene: 0.0)
        self.smoke = smoke  # (원본 길이) — 연기 배경 진행을 영상 전체에 한 번만 대응할 때


def fits(template, words, chars):
    kind = template.get('kind', 'line')
    if kind == 'word':
        return len(words) == 1 and chars <= 8
    if kind in ('two', 'flip'):
        return len(words) >= 3
    if kind == 'multi':
        return len(words) >= 2
    return len(words) >= 1 and not (len(words) == 1 and chars <= 3)


def assign_in_order(pool, paragraphs, fit=fits, last=None):
    """원본 장면 순서를 따라가며 문구 형태가 맞는 다음 템플릿을 고른다. 같은 템플릿 연속 사용을 피한다.
    last가 있으면 마지막 문구는 원본 마지막 장면으로 맺는다."""
    out, pointer = [], 0
    for k, text in enumerate(paragraphs):
        if last is not None and k == len(paragraphs) - 1 and len(paragraphs) > 1:
            out.append(last)
            break
        words = split_words(text)
        chars = len(''.join(words))
        chosen = None
        for step in range(len(pool)):
            tpl = pool[(pointer + step) % len(pool)]
            if fit(tpl, words, chars) and not (out and out[-1] is tpl):
                chosen = (pointer + step) % len(pool)
                break
        if chosen is None:
            chosen = pointer % len(pool)
        out.append(pool[chosen])
        pointer = chosen + 1
    return out


def assign_11971627(paragraphs):
    return assign_in_order(TEMPLATES_11971627, paragraphs)


def assign_60608725(paragraphs):
    """원본 10문구 순서를 반복하고, 마지막 문구는 원본 마지막(녹색 대형 타이핑)으로 맺는다."""
    loop = TEMPLATES_60608725[:-1]
    out = [loop[k % len(loop)] for k in range(len(paragraphs))]
    if out:
        out[-1] = TEMPLATES_60608725[-1]
    return out


def minimum_11971627(template, text):
    words = split_words(text)
    chars = len(''.join(words))
    read = max(.25, chars / 24)
    enter = template.get('enter')
    if enter in ('type', 'type_scroll'):
        frames = len(' '.join(words)) * template.get('per_char', 2.0) + 4
    elif enter == 'words':
        frames = 1 + 3 * len(words) + (12 if template.get('star') == 'intro_head' else 0)
    else:
        frames = 10
    return frames / FPS_1 + read + 8 / FPS_1


def minimum_60608725(template, text):
    words = split_words(text)
    chars = len(''.join(words))
    read = max(.35, chars / 14)
    t = ' '.join(words)
    n = len(t)
    enter = {'hero_type': 14 + n / 1.4, 'hero_word': 12 + n / 1.5, 'word_fade': 7 + n / 1.4, 'type': n + 2,
             'cursor_type': 1.3 * n + 2, 'gradient_focus': 15 + 18 * max(0, len(words) - 1), 'rise_words': 8 + 2 * len(words),
             'blur_focus': 14 + 2.5 * len(words), 'green_type_big': 2.3 * n + 22}[template['enter']]
    return (template.get('delay', 0) + enter + exit_length(template, t)) / FPS_5 + read


REGISTRY = {}


def register(sample):
    REGISTRY[sample.id] = sample


register(Sample('11971627', 'purple_words', FPS_1, TEMPLATES_11971627, assign_11971627, minimum_11971627,
                render_scene_11971627))
register(Sample('60608725', 'smoke_cool', FPS_5, TEMPLATES_60608725, assign_60608725, minimum_60608725,
                render_scene_60608725, hit=lambda scene: scene['template'].get('delay', 0) / FPS_5, smoke=22.9))


def load_sample_modules():
    """scripts/samples/s<id>.py를 불러와 등록한다. 각 모듈은 build(engine)으로 Sample을 돌려준다."""
    folder = Path(__file__).resolve().parent / 'samples'
    if not folder.is_dir():
        return
    import sys as _sys
    for path in sorted(folder.glob('s*.py')):
        spec = _importlib_util.spec_from_file_location('text_motion_sample_' + path.stem, path)
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register(module.build(_sys.modules[__name__] if __name__ in _sys.modules else globals_module()))


def globals_module():
    import types
    module = types.ModuleType('frame_spec_engine_globals')
    module.__dict__.update(globals())
    return module


SAMPLES = {}
PRESET_SAMPLE = {}


def refresh_index():
    SAMPLES.clear()
    PRESET_SAMPLE.clear()
    for identifier, sample in REGISTRY.items():
        SAMPLES[identifier] = sample.preset
        PRESET_SAMPLE[sample.preset] = identifier


def minimum_seconds(sample, template, text):
    return REGISTRY[sample].minimum(template, text)


def make_timeline(text, preset, target_seconds=None, pacing='tight'):
    sample = REGISTRY[PRESET_SAMPLE[preset]]
    raw = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
    if not raw:
        raise ValueError('제작 문구가 비어 있습니다.')
    paragraphs = [' '.join(split_words(x)) for x in raw]
    templates = sample.assign(paragraphs)
    fps = sample.fps
    base, minimum = [], []
    for tpl, para in zip(templates, paragraphs):
        m = sample.minimum(tpl, para)
        minimum.append(m)
        base.append(max(tpl['frames'] / fps, m))
    natural = sum(base)
    target = natural if target_seconds is None else float(target_seconds)
    if target + 1e-6 < sum(minimum):
        raise ValueError(f'현재 문구는 최소 {sum(minimum):.1f}초가 필요합니다. 문구를 줄이거나 길이를 늘리세요.')
    if target > natural * 1.12:
        need = math.ceil((target / 1.12 - natural) / max(.1, natural / len(base)))
        raise ValueError(f'문구가 부족합니다. 원본 리듬 기준 자연 길이는 {natural:.1f}초이고 1.12배 이상 늘리면 전환이 처집니다. '
                         f'원문에서 장면을 약 {need}개 더 뽑으세요(긴 정지로 채우지 않음).')
    if target >= natural:
        durations = [b * target / natural for b in base]
    else:
        spare = [b - m for b, m in zip(base, minimum)]
        cut = (natural - target) / max(1e-9, sum(spare))
        durations = [b - sp * cut for b, sp in zip(base, spare)]
    scenes, cursor = [], 0.0
    for k, (tpl, para, raw_text, d, m) in enumerate(zip(templates, paragraphs, raw, durations, minimum)):
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        scenes.append({'index': k, 'text': para, 'lines': lines if len(lines) > 1 else None, 'template': tpl,
                       'template_n': tpl['n'], 'sample': sample.id, 'start': cursor, 'duration': d, 'minimum_duration': m,
                       'motion': tpl.get('enter'), 'exit': tpl.get('exit'), 'renderer': VERSION, 'reference_id': sample.id,
                       'reveal': min(d * .5, .6), 'hold': d * .4, 'transition': 8 / fps, 'pacing': pacing})
        cursor += d
    return scenes, target


def estimate_beat(events, duration):
    """킥·스네어 후보 시각으로 박 길이(0.4~1.0초)와 위상을 추정한다. 실패하면 None."""
    times = sorted(e['time'] for e in events if e.get('kind') in ('kick', 'snare_candidate'))
    if len(times) < 8:
        return None
    best, best_score = None, -1
    for period in np.arange(.40, 1.0, .002):
        # 각 후보 박 길이로 위상 정렬 정도를 잰다(반 박 위치도 절반 점수).
        ph = (np.array(times) / period) % 1
        score = np.abs(np.mean(np.exp(2j * np.pi * ph))) + .5 * np.abs(np.mean(np.exp(4j * np.pi * ph)))
        score *= 1 - .15 * abs(period - .55)  # 지나치게 긴 박을 약하게 감점
        if score > best_score:
            best, best_score = period, score
    kicks = [e['time'] for e in events if e.get('kind') == 'kick'] or times
    ph = np.angle(np.mean(np.exp(2j * np.pi * (np.array(kicks) / best)))) / (2 * np.pi)
    return {'period': float(best), 'phase': float((ph % 1) * best), 'score': round(float(best_score), 3)}


def hit_offset(scene):
    """장면에서 '첫 타격'이 보이는 시각(컷 기준)."""
    return REGISTRY[scene['sample']].hit(scene)


def quantize_to_beat(project, beat):
    """장면 길이를 반 박 단위로 맞추고, 각 장면의 첫 타격이 반 박 격자에 떨어지게 컷을 놓는다."""
    scenes = project['scenes']
    total = project['duration']
    grid = beat['period'] / 2
    fps = REGISTRY[scenes[0]['sample']].fps
    natural = [sc['template']['frames'] / fps for sc in scenes]
    min_units = [max(1, math.ceil(sc['minimum_duration'] / grid - .15)) for sc in scenes]
    units = [max(m, round(nat / grid)) for nat, m in zip(natural, min_units)]
    target_units = max(sum(min_units), round(total / grid))
    # 모자라면 원본 대비 가장 짧아진 장면부터, 남으면 여유가 큰 장면부터 반 박씩 조정
    while sum(units) < target_units:
        k = min(range(len(units)), key=lambda i: units[i] * grid / natural[i])
        units[k] += 1
    while sum(units) > target_units:
        spare = [i for i in range(len(units)) if units[i] > min_units[i]]
        if not spare:
            break
        k = max(spare, key=lambda i: units[i] * grid / natural[i])
        units[k] -= 1
    # 첫 장면의 첫 타격을 첫 격자점에 맞춘다.
    first_grid = beat['phase'] % grid
    while first_grid < hit_offset(scenes[0]):
        first_grid += grid
    points = [first_grid]
    for u in units[:-1]:
        points.append(points[-1] + u * grid)
    starts = [0.0] + [pt - hit_offset(sc) for pt, sc in zip(points[1:], scenes[1:])]
    for i, sc in enumerate(scenes):
        end = starts[i + 1] if i + 1 < len(scenes) else total
        sc['start'], sc['duration'] = starts[i], max(1 / fps, end - starts[i])
        sc['beat_units'] = units[i]
    project['beat'] = dict(beat, grid=grid, method='kick_snare_phase_coherence_v1',
                           note='박 추정은 자동 추정이며 청취 확인이 필요하다.')


def bind_rhythm(project, rhythm):
    """음원이 있으면 박 격자에 장면을 맞추고, 없으면 원본 프레임 길이를 쓴다."""
    project['rhythm'] = rhythm
    scenes = project['scenes']
    beat = estimate_beat(rhythm.get('events', []), project['duration']) if rhythm.get('events') else None
    if beat:
        quantize_to_beat(project, beat)
    total = project['duration']
    for scene in scenes:
        smoke = REGISTRY[scene['sample']].smoke
        if smoke:
            # 연기 배경은 원본 진행을 영상 전체에 한 번만 대응한다.
            scene['source_rate'] = smoke / max(.1, total)
            scene['source_time'] = scene['start'] * scene['source_rate']
        scene['project_duration'] = total
        scene['scene_count'] = len(scenes)
    project['renderer'] = VERSION
    return project


def project_frame(project, t, width=640, height=360):
    scenes = project['scenes']
    i = 0
    for k, scene in enumerate(scenes):
        if scene['start'] <= t:
            i = k
    scene = scenes[i]
    sample = REGISTRY[scene['sample']]
    if 'project_duration' not in scene:
        bind_rhythm(project, project.get('rhythm') or {'events': []})
    local = min(max(0.0, t - scene['start']), scene['duration'] - 1e-4)
    info = dict(project.get('overlay', {'label': '', 'footer': ''}), t=t, previous=scenes[i - 1] if i else None)
    return sample.render(scene, local, width, height, info)


def frame(scene, preset, local, width=640, height=360):
    project = {'scenes': [dict(scene, start=0.0)], 'duration': scene['duration'], 'preset': preset}
    bind_rhythm(project, {'events': []})
    return project_frame(project, local, width, height)


load_sample_modules()
refresh_index()
