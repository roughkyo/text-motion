"""60153473: 장면마다 완전히 다른 RGB·잔상 변형 장치, 흰 글자+블룸, 동심원 배경 (30fps).

근거: references/analysis/60153473-spec.md (1프레임 시트 19장 전체).
"""
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

E = None
FPS = 30.0
RAINBOW = [(255, 50, 60), (255, 200, 40), (60, 230, 90), (50, 120, 255), (255, 255, 255)]
BAND = [(40, 90, 255), (40, 220, 230), (255, 255, 255), (255, 220, 40), (255, 50, 60)]

TEMPLATES = [
    {'n': 1, 'frames': 39, 'device': 'ribbon_type', 'enter': .75},
    {'n': 2, 'frames': 29, 'device': 'rgb_drop', 'enter': .7},
    {'n': 3, 'frames': 43, 'device': 'rgb_giant', 'enter': 1.0, 'tilt': -7},
    {'n': 4, 'frames': 49, 'device': 'rgb_blur', 'enter': .85, 'exit': .45},
    {'n': 5, 'frames': 77, 'device': 'outline_burst', 'enter': 2.1},
    {'n': 6, 'frames': 67, 'device': 'thermal', 'enter': 1.9},
    {'n': 7, 'frames': 70, 'device': 'diagonal_band', 'enter': 1.5},
    {'n': 8, 'frames': 41, 'device': 'pixel_glitch', 'enter': .9},
    {'n': 9, 'frames': 66, 'device': 'rainbow_extrude', 'enter': 2.1, 'dir': (0, 1)},
    {'n': 10, 'frames': 107, 'device': 'outline_burst', 'enter': 1.4, 'italic': True},
    {'n': 11, 'frames': 87, 'device': 'rainbow_extrude', 'enter': 2.8, 'dir': (.55, .85)},
    {'n': 12, 'frames': 77, 'device': 'diagonal_band', 'enter': 1.5, 'two': True},
]


def fit(template, words, chars):
    if template['device'] in ('rainbow_extrude', 'rgb_giant'):
        return chars <= 10
    return True


def assign(paragraphs):
    return E.assign_in_order(TEMPLATES, paragraphs, fit=fit)


def minimum(template, text):
    return template['enter'] + template.get('exit', 0) + max(.5, len(text.replace(' ', '')) / 14)


def background(width, height, frame_index, phase):
    im = Image.new('RGBA', (width, height), (5, 5, 7, 255))
    draw = ImageDraw.Draw(im)
    s = width / 640
    for r, g in ((.16 + .004 * math.sin(phase), 30), (.45 + .006 * math.sin(phase * .7), 22)):
        rr = height * r
        draw.ellipse((width / 2 - rr, height / 2 - rr, width / 2 + rr, height / 2 + rr), outline=(g, g, g + 4, 255), width=max(1, int(s)))
    for k in (-1, 0, 1):
        x = width / 2 + k * 8 * s
        draw.ellipse((x - s, height * .05 - s, x + s, height * .05 + s), fill=(150, 150, 155, 255))
    return E.grain(im, 4, frame_index)


def text_layers(lines, size, italic=False):
    """줄별 마스크와 배치(가운데 정렬)."""
    out = []
    for ln in lines:
        m = E.text_mask(ln, int(size), 'ExtraBold')
        if italic:
            w, h = m.size
            m = m.transform((int(w + h * .25), h), Image.Transform.AFFINE, (1, .25, -h * .25, 0, 1, 0), Image.Resampling.BICUBIC)
        out.append(m)
    return out


def white_bloom(layer, height):
    return E.bloom(layer, height * .02, .55, (235, 240, 255))


def render(scene, local, width, height, info):
    tpl = scene['template']
    dev = tpl['device']
    words = E.split_words(scene['text'])
    lines = scene.get('lines') or ([' '.join(x) for x in E.balanced_lines(words)] if (tpl.get('two') or len(scene['text']) > 18) and len(words) > 1 else [scene['text']])
    # 원본 정착 크기: 짧은 단어는 폭 약 0.4W·캡 0.12H, 긴 문구는 폭 0.6W 이하
    size = height * .2
    widest = max(E.styled_font('ExtraBold', int(size)).getlength(ln) for ln in lines)
    size = min(size, size * width * .6 / max(1, widest), height * .42 / len(lines))
    masks = text_layers(lines, size, tpl.get('italic'))
    line_h = size * 1.05
    cx, cy = width / 2, height / 2
    t = local
    p_enter = E.clamp(t / tpl['enter'])
    frame_index = int(round(info.get('t', t) * FPS))
    image = background(width, height, frame_index, info.get('t', t))
    layer = Image.new('RGBA', (width, height))
    rows = [(m, cy + (r - (len(masks) - 1) / 2) * line_h) for r, m in enumerate(masks)]
    rng = E.stable_rng(scene.get('index', 0), dev)
    if dev == 'ribbon_type':
        for m, y in rows:
            w = m.width
            shown = E.clamp(t / (tpl['enter'] * .55))
            head = cx - w / 2 + w * shown
            cut = int(w * shown)
            if cut > 0:
                E.place(layer, E.tinted(m.crop((0, 0, cut, m.height)), (255, 255, 255)), cx - w / 2 + cut / 2, y)
            # 글자 뒤로 오른쪽으로 길게 늘어지는 무지개 물결 띠(f1~f22). 타이핑이 끝나면 오른쪽 끝으로 빨려 들어간다.
            length = width * .55 * (1 - E.clamp((t - tpl['enter'] * .5) / (tpl['enter'] * .5)))
            if length > 2:
                band_h = m.height * .55
                amp = band_h * .9 * (1 - shown * .7)
                draw = ImageDraw.Draw(layer)
                stripe = band_h / len(RAINBOW)
                for k in range(0, int(length), 3):
                    x = head + k
                    off = amp * math.sin(k / (width * .06) - t * 9) * (1 - k / max(1, length)) ** .5
                    for j, c in enumerate(RAINBOW):
                        y0 = y - band_h / 2 + off + j * stripe
                        draw.rectangle((x, y0, x + 3, y0 + stripe), fill=(*c, 230))
    elif dev == 'rgb_drop':
        word_items, _ = E.block_layout(lines, size, ['ExtraBold'], line_gap=1.05, unit='word')
        for it in word_items:
            start = .05 + .14 * it['n']
            p = E.ease_out((t - start) / .35, 3)
            if t < start:
                continue
            m = E.text_mask(it['text'], int(size), 'ExtraBold')
            x, y = cx + it['cx'], cy + it['cy'] + height * .4 * (1 - p)
            trail = (1 - p)
            for k, c in enumerate(((255, 50, 60), (60, 230, 90), (50, 120, 255))):
                for j in range(1, 4):
                    E.place(layer, E.tinted(m, c, .45 * trail * (1 - j / 4)), x + (k - 1) * size * .04, y + m.height * j * .7 * trail)
            E.place(layer, E.tinted(m, (255, 255, 255)), x, y)
    elif dev == 'rgb_giant':
        p = E.ease_out(p_enter, 3)
        s = 2.6 - 1.6 * p
        sep = width * .05 * (1 - p)
        for m, y in rows:
            for c, sign in (((255, 40, 50), -1), ((40, 255, 80), 0), ((50, 80, 255), 1)):
                E.place(layer, E.tinted(m, c, .9 if sep > 1 else 0), cx + sign * sep, y, s, rot=tpl['tilt'], motion=(sep * 2, 0))
            E.place(layer, E.tinted(m, (255, 255, 255)), cx, y, s, rot=tpl['tilt'], alpha=p)
        dot = height * .012
        ImageDraw.Draw(layer).ellipse((cx - width * .3 - dot, cy - dot, cx - width * .3 + dot, cy + dot), fill=(255, 255, 255, int(255 * (1 - p))))
    elif dev == 'rgb_blur':
        ext = tpl.get('exit', 0)
        g = t - (scene['duration'] - ext)
        q = 1 - E.ease_out(p_enter, 2) if g < 0 else E.ease_in(E.clamp(g / ext), 2)
        for m, y in rows:
            sep = width * .03 * q
            for c, sign in (((255, 40, 50), -1), ((50, 120, 255), 1)):
                E.place(layer, E.tinted(m, c, .8 * q), cx + sign * sep, y, motion=(width * .06 * q, 0))
            gray = E.mix((255, 255, 255), (120, 120, 130), q)
            E.place(layer, E.tinted(m, gray), cx, y, motion=(width * .08 * q, 0))
    elif dev == 'outline_burst':
        for r, (m, y) in enumerate(rows):
            outline = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(3)), m.filter(ImageFilter.MinFilter(3)))
            a = E.clamp(t / (tpl['enter'] * .35))
            right = cx + m.width / 2
            if a < 1:
                # 오른쪽 끝 한 점에서 외곽선 복제들이 왼쪽으로 원근 있게 뿜어 나온다(f164~f182)
                for k in range(14):
                    q = k / 13
                    s = .05 + .95 * E.ease_out(a, 2) * (1 - q * .7)
                    x = right - (right - cx) * E.ease_out(a, 2) * (1 - q * .5) - width * .15 * q * (1 - a)
                    E.place(layer, E.tinted(outline, (255, 255, 255), .5 * (1 - q)), x + (m.width / 2) * (1 - s) * 0, y - height * .05 * q * (1 - a), s)
            else:
                b = E.clamp((t - tpl['enter'] * .55) / (tpl['enter'] * .45))
                fill_row = r == len(rows) - 1 and tpl.get('italic')
                E.place(layer, E.tinted(outline, (255, 255, 255)), cx, y)
                if b > 0 or fill_row:
                    bb = 1 if fill_row and b == 0 else b
                    cut = int(m.width * (1 - E.ease_out(bb, 2)))
                    filled = m.crop((cut, 0, m.width, m.height))
                    E.place(layer, E.tinted(filled, (255, 255, 255)), cx - m.width / 2 + cut + filled.width / 2, y)
                    if bb < 1:
                        sx = cx - m.width / 2 + cut
                        draw = ImageDraw.Draw(layer)
                        for j, c in enumerate(RAINBOW[:4]):
                            draw.rectangle((sx + j * size * .05, y - m.height / 2, sx + j * size * .05 + size * .04, y + m.height / 2), fill=(*c, 200))
    elif dev == 'thermal':
        p = E.ease_out(p_enter, 2)
        blur = height * .03 * (1 - p)
        for m, y in rows:
            if p < .25:
                E.place(layer, E.tinted(m, (60, 110, 255), .6), cx, y, .4 + p * 2, blur=height * .05)
                continue
            for c, d in (((255, 140, 40), 1), ((40, 120, 255), -1)):
                E.place(layer, E.tinted(m, c, .8 * (1 - p)), cx + d * size * .05 * (1 - p), y, 1.0, blur=blur * 1.3)
            E.place(layer, E.tinted(m, (255, 255, 255)), cx, y, 1.0, blur=blur * .8, alpha=.4 + .6 * p)
    elif dev == 'diagonal_band':
        a = E.clamp(t / tpl['enter'])
        for r, (m, y) in enumerate(rows):
            shown = E.clamp((a - r * .15) / .6)
            cut = int(m.width * shown)
            if cut > 0:
                E.place(layer, E.tinted(m.crop((0, 0, cut, m.height)), (255, 255, 255)), cx - m.width / 2 + cut / 2, y)
        if a < 1:
            # 화면을 대각선으로 가로지르는 굵은 단색 띠, 1~2프레임마다 색이 바뀌며 수평·가늘어짐(f308~f353)
            angle = 32 * (1 - E.ease_out(a, 2))
            thick = height * .26 * (1 - a) + 1
            color = BAND[(frame_index // 2) % len(BAND)]
            band = Image.new('RGBA', (int(width * 1.6), int(thick)), (*color, 235))
            E.place(layer, band, cx, cy, 1.0, rot=angle)
            # 띠 앞에 글자를 다시 얹어 띠 안에서 글자가 생기는 느낌
            for r, (m, y) in enumerate(rows):
                shown = E.clamp((a - r * .15) / .6)
                cut = int(m.width * shown)
                if cut > 0:
                    E.place(layer, E.tinted(m.crop((0, 0, cut, m.height)), (10, 10, 14)), cx - m.width / 2 + cut / 2, y, alpha=.5)
    elif dev == 'pixel_glitch':
        p = E.ease_out(p_enter, 2)
        for m, y in rows:
            bw = max(4, int(size * .22))
            for bx in range(0, m.width, bw):
                for by in range(0, m.height, bw):
                    piece = m.crop((bx, by, bx + bw, by + bw))
                    if not piece.getbbox():
                        continue
                    start = rng.uniform(0, .6)
                    q = E.clamp((p_enter - start) / .4)
                    if q <= 0:
                        continue
                    off = rng.uniform(-1, 1) * size * (1 - q)
                    color = E.mix((40, 90, 255) if rng.random() < .5 else (40, 220, 230), (255, 255, 255), q)
                    E.place(layer, E.tinted(piece, color), cx - m.width / 2 + bx + bw / 2 + off, y - m.height / 2 + by + bw / 2)
    elif dev == 'rainbow_extrude':
        dx, dy = tpl['dir']
        for m, y in rows:
            phase = t / tpl['enter']
            length = height * .45 * abs(math.sin(min(1.0, phase) * math.pi * 1.5)) if phase < 1 else height * .05
            yy = y - height * .15
            steps = int(max(1, length / 3))
            for k in range(steps, 0, -1):
                q = k / steps
                c = RAINBOW[min(3, int(q * 4))]
                E.place(layer, E.tinted(m, E.mix(c, (255, 255, 255), .1)), cx + dx * length * q, yy + dy * length * q)
            E.place(layer, E.tinted(m, (255, 255, 255)), cx, yy)
    image.alpha_composite(white_bloom(layer, height))
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('60153473', 'rgb_trail', FPS, TEMPLATES, assign, minimum, render)
