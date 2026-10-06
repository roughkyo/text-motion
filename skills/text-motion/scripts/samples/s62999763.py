"""62999763: 슬롯 머신 첫 글자, 분산 글자 조립+막대, 보케/검정/흰 배경 하드컷, 점 커서 (24fps).

근거: references/analysis/62999763-spec.md (1프레임 시트 18장).
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

E = None
FPS = 24.0
WHITE, BLACK, NAVY_INK = (255, 255, 255), (14, 14, 20), (58, 58, 160)
SLOT_HANGUL = '가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호'
SLOT_LATIN = 'ZYXWVUTSRQPONMLKJIHGFEDCBA'

TEMPLATES = [
    {'n': 1, 'frames': 24, 'device': 'blur_up', 'bg': 'bokeh', 'kind': 'word'},
    {'n': 2, 'frames': 37, 'device': 'slot', 'bg': 'black', 'kind': 'word'},
    {'n': 3, 'frames': 60, 'device': 'word_glow', 'bg': 'black', 'kind': 'line'},
    {'n': 4, 'frames': 30, 'device': 'slot', 'bg': 'black', 'kind': 'line', 'gap_shrink': True},
    {'n': 5, 'frames': 50, 'device': 'append', 'bg': 'white', 'kind': 'line'},
    {'n': 6, 'frames': 40, 'device': 'metal_track', 'bg': 'black', 'kind': 'word'},
    {'n': 7, 'frames': 36, 'device': 'scatter', 'bg': 'white', 'kind': 'word', 'grad': ((92, 92, 200), (24, 24, 70))},
    {'n': 8, 'frames': 40, 'device': 'type_dot', 'bg': 'black', 'kind': 'line'},
    {'n': 9, 'frames': 36, 'device': 'scatter', 'bg': 'black', 'kind': 'word', 'grad': ((240, 110, 220), (150, 110, 250))},
    {'n': 10, 'frames': 40, 'device': 'type_dot', 'bg': 'black', 'kind': 'line'},
    {'n': 11, 'frames': 40, 'device': 'wide_swap', 'bg': 'dark', 'kind': 'line'},
    {'n': 12, 'frames': 45, 'device': 'append', 'bg': 'white', 'kind': 'line'},
    {'n': 13, 'frames': 30, 'device': 'slot', 'bg': 'white', 'kind': 'line'},
    {'n': 14, 'frames': 60, 'device': 'glow_swap', 'bg': 'black', 'kind': 'line'},
    {'n': 15, 'frames': 37, 'device': 'slot', 'bg': 'black', 'kind': 'word', 'end': True},
]
LAST = TEMPLATES[-1]
ENTER = {'blur_up': .5, 'slot': .45, 'word_glow': .6, 'append': .35, 'metal_track': .4, 'scatter': .6, 'type_dot': .6,
         'wide_swap': .7, 'glow_swap': .8}


def fit(template, words, chars):
    if template['kind'] == 'word':
        return chars <= 8
    if template['device'] == 'append':
        return len(words) >= 2
    return True


def assign(paragraphs):
    return E.assign_in_order(TEMPLATES[:-1], paragraphs, fit=fit, last=LAST)


def minimum(template, text):
    words = E.split_words(text)
    extra = .08 * len(words) if template['device'] in ('type_dot', 'word_glow', 'wide_swap', 'glow_swap') else 0
    return ENTER[template['device']] + extra + max(.4, len(text.replace(' ', '')) / 15)


def background(kind, width, height, t, seed):
    if kind == 'white':
        im = Image.new('RGBA', (width, height), (251, 251, 253, 255))
    elif kind == 'black':
        im = Image.new('RGBA', (width, height), (4, 4, 7, 255))
    elif kind == 'dark':
        im = E.radial(width, height, .1, .1, .7, .8, (28, 40, 44), (6, 6, 9), 1.3)
    else:
        # 보케: 아주 흐린 큰 빛 덩어리(청록·보라·청색·주황)
        small = Image.new('RGBA', (160, 90), (8, 10, 16, 255))
        d = ImageDraw.Draw(small)
        blobs = [((40, 140, 140), .3, .15, .45), ((90, 70, 190), .15, .6, .3), ((50, 110, 240), .5, .7, .28), ((240, 150, 90), .75, 1.0, .18)]
        for c, x, y, r in blobs:
            x += .02 * math.sin(t * .7 + r * 10)
            d.ellipse(((x - r) * 160, (y - r) * 90, (x + r) * 160, (y + r) * 90), fill=(*c, 255))
        im = small.filter(ImageFilter.GaussianBlur(14)).resize((width, height), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(im)
    c = (210, 210, 215, 255) if kind == 'white' else (60, 60, 66, 255)
    draw.line((width * .44, height * .93, width * .56, height * .93), fill=c, width=max(1, height // 540))
    return im


def slot_char(target, k, frame_seed):
    pool = SLOT_HANGUL if '가' <= target <= '힣' else SLOT_LATIN
    return pool[(k * 3 + frame_seed) % len(pool)]


def render(scene, local, width, height, info):
    tpl = scene['template']
    dev = tpl['device']
    t = local
    f = t * FPS
    image = background(tpl['bg'], width, height, info.get('t', t), scene.get('index', 0))
    layer = Image.new('RGBA', (width, height))
    ink = BLACK if tpl['bg'] == 'white' else WHITE
    text = scene['text']
    words = E.split_words(text)
    base = height * (.16 if tpl['kind'] == 'word' else .105)  # 원본 정착 크기(Digital 캡 약 0.12H)
    size = min(base, base * width * .66 / max(1, E.styled_font('Bold', int(base)).getlength(text)))
    cx, cy = width / 2, height / 2
    font = E.styled_font('Bold', int(size))
    if dev == 'blur_up':
        p = E.ease_out(E.clamp(t / .5), 3)
        m = E.text_mask(text, int(size), 'Bold')
        sprite = E.fill_image(m, 'vgrad', [(150, 150, 250), (90, 110, 230)])
        E.place(layer, sprite, cx, cy + height * .12 * (1 - p), .7 + .3 * p, blur=(1 - p) * 6 * height / 1080, alpha=.3 + .7 * p)
    elif dev == 'slot':
        # 첫 글자가 1프레임씩 다른 글자로 넘어가다 멈추고(f24~f32), 단어 전체는 아래에서 올라와 정착
        spin_end = 8
        p = E.ease_out(E.clamp(f / 6), 3)
        shown = text if f >= spin_end else slot_char(text[0], int(f), scene.get('index', 0)) + text[1:]
        if tpl.get('gap_shrink') and len(words) > 1:
            gap = 1 + .8 * (1 - E.ease_out(E.clamp(f / 10), 2))
            items, _ = E.block_layout([shown], size, ['Bold'], unit='word')
            for it in items:
                E.place(layer, E.tinted(E.text_mask(it['text'], int(size), 'Bold'), ink), cx + it['cx'] * gap, cy)
        else:
            E.place(layer, E.tinted(E.text_mask(shown, int(size), 'Bold'), ink), cx, cy + height * .1 * (1 - p))
    elif dev in ('word_glow', 'type_dot', 'glow_swap', 'wide_swap', 'append'):
        items, _ = E.block_layout([text], size, ['Bold'], unit='word')
        step = {'word_glow': .16, 'type_dot': .1, 'glow_swap': .12, 'wide_swap': .3, 'append': 0}[dev]
        last_x = None
        for it in items:
            start = step * it['n']
            if dev == 'append':
                start = 0 if it['n'] < len(items) - 1 else .05
            if t < start:
                continue
            p = E.ease_out((t - start) / .15, 2)
            m = E.text_mask(it['text'], int(size), 'Bold')
            x, y = cx + it['cx'], cy
            last = it['n'] == len(items) - 1
            if dev == 'word_glow' and last:
                glow = E.tinted(m, (80, 140, 255))
                E.place(layer, E.bloom(glow, size * .3, .9, (60, 120, 255)), x, y, alpha=p)
            elif dev == 'glow_swap' and last:
                # 강조 단어: 강한 색 블룸, 파랑 → 분홍 → 보라(f630~f660)
                q = E.clamp((t - start) / .6)
                c = E.mix((48, 128, 255), (230, 60, 160), q * 2) if q < .5 else E.mix((230, 60, 160), (120, 90, 250), (q - .5) * 2)
                E.place(layer, E.bloom(E.tinted(m, c), size * .35, 1.0, c), x, y, alpha=p, blur=(1 - p) * 4)
            elif dev == 'wide_swap':
                # 오른쪽에서 넓은 자간·블러로 들어와 좁혀지며 정착
                wide = E.text_mask(it['text'], int(size), 'Bold', 1.0, .5 * (1 - p))
                E.place(layer, E.tinted(wide, E.mix((120, 120, 128), ink, p)), x + width * .15 * (1 - p), y, blur=3 * (1 - p), alpha=.4 + .6 * p)
            elif dev == 'append':
                # 흰 화면으로 바뀌며 같은 문장에 단어가 붙고, 남색 → 검정으로 0.3초 동안 바뀜(f137~f148)
                c = E.mix(NAVY_INK, BLACK, E.clamp(t / .35))
                E.place(layer, E.tinted(m, c), x, y, alpha=1 if not last else p)
            else:
                E.place(layer, E.tinted(m, ink), x, y, alpha=p)
            last_x = x + it['w'] / 2
        if dev == 'type_dot' and last_x is not None:
            # 문장 뒤 작은 점 커서: 세로 막대 ↔ 점, 청록 → 초록
            phase = int(f / 3) % 2
            c = E.mix((40, 220, 230), (40, 230, 90), E.clamp(t / 1.2))
            r = size * .08
            dx = last_x + size * .25
            d = ImageDraw.Draw(layer)
            if phase == 0:
                d.rectangle((dx - r * .35, cy - r * 1.6, dx + r * .35, cy + r * 1.6), fill=(*c, 255))
            else:
                d.ellipse((dx - r, cy - r, dx + r, cy + r), fill=(*c, 255))
    elif dev == 'metal_track':
        p = E.ease_out(E.clamp(t / .5), 2)
        m = E.text_mask(text, int(size * 1.15), 'Bold', 1.0, .35 * (1 - p) + .04)
        E.place(layer, E.fill_image(m, 'vgrad', [(250, 250, 255), (120, 120, 140)]), cx, cy, alpha=.3 + .7 * p)
    elif dev == 'scatter':
        # 화면 밖까지 흩어진 거대 글자들이 0.5초에 모이고, 첫 글자 앞 같은 색 막대가 짧아지며 사라짐
        p = E.ease_out(E.clamp(t / .5), 3)
        rng = E.stable_rng(scene.get('index', 0), 'scatter')
        items, _ = E.block_layout([text], size * 1.2, ['Bold'], unit='char')
        grad = tpl['grad']
        final = grad if tpl['bg'] == 'black' else [E.mix(grad[0], BLACK, p), E.mix(grad[1], BLACK, p)]
        for it in items:
            m = E.text_mask(it['text'], int(size * 1.2), 'Bold')
            ox, oy, sc = rng.uniform(-.55, .55) * width, rng.uniform(-.42, .42) * height, rng.uniform(1.6, 3.0)
            E.place(layer, E.fill_image(m, 'vgrad', final), cx + it['cx'] + ox * (1 - p), cy + oy * (1 - p), 1 + (sc - 1) * (1 - p))
        if p < 1 and items:
            first = items[0]
            left = cx + first['cx'] - first['w'] / 2 + rng.uniform(-.2, .2) * width * (1 - p)
            bar_w = width * .45 * (1 - p)
            h = size * .75 * (1 + 1.5 * (1 - p))
            bar = Image.new('L', (max(1, int(bar_w)), max(1, int(h))), 255)
            E.place(layer, E.fill_image(bar, 'hgrad', final), left - bar_w / 2, cy, 1.0)
    image.alpha_composite(layer)
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('62999763', 'letter_assemble', FPS, TEMPLATES, assign, minimum, render)
