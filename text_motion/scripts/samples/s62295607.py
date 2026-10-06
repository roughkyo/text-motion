"""62295607: 거대 반투명 교체·3D 별·오른쪽 세로 청색 광원, 흰 화면 구간 (60fps).

근거: references/analysis/62295607-spec.md (1프레임 시트 33장).
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

E = None
FPS = 60.0
WHITE, INK = (255, 255, 255), (16, 16, 20)
SKY = (120, 168, 255)
BLUE_WORD = (90, 140, 255)

TEMPLATES = [
    {'n': 1, 'frames': 160, 'device': 'star_open', 'kind': 'word'},
    {'n': 2, 'frames': 85, 'device': 'giant_swap', 'kind': 'word'},
    {'n': 3, 'frames': 95, 'device': 'giant_swap', 'kind': 'word', 'navy': True},
    {'n': 4, 'frames': 135, 'device': 'light_star_rise', 'kind': 'line', 'light': True},
    {'n': 5, 'frames': 200, 'device': 'two_line_fade', 'kind': 'two'},
    {'n': 6, 'frames': 90, 'device': 'wide_type', 'kind': 'line'},
    {'n': 7, 'frames': 62, 'device': 'rise_mask', 'kind': 'word'},
    {'n': 8, 'frames': 137, 'device': 'light_rise_stars', 'kind': 'line', 'light': True},
    {'n': 9, 'frames': 112, 'device': 'word_blue_fade', 'kind': 'line'},
    {'n': 10, 'frames': 170, 'device': 'stair_words', 'kind': 'line'},
    {'n': 11, 'frames': 110, 'device': 'star_shrink_words', 'kind': 'line'},
    {'n': 12, 'frames': 95, 'device': 'giant_swap', 'kind': 'word'},
    {'n': 13, 'frames': 150, 'device': 'end_erase', 'kind': 'line'},
]
LAST = TEMPLATES[-1]
ENTER = {'star_open': 1.35, 'giant_swap': .32, 'light_star_rise': .9, 'two_line_fade': .7, 'wide_type': .55, 'rise_mask': .3,
         'light_rise_stars': .6, 'word_blue_fade': .5, 'stair_words': .6, 'star_shrink_words': .9, 'end_erase': .5}
EXIT = {'light_star_rise': .4, 'two_line_fade': .4, 'wide_type': .3, 'rise_mask': .3, 'word_blue_fade': .3, 'end_erase': .35}


def fit(template, words, chars):
    if template['kind'] == 'word':
        return chars <= 9
    if template['kind'] == 'two':
        return len(words) >= 4
    return len(words) >= 2


def assign(paragraphs):
    # 별 오프닝(1번)은 첫 문구에만 쓴다.
    if not paragraphs:
        return []
    first = TEMPLATES[0] if len(paragraphs[0].replace(' ', '')) <= 9 else TEMPLATES[1]
    rest = E.assign_in_order(TEMPLATES[1:-1], paragraphs[1:], fit=fit, last=LAST) if len(paragraphs) > 1 else []
    return [first] + rest


def minimum(template, text):
    return ENTER[template['device']] + EXIT.get(template['device'], 0) + max(.45, len(text.replace(' ', '')) / 14)


def background(width, height, light, t, seed):
    if light:
        return Image.new('RGBA', (width, height), (250, 250, 252, 255))
    im = Image.new('RGBA', (width, height), (5, 6, 14, 255))
    breath = .8 + .2 * math.sin(t * 1.3)
    small_w, small_h = 160, 90
    yy, xx = np.mgrid[0:small_h, 0:small_w]
    # 오른쪽 가장자리 세로 청색 광원(세로로 길게 휜 형태) + 왼쪽 아래 약한 광
    curve = .965 - .03 * np.cos((yy / small_h - .5) * math.pi)
    right = np.clip(1 - np.abs(xx / small_w - curve) / .085, 0, 1) ** 2.2 * breath * .85
    right += np.clip(1 - np.hypot(xx / small_w - 1.0, (yy / small_h - .5) * .6) / .25, 0, 1) ** 2 * .5
    left = np.clip(1 - np.hypot(xx / small_w, (yy / small_h - 1) * 1.2) / .35, 0, 1) ** 2 * .35
    glow = np.clip(right + left, 0, 1)[..., None]
    arr = np.array((5, 6, 14), float) * (1 - glow) + np.array((58, 140, 255), float) * glow
    im = Image.fromarray(arr.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC).convert('RGBA')
    return E.grain(im, 3, seed)


def chrome_star(dst, cx, cy, size, spin):
    """흰 화면 구간의 흑백 크롬 별: 컬러 별을 회색조로 바꾸고 가운데를 어둡게."""
    tmp = Image.new('RGBA', dst.size)
    E.star_3d(tmp, cx, cy, size, spin)
    gray = ImageOps.grayscale(tmp.convert('RGB'))
    gray = gray.point(lambda v: int(250 - (v / 255) ** .7 * 200))
    chrome = Image.merge('RGBA', (gray, gray, gray, tmp.getchannel('A')))
    dst.alpha_composite(chrome)


def word_items(lines, size, style='Bold'):
    return E.block_layout(lines, size, [style], line_gap=1.18, unit='word')


def glyph(text, size, color, gradient=None):
    m = E.text_mask(text, int(size), 'Bold')
    if gradient:
        return E.fill_image(m, 'vgrad', gradient)
    return E.tinted(m, color)


def render(scene, local, width, height, info):
    tpl = scene['template']
    dev = tpl['device']
    light = tpl.get('light', False)
    t = local
    total = scene['duration']
    ext = EXIT.get(dev, 0)
    g = t - (total - ext)
    image = background(width, height, light, info.get('t', t), int(info.get('t', t) * FPS))
    layer = Image.new('RGBA', (width, height))
    words = E.split_words(scene['text'])
    lines = scene.get('lines') or ([' '.join(x) for x in E.balanced_lines(words)] if tpl['kind'] == 'two' and len(words) > 1 else [scene['text']])
    base = height * (.15 if tpl['kind'] == 'word' else .105)
    widest = max(E.styled_font('Bold', int(base)).getlength(ln) for ln in lines)
    size = min(base, base * width * .66 / max(1, widest))
    cx, cy = width / 2, height / 2
    ink = INK if light else WHITE
    prev = info.get('previous')
    if dev in ('giant_swap', 'star_open'):
        start = 1.0 if dev == 'star_open' else 0.0
        if dev == 'star_open' and t < 1.15:
            # 점에서 커지며 회전, 세로 막대까지 돌았다 다시 펼쳐짐(f4~f100)
            grow = E.ease_out(t / .5, 3)
            E.star_3d(layer, cx, cy, height * .11 * grow, t * .9)
        q = E.clamp((t - start) / .3)
        if t >= start:
            s = 2.4 - 1.4 * E.ease_out(q, 3)
            hold = max(0, t - start - .3)
            s *= 1 - .03 * min(1, hold / 1.5)
            gray = E.mix((70, 72, 84), WHITE, E.ease_out(q, 2))
            m = E.text_mask(scene['text'], int(size), 'Bold')
            if tpl.get('navy') or q < 1:
                top = E.mix(gray, WHITE, .3)
                bottom = E.mix(gray, SKY, .7) if tpl.get('navy') else gray
                white_q = E.clamp((t - start - .3) / .4)
                sprite = E.fill_image(m, 'vgrad', [E.mix(top, WHITE, white_q), E.mix(bottom, WHITE, white_q)])
            else:
                sprite = E.tinted(m, WHITE)
            E.place(layer, sprite, cx, cy, s, alpha=.4 + .6 * E.ease_out(q, 2), blur=4 * (1 - q) * height / 1080)
        if prev is not None and t < start + .3 and dev == 'giant_swap':
            pm = E.text_mask(prev['text'][:16], int(size), 'Bold')
            pq = E.clamp(t / .3)
            E.place(layer, E.tinted(pm, WHITE), cx, cy, 1 - .35 * pq, alpha=1 - pq)
    elif dev in ('light_star_rise', 'light_rise_stars', 'rise_mask'):
        if dev == 'light_star_rise':
            chrome_star(layer, cx + width * .05, cy, height * .3, t * .35)
        elif dev == 'light_rise_stars':
            chrome_star(layer, width * .17, height * .22, height * .09, t * .5)
            chrome_star(layer, width * .78, height * .72, height * .09, t * .5 + .3)
        chars = dev != 'light_star_rise'
        items, _ = E.block_layout(lines, size, ['Bold'], line_gap=1.18, unit='char' if chars else 'word')
        for it in items:
            step = .05 if chars else .16
            start = .05 + step * it['n']
            if t < start:
                continue
            p = E.ease_out((t - start) / .14, 3)
            m = E.text_mask(it['text'], int(size), 'Bold')
            color = E.mix((150, 150, 158), ink, p)
            if dev == 'rise_mask':
                sprite = E.fill_image(m, 'vgrad', [WHITE, (120, 124, 140)])
            else:
                sprite = E.tinted(m, color)
            # 기준선 아래에서 솟음: 위쪽만 보이도록 잘라 그린다
            rise = m.height * .7 * (1 - p)
            visible = sprite.crop((0, 0, sprite.width, max(1, int(sprite.height - rise))))
            alpha = 1.0
            if g >= 0:
                if dev == 'rise_mask':
                    # 글자별 아래로 빠짐(f1037~f1048)
                    e = E.clamp((g - .03 * it['n']) / .12)
                    visible = sprite.crop((0, 0, sprite.width, max(1, int(sprite.height * (1 - e)))))
                    rise = sprite.height * e
                else:
                    e = E.clamp((g - .07 * it['n']) / .15)
                    alpha = 1 - e
            E.place(layer, visible, cx + it['cx'], cy + it['cy'] - (sprite.height - visible.height) / 2 + rise * 0, 1.0, alpha=alpha)
    elif dev in ('two_line_fade', 'word_blue_fade', 'stair_words', 'star_shrink_words', 'end_erase'):
        offset = 0.0
        if dev == 'star_shrink_words':
            sq = E.clamp(t / .55)
            if sq < 1:
                E.star_3d(layer, cx, cy, height * .06 * (1 - E.ease_in(sq, 2)) + 1, t * 1.2)
            offset = .6
        items, _ = word_items(lines, size)
        order = E.stable_rng(scene.get('index', 0), 'fade').permutation(len(items))
        for it in items:
            start = offset + .05 + .11 * it['n']
            if t < start:
                continue
            p = E.ease_out((t - start) / .18, 2)
            blue = dev in ('two_line_fade', 'word_blue_fade') and it['n'] % 3 == 1
            color = E.mix((90, 96, 120), BLUE_WORD if blue else ink, p)
            if dev == 'word_blue_fade':
                color = E.mix(BLUE_WORD, ink, E.clamp((t - start - .3) / .4))
            x, y = cx + it['cx'], cy + it['cy']
            if dev == 'stair_words':
                y = cy + (it['n'] - (len(items) - 1) / 2) * size * .35
                x = cx + it['cx'] * .95
            alpha = p
            if g >= 0:
                if dev == 'two_line_fade':
                    e = E.clamp((g - .04 * order[it['n']]) / .15)
                elif dev in ('word_blue_fade',):
                    e = E.clamp((g - .07 * it['n']) / .12)
                elif dev == 'end_erase':
                    # 문장 끝 글자를 남기고 앞쪽부터 사라짐(f2160~f2167)
                    e = E.clamp((g - .05 * it['n']) / .12) if it['n'] < len(items) - 1 else E.clamp((g - ext * .7) / (ext * .3))
                else:
                    e = 0
                alpha *= 1 - e
                color = E.mix(color, (120, 120, 130), e)
            E.place(layer, E.tinted(E.text_mask(it['text'], int(size), 'Bold'), color), x, y, 1.0, alpha=alpha)
    elif dev == 'wide_type':
        text = scene['text']
        track = .18
        m_all = E.text_mask(text, int(size), 'Bold', 1.0, track)
        f = E.styled_font('Bold', int(size))
        advances = [f.getlength(ch) + track * size for ch in text]
        x0 = cx - sum(advances) / 2
        x = x0
        for k, ch in enumerate(text):
            start = .04 * k
            if ch != ' ' and t >= start:
                p = E.clamp((t - start) / .1)
                color = E.mix((80, 90, 130), E.mix(SKY, WHITE, .35), p)
                m = E.text_mask(ch, int(size), 'Bold')
                dy, rot, alpha = 0.0, 0.0, 1.0
                if g >= 0:
                    # 첫 글자부터 기울며 아래로 흘러내림(f876~f887)
                    e = E.clamp((g - .025 * k) / .15)
                    dy, rot, alpha = size * .9 * E.ease_in(e, 2), -18 * e, 1 - e
                E.place(layer, E.tinted(m, color), x + advances[k] / 2, cy + dy, 1.0, rot=rot, alpha=alpha * (.4 + .6 * p))
            x += advances[k]
    image.alpha_composite(layer)
    draw = ImageDraw.Draw(image)
    tiny = E.font(height * .011, 'Regular')
    c = (150, 150, 160, 255) if not light else (150, 150, 155, 255)
    draw.text((width * .03, height * .935), info.get('label', ''), font=tiny, fill=c)
    draw.text((width * .965, height * .045), '◇', font=tiny, fill=c, anchor='mm')
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('62295607', 'clean_scale', FPS, TEMPLATES, assign, minimum, render)
