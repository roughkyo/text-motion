"""57032891: 둥근 초굵은 3D 압출 글자 블록, 장면마다 다른 등장·퇴장, 모서리 색 광원 (60fps).

근거: references/analysis/57032891-spec.md (1프레임 시트 40장 전체).
시간 값은 원본 프레임(60fps)을 초로 바꾼 것이다.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

E = None
FPS = 60.0
WHITE_TOP, WHITE_BOTTOM, WHITE_SIDE = (255, 255, 255), (214, 214, 226), (70, 70, 84)
RED, NAVY, PURPLE_NAVY = (232, 32, 42), (32, 48, 200), (78, 64, 214)

# 배경: (광원 색, 광원 중심 x, y, 반경)
TEMPLATES = [
    {'n': 1, 'frames': 198, 'kind': 'multi', 'enter': 'gather', 'exit': 'fall', 'accent': RED, 'accent_line': 0,
     'glow': ((215, 36, 40), .05, 1.0, .75), 'styles': ['rounded']},
    {'n': 2, 'frames': 257, 'kind': 'multi', 'enter': 'scatter', 'exit': 'reverse', 'accent': NAVY, 'accent_line': 1,
     'glow': ((40, 52, 200), .5, 1.05, .8), 'styles': ['rounded']},
    {'n': 3, 'frames': 245, 'kind': 'multi', 'enter': 'pop_slide', 'exit': 'tilt_push', 'accent': RED, 'accent_line': 0,
     'glow': ((110, 50, 200), .95, .55, .75), 'styles': ['rounded', 'rounded_light']},
    {'n': 4, 'frames': 444, 'kind': 'single', 'enter': 'stack_spread', 'exit': 'collapse', 'accent': None,
     'glow': ((22, 160, 165), .05, .05, .8), 'styles': ['rounded'], 'reflect': True},
    {'n': 5, 'frames': 254, 'kind': 'multi', 'enter': 'ghost_chain', 'exit': 'drop', 'accent': PURPLE_NAVY, 'accent_line': 1,
     'glow': ((30, 80, 220), .95, 1.0, .75), 'styles': ['rounded']},
    {'n': 6, 'frames': 341, 'kind': 'multi', 'enter': 'slide_behind', 'exit': 'split', 'accent': None,
     'glow': ((20, 160, 180), 1.0, .5, .75), 'styles': ['rounded']},
    {'n': 7, 'frames': 264, 'kind': 'multi', 'enter': 'fly_rotate', 'exit': 'fall', 'accent': PURPLE_NAVY, 'accent_line': 0,
     'glow': ((20, 150, 150), .5, 1.05, .8), 'styles': ['rounded_light', 'rounded'], 'grid': True},
    {'n': 8, 'frames': 258, 'kind': 'stack', 'enter': 'vertical_pop', 'exit': 'fall', 'accent': RED, 'accent_line': -1,
     'glow': ((12, 95, 95), .5, 1.05, .8), 'styles': ['rounded']},
    {'n': 9, 'frames': 233, 'kind': 'single', 'enter': 'pop_up', 'exit': 'drop', 'accent': None,
     'glow': ((40, 200, 200), .05, .05, .8), 'styles': ['rounded']},
]
ENTER_TIME = {'gather': .9, 'scatter': .9, 'pop_slide': .65, 'stack_spread': 1.1, 'ghost_chain': .9, 'slide_behind': 1.4,
              'fly_rotate': .55, 'vertical_pop': .5, 'pop_up': .45}
EXIT_TIME = {'fall': .2, 'reverse': .95, 'tilt_push': 1.0, 'collapse': .35, 'drop': .55, 'split': .55}


def fit(template, words, chars):
    if template['kind'] == 'single':
        return len(words) <= 2 and chars <= 8
    if template['kind'] == 'stack':
        return 2 <= len(words) <= 4 and chars <= 12
    return True  # 여러 줄 템플릿은 한 단어도 받는다(같은 효과 반복 방지)


def assign(paragraphs):
    return E.assign_in_order(TEMPLATES, paragraphs, fit=fit)


def minimum(template, text):
    chars = len(text.replace(' ', ''))
    return ENTER_TIME[template['enter']] + EXIT_TIME[template['exit']] + max(.6, chars / 12)


def split_lines(scene, template):
    words = E.split_words(scene['text'])
    if scene.get('lines'):
        return scene['lines']
    if template['kind'] == 'single':
        return [' '.join(words)]
    if template['kind'] == 'stack':
        return words[:4]
    if len(words) <= 2:
        return words
    # 원본처럼 2~3줄, 줄마다 길이가 다른 비정형 블록
    if len(words) >= 5:
        a = max(1, len(words) // 3)
        return [' '.join(words[:a]), ' '.join(words[a:2 * a + 1]), ' '.join(words[2 * a + 1:])] if len(words) > 2 * a + 1 else \
            [' '.join(x) for x in E.balanced_lines(words)]
    return [' '.join(x) for x in E.balanced_lines(words)]


def block_size(lines, styles, width, height):
    size = height * .2
    for _ in range(2):
        widest = max(E.styled_font(styles[r % len(styles)], int(size)).getlength(ln) for r, ln in enumerate(lines))
        size = min(size * (width * .55) / max(1, widest), height * .6 / (len(lines) * .86))
    return max(10, size)


def char_colors(template, row, k, count, rows):
    accent = template['accent']
    line = template.get('accent_line', 0)
    if accent is None or (line >= 0 and row != line) or (line == -1 and False):
        return WHITE_TOP, WHITE_BOTTOM, WHITE_SIDE
    if line == -1:  # 세로 쌓기: 위 글자부터 원색 → 흰색
        q = row / max(1, rows - 1)
    else:
        q = k / max(1, count - 1)
    q = min(1, q * 1.15)
    top = E.mix(accent, (255, 255, 255), q ** .8)
    bottom = E.mix(tuple(int(c * .82) for c in accent), WHITE_BOTTOM, q ** .8)
    side = E.mix(tuple(int(c * .45) for c in accent), WHITE_SIDE, q)
    return top, bottom, side


def background(template, width, height, seed):
    color, gx, gy, r = template['glow']
    # 원본 광원은 모서리에 작게 모여 있고 화면 대부분은 거의 검정이다.
    color = tuple(int(c * .62) for c in color)
    im = E.radial(width, height, gx, gy, r * .72, r * .8, color, (7, 7, 10), 1.6)
    if template.get('grid'):
        draw = ImageDraw.Draw(im)
        for k in range(1, 12):
            x = width * k / 12
            draw.line((x, 0, x, height), fill=(30, 60, 64, 255), width=1)
        for k in range(1, 7):
            y = height * k / 7
            draw.line((0, y, width, y), fill=(30, 60, 64, 255), width=1)
    return E.grain(im, 7, seed)


def item_state(template, it, items, t, total, width, height, size, rng_seed):
    """등장 단계에서 글자 하나의 상태: dx, dy, scale, rot, alpha, blur, motion, ghost."""
    enter = template['enter']
    n, count = it['n'], len(items)
    rng = E.stable_rng(rng_seed, n)
    st = {'dx': 0.0, 'dy': 0.0, 'scale': 1.0, 'rot': 0.0, 'alpha': 1.0, 'blur': 0.0, 'motion': None}
    if enter == 'gather':
        words = sorted({(i['row'], i['word']) for i in items})
        order = E.stable_rng(rng_seed, 'order').permutation(len(words))
        rank = order[words.index((it['row'], it['word']))]
        start = .1 + .2 * rank / max(1, len(words) - 1) * min(1, len(words) / 6) * 3
        p = E.ease_out((t - start) / .14, 3)
        if t < start:
            st['alpha'] = 0
            return st
        st['scale'] = .4 + .6 * p
        st['dx'] = (it['cx'] - it['word_cx']) * 1.6 * (1 - p)
        st['alpha'] = min(1, (t - start) / .06)
    elif enter == 'scatter':
        start = .05 + .7 * n / max(1, count - 1)
        p = E.ease_out((t - start) / .22, 3)
        if t < start:
            st['alpha'] = 0
            return st
        st['dx'] = rng.uniform(-.45, .45) * width * (1 - p)
        st['dy'] = rng.uniform(-.38, .38) * height * (1 - p)
        st['rot'] = rng.uniform(-40, 40) * (1 - p)
        st['scale'] = p * (1 + .12 * math.sin(math.pi * p))
    elif enter == 'pop_slide':
        if it['row'] == 0:
            first = [i for i in items if i['row'] == 0]
            order = E.stable_rng(rng_seed, 'pop').permutation(len(first))
            rank = order[first.index(it)]
            start = .07 + .045 * rank
            p = E.clamp((t - start) / .1)
            if t < start:
                st['alpha'] = 0
                return st
            st['scale'] = p * (1 + .18 * math.sin(math.pi * p))
        else:
            start = .07 + .045 * len([i for i in items if i['row'] == 0]) + .02
            p = E.ease_out((t - start) / .15, 3)
            if t < start:
                st['alpha'] = 0
                return st
            st['dx'] = -width * .75 * (1 - p)
            st['motion'] = (width * .12 * (1 - p), 0)
    elif enter == 'stack_spread':
        spread = 1.0
        p = E.ease_out((t - spread) / .1, 3)
        st['dx'] = -it['cx'] * (1 - p)
        st['alpha'] = .55 + .45 * p if t >= .05 else 0
        st['blur'] = 6 * (1 - p) * height / 1080
    elif enter == 'ghost_chain':
        start = .2 + .055 * n
        if t < start:
            st['alpha'] = 0
            return st
        solid = start + .05
        if t < solid:
            st['alpha'] = .35
            st['dx'] = size * .25
            st['blur'] = 2
        p = E.ease_out((t - solid) / .06, 2)
        st['scale'] = .9 + .1 * p
    elif enter == 'slide_behind':
        row_items = [i for i in items if i['row'] == it['row']]
        k = row_items.index(it)
        row_start = 0 if it['row'] == 0 else .55 + .06 * len([i for i in items if i['row'] == 0])
        if k == 0:
            if t < row_start:
                st['alpha'] = 0
                return st
            st['gray'] = 1 - E.clamp((t - row_start) / .5)
            if it['row'] > 0:
                p = E.ease_out((t - row_start) / .12, 3)
                st['dy'] = -size * .9 * (1 - p)
        else:
            start = row_start + .5 + .065 * k
            p = E.ease_out((t - start) / .12, 3)
            prev = row_items[k - 1]
            st['dx'] = (prev['cx'] - it['cx']) * (1 - p)
            st['alpha'] = 0 if t < start else 1
            st['motion'] = (size * .3 * (1 - p), 0) if t >= start else None
    elif enter == 'fly_rotate':
        last = max(i['row'] for i in items)
        if it['row'] == last:
            p = E.ease_out((t - .05) / .38, 3)
            st['dy'] = height * .45 * (1 - p)
            st['rot'] = -40 * (1 - p) - 8 * p + 8  # 최종 0°(블록 흔들림이 기울기를 더함)
            st['scale'] = .3 + .7 * p
            st['group'] = 'last'
        else:
            start = .08 + .035 * n
            p = E.clamp((t - start) / .09)
            if t < start:
                st['alpha'] = 0
                return st
            st['scale'] = p * (1 + .2 * math.sin(math.pi * p))
    elif enter == 'vertical_pop':
        start = .07 + .14 * it['row']
        p = E.clamp((t - start) / .11)
        if t < start:
            st['alpha'] = 0
            return st
        st['scale'] = p * (1 + .2 * math.sin(math.pi * p))
    elif enter == 'pop_up':
        start = .05 + .03 * n
        p = E.ease_out((t - start) / .2, 3)
        st['dy'] = height * .55 * (1 - p)
        st['rot'] = rng.uniform(15, 35) * (1 - p)
        if n == count - 1 and count > 2:
            # 마지막 글자는 따로 크게 원을 그리며 흔들리다 붙는다(? 근거)
            swing = E.clamp((t - start) / .35)
            st['dx'] = math.sin(swing * math.pi * 2) * size * .5 * (1 - swing)
            st['dy'] += -math.sin(swing * math.pi) * size * .6
            st['rot'] += 25 * math.sin(swing * math.pi * 3) * (1 - swing)
    return st


def exit_transform(template, items, g, gt, width, height, size, rng_seed):
    """퇴장 단계: (블록 회전, 블록 dx, dy, 줄별 변위 함수, 글자 상태 갱신 함수)."""
    kind = template['exit']
    q = E.clamp(g / gt)
    rot, dx, dy = 0.0, 0.0, 0.0
    row_shift = {}
    if kind == 'fall':
        rot = -8 * q
        dy = height * 1.1 * q ** 3
    elif kind == 'tilt_push':
        a = E.clamp(g / .55)
        b = E.clamp((g - .55) / .45)
        rot = -10 * E.ease_out(a, 2)
        dx, dy = width * 1.1 * b ** 2.5, -height * .5 * b ** 2.5
    elif kind == 'drop':
        dy = height * 1.1 * q ** 3
    elif kind == 'split':
        a = E.clamp(g / .3)
        b = E.clamp((g - .3) / .25)
        rot = -6 * a
        for r in range(4):
            sign = 1 if r % 2 == 0 else -1
            row_shift[r] = (sign * width * 1.0 * b ** 2.5, -sign * height * .6 * b ** 2.5)
    return rot, dx, dy, row_shift, q


def render(scene, local, width, height, info):
    tpl = scene['template']
    lines = split_lines(scene, tpl)
    styles = tpl['styles']
    size = block_size(lines, styles, width, height)
    items, block_h = E.block_layout(lines, size, styles, line_gap=.86, unit='char')
    # 단어 중심(gather용)
    for row, line in enumerate(lines):
        row_items = [i for i in items if i['row'] == row]
        word, start = 0, 0
        positions = []
        for ch_index, ch in enumerate(line):
            if ch == ' ':
                word += 1
            positions.append(word)
        for it in row_items:
            it['word'] = positions[it['k']]
        for w in set(i['word'] for i in row_items):
            members = [i for i in row_items if i['word'] == w]
            cx = sum(i['cx'] for i in members) / len(members)
            for i in members:
                i['word_cx'] = cx
    t = local
    total = scene['duration']
    gt = EXIT_TIME[tpl['exit']]
    g = t - (total - gt)
    seed = scene['index'] * 31 + 7
    frame_index = int(round(info.get('t', local) * FPS))
    image = background(tpl, width, height, frame_index)
    layer = Image.new('RGBA', (width, height))
    cx0, cy0 = width / 2, height / 2
    hold_t = max(0.0, t - ENTER_TIME[tpl['enter']])
    rot_block = 2.2 * math.sin(hold_t * math.tau / 4.5) if hold_t > 0 else 0.0
    scale_block = 1 + .012 * math.sin(hold_t * math.tau / 3.7)
    ex_rot, ex_dx, ex_dy, row_shift, q = (0, 0, 0, {}, 0)
    reverse_t = None
    if g >= 0:
        if tpl['exit'] == 'reverse':
            reverse_t = ENTER_TIME[tpl['enter']] * (1 - E.clamp(g / gt))
        else:
            ex_rot, ex_dx, ex_dy, row_shift, q = exit_transform(tpl, items, g, gt, width, height, size, seed)
    # 4번 장면: 유지 중 자간 변화(넓어짐 → 1/3 지점 빽빽하게 스냅 → 2/3 지점 다시 넓게)
    track = 0.0
    if tpl['enter'] == 'stack_spread' and hold_t > 0:
        phase = hold_t / max(.1, total - ENTER_TIME['stack_spread'] - gt)
        track = .08 * E.clamp(phase / .33) if phase < .33 else (-.04 if phase < .66 else .1)
    rows = max(i['row'] for i in items) + 1
    row_counts = {r: len([i for i in items if i['row'] == r]) for r in range(rows)}
    for it in items:
        tt = reverse_t if reverse_t is not None else t
        st = item_state(tpl, it, items, tt, total, width, height, size, seed)
        if st['alpha'] <= 0:
            continue
        row_items = [i for i in items if i['row'] == it['row']]
        k = row_items.index(it)
        top, bottom, side = char_colors(tpl, it['row'], k, row_counts[it['row']], rows)
        if st.get('gray'):
            top = E.mix(top, (110, 110, 118), st['gray'])
            bottom = E.mix(bottom, (90, 90, 98), st['gray'])
        sprite = E.extruded(it['text'], int(size), it['style'], top, bottom, side)
        x = it['cx'] + st['dx'] + (k - (len(row_items) - 1) / 2) * track * size
        y = it['cy'] + st['dy']
        if tpl['exit'] == 'collapse' and g >= 0:
            x = x * (1 - E.ease_in(q, 2))
            st['blur'] += 6 * q
            st['alpha'] *= 1 - q
        sh = row_shift.get(it['row'], (0, 0))
        E.place(layer, sprite, cx0 + x + sh[0], cy0 + y + sh[1], st['scale'], rot=st['rot'], alpha=st['alpha'],
                blur=st['blur'], motion=st['motion'])
    if tpl.get('reflect'):
        mirrored = ImageOps.flip(layer)
        fade = Image.linear_gradient('L').resize((width, height))
        fade = ImageOps.invert(fade).point(lambda a: int(a * .45))
        alpha = Image.composite(mirrored.getchannel('A'), Image.new('L', (width, height)), fade)
        mirrored.putalpha(E.ImageChops.multiply(mirrored.getchannel('A'), fade))
        shift = int(block_h * .98)
        reflect_layer = Image.new('RGBA', (width, height))
        reflect_layer.alpha_composite(mirrored.crop((0, height - int(cy0 + block_h / 2), width, height)),
                                      (0, int(cy0 + block_h / 2) + 2))
        image.alpha_composite(reflect_layer)
    total_rot = rot_block + ex_rot
    if abs(total_rot) > .05 or abs(scale_block - 1) > .001 or ex_dx or ex_dy:
        moved = Image.new('RGBA', (width, height))
        E.place(moved, layer, cx0 + ex_dx, cy0 + ex_dy, scale_block, rot=total_rot)
        layer = moved
    image.alpha_composite(layer)
    draw = ImageDraw.Draw(image)
    tiny = E.font(height * .011, 'Regular')
    footer = info.get('footer', '')
    if footer:
        tw = draw.textlength(footer[:60], font=tiny)
        draw.text((width / 2 - tw / 2, height * .935), footer[:60], font=tiny, fill=(120, 120, 130, 255))
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('57032891', 'bold_pop', FPS, TEMPLATES, assign, minimum, render)
