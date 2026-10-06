"""57523104: 압축 서체 블록, 줄마다 굵기 교대·좁은 줄간격, 사방에서 다른 방향으로 날아드는 단어와 모션블러 (30fps).

근거: references/analysis/57523104-spec.md (1프레임 시트 22장 전체).
줄별 진입/퇴장은 (방식, 지연 초, 길이 초). 줄 수가 템플릿보다 많으면 목록을 순환한다.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

E = None
FPS = 30.0
STRETCH = .86

TEMPLATES = [
    {'n': 1, 'frames': 75, 'weights': ['ExtraBold', 'Light'],
     'enter': [('right', 0, .1), ('up', .1, .13), ('right', .2, .08)], 'exit': [('right', .1, .25), ('right', .05, .3), ('right', 0, .25)]},
    {'n': 2, 'frames': 70, 'weights': ['ExtraBold', 'Light'],
     'enter': [('up', 0, .3), ('down', .4, .27), ('down', .7, .3)], 'exit': 'tilt_fall'},
    {'n': 3, 'frames': 69, 'weights': ['ExtraBold', 'Light'],
     'enter': [('rise_chars', 0, .5), ('grow_chars', .1, .55)], 'exit': [('up_far', 0, .4), ('down_far', 0, .4)]},
    {'n': 4, 'frames': 73, 'weights': ['ExtraBold', 'Light'],
     'enter': [('grow', 0, .15), ('gray', .06, .12), ('gray', .13, .12), ('gray', .2, .12), ('gray', .27, .12)], 'exit': 'spread_down'},
    {'n': 5, 'frames': 87, 'weights': ['ExtraBold', 'ExtraBold', 'ExtraBold'],
     'enter': [('rgb', 0, 1.1)], 'exit': 'rgb_reverse'},
    {'n': 6, 'frames': 58, 'weights': ['ExtraBold'],
     'enter': 'bottom_up', 'exit': 'gray_down'},
    {'n': 7, 'frames': 71, 'weights': ['ExtraBold', 'Light'],
     'enter': [('char_sizes', 0, .45), ('grow_chars', .25, .45)], 'exit': [('gray_out', .15, .2), ('gray_out', 0, .2)]},
    {'n': 8, 'frames': 72, 'weights': ['ExtraBold', 'Light', 'ExtraBold'], 'tilt': -4,
     'enter': [('rot_ul', .03, .3), ('rot_ll', 0, .3), ('rot_ll', .13, .3)], 'exit': [('up_far', 0, .3), ('up_far', .05, .3), ('down_far', .1, .3)]},
    {'n': 9, 'frames': 86, 'weights': ['ExtraBold'], 'tilt': -3,
     'enter': 'top_down_gray', 'exit': 'gray_down'},
    {'n': 10, 'frames': 89, 'weights': ['ExtraBold'],
     'enter': [('word_grow', 0, .6)], 'exit': 'word_shrink'},
    {'n': 11, 'frames': 88, 'weights': ['ExtraBold', 'Light', 'ExtraBold'],
     'enter': [('gray_words', 0, .4), ('grow_chars', .12, .5), ('gray_words', .3, .3)], 'exit': 'spread_down'},
    {'n': 12, 'frames': 116, 'weights': ['ExtraBold'],
     'enter': [('left', 0, .35), ('up', .08, .35), ('right', .16, .35), ('left', .24, .35)], 'exit': [('right', .3, .2), ('right', .2, .2), ('right', .1, .2), ('right', 0, .2)]},
]
LAST = TEMPLATES[-1]
EXIT_TIME = {'tilt_fall': .5, 'spread_down': .65, 'rgb_reverse': .9, 'gray_down': .55, 'word_shrink': .6}


def fit(template, words, chars):
    if template['n'] == 12:
        return False
    return len(words) >= 2


def assign(paragraphs):
    return E.assign_in_order(TEMPLATES[:-1], paragraphs, fit=fit, last=LAST)


def enter_span(template, lines):
    spec = template['enter']
    if spec == 'bottom_up' or spec == 'top_down_gray':
        return .12 * len(lines) + .2
    return max(d + l for _, d, l in spec)


def exit_span(template):
    spec = template['exit']
    if isinstance(spec, str):
        return EXIT_TIME[spec]
    return max(d + l for _, d, l in spec)


def split_lines(scene):
    if scene.get('lines'):
        return scene['lines']
    words = E.split_words(scene['text'])
    # 원본은 줄마다 한두 단어씩 좁게 쌓는다(SUCCESS/LOVES/SPEED, TURN/YOUR/DREAMS/INTO/PLANS).
    if len(words) <= 3:
        return words
    lines, k = [], 0
    while k < len(words):
        take = 2 if len(words[k]) <= 6 and k + 1 < len(words) else 1
        lines.append(' '.join(words[k:k + take]))
        k += take
    if len(lines) > 5:
        per = math.ceil(len(words) / 5)
        lines = [' '.join(words[i:i + per]) for i in range(0, len(words), per)]
    return lines


def minimum(template, text):
    lines = split_lines({'text': text})
    return enter_span(template, lines) + exit_span(template) + max(.6, len(text.replace(' ', '')) / 13)


def line_motion(kind, e, width, height, enter=True):
    """방식별 줄 변위. e: 등장 진행(0 시작→1 정착) 또는 퇴장 진행(0 정착→1 화면 밖)."""
    st = {'dx': 0.0, 'dy': 0.0, 'scale': 1.0, 'rot': 0.0, 'alpha': 1.0, 'gray': 0.0, 'blur': 0.0}
    if enter:
        r = 1 - E.ease_out(e, 3)
        if kind == 'right':
            st['dx'] = width * .7 * r
        elif kind == 'left':
            st['dx'] = -width * .7 * r
        elif kind == 'up':
            st['dy'] = height * .55 * r
        elif kind == 'down':
            st['dy'] = -height * .55 * r
        elif kind == 'grow':
            st['scale'] = .08 + .92 * (1 - r)
            st['alpha'] = min(1, e * 3)
        elif kind in ('gray', 'gray_words'):
            st['alpha'] = min(1, e * 2)
            st['gray'] = r
            st['blur'] = 4 * r
            st['dy'] = height * .03 * r
        elif kind == 'rot_ll':
            st['dx'], st['dy'], st['rot'] = -width * .55 * r, height * .3 * r, -14 * r
        elif kind == 'rot_ul':
            st['dx'], st['dy'], st['rot'] = width * .5 * r, -height * .3 * r, -14 * r
    else:
        q = E.ease_in(e, 2.6)
        if kind == 'right':
            st['dx'] = width * .9 * q
        elif kind == 'up_far':
            st['dy'] = -height * .7 * q
        elif kind == 'down_far':
            st['dy'] = height * .7 * q
        elif kind == 'gray_out':
            st['gray'] = e
            st['blur'] = 4 * e
            st['alpha'] = 1 - e
    return st


def background(width, height, frame_index, seed):
    im = Image.new('RGBA', (width, height), (7, 7, 9, 255))
    draw = ImageDraw.Draw(im)
    rng = E.stable_rng('smoke', seed)
    t = frame_index / FPS
    s = width / 640
    # 아주 어두운 연기 줄기(가는 곡선)가 천천히 흐른다.
    for k in range(7):
        x0, y0 = rng.uniform(0, width), rng.uniform(0, height)
        amp, freq, speed = rng.uniform(20, 60) * s, rng.uniform(.004, .012) / s, rng.uniform(.1, .3)
        pts = [(x0 + i * 6 * s, y0 + amp * math.sin(freq * i * 6 * s + t * speed * 6 + k)) for i in range(60)]
        draw.line(pts, fill=(34, 34, 38, 255), width=max(1, int(s)))
    im = E.grain(im, 4, frame_index)
    star = E.font(height * .022, 'Regular')
    draw = ImageDraw.Draw(im)
    draw.text((width * .965, height * .045), '✦', font=star, fill=(200, 200, 205, 255), anchor='mm')
    return im


def render(scene, local, width, height, info):
    tpl = scene['template']
    lines = split_lines(scene)
    weights = tpl['weights']
    size = height * .2
    widest = max(E.styled_font(weights[r % len(weights)], int(size)).getlength(ln) * STRETCH for r, ln in enumerate(lines))
    size = min(size * width * .5 / max(1, widest), height * .62 / (len(lines) * .82), height * .17)
    items, block_h = E.block_layout(lines, size, weights, line_gap=.82, stretch=STRETCH, unit='word')
    total = scene['duration']
    ent = enter_span(tpl, lines)
    ext = exit_span(tpl)
    g = local - (total - ext)
    frame_index = int(round(info.get('t', local) * FPS))
    image = background(width, height, frame_index, scene.get('index', 0))
    layer = Image.new('RGBA', (width, height))
    cx0, cy0 = width / 2, height * .45  # 원본 블록 중심은 화면 중앙보다 약간 위
    rows = len(lines)
    seed = scene.get('index', 0) * 13 + 5

    def state(it, t):
        r = it['row']
        spec = tpl['enter']
        st = {'dx': 0.0, 'dy': 0.0, 'scale': 1.0, 'rot': 0.0, 'alpha': 1.0, 'gray': 0.0, 'blur': 0.0}
        if spec == 'bottom_up':
            start = .1 * (rows - 1 - r)
            st = line_motion('gray', E.clamp((t - start) / .15), width, height)
            if t < start:
                st['alpha'] = 0
        elif spec == 'top_down_gray':
            start = .2 * r
            st = line_motion('gray', E.clamp((t - start) / .2), width, height)
            if t < start:
                st['alpha'] = 0
        else:
            kind, delay, dur = spec[r % len(spec)]
            delay = delay + (r // len(spec)) * .08
            e = E.clamp((t - delay) / dur)
            if t < delay:
                st['alpha'] = 0
                return st
            if kind in ('rise_chars', 'grow_chars', 'char_sizes', 'word_grow', 'gray_words'):
                row_items = [i for i in items if i['row'] == r]
                k = row_items.index(it)
                local_start = delay + k * dur / max(1, len(row_items)) * .7
                ek = E.clamp((t - local_start) / (dur * .45))
                if t < local_start:
                    st['alpha'] = 0
                    return st
                rng = E.stable_rng(seed, it['n'])
                if kind == 'rise_chars':
                    st['dy'] = size * .9 * (1 - E.ease_out(ek, 3))
                    st['scale'] = .2 + .8 * E.ease_out(ek, 3)
                elif kind == 'grow_chars':
                    st['scale'] = .15 + .85 * E.ease_out(ek, 3)
                elif kind == 'char_sizes':
                    st['scale'] = rng.uniform(.3, 2.4) + (1 - rng.uniform(.3, 2.4)) * 0 if ek == 0 else 1 + (rng.uniform(.3, 2.4) - 1) * (1 - E.ease_out(ek, 3))
                    st['dx'] = rng.uniform(-.15, .15) * width * (1 - E.ease_out(ek, 3))
                    st['dy'] = rng.uniform(-.1, .1) * height * (1 - E.ease_out(ek, 3))
                elif kind == 'word_grow':
                    st['scale'] = .2 + .8 * E.ease_out(ek, 3)
                elif kind == 'gray_words':
                    st = line_motion('gray', ek, width, height)
                return st
            if kind == 'rgb':
                st['rgb'] = 1 - E.ease_out(e, 2)
                st['dx'] = (width * .4 if (it['n'] % 2) else -width * .4) * (1 - E.ease_out(e, 2.5))
                return st
            st = line_motion(kind, e, width, height)
        return st

    def exit_state(it, st):
        spec = tpl['exit']
        r = it['row']
        if g < 0:
            return st
        if isinstance(spec, list):
            kind, delay, dur = spec[r % len(spec)]
            e = E.clamp((g - delay) / dur)
            ex = line_motion(kind, e, width, height, enter=False)
            st['dx'] += ex['dx']
            st['dy'] += ex['dy']
            st['gray'] = max(st['gray'], ex['gray'])
            st['blur'] += ex['blur']
            st['alpha'] *= ex['alpha']
            return st
        q = E.clamp(g / ext)
        if spec == 'spread_down':
            start = (rows - 1 - r) * .08
            e = E.clamp((g - start) / (ext - .2))
            st['dy'] += (r + 1) * size * .5 * E.ease_out(e, 2) + height * .7 * E.ease_in(e, 3)
        elif spec == 'gray_down':
            start = (rows - 1 - r) * .1
            e = E.clamp((g - start) / .25)
            st['gray'], st['blur'], st['alpha'] = e, 4 * e, 1 - e
            st['dy'] += height * .06 * e
        elif spec == 'rgb_reverse':
            st['rgb'] = q
            st['dx'] += (width * .5 if (it['n'] % 2) else -width * .5) * E.ease_in(q, 2.5)
        elif spec == 'word_shrink':
            k = it['n']
            start = (len(items) - 1 - k) * .03
            e = E.clamp((g - start) / .25)
            st['scale'] *= 1 - E.ease_in(e, 2)
        return st

    for it in items:
        st = exit_state(it, state(it, local))
        if st['alpha'] <= 0 or st['scale'] <= 0:
            continue
        prev = exit_state(it, state(it, local - 1 / FPS))
        motion = (st['dx'] - prev['dx'], st['dy'] - prev['dy'])
        color = E.mix((255, 255, 255), (120, 120, 126), st['gray'])
        mask = E.text_mask(it['text'], int(size), it['style'], STRETCH)
        x, y = cx0 + it['cx'] + st['dx'], cy0 + it['cy'] + st['dy']
        if st.get('rgb', 0) > .01:
            # RGB 글리치 복제: 채널별로 좌우 어긋남(57523104 f380~f414)
            off = width * .08 * st['rgb']
            for c, sign in (((255, 40, 40), -1), ((40, 255, 70), 1), ((40, 80, 255), -.5)):
                E.place(layer, E.tinted(mask, c, .85), x + sign * off * (1 + it['n'] % 3), y, 1.0, motion=None)
        E.place(layer, E.tinted(mask, color), x, y, st['scale'], rot=st['rot'], alpha=st['alpha'], blur=st['blur'],
                motion=motion if abs(motion[0]) + abs(motion[1]) > size * .08 else None)
    tilt = tpl.get('tilt', 0)
    if tpl['exit'] == 'tilt_fall' and g >= 0:
        q = E.clamp(g / ext)
        tilt += 14 * E.ease_in(q, 2)
        moved = Image.new('RGBA', (width, height))
        E.place(moved, layer, cx0, cy0 + height * 1.0 * E.ease_in(q, 3), 1.0, rot=-tilt)
        layer = moved
    elif tilt:
        moved = Image.new('RGBA', (width, height))
        E.place(moved, layer, cx0, cy0, 1.0, rot=tilt)
        layer = moved
    image.alpha_composite(layer)
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('57523104', 'mono_stack', FPS, TEMPLATES, assign, minimum, render)
