"""60806086: 보라 연기 위 작은 중앙 문장, 문구 14개마다 다른 등장 장치와 강조색 (29.97fps).

근거: references/analysis/60806086-spec.md (1프레임 시트 15장 전체).
60608725와 같은 계열이라 그 엔진(enter_states·compose_chars·tint_band·apply_exit)을 재사용하고,
이 샘플에만 있는 장치(기울어진 글자 비행, 이탤릭 순간 전환, 넓은 자간 흔들림, 겹침 교체)를 여기서 그린다.
시간 값은 60608725 엔진과 같은 25fps 기준 프레임으로 환산해 둔다(원본 프레임 × 25/29.97).
"""
import math

from PIL import Image, ImageDraw

E = None
FPS = 30000 / 1001
K = 25 / FPS  # 원본 프레임 → 25fps 엔진 프레임
YELLOW, BLUE, PINK = (240, 176, 32), (80, 112, 240), (240, 80, 180)
CYAN, GREEN, PURPLE, ORANGE = (48, 232, 192), (112, 224, 112), (176, 112, 240), (240, 80, 48)

TEMPLATES = [
    {'n': 1, 'frames': 50, 'enter': 'hero_type', 'accent': YELLOW, 'hero_scale': 2.6, 'exit': None, 'delay': 2},
    {'n': 2, 'frames': 30, 'enter': 'rise_words', 'exit': None, 'delay': 3},
    {'n': 3, 'frames': 38, 'enter': 'tilt_fly', 'exit': None, 'delay': 2},
    {'n': 4, 'frames': 51, 'enter': 'type', 'pass_wave': BLUE, 'exit': None, 'delay': 1},
    {'n': 5, 'frames': 48, 'enter': 'blur_focus', 'hero_scale': 2.0, 'exit': None, 'delay': 2, 'underline': (0, PINK)},
    {'n': 6, 'frames': 48, 'enter': 'cursor_type', 'exit': None, 'delay': 1},
    {'n': 7, 'frames': 51, 'enter': 'hero_word', 'accent': CYAN, 'hero_scale': 1.6, 'exit': None, 'delay': 2},
    {'n': 8, 'frames': 45, 'enter': 'type', 'pass_wave': GREEN, 'exit': None, 'delay': 1},
    {'n': 9, 'frames': 51, 'enter': 'gradient_focus', 'gradient': (PURPLE, (226, 160, 255)), 'exit': None, 'delay': 1},
    {'n': 10, 'frames': 61, 'enter': 'italic_switch', 'accent': ORANGE, 'exit': None, 'delay': 1},
    {'n': 11, 'frames': 48, 'enter': 'wide_bounce', 'exit': None, 'delay': 1},
    {'n': 12, 'frames': 71, 'enter': 'hero_word', 'accent': CYAN, 'hero_scale': 1.6, 'exit': None, 'delay': 1, 'shrink_hold': True},
    {'n': 13, 'frames': 25, 'enter': 'type', 'exit': 'push_up', 'delay': 2},
    {'n': 14, 'frames': 111, 'enter': 'type', 'exit': None, 'delay': 0},
]
LAST = TEMPLATES[-1]
ENGINE_KINDS = {'hero_type', 'rise_words', 'type', 'blur_focus', 'cursor_type', 'hero_word', 'gradient_focus'}


def assign(paragraphs):
    pool = TEMPLATES[:-1]
    out = [pool[k % len(pool)] for k in range(len(paragraphs))]
    if out and len(paragraphs) > 1:
        out[-1] = LAST
    return out


def enter_frames(template, text):
    n = len(text)
    words = len(E.split_words(text))
    return {'hero_type': 14 + n / 1.4, 'rise_words': 8 + 2 * words, 'tilt_fly': 1.3 * n + 6, 'type': 1.3 * n + 3,
            'blur_focus': 14 + 2.5 * words, 'cursor_type': 1.3 * n + 2, 'hero_word': 12 + n / 1.5,
            'gradient_focus': 15 + 18 * max(0, words - 1), 'italic_switch': 1.1 * n + 10, 'wide_bounce': 1.4 * n + 6}[template['enter']]


def minimum(template, text):
    text = ' '.join(E.split_words(text))
    return (template.get('delay', 0) + enter_frames(template, text) + (6 if template.get('exit') == 'push_up' else 0)) / 25 + \
        max(.35, len(text.replace(' ', '')) / 14)


_SIZE = [40.0]


def size_hint():
    return _SIZE[0] * 1.2


def custom_states(template, text, f):
    """이 샘플 고유 장치의 글자 상태(25fps 프레임 f)."""
    n = len(text)
    states = [{'visible': False} for _ in text]
    kind = template['enter']
    tint = None
    if kind == 'tilt_fly':
        # 글자마다 −20° 기울어진 채 오른쪽 위에서 날아와 바로 선다(f94~f109)
        for k in range(n):
            a = 1.3 * k
            if f < a or text[k] == ' ':
                continue
            p = E.ease_out((f - a) / 5, 3)
            states[k] = {'visible': True, 'alpha': .3 + .7 * p, 'dx': (1 - p) * size_hint(), 'dy': -(1 - p) * .04,
                         'blur': (1 - p) * 3, 'color': E.SOFT_WHITE, 'rot': -20 * (1 - p)}
    elif kind == 'italic_switch':
        # 타이핑(1글자/1.1프레임) 후 이탤릭으로 순간 전환하며 뒤쪽 단어가 주황빨강(f431~f491)
        done = 1.1 * n + 2
        italic = f >= done + 8
        words = E.split_words(text)
        accent_from = len(' '.join(words[:max(1, len(words) - 3)])) if len(words) > 2 else n
        for k in range(n):
            if f < 1.1 * k or text[k] == ' ':
                continue
            p = E.clamp((f - 1.1 * k) / 1.5)
            color = ORANGE if italic and k >= accent_from else E.SOFT_WHITE
            states[k] = {'visible': True, 'alpha': .5 + .5 * p, 'color': color, 'italic': italic}
    elif kind == 'wide_bounce':
        # 넓은 자간 둥근 서체, 글자마다 위아래로 흔들리며 타이핑(f492~f539)
        for k in range(n):
            a = 1.4 * k
            if f < a or text[k] == ' ':
                continue
            p = E.clamp((f - a) / 2)
            bob = math.sin((f - a) * .9) * (1 - E.clamp((f - a) / 10)) * .02
            states[k] = {'visible': True, 'alpha': .4 + .6 * p, 'dy': bob, 'space': .12, 'color': E.SOFT_WHITE}
    return states, tint


def render(scene, local, width, height, info):
    tpl = scene['template']
    text = ' '.join(E.split_words(scene['text']))
    delay = tpl.get('delay', 0) * K
    f = local * 25 - delay
    fe = scene['duration'] * 25 - delay
    size = height * .088  # 원본 25자 문장 폭 약 0.52W
    total = sum(E.char_mask(ch, int(size), 'Bold')[1] for ch in text)
    if total > width * .66:
        size *= width * .66 / total
    _SIZE[0] = size
    image = E.background('smoke_purple', width, height, scene.get('source_time', 0) + local * scene.get('source_rate', 1.0))
    layer = Image.new('RGBA', (width, height))
    if tpl['enter'] in ENGINE_KINDS:
        states, _, tint = E.enter_states(tpl, text, f, fe)
    else:
        states, tint = custom_states(tpl, text, f)
    if tpl.get('shrink_hold'):
        q = E.ease_out(E.clamp((f - enter_frames(tpl, text)) / 18), 2)
        for st in states:
            if st.get('visible', True) and st.get('visible') is not False:
                st['scale'] = st.get('scale', 1) * (1.25 - .25 * q)
                st['pos_scale'] = st.get('pos_scale', 1) * (1.25 - .25 * q)
    cx, cy = width / 2, height * .5
    for st in states:
        if 'dy' in st:
            st['dy'] = st['dy'] * height
    # 이탤릭 순간 전환은 글자 스프라이트를 기울여 그린다.
    italic = any(st.get('italic') for st in states if isinstance(st, dict))
    extent = E.compose_chars(layer, text, size, states, cx, cy, recenter=True)
    if italic:
        w, h = layer.size
        layer = layer.transform((w, h), Image.Transform.AFFINE, (1, .22, -cy * .22, 0, 1, 0), Image.Resampling.BICUBIC)
    if tint and extent:
        E.tint_band(layer, extent, cy, size, *tint)
    if tpl['enter'] == 'cursor_type' and extent:
        if int(f // 6) % 2 == 0 or f < 1.3 * len(text):
            x = extent['right'] + size * .25
            ImageDraw.Draw(layer).rectangle((x, cy - size * .48, x + max(2, size * .06), cy + size * .42), fill=(*E.SOFT_WHITE, 255))
    if tpl.get('underline') and extent and f > 6:
        widx = E.word_index(text)
        xs = [(px, w) for k, px, w in extent['chars'] if widx[k] == 0]
        if xs:
            a = min(px - w / 2 for px, w in xs)
            b = max(px + w / 2 for px, w in xs)
            grow = E.ease_out((f - 6) / 6, 2)
            E.tapered_stroke(layer, a, a + (b - a) * grow, cy + size * .66, size * .045, [PINK, (200, 90, 240)], sag=size * .03)
    if tpl.get('exit') == 'push_up':
        # 다음 문구가 겹쳐 등장하는 동안 위로 밀려나며 흐려진다(f639~f648)
        g = f - (fe - 6)
        if g > 0:
            q = E.clamp(g / 6)
            moved = Image.new('RGBA', layer.size)
            E.place(moved, layer, width / 2, height / 2 - height * .045 * q, 1.0, alpha=1 - .7 * q, blur=2 * q)
            layer = moved
    prev = info.get('previous')
    if prev is not None and prev['template'].get('exit') == 'push_up' and local < 6 / 25:
        ghost = Image.new('RGBA', (width, height))
        ptext = ' '.join(E.split_words(prev['text']))
        pstates = [{'visible': ch != ' ', 'alpha': .3 * (1 - local / (6 / 25)), 'color': E.SOFT_WHITE} for ch in ptext]
        E.compose_chars(ghost, ptext, size, pstates, cx, cy - height * .045 - height * .02 * local * 25 / 6, recenter=True)
        image.alpha_composite(ghost)
    image.alpha_composite(layer)
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('60806086', 'smoke_purple', FPS, TEMPLATES, assign, minimum, render,
                         hit=lambda scene: scene['template'].get('delay', 0) / FPS, smoke=24.9)
