"""63211445: 편집기에서 입력되는 듯한 타이핑, 선택 표현 5종, 단어 단위 불규칙 박자, 검정/흰 배경 교대 (30fps).

근거: references/analysis/63211445-spec.md (1프레임 시트 11장 전체).
"""
from PIL import Image, ImageDraw

E = None
FPS = 30.0
WHITE, INK = (255, 255, 255), (22, 22, 26)
SELECT_BLUE, LABEL_GRAY, RED = (184, 212, 248), (58, 58, 64), (240, 64, 80)

TEMPLATES = [
    {'n': 1, 'frames': 89, 'device': 'dashed', 'bg': 'black', 'unit': 'char', 'rate': 6.0, 'title': True},
    {'n': 2, 'frames': 66, 'device': 'dashed', 'bg': 'white', 'unit': 'word', 'title': True},
    {'n': 3, 'frames': 73, 'device': 'plain', 'bg': 'black', 'unit': 'char', 'rate': 2.0, 'center': True, 'title': True},
    {'n': 4, 'frames': 72, 'device': 'blue_select', 'bg': 'white', 'unit': 'word', 'title': True},
    {'n': 5, 'frames': 74, 'device': 'labels', 'bg': 'white', 'unit': 'word'},
    {'n': 6, 'frames': 59, 'device': 'dashed', 'bg': 'black', 'unit': 'word', 'title': True},
    {'n': 7, 'frames': 100, 'device': 'blue_select', 'bg': 'white', 'unit': 'char', 'rate': 1.0, 'caret': True},
    {'n': 8, 'frames': 120, 'device': 'dashed', 'bg': 'white', 'unit': 'word', 'title': True},
    {'n': 9, 'frames': 90, 'device': 'red_latest', 'bg': 'black', 'unit': 'word', 'beat': 3},
    {'n': 10, 'frames': 95, 'device': 'plain', 'bg': 'black', 'unit': 'word'},
    {'n': 11, 'frames': 156, 'device': 'plain', 'bg': 'black', 'unit': 'char', 'rate': 2.5, 'center': True, 'title': True},
]
LAST = TEMPLATES[-1]


def fit(template, words, chars):
    if template.get('title'):
        return chars <= 14
    return len(words) >= 3


def assign(paragraphs):
    return E.assign_in_order(TEMPLATES[:-1], paragraphs, fit=fit, last=LAST)


def schedule(template, text, seed):
    """각 단위(글자 또는 단어)가 나타나는 시각(초). 단어 단위는 3~10프레임 불규칙 박자."""
    if template['unit'] == 'char':
        rate = template.get('rate', 2.0)
        return [k * rate / FPS for k in range(len(text))]
    words = E.split_words(text)
    rng = E.stable_rng(seed, 'beat')
    times, t = [], 0.0
    for _ in words:
        times.append(t)
        t += (template.get('beat') or int(rng.integers(3, 11))) / FPS
    return times


def minimum(template, text):
    times = schedule(template, text, 0)
    return (times[-1] if times else 0) + .5 + max(.4, len(text.replace(' ', '')) / 15)


def wrap(words, font, max_w):
    lines, cur = [], []
    for w in words:
        trial = ' '.join(cur + [w])
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def dashed_rect(draw, box, color, scale):
    x0, y0, x1, y1 = box
    dash, gap = 5 * scale, 4 * scale
    for (a, b, horizontal, fixed) in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(b, p + dash)
            if horizontal:
                draw.line((p, fixed, q, fixed), fill=color, width=max(1, int(scale)))
            else:
                draw.line((fixed, p, fixed, q), fill=color, width=max(1, int(scale)))
            p = q + gap
    r = 2.5 * scale
    for hx in (x0, (x0 + x1) / 2, x1):
        for hy in (y0, (y0 + y1) / 2, y1):
            if hx == (x0 + x1) / 2 and hy == (y0 + y1) / 2:
                continue
            draw.rectangle((hx - r, hy - r, hx + r, hy + r), fill=(250, 250, 250, 255), outline=color)


def render(scene, local, width, height, info):
    tpl = scene['template']
    t = local
    white = tpl['bg'] == 'white'
    image = Image.new('RGBA', (width, height), (250, 250, 252, 255) if white else (4, 4, 6, 255))
    draw = ImageDraw.Draw(image)
    ink = INK if white else WHITE
    scale = height / 540
    text = scene['text']
    words = E.split_words(text)
    title = tpl.get('title')
    # 원본 실측: 제목 캡 약 0.08H, 문단 캡 약 0.05H·줄 폭 0.55W 안에서 줄바꿈
    size = height * (.14 if title else .072)
    font = E.styled_font('SemiBold', int(size))
    max_w = width * (.62 if title else .55)
    if title and font.getlength(text) > max_w:
        size *= max_w / font.getlength(text)
        font = E.styled_font('SemiBold', int(size))
    times = schedule(tpl, text, scene.get('index', 0))
    if tpl['unit'] == 'char':
        shown_chars = sum(1 for tt in times if t >= tt)
        visible = text[:shown_chars]
        vis_words = visible.split(' ')
    else:
        count = sum(1 for tt in times if t >= tt)
        vis_words = words[:count]
    vis_words = [w for w in vis_words if w] or ([] if not text else [])
    lines = wrap(vis_words, font, max_w)
    line_h = size * 1.22
    left = width * .14 if not tpl.get('center') else None
    # 제목은 가운데, 문단은 첫 줄을 고정하고 아래로 줄이 늘어난다(원본 f367~f440, f972~f1031)
    top = height * .5 if title else height * .36
    total_words = len(vis_words)
    k_word = 0
    extent = None
    space = font.getlength(' ')
    for row, line_words in enumerate(lines):
        line_text = ' '.join(line_words)
        lw = font.getlength(line_text)
        x = (width / 2 - lw / 2) if left is None else left
        y = top + row * line_h
        ascent = size * .78
        if tpl['device'] == 'blue_select' and line_text:
            draw.rectangle((x - 2 * scale, y - ascent * .62, x + lw + 2 * scale, y + ascent * .62), fill=(*SELECT_BLUE, 255))
        for w in line_words:
            ww = font.getlength(w)
            latest = k_word == total_words - 1
            color = ink
            if tpl['device'] == 'labels':
                draw.rectangle((x - 3 * scale, y - ascent * .58, x + ww + 3 * scale, y + ascent * .62), fill=(*LABEL_GRAY, 255))
                color = WHITE
            if tpl['device'] == 'red_latest' and latest and t < times[min(k_word, len(times) - 1)] + 1.2:
                draw.rectangle((x - 2 * scale, y - ascent * .58, x + ww + 2 * scale, y + ascent * .6), fill=(*RED, 255))
                color = WHITE
            draw.text((x, y), w, font=font, fill=(*color, 255), anchor='lm')
            x += ww + space
            k_word += 1
        x_end = x - space
        extent = (min(extent[0], (width / 2 - lw / 2) if left is None else left) if extent else ((width / 2 - lw / 2) if left is None else left),
                  top - ascent * .7, max(extent[2], x_end) if extent else x_end, y + ascent * .7)
    if extent and tpl['device'] == 'dashed':
        pad = 4 * scale
        dashed_rect(draw, (extent[0] - pad, extent[1] - pad, extent[2] + pad, extent[3] + pad),
                    (150, 150, 160, 255) if not white else (90, 90, 100, 255), scale)
    if extent and tpl.get('caret') and int(t * FPS / 8) % 2 == 0:
        draw.line((extent[2] + 2 * scale, extent[3] - size * 1.05, extent[2] + 2 * scale, extent[3] - size * .1), fill=(*ink, 255),
                  width=max(1, int(1.5 * scale)))
    c = (210, 210, 215, 255) if white else (60, 60, 66, 255)
    draw.line((width * .44, height * .93, width * .56, height * .93), fill=c, width=max(1, int(scale)))
    return image.convert('RGB')


def build(engine):
    global E
    E = engine
    return engine.Sample('63211445', 'selection_type', FPS, TEMPLATES, assign, minimum, render)
