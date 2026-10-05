"""샘플별 동작, 음원 트랜지언트, 동일 시간축의 프레임 합성."""
import bisect
import hashlib
import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

VERSION = 'kinetic-3-reference-flow'
# 초 단위 관찰값은 references 문서의 원프레임 근거와 대응한다.
PROFILES = {
    'purple_words': ('11971627', 1.05, 0.30, ['word', 'blur', 'tracking', 'type'], ['cut', 'blur', 'whip']),
    'bold_pop': ('57032891', 3.25, 0.75, ['pop', 'scatter', 'reflect', 'pop'], ['rotate', 'whip']),
    'mono_stack': ('57523104', 2.50, 0.50, ['stack', 'vertical', 'stack', 'glitch'], ['whip', 'vertical']),
    'rgb_trail': ('60153473', 2.12, 0.65, ['ribbon', 'rgb', 'outline', 'stretch'], ['slice', 'zoom']),
    'smoke_cool': ('60608725', 2.25, 0.75, ['hero', 'type', 'word', 'blur'], ['blur', 'overlap']),
    'smoke_purple': ('60806086', 1.75, 0.65, ['hero', 'word', 'type', 'tracking'], ['overlap', 'blur']),
    'outline_scan': ('60944035', 2.07, 0.60, ['outline', 'scan', 'outline', 'pop'], ['zoom', 'slice']),
    'clean_scale': ('62295607', 1.48, 0.50, ['scale', 'word', 'blur', 'scale'], ['blur', 'cut']),
    'letter_assemble': ('62999763', 1.00, 0.58, ['scatter', 'tracking', 'stretch', 'rotate'], ['cut', 'rotate', 'whip']),
    'selection_type': ('63211445', 2.46, 0.90, ['type', 'select', 'highlight', 'type'], ['cut']),
}


def clamp(x):
    return max(0.0, min(1.0, x))


def ease(x):
    return 1 - (1 - clamp(x)) ** 4


@lru_cache(maxsize=1)
def background_profiles():
    path = Path(__file__).resolve().parents[1] / 'references/2026-10-05-background-profiles.json'
    return json.loads(path.read_text(encoding='utf-8'))


def reference_time(preset, t, project_duration):
    profile = background_profiles()[PROFILES[preset][0]]
    duration = profile['duration']
    # 요청 길이에 원본 전체 흐름을 한 번만 대응한다. 모듈로 반복·왕복 재생을 하지 않는다.
    return min(duration-1/profile['fps'], max(0, t) * duration / max(.001, project_duration))


def background_state(preset, source_time):
    profile = background_profiles()[PROFILES[preset][0]]
    intervals = profile['light_intervals']
    index = max(0, bisect.bisect_right([s['start'] for s in intervals], source_time)-1)
    return intervals[index]


@lru_cache(maxsize=128)
def font(size, regular=False):
    names = ['C:/Windows/Fonts/malgun.ttf' if regular else 'C:/Windows/Fonts/malgunbd.ttf',
             '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc' if regular else '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' if regular else '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
             '/System/Library/Fonts/AppleSDGothicNeo.ttc']
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, max(1, int(size)))
    raise ValueError('한국어 폰트가 필요합니다: 맑은 고딕 또는 Noto Sans CJK.')


def analyze_audio(path, duration):
    """대역별 스펙트럼 증가량으로 킥/스네어 후보를 구한다. 악기 분리 모델은 아니다."""
    rate, hop, nfft = 22050, 220, 1024
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-stream_loop', '-1', '-i', str(path),
                                   '-t', str(duration), '-ac', '1', '-ar', str(rate), '-f', 'f32le', 'pipe:1'])
    signal = np.frombuffer(raw, dtype='<f4')
    signal = np.pad(signal, (nfft // 2, nfft))
    windows = np.lib.stride_tricks.sliding_window_view(signal, nfft)[::hop]
    spec = np.abs(np.fft.rfft(windows * np.hanning(nfft), axis=1))
    delta = np.maximum(0, np.diff(spec, axis=0, prepend=spec[:1]))
    freq = np.fft.rfftfreq(nfft, 1 / rate)
    events = []
    for kind, low, high, spacing in [('kick', 35, 180, 0.22), ('snare_candidate', 650, 6500, 0.32)]:
        mask = (freq >= low) & (freq < high)
        flux = np.sqrt(np.mean(delta[:, mask] ** 2, axis=1))
        if float(flux.max()) < 1e-7:
            continue
        # 중간 대역 몸통도 있어야 하이햇만 있는 피크를 덜 선택한다.
        if kind == 'snare_candidate':
            body = np.sqrt(np.mean(delta[:, (freq >= 180) & (freq < 2500)] ** 2, axis=1))
            flux *= np.minimum(1, body / (np.percentile(body, 75) + 1e-8))
        threshold = max(float(np.percentile(flux, 72)), float(np.median(flux) + 1.1 * np.std(flux)))
        peaks = [i for i in range(1, len(flux) - 1) if flux[i] > threshold and flux[i] >= flux[i-1] and flux[i] > flux[i+1] and i * hop / rate < duration]
        accepted = []
        for i in sorted(peaks, key=lambda j: float(flux[j]), reverse=True):
            t = i * hop / rate
            if all(abs(t - j * hop / rate) >= spacing for j in accepted):
                accepted.append(i)
        for i in sorted(accepted):
            events.append({'time': round(i * hop / rate, 5), 'kind': kind,
                           'strength': round(float(flux[i] / flux.max()), 4)})
    events.sort(key=lambda e: e['time'])
    return {'method': 'band_spectral_flux_v1', 'status': 'estimated_not_instrument_verified',
            'audio_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'offset_seconds': 0,
            'analysis_hop_seconds': hop / rate, 'events': events,
            'note': '스네어 후보 자동 추정. 실제 악기 및 모든 스네어 검출을 보장하지 않음. 음악 동반 미리보기 확인 필요.'}


def make_timeline(text, preset, target_seconds=None, pacing='standard'):
    if preset not in PROFILES or pacing not in ('standard', 'tight'):
        raise ValueError('알 수 없는 샘플/속도 설정입니다.')
    profile = PROFILES[preset]
    paragraphs = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
    if not paragraphs:
        raise ValueError('제작 문구가 비어 있습니다.')
    scenes = []
    for i, content in enumerate(paragraphs):
        if len(content) > 80:
            raise ValueError('한 장면이 80자를 넘습니다. 원문 근거를 유지해 짧은 문구로 편집하세요.')
        lines = content.splitlines()
        if len(lines) == 1 and preset in ('mono_stack', 'bold_pop') and len(content.split()) > 1:
            lines = content.split()
        if len(lines) > 5:
            raise ValueError('장면당 최대 5줄입니다. 문구를 분리하세요.')
        visible = len(re.sub(r'\s', '', content))
        mode = profile[3][i % len(profile[3])]
        reveal = profile[2] * (0.68 if pacing == 'tight' else 1)
        if preset == 'selection_type':
            reveal = max(0.35, (visible - 1) / (12 if pacing == 'tight' else 6))
        reveal = max(reveal, 0.10 * (len(lines) - 1) + 0.16)
        # 짧은 키워드를 1.4초씩 정지시키지 않는다. 최종 공개 후 읽기 하한은 별도 보장한다.
        hold = max(0.22, visible / (17 if pacing == 'tight' else 12))
        transition = 0.13 if pacing == 'tight' else 0.22
        minimum = reveal + hold + transition
        natural = max(minimum, profile[1] * (0.60 if pacing == 'tight' else 1))
        size = 190 if preset in ('mono_stack', 'bold_pop') else 145
        if preset.startswith('smoke') or preset == 'selection_type':
            size = 86 if pacing == 'standard' else 116
        if visible <= 5 and preset not in ('selection_type', 'smoke_cool', 'smoke_purple'):
            size = 260
        measure = font(size)
        size = min(size, size * 1560 / max(1, max(measure.getlength(line) for line in lines)), 750 / (len(lines) * 1.13))
        scenes.append({'text': content, 'lines': lines, 'font_size': size, 'duration': natural,
                       'minimum_duration': minimum, 'reveal': reveal, 'hold': hold, 'transition': transition,
                       'pacing': pacing, 'motion': mode, 'exit': profile[4][i % len(profile[4])],
                       'index': i, 'renderer': VERSION, 'reference_id': profile[0]})
    minimum = sum(s['minimum_duration'] for s in scenes)
    natural = sum(s['duration'] for s in scenes)
    target = natural if target_seconds is None else float(target_seconds)
    if not math.isfinite(target) or target <= 0:
        raise ValueError('영상 길이는 양의 유한한 숫자여야 합니다.')
    if target + 1e-6 < minimum:
        raise ValueError(f'현재 문구는 최소 {minimum:.2f}초가 필요합니다. 문구를 줄이거나 장면을 편집하세요.')
    if target > max(natural * 1.6, len(scenes) * 1.6) and pacing == 'tight':
        raise ValueError(f'문구가 부족합니다. 현재 {len(scenes)}장면의 자연스러운 길이는 약 {natural:.1f}초입니다. 원문에서 키워드를 더 추출하세요. 긴 정지로 채우지 않습니다.')
    weights = [max(0.15, s['duration'] - s['minimum_duration']) for s in scenes]
    cursor = 0.0
    for s, weight in zip(scenes, weights):
        s['project_duration'] = target
        s['duration'] = s['minimum_duration'] + (target - minimum) * weight / sum(weights)
        s['start'] = cursor
        s['hold'] = s['duration'] - s['reveal'] - s['transition']
        cursor += s['duration']
    return scenes, target


def bind_rhythm(project, rhythm):
    """읽기 하한을 지키는 범위에서만 컷을 실제 음원 피크 쪽으로 이동한다."""
    project['rhythm'] = rhythm
    scenes = project['scenes']
    peaks = [e['time'] for e in rhythm.get('events', [])]
    for i in range(1, len(scenes)):
        left, right = scenes[i-1], scenes[i]
        boundary = right['start']
        end = right['start'] + right['duration']
        eligible = [t for t in peaks if abs(t - boundary) <= 0.12 and
                    t - left['start'] >= left['minimum_duration'] and end - t >= right['minimum_duration']]
        if eligible:
            new = min(eligible, key=lambda t: abs(t - boundary))
            left['duration'] = new - left['start']
            right['start'], right['duration'] = new, end - new
    for scene in scenes:
        scene['project_duration'] = project['duration']
        scene['hold'] = scene['duration'] - scene['reveal'] - scene['transition']
        scene['events'] = rhythm.get('events', [])
    project['renderer'] = VERSION
    return project


@lru_cache(maxsize=1024)
def sprite(text, size, color, regular=False, outline=False, heavy=False):
    f = font(size, regular)
    stroke = max(1, round(size * (0.015 if heavy else 0.010))) if outline or heavy else 0
    box = f.getbbox(text, stroke_width=stroke)
    pad = max(4, round(size * 0.18))
    im = Image.new('RGBA', (max(1, math.ceil(f.getlength(text)) + pad * 2), size + pad * 2))
    ImageDraw.Draw(im).text((pad, pad - box[1]), text, font=f,
                           fill=(0, 0, 0, 0) if outline else color,
                           stroke_width=stroke, stroke_fill=color)
    return im


@lru_cache(maxsize=256)
def shaded_sprite(text,size,color):
    im = sprite(text,size,color).copy()
    rgb = np.asarray(im).copy()
    ramp = np.linspace(1.12,.35,im.height)[:,None,None]
    rgb[:,:,:3] = np.clip(rgb[:,:,:3]*ramp,0,255).astype(np.uint8)
    return Image.fromarray(rgb)


def place(dst, im, x, y, sx=1, sy=None, angle=0, alpha=1, blur=0):
    sy = sx if sy is None else sy
    if alpha <= 0 or sx <= 0 or sy <= 0:
        return
    transformed = im
    if abs(sx-1) > .005 or abs(sy-1) > .005:
        transformed = im.resize((max(1, round(im.width * sx)), max(1, round(im.height * sy))), Image.Resampling.BICUBIC)
    if angle:
        transformed = transformed.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    if blur > .2:
        transformed = transformed.filter(ImageFilter.GaussianBlur(blur))
    if alpha < 1:
        transformed = transformed.copy()
        transformed.putalpha(transformed.getchannel('A').point(lambda a: round(a * alpha)))
    dst.alpha_composite(transformed, (round(x - transformed.width/2), round(y - transformed.height/2)))


@lru_cache(maxsize=2)
def background_chunk(path, chunk):
    # 2초 단위 디코딩 캐시로 전체 영상 수백 MB를 한꺼번에 상주시키지 않는다.
    raw = subprocess.check_output(['ffmpeg','-v','error','-ss',str(chunk*2),'-i',path,
                                   '-t','2','-vf','scale=640:360,fps=30','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,360,640,3)


def reference_background(preset, t, width, height):
    color = 'cool' if preset=='smoke_cool' else 'purple'
    path = Path(__file__).resolve().parents[1]/'assets'/f'2026-10-05-smoke-{color}-continuous-clean-v4.mp4'
    if not path.exists():
        raise ValueError('연기 배경 에셋이 없습니다. build-backgrounds.py로 재생성하거나 완전한 패키지를 설치하세요.')
    duration = background_profiles()[PROFILES[preset][0]]['duration']
    phase = min(max(0, t), duration-1/30)
    chunk = int(phase//2)
    frames = background_chunk(str(path),chunk)
    index = min(len(frames)-1,int((phase-chunk*2)*30))
    im = Image.fromarray(frames[index])
    return im.resize((width,height),Image.Resampling.BILINEAR).convert('RGBA')


def background(preset, width, height, t, index):
    if preset.startswith('smoke'):
        return reference_background(preset,t,width,height)
    profile = background_profiles()[PROFILES[preset][0]]
    state = background_state(preset, t)
    # 동일 상태 안에서만 색을 보간한다. 원본의 한 프레임 전환을 페이드로 바꾸지 않는다.
    samples = [s for s in profile['edge_color_samples'] if state['start'] <= s['time'] < state['end']]
    if not samples:
        value = 248 if state['light'] else 3
        grid = np.full((2, 2, 3), value, dtype=np.uint8)
    else:
        j = max(0, bisect.bisect_right([s['time'] for s in samples], t)-1)
        a, b = samples[j], samples[min(j+1, len(samples)-1)]
        q = clamp((t-a['time']) / max(.001, b['time']-a['time']))
        grid = ((1-q)*np.array(a['grid'])+q*np.array(b['grid'])).clip(0,255).astype(np.uint8).reshape(2,2,3)
    image = Image.fromarray(grid).resize((width, height), Image.Resampling.BILINEAR).convert('RGBA')
    if preset == 'rgb_trail':
        d = ImageDraw.Draw(image)
        for radius in (.2, .38, .62):
            r = height * radius
            d.ellipse((width/2-r, height/2-r, width/2+r, height/2+r), outline=(28, 29, 35), width=1)
    return image


def star(draw, x, y, r, phase, color):
    points = []
    for j in range(8):
        a = phase + j * math.pi/4
        radius = r if j % 2 == 0 else r * .20
        points.append((x + math.cos(a) * radius * (.55 + .45 * abs(math.cos(phase))), y + math.sin(a) * radius))
    draw.polygon(points, fill=color)


def fragments(layer, t, origin, strength, seed, color):
    age = t - origin
    if not 0 <= age < .38:
        return
    w, h = layer.size
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(layer)
    for j in range(12):
        angle = rng.uniform(0, math.tau)
        distance = w * (.12 + .34 * ease(age/.38))
        x, y = w/2 + math.cos(angle) * distance, h/2 + math.sin(angle) * distance * .60
        r = w * rng.uniform(.005, .018) * (1-age/.38) * strength
        c = (*color[:3], round(200 * (1-age/.38)))
        if j % 3:
            d.polygon([(x-r, y-r), (x+r*1.7, y), (x-r*.3, y+r)], fill=c)
        else:
            d.line((x-r*3, y, x+r*3, y), fill=c, width=max(1, round(r*.4)))


def text_layer(scene, preset, local, width, height):
    layer = Image.new('RGBA', (width, height))
    scale = width / 1920
    index, mode = scene['index'], scene['motion']
    source_time = reference_time(preset, scene['start']+local, scene.get('project_duration', scene['start']+scene['duration']))
    light = background_state(preset, source_time)['light']
    base = (18, 17, 28, 255) if light else (247, 245, 252, 255)
    accents = [(152, 121, 250, 255), (80, 220, 203, 255), (236, 100, 170, 255), (86, 232, 134, 255)]
    accent = accents[index % 4]
    size = max(10, round(scene['font_size'] * scale))
    lines = scene['lines']
    line_h = size * (1.07 if preset in ('bold_pop', 'mono_stack') else 1.28)
    top = height/2 - (len(lines)-1) * line_h/2
    reveal = scene['reveal']
    out = clamp((local - (scene['duration'] - scene['transition'])) / scene['transition'])
    total_chars = sum(len(line) for line in lines)
    offset = 0
    for row, line in enumerate(lines):
        regular = preset == 'mono_stack' and row % 3 == 1
        f = font(size, regular)
        lengths = [f.getlength(c) for c in line]
        full = sum(lengths)
        left = width*.23 if preset == 'selection_type' else (width-full)/2
        # 완료된 타이핑의 시작점은 고정; 전체 문장을 재정렬하지 않는다.
        if preset == 'selection_type' and left+full > width*.91:
            left = width*.08
        y = top + row * line_h
        progress = ease((local-row*.10) / max(.12, reveal-row*.10))
        if mode in ('stack', 'vertical', 'glitch', 'scan', 'outline', 'reflect', 'ribbon', 'rgb', 'scale'):
            color = accent if mode in ('outline',) else base
            im = sprite(line, size, color, regular, mode == 'outline', preset in ('mono_stack', 'bold_pop'))
            dx = (1-progress) * width * (.8 if row%2 == 0 else -.8) if mode in ('stack', 'glitch') else 0
            dy = (1-progress) * height*.55 if mode == 'vertical' else 0
            factor = (2.3-1.3*progress) if mode in ('scale', 'rgb') else (.20+.80*progress if mode in ('scan','outline') else 1)
            if mode == 'reflect':
                reflected = ImageOps.flip(im)
                place(layer, reflected, width/2, y+size*.95, 1.25-.25*progress, .65, alpha=.22)
            if mode in ('ribbon', 'rgb'):
                for j in range(7, 0, -1):
                    col = [(255,40,70,255),(40,190,255,255),(255,215,45,255)][j%3]
                    ghost = sprite(line, size, col)
                    length = (1-progress)*width*.09*j
                    place(layer, ghost, width/2+length*(1 if index%2 else -1), y+length*.24, factor, alpha=.48)
            if mode == 'outline':
                filled = sprite(line, size, base)
                scan = (local/max(.1, scene['duration']))
                if scan > .60:
                    im = filled
                elif .25 < scan < .50:
                    mask = Image.new('L', im.size)
                    stripe = round(im.height * ((scan-.25)/.25))
                    ImageDraw.Draw(mask).rectangle((0, stripe-5*scale, im.width, stripe+size*.28), fill=255)
                    im = Image.alpha_composite(im, Image.composite(filled, Image.new('RGBA', im.size), mask))
            if mode == 'scan':
                outline = sprite(line, size, accent, outline=True)
                mask = Image.new('L', im.size)
                ImageDraw.Draw(mask).rectangle((0,0,im.width,im.height*progress), fill=255)
                im = Image.alpha_composite(outline, Image.composite(im, Image.new('RGBA', im.size), mask))
            place(layer, im, width/2+dx, y+dy, factor, alpha=progress if mode=='scale' else 1,
                  blur=(1-progress)*5*scale if mode=='rgb' else 0)
        else:
            x = left
            first_word = len(line.split()[0]) if line.split() else 0
            for j, char in enumerate(line):
                rank = offset+j
                delay = rank/max(1,total_chars-1)*max(.02,reveal-.18)
                if mode=='word':
                    words = list(re.finditer(r'\S+',scene['text'].replace('\n','')))
                    word_index = next((wi for wi,match in enumerate(words) if match.start()<=rank<match.end()),0)
                    delay = word_index/max(1,len(words))*max(.02,reveal-.18)
                p = ease((local-delay)/.18)
                if mode in ('type','select','highlight'):
                    p = 1.0 if local >= delay else 0.0
                if p <= 0 or char.isspace():
                    x += lengths[j]
                    continue
                color = accent if (mode in ('hero','word','tracking') and local < reveal+.12) or (mode=='pop' and row==0) else base
                dx, dy, factor, angle, blur = 0, 0, 1, 0, 0
                if mode in ('scatter','stretch','tracking','rotate'):
                    q = ease((local-delay*.45)/max(.12,reveal*.70))
                    dx = (1-q)*(j-(len(line)-1)/2)*size*1.4
                    dy = (1-q)*height*(.4 if j%2 else -.4) if mode in ('scatter','stretch') else 0
                    factor = 1+(1-q)*(1.8 if j%2==0 else .4)
                    angle = (1-q)*(30 if j%2 else -24) if mode in ('scatter','rotate') else 0
                elif mode == 'hero' and j < first_word and row == 0:
                    factor = 2.6-1.6*ease(max(0,local-.15)/max(.15,reveal*.40))
                    dx = (x+lengths[j]/2-width/2)*(factor-1)
                    blur = (1-p)*5*scale
                elif mode == 'pop':
                    factor = .25+.75*p+.16*math.sin(p*math.pi)
                    dy = (1-p)*height*.24
                    angle = (1-p)*(12 if j%2 else -12)
                elif mode == 'blur':
                    blur = (1-p)*12*scale
                    dy = (1-p)*size*.20
                elif mode == 'word':
                    dy = (1-p)*size*.9
                if preset == 'selection_type' and mode in ('select','highlight'):
                    d = ImageDraw.Draw(layer)
                    current_word = j >= max(0, int(total_chars*clamp(local/reveal))-4)
                    if mode != 'highlight' or current_word:
                        d.rectangle((x-2*scale,y-size*.48,x+lengths[j]+2*scale,y+size*.50), fill=(110,151,233,180) if light else (87,70,135,200))
                im = sprite(char, size, color, heavy=preset=='bold_pop')
                if preset=='letter_assemble' and mode in ('tracking','rotate'):
                    im = shaded_sprite(char,size,color)
                if mode=='blur' and out>0:
                    char_exit = clamp((out-rank/max(1,total_chars)*.35)/.65)
                    p *= 1-char_exit
                    blur += char_exit*14*scale
                if mode=='stretch':
                    place(layer, im, x+lengths[j]/2+dx, y+dy, factor, 1+(1-p)*2.5, angle=angle, alpha=p)
                else:
                    place(layer, im, x+lengths[j]/2+dx, y+dy, factor, angle=angle, alpha=p, blur=blur)
                x += lengths[j]
            if mode in ('type','select'):
                shown = sum(1 for j in range(len(line)) if local >= (offset+j)/max(1,total_chars-1)*max(.02,reveal-.18))
                extent = sum(lengths[:shown])
                d = ImageDraw.Draw(layer)
                if shown and preset == 'selection_type':
                    box = (left-4*scale,y-size*.55,left+extent+4*scale,y+size*.55)
                    d.rectangle(box, outline=accent, width=max(1,round(scale)))
                    for px in (box[0],box[2]):
                        for py in (box[1],box[3]):
                            d.rectangle((px-2*scale,py-2*scale,px+2*scale,py+2*scale), fill=base)
                if local < reveal and int(local*8)%2==0:
                    d.line((left+extent,y-size*.45,left+extent,y+size*.45),fill=base,width=max(1,round(2*scale)))
        offset += len(line)
    # 문장 전체 페이드 대신 샘플군별 퇴장 궤적을 적용한다.
    if out > 0:
        effect = scene['exit']
        target = Image.new('RGBA', layer.size)
        if effect in ('whip','vertical','rotate','zoom'):
            place(target, layer, width/2 + (out**3*width*1.1 if effect in ('whip','rotate') else 0),
                  height/2 + (out**3*height if effect=='vertical' else 0),
                  1+out**3*2 if effect=='zoom' else 1, angle=-out**3*18 if effect=='rotate' else 0,
                  alpha=1-out*.4)
        elif effect=='blur' and mode=='blur':
            target = layer
        elif effect in ('blur','overlap'):
            place(target, layer, width/2-out*width*.04, height/2, alpha=1-out, blur=out*14*scale)
        elif effect=='slice':
            for j in range(8):
                y0, y1 = round(j*height/8), round((j+1)*height/8)
                crop = layer.crop((0,y0,width,y1))
                target.alpha_composite(crop,(round(out**3*width*(1 if j%2 else -1)),y0))
        else:
            target = layer
        layer = target
    return layer


def frame(scene, preset, local, width=640, height=360):
    local = max(0, min(local, scene['duration']-1/300))
    t = scene['start']+local
    source_time = reference_time(preset, t, scene.get('project_duration', scene['start']+scene['duration']))
    image = background(preset,width,height,source_time,scene['index'])
    layer = text_layer(scene,preset,local,width,height)
    events = scene.get('events', []) if scene['pacing']=='tight' else []
    recent_kicks = [e for e in events if e['kind']=='kick' and 0 <= t-e['time'] < .18]
    if recent_kicks:
        e = recent_kicks[-1]
        pulse = .055*e['strength']*(1-(t-e['time'])/.18)**2
        enlarged = Image.new('RGBA',layer.size)
        place(enlarged,layer,width/2,height/2,1+pulse)
        layer = enlarged
    # 파편은 진입/스네어 후보에 짧게만 나타나 텍스트 유지 구간을 가리지 않는다.
    if scene['pacing']=='tight' and not preset.startswith('smoke'):
        fragments(image,t,scene['start'],.8,scene['index']+41,(165,130,255))
        for j,e in enumerate(events):
            if e['kind']=='snare_candidate':
                fragments(image,t,e['time'],e['strength'],j+100,(120,211,255))
    if preset in ('purple_words','clean_scale','letter_assemble'):
        star(ImageDraw.Draw(image),width*.80,height*.26,width*.025,t*1.4,(136,103,220,220))
    image.alpha_composite(layer)
    if preset=='rgb_trail' or (preset=='mono_stack' and scene['motion']=='glitch'):
        strength = max(0,1-local/max(.01,scene['reveal']))
        if strength > .02:
            channels = image.convert('RGB').split()
            dx = round(width*.022*strength)
            image = Image.merge('RGB',(ImageChops.offset(channels[0],dx,0),channels[1],ImageChops.offset(channels[2],-dx,0))).convert('RGBA')
    # 배경·글자 색은 원본 배경 시간표가 결정한다. 음악 후보로 전면 색을 반전하지 않는다.
    return image.convert('RGB')


def project_frame(project, t, width=640, height=360):
    scenes = project['scenes']
    i = max(0,bisect.bisect_right([s['start'] for s in scenes],t)-1)
    scene = scenes[i]
    local = t-scene['start']
    image = frame(scene,project['preset'],local,width,height)
    # 연기 문구 교체에서만 짧게 전후 텍스트를 겹친다. 배경은 이중 합성하지 않는다.
    if i and scenes[i-1]['exit']=='overlap' and local < .10:
        previous = scenes[i-1]
        old = text_layer(previous,project['preset'],previous['duration']-.10+local,width,height)
        image = image.convert('RGBA')
        image.alpha_composite(old)
        image = image.convert('RGB')
    return image
