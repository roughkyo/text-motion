"""샘플 선택, 원문 적용, 음악 선택, 스토리보드 확인 후 MP4를 만드는 로컬 도구."""
import argparse
import base64
import hashlib
import io
import json
import math
import mimetypes
import re
import secrets
import subprocess
import threading
import unicodedata
import zipfile
import importlib.util
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from xml.etree import ElementTree

from PIL import Image, ImageDraw, ImageFont

SKILL = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('text_motion_kinetic', SKILL / 'scripts/2026-10-05-kinetic-engine.py')
kinetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kinetic)
PRESETS = [
    ('11971627', 'purple_words', '보라 강조 · 단어 공개'),
    ('57032891', 'bold_pop', '굵은 글자 · 줄별 팝인'),
    ('57523104', 'mono_stack', '흑백 · 줄별 슬라이드'),
    ('60153473', 'rgb_trail', 'RGB 분리 · 잔상'),
    ('60608725', 'smoke_cool', '푸른 연기 · 첫 단어 확대·타이핑'),
    ('60806086', 'smoke_purple', '보라 연기 · 강조·겹침 전환'),
    ('60944035', 'outline_scan', '외곽선 · 채움 스캔'),
    ('62295607', 'clean_scale', '중앙 축소 · 선명화'),
    ('62999763', 'letter_assemble', '글자 조립 · 자간 수축'),
    ('63211445', 'selection_type', '왼쪽 고정 · 선택 타이핑'),
]
DATE = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d')


def extract(path):
    suffix = path.suffix.lower()
    warnings = []
    if suffix in ('.md', '.txt'):
        text = path.read_text(encoding='utf-8-sig')
        if suffix == '.md':
            text = re.sub(r'^\s{0,3}#{1,6}\s+', '', text, flags=re.M)
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.M)
            text = re.sub(r'!\[([^]]*)\]\([^)]*\)', r'\1', text)
            text = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', text)
            text = text.replace('**', '').replace('__', '').replace('`', '')
    elif suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or '' for page in reader.pages]
        if any(not page.strip() for page in pages):
            raise ValueError('텍스트가 없는 PDF 페이지가 있습니다. 스캔/이미지 페이지를 확인하고 OCR 결과를 제공해주세요.')
        text = '\n\n'.join(pages)
        warnings.append('PDF의 단·표 읽기 순서는 원본과 다를 수 있습니다. 이미지 속 글자는 제외됩니다. 추출문을 확인해주세요.')
    elif suffix == '.pptx':
        from pptx import Presentation
        result = []
        def shapes_text(shapes):
            for shape in sorted(shapes, key=lambda value: (value.top, value.left)):
                if hasattr(shape, 'shapes'):
                    yield from shapes_text(shape.shapes)
                elif shape.has_text_frame:
                    yield shape.text
                elif shape.has_table:
                    for row in shape.table.rows:
                        yield ' | '.join(cell.text for cell in row.cells)
        for slide in Presentation(path).slides:
            result.extend(shapes_text(slide.shapes))
        text = '\n\n'.join(result)
        warnings.append('PPTX 텍스트·표를 슬라이드별 위→아래 순서로 읽었습니다. 이미지·차트 안의 글자는 추출하지 않았습니다.')
    elif suffix == '.hwpx':
        with zipfile.ZipFile(path) as archive:
            sections = sorted((name for name in archive.namelist() if re.fullmatch(r'Contents/section\d+\.xml', name)), key=lambda value: int(re.search(r'section(\d+)', value)[1]))
            if not sections:
                raise ValueError('HWPX 본문 section XML을 찾지 못했습니다.')
            lines = []
            for name in sections:
                tree = ElementTree.fromstring(archive.read(name))
                # 텍스트 노드를 한 번씩만 읽어 중첩 표 문단이 중복되는 것을 막는다.
                for paragraph in tree.iter():
                    if paragraph.tag.rsplit('}', 1)[-1] != 'p':
                        continue
                    def own_text(node):
                        for child in node:
                            local = child.tag.rsplit('}', 1)[-1]
                            if local == 'p':
                                continue
                            if local == 't':
                                yield child.text or ''
                            elif local == 'lineBreak':
                                yield '\n'
                            elif local == 'tab':
                                yield '\t'
                            else:
                                yield from own_text(child)
                    value = ''.join(own_text(paragraph)).strip()
                    if value:
                        lines.append(value)
            text = '\n'.join(lines)
        warnings.append('HWPX 본문·표 텍스트를 XML 순서로 읽었습니다. 이미지 속 글자는 제외됩니다.')
    elif suffix == '.docx':
        from docx import Document
        doc = Document(path)
        lines = []
        for child in doc.element.body:
            if child.tag.rsplit('}', 1)[-1] == 'p':
                lines.append(''.join(node.text or '' for node in child.iter() if node.tag.rsplit('}', 1)[-1] == 't'))
            elif child.tag.rsplit('}', 1)[-1] == 'tbl':
                for row in child:
                    if row.tag.rsplit('}', 1)[-1] == 'tr':
                        lines.append(' | '.join(''.join(node.text or '' for node in cell.iter() if node.tag.rsplit('}', 1)[-1] == 't') for cell in row if cell.tag.rsplit('}', 1)[-1] == 'tc'))
        text = '\n'.join(lines)
        warnings.append('DOCX 문단·표 텍스트를 읽었습니다. 이미지 속 글자는 제외됩니다.')
    else:
        raise ValueError('지원 형식은 MD, TXT, PDF, PPTX, HWPX, DOCX입니다. HWP는 HWPX로 변환해주세요.')
    text = unicodedata.normalize('NFC', text).strip()
    if not text:
        raise ValueError('추출된 텍스트가 없습니다.')
    if len(text) > 50000:
        raise ValueError('원문이 50,000자를 넘습니다. 필요한 파일 범위를 나눠주세요. 임의 요약은 하지 않았습니다.')
    return text, warnings


def split_text(text, limit=60):
    # 문구를 삭제하지 않고 장면 길이에 맞게 분할한다.
    scenes = []
    for paragraph in re.split(r'\n\s*\n', text):
        rest = paragraph.strip()
        while rest:
            cut = min(len(rest), limit)
            if len(rest) > limit:
                breaks = [match.end() for match in re.finditer(r'[\s.!?。！？]', rest[:limit]) if match.end() >= limit // 2]
                if breaks:
                    cut = breaks[-1]
            scenes.append(rest[:cut].strip())
            rest = rest[cut:].strip()
    if re.sub(r'\s+', '', ''.join(scenes)) != re.sub(r'\s+', '', text):
        raise ValueError('장면 분할 중 원문 불일치가 발견됐습니다.')
    return scenes


@lru_cache(maxsize=80)
def font(size):
    candidates = ['C:/Windows/Fonts/malgunbd.ttf', '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc', '/System/Library/Fonts/AppleSDGothicNeo.ttc']
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise ValueError('한국어 폰트를 찾지 못했습니다. 맑은 고딕 또는 Noto Sans CJK가 필요합니다.')


def layout(text):
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    size = 68 if len(text) > 35 else 84
    f = font(size)
    lines = []
    line = ''
    # 짧은 단어는 통째로 넘겨 한 글자만 다음 줄에 남는 경우를 줄인다.
    for token in re.findall(r'\S+|[^\S\n]+|\n', text):
        if token == '\n':
            lines.append(line)
            line = ''
        elif measure.textlength(line + token, font=f) <= 1500:
            line += token
        elif measure.textlength(token.strip(), font=f) <= 1500:
            lines.append(line.rstrip())
            line = token.lstrip()
        else:
            for char in token:
                if measure.textlength(line + char, font=f) > 1500:
                    lines.append(line.rstrip())
                    line = char
                else:
                    line += char
    if line:
        lines.append(line.rstrip())
    if len(lines) > 5:
        raise ValueError('한 장면이 5줄을 넘습니다. 짧은 장면으로 나누어주세요.')
    return lines, size


def timeline(text, preset, target_seconds=None, pacing='standard'):
    return kinetic.make_timeline(text, preset, target_seconds, pacing)


def frame(scene, preset, local, width=640, height=360):
    return kinetic.frame(scene, preset, local, width, height)


def validate_music(data):
    candidates = data.get('candidates', [])
    if not isinstance(candidates, list) or not 0 <= len(candidates) <= 3:
        raise ValueError('음악 후보는 최대 3곡입니다. 적합한 곡이 없으면 빈 목록을 사용하세요.')
    for candidate in candidates:
        url = urlparse(candidate['url'])
        if url.scheme != 'https' or url.hostname != 'pixabay.com' or '/music/' not in url.path or '/search/' in url.path:
            raise ValueError('곡 후보는 실제 Pixabay 음악 상세 페이지여야 합니다.')
        for key in ('title', 'artist', 'duration', 'content_id', 'reason', 'checked_at'):
            if not candidate.get(key):
                raise ValueError('음악 후보 필수 정보가 빠졌습니다: ' + key)
        # 외부 검증 자체를 대신하지 않고, 세 조건의 확인 기록이 빠진 후보를 차단한다.
        commercial = candidate.get('commercial_use', {})
        if commercial.get('free') is not True or not commercial.get('evidence') or not commercial.get('checked_at'):
            raise ValueError('상업적 무료 이용 확인 근거와 확인일이 필요합니다.')
        if commercial.get('license_url') not in ('https://pixabay.com/service/license-summary/', 'https://pixabay.com/service/terms/'):
            raise ValueError('공식 Pixabay 라이선스 근거를 연결해주세요.')
        if candidate.get('topic_match') is not True:
            raise ValueError('원문 주제 적합성을 확인한 후보만 연결할 수 있습니다.')
        sound = candidate.get('sound', {})
        if sound.get('style') != 'urban_hiphop' or not sound.get('reason'):
            raise ValueError('묵직하고 트렌디한 urban 힙합 비트의 판단 근거가 필요합니다.')
        if sound.get('status') not in ('metadata_only', 'listened', 'user_confirmed'):
            raise ValueError('사운드 확인 상태는 metadata_only, listened, user_confirmed 중 하나여야 합니다.')
    return data


def render(project, audio, output, progress_callback=lambda value: None, width=1920, height=1080, fps=30):
    if not audio or not audio.exists():
        raise ValueError('선택한 음악 파일이 필요합니다. 무음으로 생성하지 않습니다.')
    if output.exists():
        raise ValueError('기존 출력 파일을 덮어쓰지 않습니다.')
    audio_probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(audio)]))
    if not any(stream['codec_type'] == 'audio' for stream in audio_probe['streams']):
        raise ValueError('연결한 파일에 오디오 스트림이 없습니다.')
    duration = project['duration']
    frames = math.ceil(duration * fps)
    video_duration = frames / fps
    args = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-stream_loop', '-1', '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-t', str(video_duration), '-af', f'volume=0.65,afade=t=in:d=0.5,afade=t=out:st={max(0, video_duration - 0.8)}:d=0.8', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(output)]
    # 프레임 파일 수천 장을 쓰지 않고 인코더로 바로 전송한다.
    log_path = output.with_suffix('.log')
    with log_path.open('xb') as log:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in range(frames):
                t = index / fps
                process.stdin.write(kinetic.project_frame(project, t, width, height).tobytes())
                if index % fps == 0:
                    progress_callback(round(index / frames * 95))
            process.stdin.close()
            code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
    if code:
        raise ValueError('영상 인코딩 오류: ' + log_path.read_text(encoding='utf-8', errors='replace')[-1500:])
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)]))
    video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    assert video['width'] == width and video['height'] == height
    from fractions import Fraction
    assert float(Fraction(video['avg_frame_rate'])) == fps
    assert any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    assert abs(float(probe['format']['duration']) - video_duration) < 0.15
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    progress_callback(100)
    return {'width': width, 'height': height, 'fps': fps, 'duration': video_duration, 'audio': True, 'full_decode': 'passed', 'probe': probe}


def combine_sources(paths):
    # 여러 원본을 순서대로 합치되 파일 경계를 별도 기록해 단일 결과물로 만든다.
    texts, warnings, sources = [], [], []
    offset = 0
    for value in paths:
        path = Path(value).resolve()
        text, notes = extract(path)
        sources.append({'filename': path.name, 'start': offset, 'end': offset + len(text)})
        texts.append(text)
        warnings.extend(notes)
        offset += len(text) + 2
    text = '\n\n'.join(texts)
    if len(text) > 50000:
        raise ValueError('통합 원문이 50,000자를 넘습니다. 범위를 나눠주세요. 임의 요약하지 않았습니다.')
    return text, warnings, sources


@lru_cache(maxsize=20)
def original_thumbnail(path, _mtime):
    # 원본 영상의 실제 프레임을 사용한다. 재구성 프리셋 이미지를 샘플로 표시하지 않는다.
    seconds = {'11971627.mp4': '4.65', '60944035.mp4': '7.5'}.get(Path(path).name, '3')
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', seconds, '-i', path, '-frames:v', '1', '-vf', 'scale=640:-2', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1'])


def create_output_folder(workspace, topic):
    # 주제는 폴더 이름으로만 사용하고 경로 구분자·예약 문자를 제거한다.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', topic.strip())
    name = re.sub(r'\s+', '-', name).strip(' .-')[:70].rstrip(' .-') or '모션그래픽'
    output_root = (workspace / 'ouputs').resolve()
    if not output_root.is_relative_to(workspace):
        raise ValueError('ouputs 폴더는 현재 작업 폴더 안에 있어야 합니다.')
    output_root.mkdir(exist_ok=True)
    number = 1
    while True:
        suffix = '' if number == 1 else f'-{number:02d}'
        folder = output_root / f'{DATE}-{name}{suffix}'
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            # 기존 결과를 보존하며 동시에 시작한 작업도 같은 폴더를 쓰지 않는다.
            number += 1


class Studio:
    def __init__(self, args):
        self.workspace = Path(args.workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError('출력 작업 폴더가 존재하지 않습니다.')
        topic = getattr(args, 'project_topic', None)
        if not topic and args.source:
            first_source = args.source[0] if isinstance(args.source, list) else args.source
            topic = Path(first_source).stem
        self.folder = create_output_folder(self.workspace, topic or '모션그래픽')
        self.references = Path(args.reference_dir).resolve() if args.reference_dir else SKILL / 'assets/motion_graphic_sample'
        self.source = ''
        self.sources = []
        self.text = ''
        self.warnings = []
        self.preset = None
        self.sample_filename = None
        self.duration_choice = None
        self.target_seconds = None
        self.audio = None
        self.music = None
        self.revision = 0
        self.project = None
        self.preview = {'phase': 'waiting'}
        self.rhythm = None
        self.review_request = None
        self.pacing = 'tight'
        self.status = {'phase': 'idle', 'progress': 0}
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()
        candidate_file = Path(args.music_json) if args.music_json else SKILL / 'assets/2026-10-05-music-demo.json'
        self.music_data = validate_music(json.loads(candidate_file.read_text(encoding='utf-8')))
        self.demo_candidates = not bool(args.music_json)
        if args.source:
            paths = args.source if isinstance(args.source, list) else [args.source]
            self.text, self.warnings, self.sources = combine_sources(paths)
            self.source = ' + '.join(item['filename'] for item in self.sources)
        if getattr(args, 'resume_state', None):
            saved = json.loads(Path(args.resume_state).read_text(encoding='utf-8'))
            self.music_data = validate_music(saved['music_data'])
            self.select_sample(saved['sample_filename'])
            self.select_duration(saved['duration_choice'], saved['target_seconds'])
            self.text = saved['text']
            self.source = saved['source']
            self.sources = saved['sources']
            self.music = saved['music']
            self.review_request = saved['review_request']
            self.revision = saved['revision'] + 1
            # 복원 시 기존 승인과 프로젝트는 재사용하지 않고 미리보기를 다시 만든다.

    def select_sample(self, filename):
        if self.status['phase'] == 'rendering':
            raise ValueError('생성 중에는 샘플을 변경할 수 없습니다.')
        matching = [item for item in PRESETS if item[0] + '.mp4' == filename]
        if not matching:
            raise ValueError('목록의 원본 MP4 파일명을 선택해주세요.')
        if not self.references or not (self.references / filename).is_file():
            raise ValueError('원본 샘플 영상이 없습니다. motion_graphic_sample 경로를 연결해주세요.')
        self.sample_filename = filename
        self.preset = matching[0][1]
        self.project = None
        self.revision += 1
        self.status = {'phase': 'idle', 'progress': 0}
        return {'sample_filename': filename, 'preset': self.preset}

    def select_duration(self, choice, seconds=None):
        if self.status['phase'] == 'rendering':
            raise ValueError('생성 중에는 영상 길이를 변경할 수 없습니다.')
        fixed = {'15': 15.0, '30': 30.0, '50-60': 55.0}
        if choice in fixed:
            target = fixed[choice]
        elif choice == 'custom':
            try:
                if isinstance(seconds, bool):
                    raise ValueError('불리언은 초 단위 입력이 아닙니다.')
                target = float(seconds)
            except (TypeError, ValueError) as error:
                raise ValueError('기타를 선택했다면 초 단위 숫자를 직접 입력해주세요.') from error
            if not math.isfinite(target) or target <= 0:
                raise ValueError('영상 길이는 0보다 큰 유한한 초 단위 숫자여야 합니다.')
        else:
            raise ValueError('15초 / 30초 / 50~60초 / 기타 중 하나를 선택해주세요.')
        self.duration_choice = choice
        self.target_seconds = target
        self.project = None
        self.revision += 1
        self.status = {'phase': 'idle', 'progress': 0}
        return {'duration_choice': choice, 'target_seconds': target}

    def prepare(self, request_id=None):
        if self.status['phase'] == 'rendering':
            raise ValueError('생성 중에는 구성을 변경할 수 없습니다.')
        if not self.sample_filename:
            raise ValueError('원본 샘플 영상 10개 중 하나를 먼저 선택해주세요.')
        if not self.text.strip():
            raise ValueError('소스 파일을 선택해주세요.')
        if self.target_seconds is None:
            raise ValueError('사용자가 영상 길이를 먼저 선택해야 합니다: 15초 / 30초 / 50~60초 / 기타 직접 입력.')
        scenes, duration = timeline(self.text, self.preset, self.target_seconds, self.pacing)
        if self.audio:
            self.rhythm = kinetic.analyze_audio(self.audio, duration)
        self.revision += 1
        self.project = {'source': self.source, 'sources': self.sources, 'text': self.text, 'sample_filename': self.sample_filename, 'preset': self.preset, 'scenes': scenes, 'duration': duration, 'duration_choice': self.duration_choice, 'target_seconds': self.target_seconds, 'pacing': self.pacing, 'revision': self.revision, 'warnings': self.warnings, 'music': self.music, 'music_file': self.audio.name if self.audio else None}
        kinetic.bind_rhythm(self.project, self.rhythm or {'events': [], 'status': 'awaiting_audio'})
        if self.review_request and request_id == self.review_request['id']:
            self.review_request['pending'] = False
            self.review_request['resolved_revision'] = self.revision
        self.status = {'phase': 'ready', 'progress': 0}
        snapshot = self.folder / f'{DATE}-storyboard-r{self.revision:03d}.json'
        snapshot.write_text(json.dumps(self.project, ensure_ascii=False, indent=2), encoding='utf-8')
        self.build_preview()
        return self.project

    def refresh_project(self):
        # 음악만 바뀌면 장면은 유지하고 새 구성 번호로 최종 확인을 무효화한다.
        if self.project:
            self.revision += 1
            self.project = dict(self.project, revision=self.revision, music=self.music, music_file=self.audio.name if self.audio else None)
            # 음원을 바꾸면 컷/비트 시간표를 재작성하고 승인 전 새 미리보기를 만든다.
            scenes, duration = timeline(self.text, self.preset, self.target_seconds, self.pacing)
            self.project.update(scenes=scenes, duration=duration)
            kinetic.bind_rhythm(self.project, self.rhythm or {'events': [], 'status': 'awaiting_audio'})
            snapshot = self.folder / f'{DATE}-storyboard-r{self.revision:03d}.json'
            snapshot.write_text(json.dumps(self.project, ensure_ascii=False, indent=2), encoding='utf-8')
            self.build_preview()
        else:
            self.revision += 1

    def build_preview(self):
        self.preview = {'phase': 'awaiting_audio', 'revision': self.revision}
        if not self.project or not self.audio:
            return
        project = json.loads(json.dumps(self.project))
        audio = self.audio
        revision = self.revision
        output = self.folder / f'{DATE}-preview-r{revision:03d}-{secrets.token_hex(2)}.mp4'
        self.preview = {'phase': 'rendering', 'revision': revision}
        def work():
            try:
                render(project, audio, output, width=960, height=540, fps=30)
                with self.lock:
                    if self.revision == revision:
                        self.preview = {'phase': 'done', 'revision': revision, 'file': output.name}
            except Exception as error:
                with self.lock:
                    if self.revision == revision:
                        self.preview = {'phase': 'error', 'revision': revision, 'error': str(error)}
        threading.Thread(target=work, daemon=True).start()

    def request_review(self, text):
        if not self.project or not text.strip():
            raise ValueError('스토리보드와 수정 요청 내용을 확인해주세요.')
        self.review_request = {'id': secrets.token_hex(4), 'text': text.strip(), 'pending': True, 'revision': self.revision}
        target = self.folder / f'{DATE}-review-request-{self.review_request["id"]}.json'
        target.write_text(json.dumps(self.review_request, ensure_ascii=False, indent=2), encoding='utf-8')
        self.refresh_project()
        return self.review_request

    def upload(self, data):
        if not self.sample_filename:
            raise ValueError('원본 샘플 영상을 먼저 선택해주세요.')
        name = Path(data['name']).name
        suffix = Path(name).suffix.lower()
        payload = base64.b64decode(data['data'], validate=True)
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError('파일은 32MB 이하로 선택해주세요.')
        target = self.folder / f'{DATE}-input-{secrets.token_hex(3)}{suffix}'
        target.write_bytes(payload)
        kind = data.get('kind', 'source')
        if kind == 'source':
            text, warnings = extract(target)
            self.text, self.warnings, self.source = text, warnings, name
            self.sources = [{'filename': name, 'start': 0, 'end': len(text)}]
            self.audio = None
            self.music = None
            # 이전 주제의 음악을 새 원문에 자동 적용하지 않는다.
            self.music_data = {'topic': '새 원문의 주제에 맞는 음악 후보를 범용스킬에 요청해주세요.', 'candidates': []}
            self.demo_candidates = False
        elif kind == 'music_json':
            self.music_data = validate_music(json.loads(payload.decode('utf-8-sig')))
            self.demo_candidates = False
            self.music = None
            self.audio = None
        elif kind == 'audio':
            if not self.music:
                raise ValueError('음악 후보를 먼저 선택해주세요.')
            if suffix not in ('.mp3', '.wav', '.m4a', '.ogg', '.flac'):
                raise ValueError('MP3/WAV/M4A/OGG/FLAC 파일을 선택해주세요.')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(target)]))
            if not any(stream['codec_type'] == 'audio' for stream in probe['streams']):
                raise ValueError('파일에 오디오 스트림이 없습니다.')
            self.rhythm = kinetic.analyze_audio(target, self.target_seconds)
            self.audio = target
            self.music = dict(self.music, connected_filename=name, audio_sha256=hashlib.sha256(payload).hexdigest())
        else:
            raise ValueError('알 수 없는 파일 종류입니다.')
        return {'name': name, 'warnings': self.warnings}

    def select_music(self, index, sound_confirmed=False):
        if not self.sample_filename:
            raise ValueError('원본 샘플 영상을 먼저 선택해주세요.')
        if index < 0 or index >= len(self.music_data['candidates']):
            raise ValueError('알 수 없는 음악 후보입니다.')
        candidate = self.music_data['candidates'][index]
        validate_music({'candidates': [candidate]})
        selected = json.loads(json.dumps(candidate))
        if candidate['sound']['status'] == 'metadata_only':
            if sound_confirmed is not True:
                raise ValueError('곡을 듣고 묵직한 urban 힙합 사운드를 확인한 뒤 선택해주세요.')
            selected['sound']['status'] = 'user_confirmed'
            selected['sound']['confirmed_at'] = DATE
        self.music = selected
        self.audio = None
        self.rhythm = None
        self.refresh_project()
        self.status = {'phase': 'idle', 'progress': 0}
        return {'selected': self.music}

    def begin_render(self, data):
        if self.preview.get('phase') != 'done' or self.preview.get('revision') != self.revision:
            raise ValueError('현재 구성의 음악 포함 30fps 미리보기를 먼저 준비해야 합니다.')
        if self.review_request and self.review_request['pending']:
            raise ValueError('수정 요청이 아직 반영되지 않았습니다. 갱신된 미리보기를 먼저 확인해주세요.')
        if not self.project or data.get('revision') != self.revision or data.get('confirmed') is not True:
            raise ValueError('현재 스토리보드를 확인해주세요.')
        if not self.music or not self.audio:
            raise ValueError('선택한 음악 파일을 연결해주세요.')
        validate_music({'candidates': [self.music]})
        if self.music['sound']['status'] == 'metadata_only':
            raise ValueError('음악을 듣고 urban 힙합 사운드 조건을 확인해주세요.')
        if self.status['phase'] == 'rendering':
            raise ValueError('이미 생성 중입니다.')
        project = json.loads(json.dumps(self.project))
        audio = self.audio
        output = self.folder / f'{DATE}-text-motion-r{self.revision:03d}-{secrets.token_hex(2)}.mp4'
        self.status = {'phase': 'rendering', 'progress': 0}
        def work():
            try:
                def update(value):
                    self.status['progress'] = value
                check = render(project, audio, output, update)
                record = {'project': project, 'music': self.music, 'validation': check}
                output.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                self.status = {'phase': 'done', 'progress': 100, 'file': output.name, 'path': str(output), 'validation': {key: check[key] for key in ('width', 'height', 'fps', 'duration', 'audio', 'full_decode')}}
            except Exception as error:
                self.status = {'phase': 'error', 'error': str(error)}
        threading.Thread(target=work, daemon=True).start()
        return self.status


def serve(studio, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, content_type='application/json; charset=utf-8', status=200):
            raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def file(self, path):
            if not path or not path.is_file():
                self.send({'error': '파일이 없습니다.'}, status=404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get('Range', '')
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
            if requested and not match:
                self.send({'error': 'Invalid range'}, status=416)
                return
            if match:
                start = int(match[1])
                end = min(size - 1, int(match[2]) if match[2] else size - 1)
                if start > end:
                    self.send({'error': 'Invalid range'}, status=416)
                    return
            self.send_response(206 if match else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if match:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/':
                    self.send((SKILL / 'assets/2026-10-05-studio.html').read_bytes(), 'text/html; charset=utf-8')
                elif parsed.path == '/api/state':
                    self.send({'token': studio.token, 'revision': studio.revision, 'duration_choice': studio.duration_choice, 'target_seconds': studio.target_seconds, 'review_request': studio.review_request, 'presets': [{'id': identifier, 'filename': identifier + '.mp4', 'key': key, 'name': name, 'reference': bool(studio.references and (studio.references / (identifier + '.mp4')).is_file())} for identifier, key, name in PRESETS], 'sample_filename': studio.sample_filename, 'preset': studio.preset, 'text': studio.text, 'source': studio.source, 'sources': studio.sources, 'music_data': studio.music_data, 'music': studio.music, 'audio': studio.audio.name if studio.audio else None, 'demo_candidates': studio.demo_candidates, 'status': studio.status, 'preview': studio.preview, 'project': studio.project, 'output_folder': str(studio.folder)})
                elif parsed.path == '/api/updates':
                    self.send({'revision': studio.revision, 'phase': studio.status['phase'], 'preview': studio.preview, 'progress': studio.status.get('progress', 0)})
                elif parsed.path.startswith('/thumbnail/'):
                    identifier = parsed.path.split('/')[-1]
                    if identifier not in [item[0] for item in PRESETS] or not studio.references:
                        raise ValueError('원본 샘플 경로가 없습니다.')
                    path = studio.references / (identifier + '.mp4')
                    self.send(original_thumbnail(str(path), path.stat().st_mtime_ns), 'image/jpeg')
                elif parsed.path == '/api/status':
                    self.send(studio.status)
                elif parsed.path == '/api/frame':
                    with studio.lock:
                        project = studio.project
                        if not project:
                            raise ValueError('스토리보드를 먼저 만들어주세요.')
                        index = min(len(project['scenes']) - 1, max(0, int(query.get('scene', ['0'])[0])))
                        scene = project['scenes'][index]
                        local = min(scene['duration'], max(0, float(query.get('t', [str(scene['reveal'] + 0.5)])[0])))
                        img = kinetic.project_frame(project, scene['start'] + local)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    self.send(buffer.getvalue(), 'image/jpeg')
                elif parsed.path.startswith('/reference/'):
                    identifier = parsed.path.split('/')[-1]
                    if identifier not in [item[0] for item in PRESETS]:
                        raise ValueError('알 수 없는 샘플입니다.')
                    self.file(studio.references / (identifier + '.mp4') if studio.references else None)
                elif parsed.path == '/audio':
                    self.file(studio.audio)
                elif parsed.path.startswith('/output/'):
                    path = (studio.folder / parsed.path.split('/')[-1]).resolve()
                    if path.parent != studio.folder or path.suffix != '.mp4':
                        raise ValueError('허용된 출력 폴더 밖입니다.')
                    self.file(path)
                else:
                    self.send({'error': 'Not found'}, status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.send({'error': str(error)}, status=400)

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                    raise ValueError('외부 사이트의 조작은 허용하지 않습니다.')
                if self.headers.get('X-Studio-Token') != studio.token:
                    raise ValueError('조작 토큰이 일치하지 않습니다.')
                length = int(self.headers.get('Content-Length', '0'))
                if length > 45 * 1024 * 1024:
                    raise ValueError('업로드 용량이 너무 큽니다.')
                data = json.loads(self.rfile.read(length))
                with studio.lock:
                    if studio.status['phase'] == 'rendering':
                        raise ValueError('생성 중에는 구성을 변경할 수 없습니다.')
                    if 'expected_revision' in data and data['expected_revision'] != studio.revision:
                        raise ValueError('사용자 선택이 바뀌었습니다. 현재 상태를 다시 확인해주세요.')
                    if self.path == '/api/sample':
                        value = studio.select_sample(data['filename'])
                    elif self.path == '/api/duration':
                        value = studio.select_duration(data['choice'], data.get('seconds'))
                    elif self.path == '/api/upload':
                        if data.get('kind') == 'audio' and data.get('selected_music_url') != (studio.music or {}).get('url'):
                            raise ValueError('연결할 음악이 현재 선택한 곡과 다릅니다.')
                        value = studio.upload(data)
                        if data.get('kind') == 'source':
                            studio.project = None
                        studio.refresh_project()
                        studio.status = {'phase': 'idle', 'progress': 0}
                    elif self.path == '/api/agent-source':
                        text = data['text'].strip()
                        if not text or len(text) > 50000:
                            raise ValueError('원문은 1~50,000자여야 합니다.')
                        studio.text, studio.sources = text, data['sources']
                        studio.source = ' + '.join(item['filename'] for item in studio.sources)
                        studio.warnings = data.get('warnings', [])
                        studio.project = None
                        studio.review_request = None
                        studio.audio = None
                        studio.music = None
                        studio.music_data = {'topic': '범용스킬이 주제와 사운드 조건에 맞는 음악 후보를 준비합니다.', 'candidates': []}
                        studio.revision += 1
                        value = {'source': studio.source}
                    elif self.path == '/api/agent-music':
                        studio.music_data = validate_music(data['music_data'])
                        studio.music = None
                        studio.audio = None
                        studio.refresh_project()
                        value = {'candidates': len(studio.music_data['candidates'])}
                    elif self.path == '/api/review-request':
                        value = studio.request_review(data['text'])
                    elif self.path == '/api/prepare':
                        if not studio.sample_filename or data.get('preset') != studio.preset:
                            raise ValueError('원본 샘플을 선택하고 해당 구성으로 갱신해주세요.')
                        new_text = data['text'].strip()
                        if new_text != studio.text:
                            studio.warnings = studio.warnings + ['제작용 문구가 원문 추출본과 다릅니다.']
                            studio.project = None
                            studio.revision += 1
                        if len(new_text) > 50000:
                            raise ValueError('원문이 50,000자를 넘습니다.')
                        studio.text = new_text
                        pacing = data.get('pacing', studio.pacing)
                        if pacing not in ('standard', 'tight'):
                            raise ValueError('지원하지 않는 모션 속도 설정입니다.')
                        studio.pacing = pacing
                        value = studio.prepare(data.get('request_id'))
                        (studio.folder / f'{DATE}-extracted-r{studio.revision:03d}.txt').write_text(studio.text, encoding='utf-8')
                        for index, scene in enumerate(value['scenes']):
                            frame(scene, studio.preset, scene['reveal'] + 0.5).save(studio.folder / f'{DATE}-storyboard-r{studio.revision:03d}-scene-{index + 1:03d}.jpg')
                    elif self.path == '/api/music':
                        value = studio.select_music(int(data['index']), data.get('sound_confirmed', False))
                    elif self.path == '/api/render':
                        value = studio.begin_render(data)
                    else:
                        raise ValueError('알 수 없는 작업입니다.')
                self.send(value)
            except Exception as error:
                self.send({'error': str(error)}, status=400)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'http://127.0.0.1:{server.server_port}', flush=True)
    print('OUTPUT ' + str(studio.folder), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--project-topic', help='출력 폴더에 사용할 짧은 프로젝트 주제')
    parser.add_argument('--reference-dir')
    parser.add_argument('--source', nargs='+')
    parser.add_argument('--music-json')
    parser.add_argument('--resume-state')
    parser.add_argument('--port', type=int, default=0)
    arguments = parser.parse_args()
    serve(Studio(arguments), arguments.port)
