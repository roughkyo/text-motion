"""Codex가 로컬 제작 서비스에 원문·스토리보드·음악을 전달하는 연결 도구."""
import argparse
import base64
import importlib.util
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def call(url, token=None, data=None):
    headers = {}
    payload = None
    if data is not None:
        headers = {'Content-Type': 'application/json', 'X-Studio-Token': token}
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
    try:
        with urlopen(Request(url, data=payload, headers=headers), timeout=120) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(json.load(error).get('error', str(error))) from error


def run(args):
    parsed = urlparse(args.studio_url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port:
        raise ValueError('현재 제작 도구의 http://127.0.0.1:포트 주소만 사용할 수 있습니다.')
    base = f'http://127.0.0.1:{parsed.port}'
    state = call(base + '/api/state')
    token = state.pop('token')
    if args.action == 'state':
        return state
    if args.action == 'snapshot':
        with Path(args.state_file).open('x', encoding='utf-8') as output:
            json.dump(state, output, ensure_ascii=False, indent=2)
        return {'saved': args.state_file, 'revision': state['revision']}
    if args.action == 'wait':
        # 선택 변경 또는 렌더 상태 변화까지만 기다리고 승인 동작은 하지 않는다.
        deadline = time.monotonic() + min(55, max(0, args.wait_seconds))
        baseline = (state['revision'], state['status'], state.get('preview'))
        while time.monotonic() < deadline:
            time.sleep(1)
            latest = call(base + '/api/state')
            latest.pop('token', None)
            if (latest['revision'], latest['status'], latest.get('preview')) != baseline:
                return latest
        return state
    payload = {'expected_revision': state['revision']}
    if args.action == 'sample':
        payload['filename'] = args.sample_filename
        return call(base + '/api/sample', token, payload)
    if args.action == 'duration':
        payload.update(choice=args.choice, seconds=args.seconds)
        return call(base + '/api/duration', token, payload)
    if args.action == 'source':
        if not args.sources:
            raise ValueError('--sources에 사용자가 제공한 원본 파일을 지정해주세요.')
        spec = importlib.util.spec_from_file_location('motion', Path(__file__).with_name('2026-10-05-text-motion.py'))
        motion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(motion)
        text, warnings, sources = motion.combine_sources(args.sources)
        payload.update(text=text, warnings=warnings, sources=sources)
        return call(base + '/api/agent-source', token, payload)
    if args.action == 'prepare':
        text = Path(args.text_file).read_text(encoding='utf-8-sig') if args.text_file else state['text']
        payload.update(preset=state['preset'], text=text, request_id=args.request_id)
        if args.pacing:
            payload['pacing'] = args.pacing
        return call(base + '/api/prepare', token, payload)
    if args.action == 'music':
        if not args.music_json:
            raise ValueError('Codex가 확인한 후보 파일을 --music-json으로 지정해주세요.')
        payload['music_data'] = json.loads(Path(args.music_json).read_text(encoding='utf-8-sig'))
        return call(base + '/api/agent-music', token, payload)
    if args.action == 'audio':
        if not state['music'] or not args.audio_file:
            raise ValueError('사용자가 선택한 곡과 Codex가 내려받은 파일이 필요합니다.')
        file = Path(args.audio_file)
        if file.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('음악 파일은 32MB 이하여야 합니다.')
        payload.update(kind='audio', name=file.name, selected_music_url=state['music']['url'], data=base64.b64encode(file.read_bytes()).decode())
        return call(base + '/api/upload', token, payload)
    raise ValueError('지원하지 않는 작업입니다.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--studio-url', required=True)
    parser.add_argument('--action', choices=('state', 'snapshot', 'wait', 'sample', 'source', 'duration', 'prepare', 'music', 'audio'), required=True)
    parser.add_argument('--sample-filename')
    parser.add_argument('--state-file')
    parser.add_argument('--wait-seconds', type=float, default=45)
    parser.add_argument('--pacing', choices=('standard', 'tight'))
    parser.add_argument('--choice', choices=('15', '30', '50-60', 'custom'))
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--text-file')
    parser.add_argument('--music-json')
    parser.add_argument('--audio-file')
    parser.add_argument('--request-id')
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), ensure_ascii=False))
