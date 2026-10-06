# text-motion

원본 샘플 영상을 선택하고 문서 내용을 적용해 음악이 포함된 텍스트 모션그래픽을 만드는 Codex 스킬입니다.

## 사용 흐름

1. PDF, MD, PPTX, HWPX 등의 소스를 제공합니다.
2. 브라우저에서 샘플 10개 중 하나와 영상 길이를 선택합니다.
3. Codex가 스토리보드와 Pixabay 음악 후보를 제공합니다.
4. 음악과 미리보기를 확인하고 수정사항을 전달합니다.
5. 최종 승인 후 MP4를 생성합니다.

## 설치

### Claude Code 플러그인

Claude Code에서 다음 명령을 실행합니다.

```text
/plugin marketplace add roughkyo/text-motion
/plugin install text-motion@text-motion-all
```

### 스킬 폴더만 설치

저장소를 복제한 뒤 `skills/text-motion` 폴더를 사용하는 환경의 스킬 디렉터리에 복사합니다. 등록 이름은 `text-motion`입니다.

```sh
git clone https://github.com/roughkyo/text-motion.git
cp -r text-motion/skills/text-motion ~/.codex/skills/
```

기존 설치용 ZIP은 [v0.3.0 릴리스](https://github.com/roughkyo/text-motion/releases/tag/v0.3.0)에서 받을 수 있습니다. `SKILL.md`와 `scripts`, `references`, `assets`를 함께 유지해야 합니다.

## 실행 환경

- Python과 `requirements.txt`의 패키지
- FFmpeg와 FFprobe
- 맑은 고딕 또는 Noto Sans CJK 등 엔진에서 지원하는 한글 폰트

```sh
python -m pip install -r requirements.txt
python -B skills/text-motion/scripts/2026-10-05-text-motion.py --workspace "작업폴더" --project-topic "프로젝트주제" --source "소스.pdf"
```

`--workspace`에는 존재하는 폴더를 지정합니다. 결과는 `ouputs/YYYY-MM-DD-프로젝트주제/`에 모이며, 중복 이름에는 `-02`, `-03`이 붙습니다.

## 포함 내용

- 원본 샘플 MP4 10개와 파생 연기 배경 2개
- 샘플별 텍스트 모션과 배경 상태 시간표
- 스토리보드·음악 검토용 로컬 웹 화면
- 1080p·30fps H.264+AAC 렌더러

## 폴더 구조

```text
.claude-plugin/          플러그인·마켓플레이스 정보
skills/text-motion/
  SKILL.md               에이전트 작업 지시서
  agents/                Codex/OpenAI 표시 정보
  scripts/               제작·분석·렌더 스크립트
  references/            모션 사양과 분석 자료
  assets/                샘플 영상·폰트·브라우저 화면
requirements.txt         Python 의존성
```

배경이 유지되는 샘플은 유지하고, 변화하는 샘플은 전환 순서와 시간 비율을 따릅니다. 모든 샘플에 공통 색 반전을 적용하지 않습니다. 선택 박스는 선택형 샘플에만 사용합니다.

## 검증과 한계

일본 관광 PDF로 30초·24장면 영상을 생성해 900프레임 전체 디코딩과 오디오를 확인했습니다. 배경·타이핑 회귀 검사 13개 및 패키지 이동 후 프레임 생성 검사를 통과했습니다.

원본 프로젝트의 완전 복제는 아닙니다. 연기 글자 제거로 세부 질감·밝기가 달라지고, 비연기 배경은 색 표본 기반 2D 근사입니다. 킥·스네어 후보는 자동 추정입니다. 다른 OS에서의 실제 실행은 검증하지 않았습니다.

샘플 영상·폰트·음원 등의 권리는 각각의 권리자에게 있습니다. 이 저장소는 제3자 에셋에 별도의 재사용 라이선스를 부여하지 않습니다. 개인 PDF와 다운로드 음악은 포함하지 않습니다.

## 제작 기록

[샘플 기반 스킬 제작·수정 기록](skills/text-motion/references/2026-10-05-text-motion-노션정리.md)

## 만든 사람

양파고 (Yang Phago)
