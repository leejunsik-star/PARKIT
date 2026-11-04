# 맥·Windows 공통 실행 정리 (2026-10-28)

대상: `docs/도구/live_detect.py`, `docs/도구/capture.py`
동료 맥 환경: macOS 26.3.1 (Apple Silicon), Python 3.12.14, Tk 9.0 / Tcl 9.0.4, PyTorch 2.8.0, NumPy 2.3.5, Pillow 12.3.0, OpenCV 4.11.0.86, 가상환경 `.venv-gui`

## 1. 처음에 맥에서 꺼진 이유

`맥환경 오류코드.txt`를 보면 코드 문제가 아니라 **파이썬 환경 문제**였습니다.

| 기록에 나온 내용 | 뜻 |
|---|---|
| `/Applications/Xcode.app/.../Python3.framework/Versions/3.9/...` | Xcode에 딸려 오는 Python 3.9.6으로 실행됨 |
| `/System/Library/Frameworks/Tk.framework/Versions/8.5` (8.5.9) | 그 파이썬은 애플이 오래전에 지원을 끝낸 Tk 8.5를 씀 |
| `Tcl_Panic` ← `TkpInit` ← `_tkinter` | 창을 만드는 첫 줄 `tk.Tk()`에서 Tk가 스스로 멈춤 |
| `EXC_CRASH (SIGABRT)`, `abort() called` | 파이썬 오류 메시지 없이 프로그램이 바로 꺼짐 |

즉 macOS 26에서는 Tk 8.5가 시작하자마자 죽습니다. `live_detect.py`가 아니라 `tk.Tk()` 한 줄짜리 파일도 똑같이 꺼집니다. python.org Python 3.12(Tk 9.0 포함)로 바꾼 것이 실제 해결책이었습니다.

동료가 코드도 고쳤다면 그 수정본은 아직 못 봤습니다. 위 원인만 보면 코드 수정 없이 파이썬만 바꿔도 실행됐을 것으로 봅니다(추정).

## 2. 지금 코드에서 맥·Windows 차이가 나는 곳

| 항목 | 지금 상태 | 맥 | Windows | 조치 |
|---|---|---|---|---|
| 파일 경로 | `HERE` + `os.path.join` | 문제 없음 | 문제 없음 | 그대로 |
| 웹캠 열기 | Windows는 DSHOW→MSMF→기본, 그 외는 기본(맥은 AVFoundation) | 문제 없음 | 문제 없음 | 그대로 |
| MJPG 설정 | `CAP_PROP_FOURCC` MJPG 요청 | 무시될 수 있음(오류 아님) | 적용 | 그대로 |
| 카메라 권한 | 코드와 무관 | 처음엔 VS Code나 터미널에 카메라 권한이 없어 웹캠이 안 열릴 수 있음 | 해당 없음 | 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 허용 후 VS Code 재시작 |
| 글꼴 | `"맑은 고딕"`, `"Consolas"` 고정 | 없는 글꼴이라 기본 글꼴로 대체(꺼지지는 않음, 모양만 다름) | 정상 | 운영체제별 글꼴 선택(맥은 Apple SD Gothic Neo, Menlo) |
| 화면 저장 | `live_detect.py`의 "현재 화면 저장"과 비교 저장이 `cv2.imwrite` 사용, 저장 위치는 `docs\도구` | 정상 | **한글 폴더라 저장이 조용히 실패할 가능성 큼** (OpenCV의 알려진 한계, Windows에서 직접 시험은 안 함. 지금 `docs\도구`에 `live_*.jpg`가 하나도 없음) | `capture.py`처럼 `imencode` + `tofile`로 저장 |
| 한글 입력 상태 R키 | `capture.py`가 `e.char == "ㄱ"`로 녹화 | 맥 한글 입력기에서는 다르게 들어올 수 있음(미확인) | 정상 | 맥에서 한 번 확인. 안 되면 영문 입력으로 R |
| 신뢰도 조절바 | `ttk.Scale.coords` 사용 | Tk 9.0에도 있는 기본 기능(맥 실물 확인 필요) | 정상 | 맥에서 한 번 확인 |
| 연산 장치 | CUDA만 확인, 없으면 CPU | CPU(Apple GPU인 MPS는 안 씀) | NVIDIA가 있으면 CUDA | 속도 비교는 둘 다 CPU로. MPS는 필요해지면 추가 |
| 오래된 Tk | 검사 없음 | Xcode 파이썬으로 실행하면 또 아무 말 없이 꺼짐 | 해당 없음 | 시작할 때 Tk 버전이 8.6 미만이면 "python.org 3.12로 실행하세요"라고 안내하고 종료 |
| OpenCV 버전 | 설명문에 `4.10.0.84` 고정(5.0은 웹캠을 못 열었음) | 4.11.0.86으로 정상 | 4.10.0.84 | 둘 다 `opencv-python>=4.10,<5` 범위로 맞춤 |

## 3. 앞으로 코드를 쓸 때 지킬 규칙

처음부터 두 환경에서 돌도록 아래를 기본으로 넣습니다.

1. 경로는 항상 스크립트 위치(`HERE`) 기준으로 만들고, `C:\`나 `/Users/...` 같은 절대 경로를 쓰지 않습니다.
2. 이미지 저장은 한 함수(`imencode` + `tofile`)로만 합니다. 한글 폴더에서도 Windows가 저장됩니다.
3. 글꼴은 맨 위에 한 번 정해 두고(`UI_FONT`, `MONO_FONT`) 운영체제에 따라 고릅니다.
4. 웹캠은 Windows만 DSHOW·MSMF를 시도하고, 맥과 리눅스는 기본 백엔드를 씁니다.
5. 단축키는 한글 입력기에 기대지 않습니다. 버튼으로도 같은 동작을 할 수 있게 둡니다.
6. Tk 표준 위젯과 표준 명령만 씁니다(내부 `ttk::...` 명령은 쓰지 않음).
7. 시작할 때 Tk 버전을 확인해서, 맥의 오래된 파이썬이면 꺼지기 전에 안내합니다. (`tkinter.TkVersion`은 창을 만들지 않고 읽을 수 있어서 이 확인 자체로는 꺼지지 않습니다.)
8. 저는 Linux의 Tk(Python 3.12)로 화면 시험을 합니다. 맥 실물 시험은 못 하므로, 큰 변경 뒤에는 아래 4번의 맥 확인 항목을 동료가 한 번 돌려 봅니다.

## 4. 환경 맞추는 방법

**파이썬 버전**
- 맥과 Windows 모두 python.org의 Python 3.12를 씁니다. 같은 버전이면 같은 패키지 버전을 그대로 쓸 수 있습니다.
- 맥에서는 Xcode나 `/usr/bin/python3`(3.9)를 쓰지 않습니다.

**가상환경**
- 이름은 둘 다 `.venv-gui`로 하되, 각자 자기 컴퓨터에서 새로 만듭니다. 폴더를 복사하거나 클라우드·USB로 공유하지 않습니다(안에 운영체제 전용 실행 파일이 들어 있음).
- 맥: `python3.12 -m venv .venv-gui` → `source .venv-gui/bin/activate`
- Windows: `py -3.12 -m venv .venv-gui` → `.venv-gui\Scripts\activate`

**패키지**
- 공통 목록 하나(`requirements-gui-tested.txt`)를 같이 씁니다. 권장 내용:
  ```
  numpy==2.3.5
  pillow==12.3.0
  opencv-python>=4.10,<5
  ultralytics
  ```
- PyTorch는 목록에서 뺍니다. Windows에서 NVIDIA GPU를 쓰려면 CUDA판을 따로 설치해야 해서, 같은 한 줄로 두 운영체제를 다 맞출 수 없습니다. 각자 pytorch.org에서 고른 명령으로 설치합니다.
- 기존 `requirements-lock.txt`는 Python 3.9용 학습·콘솔 환경 기록이라 GUI 실행에는 쓰지 않습니다.

**VS Code 인터프리터**
- `.vscode/settings.json`에 `.venv-gui/bin/python`(맥 경로)을 넣어 공유하면 Windows에서는 그 파일을 못 찾습니다.
- 공유 설정에서는 인터프리터 경로를 빼고, 각자 `Ctrl+Shift+P`(맥은 `Cmd+Shift+P`) → "Python: Select Interpreter"에서 자기 `.venv-gui`를 고릅니다. 맥은 `.venv-gui/bin/python`, Windows는 `.venv-gui\Scripts\python.exe`입니다.

**맥에서 처음 한 번 확인할 것**
1. `python -c "import tkinter; print(tkinter.TkVersion)"` 결과가 9.0인지 확인합니다.
2. `live_detect.py`를 실행해서 웹캠이 열리는지 봅니다. 안 열리면 카메라 권한을 허용하고 VS Code를 다시 켭니다.
3. 신뢰도 조절바를 눌렀을 때 누른 자리로 가는지 봅니다.
4. "현재 화면 저장"을 눌러서 jpg가 생기는지 봅니다.
5. `capture.py`에서 스페이스(촬영), R(녹화), Esc(종료)가 되는지 봅니다. 한글 입력 상태의 R도 확인합니다.

## 5. 공용판 파일 (2026-10-28 추가)

원본 `live_detect.py`, `capture.py`는 그대로 두고, 고친 버전을 새 이름으로 만들었습니다. 쓰는 법(옵션, 화면)은 원본과 같습니다.

| 파일 | 원본에서 바뀐 점 |
|---|---|
| `docs/도구/live_detect_winmac.py` | 운영체제별 글꼴, 한글 폴더에서도 되는 화면 저장(저장 실패하면 결과 창에 "저장 실패"로 표시), 맥의 오래된 파이썬(Tk 8.5)이면 창을 열기 전에 안내하고 종료 |
| `docs/도구/capture_winmac.py` | 운영체제별 글꼴, 맥의 오래된 파이썬 안내 (사진 저장은 원래도 한글 폴더에서 됐음) |

- 실행: `python live_detect_winmac.py`, `python capture_winmac.py` (옵션은 원본과 같음)
- 시험: Linux의 Python 3.12 화면에서 한글 이름 폴더에 단일·비교 화면 저장, 가짜 웹캠 2대로 사진·녹화를 확인했습니다. 맥과 Windows 실물에서는 아직 돌려 보지 않았으므로 4번의 맥 확인 목록을 한 번 돌려 봅니다.
