# 33. 동료 웹 지도 합치기: occupancy_v2 (2026-11-04)

동료가 만든 "웹 화면으로 도면과 실시간 차량 위치 보기"를 지금 판정 프로그램에 합쳤다. 원본은 그대로 두고 새 폴더 `docs\도구\occupancy_v2`에 합쳤다.

| 폴더 | 내용 | 바꿨나 |
|---|---|---|
| `docs\도구\occupancy` | 지금 판정·보정 프로그램 (2026-11-04) | 그대로 |
| `docs\도구\occupancy(동료버전)` | 동료 판 (11-01 판정 프로그램 + 웹 지도, 맥에서 만듦) | 그대로 |
| `docs\도구\occupancy_v2` | 위 둘을 합친 판 | 새로 만듦 |

## 1. 동료가 바꾼 것

동료 판은 11-01 묶음(`dataset\handoff_점유판정_20261101`)을 바탕으로 했다. 기존 파일 중 바뀐 것은 `occupancy.py`와 `config\run.json` 둘뿐이고 나머지는 새 파일이다.

- **`occupancy.py`**: 짧은 연결 코드 약 10곳. `--tracking` 또는 `--web`을 줄 때만 켜진다.
  - 분석할 때마다 차 상자의 바닥 접점(매트 cm)과 관측 시각을 차량 추적기로 보낸다.
  - 칸 상태를 갱신할 때마다 칸 상태와 차량 위치를 웹으로 내보낸다.
  - 시작할 때 웹 서버를 열고, 끝낼 때 닫는다.
- **새 파일**: `vehicle_tracker.py`(차량 ID 추적), `tracking_bridge.py`(판정 ↔ 추적기), `parking_api.py`(웹 서버, FastAPI), `web/`(React 지도 화면, 빌드된 `web/dist` 포함), `tracking_demo.py`(카메라 없는 데모), 시험 4개, `TRACKING.md`(설명서), `requirements-web.txt`, `tracking_config.example.json`.
- **`config\run.json`**: `straddle_min_cm` 1.0 → 0.1. 판정 값이라 **합치지 않았다**(4절).

## 2. occupancy_v2에서 한 일

1. 지금 `occupancy` 폴더를 복사했다(기록, 캐시, `이전\` 백업은 뺌).
2. 동료의 새 파일을 복사했다. `web\node_modules`(약 50MB)는 빼고 빌드된 `web\dist`는 넣었다.
3. 동료의 연결 코드를 지금 `occupancy.py`에 다시 넣었다. 연결 지점이 모두 지금 코드에 그대로 있어 같은 내용으로 들어갔다.
4. 지금 판에 맞춘 것:
   - 판정 시작 창에 **[웹 지도 켜기]** 체크. 웹 라이브러리가 없으면 체크를 끄고 설치 명령을 빨간 글씨로 보여 준다.
   - 판정 창에 **[웹 지도 열기]** 버튼(기본 브라우저로 `http://127.0.0.1:8765`).
   - [카메라 고르기…]로 다시 시작할 때 웹 서버를 닫고 다시 연다.
5. 이 PC(Windows, Python 3.9)에서 돌게 고친 것:
   - `vehicle_tracker.py`: `slot: int | None` → `slot: Optional[int]`. 3.10 이상 문법이라 3.9에서 불러오기부터 실패했다.
   - `requirements-web.txt`: 동료가 고정한 fastapi 0.162.0, uvicorn 0.54.0, websockets 16.1.1은 Python 3.10 이상에서만 설치된다. "이상" 조건으로 바꿔 3.9에서는 0.128.8, 0.39.0, 15.0.1이 깔린다.
   - `test_tracking_replay.py`: Windows에서 하위 프로그램의 한글 출력을 cp949로 읽다가 멈췄다. UTF-8로 주고받게 했다.
   - `TRACKING.md`: 맥 경로(`.venv-gui/bin/python runs/occupancy/...`)를 이 폴더와 Windows 기준 명령으로 다시 썼다.
   - `make_handoff.py`: 묶음 이름을 `handoff_점유판정v2_<날짜>\occupancy_v2`로 하고 웹 파일(`web\dist`, `web\src` 등, node_modules 제외)과 웹 문서를 넣는다.

## 3. 이 PC 실행 환경

| 항목 | 이 PC | 필요 |
|---|---|---|
| Python | 3.9.25 (miniconda) | 판정 3.9 이상. 웹 지도도 위 고침으로 3.9에서 된다 |
| ultralytics / opencv / numpy / torch | 8.4.178 / 4.10.0 / 2.0.2 / 2.8.0 CPU | 그대로 |
| fastapi / uvicorn / websockets | 0.128.8 / 0.39.0 / 15.0.1 (2026-11-04 설치) | 웹 지도에만 필요: `python -m pip install -r requirements-web.txt` |
| Node | 24.16 | 웹 화면을 고쳐 다시 빌드할 때만 (22.12 이상) |

맥(동료)은 python.org Python 3.12를 쓰면 같은 파일로 된다.

## 4. 시험 결과 (occupancy_v2, 이 PC)

처음에는 임시 환경(기존 패키지를 같이 보는 venv)에서 시험했고, 2026-11-04에 이 PC의 miniconda에 웹 라이브러리를 설치한 뒤 같은 시험을 다시 해 모두 통과했다. 설치로 새로 들어온 것은 fastapi, uvicorn, websockets, starlette, pydantic, pydantic-core, annotated-types, typing-inspection 8개이고, 기존 패키지는 바뀌지 않았다(`pip check` 문제 없음).

| 시험 | 결과 |
|---|---|
| test_geometry / test_pipeline / test_autoref | 41/41, 33/33, 13/13 |
| test_vehicle_tracker (동료) | 15개 통과 |
| test_tracking_integration (동료, API·WebSocket) | 5개 통과 |
| test_tracking_replay (동료) | 추적 끔/켬/20배속의 칸 상태 기록 107줄이 같음 → 웹 지도를 켜도 판정 결과는 같다 |
| 합성 데모 (`tracking_demo.py`) | 브라우저에서 칸 56개, 움직이는 차량, 차량 선택 화면 확인 |
| 웹캠 0번 + 웹 지도 | 판정 창 [웹 지도 열기] 버튼, `/api/layout` 칸 56개, `/api/state` 칸 상태(빈 칸 34, 사각 22)와 카메라 상태. [카메라 고르기]로 두 번 다시 시작해도 웹 서버가 닫혔다 다시 열림 |
| 묶음 만들기 | 파일 57개, 13.8 MB (웹 화면 포함) |

## 5. 정할 것

- **`straddle_min_cm` (칸 선에 걸친 차 판정 기준)**: 지금 1.0cm, 동료 판 0.1cm. 동료가 왜 바꿨는지 묻고, 실제 차로 두 값을 비교한 뒤 정한다. occupancy_v2는 1.0 그대로다.
- 차량 추적 값(최대 속도, 카메라 사이 같은 차 판정 거리 2cm 등)은 동료가 정한 초기값이다. 실제 미니카로 조정해야 한다(TRACKING.md "실제 CCTV 추가 검증").
- 차량 위치는 상자의 바닥 접점이라 차 중심이 아니다. 두 카메라의 접점이 2cm 넘게 다르면 같은 차가 두 대로 나올 수 있다.
