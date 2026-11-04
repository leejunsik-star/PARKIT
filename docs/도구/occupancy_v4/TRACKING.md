# PARKIT 2D 차량 추적 확장

구현/검증: 2026-11-03. 실제 CCTV 및 실차 탐지 성능은 아직 검증하지 않았다.
기존 점유 판정은 그대로 두고, 새 기능은 `--tracking` 또는 `--web`으로 켠다.

## 먼저 확인한 기존 구조

- `occupancy.py`: `Engine`이 입력을 열고 `step_live()` / `step_files()` → `analyze()` → `finish_tick()` 순으로 실행한다. `analyze()`는 모델 추론 → `judge_frame()` → `Cam.res` 저장 → `Logs.frame()` 기록을 한다. `finish_tick()`는 카메라 상태를 판단하고 기존 `Tracker.update()`를 호출한다.
- `parkit_judge.py`: `judge_frame()`은 `FrameResult`를 반환한다. `boxes`, 주차면별 `obs/why/cover`를 포함한다. 각 `BoxResult`의 `contact_px`는 상자 아래 변 중앙, `contact_mat`는 **전체 H**로 변환한 매트 cm 좌표다. `slot_mat`는 칸별 H 좌표로 서로 다르다. `slot`과 `kind`는 점유 판정 결과이지 차량 ID가 아니다.
- `parkit_geom.py`: `SlotTable`의 `mat_size`, `Slot.cell/inner/no`를 웹 지도로 재사용한다. `CamGeom`은 해상도를 보정하고 전체 H 및 칸별 H를 제공한다. `apply_h()`는 픽셀→매트 변환이다. 매트 원점은 왼쪽 위, x 오른쪽/y 아래, cm 단위다.
- `calibrate.py`: `CalibModel.recompute()` → `build_camera_config()` → `CamGeom`; `save()`는 `cam_*.json`과 확인 이미지를 저장한다. 기준점, 감시 칸, 수정한 칸 꼭짓점, 카메라 자세를 그대로 사용한다. 이 파일은 수정하지 않았다.
- `parkit_state.py`: `Tracker`는 **주차면 상태 전용**이다. `merge_obs()`가 카메라별 PRESENT/ABSENT/OCCLUDED/OFFLINE을 합치고 시간 조건을 적용한다. `on_change()` 콜백은 상태 변경을 로그/리스너로 보낸다. 차량 ID 추적에는 재사용하지 않았다.
- `parkit_video.py`: `LiveSource.latest()`는 `(frame, 수신 wall time, frame 번호)`를 반환한다. `FileSource.get()`은 `(영상 시각, frame 번호, frame)`을 반환한다. 웹캠을 여는 동작은 macOS 메인 스레드에서 수행한다. 프레임 수신은 카메라별 스레드, 추론은 기존 Engine 스레드다.
- `parkit_draw.py`: `PlanView`는 `SlotTable`을 사용해 OpenCV 평면도를 그린다. 기존 `OccApp.plan_points()`는 `contact_mat` 접점과 색만 반환하며 ID는 없다. 관리자 화면의 이 동작은 그대로 유지했다.
- 모델: `make_detector()` → `parkit_checkpoint()`로 자체 체크포인트를 확인하고 `TinyDetector` 또는 YOLO `ModelDetector`를 선택한다. Tiny의 기존 `model.py` 검색/전처리, GUI 모델 변경, 신뢰도 변경은 변경하지 않았다.
- 설정: `load_run()`은 기본값과 run JSON을 병합한다. 상대 경로는 JSON 기준이며, model/slot_table 후보 목록의 첫 존재 경로를 사용한다. cameras에는 config/source/width/height 등을 지정한다. `params`, `overrides`, `presets`, `display`, `log`는 유지한다.
- tkinter 위젯은 메인 스레드에서만 다룬다. 신규 API는 tkinter나 Engine 필드에 직접 접근하지 않고 독립 상태 저장소만 읽는다.
- 구현 전 테스트: `test_geometry.py` **35/35**, `test_pipeline.py` **33/33** 통과. 기존 로그를 지우지 않도록 테스트 출력 경로만 별도 디렉터리로 지정했다.

## 수정/추가 파일

기존 실행 코드 수정은 **occupancy.py 한 파일**이다. 문서 README에는 안내 링크만 추가했다.

| 파일 | 역할 |
|---|---|
| `occupancy.py` (수정) | 선택적 확장 초기화, 분석 결과/관측 시각 전달, tick 후 상태 발행, API 시작/종료, CLI 옵션 추가 |
| `vehicle_tracker.py` (신규) | 전역 ID, 속도/시간 게이트, 예측 위치와 거리 비용, Hungarian 일대일 할당, 불확실 매칭 보류, 후보/누락/종료, 좌표 평활화 |
| `tracking_bridge.py` (신규) | 기존 BoxResult 읽기, 관측 순서 정렬, SlotTable 레이아웃 직렬화, 최신 상태 저장소 |
| `parking_api.py` (신규) | FastAPI HTTP/WS, 클라이언트별 송신 제한, 선택적 React 빌드 정적 제공 |
| `tracking_demo.py` (신규) | 실제 모델/카메라 없이 합성 관측으로 지도 확인. 점유 색도 합성 예시 |
| `tracking_config.example.json` (신규) | 추적 파라미터 예시; 기존 run/cam JSON과 별도 |
| `requirements-web.txt` (신규) | FastAPI/uvicorn/websockets 의존성; 탐지 패키지는 업그레이드하지 않음 |
| `web/src/main.jsx` (신규) | React + SVG 지도, 주차면 상태, 차량 선택, North-up Follow, 연결/추적 상태 |
| `web/src/tracking.js` (신규) | 위치 보간, 유효 시간 판정, Follow viewBox, WebSocket 재연결/무응답 감시 |
| `web/src/style.css` (신규) | 데스크톱/좁은 화면 배치, 상태 색, 차량 아이콘 |
| `web/package.json`, `package-lock.json`, `vite.config.js`, `index.html`, `.gitignore` (신규) | 웹 설치/빌드/개발 프록시 및 생성물 제외 |
| `test_vehicle_tracker.py` (신규) | 차량 이동/교차/다중 카메라/누락/시간/할당 테스트 |
| `test_tracking_integration.py` (신규) | Engine·기하·주차면·차량 통합, API/WS, 상태 저장소, 느린 클라이언트 테스트 |
| `test_tracking_replay.py` (신규) | 같은 합성 영상으로 추적 off/on/20배속의 기존 점유 판정 결과 비교 |
| `web/src/tracking.test.js` (신규) | 보간·Follow·선택 ID 소실·재연결·무응답 단위 테스트 |
| `gui_smoke_tracking.py` (신규) | GT 합성 영상으로 tkinter 창을 열고 5초 후 확인/종료 |
| `tracking_reports/` (신규) | 실제 시험 결과, 변경 전/후 해시, 브라우저 검증 기록/화면 |
| `TRACKING.md` (신규) | 이 문서 |

## 데이터 흐름

```text
기존 LiveSource / FileSource
  → 기존 모델 (YOLO 또는 PARKIT Tiny)
  → 기존 judge_frame() → FrameResult
       ├─ 기존 주차면 obs → 기존 Tracker → 기존 로그/tkinter
       └─ boxes.contact_mat + camera_id + 프레임 관측 시각
            → TrackingBridge.pending (한 tick 동안만)
            → VehicleTracker (시간순, 카메라별 한 프레임 한 번만)
            → 주차면 상태의 복사본과 합쳐 StateStore.publish()
            → FastAPI 최신 스냅샷 → WebSocket → React → SVG
```

StateStore는 누적 큐가 아닌 **최신 스냅샷 하나**다. 생성자는 네트워크를 호출하지 않고 잠깐 잠금 안에서 참조만 교체한다. 읽는 쪽은 참조를 받은 뒤 잠금 밖에서 복사/전송한다. 느린 웹 클라이언트는 중간 스냅샷을 건너뛰며, 송신이 2초 넘게 막히면 연결을 정리한다. 새 접속에는 전체 상태를 즉시 보내며 이후 최대 5Hz로 heartbeat를 보낸다.

### 좌표/시간/ID의 의미

- 출력 위치는 **상자 접점의 매트 좌표**이며 차체 중심이 아니다. 아이콘은 그 점을 중심으로 그린 근사 표식이다.
- 아래가 잘렸거나 화면 경계에 닿은 상자, 매트 밖 좌표, NaN 등은 차량 좌표 갱신에 사용하지 않는다. 기존 점유 알고리즘은 그대로 그 상자를 처리한다.
- live는 `LiveSource`가 기록한 호스트 수신 시각을 Engine 시작 기준 초로 변환한다. RTSP 센서 노출 시각/장치간 하드웨어 동기화를 구현한 것은 아니다.
- 파일 입력만 있으면 영상 시각을 사용한다. 두 녹화 파일의 시작점을 맞춰야 한다. 현재 임의의 파일 시간 오프셋 보정 UI는 없다.
- 파일/웹캠 혼합은 기존 Engine 규칙대로 파일 시각을 Engine 경과 시각에 대응한다. 차량 추적 실측은 동기화된 입력을 권장한다.
- 관측마다 camera_id/observed_at를 보존한다. 서로 다른 시각의 좌표를 동시 관측처럼 평균내지 않는다. 정확히 같은 시각의 허용된 중복만 평균내며, 속도는 같은 카메라의 연속 관측으로 계산한다.
- timestamp TTL을 넘긴 입력, 이미 처리한 프레임, 전역 처리 순서를 거슬러 도착한 관측은 버린다. 추론 지연 후 새 프레임이 과거 발행 시각보다 앞서 있다는 이유만으로 버리지는 않는다.
- ID는 `세션 UUID-번호`다. 새 실행에서는 새 세션이 생겨 이전 선택 ID를 다른 차에 재사용하지 않는다. 아이콘에는 번호, 선택 패널에는 전체 ID를 표시한다.
- 같은 차량으로 보이는 관측을 기하 조건으로 **추정 연결**한다. 외형/번호판 확인은 없으므로 물리적 동일성을 보증하지 않는다.
- 실제 차체 방향은 `heading: null`, 이동 방향도 첫 버전에서는 `motion_heading: null`, `heading_source: unknown`이다. 아이콘은 항상 지도 위쪽을 향한다.

### 매칭과 추적 상태

1. 프레임 시각 차이와 최대 속도로 이동 가능한 범위를 검사한다.
2. 최근 속도로 예측한 위치와 새 관측의 거리로 비용을 만든다. 예측은 매칭용이며 누락 차량을 화면에서 계속 전진시키지 않는다.
3. 새 카메라 연결은 더 작은 거리와 시간 범위를 적용한다. 두 후보의 비용 차이가 작으면 연결을 보류하고 UNCERTAIN을 표시한다.
4. 허용된 비용에 Hungarian 일대일 할당을 적용한다. 한 프레임에서 하나의 차량 ID에 여러 탐지를 붙이지 않는다.
5. 새 관측은 CANDIDATE부터 시작한다. 서로 충분히 떨어진 시각에 3번 확인하면 TRACKED로 전환한다. 동일 시각의 두 카메라 관측을 확인 횟수 2회로 세지 않는다.
6. 1.2초 관측이 없으면 LOST, 4초가 지나면 REMOVED. 후보는 1.5초 동안 재확인이 없으면 종료한다. REMOVED는 2초 더 보내고 삭제하며 같은 ID를 재발급하지 않는다.
7. 선택 ID가 사라져도 다른 차를 자동 선택하지 않는다. 마지막 중심을 유지하면서 재선택 안내를 표시한다.

기본 수치는 미니카 실측으로 조정할 초기값이다. 접점 차이가 2cm를 넘는 두 카메라는 같은 차량을 중복 ID로 표시할 수 있다. 이를 없애려고 문턱을 크게 늘리면 가까운 별개 차량을 합칠 위험이 있다.

## 설치 / 실행 (occupancy_v2 → v4, 2026-11-04 다시 씀)

> 동료가 쓴 원래 명령은 맥과 `runs/occupancy/` 폴더 기준이었다. 이 절은 이 저장소의 `docs\도구\occupancy_v2` 폴더와 Windows 기준으로 다시 썼다. 맥은 `python` 대신 `.venv-gui/bin/python`(Python 3.12)을 쓰면 된다.
> 원래 판(맥, 11-01 판정 프로그램 위에 만든 것)은 `docs\도구\occupancy(동료버전)`에 그대로 있다.

명령은 모두 `docs\도구\occupancy_v4` 폴더에서 실행한다.

### 처음 한 번

```bash
python -m pip install -r requirements-web.txt
```

- 판정 라이브러리(`requirements.txt`: ultralytics, opencv, numpy)는 이미 있어야 한다. 웹 지도는 fastapi, uvicorn, websockets만 더 쓴다.
- **Python 3.9에서도 된다.** 동료 판은 3.10 이상 문법(`int | None`)이 한 줄 있어 3.9에서 안 열렸는데, `Optional[int]`로 고쳤다. `requirements-web.txt`도 버전을 "이상"으로 바꿔서, 3.9에서는 pip이 fastapi 0.128.8, uvicorn 0.39.0, websockets 15.0.1을 고른다.
- 웹 화면은 빌드된 `web/dist`가 들어 있어서 **Node는 필요 없다.** 화면 코드(`web/src`)를 고칠 때만 Node 22.12 이상으로 다시 빌드한다(아래 "웹 화면 개발").

### 실행

```bash
python occupancy.py
```

- 시작 창에서 카메라를 고르고 **[웹 지도 켜기]**에 체크한 뒤 [판정 시작]. 웹 라이브러리가 없으면 체크가 꺼져 있고 설치 명령이 빨간 글씨로 나온다.
- 판정 창의 **[웹 지도 열기]** 버튼을 누르거나 브라우저에서 `http://127.0.0.1:8765`를 연다. 웹 서버는 모델을 불러온 뒤 켜진다.
- 판정 창의 [카메라 고르기…]로 다시 시작해도 웹 서버는 닫았다가 다시 연다(같은 주소).

명령으로 직접 줄 수도 있다.

```bash
python occupancy.py --web
python occupancy.py --web --run config/run_2대.json
python occupancy.py --web --no-window
python occupancy.py --tracking
python occupancy.py --web --tracking-config tracking_config.example.json
python occupancy.py --web --web-host 0.0.0.0
```

- `--web-host 0.0.0.0`: 같은 와이파이의 휴대폰·다른 PC에서 `http://<이 PC 주소>:8765`로 볼 수 있다. 인증이 없으니 믿을 수 있는 망에서만 쓴다. Windows 방화벽이 처음에 허용을 물을 수 있다.
- `--web`도 `--tracking`도 없으면 웹 라이브러리를 불러오지 않고 포트도 열지 않는다(기존 판정과 같다).
- 카메라 2대는 먼저 `python calibrate.py`로 CAM-2를 보정해 `config\cam_CAM-2.json`을 만든다.

### 카메라 없는 합성 데모

```bash
python tracking_demo.py
```

브라우저 `http://127.0.0.1:8765` → 회색 차량 선택 → **내 차량 따라가기**. 실제 카메라·모델은 열지 않는다. `Ctrl+C`로 끝낸다. **판정 프로그램과 데모는 같은 포트를 동시에 쓸 수 없다.**

### 웹 화면 개발 (Node 22.12 이상)

```bash
npm ci --prefix web
npm run build --prefix web
```

`occupancy.py --web`을 켠 상태에서 `npm run dev --prefix web`으로 `http://127.0.0.1:5173`을 연다. Vite가 `/api`와 `/ws`를 8765로 넘긴다.

## API 요약

- `GET /api/layout`: units, coordinate_system, mat.width_cm/height_cm, slots[].no/row/cell/inner.
- `GET /api/state`: session_id, sequence, mode, vehicles, slots, cameras, updated_at, server_time, publication_age_s, tracking_time, clock, engine_status, tracking_config.
- `WS /ws/parking`: 같은 전체 상태 JSON. 새 연결 즉시 전송하고 heartbeat 갱신.
- 차량: id, x_cm/y_cm(평활화), observed_x_cm/y_cm, status, last_seen/age_s, observations(camera_id/observed_at/valid), uncertain/uncertainty_cm, heading, slot/kind.
- 주차면 상태는 기존 `EMPTY / OCCUPIED / DYNAMIC_BLIND / STATIC_BLIND`를 변경 없이 표시한다.
- WS가 살아 있어도 producer의 publication_age_s가 2초를 넘으면 위치/점유 표시를 오래된 것으로 표시한다. 연결이 끊기면 마지막 위치만 남고 자동 재연결한다. 5초 무응답도 연결을 다시 시도한다.
- UI는 약 350ms 구간 안에서 이전 표시 위치→새 위치를 보간하며 미래 좌표로 외삽하지 않는다. 서버 좌표에도 시간 기반 지수 평활화를 적용하므로 소량 지연이 있다.

## 시험 명령 및 실제 결과

```bash
python test_geometry.py
python test_pipeline.py
python test_vehicle_tracker.py
python test_tracking_integration.py
python test_tracking_replay.py
npm test --prefix web
python gui_smoke_tracking.py
```

`test_tracking_integration.py`는 `httpx`도 필요하다(시험할 때만).

아래 표는 동료가 맥에서 원래 판으로 낸 결과다. occupancy_v2에서 다시 낸 결과는 docs/33 문서에 있다.

`test_pipeline.py` 자체는 기존 `logs/test_pipeline`을 다시 생성한다. 이번 실행에서는 기존 로그를 보존하려고 모듈의 OUT을 `logs/tracking_baseline`, `logs/tracking_regression`으로 지정했다. GUI 시험에는 `--run runs/occupancy/logs/tracking_baseline/run_test.json`을 전달했다.

| 검증 | 결과 |
|---|---|
| 기존 기하 테스트 (구현 전/후) | 35/35 통과 |
| 기존 전체 파이프라인 (구현 전/후) | 33/33 통과 |
| 차량 추적 단위 테스트 | 15개 통과 |
| Engine / 저장소 / HTTP / WebSocket 통합 | 5개 통과 |
| 추적 off/on/20배속 영상 회귀 | 상태 변경 107행 동일 (시스템 wall 기록만 제외) |
| 프런트엔드 상태/재연결 테스트 | 5개 통과 |
| React production build | 성공 |
| tkinter 창 실행 (GT 합성 영상) | ready=True, error=None, window=True, 차량 4개 확인 |
| 실제 브라우저 합성 데모 | 56개 칸, 차량 선택, Follow 중심 좌표, 서버 중단 표시, 자동 재연결, 서버 세션 변경 시 기존 ID 재선택 안내 확인 |

통합 테스트에는 `httpx`가 추가로 필요하다(현재 환경에 설치되어 있음). 이번 설치된 Starlette는 httpx TestClient의 향후 변경 경고를 내지만 테스트는 통과했다. 런타임에는 httpx 테스트 클라이언트가 필요하지 않다.

## 기존 기능 영향 / 남은 한계

- 점유 판정, Homography, 기존 Tracker, 모델 선택, draw, video, config와 기존 테스트 파일의 내용은 바꾸지 않았다. `tracking_reports/unchanged_files.json`에서 해시 비교를 확인할 수 있다.
- 기존 CLI에 옵션만 추가했다. 웹 확장을 끄면 기존 동작을 유지한다. 켜면 차량 수에 따른 할당/직렬화 연산은 추가되지만 네트워크 대기 때문에 추론을 막지는 않는다. 실제 FPS 영향은 아직 실측하지 않았다.
- 기하 정보만으로 완전한 가림/겹침 뒤 동일 차량임을 증명할 수 없다. 일부 교차 시나리오는 통과했지만 모든 교차에서 ID 유지 보장은 없다. 불확실하면 보류하거나 새 ID로 시작한다.
- 비슷한 위치에 나타난 다른 차량의 오연결 가능성은 완전히 제거할 수 없다. 모델 오탐이 지속되면 후보가 확정될 수 있다.
- 관측 한 개를 두 카메라에서 각각 새 ID로 이미 만들었고 나중에 합쳐야 하는 경우 기존 ID끼리의 소급 병합은 하지 않는다. 사용자 선택 ID를 임의로 교체하지 않기 위해서다.
- 실제 차체 중심/방향, Re-ID, 경로 계획, 영구 추적 이력, Firebase/DB, 3D, 공개 서비스 인증은 구현 범위 밖이다.
- 기본 15프레임 간격과 모델 지연이 빠른 차량에 충분하지 않을 수 있다. 추적 게이트와 분석 간격은 실제 영상으로 조절해야 한다.

## 실제 CCTV 추가 검증

1. CAM-1/2를 고정하고 같은 매트 점으로 각각 보정, 해상도/화면 비율 확인.
2. 같은 정지 차량에 대한 두 contact_mat의 차이를 cm로 측정(카메라 각도 때문에 접점이 서로 다를 수 있음).
3. 차량 1대의 정지·출발·최대 속도·시야 이동에서 ID 유지 및 위치 오차 측정.
4. 가까운 차량 2~3대의 교차/나란한 이동/부분 가림에서 ID 교환·중복·미탐 기록.
5. 실제 모델 오탐이 후보를 확정하는지 확인하고 탐지 신뢰도/후처리는 별도 평가.
6. 카메라 지연·RTSP 버퍼·분석 주기와 관측 TTL 관계, 연결 단절/재연결 확인.
7. 차량 선택 후 잠시 가림/완전 이탈/프로그램 재시작 시 자동으로 다른 차를 선택하지 않는지 확인.
8. 웹 켜기 전후 CPU·메모리·추론 FPS, 여러 클라이언트, Windows와 최종 Mac에서 실측.

참고한 공식 문서: [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/), [React useEffect](https://react.dev/reference/react/useEffect), [Vite](https://vite.dev/guide/).

## 2026-11-04 이동 방향 아이콘 회전

같은 카메라의 연속 관측에서 0.5cm 이상 이동하고 속도가 1cm/s 이상이면
이동 방향을 계산한다. `motion_heading`은 지도 위쪽 0°, 오른쪽 90°의 시계 방향 각도다.
처음에는 null이며, 정지 또는 기준 이하의 흔들림에서는 마지막 방향을 유지한다.
카메라 간 접점 차이만으로 방향을 계산하지 않는다. 방향 판정에 쓰는 연속 관측 간격은
기존 속도 계산과 동일한 0.02~1.2초다. 매우 느린 이동은 방향 갱신이 지연될 수 있다.
`heading`(차체 방향)은 계속 null이고 `heading_source`는 방향 추정 후 motion이다.
웹은 최단 각도로 350ms 보간하여 차체 그림만 회전한다. 차량 ID 글자와 지도는 회전하지 않는다.
후진 시 실제 차체 앞쪽과 다를 수 있다. 백엔드 재시작과 웹 빌드 후 브라우저 새로고침이 필요하다.

## 2026-11-04 통로 차 접점 반 폭 보정 (occupancy_v4, docs/36)

반대쪽 카메라 두 대가 같은 통로 차를 보면, 상자 아래 끝(접점)은 각 카메라에 가까운 옆면이라
두 점이 차 폭(3.2cm)만큼 떨어진다. 다른 카메라 점은 `cross_camera_radius_cm`(2cm) 안에서만 같은 차로 이어서
한 차가 두 대로 나왔다. v4는 추적기에 넣기 전에 **통로 차(kind `aisle`) 접점만** 카메라에서 먼 쪽
(`CamGeom.far_dir`, 카메라 자세로 계산)으로 `aisle_contact_shift_cm`만큼 민다.

- 기본값: 판정 숫자 `car_width_cm`의 반(지금 1.6cm). `run.json`의 `"vehicle_tracking": {"aisle_contact_shift_cm": 1.6}`
  또는 `--tracking-config` 파일로 바꾼다. 0이면 보정하지 않는다(v3와 같음).
- 칸에 선 차(`slot`, `unmonitored`)의 접점은 **그대로** 추적기와 웹 지도에 보낸다(숨기지 않음).
  `slot_cut`, `uncertain`, 가장자리 상자는 이전처럼 좌표 갱신에 쓰지 않는다.
- 칸 판정(P2~P9)은 바꾸지 않았다. 판정 창에 그리는 접점도 보정 전 자리 그대로다.
- 합성 카메라(북쪽·남쪽 정면) 시험: 통로를 따라 달리는 차의 두 카메라 접점 거리 3.2cm → 0.3cm 안, 한 대로 이어짐
  (`python test_aisle_contact_shift.py`).

## 2026-11-04 웹 지도 "내 차만 보기" (occupancy_v4, docs/36 4절)

내 차를 고른 뒤 MY VEHICLE 칸의 [내 차만 보기]를 누르면 지도에 내 차만 그린다([다른 차량도 보기]로 되돌림).
칸 색은 그대로 보인다. 내 차가 목록에서 사라지면(추적 끝) 다시 고를 수 있게 모든 차를 보여 주고 안내를 띄운다.
[선택 해제]를 누르면 옵션도 꺼진다. 규칙은 `web/src/tracking.js`의 `visibleVehicles`에 있고 웹 시험으로 확인한다.
