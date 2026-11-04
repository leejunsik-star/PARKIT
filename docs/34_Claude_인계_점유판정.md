# 34. Claude 인계: 점유 판정 작업 (2026-11-04 기준)

다른 Claude(Claude Code) 세션이 점유 판정 작업을 이어받을 수 있게 정리한 문서다. 이 문서를 먼저 읽고, 필요한 부분만 아래 문서와 코드를 연다.
작성 시점: 2026-11-04 11시쯤. 그 뒤에 바뀐 것은 파일 날짜와 문서로 확인한다.

## 0. 먼저 지킬 것 (hun과 일하는 방식)

- **한국어**로 답하고, 코드 주석과 문서도 한국어로 쓴다. 결과 문서는 `docs\번호_제목.md`.
- **요청 전에는 코드를 고치지 않는다.** 확인·분석을 부탁받으면 결과와 제안만 낸다. 큰 변경은 계획을 먼저 보여 주고 승인("계획대로 진행")을 받는다.
- **원본은 남긴다.** 기존 파일을 크게 고칠 때는 사본을 두거나 새 이름·새 폴더로 만든다. 지금까지 쓴 방법:
  - 고치기 전 사본: `docs\도구\occupancy\이전\<날짜>_<설명>\`
  - 새 판은 새 파일/폴더: `calibrate_v2.py`, `occupancy_v2\`
  - hun이 "둘 다 고쳐" 같이 명시하면 원본을 고친다(예: 카메라 고르기 기능은 calibrate.py에 직접 넣음).
- `E:\ccapstione\first`(1학기 원본)는 읽기만. `dataset\data*`, `labels*`는 원본(복사해서 쓰기).
- 코드는 **Windows와 맥 모두** 돌게 쓴다. 맥은 웹캠을 메인 스레드에서 열어야 한다(parkit_video의 `MAIN_THREAD_OPEN`, `reopen_on_main`).
- 칸 번호, 매트 치수, 판정 숫자는 **설정 파일에만** 둔다(코드에 고정하지 않음).
- hun의 설정 파일(`config\cam_*.json` 등)은 hun이 직접 보정해서 저장한다. 시험할 때 덮어쓰지 않는다(경로를 임시 폴더로 돌려서 시험).

## 1. 폴더 지도

| 위치 | 내용 | 상태 |
|---|---|---|
| `docs\도구\occupancy` | **지금 주 판** 판정·보정 프로그램 | 계속 씀 |
| `docs\도구\occupancy_v2` | occupancy + 동료 웹 지도(차량 추적) 합친 판 (docs/33) | 새로 만듦(11-04) |
| `docs\도구\occupancy(동료버전)` | 동료 판 원본(11-01 판 + 웹 지도, 맥에서 만듦) | 읽기만 |
| `docs\도구\occupancy\이전\20261101_OpenCV창판` | tkinter 이전 OpenCV 창 판 | 백업 |
| `docs\도구\occupancy\이전\20261104_카메라선택전` | 카메라 고르기 넣기 전 calibrate.py, calibrate_v2.py, occupancy.py | 백업 |
| `docs\칸표_매트좌표_초안.json` | 칸 표 원본(56칸, 매트 cm). `occupancy\data\`에 복사본 | 네 줄 모두 실측(docs/30 5절) |
| `docs\도구\models` | 탐지 모델 M1~M8(M8이 기본), PARKIT Tiny(mymodel_*.pt + model.py, 동료 자체 모델) | |
| `dataset\handoff_점유판정_<날짜>` | 동료에게 보낸 묶음(make_handoff.py로 만듦). 11-01, 11-03 | |

관련 문서: 22(바닥 접점 방법), 25(기능 P/D/L/K 번호), 26(판정 규칙 P2·P3·P5), 27(첫 프롬프트), **28(사용법, 가장 자세함)**, 30(코드 확인·실측 치수), 31(ver2 프롬프트), **32(ver2 기준점 자동 찾기)**, **33(동료 웹 지도 합치기)**.

## 2. 작동 구조

```
calibrate.py  ─ 카메라마다 기준점 찍기 → 호모그래피(화면↔매트 cm) + 카메라 자세(solvePnP) + 감시 칸 + 고친 꼭짓점
              └→ config\cam_<이름>.json (+ 확인 사진)
occupancy.py  ─ 실행 설정(run.json 또는 시작 창에서 고른 카메라) → 카메라마다:
                 프레임 → N프레임마다 모델(M8 등)로 차 상자(P1)
                 → 상자 바닥 접점을 호모그래피로 매트 cm → 칸 판정(P2~P3), 가림(P5)   [parkit_judge]
                 → 칸마다 카메라 판정 합치기(P9) → 시간 거르기(P6) → 칸 상태      [parkit_state]
                 → 화면(tkinter OccApp), 기록(logs\<날짜_시각>\state_changes.csv, run_info.json)
```

- **칸별 화면 좌표는 저장하지 않는다.** occupancy가 시작할 때 칸 표(cm) × 호모그래피로 계산한다(`parkit_geom.CamGeom`). 그래서 칸 표를 고치면 다시 켜기만 하면 된다(기준점 매트 좌표가 바뀌면 다시 보정).
- **호모그래피** → 칸 위치, 접점의 칸 판정. **카메라 자세(렌즈 위치·화각)** → P2b 예상 접점(차 높이 때문에 상자 아래 끝 ≠ 바닥 접점), P5 가림 영역. 자세가 없으면 정면 근사.
- **카메라 2대**: 카메라마다 따로 판정 → 칸마다 `merge_obs`: 가리지 않은 카메라 결과만 쓰고 하나라도 "차 있음"이면 차 있음, 모두 가림이면 가림, 모두 끊김이면 DYNAMIC_BLIND(CAMERA_OFFLINE), 감시 칸이 아니면 STATIC_BLIND.
- **상태**: EMPTY, OCCUPIED, DYNAMIC_BLIND(INITIALIZING/OCCLUDED/CAMERA_OFFLINE/CAMERA_STOPPED), STATIC_BLIND(NO_COVERAGE). 기본 시간: 차 있음 3초, 빈 칸 5초.
- **시간**: 입력이 모두 영상 파일이면 영상 시각(결과가 늘 같음, C2), 웹캠이 있으면 실제 시각.
- **스레드**: Engine은 별도 스레드, tkinter와 웹캠 열기는 메인 스레드(`main_thread_work`).

## 3. 파일별 역할 (docs\도구\occupancy)

| 파일 | 역할 |
|---|---|
| `calibrate.py` | 보정 도구(tkinter). 단계 ① 기준점(IE-/IE+/IB-/IB+ = 칸 빈 공간 inner 네 모서리) ② 감시 칸 ③ 꼭짓점 고치기(inner 기준) ④ 보조선(교차점 SNAP). 키보드 미세 이동, 확대 창, 화각 추정(F), 해상도·웹캠 바꾸기, **시작 창 `CamPicker`**(웹캠 미리보기, 카메라마다 웹캠 번호, [웹캠 번호만 저장]), 창 안 [보정할 카메라] 바꾸기 |
| `calibrate_v2.py` | 보정 ver2: 손 점 4개 → Enter로 기준선(칠한 선) 자동 찾기 → 자동 기준점 더함. calibrate.py를 불러와 넓힘(`v1.CalibModel = CalibModelV2`로 바꿔 끼움). 저장 `config\v2\` |
| `parkit_autoref.py` | ver2 핵심(선 예상·단면·RANSAC 직선·교점·손 점 검사). 매트 전용 숫자 없음 |
| `make_ref_lines.py` | 칸 표 → `data\기준선_매트.json`(이 매트의 기준선 목록, 모양 숫자 MAT_SHAPE) |
| `occupancy.py` | 판정 프로그램. `Engine`(판정), `OccApp`(창), **`CameraStart`**(시작 창: 보정한 카메라 체크, 웹캠 번호·미리보기), [카메라 고르기…]로 다시 시작(main의 while 루프) |
| `parkit_config.py` | 실행 설정 읽기(`load_run`, 상대 경로는 설정 파일 기준, 후보 목록 중 있는 것), `Params`(칸별>줄별>params>기본) |
| `parkit_geom.py` | 칸 표(`SlotTable`), 호모그래피, solvePnP(여러 방법 중 고름), `CamGeom`, `build_camera_config`, `front_sign`(지평선 대비) |
| `parkit_judge.py` | 한 프레임 판정 P2~P5 |
| `parkit_state.py` | 상태 기계 P6~P10, `merge_obs`(P9) |
| `parkit_video.py` | 웹캠(`LiveSource`, `open_camera`: Windows DSHOW→MSMF→ANY), 영상(`FileSource`), 녹화 |
| `parkit_draw.py`, `parkit_tk.py` | 그리기, tkinter 공통(ImageView, ScrollFrame, HelpWindow, 한글 키 처리) |
| `parkit_synth.py`, `parkit_synth_mat.py` | 시험용 합성 화면(가상 카메라, 실제 매트처럼 그린 합성 매트) |
| `test_geometry.py`, `test_pipeline.py`, `test_autoref.py` | 합성 시험 (41 / 33 / 13개) |
| `make_handoff.py` | 동료에게 보낼 묶음(zip) 만들기 |

occupancy_v2에는 위에 더해 `vehicle_tracker.py`, `tracking_bridge.py`, `parking_api.py`(FastAPI), `web\`(React 지도, 빌드된 `web\dist`), `tracking_demo.py`, 동료 시험 3개, `TRACKING.md`가 있다.

## 4. 설정 파일

- `config\run.json`: 칸 표·모델 경로(후보 목록), conf 0.25, imgsz 640, analyze_every_frames 15, cameras(1대: cam_CAM-1.json), params(판정 숫자, 28 문서 5절), overrides(줄·칸별), presets(demo), log.
- `config\run_2대.json`: cameras 2대. `config\run_v2.json`: ver2 보정(`v2/cam_CAM-1.json`) + `ref_lines` + `autoref` 설정.
- `config\cam_<이름>.json`: name, source(웹캠 번호), calib_size, ref_points, homography, homography_err_cm, intrinsics(hfov), pose(rvec, tvec, cam_pos_cm, pitch, roll, reproj_px), monitored, edited_corners(+kind inner), expected_contact, screen_region, guide_lines. ver2 파일은 같은 형식 + `auto_ref` 키(occupancy는 읽지 않음).
- `config\webcams.json`: 카메라 이름 → 웹캠 번호(calibrate 시작 창이 씀, 보정 파일이 없는 카메라 번호도 기억). 판정에는 보정 파일의 `source`가 쓰인다.

## 5. 지금 상태 (2026-11-04 11시)

**환경 (hun의 Windows PC)**: Python 3.9.25(miniconda, `C:\Users\hun\miniconda3`), ultralytics 8.4.178, opencv 4.10.0, numpy 2.0.2, torch 2.8.0 CPU, Pillow 11.3, fastapi 0.128.8 / uvicorn 0.39.0 / websockets 15.0.1(11-04 설치), Node 24.16. 웹캠은 지금 1대(0번)만 연결. 동료는 맥, python.org Python 3.12.

**보정 파일 (docs\도구\occupancy\config)**
| 파일 | 내용 |
|---|---|
| `cam_CAM-1.json` | 09:13 hun 저장. 웹캠 0, 손 점 10개, 오차 0.24 / 0.81cm, 화각 48도, 감시 34칸. 카메라는 매트 42~56번 줄 쪽(남쪽), 렌즈 약 (35.5, 71.0, 높이 15.4cm) |
| `cam_CAM-2.json` | 10:35 저장. **시험용으로 보임**: 손 점 4개(오차 0 = 딱 맞춤), 웹캠 0(CAM-1과 같은 웹캠), 렌즈 높이 54cm(믿기 어려움). 실제 2대 연결 뒤 다시 보정해야 함 |
| `v2\cam_CAM-1.json` | 10:45 hun 저장(ver2). 손 점 6개, 자동 점 끔, 오차 0.15 / 0.29cm, 감시 27칸 |
| `webcams.json` | CAM-2 → 0 |

occupancy_v2\config는 11-04 복사 시점의 같은 파일이다(그 뒤로 따로 바뀔 수 있음).

**시험**: test_geometry 41/41, test_pipeline 33/33, test_autoref 13/13 통과. occupancy_v2는 여기에 동료 시험(추적 15, 통합 5, 재생 1) 통과.

## 6. 지금까지 한 일 (순서대로, 요약)

1. 점유 판정 프로그램 처음 작성(docs/22·25·26·27 기준) → OpenCV 창 판 → **tkinter 판**으로 다시 만듦.
2. 보정 도구 개선: IE/IB만 쓰기, 키보드 미세 이동(분홍 겹원), 끄는 동안 재계산 안 함, 칸 선 숨기기, 확대 창, 보조선과 교차점 SNAP, 해상도·웹캠 바꾸기, 화각 추정, 모두 지우고 새로 보정, ③ 꼭짓점을 inner 기준으로, 지평선이 보이는 카메라 방향 버그(front_sign), solvePnP 여러 방법.
3. 판정 프로그램: 분석 간격 N프레임, 모든 프레임/분석한 프레임만 보기, 모델·신뢰도 고르기(YOLO + PARKIT Tiny), 웹캠·해상도 바꾸기, 맥 호환, requirements, make_handoff.
4. 칸 표 실측 반영 확인(다른 세션, docs/30).
5. **보정 ver2**(docs/31·32): 기준점 자동 찾기. 합성 시험 13/13. 실제 사진 비교: 칸 위치·판정은 v1과 같음(칸 중심 차이 평균 0.01~0.04cm, 같은 5칸 점유).
6. **카메라 고르기**: calibrate 시작 창(웹캠 미리보기, 카메라마다 웹캠 번호, 번호 저장), 보정 창 안 카메라 바꾸기, occupancy 시작 창(쓸 카메라 체크)과 [카메라 고르기…].
7. **동료 웹 지도 합치기 → occupancy_v2**(docs/33): 연결 코드 약 10곳, Python 3.9 고침, 시작 창 [웹 지도 켜기], 판정 창 [웹 지도 열기], 웹 라이브러리 설치.

## 7. 남은 일·알려진 문제

**hun이 정할 것**
- `straddle_min_cm`: 지금 1.0, 동료 판 0.1. 동료에게 이유를 묻고 실제 차로 비교(occupancy_v2도 1.0).
- 칸 표 고치기(hun 실측, 11-04): **U자 폭 0.9 → 1.0cm**, **28~41번 줄 깊이 6.6 → 6.7cm**(입구·안쪽 중 어느 쪽인지 모름). 아직 반영 안 함. 칸 표 원본은 docs\칸표_매트좌표_초안.json(복사본 data\ 도 맞춰야 함). 바꾸면 `make_ref_lines.py`로 기준선 목록도 다시 만든다.

**확인이 필요한 것**
- **줄마다 깊이가 0.2~0.5cm 어긋남**: 실제 사진에서 호모그래피 하나로 모든 줄이 맞지 않는다(1~14번 줄 입구가 실제 U자 끝보다 약 0.4cm 통로 쪽). v1 손 점도 같은 현상. hun은 각 줄의 깊이(U자 바깥 끝~카스토퍼)가 6.7cm로 같다고 실측함 → 남은 원인 후보는 **줄 사이 거리(통로·분리대 폭)** 또는 **매트가 평평하지 않음**. 매트 위쪽 끝에서 각 줄 카스토퍼 칸 쪽 면까지 거리(칸 표: 2.8 / 26.4 / 31.0 / 52.7)를 재 달라고 요청한 상태(답 없음).
- 화각: 화각 자동 추정(F)을 하면 48 → 약 45.3도, 재투영 10 → 5px, 렌즈 y 약 2cm 바뀜. hun 보정에서 F를 눌러 두길 권함(P2b·P5에 영향).
- CAM-1 손 점 `10-IB+`가 0.8cm 어긋남(잘못 찍은 것 같음).
- ver2 실제 사진 비교는 occupancy.py 화면 저장(칸 선이 그려진 사진)을 메워서 했다. 그림 없는 원본 사진으로 다시 하는 것이 좋음.
- ver2: 1~14번 줄은 칸 번호 글자가 U자 끝 바깥에 있어 입구선(tip_row)이 글자를 잡을 수 있음.
- **웹캠 2대 실제 연결은 아직 시험 못 함**, 맥에서도 최근 기능(카메라 고르기, occupancy_v2) 미확인.
- 차량 추적 값(최대 속도, 카메라 사이 같은 차 판정 거리 2cm 등)은 동료 초기값. 실제 미니카로 조정 필요(TRACKING.md "실제 CCTV 추가 검증").

**하지 않기로 한 것**
- ver2(calibrate_v2)로 2대 실행 설정(`run_v2_2대.json`)은 만들지 않음(hun: "ver2까지 적용할 필요는 없어").
- "칸 표가 바뀌면 알리기" 기능은 보류(hun: "지금은 두기").

## 8. 시험과 실행 명령

`docs\도구\occupancy`(또는 `occupancy_v2`)에서:

```bash
python test_geometry.py
python test_pipeline.py
python test_autoref.py
python calibrate.py
python occupancy.py
```

- 모든 시험은 마지막 줄 "합계: N / N 통과". test_pipeline은 1~2분, test_autoref는 5분쯤 걸리니 백그라운드로 돌린다.
- occupancy_v2 웹: `python test_vehicle_tracker.py`, `python test_tracking_integration.py`, `python test_tracking_replay.py`, `python tracking_demo.py`(브라우저 `http://127.0.0.1:8765`), `python occupancy.py --web`.
- 창 없이 판정: `python occupancy.py --run config/run.json --source CAM-1=<영상.mp4> --no-window --speed 0`.

## 9. 작업할 때 겪은 함정

- **한글 출력**: Windows 콘솔은 cp949다. 파이썬을 돌릴 때 `PYTHONIOENCODING=utf-8`을 주고, 하위 프로세스 출력은 `encoding='utf-8'`로 읽는다. 한글 경로 이미지는 `cv2.imread` 대신 `cv2.imdecode(np.fromfile(...))`.
- **Bash heredoc 안의 역슬래시·따옴표**가 깨지기 쉽다. 긴 파이썬 수정은 스크립트 파일로 써서 실행한다.
- **웹캠은 한 프로그램만 잡는다.** calibrate나 occupancy 창이 열려 있으면 다른 쪽이 못 연다. 닫은 직후에는 풀리는 데 조금 걸려서 코드에 0.3~1초 기다림과 다시 시도가 있다.
- **tkinter를 스크립트로 시험할 때** `root.update()`만 부르면 40ms 주기 갱신이 끝나지 않아 멈춘다. `after` 예약을 지우고 `tick()`을 직접 부르거나, mainloop 안에서 `after`로 시험 동작을 예약한다.
- 시험 중 실행한 프로그램을 끝낼 때 **모든 python.exe를 죽이지 않는다**(hun의 프로그램이 같이 꺼짐). 띄운 프로세스 번호로만 끈다.
- 시험이 `config\` 에 쓰지 않게 경로(`cam_config_path`, `WEBCAM_MAP`, `CONFIG_DIR`)를 임시 폴더로 돌린다.
- 맥에서 웹캠은 메인 스레드에서만 연다. 새 화면에서 웹캠을 열 때도 메인 스레드에서(시작 창 미리보기도 메인 스레드).
- Python 3.9라서 `int | None`, `match` 같은 3.10 문법을 쓰면 안 된다(동료 코드에서 실제로 걸림).
