# 35. 동료 수정(이동 방향 아이콘) 합치기 → occupancy_v3 (2026-11-04)

동료가 고친 occupancy 폴더(저장소 맨 위 `occupancy\`에 압축을 풀어 둠)를 확인하고, `docs\도구\occupancy_v3`로 합쳤다.
`occupancy`, `occupancy_v2`, `occupancy(동료버전)`, 동료가 올린 `occupancy\`는 하나도 고치지 않았다.

## 1. 동료가 올린 폴더 확인

- **바탕**: 11-01 판에 웹 지도를 더한 동료 판이다. `docs\도구\occupancy(동료버전)`과 같은 계열이라 ver2 파일(`calibrate_v2.py`, `parkit_autoref.py` 등)은 없다.
- **한글 이름**: 맥 압축이라 한글 파일 이름이 자모로 나뉘어 있다(NFD). 예: `28_점유판정_프로그램_사용법.md`, `data\칸표_매트좌표_초안.json`, `config\run_2대.json`. 화면에는 같은 글자로 보여도 Windows 프로그램에서는 다른 이름으로 다룰 수 있다. 안의 내용은 이전과 같다.
- **줄 끝**: 대부분의 파일이 CRLF로 바뀌어 있다. 줄 끝을 빼고 비교하면 calibrate.py, occupancy.py, parkit_*.py, 시험 파일, README.md, run.json, run_2대.json, 칸 표는 `occupancy(동료버전)`과 **내용이 같다**.
- **실제로 바뀐 것**(11-04 02:19, 기능 1개): **웹 지도 차량 아이콘을 이동 방향으로 회전**

| 파일 | 바뀐 내용 |
|---|---|
| `vehicle_tracker.py` | 설정 `heading_min_displacement_cm`(0.5), `heading_min_speed_cm_s`(1.0) 추가. 같은 카메라의 연속 관측에서 기준 이상 움직이면 `motion_heading`(지도 위쪽 0°, 시계 방향) 계산. 기준보다 덜 움직이면(정지 포함) 마지막 방향 유지. 카메라 사이의 접점 차이로는 방향을 정하지 않음. 스냅숏에 `motion_heading`, `heading_source`("motion"/"unknown") 넣음 |
| `test_vehicle_tracker.py` | 시험 2개 추가(상하좌우·대각선·정지, 카메라 바뀜은 방향 아님) |
| `web\src\tracking.js` | `interpolateHeading`(최단 각도), `Positions.headingAt` 추가. 350ms 보간 |
| `web\src\main.jsx` | 차 그림만 `rotate()`로 돌림(ID 글자는 안 돌림), 안내 문구 바꿈 |
| `web\src\tracking.test.js` | 웹 시험 1개 추가 |
| `web\dist` | 위 src로 새로 빌드한 결과(`index-BpaY-uGs.js`) |
| `TRACKING.md` | "2026-11-04 이동 방향 아이콘 회전" 절 추가 |

- **보정 파일**(`config\cam_CAM-1.json`, `cam_CAM-2.json`): 동료 쪽 시험 보정으로 보인다. 1280×720, 기준점 35/33개, 오차가 평균 약 16cm·최대 약 34cm라 **쓰지 않았다**.
- **그 밖**: `logs\`(동료 실행 기록, 시험용 mp4), `tracking_reports\`(동료 시험 기록), `web\node_modules`(맥에서 설치)가 들어 있다. 코드에는 영향이 없어 가져오지 않았다.

## 2. 합친 방법

1. `occupancy_v2`를 통째로 `occupancy_v3`로 복사했다. `__pycache__`, `node_modules`, `logs\` 안의 실행 기록(2026…)은 뺐다.
   - 설정(`config\`)은 v2에 있던 **hun의 11-04 18시 2대 보정**을 그대로 가져왔다(CAM-1 웹캠 0, CAM-2 웹캠 1, `straddle_min_cm` 0.1).
2. 동료가 바꾼 파일 5개(`vehicle_tracker.py`, `test_vehicle_tracker.py`, `web\src`의 3개)를 덮어썼다. v2의 이 파일들은 `vehicle_tracker.py`의 한 줄을 빼면 동료 이전 판과 같아서 충돌이 없다.
3. `web\dist`는 동료가 새로 빌드한 것으로 바꿨다. 이 PC에는 Node가 없어 다시 빌드하지 못했다. 빌드 결과에 새 기능(`headingAt`, 안내 문구)이 들어 있는 것은 확인했다.
4. **Python 3.9 호환 고침**: 동료 새 판에 `slot: int | None`, `motion_heading: float | None`이 다시 들어가 있었다. dataclass라 hun의 PC(3.9)에서는 import할 때 바로 오류가 난다. 그래서 `Optional[int]`, `Optional[float]`로 바꿨다(v2에서 한 고침과 같음). v3의 모든 .py를 3.9 문법으로 검사했고 문제는 0개였다.
5. `tracking_config.example.json`에 새 설정 2개(`heading_min_displacement_cm` 0.5, `heading_min_speed_cm_s` 1.0)를 적어 넣었다. 동료 판에는 코드 기본값만 있었다. 실제 값은 run.json의 `vehicle_tracking` 또는 `--tracking-config` 파일로 바꾼다.
6. 폴더 이름: `README.md` 머리, `TRACKING.md` 실행 폴더, `make_handoff.py` 묶음 이름(`handoff_점유판정v3_<날짜>`, 안의 폴더 `occupancy_v3`)을 v3로 바꿨다. `TRACKING.md` 끝에는 동료가 쓴 설명 절을 붙였다.

## 3. 시험 (이 PC: Python 3.14.7, numpy 2.5.4, ultralytics 8.4.182)

| 시험 | 결과 |
|---|---|
| test_vehicle_tracker | 17개 통과(동료가 더한 방향 시험 2개 포함) |
| test_tracking_integration | 5개 통과 |
| test_tracking_replay | 통과(추적 끔/켬/20배속 상태 변화 107줄 같음) |
| test_geometry | 41 / 41 통과 |
| test_pipeline | 33 / 33 통과 |
| test_autoref | **12 / 13**: "b. 대각선" 1개 실패(기준점 오차 최대 0.337cm, 기준 0.3cm) |

- test_autoref 실패는 합치기 때문이 아니다. 이 시험이 쓰는 파일(`parkit_autoref.py`, `parkit_synth_mat.py`, `parkit_geom.py`, `test_autoref.py` 등)은 v2, occupancy와 **한 글자도 다르지 않다**(v2와 v3의 차이는 2절의 파일뿐). hun의 PC(Python 3.9, numpy 2.0.2)에서는 13/13이었으니, 이 PC의 라이브러리 판 차이(Python 3.14, numpy 2.5.4)로 합성 화면이나 RANSAC 결과가 조금 달라진 것으로 본다. hun의 PC에서 v3로 다시 돌려 확인한다.
- test_pipeline이 `logs\test_pipeline\`에 만든 시험 영상은 지웠다.

- 웹 시험(`npm test`)과 웹 빌드는 Node가 없어 돌리지 못했다.
- Python 3.9 환경이 이 PC에 없어서 3.9 실행은 hun의 PC에서 확인해야 한다(문법 검사만 함).

## 4. 남은 것

- 실제 미니카로 웹 지도 아이콘 방향 확인. 후진하면 차 앞쪽과 반대로 보일 수 있다(동료 설명).
- 방향 기준값(0.5cm, 1cm/s)은 동료 초기값이다. 실제 미니카로 맞춘다(2절 5번의 설정으로 바꿈).
- 저장소 맨 위의 `occupancy\` 폴더(동료가 올린 원본)는 그대로 있다. git에 올리지 않으려면 지우거나 다른 곳으로 옮겨야 한다(hun이 정함).
