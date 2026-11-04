# PARKIT 점유 판정 프로그램

1:64 주차장 매트를 웹캠으로 찍어 칸마다 차가 있는지 판정한다. 이 폴더만 있으면 Windows와 맥에서 돈다.
자세한 사용법은 같은 폴더의 `28_점유판정_프로그램_사용법.md`에 있다. 원본은 Parkit 저장소의 `docs/28_점유판정_프로그램_사용법.md`다.

## 폴더 구성

| 이름 | 내용 |
|---|---|
| `calibrate.py` | **실행.** 보정 도구: 카메라마다 기준점을 찍고 감시 칸을 골라 `config/cam_<이름>.json`에 저장 |
| `occupancy.py` | **실행.** 판정 프로그램: 웹캠·영상에서 차를 찾아 칸 상태를 화면과 `logs/`에 낸다 |
| `parkit_*.py` | 위 두 프로그램이 불러 쓰는 모듈 |
| `test_geometry.py`, `test_pipeline.py` | 합성 시험 (코드를 고친 뒤 확인용) |
| `make_handoff.py` | 이 폴더를 남에게 보낼 묶음(zip)으로 만들기 |
| `requirements.txt` | 설치할 라이브러리 |
| `config/` | 실행 설정(`run.json`, `run_2대.json`)과 카메라 보정 파일(`cam_*.json`) |
| `data/칸표_매트좌표_초안.json` | 칸 56개의 매트 좌표 |
| `models/M8.pt` | 차 탐지 모델 (YOLO26n, 클래스 car). 기본 모델 |
| `models/mymodel_*.pt`, `models/model.py` | 직접 만든 PARKIT Tiny 모델과 구조 파일. 판정 프로그램 창의 [모델]에서 고른다 |
| `logs/` | 실행 기록 |

## 설치

Python은 Windows 3.9 이상, 맥은 **python.org의 Python 3.12**를 쓴다(맥의 시스템 파이썬은 창이 꺼진다).

**Windows**
```
cd occupancy
python -m pip install -r requirements.txt
```

**맥**
```
python3.12 -m venv .venv-gui
source .venv-gui/bin/activate
cd occupancy
python -m pip install -r requirements.txt
```
처음 웹캠을 열 때 터미널(또는 VS Code)의 카메라 사용을 허용한다(시스템 설정 → 개인정보 보호 및 보안 → 카메라).
필요한 라이브러리와 버전 조건, 설치 확인 방법은 28 문서 2절에 있다.

## 빠른 시작

1. 웹캠을 매트 옆에 고정한다.
2. 보정한다. 기준점 6~8개를 찍고 S로 저장한다.
   ```
   python calibrate.py --cam CAM-1 --source 0
   ```
   - `python calibrate.py`만 쳐도 CAM-1, 웹캠 0번(저장된 입력이 있으면 그것)으로 켜진다.
   - 같이 온 `config/cam_CAM-1.json`은 보낸 사람의 카메라 자리에서 한 보정이다. 카메라 자리가 다르면 다시 찍는다.
3. 판정한다.
   ```
   python occupancy.py
   ```
   - 녹화 영상으로 돌리려면 `python occupancy.py --source CAM-1=녹화.mp4`
   - 카메라 2대는 `python occupancy.py --run config/run_2대.json`

두 프로그램 모두 창에서 **? 키**를 누르면 단축키 표가 나온다. 웹캠 번호, 해상도, 모델, 신뢰도, 분석 간격은 창에서 고른다.

## 확인 (선택)
```
python test_geometry.py
python test_pipeline.py
```
둘 다 마지막에 "합계: N / N 통과"가 나오면 된다.

## 선택적 2D 실시간 차량 추적 (2026-11-03)

기존 점유 판정은 유지하고 `python occupancy.py --web`으로 차량 Global ID와 React + SVG 지도를 함께 실행할 수 있습니다. 처음에는 `requirements-web.txt` 설치와 `web/` 빌드가 필요합니다. 카메라 없이 확인하려면 `python tracking_demo.py`를 사용합니다.

[설치·실행·구조·시험 결과·실제 CCTV 검증 항목](TRACKING.md)을 먼저 확인하세요. 기본 웹 주소는 `http://127.0.0.1:8765`입니다. 실제 CCTV 추적 정확도는 아직 검증하지 않았습니다.
