# PARKIT 팀 작업환경

팀 기준 저장소: https://github.com/hun509/parkit

## 지금 실행할 프로그램

| 목적 | 파일 |
|---|---|
| 실제 카메라 점유 판정·차량 추적·2D 웹 지도 | `docs/도구/occupancy_v7/occupancy.py` |
| 카메라 보정 | `docs/도구/occupancy_v7/calibrate.py` 또는 `calibrate_v2.py` |
| 탐지 모델만 확인 | `docs/도구/live_detect_winmac.py` |
| 합성 데이터 데모 | `docs/도구/occupancy_v7/tracking_demo.py` |
| 공용 모델 | `docs/도구/models/` |
| 자체 CNN 학습 코드 | `docs/도구/tiny_training/` |

기존 `runs/occupancy`와 `runs/train` 실행 경로는 정리했습니다. 예전 발표자료의 실행 경로 대신 이 문서를 사용하세요.
`M8.pt`는 팀 YOLO 모델이고, `mymodel_M8.pt`는 자체 PARKIT Tiny 모델입니다. 서로 다른 구조입니다.

## 환경 준비

각 컴퓨터에서 Python 3.12 가상환경을 만듭니다. 가상환경 폴더는 다른 OS로 복사하지 않습니다.
Mac GUI는 Tk 9를 포함하는 Python을 사용하세요. 현재 Mac의 `.venv-gui`는 유지했습니다.

Mac (프로젝트 루트):
```sh
python3.12 -m venv .venv-gui
source .venv-gui/bin/activate
python -m pip install -r docs/도구/occupancy_v7/requirements.txt -r docs/도구/occupancy_v7/requirements-web.txt
python docs/도구/occupancy_v7/occupancy.py --web
```

Windows PowerShell (프로젝트 루트):
```powershell
py -3.12 -m venv .venv-gui
.venv-gui\Scripts\python.exe -m pip install -r docs/도구/occupancy_v7/requirements.txt -r docs/도구/occupancy_v7/requirements-web.txt
.venv-gui\Scripts\python.exe docs/도구/occupancy_v7/occupancy.py --web
```

카메라 선택 창에서 연결한 장치를 고릅니다. 웹 지도는 http://127.0.0.1:8765 에서 봅니다.
각 카메라는 실제 설치 위치에서 별도로 보정해야 합니다. 보정 파일을 복사해 소스 번호만 바꾸는 것은 올바른 보정이 아닙니다.

## VS Code

`Python: Select Interpreter`에서 해당 컴퓨터의 `.venv-gui`를 선택합니다.
실행 및 디버그에서 `PARKIT 점유·추적·웹 (v7)`, `PARKIT 카메라 보정 (v7)`, `PARKIT 탐지만 확인` 중 선택 후 F5로 실행합니다.
현재 파일 실행 버튼은 선택한 파일만 실행하므로 위 실행 구성을 사용하는 것이 확실합니다.

## 폴더 관리

- `dataset/`: 원본 사진·라벨과 학습 분할 데이터. 사진은 Git에 포함되지 않으므로 별도 공유.
- `docs/도구/`: 팀 실행 코드. 현재 실행 기준은 v7.
- `docs/발표자료/`: 완성된 발표 파일과 대본.
- `docs/자체모델_개발자료/`: 자체 CNN 관련 설명·학습 결과 기록. 과거 실행 경로는 현재 경로와 다를 수 있음.
- `first/`, `parkit_runs_2차/`, `parkit_runs_3차/`, 이전 occupancy 버전: 팀에서 관리하는 원본·비교 자료로 유지.

팀 버전별 `models/`, `data/`, `web/dist/`는 해당 폴더만 전달할 때도 실행되도록 포함된 구성입니다. 단순 중복으로 지우지 않았습니다.
정리 내역과 검증 결과: `docs/45_팀작업환경_폴더정리.md`.

## 아이폰·웹캠 3~4대 혼합 촬영

카메라별 상부/사선 촬영 방식 선택과 보정 순서는 [혼합 촬영 안내](docs/46_아이폰_웹캠_혼합_4대_촬영.md)를 참고하세요. 기존 실제 보정 파일은 그대로 유지했습니다.
