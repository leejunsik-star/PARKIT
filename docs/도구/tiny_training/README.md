# PARKIT Tiny 자체 CNN 학습

기존 프로젝트 루트의 `parkit_detector`를 이 위치로 이동했습니다. 팀 YOLO 학습과 별개인 자체 모델 학습 기능입니다.

가상환경을 활성화한 후 이 폴더에서 실행합니다 (Mac/Windows 공통).
```sh
python -m pip install -r requirements.txt
python -m parkit_detector.run --help
python -m parkit_detector.check
```

학습 데이터는 `images/train`, `images/val`, `labels/train`, `labels/val` 구조로 준비한 뒤 `--root`에 **실제 데이터 경로**를 지정합니다.
`run.py`는 YOLO의 train_M8.txt 목록을 직접 읽지 않습니다. M8의 선택 이미지와 라벨을 위 구조로 준비해야 합니다.
```sh
python -m parkit_detector.run train --root /실제/데이터경로 --out /실제/출력경로 --device cpu --epochs 100 --batch 4
```
Colab CUDA 환경에서는 `--device cuda`를 사용합니다. 이 정리 작업에서 새 학습은 실행하지 않았습니다.

- `model.py`: 자체 모델 구조·손실·예측 후처리.
- `data.py`: 이미지/라벨 로드·전처리·증강.
- `run.py`: 학습·평가·체크포인트 저장·벤치마크.
- `calibrate.py`: 검증 데이터 기반 신뢰도 평가 (카메라 보정과 다른 도구).
- `check.py`: 가짜 입력을 이용한 구조·손실·역전파 검사.
- `inspect_checkpoint.py`, `report.py`: 가중치 및 결과 확인.

학습된 모델은 `docs/도구/models/mymodel_M1.pt`, `mymodel_M5.pt`, `mymodel_M8.pt`에 있습니다.
새 모델은 이름을 구분해 공용 모델 폴더에 저장한 후 탐지 화면에서 직접 선택하세요.
이전 runs 경로에 자동 설치하던 `install_m8.py`와 오래된 전달용 Colab 압축본은 제거했습니다.
