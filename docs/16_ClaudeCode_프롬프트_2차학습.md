# Claude Code 작업 프롬프트: 2차 학습 데이터 정리 (2026-10-30)

15 문서의 실험 계획을 PC에서 준비하는 작업이다. 아래 코드 블록을 Claude Code에 그대로 붙여 넣는다. 학습은 코랩에서 직접 돌린다.

## 1. 붙여 넣을 프롬프트

```text
[목표]
PARKIT(1:64 주차장 미니카 탐지, YOLO26n 파인튜닝)의 2차 학습을 준비한다.
1) 모든 데이터를 배치 단위로 train/val/test로 나눈다.
2) 데이터 조합만 다른 모델 5개(M1~M5)를 코랩에서 학습할 수 있게 묶는다.
3) 학습된 모델들을 PC에서 출처별 시험 세트로 채점하는 스크립트를 만든다.
4) 동료에게 넘길 handoff_2차 폴더를 만든다.

[대상]
- 프로젝트 폴더: E:\ccapstione\Parkit
- 데이터: E:\ccapstione\Parkit\dataset
  - data_1차(home) 39장 / labels_1차(home) (A_1~A_7, B_1~B_6 배치, 파일명 A_1_0001.jpg 형식)
  - data_2차(school) 54장 s{1-9}_{1-6}.jpg / labels_2차(school). data_2차(school)\중복 폴더는 쓰지 않는다
  - data_3차(carX) 30장 x_0001~x_0030.jpg, 차 없는 매트 사진 / labels_3차(carX) 빈 txt
  - data_4차(little_car) 26장 little_car_{1-4}_{번호}.jpg / labels_4차(little_car)
  - data_neg_coco 60장 neg_coco_*.jpg + 출처.csv / labels_neg_coco 빈 txt
  - 각 labels 폴더의 labels.txt(내용 car)는 클래스 목록이라 사진 라벨이 아니다
  - handoff_1차: 1차 데이터의 기존 나누기(train A_1~A_5·B_1~B_4, val A_6·B_5, test A_7·B_6)와 data.yaml
- 도구: E:\ccapstione\Parkit\docs\도구 (best.pt = 1차 학습 모델, coco_negatives.py = COCO 차 없는 사진 생성기)
- 계획 문서: docs\15_학습데이터_조합과_검증.md, docs\10_추가학습_계획.md(7-1절 코랩 절차), docs\14_오탐지_대책.md, docs\13_맥_Windows_공통실행.md(코드 규칙)

[현재 상태]
- 1차·3차·COCO 라벨은 확정이다.
- 2차·4차 라벨은 best.pt 자동 라벨을 사람이 Make Sense로 검수하는 중이다.
- 지난 학습 설정: yolo26n.pt, imgsz=640, epochs=100, batch=16, hsv_h=0.05 (코랩 T4 GPU).
- PC에는 GPU가 없다(Ryzen 5 5600G 내장 그래픽). 학습은 코랩, 나누기와 채점은 PC에서 한다.

[나누기 기준 (배치째로 나눈다. 같은 배치의 사진이 train과 test에 섞이면 안 된다)]
- 1차: handoff_1차와 똑같이
- 2차: test s8·s9, val s7, train s1~s6
- 4차: test little_car_4, val little_car_2, train little_car_1·little_car_3
- 3차: test x_0025~x_0030, val x_0022~x_0024, train 후보 x_0001~x_0021
- COCO: train 후보는 data_neg_coco 60장. test는 coco_negatives.py --seed 2 --n 30 --out dataset\data_neg_coco_test 로 새로 만든다.
  seed 1 묶음과 COCO 번호가 겹치는 사진이 있으면 빼고 다른 사진으로 채운다.
  필요하면 겹치는 번호를 빼는 옵션을 coco_negatives.py에 추가한다.

[모델 5개의 train 구성 (val·test는 모두 같다)]
- M1: 1차
- M2: 1차 + 2차
- M3: 1차 + 2차 + 4차
- M4: M3 + 차 없는 사진 약 15장(3차 후보에서 8장 + COCO 후보에서 7장, 고정 seed로 고르게 고름)
- M5: M3 + 차 없는 사진 후보 전부(3차 21장 + COCO 60장)
사진을 모델마다 복사하지 말고, 한 폴더에 나눈 뒤 모델별 이미지 목록 txt와 data yaml로 구성한다.

[만들 것]
1. dataset\split_2차\ : images·labels 아래 train/val/test, 모델별 목록(train_M1.txt ~ train_M5.txt), 모델별 yaml, 시험 세트별 yaml
   - T1 집(1차) / T2 학교(2차) / T3 차 적음(4차) / T4 빈 매트(3차) / T5 COCO
   - 경로는 yaml 위치 기준 상대경로로 해서 코랩과 PC에서 둘 다 쓰게 한다.
   - 파일명이 출처별로 겹치지 않는지 확인한다.
2. 코랩에 올릴 압축 파일 하나(split_2차 전체)
3. docs\도구\colab_train_2차.ipynb (또는 셀별로 붙여 넣을 코드 문서)
   - 드라이브 연결 → /content로 압축 풀기 → M1~M5를 같은 설정으로 차례로 학습 → 모델마다 끝나는 즉시 best.pt와 results.png를 드라이브에 저장
   - 연결이 끊겨도 끝난 모델은 건너뛰고 이어서 학습할 수 있게 한다.
4. docs\도구\eval_2차.py (PC CPU에서 실행)
   - 모델 여러 개 × 시험 세트 T1~T5를 채점해서 표 하나로 만든다(csv와 마크다운).
   - T1~T3: Precision, Recall, mAP50 (ultralytics model.val, split=test)
   - T4·T5: 신뢰도 0.25에서 잘못 잡은 상자 수, 오탐이 있는 사진 비율
   - 기존 best.pt도 같은 표에 넣는다.
5. dataset\handoff_2차\ : 동료에게 넘길 이미지와 라벨만
   - M4 구성(추천안)의 images·labels train/val/test와 data.yaml(path: .)만 넣는다. 문서는 넣지 않는다.
6. docs\17_2차학습_실행방법.md : 위 파일들 쓰는 순서와 결과 표 빈칸

[제약 조건]
- 시작 전에 2차·4차 라벨 검수가 끝났는지, 최종 라벨이 labels_2차(school)·labels_4차(little_car)에 있는지 나에게 먼저 물어본다. 끝나지 않았으면 나누기와 압축은 하지 않는다.
- data_*·labels_* 원본 폴더는 수정하거나 옮기지 않는다. 복사만 한다.
- E:\ccapstione\first 는 1학기 백업이라 읽기만 하고 절대 수정하지 않는다.
- 기존 스크립트(live_detect.py, capture.py 등)는 수정하지 않는다. coco_negatives.py는 위 겹침 처리에 필요한 만큼만 고친다.
- 코드는 Windows와 맥에서 모두 돌게 쓴다(docs\13 3절 규칙: 스크립트 위치 기준 경로, 한글 경로에서도 되는 이미지 저장 등).
- 설명, 주석, 문서는 한국어로 쓴다. 결과 문서는 docs 폴더에 마크다운으로 둔다.

[완료 조건]
- split_2차의 출처별·나누기별 사진 수와 라벨 수가 일치하고, 같은 배치가 두 나누기에 걸치지 않음을 스크립트로 확인해서 결과를 보여준다.
- 모든 사진에 같은 이름의 라벨 txt가 있다(차 없는 사진은 빈 파일).
- eval_2차.py를 기존 best.pt 하나로 실제 실행해서 표가 나온다.
- 코랩 노트북은 코랩에서 열면 바로 실행할 수 있는 상태다(실제 학습은 내가 돌린다).
- handoff_2차에 data.yaml과 images·labels만 있고 개수가 맞다.

[작업 방식]
- 큰 변경 전에 계획(만들 파일, 나누기 결과 표)을 먼저 보여주고 확인을 받는다.
- 끝나면 만든 파일과 바꾼 파일, 사진 수 표를 요약해서 보여준다.
```

## 2. CLAUDE.md에 넣을 내용 제안

`E:\ccapstione\Parkit\CLAUDE.md`에 두면 Claude Code가 매번 읽는다.

```markdown
# PARKIT

1:64 주차장 매트(80×55cm)에 미니카를 올리고 USB 웹캠으로 찍어, 칸별 주차 여부를 판단하는 캡스톤 프로젝트.
탐지 모델은 YOLO26n을 미니카 사진으로 파인튜닝해서 쓴다(클래스 1개: car).

## 규칙
- 한국어로 소통하고, 주석과 문서도 한국어로 쓴다.
- 결과 문서는 docs 폴더에 마크다운으로 둔다(번호_제목.md).
- E:\ccapstione\first 는 1학기 앱 원본 백업이다. 읽기만 하고 절대 수정하지 않는다.
- 요청 전에는 코드를 고치지 않는다. 기존 파일을 고쳐야 하면 원본을 두고 새 이름으로 만든다(예: *_winmac.py).
- dataset의 data_*, labels_* 폴더는 원본이다. 옮기거나 고치지 말고 복사해서 쓴다.
- 코드는 Windows와 맥에서 모두 돌게 쓴다(docs\13_맥_Windows_공통실행.md 3절).

## 폴더
- docs\도구 : 실행 스크립트와 모델(best.pt = 1차 학습 모델)
- dataset\data_N차(설명) / labels_N차(설명) : 차수별 사진과 YOLO 라벨. labels.txt는 클래스 목록(car)
  - 1차 home, 2차 school, 3차 carX(차 없는 매트), 4차 little_car(차 적음), neg_coco(COCO 차 없는 사진)
- dataset\handoff_N차 : 동료에게 넘기는 묶음
- 학습은 Google Colab(T4)에서, 데이터 정리와 채점은 PC(CPU)에서 한다.

## 주요 문서
- docs\10 추가학습 계획(코랩 절차), docs\14 오탐지 대책, docs\15 학습데이터 조합과 검증
```
