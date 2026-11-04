"""2차 학습 모델 채점 (PARKIT, 17 문서) — PC CPU에서 실행

모델 여러 개를 시험 세트 T1~T5(split_2차의 test_T*.yaml)로 채점해서 표 하나(csv, 마크다운)로 만든다.
    T1 집(1차) · T2 학교(2차) · T3 차 적음(4차) : Precision, Recall, mAP50 (ultralytics model.val, split=test)
    T4 빈 매트(3차) · T5 COCO                    : 신뢰도 기준(기본 0.25)에서 잘못 잡은 상자 수, 오탐 있는 사진 비율

사용법 (docs\\도구 에서):
    python eval_2차.py                                 best.pt + models 폴더의 *.pt 전부
    python eval_2차.py best.pt models\\M1.pt models\\M4.pt   고른 모델만
    python eval_2차.py --conf 0.4                      T4·T5 신뢰도 기준 바꾸기
    python eval_2차.py --out 결과\\2차채점              결과 폴더 바꾸기

모델 이름은 파일 이름(확장자 뺀 것)으로 표에 들어간다. 코랩에서 받은 best.pt는 M1.pt ~ M5.pt로 이름을 바꿔
docs\\도구\\models 에 두면 된다(17 문서 3절).
결과: <out>\\eval_2차.csv, <out>\\eval_2차.md (기본 out = docs\\도구\\eval_2차_결과)
"""

import argparse, csv, glob, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.normpath(os.path.join(HERE, "..", "..", "dataset", "split_2차"))
DEFAULT_OUT = os.path.join(HERE, "eval_2차_결과")
POS = [("T1", "집"), ("T2", "학교"), ("T3", "차 적음")]
NEG = [("T4", "빈 매트"), ("T5", "COCO")]


def default_models():
    ms = [os.path.join(HERE, "best.pt")]
    ms += sorted(glob.glob(os.path.join(HERE, "models", "*.pt")))
    return [m for m in ms if os.path.exists(m)]


def test_images(t):
    """test_T*.txt의 ./images/test/... 줄을 실제 경로로 바꾼다."""
    with open(os.path.join(SPLIT, f"test_{t}.txt"), encoding="utf-8") as f:
        return [os.path.join(SPLIT, *x[2:].split("/")) for x in f.read().splitlines() if x]


def eval_pos(model, t, work):
    r = model.val(data=os.path.join(SPLIT, f"test_{t}.yaml"), split="test", imgsz=640, batch=4,
                  device="cpu", workers=0, plots=False, verbose=False,
                  project=work, name=t, exist_ok=True)
    return {"P": r.box.mp, "R": r.box.mr, "mAP50": r.box.map50}


def eval_neg(model, t, conf):
    imgs = test_images(t)
    boxes, hit = 0, 0
    for i in range(0, len(imgs), 8):  # 경로를 넘기면 ultralytics가 한글 경로도 읽는다
        for r in model.predict(imgs[i:i + 8], imgsz=640, conf=conf, device="cpu", verbose=False):
            n = len(r.boxes)
            boxes += n
            hit += n > 0
    return {"오탐상자": boxes, "오탐사진": hit, "사진": len(imgs)}


def main():
    ap = argparse.ArgumentParser(description="2차 학습 모델 채점 (T1~T5)")
    ap.add_argument("models", nargs="*", help="채점할 .pt (생략하면 best.pt + models\\*.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="T4·T5에서 상자로 셀 신뢰도 기준 (기본 0.25)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="결과 폴더")
    a = ap.parse_args()

    if not os.path.isdir(SPLIT):
        print(f"split_2차가 없음: {SPLIT}\n먼저 make_split_2차.py를 실행하세요.")
        return 1
    models = [os.path.abspath(m) for m in a.models] or default_models()
    if not models:
        print("채점할 모델이 없음")
        return 1
    from ultralytics import YOLO  # 불러오는 데 시간이 걸려서 확인 뒤에 읽는다

    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "_val")  # model.val이 만드는 폴더 (끝나면 지움)
    rows = []
    for mp in models:
        name = os.path.splitext(os.path.basename(mp))[0]
        print(f"\n=== {name} ({mp})")
        t0 = time.time()
        model = YOLO(mp)
        row = {"모델": name}
        for t, label in POS:
            m = eval_pos(model, t, work)
            row.update({f"{t} P": m["P"], f"{t} R": m["R"], f"{t} mAP50": m["mAP50"]})
            print(f"  {t} {label}: P {m['P']:.3f}  R {m['R']:.3f}  mAP50 {m['mAP50']:.3f}")
        for t, label in NEG:
            m = eval_neg(model, t, a.conf)
            row.update({f"{t} 오탐 상자": m["오탐상자"], f"{t} 오탐 사진 비율": m["오탐사진"] / m["사진"],
                        f"{t} 사진": m["사진"]})
            print(f"  {t} {label}: 오탐 상자 {m['오탐상자']}개, 오탐 있는 사진 {m['오탐사진']}/{m['사진']}")
        print(f"  ({time.time() - t0:.0f}초)")
        rows.append(row)
    shutil.rmtree(work, ignore_errors=True)

    cols = ["모델"] + [f"{t} {k}" for t, _ in POS for k in ("P", "R", "mAP50")] \
        + [f"{t} {k}" for t, _ in NEG for k in ("오탐 상자", "오탐 사진 비율")]
    csv_path = os.path.join(a.out, "eval_2차.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], (str, int)) else f"{r[c]:.4f}" for c in cols])

    n4, n5 = rows[0]["T4 사진"], rows[0]["T5 사진"]
    head = ["모델"] + [f"{t} {k}" for t, _ in POS for k in ("P", "R", "mAP50")] \
        + [f"T4 오탐 상자 ({n4}장)", "T4 오탐 사진", f"T5 오탐 상자 ({n5}장)", "T5 오탐 사진"]
    md = [f"# 2차 학습 모델 채점 결과 ({time.strftime('%Y-%m-%d %H:%M')})", "",
          "- T1 집(1차) · T2 학교(2차) · T3 차 적음(4차): ultralytics `model.val(split='test')`의 P, R, mAP50",
          f"- T4 빈 매트(3차) · T5 COCO: 신뢰도 {a.conf} 이상 상자를 모두 오탐으로 셈. 오탐 사진 = 상자가 하나라도 있는 사진 비율",
          "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["모델"]] + [f"{r[f'{t} {k}']:.3f}" for t, _ in POS for k in ("P", "R", "mAP50")]
        for t, _ in NEG:
            cells += [str(r[f"{t} 오탐 상자"]), f"{r[f'{t} 오탐 사진 비율'] * 100:.0f}%"]
        md.append("| " + " | ".join(cells) + " |")
    md_path = os.path.join(a.out, "eval_2차.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md[5:]))
    print(f"\n저장: {csv_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
