"""3차 학습 모델 채점 (PARKIT, 20 문서) — eval_2차.py를 바탕으로 만듦. PC CPU에서 실행

모델 여러 개를 시험 세트 T1~T8(split_3차의 test_T*.yaml)로 채점해서 표 하나(csv, 마크다운)로 만든다.
    차 있음 T1 집 · T2 학교 · T3 차 적음 · T6 차+물체 · T8 다른 배경
        - Precision, Recall, mAP50 (ultralytics model.val, split=test)
        - 신뢰도 기준(기본 0.25)에서 놓친 차 수, 잘못 잡은 상자 수
          (신뢰도 높은 상자부터, 아직 짝 없는 정답 중 IoU가 가장 큰 것과 짝짓는다. IoU 0.5 이상이면 맞음)
    차 없음 T4 빈 매트 · T5 COCO · T7 물체만
        - 신뢰도 기준에서 잘못 잡은 상자 수, 오탐 있는 사진 비율

사용법 (docs\\도구 에서):
    python eval_3차.py                           best.pt + models\\M5~M8.pt 중 있는 것
    python eval_3차.py best.pt models\\M5.pt      고른 모델만
    python eval_3차.py --conf 0.4                놓친 차·오탐을 셀 신뢰도 기준 바꾸기
    python eval_3차.py --out 결과\\3차채점        결과 폴더 바꾸기

모델 이름은 파일 이름(확장자 뺀 것)으로 표에 들어간다. 코랩에서 받은 best.pt는 M6.pt ~ M8.pt로 이름을 바꿔
docs\\도구\\models 에 두면 된다(20 문서 3절).
결과: <out>\\eval_3차.csv, <out>\\eval_3차.md (기본 out = docs\\도구\\eval_3차_결과)
"""

import argparse, csv, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.normpath(os.path.join(HERE, "..", "..", "dataset", "split_3차"))
DEFAULT_OUT = os.path.join(HERE, "eval_3차_결과")
DEFAULT_MODELS = ["best.pt"] + [os.path.join("models", f"M{i}.pt") for i in range(5, 9)]
POS = [("T1", "집"), ("T2", "학교"), ("T3", "차 적음"), ("T6", "차+물체"), ("T8", "다른 배경")]
NEG = [("T4", "빈 매트"), ("T5", "COCO"), ("T7", "물체만")]
IOU = 0.5


def test_images(t):
    """test_T*.txt의 ./images/test/... 줄을 실제 경로로 바꾼다."""
    with open(os.path.join(SPLIT, f"test_{t}.txt"), encoding="utf-8") as f:
        return [os.path.join(SPLIT, *x[2:].split("/")) for x in f.read().splitlines() if x]


def gt_boxes(img, w, h):
    """사진과 같은 이름의 라벨(YOLO 정규화 xywh)을 픽셀 xyxy로 읽는다."""
    lab = os.path.join(SPLIT, "labels", "test", os.path.splitext(os.path.basename(img))[0] + ".txt")
    out = []
    with open(lab, encoding="utf-8") as f:
        for r in f.read().splitlines():
            p = r.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts):
    """preds: [(신뢰도, xyxy)]. 맞은 상자 수를 돌려준다(정답 하나에 상자 하나만)."""
    used, tp = set(), 0
    for _, p in sorted(preds, key=lambda x: -x[0]):
        best, bi = IOU, None
        for i, g in enumerate(gts):
            if i not in used:
                v = iou(p, g)
                if v >= best:
                    best, bi = v, i
        if bi is not None:
            used.add(bi)
            tp += 1
    return tp


def count_at_conf(model, t, conf):
    """신뢰도 기준에서 정답 차 수, 놓친 차 수, 잘못 잡은 상자 수, 잘못 잡은 상자가 있는 사진 수."""
    imgs = test_images(t)
    n_gt = miss = fp = fp_img = 0
    for i in range(0, len(imgs), 8):  # 경로를 넘기면 ultralytics가 한글 경로도 읽는다
        for img, r in zip(imgs[i:i + 8], model.predict(imgs[i:i + 8], imgsz=640, conf=conf,
                                                       device="cpu", verbose=False)):
            h, w = r.orig_shape
            gts = gt_boxes(img, w, h)
            preds = [(float(c), tuple(float(v) for v in b))
                     for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            tp = match(preds, gts)
            n_gt += len(gts)
            miss += len(gts) - tp
            fp += len(preds) - tp
            fp_img += len(preds) > tp
    return {"정답": n_gt, "놓침": miss, "오탐": fp, "오탐사진": fp_img, "사진": len(imgs)}


def eval_val(model, t, work):
    r = model.val(data=os.path.join(SPLIT, f"test_{t}.yaml"), split="test", imgsz=640, batch=4,
                  device="cpu", workers=0, plots=False, verbose=False,
                  project=work, name=t, exist_ok=True)
    return {"P": r.box.mp, "R": r.box.mr, "mAP50": r.box.map50}


def main():
    ap = argparse.ArgumentParser(description="3차 학습 모델 채점 (T1~T8)")
    ap.add_argument("models", nargs="*", help="채점할 .pt (생략하면 best.pt + models\\M5~M8.pt 중 있는 것)")
    ap.add_argument("--conf", type=float, default=0.25, help="놓친 차·오탐을 셀 신뢰도 기준 (기본 0.25)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="결과 폴더")
    a = ap.parse_args()

    if not os.path.isdir(SPLIT):
        print(f"split_3차가 없음: {SPLIT}\n먼저 make_split_3차.py를 실행하세요.")
        return 1
    if a.models:
        models = [os.path.abspath(m) for m in a.models]
    else:
        models = [os.path.join(HERE, m) for m in DEFAULT_MODELS if os.path.exists(os.path.join(HERE, m))]
    missing = [m for m in models if not os.path.exists(m)]
    if missing or not models:
        print("모델이 없음: " + (", ".join(missing) or "채점할 모델이 없음"))
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
            m = eval_val(model, t, work)
            c = count_at_conf(model, t, a.conf)
            row.update({f"{t} P": m["P"], f"{t} R": m["R"], f"{t} mAP50": m["mAP50"],
                        f"{t} 정답 차": c["정답"], f"{t} 놓친 차": c["놓침"], f"{t} 잘못 잡은 상자": c["오탐"]})
            print(f"  {t} {label}: P {m['P']:.3f}  R {m['R']:.3f}  mAP50 {m['mAP50']:.3f}  |  "
                  f"conf {a.conf}: 놓친 차 {c['놓침']}/{c['정답']}, 잘못 잡은 상자 {c['오탐']}")
        for t, label in NEG:
            c = count_at_conf(model, t, a.conf)
            row.update({f"{t} 오탐 상자": c["오탐"], f"{t} 오탐 사진 비율": c["오탐사진"] / c["사진"],
                        f"{t} 사진": c["사진"]})
            print(f"  {t} {label}: 오탐 상자 {c['오탐']}개, 오탐 있는 사진 {c['오탐사진']}/{c['사진']}")
        print(f"  ({time.time() - t0:.0f}초)")
        rows.append(row)
    shutil.rmtree(work, ignore_errors=True)

    pos_keys = ("P", "R", "mAP50", "정답 차", "놓친 차", "잘못 잡은 상자")
    cols = ["모델"] + [f"{t} {k}" for t, _ in POS for k in pos_keys] \
        + [f"{t} {k}" for t, _ in NEG for k in ("오탐 상자", "오탐 사진 비율")]
    csv_path = os.path.join(a.out, "eval_3차.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], (str, int)) else f"{r[c]:.4f}" for c in cols])

    # 마크다운은 한 표에 넣으려고 세트마다 두 칸으로 줄인다 (자세한 값은 csv)
    r0 = rows[0]
    head = ["모델"]
    for t, label in POS:
        head += [f"{t} {label} P / R / mAP50", f"{t} 놓친 차 / 잘못 잡은 상자 ({r0[f'{t} 정답 차']}대)"]
    for t, label in NEG:
        head += [f"{t} {label} 오탐 상자 / 오탐 사진 ({r0[f'{t} 사진']}장)"]
    md = [f"# 3차 학습 모델 채점 결과 ({time.strftime('%Y-%m-%d %H:%M')})", "",
          "- 차 있음(T1 집, T2 학교, T3 차 적음, T6 차+물체, T8 다른 배경): P·R·mAP50은 ultralytics `model.val(split='test')`.",
          f"  놓친 차·잘못 잡은 상자는 신뢰도 {a.conf} 이상 상자를 정답과 IoU {IOU} 이상으로 1:1 짝지어 셈",
          f"- 차 없음(T4 빈 매트, T5 COCO, T7 물체만): 신뢰도 {a.conf} 이상 상자를 모두 오탐으로 셈. 오탐 사진 = 상자가 하나라도 있는 사진 비율",
          "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["모델"]]
        for t, _ in POS:
            cells += [f"{r[f'{t} P']:.3f} / {r[f'{t} R']:.3f} / {r[f'{t} mAP50']:.3f}",
                      f"{r[f'{t} 놓친 차']} / {r[f'{t} 잘못 잡은 상자']}"]
        for t, _ in NEG:
            cells += [f"{r[f'{t} 오탐 상자']} / {r[f'{t} 오탐 사진 비율'] * 100:.0f}%"]
        md.append("| " + " | ".join(cells) + " |")
    md_path = os.path.join(a.out, "eval_3차.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md[6:]))
    print(f"\n저장: {csv_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
