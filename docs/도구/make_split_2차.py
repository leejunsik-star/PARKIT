"""2차 학습용 데이터 나누기 (PARKIT, 17 문서)

dataset 안의 원본(data_*, labels_*)을 복사해서 배치째로 train/val/test로 나누고,
모델 5개(M1~M5)의 train 목록·yaml, 시험 세트 5개(T1~T5)의 yaml, 코랩용 압축, 동료용 handoff_2차를 만든다.
원본 폴더는 읽기만 한다.

사용법 (docs\\도구 에서):
    python make_split_2차.py              만들기 (이미 있으면 멈춤)
    python make_split_2차.py --overwrite  split_2차, split_2차.zip, handoff_2차를 지우고 다시 만들기
    python make_split_2차.py --check      만들지 않고 이미 만든 split_2차만 확인

만드는 것 (Parkit\\dataset 아래):
    split_2차\\images|labels\\train|val|test   사진과 라벨 (차 없는 사진은 빈 txt)
    split_2차\\train_M1.txt ~ train_M5.txt     모델별 train 사진 목록 (./images/train/... 형식)
    split_2차\\data_M1.yaml ~ data_M5.yaml     모델별 학습 설정 (val·test는 모두 같음)
    split_2차\\test_T1.txt|yaml ~ test_T5      시험 세트별 목록과 설정 (eval_2차.py가 씀)
    split_2차\\확인결과.txt                    사진 수·라벨 수·배치 겹침 확인 결과
    split_2차.zip                             코랩에 올릴 압축 (split_2차 폴더째)
    handoff_2차\\                              M4 구성의 images·labels와 data.yaml
yaml에는 path를 쓰지 않는다. 그러면 ultralytics가 yaml이 있는 폴더를 기준으로 경로를 찾아서 PC와 코랩에서 그대로 쓸 수 있다.
"""

import argparse, os, random, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.normpath(os.path.join(HERE, "..", "..", "dataset"))  # docs\도구 → Parkit\dataset
OUT = os.path.join(DS, "split_2차")
ZIP = os.path.join(DS, "split_2차.zip")
HANDOFF = os.path.join(DS, "handoff_2차")
SEED = 42  # M4의 차 없는 사진 고르기용
SPLITS = ("train", "val", "test")


# ---------- 출처별 배치와 나누기 규칙 ----------
def batch_1(n):   # A_1_0001.jpg → A_1
    return n[:3]


def split_1(b):   # handoff_1차와 같게
    return "test" if b in ("A_7", "B_6") else "val" if b in ("A_6", "B_5") else "train"


def batch_2(n):   # s7_3.jpg → s7
    return n.split("_")[0]


def split_2(b):
    return "test" if b in ("s8", "s9") else "val" if b == "s7" else "train"


def batch_3(n):   # x_0025.jpg → 연달아 찍은 묶음(번호 구간)
    k = int(n[2:6])
    return "x_0025-0030" if k >= 25 else "x_0022-0024" if k >= 22 else "x_0001-0021"


def split_3(b):
    return {"x_0025-0030": "test", "x_0022-0024": "val"}.get(b, "train")


def batch_4(n):   # little_car_2_0003.jpg → little_car_2
    return "_".join(n.split("_")[:3])


def split_4(b):
    return "test" if b == "little_car_4" else "val" if b == "little_car_2" else "train"


def batch_coco(n):  # neg_coco_000000023359.jpg → COCO 번호 (사진마다 따로)
    return n[len("neg_coco_"):-4]


# 키, 이름, 사진 폴더, 라벨 폴더(없으면 빈 라벨), 배치 함수, 나누기 함수, 시험 세트, 차 없는 사진인가
SOURCES = [
    ("1차", "집", "data_1차(home)", "labels_1차(home)", batch_1, split_1, "T1", False),
    ("2차", "학교", "data_2차(school)", "labels_2차(school)", batch_2, split_2, "T2", False),
    ("4차", "차 적음", "data_4차(little_car)", "labels_4차(little_car)", batch_4, split_4, "T3", False),
    ("3차", "빈 매트", "data_3차(carX)", "labels_3차(carX)", batch_3, split_3, "T4", True),
    ("COCO", "COCO 학습 후보(seed 1)", "data_neg_coco", "labels_neg_coco", batch_coco, lambda b: "train", None, True),
    ("COCO시험", "COCO 시험(seed 2)", "data_neg_coco_test", None, batch_coco, lambda b: "test", "T5", True),
]
TEST_NAMES = {"T1": "집(1차)", "T2": "학교(2차)", "T3": "차 적음(4차)", "T4": "빈 매트(3차)", "T5": "COCO"}


def collect():
    """출처별 사진 목록을 읽어 [(출처키, 파일명, 배치, 나누기, 사진경로, 라벨경로|None)]로 돌려준다."""
    items = []
    for key, _, dimg, dlab, bfn, sfn, _, _ in SOURCES:
        d = os.path.join(DS, dimg)
        # 하위 폴더(2차의 중복, COCO의 _cache)는 보지 않는다
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if not (os.path.isfile(p) and n.lower().endswith(".jpg")):
                continue
            b = bfn(n)
            lab = os.path.join(DS, dlab, n[:-4] + ".txt") if dlab else None
            items.append((key, n, b, sfn(b), p, lab))
    return items


def even_pick(names, k, rng):
    """정렬한 목록을 k칸으로 나눠 칸마다 하나씩 무작위로 고른다(고르게 퍼지게)."""
    names = sorted(names)
    out = []
    for i in range(k):
        lo, hi = len(names) * i // k, len(names) * (i + 1) // k
        out.append(names[rng.randrange(lo, hi)])
    return out


def model_lists(items):
    """모델별 train 사진 파일명 목록."""
    tr = lambda *keys: [n for k, n, _, s, _, _ in items if s == "train" and k in keys]
    m3 = tr("1차", "2차", "4차")
    rng = random.Random(SEED)
    neg4 = even_pick(tr("3차"), 8, rng) + even_pick(tr("COCO"), 7, rng)
    return {
        "M1": tr("1차"),
        "M2": tr("1차", "2차"),
        "M3": m3,
        "M4": m3 + neg4,
        "M5": m3 + tr("3차", "COCO"),
    }


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def build(items):
    for s in SPLITS:
        os.makedirs(os.path.join(OUT, "images", s))
        os.makedirs(os.path.join(OUT, "labels", s))
    for key, n, b, s, img, lab in items:
        shutil.copy2(img, os.path.join(OUT, "images", s, n))
        dst = os.path.join(OUT, "labels", s, n[:-4] + ".txt")
        if lab and os.path.exists(lab):
            shutil.copy2(lab, dst)
        elif lab is None:
            write(dst, "")  # COCO 시험 사진: 원본 라벨 폴더가 없으므로 빈 라벨을 만든다
        else:
            raise SystemExit(f"라벨 없음: {lab}")

    names_yaml = "names:\n  0: car\n"
    for m, lst in model_lists(items).items():
        write(os.path.join(OUT, f"train_{m}.txt"), "".join(f"./images/train/{n}\n" for n in lst))
        write(os.path.join(OUT, f"data_{m}.yaml"),
              f"# {m} 학습용. path가 없으면 이 yaml이 있는 폴더가 기준이다(PC·코랩 공통)\n"
              f"train: train_{m}.txt\nval: images/val\ntest: images/test\n" + names_yaml)
    for key, _, _, _, _, _, t, _ in SOURCES:
        if not t:
            continue
        lst = [n for k, n, _, s, _, _ in items if k == key and s == "test"]
        write(os.path.join(OUT, f"test_{t}.txt"), "".join(f"./images/test/{n}\n" for n in lst))
        write(os.path.join(OUT, f"test_{t}.yaml"),
              f"# 시험 세트 {t} {TEST_NAMES[t]}. eval_2차.py에서 split=test로 쓴다(train·val은 형식상 필요)\n"
              f"train: images/train\nval: images/val\ntest: test_{t}.txt\n" + names_yaml)


def make_zip():
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT):
            for f in sorted(files):
                if f.endswith(".cache"):  # ultralytics가 만드는 라벨 캐시는 넣지 않음
                    continue
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, DS).replace(os.sep, "/"))


def make_handoff(items):
    m4 = set(model_lists(items)["M4"])
    for key, n, b, s, _, _ in items:
        if s == "train" and n not in m4:
            continue
        for sub, name in (("images", n), ("labels", n[:-4] + ".txt")):
            d = os.path.join(HANDOFF, sub, s)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(OUT, sub, s, name), os.path.join(d, name))
    write(os.path.join(HANDOFF, "data.yaml"),
          "path: .  # 이 폴더 위치로 바꿔 쓰기\ntrain: images/train\nval: images/val\ntest: images/test\n"
          "names:\n  0: car\n")


# ---------- 확인 ----------
def check(items):
    lines, errors = [], []
    say = lines.append

    # 1) 파일명이 출처끼리 겹치지 않는가
    seen = {}
    for k, n, *_ in items:
        if n in seen:
            errors.append(f"파일명 겹침: {n} ({seen[n]}, {k})")
        seen[n] = k

    # 2) 같은 배치가 두 나누기에 걸치지 않는가 (COCO는 seed1·seed2를 한 출처로 봄)
    where = {}
    for k, n, b, s, *_ in items:
        g = "COCO" if k.startswith("COCO") else k
        where.setdefault((g, b), set()).add(s)
    split_batches = {}
    for (g, b), ss in sorted(where.items()):
        if len(ss) > 1:
            errors.append(f"배치가 나누기에 걸침: {g} {b} → {sorted(ss)}")
        split_batches.setdefault((g, next(iter(ss))), set()).add(b)

    # 3) 폴더 안의 사진·라벨이 1:1이고, 라벨 형식이 맞는가
    on_disk = {}
    for s in SPLITS:
        imgs = {f[:-4] for f in os.listdir(os.path.join(OUT, "images", s)) if f.lower().endswith(".jpg")}
        labs = {f[:-4] for f in os.listdir(os.path.join(OUT, "labels", s)) if f.endswith(".txt")}
        for x in sorted(imgs - labs):
            errors.append(f"라벨 없는 사진: {s}/{x}.jpg")
        for x in sorted(labs - imgs):
            errors.append(f"사진 없는 라벨: {s}/{x}.txt")
        on_disk[s] = imgs

    src_of = {n[:-4]: k for k, n, *_ in items}
    neg = {k for k, *_, is_neg in SOURCES if is_neg}
    table, boxes = {}, {}
    for k, n, b, s, *_ in items:
        lp = os.path.join(OUT, "labels", s, n[:-4] + ".txt")
        with open(lp, encoding="utf-8") as f:
            rows = [r.split() for r in f.read().splitlines() if r.strip()]
        for r in rows:
            ok = len(r) == 5 and r[0] == "0" and all(0 <= float(v) <= 1 for v in r[1:])
            if not ok:
                errors.append(f"라벨 형식 이상: {s}/{n[:-4]}.txt {' '.join(r)}")
                break
        if k in neg and rows:
            errors.append(f"차 없는 사진인데 라벨에 상자가 있음: {s}/{n}")
        if k not in neg and not rows:
            errors.append(f"차 있는 사진인데 라벨이 비어 있음: {s}/{n}")
        c = table.setdefault(k, {sp: [0, 0, 0] for sp in SPLITS})[s]
        c[0] += 1
        c[1] += os.path.exists(os.path.join(OUT, "images", s, n))
        c[2] += 1  # 라벨 파일 (위에서 열었으므로 있음)
        boxes[(k, s)] = boxes.get((k, s), 0) + len(rows)
    extra = {s: on_disk[s] - {n[:-4] for _, n, _, sp, *_ in items if sp == s} for s in SPLITS}
    for s, xs in extra.items():
        for x in sorted(xs):
            errors.append(f"원본 목록에 없는 파일이 들어 있음: {s}/{x}")

    # 4) handoff_1차와 1차 나누기가 같은가
    h1 = os.path.join(DS, "handoff_1차", "images")
    for s in SPLITS:
        a = set(os.listdir(os.path.join(h1, s)))
        b = {n for k, n, _, sp, *_ in items if k == "1차" and sp == s}
        if a != b:
            errors.append(f"1차 {s}가 handoff_1차와 다름: {sorted(a ^ b)}")

    # 5) 목록 txt가 가리키는 사진이 다 있는가
    lists = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".txt") and (f.startswith("train_M") or f.startswith("test_T")):
            with open(os.path.join(OUT, f), encoding="utf-8") as t:
                lst = [x for x in t.read().splitlines() if x]
            for x in lst:
                if not os.path.exists(os.path.join(OUT, x[2:])):
                    errors.append(f"{f}: 없는 사진 {x}")
            lists[f] = lst

    # 출력
    say("== 출처별·나누기별 사진 수 / 라벨 파일 수 / 상자 수 ==")
    say(f"{'출처':<10}" + "".join(f"{s:>20}" for s in SPLITS) + f"{'합계':>8}")
    tot = {s: [0, 0] for s in SPLITS}
    for k, v in table.items():
        cells = []
        for s in SPLITS:
            n_img, n_on, n_lab = v[s]
            cells.append(f"{n_img:>4}장 {n_lab:>4}라벨 {boxes.get((k, s), 0):>4}상자")
            tot[s][0] += n_img
            tot[s][1] += n_lab
            if n_on != n_img:
                errors.append(f"{k} {s}: 복사된 사진 {n_on} ≠ {n_img}")
        say(f"{k:<10}" + "".join(f"{c:>20}" for c in cells) + f"{sum(v[s][0] for s in SPLITS):>8}")
    say(f"{'합계':<10}" + "".join(f"{tot[s][0]:>4}장 {tot[s][1]:>4}라벨{'':>9}" for s in SPLITS))
    say(f"폴더 실제 사진 수: " + ", ".join(f"{s} {len(on_disk[s])}" for s in SPLITS))
    say("")
    say("== 나누기별 배치 ==")
    for (g, s), bs in sorted(split_batches.items(), key=lambda x: (x[0][0], SPLITS.index(x[0][1]))):
        bs = sorted(bs)
        shown = ", ".join(bs) if len(bs) <= 10 else f"{len(bs)}장(사진마다 따로)"
        say(f"  {g:<5} {s:<5}: {shown}")
    say("")
    say("== 목록 파일 ==")
    for f, lst in lists.items():
        cnt = {}
        for x in lst:
            k = src_of[os.path.basename(x)[:-4]]
            cnt[k] = cnt.get(k, 0) + 1
        say(f"  {f:<14} {len(lst):>4}장  " + ", ".join(f"{k} {c}" for k, c in cnt.items()))
    say("")
    if errors:
        say(f"문제 {len(errors)}개:")
        lines.extend("  - " + e for e in errors)
    else:
        say("확인 통과: 파일명 겹침 없음, 배치가 두 나누기에 걸치지 않음, 모든 사진에 같은 이름의 라벨 있음,")
        say("          차 없는 사진은 빈 라벨, 1차 나누기 = handoff_1차, 목록 txt의 사진 모두 있음")
    return lines, errors


def check_handoff(items):
    m4 = set(model_lists(items)["M4"])
    want = {s: {n for _, n, _, sp, *_ in items if sp == s and (s != "train" or n in m4)} for s in SPLITS}
    lines, errors = ["== handoff_2차 =="], []
    top = sorted(os.listdir(HANDOFF))
    if top != ["data.yaml", "images", "labels"]:
        errors.append(f"handoff_2차 맨 위에 다른 것이 있음: {top}")
    for s in SPLITS:
        imgs = set(os.listdir(os.path.join(HANDOFF, "images", s)))
        labs = {f[:-4] + ".jpg" for f in os.listdir(os.path.join(HANDOFF, "labels", s))}
        lines.append(f"  {s:<5}: 사진 {len(imgs):>3}  라벨 {len(labs):>3}  (기대 {len(want[s])})")
        if not (imgs == labs == want[s]):
            errors.append(f"handoff_2차 {s} 개수/이름 불일치")
    lines += ["  문제: " + e for e in errors] or ["  확인 통과"]
    return lines, errors


def main():
    ap = argparse.ArgumentParser(description="2차 학습용 데이터 나누기")
    ap.add_argument("--overwrite", action="store_true", help="만든 결과물을 지우고 다시 만들기")
    ap.add_argument("--check", action="store_true", help="만들지 않고 확인만")
    a = ap.parse_args()

    items = collect()
    if not a.check:
        made = [p for p in (OUT, ZIP, HANDOFF) if os.path.exists(p)]
        if made and not a.overwrite:
            print("이미 있음: " + ", ".join(made) + "\n다시 만들려면 --overwrite")
            return 1
        for p in made:  # 이 스크립트가 만든 결과물만 지운다(원본 data_*, labels_*는 건드리지 않음)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        build(items)

    lines, errors = check(items)
    if os.path.exists(HANDOFF) or not a.check:
        if not a.check:
            make_handoff(items)
        hl, he = check_handoff(items)
        lines += [""] + hl
        errors += he
    write(os.path.join(OUT, "확인결과.txt"), "\n".join(lines) + "\n")
    if not a.check:
        make_zip()  # 확인결과.txt까지 넣고 압축
        lines.append(f"\n압축: {ZIP} ({os.path.getsize(ZIP) / 1e6:.1f} MB)")
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
