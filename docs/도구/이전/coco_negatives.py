"""COCO에서 차 없는 사진을 골라 웹캠 화질로 바꿔 저장하는 프로그램 (PARKIT 오탐지 대책용, 14 문서)

COCO val2017(5,000장) 중 실제 자동차·오토바이·버스·기차·트럭이 없는 사진만 고르고,
책상 위 물건(병, 컵, 키보드, 마우스, 휴대폰, 책, 시계, 가위 등)이 있는 사진을 우선으로 뽑는다.
뽑은 사진은 웹캠처럼 16:9로 자르고, 살짝 흐리게 하고, 잡음을 넣고, JPEG 화질을 낮춰 저장한다.
라벨 파일은 만들지 않는다. YOLO는 라벨이 없는 사진을 "차 없음"으로 학습한다.

설치: live_detect.py용 설치를 했다면 따로 할 것 없음 (opencv-python, numpy만 씀)

사용법:
    python coco_negatives.py                    60장, Parkit\\dataset\\neg_coco에 저장
    python coco_negatives.py --n 100            100장
    python coco_negatives.py --out D:\\neg       저장 폴더 바꾸기
    python coco_negatives.py --seed 2           다른 사진 묶음으로 뽑기 (기본 1)
    python coco_negatives.py --seed 2 --exclude ..\..\dataset\data_neg_coco
                                                그 폴더(또는 출처.csv)에 있는 COCO 번호는 빼고 다른 사진으로 채우기

처음 실행하면 COCO 라벨 묶음(약 46MB)을 저장 폴더의 _cache에 한 번 받고, 사진은 고른 것만 한 장씩 받는다.
같은 폴더에 다시 실행하면 이미 받은 사진은 건너뛴다.
출처: COCO 2017 (cocodataset.org), 라벨 CC BY 4.0, 사진은 Flickr 원저작자 조건. 목록은 출처.csv에 남긴다.
"""

import argparse, csv, os, random, sys, urllib.request, zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "dataset", "neg_coco"))  # docs\도구 → Parkit\dataset\neg_coco
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_URLS = ["http://images.cocodataset.org/val2017/{}.jpg",
              "https://s3.amazonaws.com/images.cocodataset.org/val2017/{}.jpg"]

# COCO 80가지 번호(YOLO 순서)
NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
         "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
         "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
         "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
         "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
         "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
         "toothbrush").split()
VEHICLES = {2, 3, 5, 6, 7}  # car, motorcycle, bus, train, truck: 있으면 뺀다
DESK = {39, 40, 41, 42, 43, 44, 45, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 79}  # 책상 위 물건: 있으면 우선


def fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            f.write(b)
    os.replace(path + ".part", path)


def load_labels(cache):
    """val2017 사진마다 들어 있는 물체 번호 집합을 돌려준다."""
    z = os.path.join(cache, "coco2017labels.zip")
    if not os.path.exists(z):
        print("COCO 라벨 묶음 받는 중 (약 46MB, 한 번만)...")
        fetch(LABELS_URL, z)
    out = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.startswith("coco/labels/val2017/") and n.endswith(".txt"):
                ids = {int(l.split()[0]) for l in zf.read(n).decode().splitlines() if l.strip()}
                out[os.path.basename(n)[:-4]] = ids
    return out


def excluded_ids(paths):
    """--exclude로 받은 폴더(neg_coco_번호.jpg)나 출처.csv에서 COCO 번호를 모은다."""
    ids = set()
    for p in paths:
        if os.path.isdir(p):
            ids |= {f[len("neg_coco_"):-4] for f in os.listdir(p) if f.startswith("neg_coco_") and f.endswith(".jpg")}
        elif os.path.isfile(p):
            with open(p, encoding="utf-8-sig") as f:
                ids |= {r["COCO 번호"] for r in csv.DictReader(f)}
        else:
            raise SystemExit(f"--exclude 경로가 없음: {p}")
    return ids


def pick(labels, n, desk_ratio, seed, exclude=()):
    clean = {k: v for k, v in labels.items() if v and not (v & VEHICLES)}
    desk = sorted(k for k, v in clean.items() if v & DESK)
    other = sorted(k for k in clean if k not in set(desk))
    rng = random.Random(seed)
    rng.shuffle(desk)
    rng.shuffle(other)
    # 섞은 뒤에 빼므로, 겹치는 번호만 빠지고 그 자리는 다음 사진으로 채워진다
    desk = [k for k in desk if k not in exclude]
    other = [k for k in other if k not in exclude]
    nd = min(len(desk), round(n * desk_ratio))
    return desk[:nd] + other[:n - nd], len(clean)


def webcam_like(im, rng):
    """웹캠 사진처럼: 16:9 가운데 자르기 → 1280x720 → 약한 흐림 → 잡음 → (저장 때) 낮은 JPEG 화질."""
    h, w = im.shape[:2]
    ch = min(h, int(w * 9 / 16))
    cw = int(ch * 16 / 9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    im = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (1280, 720), interpolation=cv2.INTER_LINEAR)
    im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.6, 1.2))
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 5), im.shape)
    return np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def save_jpg(path, im, q):
    """한글 경로에서도 저장되도록 imencode + tofile을 쓴다."""
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)
    return ok


def main():
    ap = argparse.ArgumentParser(description="COCO에서 차 없는 사진을 골라 웹캠 화질로 저장")
    ap.add_argument("--n", type=int, default=60, help="뽑을 사진 수 (기본 60)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="저장 폴더 (기본: Parkit\\dataset\\neg_coco)")
    ap.add_argument("--desk", type=float, default=0.7, help="책상 위 물건이 있는 사진 비율 (기본 0.7)")
    ap.add_argument("--seed", type=int, default=1, help="뽑기 번호 (같으면 같은 사진이 뽑힘)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="이 폴더(neg_coco_*.jpg)나 출처.csv에 있는 COCO 번호는 뽑지 않음 (여러 개 가능)")
    a = ap.parse_args()

    cache = os.path.join(a.out, "_cache")
    raw_dir = os.path.join(cache, "원본")
    os.makedirs(raw_dir, exist_ok=True)
    labels = load_labels(cache)
    excl = excluded_ids(a.exclude)
    ids, n_clean = pick(labels, a.n, a.desk, a.seed, excl)
    if excl:
        print(f"--exclude로 뺄 COCO 번호 {len(excl)}개")
    print(f"val2017 {len(labels)}장 중 차 없는 사진 {n_clean}장, 그중 {len(ids)}장을 고름")

    rng = random.Random(a.seed)
    rows, done = [], 0
    for i, k in enumerate(ids, 1):
        dst = os.path.join(a.out, f"neg_coco_{k}.jpg")
        raw = os.path.join(raw_dir, f"{k}.jpg")
        if not os.path.exists(raw):
            for u in IMAGE_URLS:
                try:
                    fetch(u.format(k), raw)
                    break
                except Exception as e:
                    err = e
            else:
                print(f"[{i}/{len(ids)}] {k} 받기 실패: {err}")
                continue
        im = cv2.imdecode(np.fromfile(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            print(f"[{i}/{len(ids)}] {k} 열기 실패")
            continue
        q = rng.randint(60, 75)
        if save_jpg(dst, webcam_like(im, rng), q):
            done += 1
        objs = " ".join(sorted({NAMES[c] for c in labels[k]}))
        rows.append([os.path.basename(dst), k, IMAGE_URLS[0].format(k), objs])
        print(f"[{i}/{len(ids)}] {os.path.basename(dst)}  ({objs})")

    with open(os.path.join(a.out, "출처.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["파일", "COCO 번호", "원본 주소", "사진 속 물체(COCO 라벨)"])
        w.writerows(rows)
    print(f"끝: {done}장 저장 → {a.out}  (원본과 라벨 묶음은 _cache 폴더, 지워도 됨)")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
