"""PARKIT 점유 판정 — 동료에게 보낼 묶음 만들기

occupancy 폴더에서 보낼 것만 골라 새 폴더와 zip을 만든다. 받은 사람은 zip을 풀고 occupancy 폴더에서 바로 실행한다
(설치는 README.md, 자세한 사용법은 28_점유판정_프로그램_사용법.md).

넣는 것  코드(*.py), requirements.txt, README.md, 28 문서, config(run.json, run_2대.json, cam_*.json과 확인 사진),
         data/칸표_매트좌표_초안.json, models/M8.pt, 동료의 PARKIT Tiny 모델(models/mymodel_*.pt + model.py),
         --models 로 고른 모델, 빈 logs 폴더
빼는 것  logs 안 기록, 이전/, __pycache__/, 녹화 영상(*.mp4 등)
칸 표, 모델, 28 문서는 저장소 안이면 원본(docs)에서 새로 복사하고, occupancy 폴더 안 복사본(data/, models/, 28 문서)도 같이 맞춘다.

사용법 (docs\\도구\\occupancy 에서):
    python make_handoff.py                         Parkit/dataset/handoff_점유판정_<날짜>/ 와 .zip
    python make_handoff.py --models M7.pt best.pt  모델을 더 넣기 (occupancy/models, docs/도구/models, docs/도구 에서 찾음)
    python make_handoff.py --no-cam                보정 파일(cam_*.json) 빼기 (받는 쪽이 새로 보정할 때)
    python make_handoff.py --no-tiny               PARKIT Tiny 모델(mymodel_*.pt, model.py) 빼기
    python make_handoff.py --out D:\\보낼곳         다른 곳에 만들기
    python make_handoff.py --no-zip                zip 없이 폴더만
"""

import argparse, datetime, glob, os, shutil, sys

from parkit_config import CONFIG_DIR, DATA_DIR, DEFAULT_RUN, HERE, MODEL_DIR, MODEL_CANDIDATES, SLOT_TABLE_CANDIDATES, \
    resolve_first

TOOLS_DIR = os.path.dirname(HERE)                                   # 저장소 안이면 docs/도구
DOCS_DIR = os.path.dirname(TOOLS_DIR)                               # 저장소 안이면 docs
REPO_DIR = os.path.dirname(DOCS_DIR)                                # 저장소 안이면 Parkit
DOC_NAME = "28_점유판정_프로그램_사용법.md"
TABLE_NAME = "칸표_매트좌표_초안.json"


def find_model(name):
    for d in (MODEL_DIR, os.path.join(TOOLS_DIR, "models"), TOOLS_DIR):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def fresh(original, local):
    """원본이 있으면 폴더 안 복사본을 원본으로 맞추고 그 경로를, 없으면 복사본 경로를 돌려준다."""
    if original and os.path.isfile(original):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if not os.path.isfile(local) or open(original, "rb").read() != open(local, "rb").read():
            shutil.copy2(original, local)
            print(f"  폴더 안 복사본을 원본으로 맞춤: {os.path.relpath(local, HERE)}")
        return original
    return local if os.path.isfile(local) else None


def main():
    ap = argparse.ArgumentParser(description="PARKIT 점유 판정 — 동료에게 보낼 묶음 만들기")
    ap.add_argument("--out", default=None, help="만들 곳 (기본: Parkit/dataset, 저장소 밖이면 occupancy 옆)")
    ap.add_argument("--models", nargs="*", default=[], help="더 넣을 모델 파일 이름 (예: M7.pt best.pt)")
    ap.add_argument("--no-cam", action="store_true", help="보정 파일(cam_*.json, 확인 사진) 빼기")
    ap.add_argument("--no-zip", action="store_true", help="zip 만들지 않기")
    ap.add_argument("--no-tiny", action="store_true", help="PARKIT Tiny 모델(mymodel_*.pt, model.py) 빼기")
    args = ap.parse_args()

    stamp = datetime.date.today().strftime("%Y%m%d")
    base = args.out or (os.path.join(REPO_DIR, "dataset") if os.path.isdir(os.path.join(REPO_DIR, "dataset"))
                        else os.path.dirname(HERE))
    pack = os.path.join(base, f"handoff_점유판정_{stamp}")
    dst = os.path.join(pack, "occupancy")
    if os.path.exists(pack):
        shutil.rmtree(pack)
    os.makedirs(dst)
    print(f"[묶음] {dst}")

    # 칸 표, 28 문서, 모델: 원본에서 (폴더 안 복사본도 맞춤)
    table_orig = resolve_first(DEFAULT_RUN, SLOT_TABLE_CANDIDATES[:1])[0]
    table = fresh(table_orig, os.path.join(DATA_DIR, TABLE_NAME))
    doc = fresh(os.path.join(DOCS_DIR, DOC_NAME), os.path.join(HERE, DOC_NAME))
    model_orig = resolve_first(DEFAULT_RUN, MODEL_CANDIDATES[:1])[0]
    m8 = fresh(model_orig, os.path.join(MODEL_DIR, "M8.pt"))
    if table is None or m8 is None:
        sys.exit(f"칸 표 또는 M8.pt를 찾지 못했습니다: {table}, {m8}")

    copied = []

    def put(src, rel):
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        copied.append((rel, os.path.getsize(out)))

    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        put(p, os.path.basename(p))
    for name in ("requirements.txt", "README.md"):
        if os.path.isfile(os.path.join(HERE, name)):
            put(os.path.join(HERE, name), name)
    if doc:
        put(doc, DOC_NAME)
    for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "run*.json"))):
        put(p, os.path.join("config", os.path.basename(p)))
    if not args.no_cam:
        for p in sorted(glob.glob(os.path.join(CONFIG_DIR, "cam_*"))):
            put(p, os.path.join("config", os.path.basename(p)))
    put(table, os.path.join("data", TABLE_NAME))
    put(m8, os.path.join("models", "M8.pt"))
    if not args.no_tiny:
        # 동료가 만든 PARKIT Tiny: 가중치와 구조 파일(model.py)을 models/ 에 나란히 넣는다
        tiny = sorted({os.path.basename(p) for d in (MODEL_DIR, os.path.join(TOOLS_DIR, "models"))
                       for p in glob.glob(os.path.join(d, "mymodel*.pt"))})
        model_py = next((p for p in (os.path.join(MODEL_DIR, "model.py"), os.path.join(TOOLS_DIR, "model.py"))
                         if os.path.isfile(p)), None)
        if tiny and model_py is None:
            print("  [주의] model.py가 없어 PARKIT Tiny 모델을 뺐습니다.")
        elif tiny:
            put(model_py, os.path.join("models", "model.py"))
            args.models = list(args.models) + [t for t in tiny if t not in args.models]
    for name in args.models:
        p = find_model(name)
        if p is None:
            print(f"  [주의] 모델을 찾지 못해 뺐습니다: {name}")
        elif os.path.basename(p) != "M8.pt":
            put(p, os.path.join("models", os.path.basename(p)))
    os.makedirs(os.path.join(dst, "logs"))
    open(os.path.join(dst, "logs", "여기에_실행_기록이_생깁니다.txt"), "w", encoding="utf-8").write(
        "occupancy.py를 실행할 때마다 이 폴더에 <날짜_시각> 폴더가 생긴다(28 문서 4.5절).\n")

    total = sum(n for _, n in copied)
    for rel, n in copied:
        print(f"  {rel.replace(os.sep, '/'):45s} {n / 1024:9.1f} KB")
    print(f"[묶음] 파일 {len(copied)}개, {total / 1024 / 1024:.1f} MB")
    left = [os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*")) if os.path.splitext(p)[1].lower() in
            (".mp4", ".avi", ".mov", ".mkv")]
    print("[묶음] 뺀 것: logs 기록, 이전/, __pycache__/" + (f", 영상 {', '.join(left)}" if left else ""))
    if not args.no_zip:
        z = shutil.make_archive(pack, "zip", root_dir=pack, base_dir="occupancy")
        print(f"[묶음] zip: {z} ({os.path.getsize(z) / 1024 / 1024:.1f} MB)")
    print("[묶음] 받는 쪽: zip을 풀고 occupancy/README.md 대로 설치·실행")


if __name__ == "__main__":
    sys.exit(main())
