"""
영상에서 뽑은 프레임에 첫 프레임 라벨을 복사하는 스크립트

frames.py로 뽑은 사진은 "영상이름_0000.jpg, 영상이름_0001.jpg ..."처럼
영상마다 같은 앞부분과 번호를 가진다. 이 스크립트는 영상(앞부분)마다
라벨(.txt)이 있는 가장 앞 번호 프레임을 찾아, 그 라벨을 같은 영상의
나머지 프레임 이름으로 복사한다. 이미 라벨이 있는 프레임은 건드리지 않는다.
makesense.ai의 Import Annotations에 쓸 labels.txt(car 한 줄)도 만든다.

사용법:
  python copy_labels.py 사진폴더 라벨폴더
  python copy_labels.py frames labels --dry     (복사하지 않고 할 일만 출력)
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_RE = re.compile(r"^(.*)_(\d+)$")


def group_frames(img_dir):
    groups, skipped = {}, []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append(p.name)
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), p.stem))
    for frames in groups.values():
        frames.sort()
    return groups, skipped


def main():
    ap = argparse.ArgumentParser(description="첫 프레임 라벨을 같은 영상의 나머지 프레임에 복사")
    ap.add_argument("images", help="프레임 사진이 있는 폴더")
    ap.add_argument("labels", help="makesense에서 내보낸 .txt가 있는 폴더 (복사본도 여기에 저장)")
    ap.add_argument("--dry", action="store_true", help="복사하지 않고 할 일만 출력")
    args = ap.parse_args()

    img_dir, lab_dir = Path(args.images), Path(args.labels)
    if not img_dir.is_dir():
        sys.exit(f"사진 폴더가 없습니다: {img_dir}")
    if not lab_dir.is_dir():
        sys.exit(f"라벨 폴더가 없습니다: {lab_dir}")

    groups, skipped = group_frames(img_dir)
    if not groups:
        sys.exit("'이름_번호' 형식의 사진이 없습니다.")

    total = 0
    for prefix, frames in groups.items():
        src = next((lab_dir / f"{stem}.txt" for _, stem in frames
                    if (lab_dir / f"{stem}.txt").is_file()), None)
        if src is None:
            print(f"[{prefix}] 라벨이 있는 프레임이 없어 건너뜀 ({len(frames)}장)")
            continue
        boxes = sum(1 for line in src.read_text(encoding="utf-8").splitlines() if line.strip())
        todo = [stem for _, stem in frames if not (lab_dir / f"{stem}.txt").is_file()]
        print(f"[{prefix}] {src.name} (상자 {boxes}개) -> {len(todo)}장에 복사, "
              f"이미 라벨 있음 {len(frames) - len(todo)}장")
        if not args.dry:
            for stem in todo:
                shutil.copyfile(src, lab_dir / f"{stem}.txt")
        total += len(todo)

    if not args.dry:
        (lab_dir / "labels.txt").write_text("car\n", encoding="utf-8")
    if skipped:
        print(f"이름이 '이름_번호' 형식이 아니라 건너뛴 사진 {len(skipped)}장: {', '.join(skipped[:5])}"
              + (" ..." if len(skipped) > 5 else ""))
    print(("복사할 파일" if args.dry else "복사한 파일") + f" {total}개"
          + ("" if args.dry else f", labels.txt 저장: {lab_dir / 'labels.txt'}"))


if __name__ == "__main__":
    main()
