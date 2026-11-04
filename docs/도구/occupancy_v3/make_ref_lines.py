"""PARKIT 보정 ver2 — 이 매트의 기준선 목록 만들기 (칸 표 → data/기준선_매트.json)

기준점 자동 찾기(parkit_autoref.py)는 주차장마다 다른 것을 모두 "기준선 목록"으로 받는다.
이 파일은 1:64 주차장 매트 전용이다. 매트 그림의 모양(U자 선, 카스토퍼)에 관한 숫자는 이 파일과 결과 json에만 둔다.

이 매트의 기준선 (docs/31 문서 4절)
    칸 옆   : U자 선 다리의 칸 쪽 가장자리 (edge). 칸마다 양옆 두 개. 바닥 쪽 = 칸 안쪽
    입구    : 한 묶음의 U자 둥근 끝을 이은 가상선 (tip_row). 기둥에서 묶음을 끊는다. 끝 방향 = 통로 쪽
    안쪽    : 카스토퍼의 칸 쪽 면 (edge). 칸마다 짧은 선분
    기준점  : 칸 옆 선과 입구 선의 교점 = IE-, IE+ / 칸 옆 선과 안쪽 선의 교점 = IB-, IB+  (칸 표의 inner 네 모서리)
    기둥 옆, 줄 끝(반쪽 U자)처럼 칠한 모양이 다른 칸 옆은 기준선을 적지 않는다(실제 사진으로 확인 전).

사용법 (docs\\도구\\occupancy 에서):
    python make_ref_lines.py                       data/칸표_매트좌표_초안.json → data/기준선_매트.json
    python make_ref_lines.py --table 다른칸표.json --out data/기준선_다른.json
"""

import argparse, json, os, sys

from parkit_config import DATA_DIR, DEFAULT_RUN, SLOT_TABLE_CANDIDATES, resolve_first
from parkit_geom import SlotTable

# 이 매트의 그림 모양 (cm). 실제 사진으로 확인해서 고친다.
MAT_SHAPE = {
    "leg_width": 0.25,       # U자 선 다리 한 줄의 폭 (U자 전체 폭 0.9cm, 가운데는 비어 있음)
    "castor_margin": 0.4,    # 카스토퍼가 칸 빈 공간 양옆에서 떨어진 거리
    "castor_width": 0.3,     # 카스토퍼 막대의 두께 (깊이 방향)
}


def groups_of(table, row):
    """한 줄의 칸을 x 순서로, 기둥(맞닿지 않은 곳)에서 끊어 묶음으로 나눈다."""
    nos = table.rows[row]
    out, cur = [], [nos[0]]
    for a, b in zip(nos, nos[1:]):
        if table[a].right == b:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def build(table, shape=MAT_SHAPE):
    lines, points = [], []
    for row in sorted(table.rows, key=lambda r: min(table[n].y0 for n in table.rows[r])):
        for gi, grp in enumerate(groups_of(table, row)):
            s0, s1 = table[grp[0]], table[grp[-1]]
            ent = s0.entrance_y
            inward = 1.0 if s0.entrance_low else -1.0          # 입구에서 칸 안쪽으로 가는 y 방향
            tid = f"T{row}-{gi + 1}"
            lines.append({"id": tid, "kind": "tip_row", "p1": [s0.x0, ent], "p2": [s1.x1, ent],
                          "tip_dir": [0.0, -inward], "note": f"{row}번 줄 {grp[0]}~{grp[-1]}번 U자 둥근 끝"})
            for no in grp:
                sl = table[no]
                back = sl.y1 if sl.entrance_low else sl.y0
                sides = []
                if sl.left is not None and sl.left in grp:
                    sides.append(("L", sl.ix0, [1.0, 0.0], "IE-", "IB-"))
                if sl.right is not None and sl.right in grp:
                    sides.append(("R", sl.ix1, [-1.0, 0.0], "IE+", "IB+"))
                bid = f"{no}-B"
                lines.append({"id": bid, "kind": "edge", "p1": [sl.ix0 + shape["castor_margin"], back],
                              "p2": [sl.ix1 - shape["castor_margin"], back], "floor_dir": [0.0, -inward],
                              "width": shape["castor_width"], "note": f"{no}번 카스토퍼 칸 쪽 면"})
                for code, x, fdir, ie, ib in sides:
                    lid = f"{no}-{code}"
                    lines.append({"id": lid, "kind": "edge", "p1": [x, ent], "p2": [x, back], "floor_dir": fdir,
                                  "width": shape["leg_width"], "note": f"{no}번 칸 {'왼' if code == 'L' else '오른'}쪽 U자 다리"})
                    points.append({"id": f"{no}-{ie}", "a": lid, "b": tid, "mat": [x, ent]})
                    points.append({"id": f"{no}-{ib}", "a": lid, "b": bid, "mat": [x, back]})
    for l in lines:
        l["p1"] = [round(v, 4) for v in l["p1"]]
        l["p2"] = [round(v, 4) for v in l["p2"]]
    for p in points:
        p["mat"] = [round(v, 4) for v in p["mat"]]
    return lines, points


def main():
    ap = argparse.ArgumentParser(description="이 매트의 기준선 목록 만들기")
    ap.add_argument("--table", default=None, help="칸 표 (기본: 실행 설정의 칸 표)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "기준선_매트.json"))
    args = ap.parse_args()
    path = args.table or resolve_first(DEFAULT_RUN, SLOT_TABLE_CANDIDATES)[0]
    table = SlotTable(path)
    lines, points = build(table)
    data = {
        "_설명": {
            "무엇": "1:64 주차장 매트의 기준선 목록. make_ref_lines.py가 칸 표에서 만든다. 손으로 고치지 말고 다시 만든다",
            "단위": "cm, 매트 좌표 (칸 표와 같음)",
            "칸표": os.path.basename(path),
            "모양": MAT_SHAPE,
            "종류": "edge = 칠한 선의 한쪽 가장자리(floor_dir = 바닥 쪽), center = 가는 선 가운데, "
                    "tip_row = 여러 선 끝을 이은 가상선(tip_dir = 끝이 향하는 쪽)",
            "빠진 것": "기둥 옆과 줄 끝(반쪽 U자) 칸 옆은 칠한 모양이 달라 기준선을 적지 않았다",
        },
        "lines": lines,
        "points": points,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    kinds = {}
    for l in lines:
        kinds[l["kind"]] = kinds.get(l["kind"], 0) + 1
    print(f"[기준선] {args.out}: 기준선 {len(lines)}개 {kinds}, 기준점 {len(points)}개 (칸 표 {os.path.basename(path)})")


if __name__ == "__main__":
    sys.exit(main())
