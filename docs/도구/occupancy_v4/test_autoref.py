"""PARKIT 보정 ver2 — 기준점 자동 찾기 합성 시험 (docs/31 문서 5절, docs/32 문서)

a. 실제 매트처럼 그린 합성 매트(U자 선 둥근 끝, 떨어진 카스토퍼, 기둥, 칸 번호 글자, 화살표)에서
   참 기준점 4개에 ±8픽셀 잡음을 넣고 자동 찾기 → 모든 기준점 오차 평균 0.15cm, 최대 0.3cm 안, 렌즈 위치 0.5cm 안.
   카메라: 북쪽 정면(11 문서 CAM-1), 남쪽(지금 CAM-1 자리), 대각선
b. a에 차(선 일부를 가림)와 밝은 얼룩(반사광)을 넣어도 같은 기준. 차에 다 가려진 선은 "못 찾음"
c. 일반 주차장(직사각형 칸 10개 = 선 가운데 기준·안쪽 선 없음, 비스듬한 칸 4개)을 같은 코드로. 데이터만 다르다
d. 처음 점 하나를 옆 칸으로 밀려 찍으면 경고. 손 점 넷 중 둘이 틀리면(이름 바뀜, 옆 칸) 틀린 결과를 조용히 내지 않는다
e. 기준선 목록이 틀리면(기준점 좌표 ≠ 두 선 교점, 종류 값 빠짐) 읽을 때 알린다

사용법 (docs\\도구\\occupancy 에서):  python test_autoref.py      (-v: 자세히)
"""

import argparse, os, sys

import numpy as np

from parkit_autoref import RefLines, auto_find
from parkit_config import DATA_DIR, DEFAULT_RUN, SLOT_TABLE_CANDIDATES, Params, resolve_first
from parkit_geom import SlotTable, apply_h, build_camera_config
from parkit_synth_mat import VirtualCamera, camera_view, draw_lot, draw_mat_real

RESULTS = []
VERBOSE = False


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def visible_points(ref, vcam, margin=60):
    W, H = vcam.size
    out = []
    for p in ref.points:
        uv, z = vcam.project([[p["mat"][0], p["mat"][1], 0.0]])
        if z[0] > 0 and margin < uv[0][0] < W - margin and margin < uv[0][1] < H - margin:
            out.append((p, uv[0]))
    return out


def pick_four(vis, min_gap=150):
    """화면에 넓게 흩어진 서로 다른 4점 (사람이 찍듯이: 네 귀퉁이 쪽, 셋이 한 줄에 서지 않게)."""
    arr = np.array([u for _, u in vis])
    mats = np.array([p["mat"] for p, _ in vis])

    def area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2

    idx = []
    for key in (arr[:, 0] + arr[:, 1], -(arr[:, 0] - arr[:, 1]), -(arr[:, 0] + arr[:, 1]), arr[:, 0] - arr[:, 1]):
        for i in np.argsort(key):
            i = int(i)
            if i in idx or any(np.linalg.norm(arr[i] - arr[j]) <= min_gap for j in idx):
                continue
            if len(idx) >= 2 and min(area(mats[idx[a]], mats[idx[b]], mats[i]) for a in range(len(idx))
                                     for b in range(a + 1, len(idx))) < 0.02 * np.ptp(mats, 0).max() ** 2:
                continue                                    # 이미 고른 두 점과 한 줄이면 건너뜀
            idx.append(i)
            break
    return [vis[i] for i in idx]


def run_case(name, vcam, img, ref, params, noise=8.0, seed=5, lim=(0.15, 0.3, 0.5), opts=None, table=None,
             shift_one=None, mutate=None):
    """자동 찾기를 돌리고 (결과, 기준점 오차 목록, 렌즈 위치 오차)를 돌려준다."""
    vis = visible_points(ref, vcam)
    four = pick_four(vis)
    rng = np.random.default_rng(seed)
    manual = [{"id": p["id"], "mat": p["mat"].tolist(), "px": (uv + rng.uniform(-noise, noise, 2)).tolist()} for p, uv in four]
    if shift_one is not None:
        manual[0]["px"] = shift_one(manual[0])
    if mutate is not None:
        manual = mutate(manual)
    res = auto_find(img, ref, manual, vcam.size, opts)
    if not res.ok:
        return res, None, None, manual
    errs = np.array([np.linalg.norm(apply_h(res.H, [uv])[0] - p["mat"]) for p, uv in vis])
    pos_err = None
    if table is not None:
        refs = [m for m in manual if m["id"] not in res.manual_bad] + \
               [{"id": a["id"], "mat": a["mat"], "px": a["px"]} for a in res.used_refs()]
        cam = build_camera_config("T", 0, vcam.size, refs, table, params, 50.0)
        if cam.get("pose"):
            true = np.array([vcam.C[0], vcam.C[1], -vcam.C[2]])
            pos_err = float(np.linalg.norm(np.array(cam["pose"]["cam_pos_cm"]) - true))
    return res, errs, pos_err, manual


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    VERBOSE = ap.parse_args().v

    table = SlotTable(resolve_first(DEFAULT_RUN, SLOT_TABLE_CANDIDATES)[0])
    params = Params({}, table)
    ref = RefLines.load(os.path.join(DATA_DIR, "기준선_매트.json"))
    top = draw_mat_real(table, 30)
    cams = (("북쪽 정면 (11 문서 CAM-1)", VirtualCamera((51, -8, 15))),
            ("남쪽 (지금 CAM-1 자리)", VirtualCamera((33.9, 72.4, 16), yaw_dir=(0, -1), pitch=25)),
            ("대각선", VirtualCamera((86, -8, 15), look=(50, 26))))

    print("\na. 합성 매트, 참 기준점 4개 + ±8픽셀 잡음 (기준: 오차 평균 0.15 / 최대 0.3cm, 렌즈 0.5cm)")
    for nm, vc in cams:
        img = camera_view(vc, top, 30)
        res, errs, pos, manual = run_case(nm, vc, img, ref, params, table=table)
        if errs is None:
            check(f"{nm}: 자동 찾기", False, "; ".join(res.warnings))
            continue
        c = res.counts()
        check(f"{nm}: 손 점 {[m['id'] for m in manual]} → 자동 점 {len(res.used_refs())}개, 선 찾음 {c['found']}/애매 "
              f"{c['ambiguous']}/못 찾음 {c['missing']}, 기준점 {len(errs)}개 오차 평균 {errs.mean():.3f} 최대 {errs.max():.3f}cm, "
              f"렌즈 {pos:.2f}cm", errs.mean() <= 0.15 and errs.max() <= 0.3 and pos is not None and pos <= 0.5)
        if res.warnings:
            print("     경고:", res.warnings)

    print("\nb. 차와 반사광 (34번 칸에 넓은 차를 세워 34번 양옆 선을 다 가림, 다른 칸·통로에도 차, 밝은 얼룩 3곳)")
    cars = [(33.0, 37.5, 30.7, 37.9, 2.2, (40, 40, 150)),            # 34번: 양옆 U자 다리까지 다 가림
            (5.6 + 41.3, 5.6 + 44.5, 19.8, 26.3, 2.0, (150, 60, 30)),   # 5번 칸 근처
            (46.0, 49.2, 46.1, 52.6, 2.0, (30, 140, 30)),            # 48번 칸
            (20.0, 26.5, 41.0, 44.2, 2.0, (120, 120, 40))]           # 통로 B
    glare = [(45.0, 47.5, 0.8), (20.0, 33.0, 0.7), (60.0, 22.0, 0.7)]
    for nm, vc in cams:
        img = camera_view(vc, top, 30, cars=cars, glare=glare, seed=3)
        res, errs, pos, manual = run_case(nm, vc, img, ref, params, table=table, seed=7)
        if errs is None:
            check(f"{nm}: 자동 찾기 (차·반사광)", False, "; ".join(res.warnings))
            continue
        hidden = {lid: res.fits[lid].status for lid in ("34-L", "34-R") if lid in res.fits}
        check(f"{nm}: 자동 점 {len(res.used_refs())}개, 오차 평균 {errs.mean():.3f} 최대 {errs.max():.3f}cm, 렌즈 {pos:.2f}cm, "
              f"가려진 선 {hidden}", errs.mean() <= 0.15 and errs.max() <= 0.3 and pos <= 0.5
              and all(s == "missing" for s in hidden.values()) and hidden)

    print("\nc. 일반 주차장 (같은 코드, 데이터만 다름: 직사각형 칸 10개 선 가운데 기준 + 비스듬한 칸 4개)")
    lot_ref, segs, size = make_lot()
    lot_top = draw_lot(size, segs, 1.0)
    # 같은 기준선 목록을 카메라 두 대가 나눠 본다: 직사각형 칸 구역, 비스듬한 칸 구역 (CCTV 높이 8~9m)
    for nm, vc in (("직사각형 칸 구역", VirtualCamera((1350, -900, 900), look=(1350, 380), hfov=75)),
                   ("비스듬한 칸 구역", VirtualCamera((3500, -700, 800), look=(3750, 380), hfov=70))):
        img = camera_view(vc, lot_top, 1.0, cars=[(520, 730, 130, 570, 150, (60, 60, 160)), (3420, 3600, 180, 560, 150, (150, 70, 40))],
                          glare=[(1600, 350, 40), (3900, 300, 40)])
        res, errs, pos, manual = run_case(nm, vc, img, lot_ref, params, noise=8.0)
        if errs is None:
            check(f"일반 주차장 {nm}", False, "; ".join(res.warnings))
            continue
        # 매트(칸 폭 4cm)의 0.15 / 0.3cm 기준을 칸 폭 250cm에 맞춰 같은 비율로: 평균 9cm, 최대 19cm
        check(f"일반 주차장 {nm}: 자동 점 {len(res.used_refs())}개, 기준점 {len(errs)}개 오차 평균 {errs.mean():.1f} "
              f"최대 {errs.max():.1f}cm (칸 폭 250cm)", errs.mean() <= 9.0 and errs.max() <= 19.0)

    print("\nd. 처음 점 하나를 옆 칸으로 밀려 찍음")
    nm, vc = cams[1]
    img = camera_view(vc, top, 30)

    def shift(m):
        # 같은 이름(id)인데 옆 칸의 같은 자리를 찍었다: 매트 x로 칸 하나(약 4cm) 옆
        uv, _ = vc.project([[m["mat"][0] + 4.03, m["mat"][1], 0.0]])
        return (uv[0] + np.random.default_rng(1).uniform(-3, 3, 2)).tolist()
    res, errs, pos, manual = run_case(nm, vc, img, ref, params, shift_one=shift)
    bad = manual[0]["id"]
    check(f"{nm}: 손 점 {bad}를 옆 칸에 찍음 → 경고", bad in res.manual_bad and any(bad in w for w in res.warnings),
          (res.warnings[0][:70] + "…") if res.warnings else "경고 없음")
    if errs is not None:
        check(f"  그 점을 빼고 계산한 결과는 정상 (오차 평균 {errs.mean():.3f} 최대 {errs.max():.3f}cm)",
              errs.mean() <= 0.15 and errs.max() <= 0.3)

    def swap_two(man):
        # 두 점의 이름을 서로 바꿔 찍음
        man[1]["px"], man[2]["px"] = man[2]["px"], man[1]["px"]
        return man

    def shift_two(man):
        # 두 점을 옆 칸(매트 x로 약 4cm, 서로 반대쪽)에 찍음
        for m, dx in ((man[0], 4.03), (man[2], -4.03)):
            m["px"] = vc.project([[m["mat"][0] + dx, m["mat"][1], 0.0]])[0][0].tolist()
        return man
    for what, fn in (("이름 둘이 서로 바뀜", swap_two), ("둘을 옆 칸에 찍음", shift_two)):
        res, errs, pos, manual = run_case(nm, vc, img, ref, params, mutate=fn)
        if res.ok:
            check(f"{nm}: 손 점 넷 중 {what} → 결과를 낸다면 맞아야 함 (오차 평균 {errs.mean():.3f} 최대 {errs.max():.3f}cm)",
                  errs.mean() <= 0.15 and errs.max() <= 0.3)
        else:
            check(f"{nm}: 손 점 넷 중 {what} → 자동 결과를 쓰지 않고 알림", bool(res.warnings),
                  (res.warnings[-1][:70] + "…") if res.warnings else "알림 없음")

    print("\ne. 기준선 목록 검사")
    base = {"lines": [{"id": "A", "kind": "center", "p1": [0, 0], "p2": [0, 10]},
                      {"id": "B", "kind": "center", "p1": [-5, 5], "p2": [5, 5]}],
            "points": [{"id": "P", "a": "A", "b": "B", "mat": [0, 5]}]}
    try:
        RefLines(base)
        ok1 = True
    except ValueError:
        ok1 = False
    bad1 = dict(base, points=[{"id": "P", "a": "A", "b": "B", "mat": [0, 6]}])
    bad2 = {"lines": [{"id": "E", "kind": "edge", "p1": [0, 0], "p2": [0, 10]}], "points": []}
    errs = []
    for b in (bad1, bad2):
        try:
            RefLines(b)
        except ValueError as e:
            errs.append(str(e).splitlines()[1].strip())
    check("맞는 목록은 읽고, 기준점 좌표가 교점과 다르거나 edge에 floor_dir이 없으면 알림", ok1 and len(errs) == 2,
          " / ".join(errs))

    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return 0 if n_ok == len(RESULTS) else 1


def make_lot():
    """일반 주차장 예 (단위 cm): 직사각형 칸 10개(칸 사이 선 가운데, 안쪽 선 없음, 앞은 선 끝만) + 비스듬한(60°) 칸 4개."""
    w = 12.0
    lines, points, segs = [], [], []
    for i in range(11):                                    # 직사각형 칸 사이 선 (칸 폭 250, 깊이 500)
        x = 100 + i * 250.0
        lines.append({"id": f"S{i}", "kind": "center", "p1": [x, 100], "p2": [x, 600], "width": w})
        segs.append(((x, 100), (x, 600), w))
    lines.append({"id": "BACK", "kind": "center", "p1": [94, 600], "p2": [2606, 600], "width": w})
    segs.append(((94, 600), (2606, 600), w))
    lines.append({"id": "FRONT", "kind": "tip_row", "p1": [100, 100], "p2": [2600, 100], "tip_dir": [0, -1]})
    for i in range(11):
        points.append({"id": f"R{i}-front", "a": f"S{i}", "b": "FRONT", "mat": [100 + i * 250.0, 100]})
        points.append({"id": f"R{i}-back", "a": f"S{i}", "b": "BACK", "mat": [100 + i * 250.0, 600]})
    dx, depth = 500 / np.tan(np.radians(60)), 500.0     # 비스듬한 칸: 앞선·뒷선 사이 60°로 기운 칸 사이 선
    for i in range(5):
        x0 = 3000 + i * 290.0
        p1, p2 = (x0, 100.0), (x0 + dx, 100.0 + depth)
        lines.append({"id": f"A{i}", "kind": "center", "p1": list(p1), "p2": list(p2), "width": w})
        segs.append((p1, p2, w))
    lines.append({"id": "AFRONT", "kind": "center", "p1": [2960, 100], "p2": [4250, 100], "width": w})
    lines.append({"id": "ABACK", "kind": "center", "p1": [3250, 600], "p2": [4540, 600], "width": w})
    segs += [((2960, 100), (4250, 100), w), ((3250, 600), (4540, 600), w)]
    for i in range(5):
        x0 = 3000 + i * 290.0
        points.append({"id": f"A{i}-front", "a": f"A{i}", "b": "AFRONT", "mat": [x0, 100.0]})
        points.append({"id": f"A{i}-back", "a": f"A{i}", "b": "ABACK", "mat": [x0 + dx, 600.0]})
    data = {"lines": lines, "points": points,
            "scan": {"search": [60.0, 30.0, 15.0], "floor_clear": 30.0, "manual_warn": 30.0, "point_tol": 0.5}}
    return RefLines(data), segs, (4700, 800)


if __name__ == "__main__":
    sys.exit(main())
