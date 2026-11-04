"""PARKIT 점유 판정 — 합성 시험 (28 문서 6절, 26 문서 계산값과 비교)

11 문서 배치로 가상 카메라를 만들고, 기준점을 찍은 것처럼 만들어(0.5픽셀 잡음) 실제 보정 계산
(build_camera_config)을 거친 뒤, 가상 차 상자를 판정 코드(judge_frame)에 넣어 본다.

사용법 (docs\\도구\\occupancy 에서):
    python test_geometry.py            모든 시험
    python test_geometry.py -v         틀린 경우를 하나하나 보여 준다
끝에 통과/실패 수를 내고, 하나라도 실패하면 종료 코드 1.
"""

import argparse, os, sys

import numpy as np

from parkit_config import DEFAULT_RUN, HERE, SLOT_TABLE_CANDIDATES, Params, resolve_first
from parkit_geom import CamGeom, ResolutionMismatch, SlotTable, apply_h, build_camera_config, contact_px
from parkit_judge import ABSENT, OCCLUDED, PRESENT, judge_frame
from parkit_synth import VirtualCamera, cam1, cam2, cam_diag, pick_refs, visible_boxes

TABLE = resolve_first(DEFAULT_RUN, SLOT_TABLE_CANDIDATES)[0]     # docs 원본, 없으면 data/ 복사본
RESULTS = []
VERBOSE = False


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def calibrate(vcam, table, params, name="CAM", size=None, edited=None, extra_refs=()):
    ids = pick_refs(table, vcam)
    refs = vcam.ref_clicks(table, list(ids) + list(extra_refs), noise_px=0.5, seed=1)
    return build_camera_config(name, 0, vcam.size, refs, table, params, 50.0, edited=edited)


def present_slots(res):
    return sorted(n for n, o in res.obs.items() if o == PRESENT)


def occluded_slots(res):
    return sorted(n for n, o in res.obs.items() if o == OCCLUDED)


def sweep(vcam, geom, table, params, rows, label, use_expected=True):
    """26 문서 5절: 차 폭 × 길이 × 좌우 × 깊이를 바꿔 가며 칸마다 넣는다."""
    if not use_expected:
        saved = dict(geom.expected)
        for no in geom.expected:
            sl = table[no]
            geom.expected[no] = np.array([sl.center[0], sl.entrance_y])
    total = good = 0
    bad = []
    for no in sorted(geom.monitored):
        sl = table[no]
        if sl.row not in rows:
            continue
        for w in (2.8, 3.2, 3.6):
            for l in (6.0, 6.5, 7.0):
                for dx in (-1.0, 0.0, 1.0):
                    for dy in (0.0, 0.5):
                        b = vcam.car_in_slot(sl, w, l, 2.0, dx, dy)
                        total += 1
                        if b is None:
                            bad.append((no, w, l, dx, dy, "상자 없음"))
                            continue
                        r = judge_frame(geom, params, [b + (0.9,)])
                        got = present_slots(r)
                        if got == [no] and not occluded_slots(r):
                            good += 1
                        else:
                            bad.append((no, w, l, dx, dy, got))
    if not use_expected:
        geom.expected.update(saved)
    if VERBOSE and bad and use_expected:
        for x in bad[:20]:
            print("     틀림:", x)
    return good, total, bad


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    VERBOSE = ap.parse_args().v

    table = SlotTable(TABLE)
    params = Params({}, table)
    c1, c2, cd = cam1(), cam2(), cam_diag()
    cfg1 = calibrate(c1, table, params, "CAM-1")
    cfg2 = calibrate(c2, table, params, "CAM-2")
    cfgd = calibrate(cd, table, params, "DIAG")
    g1 = CamGeom(cfg1, table, c1.size, params)
    gd = CamGeom(cfgd, table, cd.size, params)

    print("\n1. 보정 계산 (기준점 8개, 잡음 0.5픽셀)")
    for nm, cfg, true in (("CAM-1", cfg1, (51, -8, 15)), ("CAM-2", cfg2, (23, -8, 15)), ("대각선", cfgd, (86, -8, 15))):
        pos = cfg["pose"]["cam_pos_cm"]
        e = cfg["homography_err_cm"]["max"]
        d = float(np.linalg.norm(np.array(pos) - true))
        check(f"{nm} 기준점 오차 {e:.3f}cm < 0.5, 렌즈 위치 {pos} (참값 {true}, 차이 {d:.2f}cm)", e < 0.5 and d < 1.0)

    print("\n2. 감시 칸 자동 선택 (11 문서 1절)")
    want1 = list(range(1, 8)) + list(range(34, 42))
    want2 = list(range(8, 15)) + list(range(28, 36))
    check(f"CAM-1 = 1~7, 34~41번", cfg1["monitored"] == want1, str(cfg1["monitored"]))
    check(f"CAM-2 = 8~14, 28~35번", cfg2["monitored"] == want2, str(cfg2["monitored"]))
    need = set(range(1, 15)) | set(range(28, 42))
    check("대각선: 1~14번, 28~41번 모두 보임", need <= set(cfgd["monitored"]))
    gd.monitored = need      # 시험은 두 줄만 감시

    print("\n3. P2·P2b 칸 고르기 (26 문서 5절 표: 차 폭 2.8/3.2/3.6 × 길이 6.0/6.5/7.0 × 좌우 ±1 × 깊이 0/0.5)")
    for nm, vc, g, rows, want in (("대각선", cd, gd, ("1-14",), 756), ("대각선", cd, gd, ("28-41",), 756),
                                  ("정면 CAM-1", c1, g1, ("1-14",), 378), ("정면 CAM-1", c1, g1, ("28-41",), 432)):
        good, total, _ = sweep(vc, g, table, params, rows, nm)
        check(f"{nm} {rows[0]}번 줄: {good} / {total} (문서 {want} / {want})", good == total == want)
    good, total, _ = sweep(cd, gd, table, params, ("1-14",), "대각선", use_expected=False)
    print(f"  (참고) 대각선에서 P2b를 끄고 칸 입구 가운데를 쓰면 1~14번 줄 {good} / {total} — 26 문서 5절 \"거의 모든 칸이 옆 칸\"")

    print("\n4. P3 두 칸 걸침 (26 문서 2.2절, 4·5번 칸 경계)")
    sl4 = table[4]
    for cx, want in ((49.34, [4, 5]), (48.84, [5]), (47.33, [5])):
        b = c1.box(cx - 1.6, cx + 1.6, sl4.center[1] - 3.25, sl4.center[1] + 3.25, 2.0)
        r = judge_frame(g1, params, [b + (0.9,)])
        cxm = apply_h(g1.H, [contact_px(b)])[0][0]
        check(f"접점 x {cxm:.2f} → {want}", present_slots(r) == want, f"결과 {present_slots(r)}")

    print("\n5. P5 가림 비율 (26 문서 3.3절, 4번 칸 앞 통로에 높이 2cm 차, 4번 칸은 비어 있음)")
    for y0, y1, dx, want, occ in ((16.3, 19.5, 0, 42, True), (15.5, 18.7, 0, 33, True),
                                  (13.0, 16.2, 0, 3, False), (15.5, 18.7, -3.35, 16, False)):
        cx = sl4.center[0] + dx
        b = c1.box(cx - 3.25, cx + 3.25, y0, y1, 2.0)
        r = judge_frame(g1, params, [b + (0.9,)])
        cov = r.cover[4] * 100
        check(f"y {y0}~{y1}, 좌우 {dx:+}: 덮는 비율 {cov:.1f}% (문서 {want}%), 4번 = {r.obs[4]}",
              abs(cov - want) <= 2.0 and r.obs[4] == (OCCLUDED if occ else ABSENT) and not present_slots(r))

    print("\n6. P2a 아래가 잘린 상자 (26 문서 1.6절, 4번 칸 차 + 칸 앞 통로 차)")
    doc = {(2, 18.7): 22.8, (2, 19.5): 23.7, (3, 18.7): 25.4, (3, 19.5): 26.4,
           (4, 18.7): 28.4, (4, 19.5): 29.5, (5, 18.7): 32.1, (5, 19.5): 33.3}
    cx, cy = sl4.center
    parked = (cx - 1.6, cx + 1.6, cy - 3.25, cy + 3.25, 2.0)
    for hb in (2, 3, 4, 5):
        for back in (18.7, 19.5):
            aisle = (cx - 3.25, cx + 3.25, back - 3.2, back, float(hb))
            boxes = visible_boxes(c1, [parked, aisle])
            r = judge_frame(g1, params, boxes)
            cut = [b for b in r.boxes if b.cut_by is not None]
            # 큰 통로 차가 이웃 3·5번 칸까지 30% 넘게 덮으면 그 칸들이 가려짐이 되는 것은 P5대로 맞다
            if hb <= 4:
                yc = cut[0].contact_mat[1] if cut else float("nan")
                ok = present_slots(r) == [4] and abs(yc - doc[(hb, back)]) <= 0.3
                check(f"가리는 차 높이 {hb}cm, 뒤 끝 y {back}: 잘린 점 y {yc:.2f} (문서 {doc[(hb, back)]}) → 4번",
                      ok, f"차 있음 {present_slots(r)}, 가려짐 {occluded_slots(r)}")
            else:
                # 이 자리의 5cm 차는 2cm 차를 완전히 가린다(상자가 없음) → P5가 4번을 가려짐으로 둔다
                ok = not present_slots(r) and 4 in occluded_slots(r) and not cut
                check(f"가리는 차 높이 5cm, 뒤 끝 y {back}: 4번 차가 완전히 가려져 상자 없음 → P5로 4번 가려짐",
                      ok, f"차 있음 {present_slots(r)}, 가려짐 {occluded_slots(r)}")
                # 26 문서처럼 윗부분만 보여 아래 변이 가리는 차 지붕에 걸린 상자가 잡혔다고 넣어 본다
                ab = c1.box(*parked)
                tb = c1.box(*aisle)
                yp = g1.to_px([(cx, doc[(hb, back)])])[0][1]
                hyp = (ab[0], min(ab[1], yp - 40), ab[2], yp, 0.9)
                r = judge_frame(g1, params, [hyp, tb + (0.9,)])
                ok = not present_slots(r) and {4, 38} <= set(occluded_slots(r))
                check(f"  같은 장면에서 아래 변이 y {doc[(hb, back)]}에 걸린 상자가 잡히면 P2a로 4·38번 가려짐",
                      ok, f"차 있음 {present_slots(r)}, 가려짐 {occluded_slots(r)}")
    # 4.5cm 차: 2cm 차의 윗부분이 보이고 잘린 점이 뒷줄 판정 구역까지 간다 → P2a 여러 칸
    aisle = (cx - 3.25, cx + 3.25, 16.3, 19.5, 4.5)
    r = judge_frame(g1, params, visible_boxes(c1, [parked, aisle]))
    cut = [b for b in r.boxes if b.cut_by is not None]
    yc = cut[0].contact_mat[1] if cut else float("nan")
    check(f"가리는 차 높이 4.5cm, 뒤 끝 y 19.5: 잘린 점 y {yc:.2f} → P2a로 4·38번 가려짐",
          bool(cut) and not present_slots(r) and {4, 38} <= set(occluded_slots(r)),
          f"차 있음 {present_slots(r)}, 가려짐 {occluded_slots(r)}")
    # 앞줄 주차 차가 뒷줄 차의 아래를 가릴 때 (26 문서 1.6절: 31.0 → 31.6)
    sl38 = table[38]
    back_car = (sl38.center[0] - 1.6, sl38.center[0] + 1.6, sl38.center[1] - 3.25, sl38.center[1] + 3.25, 2.0)
    boxes = visible_boxes(c1, [parked, back_car])
    r = judge_frame(g1, params, boxes)
    cut = [b for b in r.boxes if b.cut_by is not None]
    yc = cut[0].contact_mat[1] if cut else float("nan")
    check(f"4번 차가 38번 차를 가림: 잘린 점 y {yc:.2f} (문서 31.6) → 4·38번", present_slots(r) == [4, 38],
          f"{present_slots(r)}")
    # 통로 차가 가렸는데 YOLO가 4번 차를 아예 못 잡은 경우는 5번 시험(P5)이 맡는다

    print("\n7. 통로 차 (P4)")
    bad = []
    for x in np.arange(10, 70, 0.5):
        for y0 in (12.8, 14.5, 16.0):
            b = c1.box(x - 3.25, x + 3.25, y0, y0 + 3.2, 2.0)
            if b is None:
                continue
            r = judge_frame(g1, params, [b + (0.9,)])
            if present_slots(r):
                bad.append((x, y0, present_slots(r)))
    check("통로 A에 가로로 선 차(x 10~70, 0.5cm 간격)는 어느 칸도 '차 있음'으로 만들지 않음", not bad,
          f"틀린 경우 {len(bad)}개 {bad[:3]}")

    print("\n8. C6 해상도 (1920×1080으로 보정, 1280×720으로 실행)")
    c1s = c1.scaled((1280, 720))
    g1s = CamGeom(cfg1, table, (1280, 720), params)
    same = True
    for no in sorted(g1.monitored):
        sl = table[no]
        for dx in (-1.0, 0.0, 1.0):
            a = judge_frame(g1, params, [c1.car_in_slot(sl, dx=dx) + (0.9,)])
            b = judge_frame(g1s, params, [c1s.car_in_slot(sl, dx=dx) + (0.9,)])
            if a.obs != b.obs:
                same = False
    for hb in (2, 5):
        aisle = (cx - 3.25, cx + 3.25, 15.5, 18.7, float(hb))
        a = judge_frame(g1, params, visible_boxes(c1, [parked, aisle]))
        b = judge_frame(g1s, params, visible_boxes(c1s, [parked, aisle]))
        same &= a.obs == b.obs
    for y0, y1 in ((16.3, 19.5), (15.5, 18.7)):
        a = judge_frame(g1, params, [c1.box(cx - 3.25, cx + 3.25, y0, y1, 2.0) + (0.9,)])
        b = judge_frame(g1s, params, [c1s.box(cx - 3.25, cx + 3.25, y0, y1, 2.0) + (0.9,)])
        same &= a.obs == b.obs and abs(a.cover[4] - b.cover[4]) < 0.01
    check("칸 넣기, 잘린 상자, 가림 판정이 1920×1080과 같음", same)
    try:
        CamGeom(cfg1, table, (640, 480), params)
        check("640×480(4:3)은 다시 보정하라고 알리고 멈춤", False)
    except ResolutionMismatch as e:
        check("640×480(4:3)은 다시 보정하라고 알리고 멈춤", True, str(e)[:60] + "…")

    print("\n9. 칸 꼭짓점 손으로 고치기 (22 문서 3.3절: 3번 칸만 인쇄가 x +1.5cm 밀렸다고 가정)")
    sl3 = table[3]
    shift = np.array([1.5, 0.0])
    real_inner = sl3.inner + shift                     # ③은 빈 공간(inner) 네 모서리를 고친다
    uv, _ = c1.project(np.c_[real_inner, np.zeros(4)])
    cfg_e = calibrate(c1, table, params, "CAM-1", edited={3: uv.tolist()})
    g_e = CamGeom(cfg_e, table, c1.size, params)
    res_plain, res_edit = [], []
    for dx in (-1.0, 0.0, 1.0):
        cxr = sl3.center[0] + shift[0] + dx
        b = c1.box(cxr - 1.6, cxr + 1.6, sl3.center[1] - 3.25, sl3.center[1] + 3.25, 2.0)
        res_plain.append(present_slots(judge_frame(g1, params, [b + (0.9,)])))
        res_edit.append(present_slots(judge_frame(g_e, params, [b + (0.9,)])))
    print(f"  고치기 전: {res_plain}")
    check(f"고친 뒤 실제 3번 칸 자리의 차(좌우 -1/0/+1)가 모두 3번", all(r == [3] for r in res_edit), str(res_edit))
    r4 = present_slots(judge_frame(g_e, params, [c1.car_in_slot(table[4]) + (0.9,)]))
    r2 = present_slots(judge_frame(g_e, params, [c1.car_in_slot(table[2]) + (0.9,)]))
    check("이웃 2·4번 칸 판정은 그대로", r4 == [4] and r2 == [2], f"{r2}, {r4}")

    print("\n10. P8 칸·줄별 값 덮어쓰기")
    p8 = Params({"params": {"occlusion_ratio": 0.35}, "overrides": {"rows": {"28-41": {"zone_margin_back_cm": 1.5}},
                                                                    "slots": {"34": {"car_width_cm": 3.0}}}}, table)
    ok = (p8(1)["occlusion_ratio"] == 0.35 and p8(40)["zone_margin_back_cm"] == 1.5 and
          p8(4)["zone_margin_back_cm"] == 1.0 and p8(34)["car_width_cm"] == 3.0 and p8(35)["car_width_cm"] == 3.2)
    check("공통 → 줄별 → 칸별 순서로 덮어씀", ok)

    print("\n11. 카메라 방향과 높이 (반대쪽, 낮게 보아 지평선이 화면에 보임, 반대쪽 대각선)")
    others = (("반대쪽(42~56번 줄 쪽) 정면", VirtualCamera((40, 63, 15), yaw_dir=(0, -1))),
              ("반대쪽 낮게(높이 6, 아래로 8°, 지평선 보임)", VirtualCamera((40, 63, 6), yaw_dir=(0, -1), pitch=8)),
              ("15~27번 쪽 낮게(높이 6, 아래로 8°, 지평선 보임)", VirtualCamera((51, -8, 6), pitch=8)),
              ("반대쪽 모서리 대각선", VirtualCamera((-6, 63, 15), look=(30, 28))))
    for nm, vc in others:
        cfg_o = calibrate(vc, table, params, "O")
        g_o = CamGeom(cfg_o, table, vc.size, params)
        pos = np.array(cfg_o["pose"]["cam_pos_cm"]) if cfg_o["pose"] else np.array([np.nan] * 3)
        true = np.array([vc.C[0], vc.C[1], -vc.C[2]])
        mon = [n for n in g_o.monitored if table[n].row in ("1-14", "28-41")]
        good = total = 0
        for no in mon:
            sl = table[no]
            for w, l, dx, dy in ((3.2, 6.5, 0.0, 0.0), (2.8, 6.0, -1.0, 0.5), (3.6, 7.0, 1.0, 0.0)):
                b = vc.car_in_slot(sl, w, l, 2.0, dx, dy)
                if b is None:
                    continue
                total += 1
                r = judge_frame(g_o, params, [b + (0.9,)])
                good += present_slots(r) == [no] and not occluded_slots(r)
        check(f"{nm}: 렌즈 위치 {np.round(pos, 1)} (참값 {np.round(true, 1)}), 칸 {len(g_o.slot_H)}/56, "
              f"감시 칸 {len(cfg_o['monitored'])}개, 1~14·28~41번 감시 칸 판정 {good}/{total}",
              np.linalg.norm(pos - true) < 1.0 and len(mon) > 0 and good == total)
        if "지평선" in nm:
            # 지평선 위(벽 쪽)에 잡힌 상자는 칸에 넣지 않는다
            sky = (900.0, 5.0, 1000.0, 40.0, 0.9)
            r = judge_frame(g_o, params, [sky])
            check(f"  {nm}: 지평선 위 상자는 칸에 들어가지 않음", not present_slots(r), str(present_slots(r)))

    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
