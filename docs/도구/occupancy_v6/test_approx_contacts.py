"""웹 지도 "위치 대략" 시험 (docs/43).

아래 변이 바닥 접점이 아닌 상자도 대략 위치로 웹 지도에 보내는지 본다.
    - 다른 차에 아래가 가린 차(cut): 자기 시선 위, 가리는 차 뒤 끝을 막 지난 곳. 감시 칸으로 정해졌으면 그 칸 예상 접점
    - 화면 옆 끝에 잘린 차(edge): 접점 그대로(깊이는 맞음). 화면 아래 끝에 잘린 차는 버림
잘린 상자의 원래 접점(앞차 위에 찍힘)은 쓰지 않는다.

실행: python test_approx_contacts.py   (마지막 줄 "합계: N / N 통과")
"""
import sys
from types import SimpleNamespace

import numpy as np

from parkit_config import DEFAULT_RUN, Params, load_run
from parkit_geom import CamGeom, SlotTable, build_camera_config
from parkit_judge import judge_frame
from parkit_synth import VirtualCamera, pick_refs, visible_boxes
from tracking_bridge import TrackingBridge

RESULTS = []
W, L, H = 3.2, 6.5, 2.0


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def live(br):
    return [v for v in br.store.read()["vehicles"] if v["status"] != "REMOVED"]


def run(br, frames, geom=None, dt=0.2):
    """frames: 프레임마다 judge_frame 결과(또는 boxes를 가진 것). 0.2초 간격으로 넣는다."""
    for i, r in enumerate(frames):
        br.submit("A", i * dt, r, geom=geom)
        br.publish(i * dt, {}, {"A": "ok"})
    return live(br)


def main():
    table = SlotTable(load_run(DEFAULT_RUN)["slot_table"])
    params = Params({}, table)

    print("a. 브리지만")
    def box(x, y, **k):
        d = dict(contact_mat=np.array([x, y], float), conf=.9, slot=None, kind="aisle", cut_by=None, edge=False,
                 edge_bottom=False, approx_mat=None)
        d.update(k)
        return SimpleNamespace(**d)
    vs = run(TrackingBridge(table), [SimpleNamespace(boxes=[box(20, 30, cut_by=0, approx_mat=np.array([22.0, 16.0]))])] * 5)
    check("가린 차: 원래 접점(20, 30)이 아니라 대략 위치(22, 16)로, approx=cut",
          len(vs) == 1 and np.allclose((vs[0]["observed_x_cm"], vs[0]["observed_y_cm"]), (22, 16)) and vs[0]["approx"] == "cut")
    br = TrackingBridge(table)
    vs = run(br, [SimpleNamespace(boxes=[box(20, 30, cut_by=0)])] * 5)
    check("대략 위치가 없는 가린 차는 예전처럼 버림", vs == [] and br.rejected_contacts == 5)
    vs = run(TrackingBridge(table), [SimpleNamespace(boxes=[box(5, 16, edge=True)])] * 5)
    check("화면 옆 끝에 잘린 차: 접점 그대로, approx=edge", len(vs) == 1 and vs[0]["approx"] == "edge")
    vs = run(TrackingBridge(table), [SimpleNamespace(boxes=[box(5, 16, edge=True, edge_bottom=True)])] * 5)
    check("화면 아래 끝에 잘린 차는 버림", vs == [])
    vs = run(TrackingBridge(table), [SimpleNamespace(boxes=[box(30, 16)])] * 5)
    check("보통 차는 approx 없음", len(vs) == 1 and vs[0]["approx"] is None)
    # 같은 순간 두 카메라: 정확한 점과 대략 위치가 같이 오면 정확한 점을 쓴다
    br = TrackingBridge(table)
    for i in range(5):
        t = i * 0.2
        br.submit("A", t, SimpleNamespace(boxes=[box(30, 16)]))
        br.submit("B", t, SimpleNamespace(boxes=[box(30.5, 16.5, cut_by=0, approx_mat=np.array([30.5, 16.5]))]))
        br.publish(t, {}, {"A": "ok", "B": "ok"})
    vs = live(br)
    check("두 카메라: 정확한 점이 있으면 위치는 그 점, approx 없음",
          len(vs) == 1 and vs[0]["approx"] is None and abs(vs[0]["observed_x_cm"] - 30) < 1e-6,
          f"{[(round(v['observed_x_cm'], 2), v['approx']) for v in vs]}")

    print("\nb. 합성 카메라: 칸에 선 앞차 + 뒤 통로 차 (뒤차 아래가 앞차에 가림)")
    for name, vc in (("남쪽 정면", VirtualCamera((40, 63, 15), yaw_dir=(0, -1))),
                     ("북쪽 정면 CAM-1", VirtualCamera((51, -8, 15)))):
        g = CamGeom(build_camera_config("A", 0, vc.size, vc.ref_clicks(table, pick_refs(table, vc), noise_px=.5, seed=1),
                                        table, params, 50.0), table, vc.size, params)
        br = TrackingBridge(table, contact_shift_cm=W / 2)
        errs, n_cut, n_two = [], 0, 0
        for no in sorted(n for n in g.monitored if table[n].row in ("1-14", "28-41")):
            sl = table[no]
            # 앞차 칸 바로 뒤(카메라에서 먼 쪽)의 통로: 남쪽 카메라는 1~14번 뒤 y 14.6, 북쪽 카메라는 28~41번 뒤 y 41.8
            south = vc.C[1] > 30
            if (south and sl.row != "1-14") or (not south and sl.row != "28-41"):
                continue
            ay = 14.6 if south else 41.8
            front = (sl.center[0] - W / 2, sl.center[0] + W / 2, sl.center[1] - L / 2, sl.center[1] + L / 2, H)
            rear = (sl.center[0] - L / 2, sl.center[0] + L / 2, ay - W / 2, ay + W / 2, H)
            boxes = visible_boxes(vc, [front, rear])
            if len(boxes) != 2:
                continue
            r = judge_frame(g, params, boxes)
            cut = [b for b in r.boxes if b.cut_by is not None and b.kind == "aisle"]
            if len(cut) != 1:
                continue
            n_cut += 1
            alone = judge_frame(g, params, [vc.box(*rear) + (0.9,)]).boxes[0]
            if alone.edge:
                continue
            ref = np.array(br.shifted_contact(alone, g))
            got = np.array(br.shifted_contact(cut[0], g, cut[0].approx_mat))
            errs.append(float(np.linalg.norm(got - ref)))
            vs = run(TrackingBridge(table, contact_shift_cm=W / 2), [r] * 6, geom=g)
            n_two += len(vs) == 2 and sorted(str(v["approx"]) for v in vs) == ["None", "cut"]
        e = np.array(errs)
        check(f"{name}: 뒤 통로 차가 잘린 경우 {n_cut}가지 모두 웹 지도에 앞차와 따로 나옴(뒤차 approx=cut)",
              n_cut >= 5 and n_two == n_cut, f"{n_two} / {n_cut}")
        check(f"{name}: 대략 위치 오차 4cm 안 (가리지 않았을 때 위치와 비교)", len(e) and e.max() <= 4.0,
              f"중앙 {np.median(e):.2f}, 최대 {e.max():.2f}cm")

    print("\nc. 합성 카메라: 화면 옆 끝에 잘린 통로 차")
    vc = VirtualCamera((40, 63, 15), yaw_dir=(0, -1))
    g = CamGeom(build_camera_config("A", 0, vc.size, vc.ref_clicks(table, pick_refs(table, vc), noise_px=.5, seed=1),
                                    table, params, 50.0), table, vc.size, params)
    shown = total = 0
    for x in np.arange(0, 78.4, 1.0):
        r = judge_frame(g, params, [vc.box(x - L / 2, x + L / 2, 41.8 - W / 2, 41.8 + W / 2, H) + (0.9,)]) \
            if vc.box(x - L / 2, x + L / 2, 41.8 - W / 2, 41.8 + W / 2, H) else None
        if r is None or not r.boxes[0].edge or r.boxes[0].edge_bottom:
            continue
        total += 1
        vs = run(TrackingBridge(table, contact_shift_cm=W / 2), [r] * 6, geom=g)
        shown += len(vs) == 1 and vs[0]["approx"] == "edge"
    check(f"옆 끝에 잘린 통로 차 {total}가지 모두 웹 지도에 나옴(approx=edge)", total >= 5 and shown == total, f"{shown} / {total}")

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
