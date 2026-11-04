"""v4 통로 차 접점 반 폭 보정 시험 (docs/36).

반대쪽 카메라 두 대(북쪽, 남쪽)가 같은 통로 차를 보면, 상자 아래 끝은 각자 가까운 옆면이라
두 접점이 차 폭(3.2cm)만큼 떨어진다. 추적기는 다른 카메라 점을 2cm 안에서만 같은 차로 잇기 때문에
두 대로 나온다. 접점을 카메라에서 먼 쪽(far_dir)으로 차 폭의 반만큼 밀면 한 점으로 모이는지 본다.
칸에 선 차의 접점은 그대로 보내는지(웹 지도에 계속 보이는지)도 본다.

실행: python test_aisle_contact_shift.py   (마지막 줄 "합계: N / N 통과")
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from occupancy import Engine
from parkit_config import DEFAULT_RUN, Params, load_run, save_json
from parkit_geom import CamGeom, SlotTable, build_camera_config
from parkit_judge import judge_frame
from parkit_synth import VirtualCamera, pick_refs
from tracking_bridge import TrackingBridge

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def calibrate(vcam, table, params, name):
    refs = vcam.ref_clicks(table, pick_refs(table, vcam), noise_px=0.5, seed=1)
    return build_camera_config(name, 0, vcam.size, refs, table, params, 50.0)


def box(x, y, kind="aisle", slot=None, cut=None, edge=False):
    return SimpleNamespace(contact_mat=np.array([x, y], float), conf=0.9, slot=slot, kind=kind,
                           cut_by=cut, edge=edge)


def fake_geom(cam_ground):
    """CamGeom.far_dir를 그대로 쓰는 가짜 카메라(카메라 바닥 위치만 있음)."""
    g = SimpleNamespace(cam_ground=np.array(cam_ground, float))
    g.far_dir = lambda p: CamGeom.far_dir(g, p)
    return g


def vehicles(bridge):
    return [v for v in bridge.store.read()["vehicles"] if v["status"] != "REMOVED"]


def main():
    table = SlotTable(load_run(DEFAULT_RUN)["slot_table"])
    params = Params({}, table)
    half = params(None)["car_width_cm"] / 2

    print("a. 보정 규칙")
    south = fake_geom((40, 70))            # 남쪽 카메라: 먼 쪽은 -y
    br = TrackingBridge(table, contact_shift_cm=half)
    p = br.shifted_contact(box(40, 16.2), south)
    check("통로 차 접점은 카메라 반대쪽으로 차 폭의 반(1.6cm)만큼 밀림", np.allclose(p, (40, 14.6)), f"{np.round(p, 3)}")
    p = br.shifted_contact(box(40, 16.2, kind="slot", slot=4), south)
    check("칸에 선 차(slot) 접점은 그대로", np.allclose(p, (40, 16.2)), f"{np.round(p, 3)}")
    p = br.shifted_contact(box(40, 16.2, kind="unmonitored", slot=4), south)
    check("감시하지 않는 칸의 차(unmonitored) 접점도 그대로", np.allclose(p, (40, 16.2)))
    p = br.shifted_contact(box(40, 16.2), None)
    check("카메라 정보(geom)가 없으면 접점 그대로(동료 시험·데모 호환)", np.allclose(p, (40, 16.2)))
    br0 = TrackingBridge(table, {"aisle_contact_shift_cm": 0}, contact_shift_cm=half)
    check("설정 aisle_contact_shift_cm 0이면 보정 끔", np.allclose(br0.shifted_contact(box(40, 16.2), south), (40, 16.2)))
    br2 = TrackingBridge(table, {"aisle_contact_shift_cm": 1.0, "match_radius_cm": 8.0}, contact_shift_cm=half)
    check("설정값이 기본값보다 앞서고, 나머지 추적 설정은 그대로 들어감",
          br2.contact_shift_cm == 1.0 and br2.config.match_radius_cm == 8.0)
    try:
        TrackingBridge(table, {"aisle_contact_shift_cm": -1})
        bad = False
    except ValueError:
        bad = True
    check("음수 보정값은 오류", bad)

    # 칸에 선 차는 추적기·웹에 계속 들어간다 (숨기지 않음)
    br = TrackingBridge(table, contact_shift_cm=half)
    for i in range(5):
        br.submit("A", i * 0.2, SimpleNamespace(boxes=[box(20, 10, kind="slot", slot=4)]), geom=south)
        br.publish(i * 0.2, {}, {"A": "ok"})
    vs = vehicles(br)
    check("칸에 선 차도 웹 지도 차량으로 나옴(접점 그대로)",
          len(vs) == 1 and abs(vs[0]["observed_x_cm"] - 20) < 1e-6 and abs(vs[0]["observed_y_cm"] - 10) < 1e-6,
          f"{[(round(v['observed_x_cm'], 2), round(v['observed_y_cm'], 2), v['kind']) for v in vs]}")

    print("\nb. 합성 카메라 두 대(북쪽·남쪽 정면)로 통로 차")
    cams = {"N": VirtualCamera((40, -8, 15)), "S": VirtualCamera((40, 63, 15), yaw_dir=(0, -1))}
    geoms = {k: CamGeom(calibrate(vc, table, params, k), table, vc.size, params) for k, vc in cams.items()}
    w, l, h = params(None)["car_width_cm"], params(None)["car_length_cm"], params(None)["car_height_cm"]
    raw_d, shift_d = [], []
    for y in (14.6, 41.8):                 # 통로 두 곳(docs/36), 통로를 따라 달리는 차(가로)
        for x in np.arange(15, 66, 5.0):
            pts = {}
            for k, vc in cams.items():
                b = vc.box(x - l / 2, x + l / 2, y - w / 2, y + w / 2, h)
                if b is None:
                    continue
                r = judge_frame(geoms[k], params, [b + (0.9,)])
                a = [bb for bb in r.boxes if bb.kind == "aisle" and bb.cut_by is None and not bb.edge]
                if a:
                    pts[k] = (np.array(a[0].contact_mat, float),
                              np.array(TrackingBridge(table, contact_shift_cm=half).shifted_contact(a[0], geoms[k])))
            if len(pts) == 2:
                raw_d.append(float(np.linalg.norm(pts["N"][0] - pts["S"][0])))
                shift_d.append(float(np.linalg.norm(pts["N"][1] - pts["S"][1])))
    raw_d, shift_d = np.array(raw_d), np.array(shift_d)
    # 화각 때문에 두 카메라가 같이 보는 곳(화면 가장자리에 닿지 않은 상자)은 통로 가운데 x 35~45쯤뿐
    check(f"두 카메라가 같이 본 통로 위치 {len(raw_d)}곳 (통로 2곳 × x 15~65 중)", len(raw_d) >= 5)
    check("보정 전: 두 접점이 차 폭쯤 떨어짐(추적기 2cm 기준 넘음)", len(raw_d) and raw_d.min() > 2.0,
          f"평균 {raw_d.mean():.2f}, 최소 {raw_d.min():.2f}, 최대 {raw_d.max():.2f}cm")
    check("보정 후: 0.5cm 안으로 모임", len(shift_d) and shift_d.max() <= 0.5,
          f"평균 {shift_d.mean():.2f}, 최대 {shift_d.max():.2f}cm")

    # 추적기에 넣어 보면: 보정 전 두 대, 보정 후 한 대
    def run(shift):
        br = TrackingBridge(table, contact_shift_cm=shift)
        y, t = 14.6, 0.0
        for i in range(16):                 # 0.2초마다, 초속 5cm로 +x 방향
            x = 25 + i
            t = i * 0.2
            for k, dt in (("N", 0.0), ("S", 0.05)):
                b = cams[k].box(x - l / 2, x + l / 2, y - w / 2, y + w / 2, h)
                br.submit(k, t + dt, judge_frame(geoms[k], params, [b + (0.9,)]), geom=geoms[k])
            br.publish(t + 0.05, {}, {"N": "ok", "S": "ok"})
        return vehicles(br)
    v0, v1 = run(0.0), run(half)
    check("보정 끔: 같은 통로 차가 두 대로 나옴(지금 문제)", len(v0) == 2, f"{len(v0)}대")
    check("보정 켬: 한 대로 이어짐", len(v1) == 1,
          f"{len(v1)}대, 위치 {[(round(v['x_cm'], 2), round(v['y_cm'], 2)) for v in v1]}, 참값 y 14.6")
    check("보정 켬: 위치가 차 가운데(y 14.6) 0.5cm 안", len(v1) == 1 and abs(v1[0]["y_cm"] - 14.6) <= 0.5)

    print("\nc. 판정 프로그램(Engine) 연결")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        vc = cams["S"]
        cal = calibrate(vc, table, params, "A")
        cal["source"] = "unused.mp4"
        save_json(str(path / "cam.json"), cal)
        for over, want in ((None, half), ({"aisle_contact_shift_cm": 0.0}, 0.0)):
            run_cfg = {"slot_table": table.path, "cameras": [{"config": "cam.json"}], "log": {"dir": "logs"}}
            if over is not None:
                run_cfg["vehicle_tracking"] = over
            save_json(str(path / "run.json"), run_cfg)
            args = SimpleNamespace(run=str(path / "run.json"), preset=None, source=None, log_frames=False,
                                   every=None, speed=0, model=None, conf=None, quiet=True, detector="gt",
                                   tracking=True, tracking_config=None, web=False, duration=0)
            e = Engine(args, headless=True)
            try:
                c = e.cams[0]
                c.geom = CamGeom(cal, table, vc.size, e.params)
                c.size = vc.size
                b = vc.box(40 - l / 2, 40 + l / 2, 14.6 - w / 2, 14.6 + w / 2, h) + (0.95,)
                e.detector = lambda *_: [b]
                frame = np.zeros((vc.size[1], vc.size[0], 3), dtype=np.uint8)
                for i in range(6):
                    e.t = i * 0.5
                    e.analyze(c, frame, i, observed_at=e.t)
                    e.finish_tick()
                vs = vehicles(e.vehicle_tracking)
                raw = np.array(c.res.boxes[0].contact_mat, float)
                ok = (e.vehicle_tracking.contact_shift_cm == want and len(vs) == 1
                      and abs(vs[0]["observed_y_cm"] - (raw[1] - want)) < 0.05)
                check(f"설정 {over}: 보정 {want}cm, 웹 차량 점 y {vs[0]['observed_y_cm'] if vs else None:.2f} "
                      f"(접점 y {raw[1]:.2f})", ok)
            finally:
                e.detector = None
                e.close()

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
