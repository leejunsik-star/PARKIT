"""웹 지도 "보일 차량" 서버 쪽 시험 (docs/42, vehicle_tracking.parked_vehicles).

거르기는 웹 화면(web/src/tracking.js shownByParking, 시험은 web에서 npm test)이 한다.
서버는 늘 모든 차를 보내고, 웹이 거르는 데 필요한 것(차마다 칸 slot·종류 kind, 칸 상태, 처음 값)을 보내는지 본다.
    show          : 전체 차량 (기본)
    hide_occupied : 주차 확정 차량 숨기기
    aisle_only    : 주차칸 차량 모두 숨기기

실행: python test_parked_vehicles.py   (마지막 줄 "합계: N / N 통과")
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from occupancy import Engine
from parkit_config import DEFAULT_RUN, Params, load_run, save_json
from parkit_geom import CamGeom, SlotTable, build_camera_config
from parkit_synth import VirtualCamera, pick_refs
from tracking_bridge import TrackingBridge

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def web_filter(state, mode):
    """web/src/tracking.js shownByParking과 같은 규칙 (서버가 보낸 것만으로 거를 수 있는지 확인용)."""
    status = {s["no"]: s["status"] for s in state["slots"]}
    out = []
    for v in state["vehicles"]:
        if v["status"] == "REMOVED":
            continue
        hide = mode != "show" and v["kind"] != "aisle" and (
            mode == "aisle_only" or (v["slot"] is not None and status.get(v["slot"]) == "OCCUPIED"))
        if not hide:
            out.append(v["kind"])
    return sorted(out)


def main():
    table = SlotTable(load_run(DEFAULT_RUN)["slot_table"])

    print("a. 설정 값")
    check("설정이 없으면 show", TrackingBridge(table).parked_mode == "show")
    for m in ("show", "hide_occupied", "aisle_only"):
        br = TrackingBridge(table, {"parked_vehicles": m})
        br.publish(0.0, {}, {})
        check(f"{m}: 웹 상태 tracking_config.parked_vehicles로 처음 값을 보냄",
              br.store.read()["tracking_config"]["parked_vehicles"] == m)
    try:
        TrackingBridge(table, {"parked_vehicles": "hide"})
        bad = False
    except ValueError:
        bad = True
    check("모르는 값은 오류", bad)

    print("\nb. 판정 프로그램(Engine): 남쪽 정면 카메라, 칸에 선 차 1대 + 통로를 달리는 차 1대")
    params = Params({}, table)
    vc = VirtualCamera((40, 63, 15), yaw_dir=(0, -1))
    refs = vc.ref_clicks(table, pick_refs(table, vc), noise_px=0.5, seed=1)
    cal = build_camera_config("A", 0, vc.size, refs, table, params, 50.0)
    cal["source"] = "unused.mp4"
    near = sorted((n for n in cal["monitored"] if table[n].row == "28-41"), key=lambda n: table[n].x0)
    slot_no = near[len(near) // 2]
    w, l, h = params(None)["car_width_cm"], params(None)["car_length_cm"], params(None)["car_height_cm"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        save_json(str(path / "cam.json"), cal)
        run_cfg = {"slot_table": table.path, "cameras": [{"config": "cam.json"}], "log": {"dir": "logs"},
                   "vehicle_tracking": {"parked_vehicles": "hide_occupied"}}
        save_json(str(path / "run.json"), run_cfg)
        args = SimpleNamespace(run=str(path / "run.json"), preset=None, source=None, log_frames=False,
                               every=None, speed=0, model=None, conf=None, quiet=True, detector="gt",
                               tracking=True, tracking_config=None, web=False, duration=0)
        e = Engine(args, headless=True)
        try:
            c = e.cams[0]
            c.geom = CamGeom(cal, table, vc.size, e.params)
            c.size = vc.size
            frame = np.zeros((vc.size[1], vc.size[0], 3), dtype=np.uint8)
            seen = {}
            for i in range(14):
                e.t = i * 0.5
                ax = 25 + i * 1.0           # 통로 차는 위쪽 통로(y 14.6)를 달림 (x 21까지는 화면 끝에 닿아 좌표를 안 씀)
                dets = [vc.car_in_slot(table[slot_no]) + (0.95,),
                        vc.box(ax - l / 2, ax + l / 2, 14.6 - w / 2, 14.6 + w / 2, h) + (0.9,)]
                e.detector = lambda *_, d=dets: d
                e.analyze(c, frame, i, observed_at=e.t)
                e.finish_tick()
                seen[e.t] = e.vehicle_tracking.store.read()
            early, late = seen[1.5], seen[6.5]
            st = {s["no"]: s["status"] for s in late["slots"]}
            parked = [v for v in late["vehicles"] if v["kind"] == "slot"]
            check(f"서버는 주차 확정 뒤에도 모든 차를 보냄 (6.5초 {slot_no}번 {st.get(slot_no)})",
                  st.get(slot_no) == "OCCUPIED" and web_filter(late, "show") == ["aisle", "slot"])
            check(f"주차된 차에 칸 번호와 종류가 붙음 (slot {parked[0]['slot'] if parked else None})",
                  len(parked) == 1 and parked[0]["slot"] == slot_no)
            check("처음 값은 run.json의 hide_occupied", late["tracking_config"]["parked_vehicles"] == "hide_occupied")
            for m, want_early, want_late in (("show", ["aisle", "slot"], ["aisle", "slot"]),
                                             ("hide_occupied", ["aisle", "slot"], ["aisle"]),
                                             ("aisle_only", ["aisle"], ["aisle"])):
                a, b = web_filter(early, m), web_filter(late, m)
                check(f"웹 규칙 {m}: 1.5초(판정 전) {a}, 6.5초(주차 확정 뒤) {b}", a == want_early and b == want_late)
        finally:
            e.detector = None
            e.close()

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
