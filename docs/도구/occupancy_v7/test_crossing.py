"""두 차가 통로에서 마주 지나갈 때 차량 ID 시험 (docs/44).

합성 카메라로 두 차가 통로를 서로 반대로 지나가게 하고, 판정(judge_frame) → 추적(TrackingBridge)을 거쳐 센다.
    ID 끊김 : 한 차에 확정(TRACKED) ID가 2개 이상 붙은 횟수 (ID 수 - 1 을 차마다 더함)
    ID 바뀜 : 한 ID가 두 차 모두에 붙은 경우 수 (A의 ID가 B로 넘어감). 끊김·바뀜 모두 2프레임 연속 붙은 것만 센다
    내 차 놓침: 처음 확정 ID를 "내 차"로 고르고, 웹 지도 규칙(내 차 후보 자동 이어 붙이기, docs/44)대로 따라갔을 때
               끝까지 내 차를 따라갔는지 / 다른 차로 잘못 넘어갔는지 / 확인이 필요했는지
상자 모델
    cut  : 앞차가 뒤차 상자 폭의 50% 넘게 덮으면 뒤차 상자 아래를 앞차 위 끝에서 자름
    real : cut + 덜 덮이면 덮인 쪽 폭을 깎음(보이는 부분만) + 두 상자 IoU > 0.7이면 뒤차 상자 없어짐(NMS)

실행: python test_crossing.py           (표와 합계)
      python test_crossing.py --check   (합계가 기준보다 나쁘면 종료 코드 1)
"""
import argparse
import itertools
import sys

import numpy as np

from parkit_config import DEFAULT_RUN, Params, load_run
from parkit_geom import CamGeom, SlotTable, build_camera_config
from parkit_judge import judge_frame
from parkit_synth import VirtualCamera, pick_refs, visible_boxes
from tracking_bridge import TrackingBridge

W, L, H = 3.2, 6.5, 2.0
CAMS = {"남쪽 정면": dict(pos=(40, 63, 15), yaw_dir=(0, -1)), "북쪽 정면": dict(pos=(51, -8, 15)),
        "대각선": dict(pos=(86, -8, 15), look=(50, 26)), "남쪽 대각선": dict(pos=(-6, 63, 15), look=(30, 28))}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def real_boxes(vc, cars):
    raw = [vc.box(*c) for c in cars]
    out = []
    for i, a in enumerate(raw):
        if a is None:
            continue
        x1, y1, x2, y2 = a
        drop = False
        for j, b in enumerate(raw):
            if j == i or b is None or b[3] <= a[3]:
                continue
            ov = min(x2, b[2]) - max(x1, b[0])
            if ov <= 0 or b[1] >= y2:
                continue
            if iou(a, b) > 0.7:
                drop = True
                break
            if ov >= 0.5 * (x2 - x1):
                y2 = max(y1 + 1, b[1])
            elif b[0] <= x1:
                x1 = min(x2 - 1, b[2])
            else:
                x2 = max(x1 + 1, b[0])
        if not drop and y2 - y1 > 3 and x2 - x1 > 3:
            out.append((x1, y1, x2, y2, 0.9))
    return out


def follow_mine(state, mine, memory=None):
    """웹 지도 규칙(web/src/tracking.js myCarState)을 파이썬으로 흉내: 내 차 ID를 이어 간다.
    돌려주는 것: (새 내 차 ID, "ok"/"auto"/"ask"/"lost")"""
    alive = {v["id"]: v for v in state["vehicles"] if v["status"] != "REMOVED"}
    v = alive.get(mine)
    if v is not None and v["status"] != "LOST":
        others = [i for i in v.get("related", []) if i in alive]
        return mine, ("ask" if others else "ok")
    succ, seen, frontier = [], {mine}, [mine]
    while frontier:
        nxt = []
        for f in frontier:
            for l in state.get("successors", {}).get(f, []):
                if l["id"] in seen:
                    continue
                seen.add(l["id"])
                (succ.append(l["id"]) if l["id"] in alive else nxt.append(l["id"]))
        frontier = nxt
    if len(succ) == 1:
        return succ[0], "auto"
    return mine, ("ask" if succ else "lost")


def scene_positions(scene, table, y0, gap, speed, xs):
    """장면별 차 위치 함수와 시간. 돌려주는 것: (t -> [(x, y, 가로?)...], 끝 시각)
        cross : 두 차가 통로에서 마주 지나감 (옆 간격 gap)
        follow: 두 차가 같은 방향으로 앞뒤 8cm 간격으로 감
        parked: 한 차가 통로를 지나가고, 통로 옆 칸에 차 한 대가 서 있음"""
    if scene == "cross":
        return (lambda t: [(xs[0] + speed * t, y0 - gap, True), (xs[1] - speed * t, y0 + gap, True)]), (xs[1] - xs[0]) / speed
    if scene == "follow":
        return (lambda t: [(xs[0] + 8 + speed * t, y0, True), (xs[0] + speed * t, y0, True)]), (xs[1] - xs[0] - 8) / speed
    row = "1-14" if y0 < 28 else "28-41"
    sl = min((s for s in table if s.row == row), key=lambda s: abs(s.center[0] - 40))
    return (lambda t: [(xs[0] + speed * t, y0, True), (sl.center[0], sl.center[1], False)]), (xs[1] - xs[0]) / speed


def simulate(vc, g, params, y0, model, fps=2.0, gap=1.75, speed=5.0, xs=(15, 65), config=None, scene="cross", jitter=0.0):
    table = g.table
    rng = np.random.default_rng(1)
    br = TrackingBridge(table, config, contact_shift_cm=W / 2)
    ids_of = {0: [], 1: []}
    mine, mine_car, outcome = None, None, set()
    where, dur = scene_positions(scene, table, y0, gap, speed, xs)
    for k in range(int(dur * fps) + 1):
        t = k / fps
        p3 = where(t)
        pos = [(x, y) for x, y, _ in p3]
        cars = [(x - L / 2, x + L / 2, y - W / 2, y + W / 2, H) if hz else (x - W / 2, x + W / 2, y - L / 2, y + L / 2, H)
                for x, y, hz in p3]
        boxes = visible_boxes(vc, cars) if model == "cut" else real_boxes(vc, cars)
        if jitter > 0:                                   # 상자 네 변을 ±jitter 픽셀 흔듦 (YOLO 흔들림 흉내)
            boxes = [tuple(np.asarray(b[:4]) + rng.uniform(-jitter, jitter, 4)) + (b[4],) for b in boxes]
        r = judge_frame(g, params, boxes)
        br.submit("A", t, r, geom=g)
        br.publish(t, {}, {"A": "ok"})
        st = br.store.read()
        vs = [v for v in st["vehicles"] if v["status"] == "TRACKED"]
        # 확정 차를 참 차와 짝짓기: 5cm 안이고, 다른 차보다 1.5cm 넘게 더 가까울 때만 (엇갈리는 순간의 착시를 세지 않음)
        for v in vs:
            d = [np.hypot(v["x_cm"] - x, v["y_cm"] - y) for x, y in pos]
            ci = int(np.argmin(d))
            if d[ci] < 5 and d[1 - ci] - d[ci] > 1.5:
                ids_of[ci].append(v["id"])
        if mine is None:
            for ci in (0, 1):
                if ids_of[ci]:
                    mine, mine_car = ids_of[ci][-1], ci
                    break
        elif mine is not None:
            mine, how = follow_mine(st, mine, None)
            outcome.add(how)
    def steady(seq):
        """2프레임 연속 그 차에 붙은 ID만 (한 프레임 튄 것은 빼기)"""
        return {a for a, b in zip(seq, seq[1:]) if a == b}
    breaks = sum(max(0, len(steady(ids_of[c])) - 1) for c in (0, 1))
    swaps = len(steady(ids_of[0]) & steady(ids_of[1]))
    # 내 차를 끝까지 따라갔는지: 마지막 내 차 ID가 끝 무렵 내 차(mine_car)에 붙어 있었나
    end_ok = mine is not None and mine_car is not None and ids_of[mine_car][-3:].count(mine) >= 1
    wrong = mine is not None and mine_car is not None and mine in ids_of[1 - mine_car][-3:]
    return {"끊김": breaks, "바뀜": swaps, "내 차 끝까지": end_ok, "내 차 잘못 넘어감": wrong,
            "확인 필요": "ask" in outcome, "자동 이음": "auto" in outcome}


SCENES = {"cross": "마주 지나감", "follow": "같은 방향 줄지어", "parked": "주차 차 옆 지나감"}


def run_all(config=None, verbose=True, scenes=tuple(SCENES), jitter=0.0):
    table = SlotTable(load_run(DEFAULT_RUN)["slot_table"])
    params = Params({}, table)
    keys = ("경우", "끊김", "바뀜", "내 차 끝까지", "내 차 잘못 넘어감", "확인 필요", "자동 이음")
    tot = {k: 0 for k in keys}
    tot["장면별"] = {sc: {k: 0 for k in keys} for sc in scenes}
    for name, kw in CAMS.items():
        kw = dict(kw)
        vc = VirtualCamera(kw.pop("pos"), **kw)
        refs = vc.ref_clicks(table, pick_refs(table, vc), noise_px=.5, seed=1)
        g = CamGeom(build_camera_config("A", 0, vc.size, refs, table, params, 50.0), table, vc.size, params)
        for scene in scenes:
            gaps = (1.75, 2.5) if scene == "cross" else (1.75,)
            for y0, model, gap, speed in itertools.product((14.6, 41.8), ("cut", "real"), gaps, (5.0, 8.0)):
                r = simulate(vc, g, params, y0, model, gap=gap, speed=speed, config=config, scene=scene, jitter=jitter)
                for d in (tot, tot["장면별"][scene]):
                    d["경우"] += 1
                    for k in ("끊김", "바뀜"):
                        d[k] += r[k]
                    for k in ("내 차 끝까지", "내 차 잘못 넘어감", "확인 필요", "자동 이음"):
                        d[k] += bool(r[k])
                if verbose and (r["끊김"] or r["바뀜"] or not r["내 차 끝까지"] or r["내 차 잘못 넘어감"]):
                    print(f"  {SCENES[scene]} {name} 통로 y{y0} {model} 옆 간격 {gap} 속도 {speed}: {r}")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tot = run_all()
    print("\n합계 (숫자는 경우의 수, 끊김·바뀜은 횟수):", tot)
    if a.check:
        # 기준 (docs/44): ID 바뀜 0, 마주 지나감·주차 차 옆에서 내 차를 잘못 넘겨받은 경우 0, 내 차를 끝까지 따라간 경우 90% 이상.
        # 줄지어 감(follow)의 "잘못 넘어감"은 화면 끝으로 나가는 앞차의 대략 위치가 늦게 따라가 뒤차와 가까워지는 것으로,
        # v6에서도 같아 기준에서 뺀다(docs/44 남은 문제)
        sc = tot["장면별"]
        ok = (tot["바뀜"] == 0 and sc["cross"]["내 차 잘못 넘어감"] == 0 and sc["parked"]["내 차 잘못 넘어감"] == 0
              and tot["내 차 끝까지"] >= 0.9 * tot["경우"])
        print("확인:", "통과" if ok else "실패")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
