"""PARKIT 점유 판정 — 프로그램 전체 시험 (시간 거르기 P6, 가림 P5, 걸침 P3, 2대 합치기 P9, 끊김 P10, 영상 재생 C2)

1부: 상태 기계(parkit_state)만 따로 시험한다.
2부: 가상 CAM-1, CAM-2로 합성 영상 두 개(1280x720, 초당 10장)와 정답 상자 파일(<영상>.gt.json)을 만들고,
     1920x1080으로 보정한 카메라 설정과 실행 설정을 만든 뒤 occupancy.py를 창 없이 돌린다.
     기록(state_changes.csv)에서 상태가 바뀐 시각이 시나리오와 맞는지 본다.
     같은 영상을 속도를 바꿔 두 번 돌려 결과가 같은지도 본다(C2).
정답 상자를 쓰는 이유: 그려서 만든 차는 M8이 잡지 못한다. 실제 모델 경로는 웹캠과 녹화 영상으로 확인한다(28 문서).

사용법 (docs\\도구\\occupancy 에서):
    python test_pipeline.py
만든 영상과 기록은 logs\\test_pipeline\\ 에 남는다.
"""

import csv, glob, json, os, shutil, subprocess, sys

import cv2
import numpy as np

from parkit_config import HERE, Params, save_json
from parkit_geom import SlotTable, build_camera_config
from parkit_judge import ABSENT, OCCLUDED, PRESENT
from parkit_state import DYNAMIC_BLIND, EMPTY, OCCUPIED, STATIC_BLIND, Tracker, merge_obs
from parkit_synth import cam1, cam2, draw_mat_image, pick_refs, render_frame, visible_boxes

TABLE = os.path.normpath(os.path.join(HERE, "..", "..", "칸표_매트좌표_초안.json"))
OUT = os.path.join(HERE, "logs", "test_pipeline")
RESULTS = []
FPS = 10
RUN_SIZE = (1280, 720)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1부: 상태 기계

def part1():
    print("\n1부. 상태 기계 (P6, P5, P9, P10, T1~T8)")
    params = Params({}, None)
    ev = []

    def run(cover, script, until, dt=0.5):
        tr = Tracker([1, 2, 3], cover, params)
        ev.clear()
        tr.on_change(lambda e: ev.append((round(e["time"], 2), e["slot"], e["status"], e["reason"], e["last_known"])))
        t = 0.0
        while t <= until + 1e-9:
            obs, state = script(t)
            tr.update(t, obs, state)
            t = round(t + dt, 3)
        return tr

    # 켤 때: 감시 칸은 DYNAMIC_BLIND(INITIALIZING), 아닌 칸은 STATIC_BLIND
    tr = run({1: ["A"]}, lambda t: ({"A": {1: ABSENT}}, {"A": "ok"}), 0)
    first = {e[1]: (e[2], e[3]) for e in ev}
    check("켤 때 감시 칸 DYNAMIC_BLIND(INITIALIZING), 아닌 칸 STATIC_BLIND(NO_COVERAGE)",
          first[1] == (DYNAMIC_BLIND, "INITIALIZING") and first[2] == (STATIC_BLIND, "NO_COVERAGE"))

    # 입차 3초, 출차 5초
    run({1: ["A"]}, lambda t: ({"A": {1: PRESENT if 2 <= t < 10 else ABSENT}}, {"A": "ok"}), 20)
    ch = [(e[0], e[2]) for e in ev if e[1] == 1][1:]
    check("2초에 차가 들어오면 3초 뒤 OCCUPIED, 10초에 빠지면 5초 뒤 EMPTY",
          ch == [(5.0, OCCUPIED), (15.0, EMPTY)], str(ch))

    # 한두 프레임 놓쳐도 그대로
    run({1: ["A"]}, lambda t: ({"A": {1: ABSENT if t in (6.0, 6.5) else PRESENT}}, {"A": "ok"}), 12)
    ch = [(e[0], e[2]) for e in ev if e[1] == 1][1:]
    check("OCCUPIED 중 1초 놓침은 상태를 바꾸지 않음", ch == [(3.0, OCCUPIED)], str(ch))

    # 가림: 1초 넘게 → DYNAMIC_BLIND, 풀리면 다시 3초/5초
    def occl(t):
        o = PRESENT if t < 8 else (OCCLUDED if t < 12 else PRESENT)
        return {"A": {1: o}}, {"A": "ok"}
    run({1: ["A"]}, occl, 20)
    ch = [(e[0], e[2], e[3], e[4]) for e in ev if e[1] == 1][1:]
    check("가림 1초 넘게 → DYNAMIC_BLIND(OCCLUDED, 직전 OCCUPIED), 풀린 뒤 3초 → OCCUPIED",
          ch == [(3.0, OCCUPIED, None, None), (9.5, DYNAMIC_BLIND, "OCCLUDED", OCCUPIED), (15.0, OCCUPIED, None, None)],
          str(ch))
    run({1: ["A"]}, lambda t: ({"A": {1: OCCLUDED if 6 <= t < 7 else ABSENT}}, {"A": "ok"}), 14)
    ch = [(e[0], e[2]) for e in ev if e[1] == 1][1:]
    check("1초 이하 가림(지나가는 차)은 DYNAMIC_BLIND를 만들지 않음", ch == [(5.0, EMPTY)], str(ch))

    # P9 두 카메라 합치기
    check("P9 한쪽 가려짐 → 다른 쪽 결과", merge_obs({"A": OCCLUDED, "B": ABSENT}) == ABSENT and
          merge_obs({"A": OCCLUDED, "B": PRESENT}) == PRESENT)
    check("P9 둘 다 정상인데 다르면 차 있음", merge_obs({"A": ABSENT, "B": PRESENT}) == PRESENT)
    check("P9 둘 다 가려짐 → 가려짐, 하나 끊김 + 하나 가려짐 → 가려짐",
          merge_obs({"A": OCCLUDED, "B": OCCLUDED}) == OCCLUDED and merge_obs({"A": "offline", "B": OCCLUDED}) == OCCLUDED)

    # P10 끊김: 그 카메라만 보는 칸만 DYNAMIC_BLIND
    def off(t):
        return ({"A": {1: PRESENT, 2: PRESENT}, "B": {2: PRESENT}},
                {"A": "ok" if t < 10 else "offline", "B": "ok"})
    run({1: ["A"], 2: ["A", "B"]}, off, 16)
    ch1 = [(e[0], e[2], e[3], e[4]) for e in ev if e[1] == 1][1:]
    ch2 = [(e[0], e[2]) for e in ev if e[1] == 2][1:]
    check("P10 A가 끊기면 A만 보는 1번은 DYNAMIC_BLIND(CAMERA_OFFLINE), 2번은 B로 그대로",
          ch1 == [(3.0, OCCUPIED, None, None), (10.0, DYNAMIC_BLIND, "CAMERA_OFFLINE", OCCUPIED)] and ch2 == [(3.0, OCCUPIED)],
          f"{ch1} / {ch2}")

    # 칸별 시간 값 (P8)
    p8 = Params({"overrides": {"slots": {"1": {"t_occupied_s": 1.0}}}}, None)
    tr = Tracker([1, 2], {1: ["A"], 2: ["A"]}, p8)
    got = []
    tr.on_change(lambda e: got.append((e["time"], e["slot"], e["status"])))
    for k in range(9):
        tr.update(k * 0.5, {"A": {1: PRESENT, 2: PRESENT}}, {"A": "ok"})
    check("P8 칸별 시간 값: 1번만 1초, 2번은 3초", (1.0, 1, OCCUPIED) in got and (3.0, 2, OCCUPIED) in got, str(got[2:]))


# ---------------------------------------------------------------- 2부: 합성 영상

def car_at(cx, cy, w=3.2, l=6.5, h=2.0, color=(40, 40, 160)):
    return (cx - w / 2, cx + w / 2, cy - l / 2, cy + l / 2, h, color)


def aisle_car(cx, y0, h=2.0, color=(150, 90, 30)):
    return (cx - 3.25, cx + 3.25, y0, y0 + 3.2, h, color)


def scene(t, table):
    """시각 t의 매트 위 차들. 시나리오 (기댓값은 part2의 EXPECT)."""
    cars = []
    s = table.slots
    if 2 <= t < 20:
        cars.append(car_at(*s[4].center))                                   # E1 4번 입차/출차
    if 6 <= t < 10:
        cars.append(aisle_car(75 - 20 * (t - 6), 14.5))                     # E2 통로를 지나가는 차 (20cm/s)
    if 10 <= t < 16:
        cars.append(aisle_car(s[6].center[0], 16.3))                        # E3 6번 칸 앞에 서서 가림
    if 12 <= t < 30:
        cars.append(car_at(s[2].x0, s[2].center[1]))                         # E4 2·3번 경계에 걸침
    if 13 <= t < 17:
        cars.append(aisle_car(s[4].center[0], 16.3))                        # E7 차 있는 4번 앞에 통로 차 (P2a)
    if 3 <= t < 40:
        cars.append(car_at(*s[35].center))                                  # E5 두 카메라가 같이 보는 35번
    cars.append(car_at(*s[30].center))                                      # E6 CAM-2만 보는 30번 (계속)
    return cars


EXPECT = [
    # (칸, 상태, 이유, 기대 시각, 설명)
    (4, OCCUPIED, None, 5.0, "E1 4번 입차 2초 + 3초"),
    (4, EMPTY, None, 25.0, "E1 4번 출차 20초 + 5초 (E7 통로 차가 13~17초에 가려도 그대로)"),
    (6, EMPTY, None, 5.0, "빈 칸 켠 뒤 5초"),
    (6, DYNAMIC_BLIND, "OCCLUDED", 11.5, "E3 6번 앞 통로 차 10초부터, 1초 넘게"),
    (6, EMPTY, None, 21.0, "E3 통로 차 16초에 떠남 + 5초"),
    (5, DYNAMIC_BLIND, "OCCLUDED", 14.5, "E3 + E7 통로 차가 함께 5번을 덮음 (13초부터, 1초 넘게)"),
    (5, EMPTY, None, 21.0, "E3가 16초에 떠나 E7 혼자로는 30% 아래 + 5초 (P5는 통로 차 상자들의 합집합)"),
    (2, OCCUPIED, None, 15.0, "E4 걸침 12초 + 3초 (2번)"),
    (3, OCCUPIED, None, 15.0, "E4 걸침 12초 + 3초 (3번)"),
    (2, EMPTY, None, 35.0, "E4 30초에 떠남 + 5초"),
    (35, OCCUPIED, None, 6.0, "E5 35번 3초 + 3초 (두 카메라)"),
    (35, EMPTY, None, 45.0, "E5 40초에 떠남 + 5초 (CAM-2 끊긴 뒤 CAM-1만으로)"),
    (30, OCCUPIED, None, 3.0, "E6 30번 처음부터 차, 3초"),
    (30, DYNAMIC_BLIND, "CAMERA_OFFLINE", 55.0, "E6 CAM-2 영상 25초에 끝 + 30초"),
    (12, DYNAMIC_BLIND, "CAMERA_OFFLINE", 55.0, "CAM-2만 보는 12번도 끊김"),
]
NEVER_OCC = [1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41]
LEN = {"CAM-1": 62.0, "CAM-2": 25.0}


def make_videos(table, params):
    os.makedirs(OUT, exist_ok=True)
    mat = draw_mat_image(table, 12)
    files = {}
    for name, vc in (("CAM-1", cam1()), ("CAM-2", cam2())):
        refs = vc.ref_clicks(table, pick_refs(table, vc), noise_px=0.5, seed=2)
        cfg = build_camera_config(name, os.path.join(OUT, f"{name}.mp4"), vc.size, refs, table, params, 50.0)
        cfg["date"] = "합성 시험"
        save_json(os.path.join(OUT, f"cam_{name}.json"), cfg)
        vs = vc.scaled(RUN_SIZE)
        path = os.path.join(OUT, f"{name}.mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RUN_SIZE)
        gt = {}
        n = int(LEN[name] * FPS)
        for i in range(n):
            cars = scene(i / FPS, table)
            w.write(render_frame(vs, mat, 12, cars))
            gt[str(i)] = [list(map(lambda v: round(v, 2), b)) for b in visible_boxes(vs, [c[:5] for c in cars])]
        w.release()
        with open(path + ".gt.json", "w", encoding="utf-8") as f:
            json.dump({"fps": FPS, "frames": gt}, f)
        files[name] = (path, cfg["monitored"])
    run = {
        "slot_table": TABLE, "model": "../../models/M8.pt", "conf": 0.25, "analyze_fps": 2,
        "cameras": [{"config": f"cam_{n}.json", "width": RUN_SIZE[0], "height": RUN_SIZE[1]} for n in ("CAM-1", "CAM-2")],
        "log": {"dir": "logs", "frames": True},
    }
    save_json(os.path.join(OUT, "run_test.json"), run)
    return files


def run_program(speed):
    before = set(glob.glob(os.path.join(OUT, "logs", "*")))
    cmd = [sys.executable, os.path.join(HERE, "occupancy.py"), "--run", os.path.join(OUT, "run_test.json"),
           "--detector", "gt", "--no-window", "--speed", str(speed), "--quiet"]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
        return None
    new = sorted(set(glob.glob(os.path.join(OUT, "logs", "*"))) - before)
    with open(os.path.join(new[-1], "state_changes.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), new[-1]


def part2():
    print("\n2부. 합성 영상 두 개로 프로그램 전체 (보정 1920x1080, 영상 1280x720, 정답 상자)")
    table = SlotTable(TABLE)
    params = Params({}, table)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    files = make_videos(table, params)
    print(f"  영상: {', '.join(os.path.basename(p) for p, _ in files.values())}  (CAM-1 감시 {files['CAM-1'][1]}, "
          f"CAM-2 감시 {files['CAM-2'][1]})")
    r = run_program(0)
    if r is None:
        check("occupancy.py 실행", False)
        return
    rows, logdir = r
    print(f"  기록: {logdir}")
    ch = [(float(x["t_s"]), int(x["slot"]), x["status"], x["reason"]) for x in rows if x["prev"]]
    for slot, status, reason, t_exp, desc in EXPECT:
        hit = [c for c in ch if c[1] == slot and c[2] == status and (reason is None or c[3] == reason)]
        near = [c for c in hit if abs(c[0] - t_exp) <= 0.6]
        got = ", ".join(f"{c[0]:.1f}s" for c in hit) or "없음"
        check(f"{slot}번 → {status}{' ' + reason if reason else ''} 약 {t_exp}s ({desc})", bool(near), f"실제 {got}")
    bad = [c for c in ch if c[1] in NEVER_OCC and c[2] == OCCUPIED]
    check("차를 넣지 않은 칸은 OCCUPIED가 되지 않음 (지나가는 차 E2 포함)", not bad, str(bad[:5]))
    occl = sorted({c[1] for c in ch if c[3] == "OCCLUDED"})
    check("가려짐은 5·6번만 (E3 6번, E3+E7 통로 차가 함께 5번을 덮음. E2는 1초 안, 4번은 잘린 채 잡혀 차 있음 우선)",
          occl == [5, 6], str(occl))
    four = [c for c in ch if c[1] == 4]
    check("4번은 OCCUPIED(5초) → EMPTY(25초) 두 번만 바뀜", [c[2] for c in four] == [OCCUPIED, EMPTY], str(four))
    cam1_only = [c for c in ch if c[1] in (1, 2, 3, 4, 5, 6, 7, 34, 35, 36, 37, 38, 39, 40, 41) and c[3] == "CAMERA_OFFLINE"]
    check("CAM-2가 끊겨도 CAM-1이 보는 칸(34, 35번 포함)은 끊김이 되지 않음", not cam1_only, str(cam1_only[:3]))
    blind = sorted({c[1] for c in ch if c[3] == "CAMERA_OFFLINE"})
    check("끊김(CAMERA_OFFLINE)은 CAM-2만 보는 8~14, 28~33번", blind == list(range(8, 15)) + list(range(28, 34)), str(blind))
    sta = [x for x in rows if not x["prev"] and x["status"] == STATIC_BLIND]
    check("감시 칸이 아닌 칸 28개는 STATIC_BLIND (56 - 15 - 15 + 겹친 2)", len(sta) == 28 and
          not any(int(x["slot"]) in range(1, 15) for x in sta), f"{len(sta)}개")
    frames_csv = os.path.join(logdir, "frames.csv")
    with open(frames_csv, encoding="utf-8-sig") as f:
        fr = list(csv.DictReader(f))
    cut = [x for x in fr if "P2a" in x["p2a"]]
    check("frames.csv(L2)에 상자, 접점 좌표, 고른 칸, P2a/P5 표시가 남음",
          len(fr) > 100 and cut and any(x["p5_occluded"] for x in fr),
          f"{len(fr)}줄, P2a 표시 {len(cut)}줄")

    # C2: 같은 영상은 재생 속도와 상관없이 같은 결과
    r2 = run_program(20)
    if r2 is None:
        check("빠르게/천천히 돌린 결과가 같음", False)
        return
    a = [(x["t_s"], x["slot"], x["prev"], x["status"], x["reason"], x["last_known"]) for x in rows]
    b = [(x["t_s"], x["slot"], x["prev"], x["status"], x["reason"], x["last_known"]) for x in r2[0]]
    check("최대 속도와 20배속으로 돌린 상태 변화 기록이 같음 (C2)", a == b, f"{len(a)}줄 / {len(b)}줄")


def main():
    part1()
    part2()
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
