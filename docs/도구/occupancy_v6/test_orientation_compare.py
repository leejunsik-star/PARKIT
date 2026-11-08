"""PARKIT 방향 추정 합성 시험 — v5와 v6 비교 (docs/40, docs/39 A5절과 같은 경우)

칸마다 차를 여러 위치에 놓고, 카메라 자세로 화면에 그린 상자(합성 상자)를 judge_frame()에 넣는다.
    ① 세로로 선 차 15가지 (차 폭 2.8·3.2·3.6cm × 좌우 -1~+1cm 5가지)
    ② 두 칸 사이에 가로로 선 차 9가지 (칸 오른쪽 경계 근처, 좌우 -2~+2cm 0.5cm 간격)
    ③ 칸 가운데에 가로로 선 차 5가지 (좌우 -0.6~+0.6cm, 3칸을 막음)
1~14번·28~41번 줄에서 카메라가 감시하는 칸마다 반복한다(양 끝 칸 제외).
정답은 실제 차가 빈 공간(inner)에 straddle 기준(0.1cm) 이상 들어간 칸 모두다. "차 있음"으로 낸 칸이 정답과 같으면 맞음.
표의 숫자는 cm가 아니라 경우의 수다. 판정 숫자는 각 폴더의 config/run.json, 카메라는 config/cam_CAM-1.json, cam_CAM-2.json.

사용:
    python test_orientation_compare.py                 # v5(옆 폴더 occupancy_v5)와 v6 비교
    python test_orientation_compare.py --v5 <v5 폴더>
    python test_orientation_compare.py --jitter 3      # 상자 네 변을 ±3픽셀 흔들어서 (YOLO 상자 흔들림 흉내)
두 폴더는 모듈 이름이 같아서, 버전마다 따로 파이썬을 띄워 돌린다(--worker).
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("cam_CAM-1.json", "cam_CAM-2.json")
KINDS = (("vert", "① 세로"), ("side2", "② 가로 2칸"), ("side3", "③ 가로 3칸"))
SIZE = (1920, 1080)
CAR_W, CAR_L, CAR_H = 3.2, 6.5, 2.0
THR = 0.1


def worker(folder, extra_params, jitter=0.0):
    """한 버전 폴더의 코드로 시험하고 결과를 JSON으로 출력한다.
    jitter > 0이면 상자 네 변을 각각 ±jitter 픽셀 안에서 흔든다(YOLO 상자 흔들림 흉내, 같은 씨앗)."""
    sys.path.insert(0, folder)
    import numpy as np
    from parkit_config import Params, load_json, resolve_first
    from parkit_geom import CamGeom, SlotTable, car_box
    from parkit_judge import PRESENT, judge_frame

    run_path = os.path.join(folder, "config", "run.json")
    cfg = load_json(run_path)
    cfg.setdefault("params", {}).update(extra_params)
    table = SlotTable(resolve_first(run_path, cfg["slot_table"])[0])
    params = Params(cfg, table)
    rng = np.random.default_rng(0)
    out = {}
    for camf in CAMS:
        cam = load_json(os.path.join(folder, "config", camf))
        g = CamGeom(cam, table, SIZE, params)
        st = {k: {"n": 0, "ok": 0, "unk": 0, "extra": 0, "miss": 0, "orient_ok": 0} for k, _ in KINDS}
        for row in ("1-14", "28-41"):
            nos = sorted((n for n in g.monitored if table[n].row == row), key=lambda n: table[n].x0)
            for no in nos[1:-1]:
                sl = table[no]
                cy = sl.center[1]
                cases = []
                for w in (2.8, 3.2, 3.6):
                    for dx in (-1, -0.5, 0, 0.5, 1):
                        cx = sl.center[0] + dx
                        cases.append(("vert", cx - w / 2, cx + w / 2, cy - CAR_L / 2, cy + CAR_L / 2))
                for dx in np.arange(-2.0, 2.01, 0.5):
                    cx = sl.ix1 + 0.45 + dx
                    cases.append(("side2", cx - CAR_L / 2, cx + CAR_L / 2, cy - CAR_W / 2, cy + CAR_W / 2))
                for dx in (-0.6, -0.3, 0, 0.3, 0.6):
                    cx = sl.center[0] + dx
                    cases.append(("side3", cx - CAR_L / 2, cx + CAR_L / 2, cy - CAR_W / 2, cy + CAR_W / 2))
                for kind, x0, x1, y0, y1 in cases:
                    b = car_box(g.K, g.rvec, g.tvec, x0, x1, y0, y1, CAR_H, g.size)
                    if b is None:
                        continue
                    truth = {s.no for s in table if s.row == row and min(x1, s.ix1) - max(x0, s.ix0) >= THR}
                    if not truth <= g.monitored:
                        continue
                    if jitter > 0:
                        b = tuple(float(v) for v in np.asarray(b) + rng.uniform(-jitter, jitter, 4))
                    r = judge_frame(g, params, [tuple(b) + (0.9,)])
                    got = {n for n, o in r.obs.items() if o == PRESENT}
                    unk = {n for n, o in r.obs.items() if o is None}
                    s = st[kind]
                    s["n"] += 1
                    s["ok"] += got == truth and not unk
                    s["unk"] += bool(unk) and got <= truth <= got | unk      # 틀린 칸은 없고 일부를 모름으로 둠
                    s["extra"] += bool(got - truth)
                    s["miss"] += bool(truth - got - unk)                   # 막힌 칸을 빈 칸으로 냄
                    o = getattr(r.boxes[0], "orient", "vert") or "vert"
                    s["orient_ok"] += (o == "vert") == (kind == "vert")
        out[camf] = st
    print(json.dumps(out))


def run_version(folder, extra=None, jitter=0.0):
    cmd = [sys.executable, "-B", os.path.abspath(__file__), "--worker", folder,
           "--params", json.dumps(extra or {}), "--jitter", str(jitter)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise SystemExit(f"실패: {folder}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", default=os.path.join(HERE, "..", "occupancy_v5"))
    ap.add_argument("--worker")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--jitter", type=float, default=0.0, help="상자 네 변을 ±픽셀만큼 흔든다 (1920×1080 기준)")
    a = ap.parse_args()
    if a.worker:
        worker(os.path.abspath(a.worker), json.loads(a.params), a.jitter)
        return
    v5 = os.path.abspath(a.v5)
    rows = [("v5 (기준 0.1)", run_version(v5, None, a.jitter)),
            ("v6 방향 추정 끔", run_version(HERE, {"orientation_enable": False}, a.jitter)),
            ("v6 run.json (여유 0.4)", run_version(HERE, None, a.jitter)),
            ("v6 여유 0", run_version(HERE, {"orient_margin": 0.0}, a.jitter))]
    if a.jitter > 0:
        print(f"상자 흔들림 ±{a.jitter:g}px\n")
    head = "| 판 | " + " | ".join(f"{c[4:9]} {k}" for c in CAMS for _, k in KINDS) + " |"
    line = "|" + "---|" * (1 + len(CAMS) * len(KINDS))

    def table(title, cell):
        print(title + "\n")
        print(head)
        print(line)
        for name, res in rows:
            print(f"| {name} | " + " | ".join(cell(res[c][k]) for c in CAMS for k, _ in KINDS) + " |")
        print()

    table("맞은 경우 / 시험한 경우 (숫자는 경우의 수). 맞음 = '차 있음'으로 낸 칸이 정답과 같고 '모름' 칸이 없음",
          lambda s: f"{s['ok']} / {s['n']}")
    table("모름 (틀린 칸은 없지만 일부 칸을 모름으로 둔 경우의 수)", lambda s: str(s["unk"]))
    table("틀린 경우: 빈 칸을 차 있음으로 / 막힌 칸을 빈 칸으로 (둘 다 있으면 양쪽에 셈)",
          lambda s: f"{s['extra']} / {s['miss']}")
    r5, r6 = rows[0][1], rows[2][1]
    print("v6 run.json 방향 맞춤 (애매해서 세로로 둔 가로 차는 틀린 것으로 셈): " + ", ".join(
        f"{c[4:9]} {sum(r6[c][k]['orient_ok'] for k, _ in KINDS)} / {sum(r6[c][k]['n'] for k, _ in KINDS)}" for c in CAMS))
    same = rows[0][1] == rows[1][1]
    print(f"v6 방향 추정 끔 = v5: {'같음' if same else '다름'}")
    if a.jitter > 0:
        print("흔들림 시험은 표만 보인다(통과 확인 없음).")
        return
    # 통과 기준: 끄면 v5와 같고, 켜면 어느 경우든 틀린 경우가 v5보다 늘지 않으며, 가로 2칸·3칸은 v5보다 많이 맞는다
    no_worse = all(r6[c][k][e] <= r5[c][k][e] for c in CAMS for k, _ in KINDS for e in ("extra", "miss"))
    better = all(r6[c][k]["ok"] > r5[c][k]["ok"] for c in CAMS for k in ("side2", "side3"))
    ok = same and no_worse and better
    print("확인:", "통과 (끄면 v5와 같음, 틀린 경우가 v5보다 늘지 않음, 가로 주차는 더 많이 맞음)" if ok else
          f"실패 (v5와 같음 {same}, 틀림 늘지 않음 {no_worse}, 가로 더 맞음 {better})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
