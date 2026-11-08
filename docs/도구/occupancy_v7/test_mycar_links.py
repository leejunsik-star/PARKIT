"""내 차 후보 서버 쪽 시험 (docs/44): related(섞였을 수 있는 차), successors(뒤를 이을 수 있는 차).

웹 쪽 규칙(myCarState)은 web에서 npm test로 본다.
실행: python test_mycar_links.py   (마지막 줄 "합계: N / N 통과")
"""
import sys

from vehicle_tracker import Observation, TrackingConfig, VehicleTracker
import test_crossing as tc

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def feed(tr, t, pts):
    tr.update("A", t, [Observation("A", t, x, y) for x, y in pts], now=t)


def main():
    print("a. 놓친 차 뒤에 새 ID가 생기면 연결 (successors)")
    tr = VehicleTracker(TrackingConfig(reacquire_radius_cm=0.01, reacquire_speed_frac=1e-6), session_id="s")
    for i in range(6):                                   # 0.5초마다 오른쪽으로 초속 5cm
        feed(tr, i * 0.5, [(20 + i * 2.5, 15)])
    old = next(iter(tr.tracks))
    feed(tr, 4.5, [(20 + 9 * 2.5, 15), (60, 40)])       # 2초 놓친 뒤: 갔을 법한 곳 1대 + 먼 곳 1대
    succ = tr.successors(4.5)
    new_ids = [v.id for v in tr.tracks.values() if v.id != old]
    near = [v.id for v in tr.tracks.values() if v.id != old and v.raw[0] < 50]
    check("갔을 법한 곳의 새 ID만 이어짐 (먼 곳 새 차는 아님)", succ.get(old) and [l["id"] for l in succ[old]] == near,
          f"{succ}")
    check("연결은 link_keep_s(30초) 뒤 사라짐", not tr.successors(4.5 + 31).get(old))
    st = tr.snapshot(4.5)
    check("차량마다 related, first_seen이 나감", all("related" in v and "first_seen" in v for v in st))

    print("\nb. 엇갈려 애매하면 서로 related")
    tr = VehicleTracker(TrackingConfig(), session_id="s")
    feed(tr, 0.0, [(30, 15), (34, 15.5)])
    feed(tr, 0.5, [(31, 15), (33, 15.5)])
    feed(tr, 1.0, [(32, 15.2), (32.1, 15.3)])           # 두 점이 거의 겹침 → 애매
    ids = list(tr.tracks)
    rel = {v["id"]: v["related"] for v in tr.snapshot(1.0)}
    check("두 차가 서로를 related로 가짐", rel.get(ids[0]) == [ids[1]] and rel.get(ids[1]) == [ids[0]], f"{rel}")
    check("lost_after_s 지나면 related 비워짐", all(not v["related"] for v in tr.snapshot(1.0 + 1.3)))

    print("\nc. 합성 카메라에서 일부러 ID를 끊었을 때(다시 찾기 끔) 웹 규칙으로 내 차를 이어 가는지")
    t = tc.run_all({"reacquire_radius_cm": 0.01, "reacquire_speed_frac": 1e-6}, verbose=False, scenes=("cross",))
    check(f"마주 지나감 {t['경우']}가지: 끊김 {t['끊김']}번이어도 내 차를 끝까지 {t['내 차 끝까지']}가지, 잘못 넘어감 {t['내 차 잘못 넘어감']}",
          t["끊김"] > 0 and t["내 차 잘못 넘어감"] == 0 and t["내 차 끝까지"] >= 0.9 * t["경우"],
          f"자동 이음 {t['자동 이음']}가지, 확인 필요 {t['확인 필요']}가지")

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n합계: {n_ok} / {len(RESULTS)} 통과")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
