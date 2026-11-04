"""PARKIT 점유 판정 — 칸 상태 기계 (25 문서 P6, P7, P9, P10 / 02 문서 T1~T8)

카메라마다 한 프레임 판정(parkit_judge) 결과를 받아 칸마다 합치고(P9), 시간을 걸러(P6) 상태를 정한다.
상태가 바뀌면 등록한 콜백(on_change)들을 부른다. 기록 파일(L1)과 나중의 Firebase가 여기에 붙는다.

상태
    EMPTY, OCCUPIED, DYNAMIC_BLIND, STATIC_BLIND
DYNAMIC_BLIND 이유(reason)
    INITIALIZING(켠 직후), OCCLUDED(가림), CAMERA_OFFLINE(프레임이 안 옴), CAMERA_STOPPED(해상도가 달라 멈춤)
STATIC_BLIND 이유
    NO_COVERAGE(어느 카메라의 감시 칸도 아님)
"""

from parkit_judge import ABSENT, OCCLUDED, OFFLINE, PRESENT

EMPTY, OCCUPIED, DYNAMIC_BLIND, STATIC_BLIND = "EMPTY", "OCCUPIED", "DYNAMIC_BLIND", "STATIC_BLIND"


class SlotState:
    def __init__(self, no, cams):
        self.no = no
        self.cams = list(cams)              # 이 칸을 감시하는 카메라
        if cams:
            self.status, self.reason = DYNAMIC_BLIND, "INITIALIZING"      # T6
        else:
            self.status, self.reason = STATIC_BLIND, "NO_COVERAGE"        # T7
        self.last_known = None
        self.since = None                   # 지금 상태가 된 시각
        self.obs = None                     # 이번 합친 판정
        self.obs_cams = {}                  # 카메라별 이번 판정
        self.streak = None                  # PRESENT / ABSENT 가 이어진 종류
        self.streak_start = None
        self.occl_start = None
        self.why = ""

    def remaining(self, t, p):
        """상태가 바뀌기까지 남은 초(D3). 바뀔 것이 없으면 None."""
        if self.status == STATIC_BLIND or self.obs is None:
            return None
        if self.obs == OCCLUDED and self.status in (EMPTY, OCCUPIED) and self.occl_start is not None:
            return max(0.0, p["t_blind_s"] - (t - self.occl_start))
        if self.streak is None:
            return None
        target = OCCUPIED if self.streak == PRESENT else EMPTY
        if self.status == target:
            return None
        need = p["t_occupied_s"] if target == OCCUPIED else p["t_empty_s"]
        return max(0.0, need - (t - self.streak_start))


def merge_obs(per_cam):
    """P9: 카메라별 판정 → 하나. 가려지지 않은 카메라 결과를 쓰고, 그중 하나라도 차 있음이면 차 있음."""
    vals = list(per_cam.values())
    normal = [o for o in vals if o in (PRESENT, ABSENT)]
    if normal:
        return PRESENT if PRESENT in normal else ABSENT
    if OCCLUDED in vals:
        return OCCLUDED
    if any(o is None for o in vals):
        return None                    # 모름: 이번에는 상태를 건드리지 않는다
    return OFFLINE


class Tracker:
    """coverage: {칸 번호: [카메라 이름, ...]} (감시 칸이 없는 칸은 빈 목록이나 빠짐)
    params(no) -> 판정 값 dict. all_slots: 칸 번호 전체 (STATIC_BLIND 포함)."""

    def __init__(self, all_slots, coverage, params):
        self.params = params
        self.slots = {no: SlotState(no, coverage.get(no, [])) for no in sorted(all_slots)}
        self.callbacks = []
        self.started = False

    def on_change(self, fn):
        """fn(event dict). event: time, slot, prev, status, reason, last_known, cams, why"""
        self.callbacks.append(fn)

    def _emit(self, s, t, prev, prev_reason):
        ev = {"time": t, "slot": s.no, "prev": prev, "prev_reason": prev_reason, "status": s.status,
              "reason": s.reason, "last_known": s.last_known,
              "cams": ",".join(f"{c}:{o}" for c, o in sorted(s.obs_cams.items())), "why": s.why}
        for fn in self.callbacks:
            fn(ev)

    def _set(self, s, t, status, reason=None):
        if s.status == status and s.reason == reason:
            return
        prev, prev_reason = s.status, s.reason
        if status == DYNAMIC_BLIND:
            if prev in (EMPTY, OCCUPIED):
                s.last_known = prev           # 02 문서 lastKnownStatus
        else:
            s.last_known = None
        s.status, s.reason, s.since = status, reason, t
        self._emit(s, t, prev, prev_reason)

    def start(self, t):
        """처음 상태(T6, T7)를 기록으로 내보낸다."""
        if self.started:
            return
        self.started = True
        for s in self.slots.values():
            s.since = t
            self._emit(s, t, None, None)

    def update(self, t, cam_obs, cam_state):
        """t: 지금 시각(초). cam_obs: {카메라: {칸: PRESENT/ABSENT/OCCLUDED}} (그 카메라의 마지막 판정).
        cam_state: {카메라: "ok" / "stale"(잠깐 프레임 없음) / "offline" / "stopped"}."""
        self.start(t)
        for s in self.slots.values():
            if not s.cams:
                continue
            p = self.params(s.no)
            per = {}
            for c in s.cams:
                st = cam_state.get(c, "offline")
                if st == "stale":
                    per[c] = None              # 잠깐 프레임이 안 옴: 이 카메라는 모름(다른 카메라로 정한다)
                elif st != "ok":
                    per[c] = OFFLINE
                else:
                    per[c] = (cam_obs.get(c) or {}).get(s.no)
            s.obs_cams = per
            obs = merge_obs(per)
            s.obs = obs
            if obs is None:
                continue                       # 판정이 없음 (켠 직후, 잠깐 끊김, P3를 판정 못한 옆 칸)
            if obs == OFFLINE:
                # P10 / C6: 이 칸을 보는 카메라가 모두 끊김
                reason = "CAMERA_STOPPED" if all(cam_state.get(c) == "stopped" for c in s.cams) else "CAMERA_OFFLINE"
                s.streak = s.streak_start = s.occl_start = None
                self._set(s, t, DYNAMIC_BLIND, reason)
                continue
            if obs == OCCLUDED:
                # P5: 가려짐이 t_blind_s 넘게 이어지면 DYNAMIC_BLIND. 그동안 시간 거르기는 멈추고 처음부터 다시 센다(T5)
                if s.occl_start is None:
                    s.occl_start = t
                s.streak = s.streak_start = None
                if s.status in (EMPTY, OCCUPIED) and t - s.occl_start > p["t_blind_s"]:
                    self._set(s, t, DYNAMIC_BLIND, "OCCLUDED")
                continue
            s.occl_start = None
            if s.streak != obs:
                s.streak, s.streak_start = obs, t
            target = OCCUPIED if obs == PRESENT else EMPTY
            need = p["t_occupied_s"] if target == OCCUPIED else p["t_empty_s"]
            if s.status != target and t - s.streak_start >= need - 1e-9:
                self._set(s, t, target, None)

    def set_why(self, cam_why):
        """D3, L1에 쓸 이유 글. cam_why: {카메라: {칸: 글}}"""
        for s in self.slots.values():
            parts = [f"{c}:{w[s.no]}" for c, w in cam_why.items() if w.get(s.no)]
            s.why = " | ".join(parts)

    def counts(self):
        out = {EMPTY: 0, OCCUPIED: 0, DYNAMIC_BLIND: 0, STATIC_BLIND: 0}
        for s in self.slots.values():
            out[s.status] += 1
        return out
