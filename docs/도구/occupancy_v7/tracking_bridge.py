"""Read-only adapter from Engine results to independent tracks and web snapshots."""
import copy
import threading
import time
from vehicle_tracker import Observation, TrackingConfig, VehicleTracker


class StateStore:
    """One immutable-by-contract snapshot, not a growing network queue.

    Copying and JSON serialization happen outside the lock. The inference thread
    only swaps a pointer; no sockets, callbacks or client locks on its path.
    """
    def __init__(self, layout):
        self.layout = copy.deepcopy(layout)
        self._lock = threading.Lock()
        self._value = {"vehicles": [], "slots": [], "cameras": [], "engine_status": "starting",
                       "updated_at": time.time(), "sequence": 0, "session_id": None}
        self._published = time.monotonic()

    def publish(self, value):
        with self._lock:
            self._value = value
            self._published = time.monotonic()

    def read(self):
        with self._lock:
            value, published = self._value, self._published
        result = copy.deepcopy(value)
        result['server_time'] = time.time()
        result['publication_age_s'] = max(0., time.monotonic() - published)
        return result


def layout_from_table(table):
    return {"units": "cm", "coordinate_system": "top_left_x_right_y_down",
            "mat": {"width_cm": float(table.mat_size[0]), "height_cm": float(table.mat_size[1])},
            "slots": [{"no": s.no, "row": s.row, "cell": s.cell.tolist(), "inner": s.inner.tolist()}
                      for s in table]}


# 웹 지도 "보일 차량" 처음 값 (설정 vehicle_tracking.parked_vehicles, docs/42). 서버는 늘 모든 차를 보내고,
# 거르기는 웹 화면(web/src/tracking.js shownByParking)이 보는 사람마다 한다. 화면에서 바꾼 값은 그 브라우저가 기억한다.
#   show          : 전체 차량 (기본)
#   hide_occupied : 주차 확정 차량 숨기기 — 차가 선 칸이 OCCUPIED(판정 끝)면 숨김. 들어가는 중(판정 전)에는 보임
#   aisle_only    : 주차칸 차량 모두 숨기기 — 칸에 넣은 차는 판정과 상관없이 숨김. 통로 차(kind aisle)만
PARKED_MODES = ("show", "hide_occupied", "aisle_only")


class TrackingBridge:
    def __init__(self, table, config=None, clock='wall', camera_ids=(), mode='detection',
                 contact_shift_cm=0.0):
        # v4: 통로 차(kind aisle) 접점을 카메라에서 먼 쪽으로 미는 거리(cm). 반대쪽 카메라 두 대가
        # 같은 통로 차를 차 폭만큼 다른 곳(가까운 옆면)에 찍는 차이를 줄인다(docs/36).
        # 설정 vehicle_tracking.aisle_contact_shift_cm 이 있으면 그 값, 없으면 contact_shift_cm(기본: 차 폭의 반).
        # 0이면 보정하지 않는다. 칸에 선 차(slot 등)의 접점은 그대로 보낸다
        config = dict(config or {})
        self.contact_shift_cm = float(config.pop("aisle_contact_shift_cm", contact_shift_cm))
        if not self.contact_shift_cm >= 0:
            raise ValueError("vehicle_tracking.aisle_contact_shift_cm must be zero or positive")
        self.parked_mode = str(config.pop("parked_vehicles", "show"))
        if self.parked_mode not in PARKED_MODES:
            raise ValueError(f"vehicle_tracking.parked_vehicles must be one of {PARKED_MODES}")
        self.config = TrackingConfig(**config)
        self.tracker = VehicleTracker(self.config)
        self.store = StateStore(layout_from_table(table))
        self.bounds = tuple(float(v) for v in table.mat_size)
        self.clock = clock
        self.mode = mode
        self.pending = []
        self.camera_times = {c: None for c in camera_ids}
        self.sequence = 0
        self.rejected_contacts = 0
        self.approx_contacts = 0       # 대략 위치로 보낸 점 수 (docs/43)

    def shifted_contact(self, b, geom, point=None):
        """추적기에 넣을 매트 점(기본은 접점, point를 주면 그 점). 통로 차만 far_dir(카메라 반대쪽)로
        contact_shift_cm 민다. geom(카메라 CamGeom)이 없거나 보정 거리가 0이면 그대로."""
        x, y = map(float, b.contact_mat if point is None else point)
        if geom is None or self.contact_shift_cm <= 0 or b.kind != 'aisle':
            return x, y
        d = geom.far_dir((x, y))
        return x + float(d[0]) * self.contact_shift_cm, y + float(d[1]) * self.contact_shift_cm

    def submit(self, camera_id, observed_at, result, geom=None):
        obs = []
        for b in result.boxes:
            if b.contact_mat is None:
                continue
            # 아래 변이 바닥 접점이 아닌 상자 (docs/43):
            #   아래가 다른 차에 가림 → 판정이 구한 대략 위치(approx_mat)가 있으면 그 점, 없으면 버림
            #   화면 옆 끝에 잘림 → 깊이(아래 변)는 맞으므로 접점 그대로, 화면 아래 끝에 잘림 → 버림
            # 대략 위치는 approx("cut"/"edge")를 붙여 보내고 웹 지도에 "위치 대략"으로 표시한다
            cut = getattr(b, 'cut_by', None) is not None or b.kind in ('uncertain', 'slot_cut')
            approx_mat = getattr(b, 'approx_mat', None)
            approx, ok = None, True
            if cut:
                approx, ok = 'cut', approx_mat is not None
            elif b.edge:
                approx, ok = 'edge', not getattr(b, 'edge_bottom', True)
            x, y = self.shifted_contact(b, geom, approx_mat if cut and ok else None)
            reliable = ok and 0 <= x <= self.bounds[0] and 0 <= y <= self.bounds[1]
            # v7: 가린 차는 시선 구간(대략 위치 → 잘린 접점)도 같이 보낸다. 추적기가 이미 아는 차면 구간 위에서
            # 그 차의 예상 위치와 가장 가까운 점을 쓴다 (docs/44)
            ray = None
            r = getattr(b, 'approx_ray', None)
            if cut and ok and r is not None:
                ray = ((x, y), self.shifted_contact(b, geom, r[1]))
            if not reliable:
                self.rejected_contacts += 1
            elif approx:
                self.approx_contacts += 1
            obs.append(Observation(camera_id, observed_at, x, y, b.conf, b.slot, b.kind, reliable, approx, ray))
        self.pending.append((observed_at, camera_id, obs))

    def publish(self, now, slots, camera_states, engine_status='running'):
        for t, camera_id, obs in sorted(self.pending, key=lambda b: (b[0], b[1])):
            if self.tracker.update(camera_id, t, obs, now=max(now, t)):
                self.camera_times[camera_id] = t
        self.pending.clear()
        now = max(now, self.tracker.now)
        if now == float('-inf'):
            now = 0.
        self.sequence += 1
        self.store.publish({
            "session_id": self.tracker.session_id, "sequence": self.sequence,
            "mode": self.mode,
            "updated_at": time.time(), "tracking_time": now, "clock": self.clock,
            "engine_status": engine_status,
            "vehicles": self.tracker.snapshot(now),
            "successors": self.tracker.successors(now),   # v7 내 차 후보: 옛 ID → 뒤를 이을 수 있는 새 ID (docs/44)
            "slots": [{"no": no, "status": s.status, "reason": s.reason, "last_known": s.last_known}
                      for no, s in sorted(slots.items())],
            "cameras": [{"id": cam, "status": state, "observed_at": self.camera_times.get(cam),
                         "valid": self.camera_times.get(cam) is not None and
                                  now - self.camera_times[cam] <= self.config.observation_ttl_s}
                        for cam, state in camera_states.items()],
            "tracking_config": {"lost_after_s": self.config.lost_after_s,
                                "remove_after_s": self.config.remove_after_s,
                                "aisle_contact_shift_cm": self.contact_shift_cm,
                                "parked_vehicles": self.parked_mode},
            "rejected_contacts": self.rejected_contacts,
            "approx_contacts": self.approx_contacts,
        })
