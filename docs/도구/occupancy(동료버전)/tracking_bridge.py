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


class TrackingBridge:
    def __init__(self, table, config=None, clock='wall', camera_ids=(), mode='detection'):
        self.config = TrackingConfig(**(config or {}))
        self.tracker = VehicleTracker(self.config)
        self.store = StateStore(layout_from_table(table))
        self.bounds = tuple(float(v) for v in table.mat_size)
        self.clock = clock
        self.mode = mode
        self.pending = []
        self.camera_times = {c: None for c in camera_ids}
        self.sequence = 0
        self.rejected_contacts = 0

    def submit(self, camera_id, observed_at, result):
        obs = []
        for b in result.boxes:
            if b.contact_mat is None:
                continue
            x, y = map(float, b.contact_mat)
            # Truncated bottom edges are not reliable ground contacts. Preserve
            # the old vehicle as missing instead of moving it to a false point.
            reliable = (b.cut_by is None and b.kind not in ('uncertain', 'slot_cut') and not b.edge
                        and 0 <= x <= self.bounds[0] and 0 <= y <= self.bounds[1])
            if not reliable:
                self.rejected_contacts += 1
            obs.append(Observation(camera_id, observed_at, x, y, b.conf, b.slot, b.kind, reliable))
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
            "slots": [{"no": no, "status": s.status, "reason": s.reason, "last_known": s.last_known}
                      for no, s in sorted(slots.items())],
            "cameras": [{"id": cam, "status": state, "observed_at": self.camera_times.get(cam),
                         "valid": self.camera_times.get(cam) is not None and
                                  now - self.camera_times[cam] <= self.config.observation_ttl_s}
                        for cam, state in camera_states.items()],
            "tracking_config": {"lost_after_s": self.config.lost_after_s,
                                "remove_after_s": self.config.remove_after_s},
            "rejected_contacts": self.rejected_contacts,
        })
