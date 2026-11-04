"""Independent vehicle IDs in mat centimetres; never modifies slot-state Tracker.

Geometry-only association is evidence, not proof of physical identity. Ambiguous
matches are withheld. No appearance Re-ID or body heading is inferred.
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import uuid


@dataclass
class TrackingConfig:
    max_speed_cm_s: float = 35.0
    position_noise_cm: float = 1.0
    match_radius_cm: float = 10.0
    cross_camera_radius_cm: float = 2.0
    cross_camera_window_s: float = 1.0
    observation_ttl_s: float = 1.5
    ambiguity_margin_cm: float = 0.65
    confirm_hits: int = 3
    confirm_interval_s: float = 0.1
    lost_after_s: float = 1.2
    remove_after_s: float = 4.0
    candidate_timeout_s: float = 1.5
    removed_retention_s: float = 2.0
    smoothing_tau_s: float = 0.15
    prediction_horizon_s: float = 0.7

    def __post_init__(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"tracking.{key} must be positive and finite")
        if not isinstance(self.confirm_hits, int):
            raise ValueError("confirm_hits must be an integer")
        if self.remove_after_s <= self.lost_after_s:
            raise ValueError("remove_after_s must exceed lost_after_s")


@dataclass(frozen=True)
class Observation:
    camera_id: str
    observed_at: float       # engine clock: relative wall seconds OR video seconds
    x: float
    y: float
    confidence: float = 1.0
    slot: Optional[int] = None
    kind: str = "aisle"
    reliable: bool = True


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign(costs):
    """Rectangular minimum-cost one-to-one assignment (rows <= columns).

    Hungarian augmenting paths, O(rows**2 * columns). Caller supplies dummy
    unmatched columns so prohibited matches are never required. No SciPy needed.
    """
    n = len(costs)
    if not n:
        return []
    m = len(costs[0])
    if m < n:
        raise ValueError("assignment requires at least as many columns as rows")
    u, v, p, way = [0.] * (n + 1), [0.] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


@dataclass
class Vehicle:
    id: str
    raw: tuple
    position: tuple
    first_seen: float
    last_seen: float
    last_hit: float
    hits: int = 1
    velocity: tuple = (0., 0.)
    sources: dict = field(default_factory=dict)
    status: str = "CANDIDATE"
    uncertain_until: float = -math.inf
    slot: Optional[int] = None
    kind: str = "aisle"


class VehicleTracker:
    def __init__(self, config=None, session_id=None):
        self.cfg = config or TrackingConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.next_id = 1
        self.tracks = {}
        self.camera_times = {}
        self.now = -math.inf
        self.last_observation_time = -math.inf

    def _prediction(self, tr, t):
        dt = max(0., min(t - tr.last_seen, self.cfg.prediction_horizon_s))
        return (tr.raw[0] + tr.velocity[0] * dt, tr.raw[1] + tr.velocity[1] * dt)

    def _cost(self, tr, ob):
        c = self.cfg
        dt = ob.observed_at - tr.last_seen
        if dt < -1e-6 or dt > c.remove_after_s or tr.status == "REMOVED":
            return None
        delta = distance(tr.raw, (ob.x, ob.y))
        if delta > c.max_speed_cm_s * max(0., dt) + c.position_noise_cm:
            return None
        err = distance(self._prediction(tr, ob.observed_at), (ob.x, ob.y))
        gate = min(c.match_radius_cm, c.position_noise_cm + c.max_speed_cm_s * max(0., dt))
        # A camera's old association cannot remain valid indefinitely.
        own = tr.sources.get(ob.camera_id)
        known = own is not None and ob.observed_at - own.observed_at <= c.remove_after_s
        if known and dt > c.lost_after_s:
            gate = min(gate, c.cross_camera_radius_cm)
        if not known:
            if dt > c.cross_camera_window_s:
                return None
            gate = min(gate, c.cross_camera_radius_cm)
        return err if err <= gate else None

    def update(self, camera_id, observed_at, observations, now=None):
        """Consume a camera frame once. Caller sorts pending frames by timestamp.

        Late frames are rejected rather than rewinding tracks; empty frames also
        advance camera freshness. No stale frame is counted again as a new hit.
        """
        c = self.cfg
        now = observed_at if now is None else now
        if not math.isfinite(observed_at) or not math.isfinite(now):
            return False
        if observed_at <= self.camera_times.get(camera_id, -math.inf):
            return False
        if observed_at < self.last_observation_time - 1e-6 or observed_at > now + 1e-6 or now - observed_at > c.observation_ttl_s:
            return False
        self.advance(observed_at)
        self.last_observation_time = observed_at
        self.camera_times[camera_id] = observed_at
        obs = [o for o in observations if o.camera_id == camera_id and o.observed_at == observed_at
               and o.reliable and all(math.isfinite(x) for x in (o.x, o.y, o.confidence))]
        tracks = [tr for tr in self.tracks.values() if tr.status != "REMOVED"]
        invalid = 1e9
        costs = [[self._cost(tr, o) for o in obs] for tr in tracks]
        # Reject close alternatives on BOTH sides. Do not force a Hungarian tie
        # into a confident identity at a crossing or in overlap between cameras.
        ambiguous_rows, ambiguous_cols = set(), set()
        for i, row in enumerate(costs):
            vals = sorted((v, j) for j, v in enumerate(row) if v is not None)
            if len(vals) > 1 and vals[1][0] - vals[0][0] <= c.ambiguity_margin_cm:
                ambiguous_rows.add(i)
                ambiguous_cols.update(j for v, j in vals if v <= vals[0][0] + c.ambiguity_margin_cm)
        for j in range(len(obs)):
            vals = sorted((costs[i][j], i) for i in range(len(tracks)) if costs[i][j] is not None)
            if len(vals) > 1 and vals[1][0] - vals[0][0] <= c.ambiguity_margin_cm:
                ambiguous_cols.add(j)
                ambiguous_rows.update(i for v, i in vals if v <= vals[0][0] + c.ambiguity_margin_cm)
        for i in ambiguous_rows:
            tracks[i].uncertain_until = observed_at + c.lost_after_s
        matrix = [[v if v is not None and i not in ambiguous_rows and j not in ambiguous_cols else invalid
                   for j, v in enumerate(row)] + [c.match_radius_cm + 1] * len(tracks)
                  for i, row in enumerate(costs)]
        used = set()
        for i, j in assign(matrix):
            if j >= len(obs) or matrix[i][j] >= invalid:
                continue
            self._accept(tracks[i], obs[j])
            used.add(j)
        for j, o in enumerate(obs):
            # Withhold candidates involved in ambiguous association as well.
            if j in used or j in ambiguous_cols or any(costs[i][j] is not None for i in ambiguous_rows):
                continue
            gid = f"{self.session_id}-{self.next_id:04d}"
            self.next_id += 1
            tr = Vehicle(gid, (o.x, o.y), (o.x, o.y), observed_at, observed_at, observed_at,
                         sources={camera_id: o}, slot=o.slot, kind=o.kind)
            self.tracks[gid] = tr
        self.advance(max(self.now, observed_at))
        return True

    def _accept(self, tr, ob):
        c = self.cfg
        t, pt = ob.observed_at, (ob.x, ob.y)
        dt = t - tr.last_seen
        prev = tr.sources.get(ob.camera_id)
        # Estimate motion from SAME camera, avoiding a false velocity caused by
        # different cameras' contact-point biases at the same instant.
        if prev and 0.02 <= t - prev.observed_at <= c.lost_after_s:
            elapsed = t - prev.observed_at
            vel = ((pt[0] - prev.x) / elapsed, (pt[1] - prev.y) / elapsed)
            speed = math.hypot(*vel)
            factor = min(1., c.max_speed_cm_s / max(speed, 1e-9))
            tr.velocity = tuple(.3 * a + .7 * b * factor for a, b in zip(tr.velocity, vel))
        elif dt > c.lost_after_s:
            tr.velocity = (0., 0.)
        tr.sources[ob.camera_id] = ob
        if dt <= 1e-6:
            simultaneous = [o for o in tr.sources.values() if abs(o.observed_at - t) < 1e-6]
            pt = (sum(o.x for o in simultaneous) / len(simultaneous),
                  sum(o.y for o in simultaneous) / len(simultaneous))
            alpha = .5
        else:
            alpha = 1 - math.exp(-dt / c.smoothing_tau_s)
        tr.position = tuple(a + alpha * (b - a) for a, b in zip(tr.position, pt))
        tr.raw, tr.last_seen = pt, t
        tr.slot, tr.kind = ob.slot, ob.kind
        if t - tr.last_hit >= c.confirm_interval_s - 1e-9:
            tr.hits += 1
            tr.last_hit = t
        tr.uncertain_until = -math.inf

    def advance(self, now):
        if now < self.now:
            return
        self.now = now
        c = self.cfg
        for gid, tr in list(self.tracks.items()):
            age = now - tr.last_seen
            ttl = c.candidate_timeout_s if tr.hits < c.confirm_hits else c.remove_after_s
            if age > ttl + c.removed_retention_s:
                del self.tracks[gid]
                continue
            if age > ttl:
                tr.status = "REMOVED"
            elif age > c.lost_after_s:
                tr.status = "LOST"
            elif now <= tr.uncertain_until:
                tr.status = "UNCERTAIN"
            elif tr.hits < c.confirm_hits:
                tr.status = "CANDIDATE"
            else:
                tr.status = "TRACKED"

    def snapshot(self, now):
        self.advance(now)
        out = []
        for tr in self.tracks.values():
            age = max(0., now - tr.last_seen)
            out.append({"id": tr.id, "x_cm": tr.position[0], "y_cm": tr.position[1],
                        "observed_x_cm": tr.raw[0], "observed_y_cm": tr.raw[1],
                        "status": tr.status, "last_seen": tr.last_seen, "age_s": age,
                        "heading": None, "motion_heading": None,
                        "heading_source": "unknown", "slot": tr.slot, "kind": tr.kind,
                        "association": "geometry_only", "confirmed": tr.hits >= self.cfg.confirm_hits,
                        "uncertain": tr.status != "TRACKED" or age > self.cfg.observation_ttl_s,
                        "uncertainty_cm": self.cfg.position_noise_cm + self.cfg.max_speed_cm_s * age,
                        "observations": [{"camera_id": o.camera_id, "observed_at": o.observed_at,
                                          "x_cm": o.x, "y_cm": o.y,
                                          "valid": now - o.observed_at <= self.cfg.observation_ttl_s}
                                         for o in tr.sources.values()]})
        return out
