"""Independent vehicle IDs in mat centimetres; never modifies slot-state Tracker.

Geometry-only association is evidence, not proof of physical identity. Ambiguous
matches are withheld. No appearance Re-ID or body heading is inferred.
"""
from dataclasses import dataclass, field, replace
from typing import Optional  # Python 3.9 호환: 'int | None' 대신 Optional[int] (v3 합치기)
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
    heading_min_displacement_cm: float = 0.5
    heading_min_speed_cm_s: float = 1.0
    approx_radius_cm: float = 5.0      # v6: 대략 위치(가림·화면 끝)를 같은 차로 이을 때 더 주는 여유 (docs/43)
    # v7 (docs/44): 마주 지나갈 때 ID가 덜 끊기게
    reacquire_horizon_s: float = 3.0   # 놓친 차를 다시 찾을 때 지난 속도로 내다보는 최대 시간
    reacquire_radius_cm: float = 4.0   # 그 예측 위치에서 다시 잇는 기본 반경
    reacquire_speed_frac: float = 0.5  # 반경에 더함: 이 비율 × 속력 × 놓친 시간 (속도를 잘못 알았을 때의 여유)
    approx_penalty_cm: float = 0.7     # 대략 위치의 비교 값에 더함: 같은 차를 두고 다투면 정확한 점이 먼저 이어진다
    # v7 내 차 후보 (docs/44): ID가 끊기거나 섞였을 때 웹 지도가 내 차를 이어 주거나 후보로 보여 줄 수 있게
    link_radius_cm: float = 6.0        # 놓친 차가 갔을 법한 곳에서 이 반경 안에 새 ID가 생기면 "뒤를 이을 수 있는 차"
    link_speed_frac: float = 0.5       # 반경에 더함: 이 비율 × 속력 × 놓친 시간
    link_keep_s: float = 30.0          # 그 연결을 이만큼 기억해 웹에 보낸다

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
    approx: Optional[str] = None   # v6 대략 위치: "cut"(다른 차에 가림) / "edge"(화면 옆 끝) / None (docs/43)
    ray: Optional[tuple] = None    # v7 가린 차: 시선 위 구간 ((x0, y0), (x1, y1)). 진짜 위치는 이 구간 위 어딘가 (docs/44)


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
    motion_heading: Optional[float] = None
    sources: dict = field(default_factory=dict)
    status: str = "CANDIDATE"
    uncertain_until: float = -math.inf
    slot: Optional[int] = None
    kind: str = "aisle"
    related: set = field(default_factory=set)   # v7: 엇갈려 이 차와 ID가 섞였을 수 있는 차들 (docs/44)
    related_until: float = -math.inf


class VehicleTracker:
    def __init__(self, config=None, session_id=None):
        self.cfg = config or TrackingConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.next_id = 1
        self.tracks = {}
        self.camera_times = {}
        self.now = -math.inf
        self.last_observation_time = -math.inf
        self.links = []        # v7: [{"from": 옛 ID, "to": 새 ID, "err_cm": 예측과 거리, "at": 시각}] (docs/44)

    def _prediction(self, tr, t):
        dt = max(0., min(t - tr.last_seen, self.cfg.prediction_horizon_s))
        return (tr.raw[0] + tr.velocity[0] * dt, tr.raw[1] + tr.velocity[1] * dt)

    @staticmethod
    def _latest_approx(tr):
        last = max(tr.sources.values(), key=lambda o: o.observed_at, default=None)
        return bool(last is not None and last.approx)

    def _exact_elsewhere(self, tr, camera_id, t):
        """다른 카메라가 이 차를 아직 정확한 점으로 보고 있는지 (docs/43)."""
        return any(not o.approx and o.camera_id != camera_id and t - o.observed_at <= self.cfg.observation_ttl_s
                   for o in tr.sources.values())

    @staticmethod
    def _on_ray(ob, p):
        """가린 차 관측(ob.ray)이면 점 p에서 가장 가까운 구간 위의 점, 아니면 관측 점."""
        if not ob.ray:
            return (ob.x, ob.y)
        (x0, y0), (x1, y1) = ob.ray
        dx, dy = x1 - x0, y1 - y0
        n = dx * dx + dy * dy
        if n < 1e-12:
            return (x0, y0)
        u = max(0., min(1., ((p[0] - x0) * dx + (p[1] - y0) * dy) / n))
        return (x0 + u * dx, y0 + u * dy)

    def _ob_point(self, tr, ob):
        """이 차(tr)에 이을 때 쓸 관측 점 (docs/44). 가린 차는 시선 구간 위에서 이 차의 예상 위치와 가장 가까운 점."""
        if not ob.ray:
            return (ob.x, ob.y)
        c = self.cfg
        dt = ob.observed_at - tr.last_seen
        own = tr.sources.get(ob.camera_id)
        if own is not None and ob.observed_at - own.observed_at <= c.remove_after_s and dt > c.lost_after_s:
            h = min(dt, c.reacquire_horizon_s)
            pred = (tr.raw[0] + tr.velocity[0] * h, tr.raw[1] + tr.velocity[1] * h)
        else:
            pred = self._prediction(tr, ob.observed_at)
        return self._on_ray(ob, pred)

    def _cost(self, tr, ob):
        c = self.cfg
        dt = ob.observed_at - tr.last_seen
        if dt < -1e-6 or dt > c.remove_after_s or tr.status == "REMOVED":
            return None
        # v6: 대략 위치(가림·화면 끝)가 끼면 그 오차만큼 여유를 더 준다. 대략 위치가 없으면 예전과 같다 (docs/43)
        slack = c.approx_radius_cm if (ob.approx or self._latest_approx(tr)) else 0.
        p = self._ob_point(tr, ob)
        delta = distance(tr.raw, p)
        if delta > c.max_speed_cm_s * max(0., dt) + c.position_noise_cm + slack:
            return None
        err = distance(self._prediction(tr, ob.observed_at), p)
        gate = min(c.match_radius_cm, c.position_noise_cm + c.max_speed_cm_s * max(0., dt) + slack)
        cross = max(c.cross_camera_radius_cm, slack)
        penalty = c.approx_penalty_cm if ob.approx else 0.
        # A camera's old association cannot remain valid indefinitely.
        own = tr.sources.get(ob.camera_id)
        known = own is not None and ob.observed_at - own.observed_at <= c.remove_after_s
        if known and dt > c.lost_after_s:
            # v7: 같은 카메라로 놓친 차는 멈춘 자리 2cm가 아니라, 놓친 동안 지난 속도로 갔을 곳 근처에서 다시 찾는다 (docs/44)
            h = min(dt, c.reacquire_horizon_s)
            pred = (tr.raw[0] + tr.velocity[0] * h, tr.raw[1] + tr.velocity[1] * h)
            err = distance(pred, p)
            gate = min(c.match_radius_cm, c.reacquire_radius_cm + slack +
                       c.reacquire_speed_frac * math.hypot(*tr.velocity) * dt)
            return err + penalty if err <= gate else None
        if not known:
            if dt > c.cross_camera_window_s:
                return None
            gate = min(gate, cross)
        return err + penalty if err <= gate else None

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
        # v7: 애매해서 못 이은 차들은 서로 "섞였을 수 있는 차"로 표시한다(같은 점을 두고 다툰 차들) (docs/44)
        for j in ambiguous_cols:
            group = {i for i in range(len(tracks)) if costs[i][j] is not None}
            for i in group:
                if len(group) > 1:
                    tracks[i].related |= {tracks[k].id for k in group if k != i}
                    tracks[i].related_until = observed_at + c.lost_after_s
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
            # 대략 위치는, 다른 카메라가 정확히 보고 있는 차 가까이(approx_radius_cm 안)면 새 차로 만들지 않는다:
            # 화면 끝에 걸린 카메라의 어긋난 점이 같은 차를 한 대 더 만드는 것을 막는다 (docs/43)
            if o.approx and any(distance(tr.raw, (o.x, o.y)) <= c.approx_radius_cm and
                                self._exact_elsewhere(tr, camera_id, observed_at) for tr in tracks):
                continue
            gid = f"{self.session_id}-{self.next_id:04d}"
            self.next_id += 1
            self._link_new(gid, o, observed_at)
            tr = Vehicle(gid, (o.x, o.y), (o.x, o.y), observed_at, observed_at, observed_at,
                         sources={camera_id: o}, slot=o.slot, kind=o.kind)
            self.tracks[gid] = tr
        self.advance(max(self.now, observed_at))
        return True

    def _link_new(self, gid, o, t):
        """새 ID가 생길 때: 잠시 놓친 차(최근 갱신이 없거나 놓침·애매함)가 지난 속도로 갔을 곳 가까이면
        "옛 ID → 새 ID" 연결을 남긴다. 웹 지도가 내 차를 이어 주거나 후보로 보여 줄 때 쓴다 (docs/44)."""
        c = self.cfg
        for tr in self.tracks.values():
            gap = t - tr.last_seen
            if gap <= 1e-6 or gap > c.remove_after_s + c.removed_retention_s:
                continue
            if tr.status not in ("LOST", "UNCERTAIN", "REMOVED") and gap < 0.4:
                continue                       # 방금도 보인 차는 새 차의 앞 ID가 아니다
            h = min(gap, c.reacquire_horizon_s)
            pred = (tr.raw[0] + tr.velocity[0] * h, tr.raw[1] + tr.velocity[1] * h)
            p = self._on_ray(o, pred)
            err = distance(pred, p)
            slack = c.approx_radius_cm if o.approx else 0.
            if err <= c.link_radius_cm + slack + c.link_speed_frac * math.hypot(*tr.velocity) * gap:
                self.links.append({"from": tr.id, "to": gid, "err_cm": round(err, 2), "at": t})

    def successors(self, now):
        """옛 ID → 뒤를 이을 수 있는 새 ID 목록 (link_keep_s 안의 연결만, 예측과 가까운 순)."""
        self.links = [l for l in self.links if now - l["at"] <= self.cfg.link_keep_s]
        out = {}
        for l in sorted(self.links, key=lambda l: l["err_cm"]):
            out.setdefault(l["from"], []).append({"id": l["to"], "err_cm": l["err_cm"], "at": l["at"]})
        return out

    def _accept(self, tr, ob):
        c = self.cfg
        if ob.ray:
            p = self._ob_point(tr, ob)            # 시선 구간 위에서 이 차에 맞는 점으로 바꿔 쓴다 (docs/44)
            ob = replace(ob, x=p[0], y=p[1], ray=None)
        t, pt = ob.observed_at, (ob.x, ob.y)
        if ob.approx and self._exact_elsewhere(tr, ob.camera_id, t):
            # 다른 카메라가 정확히 보고 있으면 대략 위치로 옮기지 않고, 이 카메라도 이 차를 본다는 것만 남긴다 (docs/43)
            tr.sources[ob.camera_id] = ob
            return
        dt = t - tr.last_seen
        prev = tr.sources.get(ob.camera_id)
        # Estimate motion from SAME camera, avoiding a false velocity caused by
        # different cameras' contact-point biases at the same instant.
        # v6: 대략 위치와 정확한 점 사이의 움직임은 어긋남이라 속도로 보지 않는다 (docs/43)
        if prev and prev.approx == ob.approx and 0.02 <= t - prev.observed_at <= c.lost_after_s:
            elapsed = t - prev.observed_at
            vel = ((pt[0] - prev.x) / elapsed, (pt[1] - prev.y) / elapsed)
            speed = math.hypot(*vel)
            # Same-camera displacement only: camera handoff offsets are not motion.
            # SVG/map convention: 0 degrees up, 90 right, clockwise positive.
            if (distance((prev.x, prev.y), pt) >= c.heading_min_displacement_cm
                    and speed >= c.heading_min_speed_cm_s):
                tr.motion_heading = math.degrees(math.atan2(vel[0], -vel[1])) % 360
            # Below the deadband, retain the last direction (including at rest).
            factor = min(1., c.max_speed_cm_s / max(speed, 1e-9))
            tr.velocity = tuple(.3 * a + .7 * b * factor for a, b in zip(tr.velocity, vel))
        elif dt > c.lost_after_s:
            tr.velocity = (0., 0.)
        tr.sources[ob.camera_id] = ob
        if dt <= 1e-6:
            simultaneous = [o for o in tr.sources.values() if abs(o.observed_at - t) < 1e-6]
            exact = [o for o in simultaneous if not o.approx]
            simultaneous = exact or simultaneous       # 같은 순간 정확한 점이 있으면 대략 위치는 섞지 않는다 (docs/43)
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
            # 위치 대략: 아직 유효한 관측이 모두 대략 위치면 그 이유, 유효한 관측이 없으면 마지막 관측의 이유 (docs/43)
            valid = [o for o in tr.sources.values() if now - o.observed_at <= self.cfg.observation_ttl_s]
            last = max(tr.sources.values(), key=lambda o: o.observed_at, default=None)
            approx = (max(valid, key=lambda o: o.observed_at).approx if valid and all(o.approx for o in valid)
                      else None if valid else (last.approx if last else None))
            related = sorted(tr.related) if now <= tr.related_until else []
            if now > tr.related_until:
                tr.related = set()
            out.append({"id": tr.id, "approx": approx, "related": related, "first_seen": tr.first_seen, "x_cm": tr.position[0], "y_cm": tr.position[1],
                        "observed_x_cm": tr.raw[0], "observed_y_cm": tr.raw[1],
                        "status": tr.status, "last_seen": tr.last_seen, "age_s": age,
                        "heading": None, "motion_heading": tr.motion_heading,
                        "heading_source": "motion" if tr.motion_heading is not None else "unknown", "slot": tr.slot, "kind": tr.kind,
                        "association": "geometry_only", "confirmed": tr.hits >= self.cfg.confirm_hits,
                        "uncertain": tr.status != "TRACKED" or age > self.cfg.observation_ttl_s,
                        "uncertainty_cm": self.cfg.position_noise_cm + self.cfg.max_speed_cm_s * age,
                        "observations": [{"camera_id": o.camera_id, "observed_at": o.observed_at,
                                          "x_cm": o.x, "y_cm": o.y,
                                          "valid": now - o.observed_at <= self.cfg.observation_ttl_s}
                                         for o in tr.sources.values()]})
        return out
