"""PARKIT 점유 판정 — 기하 계산 (25 문서 C3~C6, 26 문서 3.2·5절)

화면 코드와 섞지 않는다. 보정 도구, 판정 프로그램, 합성 시험이 같이 쓴다.

좌표
    매트: 15~27번 줄을 위로 두고 왼쪽 위 모서리가 (0, 0), x 오른쪽, y 아래쪽, cm.
    3차원: 매트 좌표에 z를 더한다. z는 매트 안쪽(아래)이 +라서, 높이 h인 점은 z = -h.
    화면: 픽셀 (u, v), 왼쪽 위가 (0, 0).

호모그래피 H는 "화면 픽셀 → 매트 cm"이다(22 문서 3.2절). 카메라 자세(rvec, tvec)는
높이가 있는 점(차 상자, 칸을 올린 모양)을 화면에 투영할 때만 쓴다.
"""

import json, math

import cv2
import numpy as np


# ---------------------------------------------------------------- 칸 표 (C3)

class Slot:
    """칸 하나. cell, inner는 [왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래] 순서(매트 좌표)."""

    def __init__(self, d):
        self.no = int(d["no"])
        self.row = str(d["row"])
        self.cell = np.array(d["cell"], float)
        self.inner = np.array(d["inner"], float)
        self.entrance_y = float(d["entrance_y"])
        self.measured = d.get("x_source") == "실측"
        self.x0, self.x1 = float(self.cell[:, 0].min()), float(self.cell[:, 0].max())
        self.y0, self.y1 = float(self.cell[:, 1].min()), float(self.cell[:, 1].max())
        self.ix0, self.ix1 = float(self.inner[:, 0].min()), float(self.inner[:, 0].max())
        self.center = np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2])
        # 입구가 y가 작은 쪽인지 (1~14번 줄은 True, 28~41번 줄은 False)
        self.entrance_low = abs(self.entrance_y - self.y0) <= abs(self.entrance_y - self.y1)
        self.left = None   # 같은 줄에서 x가 작은 쪽 이웃 칸 번호
        self.right = None  # x가 큰 쪽 이웃 칸 번호

    def zone_y(self, margin_entrance, margin_back):
        """판정 구역의 깊이 범위 (cell을 입구 쪽, 안쪽으로 늘린 것, 22 문서 4.1절)."""
        if self.entrance_low:
            return self.y0 - margin_entrance, self.y1 + margin_back
        return self.y0 - margin_back, self.y1 + margin_entrance


class SlotTable:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        info = d.get("_설명", {})
        self.mat_size = tuple(info.get("매트", [78.3, 55.4]))
        self.slots = {}
        for s in d["slots"]:
            if "cell" in s and "inner" in s:
                sl = Slot(s)
                self.slots[sl.no] = sl
        self.rows = {}
        for sl in self.slots.values():
            self.rows.setdefault(sl.row, []).append(sl.no)
        # 같은 줄에서 cell 가장자리가 맞닿은 칸을 이웃으로 둔다(P3)
        for nos in self.rows.values():
            nos.sort(key=lambda n: self.slots[n].x0)
            for a, b in zip(nos, nos[1:]):
                sa, sb = self.slots[a], self.slots[b]
                if abs(sa.x1 - sb.x0) < 0.3:
                    sa.right, sb.left = b, a
        self.path = path

    def __iter__(self):
        return iter(self.slots[n] for n in sorted(self.slots))

    def __getitem__(self, no):
        return self.slots[int(no)]

    def slot_at(self, p, use_inner=False):
        """매트 좌표 p가 들어간 칸 번호(없으면 None). 평면도 클릭에 쓴다."""
        for sl in self:
            poly = sl.inner if use_inner else sl.cell
            if cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0:
                return sl.no
        return None


# 기준점 종류: 코드, 설명, (inner, 입구/안쪽, x 작은/큰 쪽)
# 칸 사이 U선이 없는 주차장도 있으므로, 어느 주차장에나 있는 "빈 공간(inner)의 네 모서리"만 쓴다.
REF_KINDS = [
    ("IE-", "입구 끝 · x 작은 쪽", "inner", "ent", 0),
    ("IE+", "입구 끝 · x 큰 쪽", "inner", "ent", 1),
    ("IB-", "안쪽 끝(카스토퍼) · x 작은 쪽", "inner", "back", 0),
    ("IB+", "안쪽 끝(카스토퍼) · x 큰 쪽", "inner", "back", 1),
]


def ref_points(table):
    """보정에 쓸 기준점 목록. 실측 줄이 앞에 온다(K2).
    돌려주는 것: [{"id": "4-IE-", "slot": 4, "kind": "IE-", "label": 설명, "mat": (x, y), "measured": bool}]"""
    out = []
    for sl in sorted(table, key=lambda s: (not s.measured, s.no)):
        for code, label, which, end, side in REF_KINDS:
            poly = sl.inner if which == "inner" else sl.cell
            xs = (poly[:, 0].min(), poly[:, 0].max())
            y_ent = sl.entrance_y
            y_back = sl.y1 if sl.entrance_low else sl.y0
            y = y_ent if end == "ent" else y_back
            out.append({"id": f"{sl.no}-{code}", "slot": sl.no, "kind": code,
                        "label": f"{sl.no}번 칸 {label}", "mat": (float(xs[side]), float(y)),
                        "measured": sl.measured})
    w, h = table.mat_size
    for code, label, p in [("TL", "왼쪽 위", (0, 0)), ("TR", "오른쪽 위", (w, 0)),
                           ("BR", "오른쪽 아래", (w, h)), ("BL", "왼쪽 아래", (0, h))]:
        out.append({"id": f"MAT-{code}", "slot": None, "kind": code, "label": f"매트 모서리 {label}",
                    "mat": (float(p[0]), float(p[1])), "measured": True})
    return out


# ---------------------------------------------------------------- 카메라 (C4)

def intrinsics(width, height, hfov_deg):
    """가로 화각으로 초점거리를 정한 카메라 행렬. 렌즈 왜곡은 없다고 본다."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


def to3d(pts, z=0.0):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.hstack([pts, np.full((len(pts), 1), z)])


def apply_h(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, float)).reshape(-1, 2)


def homography(px, mat):
    """화면 픽셀 → 매트 cm 호모그래피와 점마다 오차(cm). 점이 4개보다 적으면 (None, None)."""
    px = np.asarray(px, float).reshape(-1, 2)
    mat = np.asarray(mat, float).reshape(-1, 2)
    if len(px) < 4:
        return None, None
    H, _ = cv2.findHomography(px, mat, 0)
    if H is None:
        return None, None
    err = np.linalg.norm(apply_h(H, px) - mat, axis=1)
    return H, err


def solve_pose(px, mat, K):
    """기준점(높이 0)으로 카메라 자세를 구한다(26 문서 3.2절). 카메라가 매트 어느 쪽에 있든 같은 방법이다.
    평면 위 점으로 구하면 답이 둘 나올 수 있어서(거울상), IPPE, SQPnP, 반복법을 모두 써 보고
    "카메라가 매트 위(z<0), 기준점이 모두 카메라 앞, 재투영 오차가 가장 작은" 답을 고른다.
    돌려주는 것: (rvec, tvec, 점마다 재투영 오차 픽셀) 또는 None"""
    px = np.asarray(px, float).reshape(-1, 1, 2)
    obj = to3d(mat).reshape(-1, 1, 3)
    if len(px) < 4:
        return None
    cands = []
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, px, K, None, flags=flag)
            cands += list(zip(rvecs, tvecs))
        except cv2.error:
            pass
    best = None
    for rv, tv in cands:
        try:
            rv, tv = cv2.solvePnPRefineLM(obj, px, K, None, rv.copy(), tv.copy())
        except cv2.error:
            continue
        if camera_position(rv, tv)[2] >= 0:      # 매트 아래에 있는 답은 버린다
            continue
        R, _ = cv2.Rodrigues(rv)
        if ((obj.reshape(-1, 3) @ R.T + tv.reshape(3))[:, 2] <= 0).any():
            continue                             # 기준점이 카메라 뒤에 있는 답도 버린다
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - px.reshape(-1, 2), axis=1)
        if best is None or err.mean() < best[2].mean():
            best = (rv.reshape(3), tv.reshape(3), err)
    return best


def estimate_hfov(px, mat, width, height, lo=30.0, hi=90.0):
    """기준점에 가장 잘 맞는 가로 화각(도)을 찾는다. 점이 6개 이상일 때만 의미가 있다."""
    def cost(f):
        r = solve_pose(px, mat, intrinsics(width, height, f))
        return 1e9 if r is None else float(np.sqrt((r[2] ** 2).mean()))
    a, b = lo, hi
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = cost(d)
    f = (a + b) / 2
    return f, cost(f)


def camera_position(rvec, tvec):
    """매트 좌표에서 렌즈 위치 (x, y, z). 높이는 -z."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return (-R.T @ np.asarray(tvec, float).reshape(3)).reshape(3)


def camera_angles(rvec):
    """보는 방향(yaw, 매트 x축에서 잰 도), 아래로 숙인 각(pitch), 옆으로 기운 각(roll)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    right, down, fwd = R[0], R[1], R[2]          # 카메라 축을 매트 좌표로
    pitch = math.degrees(math.asin(max(-1, min(1, fwd[2]))))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    roll = math.degrees(math.asin(max(-1, min(1, right[2]))))
    return yaw, pitch, roll


def project(K, rvec, tvec, pts3):
    """3차원 매트 점 → 화면 픽셀, 그리고 카메라 앞쪽 거리(0 이하면 카메라 뒤)."""
    pts3 = np.asarray(pts3, float).reshape(-1, 3)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    cam = pts3 @ R.T + np.asarray(tvec, float).reshape(3)
    uv = cam @ np.asarray(K, float).T
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def car_box(K, rvec, tvec, x0, x1, y0, y1, h, size=None):
    """매트 위 직육면체(x0~x1, y0~y1, 높이 h)가 화면에서 차지하는 상자 (x1, y1, x2, y2).
    카메라 뒤로 넘어가면 None. size=(폭, 높이)를 주면 화면 안으로 자른다(YOLO 상자처럼)."""
    pts = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0.0, -h)]
    uv, z = project(K, rvec, tvec, pts)
    if (z <= 1e-6).any():
        return None
    b = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    if size is not None:
        w, hh = size
        b = [min(max(b[0], 0), w), min(max(b[1], 0), hh), min(max(b[2], 0), w), min(max(b[3], 0), hh)]
        if b[2] - b[0] < 1 or b[3] - b[1] < 1:
            return None
    return tuple(float(v) for v in b)


def contact_px(box):
    """바닥 접점 = 상자 아래 변 가운데 (26 문서 1.1절)."""
    return ((box[0] + box[2]) / 2.0, box[3])


def expected_contact(slot, K, rvec, tvec, H, car_w, car_l, car_h, size):
    """P2b: 보통 크기 차가 칸 가운데에 바르게 섰을 때 상자 아래 변 가운데의 매트 좌표(26 문서 5절)."""
    cx, cy = slot.center
    b = car_box(K, rvec, tvec, cx - car_w / 2, cx + car_w / 2, cy - car_l / 2, cy + car_l / 2, car_h, size)
    if b is None:
        return None
    return apply_h(H, [contact_px(b)])[0]


def lifted_offsets(K, rvec, tvec, pts, h):
    """바닥 점들을 높이 h만큼 올리면 화면에서 몇 픽셀 움직이는지 (P5, 26 문서 3.2절)."""
    a, za = project(K, rvec, tvec, to3d(pts, 0.0))
    b, zb = project(K, rvec, tvec, to3d(pts, -h))
    if (za <= 0).any() or (zb <= 0).any():
        return None
    return b - a


def screen_region(floor_px, offsets):
    """칸 바닥 사각형과 그것을 올린 모양을 합친 볼록 영역(화면 픽셀)."""
    pts = np.vstack([floor_px, floor_px + offsets]).astype(np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


def inside_image(pts, size, margin=0.0):
    w, h = size
    pts = np.asarray(pts, float).reshape(-1, 2)
    return bool(((pts[:, 0] >= margin) & (pts[:, 0] <= w - margin) &
                 (pts[:, 1] >= margin) & (pts[:, 1] <= h - margin)).all())


def front_sign(Hinv, mats):
    """매트 점 → 화면 동차 좌표의 w 부호 중 "카메라 앞"인 쪽. 기준점(화면에 보인 점)들의 부호로 정한다.
    화면에 지평선(매트 너머 벽 등)이 보이면 화면 구석 픽셀은 매트 바깥이라, 부호를 미리 정해 둘 수 없다."""
    mats = np.asarray(mats, float).reshape(-1, 2)
    if len(mats) == 0:
        return 1.0
    w = (np.c_[mats, np.ones(len(mats))] @ np.asarray(Hinv, float).T)[:, 2]
    return 1.0 if np.median(w) >= 0 else -1.0


def in_front(Hinv, mats, sign):
    """매트 점들이 모두 카메라 앞(지평선 이쪽)에 있는지."""
    q = np.c_[np.asarray(mats, float).reshape(-1, 2), np.ones(len(mats))] @ np.asarray(Hinv, float).T
    return bool((q[:, 2] * sign > 1e-12).all())


def auto_monitored(table, H, size, margin=0.0, rows=None, ref_mats=None):
    """K5: cell 네 꼭짓점이 모두 화면 안에 들어오는(카메라 앞에 있는) 칸."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return []
    sign = front_sign(Hinv, ref_mats if ref_mats is not None else [])
    out = []
    for sl in table:
        if rows and sl.row not in rows:
            continue
        if not in_front(Hinv, sl.cell, sign):    # 카메라 뒤(지평선 너머)
            continue
        if inside_image(apply_h(Hinv, sl.cell), size, margin):
            out.append(sl.no)
    return out


def edited_to_inner(table, edited, kind):
    """고친 꼭짓점을 빈 공간(inner) 네 모서리 기준으로 맞춘다.
    지금 보정 도구는 inner 모서리(기준점 IE/IB와 같은 자리)를 고친다. 2026-11-03 전 설정은 cell(칸 경계) 모서리였으므로 바꿔 준다."""
    if kind == "inner":
        return {int(k): np.asarray(v, float) for k, v in edited.items()}
    out = {}
    for k, v in edited.items():
        sl = table[int(k)]
        Hs = cv2.getPerspectiveTransform(np.asarray(v, np.float32), sl.cell.astype(np.float32))   # cell 픽셀 → 매트
        out[int(k)] = apply_h(np.linalg.inv(Hs), sl.inner)
    return out


def aspect_same(a, b, tol=0.01):
    return abs(a[0] / a[1] - b[0] / b[1]) <= tol * (a[0] / a[1])


# ---------------------------------------------------------------- 카메라 하나의 판정용 기하 (C4, C6)

class ResolutionMismatch(Exception):
    pass


class CamGeom:
    """카메라 설정(json dict) + 칸 표 + 실행 해상도로, 판정에 필요한 값을 미리 계산해 둔다.

    params(no) -> dict 로 칸별 판정 값(P8)을 받는다. 차 크기, 올리는 높이가 여기서 나온다.
    실행 해상도의 화면 비율이 보정 때와 다르면 ResolutionMismatch (C6).
    """

    def __init__(self, cam, table, run_size, params):
        self.name = cam.get("name", "CAM")
        self.table = table
        cal = tuple(cam["calib_size"])
        run = tuple(run_size)
        if not aspect_same(cal, run):
            raise ResolutionMismatch(
                f"{self.name}: 보정 해상도 {cal[0]}x{cal[1]}와 실행 해상도 {run[0]}x{run[1]}의 화면 비율이 다릅니다. "
                f"이 해상도로 다시 보정하세요(calibrate.py). 이 카메라는 멈춥니다.")
        self.size = run
        s = run[0] / cal[0]
        self.scale = s
        S = np.diag([s, s, 1.0])
        self.H = np.asarray(cam["homography"], float) @ np.linalg.inv(S)   # 실행 픽셀 → 매트
        self.Hinv = np.linalg.inv(self.H)
        pose = cam.get("pose") or {}
        self.rvec = np.asarray(pose["rvec"], float) if pose.get("rvec") is not None else None
        self.tvec = np.asarray(pose["tvec"], float) if pose.get("tvec") is not None else None
        intr = cam.get("intrinsics", {})
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr.get("fy", intr["fx"]), intr["cy"]], [0, 0, 1.0]]) \
            if "fx" in intr else intrinsics(cal[0], cal[1], intr.get("hfov_deg", 50.0))
        self.K = S @ K
        self.has_pose = self.rvec is not None
        self.cam_ground = camera_position(self.rvec, self.tvec)[:2] if self.has_pose else None
        self.monitored = set(int(n) for n in cam.get("monitored", []))
        # 카메라 앞 부호: 기준점이 있는 쪽 (지평선이 화면에 보여도 맞게)
        self.front = front_sign(self.Hinv, [r["mat"] for r in cam.get("ref_points", [])])
        # 고친 꼭짓점: 빈 공간(inner) 네 모서리 [왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래] (예전 설정은 cell)
        edited = edited_to_inner(table, {k: np.asarray(v, float) * s for k, v in (cam.get("edited_corners") or {}).items()},
                                 cam.get("edited_corners_kind", "cell"))
        stored = {int(k): v for k, v in (cam.get("expected_contact") or {}).items()}
        stored_car = cam.get("expected_car")

        self.slot_H = {}       # 칸별 픽셀 → 매트 (고친 칸은 칸 꼭짓점 ↔ cell 변환, 22 문서 3.3절)
        self.floor_px = {}     # cell 네 꼭짓점 (화면)
        self.inner_px = {}
        self.region = {}       # P5 칸 화면 영역
        self.region_area = {}
        self.expected = {}     # P2b 예상 접점 (매트), 세로로 선 차
        self.expected_horiz = {}   # 가로로 선 차의 예상 접점 (매트, docs/40). 카메라 자세가 있을 때만
        self.exp_box = {}      # 칸 -> {"vert": (접점 x, 접점 y, 화면 폭, 높이), "horiz": (...)} 예상 상자 (docs/40)
        self.edited = set(edited)
        for sl in table:
            p = params(sl.no)
            if sl.no in edited:
                Hs = cv2.getPerspectiveTransform(edited[sl.no].astype(np.float32), sl.inner.astype(np.float32))
                floor = apply_h(np.linalg.inv(Hs), sl.cell)
            else:
                Hs = self.H
                if not in_front(self.Hinv, sl.cell, self.front):
                    continue                  # 지평선 너머 칸은 이 카메라와 상관없다
                floor = apply_h(self.Hinv, sl.cell)
            self.slot_H[sl.no] = Hs
            self.floor_px[sl.no] = floor
            self.inner_px[sl.no] = apply_h(np.linalg.inv(Hs), sl.inner)
            if self.has_pose:
                off = lifted_offsets(self.K, self.rvec, self.tvec, sl.cell, p["occlusion_height_cm"])
                if off is not None:
                    reg = screen_region(floor, off)
                    self.region[sl.no] = reg
                    self.region_area[sl.no] = float(cv2.contourArea(reg.astype(np.float32)))
                car = [p["car_width_cm"], p["car_length_cm"], p["car_height_cm"]]
                if sl.no in stored and stored_car and np.allclose(stored_car, car):
                    e = stored[sl.no]
                    self.expected[sl.no] = None if e is None else np.asarray(e, float)
                else:
                    self.expected[sl.no] = expected_contact(sl, self.K, self.rvec, self.tvec, self.H,
                                                            car[0], car[1], car[2], self.size)
                # 방향 추정 (docs/40): 칸 가운데에 세로·가로로 선 보통 크기 차의 예상 상자 (접점, 화면 폭·높이).
                # 여기서(보정 파일을 읽을 때) 한 번만 계산하고, 프레임마다는 이 표와 비교만 한다
                cx, cy = sl.center
                sz = {}
                for o, (hx, hy) in (("vert", (car[0] / 2, car[1] / 2)), ("horiz", (car[1] / 2, car[0] / 2))):
                    b = car_box(self.K, self.rvec, self.tvec, cx - hx, cx + hx, cy - hy, cy + hy, car[2], self.size)
                    if b is not None and b[2] > b[0] and b[3] > b[1]:
                        c = apply_h(self.H, [contact_px(b)])[0]
                        sz[o] = (float(c[0]), float(c[1]), b[2] - b[0], b[3] - b[1])
                        if o == "horiz":
                            self.expected_horiz[sl.no] = c          # P3에서 차 가운데를 되돌릴 때도 쓴다
                if "vert" in sz:
                    self.exp_box[sl.no] = sz
            else:
                # 자세가 없으면 칸 가운데 입구를 예상 접점으로 쓴다(정면 카메라 근사)
                self.expected[sl.no] = np.array([sl.center[0], sl.entrance_y])

    def to_mat(self, px, slot=None):
        H = self.H if slot is None else self.slot_H.get(slot, self.H)
        return apply_h(H, px)

    def to_px(self, mat):
        return apply_h(self.Hinv, mat)

    def on_floor(self, px):
        """화면 점이 매트 바닥 쪽(지평선 아래)인지. 지평선 위(벽, 커튼)의 점은 매트 좌표가 뒤집혀 나온다."""
        w = float(np.asarray(self.H, float)[2] @ [float(px[0]), float(px[1]), 1.0])
        # 바닥 점은 H로 바꾼 w와 Hinv로 되돌린 w의 부호가 맞물린다: w(바닥) 부호 = front 부호
        return w * self.front > 0

    def far_dir(self, p_mat):
        """매트 점에서 카메라 반대쪽(먼 쪽) 단위 방향."""
        if self.cam_ground is None:
            return np.array([0.0, 1.0])
        d = np.asarray(p_mat, float) - self.cam_ground
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([0.0, 1.0])


def build_camera_config(name, source, size, ref, table, params, hfov_deg, monitored=None, edited=None,
                        intr_fx=None):
    """기준점에서 카메라 설정 dict를 만든다(K3, P2b, P5). 보정 도구와 합성 시험이 같이 쓴다.
    ref: [{"id", "mat", "px"}] (보정 해상도 픽셀)."""
    px = [r["px"] for r in ref]
    mat = [r["mat"] for r in ref]
    H, err = homography(px, mat)
    if H is None:
        raise ValueError("기준점이 4개 이상 있어야 합니다.")
    K = intrinsics(size[0], size[1], hfov_deg)
    if intr_fx:
        K[0, 0] = K[1, 1] = intr_fx
    pose = solve_pose(px, mat, K)
    ref_out = [dict(r, err_cm=round(float(e), 3)) for r, e in zip(ref, err)]
    cam = {
        "name": name, "source": source, "calib_size": [int(size[0]), int(size[1])],
        "intrinsics": {"hfov_deg": round(float(hfov_deg), 3), "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                       "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "ref_points": ref_out,
        "homography": np.asarray(H).tolist(),
        "homography_err_cm": {"mean": round(float(err.mean()), 3), "max": round(float(err.max()), 3)},
        "pose": None,
        "monitored": sorted(int(n) for n in (monitored if monitored is not None
                                             else auto_monitored(table, H, size, ref_mats=mat))),
        "edited_corners": {str(k): np.asarray(v, float).tolist() for k, v in (edited or {}).items()},
        "edited_corners_kind": "inner",          # 고친 꼭짓점 = 빈 공간(inner) 네 모서리 (화면 픽셀)
    }
    if pose is not None:
        rv, tv, perr = pose
        yaw, pitch, roll = camera_angles(rv)
        pos = camera_position(rv, tv)
        cam["pose"] = {"rvec": rv.tolist(), "tvec": tv.tolist(),
                       "cam_pos_cm": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(-pos[2]), 2)],
                       "yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2),
                       "reproj_px": {"mean": round(float(perr.mean()), 3), "max": round(float(perr.max()), 3)}}
    # 칸별 예상 접점과 화면 영역을 CamGeom으로 계산해 같이 저장한다
    g = CamGeom(cam, table, size, params)
    p0 = params(None)
    cam["expected_car"] = [p0["car_width_cm"], p0["car_length_cm"], p0["car_height_cm"]]
    cam["expected_contact"] = {str(k): (None if v is None else [round(float(v[0]), 3), round(float(v[1]), 3)])
                               for k, v in g.expected.items()}
    cam["screen_region"] = {str(k): np.round(v, 1).tolist() for k, v in g.region.items()}
    return cam
