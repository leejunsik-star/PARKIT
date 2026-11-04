"""PARKIT 점유 판정 — 합성 시험용 가상 카메라와 합성 영상 (test_geometry.py, test_pipeline.py가 쓴다)

11 문서 배치(렌즈 높이 15cm, 매트 위쪽 끝에서 8cm 밖, 아래로 30°, 가로 화각 50°)로 가상 카메라를 만들고,
매트 위 직육면체 차를 화면 상자로 바꾼다. 실제 판정 코드는 이 파일을 쓰지 않는다.
"""

import math

import cv2
import numpy as np

from parkit_geom import car_box, intrinsics, ref_points


class VirtualCamera:
    """pos=(x, y, 높이) cm. look=(x, y)를 주면 매트의 그 점을 보고, 없으면 yaw_dir 방향으로 pitch만큼 숙인다."""

    def __init__(self, pos, size=(1920, 1080), hfov=50.0, look=None, pitch=30.0, yaw_dir=(0, 1), roll=0.0):
        self.size = size
        self.K = intrinsics(size[0], size[1], hfov)
        C = np.array([pos[0], pos[1], -pos[2]], float)
        if look is not None:
            f = np.array([look[0], look[1], 0.0]) - C
        else:
            d = np.array([yaw_dir[0], yaw_dir[1], 0.0], float)
            d /= np.linalg.norm(d)
            f = d * math.cos(math.radians(pitch)) + np.array([0, 0, 1.0]) * math.sin(math.radians(pitch))
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, -1.0])
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        if roll:
            c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
            r, u = c * r + s * u, -s * r + c * u
        self.R = np.stack([r, u, f])
        self.t = -self.R @ C
        self.rvec = cv2.Rodrigues(self.R)[0].reshape(3)
        self.C = C

    def scaled(self, size):
        """같은 자세, 다른 해상도(화면 비율은 같다고 본다)."""
        v = VirtualCamera.__new__(VirtualCamera)
        v.__dict__.update(self.__dict__)
        s = size[0] / self.size[0]
        v.size = size
        v.K = np.diag([s, s, 1.0]) @ self.K
        return v

    def project(self, pts3):
        pts3 = np.asarray(pts3, float).reshape(-1, 3)
        cam = pts3 @ self.R.T + self.t
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3], cam[:, 2]

    def box(self, x0, x1, y0, y1, h):
        return car_box(self.K, self.rvec, self.t, x0, x1, y0, y1, h, self.size)

    def car_in_slot(self, slot, w=3.2, l=6.5, h=2.0, dx=0.0, dy=0.0):
        """칸 가운데에 바르게 선 차. dx는 줄 방향, dy는 깊이 방향(+는 칸 안쪽)으로 옮긴 거리."""
        cx, cy = slot.center
        cy += dy if slot.entrance_low else -dy
        cx += dx
        return self.box(cx - w / 2, cx + w / 2, cy - l / 2, cy + l / 2, h)

    def ref_clicks(self, table, ids, noise_px=0.0, seed=0):
        """기준점 id들을 이 카메라로 찍은 것처럼 만든다(보정 도구 대신)."""
        rng = np.random.default_rng(seed)
        allp = {r["id"]: r for r in ref_points(table)}
        out = []
        for i in ids:
            m = allp[i]["mat"]
            uv, z = self.project([[m[0], m[1], 0.0]])
            p = uv[0] + rng.normal(0, noise_px, 2) if noise_px else uv[0]
            out.append({"id": i, "mat": list(m), "px": [float(p[0]), float(p[1])]})
        return out


def cam1():
    """11 문서 CAM-1: x 51, 매트 위쪽 끝에서 8cm 밖, 높이 15, 아래로 30°, 정면."""
    return VirtualCamera((51, -8, 15))


def cam2():
    """11 문서 CAM-2: x 23."""
    return VirtualCamera((23, -8, 15))


def cam_diag():
    """26 문서 5절 대각선 카메라: x 86, y -8, 높이 15, 매트 (50, 26)을 본다."""
    return VirtualCamera((86, -8, 15), look=(50, 26))


def pick_refs(table, vcam, want=8, rows=("1-14", "28-41"), margin=40):
    """화면 안에 보이는 실측 기준점 중에서 넓게 퍼지고 깊이가 섞이도록 고른다."""
    W, H = vcam.size
    cands = []
    for r in ref_points(table):
        if r["slot"] is None or table[r["slot"]].row not in rows or not r["kind"].startswith("I"):
            continue
        uv, z = vcam.project([[r["mat"][0], r["mat"][1], 0.0]])
        if z[0] > 0 and margin <= uv[0][0] <= W - margin and margin <= uv[0][1] <= H - margin:
            cands.append((r, uv[0]))
    chosen = []
    # 줄과 끝(입구/안쪽)마다 화면 왼쪽 끝, 오른쪽 끝 점을 고른다
    groups = {}
    for r, uv in cands:
        key = (table[r["slot"]].row, r["kind"][1])
        groups.setdefault(key, []).append((r, uv))
    for key, g in sorted(groups.items()):
        g.sort(key=lambda t: t[1][0])
        for r, uv in (g[0], g[-1]):
            if r["id"] not in chosen:
                chosen.append(r["id"])
    return chosen[:max(want, 4)]


# ---------------------------------------------------------------- 합성 영상 (test_pipeline.py)

def draw_mat_image(table, px_per_cm=12):
    """칸 표로 위에서 본 매트 그림(합성 영상 바탕)."""
    w, h = table.mat_size
    im = np.full((int(h * px_per_cm), int(w * px_per_cm), 3), (70, 70, 70), np.uint8)
    for sl in table:
        cell = (sl.cell * px_per_cm).astype(np.int32)
        inner = (sl.inner * px_per_cm).astype(np.int32)
        cv2.fillConvexPoly(im, cell, (235, 235, 235))
        cv2.fillConvexPoly(im, inner, (80, 80, 80))
    return im


def render_frame(vcam, mat_im, px_per_cm, cars):
    """가상 카메라 화면 한 장. cars: [(x0, x1, y0, y1, h, (b, g, r))]. 먼 차부터 그린다."""
    W, H = vcam.size
    S = np.diag([1.0 / px_per_cm, 1.0 / px_per_cm, 1.0])
    # 매트 그림 픽셀 → 매트 cm → 화면
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) * [mat_im.shape[1] / px_per_cm, mat_im.shape[0] / px_per_cm]
    uv, _ = vcam.project(np.c_[corners, np.zeros(4)])
    src = (corners * px_per_cm).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    frame = cv2.warpPerspective(mat_im, M, (W, H), borderValue=(40, 50, 40))
    cam_xy = vcam.C[:2]
    def dist(c):
        return -np.hypot((c[0] + c[1]) / 2 - cam_xy[0], (c[2] + c[3]) / 2 - cam_xy[1])
    for x0, x1, y0, y1, h, color in sorted(cars, key=dist):
        pts = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0.0, -h)])
        p, z = vcam.project(pts)
        if (z <= 0).any():
            continue
        hull = cv2.convexHull(p.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(frame, hull, color)
        top = vcam.project([(x, y, -h) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])[0]
        cv2.fillConvexPoly(frame, top.astype(np.int32), tuple(min(255, c + 60) for c in color))
    return frame


def visible_boxes(vcam, cars):
    """cars의 화면 상자. 더 가까운 차가 아래를 가리면 상자 아래를 가리는 차 위 끝에서 자른다(YOLO가 보이는 부분만 잡는다고 본다)."""
    raw = []
    for c in cars:
        b = vcam.box(*c[:5])
        if b is not None:
            raw.append(b)
    out = []
    for a in raw:
        x1, y1, x2, y2 = a
        for b in raw:
            if b is a or b[3] <= a[3]:
                continue
            ov = min(x2, b[2]) - max(x1, b[0])
            if ov >= 0.5 * (x2 - x1) and b[1] < y2:
                y2 = max(y1 + 1, b[1])
        if y2 - y1 > 3:
            out.append((x1, y1, x2, y2, 0.9))
    return out
