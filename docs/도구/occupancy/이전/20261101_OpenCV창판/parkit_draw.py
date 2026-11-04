"""PARKIT 점유 판정 — 그리기 (25 문서 C5, D1, D2)

보정 도구와 판정 프로그램이 같이 쓴다. OpenCV 글자는 한글이 안 나오므로 영어 약어로 쓴다(뜻은 28 문서).
이미지 저장/읽기는 한글 경로에서도 되도록 imencode/imdecode를 쓴다(13 문서 3절).
"""

import os

import cv2
import numpy as np

# 상태 색 (BGR) — EMPTY 초록, OCCUPIED 빨강, DYNAMIC_BLIND 노랑, STATIC_BLIND 회색 (25 문서 D1)
COLORS = {
    "EMPTY": (60, 200, 60),
    "OCCUPIED": (40, 40, 230),
    "DYNAMIC_BLIND": (0, 215, 255),
    "STATIC_BLIND": (150, 150, 150),
    None: (230, 230, 230),
}
SHORT = {"EMPTY": "EMP", "OCCUPIED": "OCC", "DYNAMIC_BLIND": "DYN", "STATIC_BLIND": "STA"}
BOX_COLORS = {"slot": (40, 40, 230), "slot_cut": (200, 60, 230), "aisle": (255, 160, 0),
              "uncertain": (0, 140, 255), "unmonitored": (160, 160, 160)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def save_image(path, im, quality=95):
    """한글 경로에서도 저장된다(Windows의 cv2.imwrite는 한글 경로에서 조용히 실패)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, im, params)
    if ok:
        buf.tofile(path)
    return ok


def read_image(path):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def text(im, s, org, scale=0.5, color=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    """배경 상자가 있는 글자."""
    (w, h), b = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(im, (x - 2, y - h - 3), (x + w + 2, y + b + 1), bg, -1)
    cv2.putText(im, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w


def poly(im, pts, color, thick=1, scale=1.0, closed=True):
    p = np.round(np.asarray(pts, float) * scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(im, [p], closed, color, thick, cv2.LINE_AA)


def draw_slots(im, geom, status=None, scale=1.0, thick=2, slots=None, show_zone=False, show_region=False,
               label=True, params=None):
    """카메라 화면 위에 칸 선(inner)을 상태 색으로 그린다(C5, D1). scale은 화면에 줄여 그릴 때의 비율."""
    nos = slots if slots is not None else sorted(geom.monitored)
    for no in nos:
        if no not in geom.inner_px:
            continue
        st = status.get(no) if status else None
        c = COLORS.get(st, COLORS[None])
        poly(im, geom.inner_px[no], c, thick, scale)
        poly(im, geom.floor_px[no], tuple(int(v * 0.6) for v in c), 1, scale)
        if show_zone and params is not None:
            sl = geom.table[no]
            p = params(no)
            y0, y1 = sl.zone_y(p["zone_margin_entrance_cm"], p["zone_margin_back_cm"])
            z = np.array([[sl.x0, y0], [sl.x1, y0], [sl.x1, y1], [sl.x0, y1]])
            from parkit_geom import apply_h
            poly(im, apply_h(np.linalg.inv(geom.slot_H[no]), z), (200, 200, 0), 1, scale)
        if show_region and no in geom.region:
            poly(im, geom.region[no], (255, 120, 0), 1, scale)
        if label:
            c0 = geom.inner_px[no].mean(axis=0) * scale
            text(im, str(no), (c0[0] - 8, c0[1] + 5), 0.45, c, 1)


def draw_boxes(im, res, scale=1.0):
    """차 상자, 바닥 접점, 고른 칸 (D1)."""
    for b in res.boxes:
        c = BOX_COLORS.get(b.kind, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b.box]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cx, cy = int(round(b.contact_px[0] * scale)), int(round(b.contact_px[1] * scale))
        cv2.circle(im, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(im, (cx, cy), 5, c, 2)
        lab = f"{b.conf:.2f} "
        if b.slot is not None:
            lab += f"#{b.slot}"
            if b.straddle:
                lab += "+" + "+".join(str(n) for n in b.straddle)
        else:
            lab += {"aisle": "aisle", "uncertain": "?" + ",".join(map(str, b.occluded_slots))}.get(b.kind, b.kind)
        if b.cut_by is not None:
            lab += " cut"
        text(im, lab, (x1, max(12, y1 - 4)), 0.45, (255, 255, 255), 1, c)


class PlanView:
    """위에서 본 매트 평면도 (D2). 칸 표만으로 그린다."""

    def __init__(self, table, px_per_cm=10, margin=12):
        self.table = table
        self.k = float(px_per_cm)
        self.m = margin
        w, h = table.mat_size
        self.size = (int(w * self.k + 2 * margin), int(h * self.k + 2 * margin))

    def to_px(self, p):
        p = np.asarray(p, float)
        return p * self.k + self.m

    def to_mat(self, px):
        return (np.asarray(px, float) - self.m) / self.k

    def render(self, status, points=(), selected=None, cams=None, title=None):
        """status: {칸: 상태}. points: [(매트 좌표, 색)]. cams: [(이름, 매트 x, y)]."""
        W, H = self.size
        im = np.full((H, W, 3), 45, np.uint8)
        w, h = self.table.mat_size
        cv2.rectangle(im, tuple(self.to_px((0, 0)).astype(int)), tuple(self.to_px((w, h)).astype(int)), (85, 85, 85), -1)
        for sl in self.table:
            st = status.get(sl.no)
            c = COLORS.get(st, COLORS[None])
            cell = np.round(self.to_px(sl.cell)).astype(np.int32)
            inner = np.round(self.to_px(sl.inner)).astype(np.int32)
            cv2.fillConvexPoly(im, cell, (200, 200, 200))
            cv2.fillConvexPoly(im, inner, c)
            if st == "STATIC_BLIND":
                x0, y0 = inner.min(axis=0)
                x1, y1 = inner.max(axis=0)
                for d in range(0, int(x1 - x0 + y1 - y0), 6):
                    cv2.line(im, (int(x0 + d), int(y0)), (int(x0), int(y0 + d)), (120, 120, 120), 1)
            if sl.no == selected:
                cv2.polylines(im, [cell], True, (255, 255, 255), 3)
            cc = self.to_px(sl.center).astype(int)
            s = str(sl.no)
            (tw, th), _ = cv2.getTextSize(s, FONT, 0.38, 1)
            cv2.putText(im, s, (cc[0] - tw // 2, cc[1] + th // 2), FONT, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        for p, c in points:
            q = self.to_px(p).astype(int)
            cv2.circle(im, tuple(q), 4, (255, 255, 255), -1)
            cv2.circle(im, tuple(q), 4, c, 1)
        for name, x, y in cams or []:
            q = self.to_px((x, max(y, -self.m / self.k + 0.5))).astype(int)
            cv2.drawMarker(im, tuple(q), (255, 255, 0), cv2.MARKER_TRIANGLE_DOWN, 14, 2)
            text(im, name, (q[0] + 8, q[1] + 10), 0.4, (255, 255, 0), 1, None)
        if title:
            text(im, title, (6, H - 6), 0.45, (255, 255, 255), 1)
        return im
