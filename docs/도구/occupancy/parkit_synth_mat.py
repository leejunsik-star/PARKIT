"""PARKIT 보정 ver2 — 시험용 합성 화면 (test_autoref.py가 쓴다. 판정·보정 코드는 이 파일을 쓰지 않는다)

draw_mat_real : 칸 표로 "실제 매트처럼" 위에서 본 그림을 그린다.
                회색 바닥(질감), 칸 사이 U자 선(가운데가 빈 머리핀, 둥근 끝은 입구 쪽), 칸마다 떨어진 카스토퍼,
                기둥 X 칸, 줄 끝 반쪽 U자, 칸 번호 글자, 통로 화살표, 횡단보도, 보도 선
draw_lot      : 일반 주차장 그림 (칠한 직선 목록)
camera_view   : 위 그림을 가상 카메라 화면으로 바꾸고 차(직육면체), 반사광, 잡음, 흐림을 더한다
"""

import math

import cv2
import numpy as np

from parkit_synth import VirtualCamera  # noqa: F401  (시험에서 같이 쓴다)

FLOOR, PAINT = 105, 232


def _px(p, k):
    return (int(round(p[0] * k)), int(round(p[1] * k)))


def _rect(im, x0, y0, x1, y1, k, val):
    cv2.rectangle(im, _px((min(x0, x1), min(y0, y1)), k), _px((max(x0, x1), max(y0, y1)), k), val, -1, cv2.LINE_AA)


def _floor(h, w, seed, base=FLOOR):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 9, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    fine = rng.normal(0, 4, (h, w)).astype(np.float32)
    return np.clip(base + noise + fine, 0, 255)


def draw_mat_real(table, k=30, shape=None, seed=0, numbers=True, extras=True):
    """칸 표 → 위에서 본 매트 그림 (BGR, 1cm = k픽셀)."""
    shape = shape or {"leg_width": 0.25, "castor_margin": 0.4, "castor_width": 0.3, "u_width": 0.9}
    leg, uw = shape["leg_width"], shape.get("u_width", 0.9)
    r_out, r_in = uw / 2, uw / 2 - leg
    W, H = table.mat_size
    im = _floor(int(H * k), int(W * k), seed)
    rows = {}
    for sl in table:
        rows.setdefault(sl.row, []).append(sl)
    for row, sls in rows.items():
        sls.sort(key=lambda s: s.x0)
        ent = sls[0].entrance_y
        s_in = 1.0 if sls[0].entrance_low else -1.0
        back = sls[0].y1 if sls[0].entrance_low else sls[0].y0
        # 칸 사이 U자 (머리핀): 맞닿은 이웃 사이마다
        for a, b in zip(sls, sls[1:]):
            if abs(a.x1 - b.x0) > 0.3:
                # 기둥: 두 묶음 사이 빈 곳에 X 칸
                gx0, gx1 = a.ix1 + 0.6, b.ix0 - 0.6
                if gx1 - gx0 > 1.0:
                    y0, y1 = sorted((ent + s_in * 1.0, back - s_in * 1.0))
                    cv2.rectangle(im, _px((gx0, y0), k), _px((gx1, y1), k), PAINT, max(2, int(0.15 * k)), cv2.LINE_AA)
                    cv2.line(im, _px((gx0, y0), k), _px((gx1, y1), k), PAINT, max(2, int(0.15 * k)), cv2.LINE_AA)
                    cv2.line(im, _px((gx0, y1), k), _px((gx1, y0), k), PAINT, max(2, int(0.15 * k)), cv2.LINE_AA)
                continue
            xb = a.x1
            cy = ent + s_in * r_out
            _rect(im, xb - r_out, cy, xb + r_out, back, k, PAINT)
            cv2.circle(im, _px((xb, cy), k), int(round(r_out * k)), PAINT, -1, cv2.LINE_AA)
            _rect(im, xb - r_in, cy, xb + r_in, back + s_in * 0.05, k, FLOOR)
            cv2.circle(im, _px((xb, cy), k), int(round(r_in * k)), FLOOR, -1, cv2.LINE_AA)
        # 줄 끝·기둥 옆: 반쪽 U자 (다리 한 줄)
        for i, sl in enumerate(sls):
            if i == 0 or abs(sls[i - 1].x1 - sl.x0) > 0.3:
                _rect(im, sl.ix0 - leg, ent + s_in * 0.2, sl.ix0, back, k, PAINT)
            if i == len(sls) - 1 or abs(sl.x1 - sls[i + 1].x0) > 0.3:
                _rect(im, sl.ix1, ent + s_in * 0.2, sl.ix1 + leg, back, k, PAINT)
        for sl in sls:
            # 카스토퍼: 칸 쪽 면이 칸 표의 안쪽 끝
            _rect(im, sl.ix0 + shape["castor_margin"], back, sl.ix1 - shape["castor_margin"],
                  back + s_in * shape["castor_width"], k, PAINT)
            if numbers:
                txt = str(sl.no)
                fs = 0.6 * k / 22.0
                th = max(1, int(round(0.1 * k)))
                (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                c = sl.center
                yy = ent + s_in * 1.3
                org = (int(c[0] * k - tw / 2), int(yy * k + (tht / 2 if s_in > 0 else tht / 2)))
                cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, PAINT, th, cv2.LINE_AA)
    if extras:
        # 통로 화살표, 횡단보도, 보도 선, 중앙 분리대 (판정과 상관없는 흰 그림)
        for y, d in ((16.0, -1), (40.2, 1)):
            for x in (25.0, 55.0):
                cv2.arrowedLine(im, _px((x - 4 * d, y), k), _px((x + 4 * d, y), k), PAINT, max(2, int(0.35 * k)),
                                cv2.LINE_AA, tipLength=0.25)
        for i in range(6):
            _rect(im, 70.0 + i * 0.9, 13.0, 70.5 + i * 0.9, 19.0, k, PAINT)
        for y in (12.5, 42.9):
            _rect(im, 0, y - 0.1, W, y + 0.1, k, PAINT)
        _rect(im, 0, 27.9, W, 28.1, k, 200)
        _rect(im, 0, 29.3, W, 29.5, k, 200)
    im = cv2.GaussianBlur(im, (0, 0), 0.5)
    im = np.clip(im, 0, 255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def draw_lot(size, segments, k, seed=0, base=95, paint=225):
    """일반 주차장: size=(폭, 높이) 매트 단위, segments=[(p1, p2, 폭)] 칠한 직선(끝은 각짐)."""
    W, H = size
    im = _floor(int(H * k), int(W * k), seed, base)
    for p1, p2, w in segments:
        p1, p2 = np.array(p1, float), np.array(p2, float)
        d = (p2 - p1) / np.linalg.norm(p2 - p1)
        n = np.array([-d[1], d[0]]) * w / 2
        quad = np.array([p1 + n, p2 + n, p2 - n, p1 - n]) * k
        cv2.fillConvexPoly(im, np.round(quad).astype(np.int32), paint, cv2.LINE_AA)
    im = np.clip(cv2.GaussianBlur(im, (0, 0), 0.5), 0, 255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def camera_view(vcam, top, k, cars=(), glare=(), noise=3.0, blur=0.7, seed=1):
    """위에서 본 그림 → 가상 카메라 화면. cars=[(x0, x1, y0, y1, 높이, (b,g,r))], glare=[(x, y, 반지름)] (매트 단위)."""
    W, H = vcam.size
    mh, mw = top.shape[:2]
    corners = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], float) / k
    uv, _ = vcam.project(np.c_[corners, np.zeros(4)])
    M = cv2.getPerspectiveTransform((corners * k).astype(np.float32), uv.astype(np.float32))
    frame = cv2.warpPerspective(top, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(60, 70, 60)).astype(np.float32)
    cam_xy = vcam.C[:2]
    for x0, x1, y0, y1, h, col in sorted(cars, key=lambda c: -np.hypot((c[0] + c[1]) / 2 - cam_xy[0], (c[2] + c[3]) / 2 - cam_xy[1])):
        pts = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0.0, -h)])
        p, z = vcam.project(pts)
        if (z <= 0).any():
            continue
        cv2.fillConvexPoly(frame, cv2.convexHull(p.astype(np.float32)).astype(np.int32), col, cv2.LINE_AA)
        top_q = vcam.project([(x, y, -h) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])[0]
        cv2.fillConvexPoly(frame, top_q.astype(np.int32), tuple(min(255, c + 50) for c in col), cv2.LINE_AA)
    for x, y, r in glare:
        c, _ = vcam.project([[x, y, 0.0]])
        e, _ = vcam.project([[x + r, y, 0.0]])
        rad = max(3, int(np.linalg.norm(e[0] - c[0])))
        blob = np.zeros((H, W), np.float32)
        cv2.circle(blob, tuple(int(v) for v in c[0]), rad, 1.0, -1, cv2.LINE_AA)
        blob = cv2.GaussianBlur(blob, (0, 0), rad * 0.5)
        frame += blob[:, :, None] * 140
    rng = np.random.default_rng(seed)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)
