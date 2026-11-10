"""카메라별 위치 추정. 기존 사선 판정은 그대로, 상부 촬영만 독립 처리."""
import cv2
import numpy as np

MODE_LABELS = {"ground": "사선 촬영 · 기존 바닥 접점", "center": "상부 촬영 · 박스 정중앙"}


def contact_mode(config):
    mode = config.get("contact_mode", "ground")
    if mode not in MODE_LABELS:
        raise ValueError(f"contact_mode는 ground 또는 center여야 합니다: {mode}")
    return mode


def judge_center(geom, params, dets):
    """상부 박스 중심을 바닥의 근사 위치로 사용한다.

    사선용 예상 하단 접점, 앞/뒤 가림, 폭 추정식을 재사용하지 않는다.
    온전한 박스의 중심이 들어간 칸만 점유로 보고, 박스가 걸친 다른 칸은
    차체 방향·실제 발자국을 모르므로 모름으로 남긴다(두 칸 점유 확정 아님).
    """
    from parkit_geom import apply_h
    from parkit_judge import FrameResult, BoxResult, PRESENT, ABSENT
    result = FrameResult()
    result.obs = {n: ABSENT for n in geom.monitored}
    result.why = {n: "상부: 중심 관측 없음" for n in geom.monitored}
    w, h = geom.size
    boxes = [BoxResult(i, d[:4], d[4] if len(d) > 4 else 1.) for i, d in enumerate(dets)]
    result.boxes = boxes
    for b in boxes:
        x1, y1, x2, y2 = b.box
        b.contact_px = ((x1+x2)/2, (y1+y2)/2)
        b.edge = x1 <= 1 or y1 <= 1 or x2 >= w-1 or y2 >= h-1
        b.edge_bottom = y2 >= h-1
        if not np.isfinite(b.box).all() or x2 <= x1 or y2 <= y1 or not geom.on_floor(b.contact_px):
            b.kind = "uncertain"
            continue
        b.contact_mat = apply_h(geom.H, [b.contact_px])[0]
        if not np.isfinite(b.contact_mat).all():
            b.contact_mat = None
            continue
        # 겹친 상자는 차량 가림과 중복 탐지를 구분할 수 없으므로 확정을 보류한다.
        overlap = any(a is not b and min(x2,a.box[2]) > max(x1,a.box[0])
                      and min(y2,a.box[3]) > max(y1,a.box[1]) for a in boxes)
        uncertain = b.edge or overlap
        rect = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        touched, centers = [], []
        for sl in geom.table:
            poly = np.float32(geom.inner_px[sl.no])
            area, _ = cv2.intersectConvexConvex(rect, cv2.convexHull(poly))
            if area > 1e-3:
                touched.append(sl.no)
            if cv2.pointPolygonTest(poly, b.contact_px, False) >= 0:
                centers.append(sl.no)
        for no in touched:
            if no in result.obs and result.obs[no] != PRESENT:
                result.obs[no] = None
                result.why[no] = "상부: 경계/가림으로 점유 불확실"
        if uncertain or len(centers) > 1:
            b.kind = "uncertain"
            continue
        if centers:
            no = centers[0]
            b.slot = no
            b.slot_mat = apply_h(geom.slot_H[no], [b.contact_px])[0]
            b.kind = "slot" if no in geom.monitored else "unmonitored"
            if no in result.obs:
                result.obs[no] = PRESENT
                result.why[no] = "상부: 박스 중심이 주차면 안"
    return result
