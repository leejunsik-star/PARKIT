"""PARKIT 점유 판정 — 한 프레임 판정 (25 문서 P2, P2a, P2b, P3, P4, P5 / 26 문서)

카메라 하나의 한 프레임에서 찾은 상자들 → 감시 칸마다 "차 있음 / 없음 / 가려짐".
시간 거르기와 카메라 합치기는 parkit_state.py가 한다. 화면 코드는 없다.

판정 순서
    1. 상자마다 바닥 접점(아래 변 가운데)을 구한다.
    2. P2a: 아래 변이 화면에서 더 아래(카메라에 가까운) 다른 상자 안에 있으면
       가로 겹침 ≥ 폭 × cut_overlap_ratio 이면 "아래가 잘린 상자", 그보다 적으면 "옆이 가려진 상자".
    3. 카메라에 가까운 상자부터 칸을 정한다.
       - P2/P2b: 칸마다 접점을 그 칸 좌표로 바꾸고, 깊이가 판정 구역 안이면서
         예상 접점과의 줄 방향 거리가 expect_max_dist_cm 안인 칸 중 가장 가까운 칸.
       - P2a: 잘린 상자는 "가리는 차의 뒤 끝 ~ 바꾼 점" 구간이 지나는 칸을 본다(가리는 차 자신의 칸은 뺀다).
         한 칸이면 그 칸, 여러 칸이면 그 칸들을 가려짐, 없으면 통로 차.
       - P4: 어느 칸도 아니면 통로 차.
    4. P3: 가려지지 않은 상자만, 차 가운데 ± 차 폭/2가 옆 칸 inner에 straddle_min_cm 이상 들어가면 옆 칸도 차 있음.
    5. P5: 통로 차 상자들(합집합)이 칸 화면 영역을 occlusion_ratio 이상 덮고, 그 칸에 차가 직접 잡히지 않았으면 가려짐.
    6. 칸마다 우선순위: 차 있음 > 가려짐(P2a) > 가려짐(P5) > 모름(None) > 차 없음.
       "모름"은 옆이 가려진 상자라 걸침(P3)을 판정할 수 없는 옆 칸이다. 시간 거르기를 멈추지도 다시 세지도 않는다.
"""

import cv2
import numpy as np

from parkit_geom import apply_h, contact_px

PRESENT, ABSENT, OCCLUDED, OFFLINE = "present", "absent", "occluded", "offline"


class BoxResult:
    def __init__(self, i, box, conf):
        self.i = i
        self.box = tuple(float(v) for v in box[:4])
        self.conf = float(conf)
        self.contact_px = contact_px(self.box)
        self.contact_mat = None      # 전체 호모그래피로 바꾼 점
        self.slot = None             # 고른 칸 (P2)
        self.slot_mat = None         # 그 칸 좌표로 바꾼 점
        self.dist = None             # 예상 접점과 줄 방향 거리
        self.straddle = []           # P3로 더한 칸
        self.cut_by = None           # P2a 아래를 가린 상자 번호
        self.side_by = None          # 옆을 가린 상자 번호
        self.occluded_slots = []     # P2a 여러 칸이라 가려짐으로 둔 칸
        self.kind = "aisle"          # slot / slot_cut / aisle / uncertain / unmonitored
        self.edge = False            # 화면 가장자리에 닿음 (P3에 쓰지 않음)

    def flags(self):
        f = []
        if self.cut_by is not None:
            f.append(f"P2a-cut({self.cut_by})")
        if self.side_by is not None:
            f.append(f"side({self.side_by})")
        if self.edge:
            f.append("edge")
        return ";".join(f)


class FrameResult:
    def __init__(self):
        self.boxes = []
        self.obs = {}          # 감시 칸 -> PRESENT / ABSENT / OCCLUDED
        self.cover = {}        # 감시 칸 -> P5 덮인 비율
        self.why = {}          # 감시 칸 -> 이유 글 (D3, L2)


def _assign(geom, params, px):
    """P2 + P2b. 화면 점 px를 칸에 넣는다. (칸, 그 칸 좌표의 점, 거리) 또는 None.
    감시 칸이 아닌 칸에도 넣는다(그래야 감시 칸 옆에 선 차를 통로 차로 착각하지 않는다)."""
    best = None
    for no, Hs in geom.slot_H.items():
        e = geom.expected.get(no)
        if e is None:
            continue
        sl = geom.table[no]
        p = params(no)
        q = apply_h(Hs, [px])[0]
        y0, y1 = sl.zone_y(p["zone_margin_entrance_cm"], p["zone_margin_back_cm"])
        if not (y0 <= q[1] <= y1):
            continue
        d = abs(q[0] - e[0])
        if d <= p["expect_max_dist_cm"] and (best is None or d < best[2]):
            best = (no, q, d)
    return best


def _segment_slots(geom, params, a_mat, b_mat, exclude, step=0.1):
    """매트 위 a → b 구간을 따라가며 들어가는 칸들(P2a)."""
    a, b = np.asarray(a_mat, float), np.asarray(b_mat, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    pts_mat = a + (b - a) * np.linspace(0, 1, n)[:, None]
    pts_px = apply_h(geom.Hinv, pts_mat)
    hit = []
    for px in pts_px:
        r = _assign(geom, params, px)
        if r is not None and r[0] not in exclude and r[0] not in hit:
            hit.append(r[0])
    return hit


def _cover_ratio(region, boxes):
    """칸 화면 영역 중 상자들(합집합)이 덮은 비율 (P5)."""
    reg = np.asarray(region, np.float32)
    x0, y0 = np.floor(reg.min(axis=0)).astype(int)
    x1, y1 = np.ceil(reg.max(axis=0)).astype(int)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    hits = [b for b in boxes if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0]
    if not hits:
        return 0.0
    if len(hits) == 1:
        b = hits[0]
        rect = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
        a_reg = cv2.contourArea(reg)
        if a_reg <= 0:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(rect, reg)
        return float(inter / a_reg)
    # 상자가 여럿이면 작은 그림에 칠해서 합집합 넓이를 잰다
    k = min(1.0, 300.0 / max(w, h))
    m_reg = np.zeros((int(h * k) + 2, int(w * k) + 2), np.uint8)
    cv2.fillConvexPoly(m_reg, np.round((reg - [x0, y0]) * k).astype(np.int32), 1)
    m_box = np.zeros_like(m_reg)
    for b in hits:
        p0 = (int(round((b[0] - x0) * k)), int(round((b[1] - y0) * k)))
        p1 = (int(round((b[2] - x0) * k)), int(round((b[3] - y0) * k)))
        cv2.rectangle(m_box, p0, p1, 1, -1)
    s = int(m_reg.sum())
    return float((m_reg & m_box).sum() / s) if s else 0.0


def judge_frame(geom, params, dets):
    """dets: [(x1, y1, x2, y2, conf), ...] (실행 해상도 픽셀). 돌려주는 것: FrameResult."""
    res = FrameResult()
    W, H = geom.size
    boxes = [BoxResult(i, d[:4], d[4] if len(d) > 4 else 1.0) for i, d in enumerate(dets)]
    res.boxes = boxes
    for b in boxes:
        b.contact_mat = apply_h(geom.H, [b.contact_px])[0]
        b.edge = b.box[0] <= 1 or b.box[2] >= W - 1 or b.box[3] >= H - 1

    # P2a: 아래가 잘린 상자, 옆이 가려진 상자
    touch = params(None)["cut_touch_ratio"] * H
    for a in boxes:
        ax1, ay1, ax2, ay2 = a.box
        best = None
        for b in boxes:
            if b is a:
                continue
            bx1, by1, bx2, by2 = b.box
            if not (by2 > ay2 + 0.5 and by1 <= ay2 + touch):   # a의 아래 변이 b 안, b가 더 앞
                continue
            ov = min(ax2, bx2) - max(ax1, bx1)
            if ov <= 0:
                continue
            if best is None or ov > best[1]:
                best = (b.i, ov)
        if best is not None:
            p0 = params(None)
            if best[1] >= p0["cut_overlap_ratio"] * (ax2 - ax1):
                a.cut_by = best[0]
            elif best[1] >= p0["side_overlap_ratio"] * (ax2 - ax1):
                a.side_by = best[0]

    present = {}            # 칸 -> 이유
    occluded = {}
    unknown = {}            # 이번 프레임은 판정하지 않는 칸 (시간 거르기를 멈추지도 다시 세지도 않음)
    order = sorted(boxes, key=lambda b: -b.box[3])      # 카메라에 가까운 상자부터
    for a in order:
        if a.cut_by is None:
            r = _assign(geom, params, a.contact_px)
            if r is None:
                a.kind = "aisle"
                continue
            a.slot, a.slot_mat, a.dist = r
        else:
            # 가리는 차 b의 바닥 뒤 끝 ~ a를 바꾼 점 (26 문서 1.6절)
            b = boxes[a.cut_by]
            pb = b.contact_mat
            if b.slot is not None:
                depth = params(b.slot)["car_length_cm"]
                exclude = {b.slot, *b.straddle}
            else:
                depth = params(None)["occluder_depth_cm"]
                exclude = set()
            far = pb + geom.far_dir(pb) * depth
            hit = _segment_slots(geom, params, far, a.contact_mat, exclude)
            if not hit:
                a.kind = "aisle"
                continue
            if len(hit) > 1:
                a.kind = "uncertain"
                a.occluded_slots = hit
                for no in hit:
                    if no in geom.monitored:
                        occluded.setdefault(no, f"P2a box{a.i} cut by box{b.i}: {hit}")
                continue
            a.slot = hit[0]
            a.slot_mat = apply_h(geom.slot_H.get(a.slot, geom.H), [a.contact_px])[0]
            a.kind = "slot_cut"
        if a.slot not in geom.monitored:
            a.kind = "unmonitored"
            continue
        if a.kind != "slot_cut":
            a.kind = "slot"
        present.setdefault(a.slot, f"box{a.i}")

        # P3 걸침: 가려지지 않은, 화면 가장자리에 닿지 않은 상자만.
        # 옆이 가려졌거나 가장자리에 닿은 상자는 걸침을 판정할 수 없으므로, 걸쳤을 수도 있는 옆 칸을 이번 프레임 "모름"으로 둔다
        if a.kind in ("slot", "slot_cut"):
            sl = geom.table[a.slot]
            p = params(a.slot)
            e = geom.expected[a.slot]
            cx = a.slot_mat[0] - (e[0] - sl.center[0])       # 예상 접점 차이만큼 되돌린 차 가운데
            lo, hi = cx - p["car_width_cm"] / 2, cx + p["car_width_cm"] / 2
            sure = a.kind == "slot" and a.side_by is None and not a.edge
            for nb in (sl.left, sl.right):
                if nb is None:
                    continue
                ns = geom.table[nb]
                ov = min(hi, ns.ix1) - max(lo, ns.ix0)
                if ov >= params(nb)["straddle_min_cm"] - 1e-9:
                    if sure:
                        a.straddle.append(nb)
                        if nb in geom.monitored:
                            present.setdefault(nb, f"P3 box{a.i} {ov:.2f}cm")
                    elif nb in geom.monitored:
                        unknown.setdefault(nb, f"P3 skipped box{a.i} ({'cut' if a.cut_by is not None else 'side' if a.side_by is not None else 'edge'})")

    # P5 가림: 통로 차 상자만 가리는 차로 쓴다
    aisle = [b.box for b in boxes if b.kind == "aisle"]
    for no in geom.monitored:
        if no not in geom.slot_H:
            continue
        cov = 0.0
        if aisle and no in geom.region:
            cov = _cover_ratio(geom.region[no], aisle)
        res.cover[no] = cov
        if no in present:
            res.obs[no] = PRESENT
            res.why[no] = present[no]
        elif no in occluded:
            res.obs[no] = OCCLUDED
            res.why[no] = occluded[no]
        elif cov >= params(no)["occlusion_ratio"]:
            res.obs[no] = OCCLUDED
            res.why[no] = f"P5 cover {cov * 100:.0f}%"
        elif no in unknown:
            res.obs[no] = None
            res.why[no] = unknown[no]
        else:
            res.obs[no] = ABSENT
            res.why[no] = ""
    return res
