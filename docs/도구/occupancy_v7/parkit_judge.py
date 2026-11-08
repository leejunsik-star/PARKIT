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
       - 방향 추정(docs/40, orientation_enable): 온전한 상자는 칸마다 세로·가로로 선 차의 예상 상자 크기와
         예상 접점을 비교해 비교 값이 가장 작은 칸과 방향을 같이 고른다. 가로 차는 x 허용 거리가 expect_max_dist_side_cm.
         가로는 세로보다 orient_margin 넘게 잘 맞을 때만 고르고, 그보다 애매하면 세로로 보되 가로였다면 더 막았을 칸은 "모름".
         잘렸거나 옆이 가려졌거나 화면 끝에 닿은 상자는 크기를 믿을 수 없어 세로로만 본다.
       - P2a: 잘린 상자는 "가리는 차의 뒤 끝 ~ 바꾼 점" 구간이 지나는 칸을 본다(가리는 차 자신의 칸은 뺀다).
         가리는 차가 통로 차면 구간을 잘린 차 자기 시선 위에서, 뒤 끝과 같은 깊이부터 시작한다(docs/38).
         한 칸이면 그 칸, 여러 칸이면 그 칸들을 가려짐, 없으면 통로 차.
       - P4: 어느 칸도 아니면 통로 차.
    4. P3: 가려지지 않은 상자만, 차 가운데 ± 차 폭/2가 옆 칸 inner에 straddle_min_cm 이상 들어가면 옆 칸도 차 있음.
       가로로 고른 차는 차 폭 대신 차 길이(car_length_cm)를 쓰고, 두 칸 넘게 이어서 본다.
       P2a에서 가리는 차가 가로면 뒤 끝 깊이로 car_width_cm을 쓴다.
    5. P5: 통로 차 상자들(합집합)이 칸 화면 영역을 occlusion_ratio 이상 덮거나, 그 칸보다 카메라에 가까운 칸에 선 차까지 합쳐
       occlusion_ratio_parked 이상 덮고, 그 칸에 차가 직접 잡히지 않았으면 가려짐(docs/38).
    6. 칸마다 우선순위: 차 있음 > 가려짐(P2a) > 가려짐(P5) > 모름(None) > 차 없음.
       "모름"은 옆이 가려진 상자라 걸침(P3)을 판정할 수 없는 옆 칸이다. 시간 거르기를 멈추지도 다시 세지도 않는다.
"""

import math

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
        self.edge_bottom = False     # 그중 화면 아래 끝에 닿음 (아래 변이 바닥 접점이 아님)
        self.approx_mat = None       # 아래가 잘린 상자의 대략 위치(매트). 웹 지도에만 쓴다 (docs/43)
        self.approx_ray = None       # v7: 그 차가 있을 수 있는 시선 구간 (대략 위치 → 잘린 접점). 웹 지도 추적에 쓴다 (docs/44)
        self.orient = None           # 칸에 넣은 차의 방향: "vert"(칸과 같은 방향) / "horiz"(칸과 수직) (docs/40)
        self.orient_alt = None       # 세로·가로가 애매할 때 가로 후보 칸 (그 칸 쪽은 "모름")

    def flags(self):
        f = []
        if self.orient == "horiz":
            f.append("horiz")
        if self.orient_alt is not None:
            f.append(f"horiz?({self.orient_alt})")
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


def _assign(geom, params, px, size=None):
    """P2 + P2b. 화면 점 px를 칸에 넣는다. (칸, 그 칸 좌표의 점, 거리, 방향, 애매한 가로 후보) 또는 None.
    감시 칸이 아닌 칸에도 넣는다(그래야 감시 칸 옆에 선 차를 통로 차로 착각하지 않는다).
    size가 없으면 v5와 같이 세로로만 보고 줄 방향 거리가 가장 가까운 칸을 고른다.
    size=(상자 폭, 높이)를 주면 방향 추정(docs/40): 칸마다 세로·가로 예상 상자와 비교해 칸과 방향을 같이 고른다.
      가로는 비교 값이 세로보다 orient_margin 넘게 작을 때만 고른다. 차이가 그보다 작으면(대각선에서 두 방향 상자가
      거의 같을 때) 세로로 고르고, 가로 후보 (칸, 그 칸 좌표의 점)를 같이 돌려준다. 가로였다면 더 막았을 칸은 "모름"이 된다."""
    if size is None:
        r = _assign_one(geom, params, px, None)
        return None if r is None else r[:5]
    best_v = _assign_one(geom, params, px, size, "vert")
    best_h = _assign_one(geom, params, px, size, "horiz")
    if best_h is None:
        return None if best_v is None else best_v[:5]
    margin = params(best_h[0])["orient_margin"]
    if best_v is None or best_h[5] < best_v[5] - margin:
        return best_h[:4] + (None,)
    if best_h[5] < best_v[5] + margin:
        return best_v[:4] + ((best_h[0], best_h[1]),)
    return best_v[:5]


def _assign_one(geom, params, px, size, want=None):
    """_assign의 칸 고르기. want가 "vert"/"horiz"면 그 방향 후보만 본다.
    돌려주는 것: (칸, 그 칸 좌표의 점, 거리, 방향, None, 비교 값) 또는 None."""
    best = None
    if not geom.on_floor(px):
        return None                      # 지평선 위(벽 등)의 점은 바닥이 아니다
    if size is not None and (size[0] <= 0 or size[1] <= 0):
        size = None
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
        es = geom.exp_box.get(no, {})
        if size is None or not p["orientation_enable"] or "vert" not in es:
            # v5와 같은 고르기 (방향 추정을 끈 칸, 크기를 믿을 수 없는 상자, 카메라 자세가 없는 칸)
            if want == "horiz":
                continue
            d = abs(q[0] - e[0])
            if d <= p["expect_max_dist_cm"] and (best is None or d < best[5]):
                best = (no, q, d, "vert", None, d)
            continue
        for o, max_x, max_y in (("vert", p["expect_max_dist_cm"], None),
                                ("horiz", p["expect_max_dist_side_cm"], p["expect_max_depth_side_cm"])):
            if (want is not None and o != want) or o not in es:
                continue
            ex, ey, ew, eh = es[o]                      # 예상 상자: 접점 x, y (매트), 화면 폭, 높이
            dx, dy = abs(q[0] - ex), abs(q[1] - ey)
            if dx > max_x or (max_y is not None and dy > max_y):
                continue
            cost = (p["orient_x_weight"] * dx + p["orient_depth_weight"] * dy +
                    p["orient_size_weight"] * (abs(math.log(size[0] / ew)) + abs(math.log(size[1] / eh))))
            if best is None or cost < best[5]:
                best = (no, q, dx, o, None, cost)
    return best


def _straddle(geom, params, no, q, horiz):
    """P3: 칸 no에 고른 차(그 칸 좌표의 접점 q)가 옆 칸 inner에 straddle_min_cm 이상 들어간 칸들 [(칸, 길이)].
    세로 차는 차 가운데 ± 차 폭/2로 바로 옆 칸만, 가로 차는 ± 차 길이/2로 옆 칸을 넘어 이어서 본다 (docs/40)."""
    sl = geom.table[no]
    p = params(no)
    e = geom.expected_horiz[no] if horiz else geom.expected[no]
    cx = q[0] - (e[0] - sl.center[0])                 # 예상 접점 차이만큼 되돌린 차 가운데
    half = (p["car_length_cm"] if horiz else p["car_width_cm"]) / 2
    lo, hi = cx - half, cx + half
    out = []
    for side in ("left", "right"):
        nb = getattr(sl, side)
        while nb is not None:
            ns = geom.table[nb]
            ov = min(hi, ns.ix1) - max(lo, ns.ix0)
            if ov < params(nb)["straddle_min_cm"] - 1e-9:
                break
            out.append((nb, ov))
            nb = getattr(ns, side) if horiz else None
    return out


def _segment_slots(geom, params, a_mat, b_mat, exclude, step=0.1):
    """매트 위 a → b 구간을 따라가며 들어가는 칸들(P2a)."""
    a, b = np.asarray(a_mat, float), np.asarray(b_mat, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    pts_mat = a + (b - a) * np.linspace(0, 1, n)[:, None]
    pts_px = apply_h(geom.Hinv, pts_mat)
    hit = []
    for px in pts_px:
        r = _assign(geom, params, px)            # 구간 위의 점은 상자 크기가 없으므로 세로로만 본다
        if r is not None and r[0] not in exclude and r[0] not in hit:
            hit.append(r[0])
    return hit


def _sight_start(geom, pb, far, a_mat):
    """P2a 구간의 시작점: 카메라 → 잘린 차 접점 시선 위에서, 가리는 차 뒤 끝(far)과 깊이가 같은 점 (docs/38).
    잘린 차의 진짜 접점은 이 시선 위에 있다. 가리는 차 가운데에서 시작하면 옆 칸을 스쳐 ?가 된다.
    카메라 자세가 없으면 시선을 알 수 없으므로 far를 그대로 쓴다."""
    if geom.cam_ground is None:
        return far
    c = np.asarray(geom.cam_ground, float)
    sight = np.asarray(a_mat, float) - c
    d = geom.far_dir(pb)                        # 깊이를 재는 방향
    den = float(sight @ d)
    if den <= 1e-9:
        return far
    return c + sight * (float((far - c) @ d) / den)


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
        b.edge_bottom = b.box[3] >= H - 1
        b.edge = b.box[0] <= 1 or b.box[2] >= W - 1 or b.edge_bottom

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
    use_orient = bool(params(None)["orientation_enable"])
    for a in order:
        if a.cut_by is None:
            # 방향 추정은 크기를 믿을 수 있는 상자(옆이 가려지지 않고 화면 끝에 닿지 않은 상자)에만 쓴다 (docs/40)
            size = None
            if use_orient and a.side_by is None and not a.edge:
                size = (a.box[2] - a.box[0], a.box[3] - a.box[1])
            r = _assign(geom, params, a.contact_px, size)
            if r is None:
                a.kind = "aisle"
                continue
            a.slot, a.slot_mat, a.dist, a.orient, alt = r
            if alt is not None:
                a.orient_alt = alt[0]
        else:
            alt = None
            # 가리는 차 b의 바닥 뒤 끝 ~ a를 바꾼 점 (26 문서 1.6절). b가 통로 차면 시작점을 a의 시선 위로 옮긴다 (docs/38)
            b = boxes[a.cut_by]
            pb = b.contact_mat
            if b.slot is not None:
                # 가로로 선 차는 시선 쪽 길이가 차 폭이다 (docs/40)
                depth = params(b.slot)["car_width_cm" if b.orient == "horiz" else "car_length_cm"]
                exclude = {b.slot, *b.straddle}
            else:
                depth = params(None)["occluder_depth_cm"]
                exclude = set()
            start = pb + geom.far_dir(pb) * depth
            sight = _sight_start(geom, pb, start, a.contact_mat)
            # 웹 지도용 대략 위치 (docs/43): 잘린 차 자기 시선 위에서 가리는 차 뒤 끝을 막 지난 곳.
            # 진짜 접점은 이 점이거나 시선을 따라 조금 더 먼 곳이다. 칸 판정에는 쓰지 않는다
            a.approx_mat = np.asarray(sight, float)
            a.approx_ray = (a.approx_mat, np.asarray(a.contact_mat, float))
            if b.slot is None:
                # 통로 차는 뒤의 여러 칸을 가리므로 잘린 차 자기 시선에서 시작한다. 주차된 차가 가릴 때는 그대로 둔다:
                # 대각선에서 옆 칸 차가 가리면 차 길이만큼 간 깊이가 진짜 접점을 지나쳐 제 칸을 놓친다 (docs/38)
                start = sight
            hit = _segment_slots(geom, params, start, a.contact_mat, exclude)
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
            a.orient = "vert"                                # 잘린 상자는 크기를 믿을 수 없어 세로로 본다
            e_slot = geom.expected.get(a.slot)
            if (a.slot in geom.monitored and e_slot is not None
                    and np.linalg.norm(np.asarray(e_slot, float) - sight) <= params(a.slot)["approx_slot_max_cm"]):
                # 감시 칸으로 정해졌으면 그 칸 예상 접점 (지도의 "주차 중" 칸과 맞다). 감시 밖 칸은 판정이 확인하지
                # 않으므로 시선 점을 그대로 쓴다: 합성 시험에서 감시 밖 칸 예상 접점은 10~12cm 틀렸다 (docs/43)
                # v7: 칸 예상 접점이 시선 점에서 approx_slot_max_cm 넘게 떨어지면 칸을 잘못 고른 것으로 보고 시선 점을 쓴다.
                #     엇갈리는 차가 엉뚱한 칸으로 6cm 넘게 튀어 ID가 끊기던 일을 막는다 (docs/44)
                a.approx_mat = np.asarray(e_slot, float)
                a.approx_ray = None                          # 칸 예상 접점은 시선 구간 위 점이 아니다
        if a.slot not in geom.monitored:
            a.kind = "unmonitored"
            continue
        if a.kind != "slot_cut":
            a.kind = "slot"
        horiz = a.orient == "horiz"
        tag = " horiz" if horiz else ""
        present.setdefault(a.slot, f"box{a.i}{tag}")

        # P3 걸침: 가려지지 않은, 화면 가장자리에 닿지 않은 상자만.
        # 옆이 가려졌거나 가장자리에 닿은 상자는 걸침을 판정할 수 없으므로, 걸쳤을 수도 있는 옆 칸을 이번 프레임 "모름"으로 둔다
        # 가로로 선 차는 줄 방향 길이가 차 길이라 2칸 넘게 막을 수 있으므로, 옆 칸을 넘어 이어서 본다 (docs/40)
        if a.kind in ("slot", "slot_cut"):
            sure = a.kind == "slot" and a.side_by is None and not a.edge
            for nb, ov in _straddle(geom, params, a.slot, a.slot_mat, horiz):
                if sure:
                    a.straddle.append(nb)
                    if nb in geom.monitored:
                        present.setdefault(nb, f"P3 box{a.i} {ov:.2f}cm{tag}")
                elif nb in geom.monitored:
                    unknown.setdefault(nb, f"P3 skipped box{a.i} ({'cut' if a.cut_by is not None else 'side' if a.side_by is not None else 'edge'})")
            if alt is not None:
                # 세로·가로가 애매한 상자: 가로였다면 더 막았을 칸은 "모름"으로 둔다 (docs/40)
                more = {alt[0]} | {nb for nb, _ in _straddle(geom, params, alt[0], alt[1], True)}
                for nb in sorted(more - {a.slot} - set(a.straddle)):
                    if nb in geom.monitored:
                        unknown.setdefault(nb, f"P3 orient? box{a.i}")

    # P5 가림: 통로 차와 칸에 선 차를 가리는 차로 쓴다 (docs/38)
    # 칸에 선 차는 아래 변이 칸 영역 아래 끝보다 화면에서 아래인 것, 즉 그 칸보다 카메라에 가까운 차만 쓴다.
    # 칸에 선 차는 떠나지 않으므로 기준을 따로 둔다: 통로 차 기준(30%)을 쓰면 잘 보이는 빈 칸까지 계속 가려짐으로 남는다
    aisle = [b.box for b in boxes if b.kind == "aisle"]
    parked = [b.box for b in boxes if b.kind in ("slot", "slot_cut", "unmonitored")]
    for no in geom.monitored:
        if no not in geom.slot_H:
            continue
        cov = cov_all = 0.0
        if no in geom.region:
            bottom = float(np.max(geom.region[no][:, 1]))
            near = [b for b in parked if b[3] > bottom]
            if aisle:
                cov = _cover_ratio(geom.region[no], aisle)
            cov_all = _cover_ratio(geom.region[no], aisle + near) if near else cov
        res.cover[no] = cov_all
        if no in present:
            res.obs[no] = PRESENT
            res.why[no] = present[no]
        elif no in occluded:
            res.obs[no] = OCCLUDED
            res.why[no] = occluded[no]
        elif cov >= params(no)["occlusion_ratio"]:
            res.obs[no] = OCCLUDED
            res.why[no] = f"P5 cover {cov * 100:.0f}%"
        elif cov_all >= params(no)["occlusion_ratio_parked"]:
            res.obs[no] = OCCLUDED
            res.why[no] = f"P5 cover {cov_all * 100:.0f}% (parked)"
        elif no in unknown:
            res.obs[no] = None
            res.why[no] = unknown[no]
        else:
            res.obs[no] = ABSENT
            res.why[no] = ""
    return res
