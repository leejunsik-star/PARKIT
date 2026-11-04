"""PARKIT 보정 ver2 — 기준점 자동 찾기 (화면 코드 없음)

손으로 찍은 기준점 4개 이상 → 호모그래피 → 기준선 목록의 선이 화면 어디 있을지 예상 → 예상선 근처에서 실제 칠한 선을 찾아
직선으로 맞춤 → 기준선끼리의 교점을 기준점으로 더함 → 다시 계산. 찾는 폭을 넓게 → 좁게 줄이며 되풀이한다.

주차장마다 다른 것은 모두 "기준선 목록"(json)에 있다. 이 파일에는 특정 매트의 치수, 선 모양, 칸 번호가 없다.

기준선 목록 형식 (docs/32 문서)
    {"lines": [{"id", "kind", "p1": [x, y], "p2": [x, y], ...종류별 값, "width"(선택), "bright"(선택, 기본 true)}],
     "points": [{"id", "a": 기준선 id, "b": 기준선 id, "mat": [x, y]}],
     "scan": {설정 덮어쓰기(선택). 매트 단위가 다른 주차장은 여기서 search, floor_clear 등을 바꾼다}}
    kind = edge    : 칠한 선의 한쪽 가장자리. "floor_dir": 바닥(선이 아닌) 쪽 매트 방향
           center  : 가는 선의 가운데
           tip_row : 여러 선의 끝을 이은 가상선. "tip_dir": 끝이 향하는 매트 방향. 그 방향으로 가장 먼 끝점들을 잇는다
    기준점 = 기준선 a, b 의 교점 (직선으로 늘린 교점). 읽을 때 "mat"과 같은지 확인한다.
"""

import json, math

import cv2
import numpy as np

from parkit_geom import apply_h, front_sign

KINDS = ("edge", "center", "tip_row")

# 찾기 설정 기본값. 실행 설정(run_v2.json)의 "autoref"와 기준선 목록의 "scan"이 덮어쓴다.
# 매트 단위 값(search, floor_clear, h_ransac, manual_warn, point_tol)은 주차장 단위에 맞춰 바꾼다(이 매트는 cm).
DEFAULTS = {
    "search": [2.0, 1.0, 0.5],      # 찾는 폭(매트 단위, 예상선 양쪽). 되풀이할 때마다 다음 값
    "sample_step_px": 5.0,          # 예상선을 따라 단면을 뽑는 간격(화면 픽셀)
    "tip_step_px": 2.0,             # tip_row 간격 (둥근 끝 꼭대기를 놓치지 않게 촘촘히)
    "min_samples": 12,              # 짧은 선(카스토퍼 등)도 단면을 이만큼은 뽑는다 (간격을 좁혀서)
    "end_margin": 0.1,              # edge/center 선분 양 끝에서 비우는 비율 (선 끝 모양에 흔들리지 않게)
    "profile_step_px": 0.5,         # 단면 위 점 간격
    "smooth_px": 0.8,               # 단면을 부드럽게 하는 폭
    "min_contrast": 20.0,           # 선과 바닥의 밝기 차 최소값 (0~255)
    "rel_strength": 0.8,            # 가장 센 후보의 이 비율 이상인 후보 중 예상선에 가장 가까운 것
                                    # (낮으면 처음 넓게 찾을 때 옆의 약한 나란한 선을 잡는다)
    "floor_clear": 0.6,             # edge/tip: 바닥 쪽이 이만큼(매트 단위) 칠해져 있지 않아야 한다
    "floor_clear_factor": 2.5,      # 선 폭이 적혀 있으면 floor_clear = 폭 × 이 값 (둘 중 큰 것)
    "width_tol": [0.4, 2.5],        # 잰 선 폭 / 적힌 선 폭 허용 범위
    "max_search_px": 220.0,         # 화면에서 찾는 폭의 최대값
    "ransac_px": 1.5,               # 직선 RANSAC 문턱 (픽셀)
    "ransac_iter": 150,
    "min_points": 6,                # 직선 하나에 필요한 점 수 (edge, center)
    "min_points_tip": 3,            # tip_row 에 필요한 끝점 수
    "found_ratio": 0.5,             # 찾음: 안쪽 점 비율
    "found_coverage": 0.4,          # 찾음: 안쪽 점이 덮는 길이 / 예상 선분 길이
    "found_rms_px": 1.2,            # 찾음: 직선에서 흩어진 정도
    "amb_coverage": 0.15,           # 애매함 기준 (이보다 못하면 못 찾음)
    "amb_rms_px": 2.5,
    "min_angle_deg": 15.0,          # 교점을 만드는 두 선의 최소 각도
    "h_ransac_px": 2.0,             # 전체 호모그래피 RANSAC 문턱 (화면 픽셀. 선 찾기 오차는 픽셀로 고르게 생긴다)
    "h_prune_px": 1.5,              # 맞춘 뒤 이 픽셀(또는 중앙값의 3배) 넘게 어긋난 자동 점은 뺀다
    "manual_warn": 1.0,             # 손 점이 자동 결과와 이만큼(매트 단위) 넘게 어긋나면 경고
    "min_auto_points": 4,           # 자동 점이 이보다 적으면 결과를 쓰지 않는다 (호모그래피를 따로 검사할 최소)
    "min_total_points": 8,          # 손 점 + 자동 점이 이보다 적으면 결과를 쓰지 않는다
    "use_ambiguous": True,          # 애매한 선에서 나온 교점도 쓸지
    "border_frac": 0.15,            # 화면 가장자리로 볼 폭 (가장자리/가운데 재투영 비교)
    "point_tol": 0.02,              # 기준점 매트 좌표 = 두 선 교점 허용 오차 (매트 단위)
    "min_spread": 0.01,             # 손 점 세 개가 만드는 삼각형 넓이 / (퍼진 폭)² 이 이보다 작으면 "한 줄"
}


def unknown_settings(d):
    """설정 dict에서 DEFAULTS에 없는 이름 (오타 찾기). "_"로 시작하는 설명 키는 뺀다."""
    return sorted(k for k in (d or {}) if not k.startswith("_") and k not in DEFAULTS)


def settings(*layers):
    p = dict(DEFAULTS)
    for d in layers:
        if d:
            p.update({k: v for k, v in d.items() if not k.startswith("_")})
    return p


# ---------------------------------------------------------------- 기준선 목록

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cross_lines(p1, p2, q1, q2):
    """두 직선(끝점으로 정함)의 교점. 나란하면 None."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12 * max(1.0, np.linalg.norm(d1) * np.linalg.norm(d2)):
        return None
    s = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    return p1 + s * d1


class RefLines:
    """기준선 목록. lines: id → dict, points: [dict]. 읽을 때 형식과 교점을 확인한다."""

    def __init__(self, data, path=None):
        self.path = path
        self.scan = data.get("scan", {}) or {}
        self.lines, self.points, errs = {}, [], []
        if unknown_settings(self.scan):
            errs.append(f"scan에 모르는 설정 이름이 있습니다: {unknown_settings(self.scan)} (parkit_autoref.DEFAULTS 참고)")
        tol = settings(self.scan)["point_tol"]
        for l in data.get("lines", []):
            lid = str(l.get("id", ""))
            if not lid or lid in self.lines:
                errs.append(f"기준선 id가 없거나 겹침: {lid!r}")
                continue
            kind = l.get("kind")
            if kind not in KINDS:
                errs.append(f"{lid}: 종류(kind)는 {KINDS} 중 하나여야 합니다: {kind!r}")
                continue
            p1, p2 = np.asarray(l["p1"], float), np.asarray(l["p2"], float)
            if np.linalg.norm(p2 - p1) <= 0:
                errs.append(f"{lid}: 두 끝점이 같습니다")
                continue
            line = {"id": lid, "kind": kind, "p1": p1, "p2": p2, "bright": bool(l.get("bright", True)),
                    "width": float(l["width"]) if l.get("width") else None, "note": l.get("note", "")}
            if kind == "edge":
                if "floor_dir" not in l:
                    errs.append(f"{lid}: edge에는 floor_dir(바닥 쪽 방향)이 있어야 합니다")
                    continue
                line["dir"] = _unit(l["floor_dir"])
            elif kind == "tip_row":
                if "tip_dir" not in l:
                    errs.append(f"{lid}: tip_row에는 tip_dir(끝이 향하는 방향)이 있어야 합니다")
                    continue
                line["dir"] = _unit(l["tip_dir"])
            else:
                u = _unit(p2 - p1)
                line["dir"] = np.array([-u[1], u[0]])
            self.lines[lid] = line
        for p in data.get("points", []):
            a, b = p.get("a"), p.get("b")
            if a not in self.lines or b not in self.lines or a == b:
                errs.append(f"기준점 {p.get('id')}: 기준선 {a}, {b}가 목록에 없거나 같습니다")
                continue
            la, lb = self.lines[a], self.lines[b]
            q = cross_lines(la["p1"], la["p2"], lb["p1"], lb["p2"])
            m = np.asarray(p["mat"], float)
            if q is None:
                errs.append(f"기준점 {p['id']}: 기준선 {a}, {b}가 나란해서 교점이 없습니다")
                continue
            if np.linalg.norm(q - m) > tol:
                errs.append(f"기준점 {p['id']}: 적힌 좌표 {m.tolist()}와 두 선의 교점 {np.round(q, 4).tolist()}가 다릅니다")
                continue
            self.points.append({"id": str(p["id"]), "a": a, "b": b, "mat": m})
        if errs:
            raise ValueError("기준선 목록 오류:\n  " + "\n  ".join(errs[:20]) + ("" if len(errs) <= 20 else f"\n  … 외 {len(errs) - 20}개"))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), path)


# ---------------------------------------------------------------- 단면 뽑기와 종류별 찾기

class Proj:
    """매트 → 화면 (호모그래피 역), 카메라 앞인지도 본다."""

    def __init__(self, H, front):
        self.Hinv = np.linalg.inv(np.asarray(H, float))
        self.front = front

    def __call__(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        q = np.c_[pts, np.ones(len(pts))] @ self.Hinv.T
        ok = q[:, 2] * self.front > 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        return uv, ok


def _profiles(gray, line, proj, size, search, P, step_px, extra=0.0):
    """예상선을 따라 단면을 뽑는다. 돌려주는 것: 화면 점 q (N,2), 바깥 방향 n (N,2), 단면 (N,T), t 눈금 (T), 픽셀/매트단위 (N)"""
    p1, p2 = line["p1"], line["p2"]
    L = np.linalg.norm(p2 - p1)
    u = (p2 - p1) / L
    ends, ok = proj([p1, p2])
    if not ok.all():
        return None
    Lpx = float(np.linalg.norm(ends[1] - ends[0]))
    if Lpx < 3:
        return None
    margin = 0.0 if line["kind"] == "tip_row" else P["end_margin"]
    n_s = int(max(P["min_samples"], min(400, Lpx * (1 - 2 * margin) / step_px)))
    s = np.linspace(margin, 1 - margin, n_s) * L
    m = p1 + s[:, None] * u
    dlt = 1e-3 * L
    q, ok0 = proj(m)
    qu, ok1 = proj(m + u * dlt)
    qd, ok2 = proj(m + line["dir"] * dlt)
    ok = ok0 & ok1 & ok2
    lt = qu - q
    lt /= np.linalg.norm(lt, axis=1, keepdims=True) + 1e-12
    n = np.c_[-lt[:, 1], lt[:, 0]]
    flip = np.sum((qd - q) * n, axis=1) < 0
    n[flip] *= -1
    scale = np.sum((qd - q) * n, axis=1) / dlt                 # 화면 픽셀 / 매트 단위 (바깥 방향)
    W, H = size
    S = np.clip(search * scale, 3.0, P["max_search_px"])
    inside = ok & (q[:, 0] > -S) & (q[:, 0] < W + S) & (q[:, 1] > -S) & (q[:, 1] < H + S) & (scale > 0)
    if inside.sum() < 2:
        return None
    q, n, scale, S, s = q[inside], n[inside], scale[inside], S[inside], s[inside]
    # 단면은 찾는 폭보다 넉넉히 뽑는다(바닥이 비었는지, 선 폭 검사에 필요). 후보 위치만 찾는 폭 안으로 제한한다
    Smax = min(float(S.max()) + float(extra * scale.max()) + 6.0, 2 * P["max_search_px"] + 60.0)
    t = np.arange(-Smax, Smax + 1e-9, P["profile_step_px"]).astype(np.float32)
    mx = (q[:, 0:1] + n[:, 0:1] * t[None, :]).astype(np.float32)
    my = (q[:, 1:2] + n[:, 1:2] * t[None, :]).astype(np.float32)
    prof = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    valid = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)          # 화면 안
    win = valid & (np.abs(t)[None, :] <= S[:, None])                        # 후보를 찾는 폭
    k = max(0.1, P["smooth_px"] / P["profile_step_px"])
    prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=k, sigmaY=1e-6)
    if not line["bright"]:
        prof = 255.0 - prof
    return {"q": q, "n": n, "prof": prof, "valid": valid, "win": win, "t": t, "scale": scale, "S": S, "s": s, "L": L}


def _edge_candidates(row, valid, win, t, step, clear_px, width_px, P):
    """밝음(선) → 어두움(바닥)으로 바뀌는 자리 후보: (t, 대비)"""
    d = np.gradient(row) / step
    out = []
    side = max(2, int(round(2.0 / step)))
    clear = max(side + 1, int(round(clear_px / step)))
    # 기울기가 충분히 가파른 극소점만 (파이썬 반복을 줄인다)
    cand = np.where(win[1:-1] & (d[1:-1] < -P["min_contrast"] / 8.0) & (d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]))[0] + 1
    for i in cand:
        a0, a1 = i - side - 2, i - 1                       # 선 쪽
        b0, b1 = i + 2, i + 2 + clear                       # 바닥 쪽 (비어 있어야 하는 길이)
        if a0 < 0 or b1 >= len(row) or not valid[a0] or not valid[b1]:
            continue
        paint = float(row[a0:a1].mean())
        floor_seg = row[b0:b1]
        floor = float(np.median(floor_seg))
        con = paint - floor
        if con < P["min_contrast"] or float(floor_seg.max()) > floor + 0.5 * con:
            continue
        if width_px:
            half = floor + 0.5 * con
            # 웹캠 화면은 선명하게 하는 처리 때문에 가장자리 바깥이 바닥보다 어둡게 파인다.
            # 그래서 기울기가 가장 가파른 점이 이미 반 높이 아래일 수 있다: 2픽셀까지 선 쪽으로 물러나 반 높이를 찾는다
            k0 = i
            while k0 > max(0, i - side) and row[k0] < half and valid[k0]:
                k0 -= 1
            j = k0
            while j > 0 and row[j] >= half - 1e-6 and valid[j]:
                j -= 1
            # 선 쪽 밝은 부분의 길이 (반 높이 자리부터)
            run = (k0 - j) * step
            if not (P["width_tol"][0] * width_px - 1.0 <= run <= P["width_tol"][1] * width_px + 2.0):
                continue
        # 소수점 위치 (기울기 최소점 포물선)
        den = d[i - 1] - 2 * d[i] + d[i + 1]
        off = 0.5 * (d[i - 1] - d[i + 1]) / den if abs(den) > 1e-9 else 0.0
        out.append((float(t[i] + np.clip(off, -1, 1) * step), con))
    return out


def _center_candidates(row, valid, win, t, step, width_px, P):
    """밝은 띠의 가운데 후보: (t, 대비, 잰 폭)"""
    out = []
    vrow = np.where(valid, row, np.nan)
    if np.isnan(vrow).all():
        return out
    bg = float(np.nanpercentile(vrow, 30))
    cand = np.where(win[1:-1] & (row[1:-1] - bg >= P["min_contrast"]) & (row[1:-1] >= row[:-2]) & (row[1:-1] > row[2:]))[0] + 1
    for i in cand:
        h = row[i] - bg
        if h < P["min_contrast"]:
            continue
        half = bg + h / 2
        l = i
        while l > 0 and valid[l - 1] and row[l - 1] >= half:
            l -= 1
        r = i
        while r < len(row) - 1 and valid[r + 1] and row[r + 1] >= half:
            r += 1
        if l == 0 or r == len(row) - 1:
            continue
        # 반 높이 지점 (선형 보간)
        lf = l - (row[l] - half) / max(1e-6, row[l] - row[l - 1])
        rf = r + (row[r] - half) / max(1e-6, row[r] - row[r + 1])
        w = (rf - lf) * step
        if width_px and not (P["width_tol"][0] * width_px - 1.0 <= w <= P["width_tol"][1] * width_px + 2.0):
            continue
        out.append((float(t[0] + (lf + rf) / 2 * step), float(h), w))
    # 같은 띠에서 나온 겹친 후보 정리
    out.sort()
    merged = []
    for c in out:
        if merged and abs(c[0] - merged[-1][0]) < 1.0:
            if c[1] > merged[-1][1]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _pick(cands, P):
    """대비가 센 후보들 중 예상선(t=0)에 가장 가까운 것."""
    if not cands:
        return None
    top = max(c[1] for c in cands)
    good = [c for c in cands if c[1] >= P["rel_strength"] * top]
    return min(good, key=lambda c: abs(c[0]))


# ---------------------------------------------------------------- 직선 맞추기

def fit_line(pts, w=None):
    """가중 주성분 직선: (점, 방향)"""
    pts = np.asarray(pts, float)
    w = np.ones(len(pts)) if w is None else np.asarray(w, float)
    c = (pts * w[:, None]).sum(0) / w.sum()
    X = (pts - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return c, vt[0]


def dist_to_line(pts, c, u):
    d = np.asarray(pts, float) - c
    return np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])


def ransac_line(pts, thr, iters, min_pts, rng):
    """RANSAC 직선. (점, 방향, 안쪽 표시) 또는 None"""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < max(2, min_pts):
        return None
    best = None
    for _ in range(iters if n > 3 else 1):
        i, j = rng.choice(n, 2, replace=False) if n > 2 else (0, 1)
        d = pts[j] - pts[i]
        if np.linalg.norm(d) < 1e-6:
            continue
        u = d / np.linalg.norm(d)
        inl = dist_to_line(pts, pts[i], u) <= thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_pts:
        return None
    for _ in range(3):
        c, u = fit_line(pts[best])
        inl = dist_to_line(pts, c, u) <= thr
        if inl.sum() < min_pts:
            break
        best = inl
    c, u = fit_line(pts[best])
    return c, u, best


class LineFit:
    def __init__(self, line):
        self.id, self.kind = line["id"], line["kind"]
        self.status = "missing"     # found / ambiguous / missing
        self.reason = "화면 밖"
        self.c = self.u = None
        self.pts = np.zeros((0, 2))
        self.inl = np.zeros(0, bool)
        self.rms = None
        self.coverage = 0.0
        self.expected = None        # 화면 예상 선분 (2,2)
        self.scale = None           # 픽셀 / 매트 단위 (가운데)

    def to_dict(self):
        return {"id": self.id, "status": self.status, "reason": self.reason, "points": int(len(self.pts)),
                "inliers": int(self.inl.sum()), "rms_px": None if self.rms is None else round(float(self.rms), 3),
                "coverage": round(float(self.coverage), 3)}


def detect_line(gray, line, proj, size, search, P, rng):
    """기준선 하나를 찾는다."""
    fit = LineFit(line)
    ends, ok = proj([line["p1"], line["p2"]])
    if not ok.all():
        return fit
    fit.expected = ends
    tip = line["kind"] == "tip_row"
    extra = max(P["floor_clear"], (line["width"] or 0) * P["floor_clear_factor"]) + 2 * (line["width"] or 0)
    pr = _profiles(gray, line, proj, size, search, P, P["tip_step_px"] if tip else P["sample_step_px"], extra)
    if pr is None:
        return fit
    fit.scale = float(np.median(pr["scale"]))
    step = P["profile_step_px"]
    width = line["width"]
    pts, offs = [], []
    for k in range(len(pr["q"])):
        sc = pr["scale"][k]
        wpx = width * sc if width else None
        clear_px = max(P["floor_clear"], (width or 0) * P["floor_clear_factor"]) * sc
        row, valid, win = pr["prof"][k], pr["valid"][k], pr["win"][k]
        if win.sum() < 6:
            pts.append(None)
            continue
        if line["kind"] == "center":
            c = _pick(_center_candidates(row, valid, win, pr["t"], step, wpx, P), P)
        else:
            cands = _edge_candidates(row, valid, win, pr["t"], step, clear_px, wpx, P)
            if tip:
                c = max(cands, key=lambda x: x[0]) if cands else None      # 끝: 가장 먼 것
            else:
                c = _pick(cands, P)
        pts.append(None if c is None else (pr["q"][k] + c[0] * pr["n"][k], c[0]))
    if tip:
        # 붙어 있는 표본끼리 한 덩어리(선 하나의 끝): 덩어리마다 가장 먼 점(둥근 끝 꼭대기)
        runs, cur = [], []
        for k, v in enumerate(pts):
            if v is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(v)
        if cur:
            runs.append(cur)
        # 꼭대기는 매트 좌표에서 끝 방향으로 가장 먼 점으로 고른다(비스듬히 보면 화면에서 가장 먼 점과 다르다)
        Hm = np.linalg.inv(proj.Hinv)
        cand = []
        for r in runs:
            q = np.array([v[0] for v in r])
            m = apply_h(Hm, q)
            cand.append(q[int(np.argmax(m @ line["dir"]))])
        cand = np.array(cand) if cand else np.zeros((0, 2))
    else:
        cand = np.array([v[0] for v in pts if v is not None]) if any(v is not None for v in pts) else np.zeros((0, 2))
    fit.pts = cand
    min_pts = P["min_points_tip"] if tip else P["min_points"]
    if len(cand) < min_pts:
        fit.reason = f"찾은 점이 적음 ({len(cand)}개)"
        return fit
    if tip:
        # 끝을 이은 직선이 여럿이면(칸 안 글자 등) 끝 방향으로 가장 먼 직선
        rest, lines = np.ones(len(cand), bool), []
        for _ in range(3):
            r = ransac_line(cand[rest], P["ransac_px"], P["ransac_iter"], min_pts, rng)
            if r is None:
                break
            c, u, inl_sub = r
            idx = np.where(rest)[0][inl_sub]
            lines.append((c, u, idx))
            rest[idx] = False
            if rest.sum() < min_pts:
                break
        if not lines:
            fit.reason = "끝점이 한 줄로 서지 않음"
            return fit
        nmean = pr["n"].mean(0)
        c, u, idx = max(lines, key=lambda L: float(np.mean(cand[L[2]] @ nmean)))
        inl = np.zeros(len(cand), bool)
        inl[idx] = True
    else:
        r = ransac_line(cand, P["ransac_px"], P["ransac_iter"], min_pts, rng)
        if r is None:
            fit.reason = "점이 한 줄로 서지 않음"
            return fit
        c, u, inl = r
    fit.c, fit.u, fit.inl = c, u, inl
    d = dist_to_line(cand[inl], c, u)
    fit.rms = float(np.sqrt(np.mean(d ** 2))) if len(d) else None
    along = cand[inl] @ u
    span = float(along.max() - along.min()) if inl.sum() > 1 else 0.0
    # 덮은 길이는 화면 안에 보이는 예상 선분 길이에 대해 잰다(선 일부가 화면 밖이어도 공정하게)
    W, H = size
    okc, a, b = cv2.clipLine((0, 0, int(W), int(H)), tuple(int(round(v)) for v in ends[0]), tuple(int(round(v)) for v in ends[1]))
    seen = float(np.hypot(b[0] - a[0], b[1] - a[1])) if okc else float(np.linalg.norm(ends[1] - ends[0]))
    if tip:
        seen *= 0.8      # 끝점은 선 사이 간격마다 하나라 양 끝까지 닿지 않는다
    fit.coverage = min(1.0, span / max(1e-6, seen))
    ratio = inl.sum() / len(cand)
    if ratio >= P["found_ratio"] and fit.coverage >= P["found_coverage"] and fit.rms <= P["found_rms_px"]:
        fit.status, fit.reason = "found", ""
    elif fit.coverage >= P["amb_coverage"] and fit.rms <= P["amb_rms_px"]:
        fit.status, fit.reason = "ambiguous", f"덮은 길이 {fit.coverage:.0%}, 안쪽 점 {ratio:.0%}, 흩어짐 {fit.rms:.2f}px"
    else:
        fit.status, fit.reason = "missing", f"덮은 길이 {fit.coverage:.0%}, 흩어짐 {fit.rms:.2f}px"
    return fit


def intersect_fits(fa, fb, min_angle_deg):
    if fa.c is None or fb.c is None:
        return None
    ang = math.degrees(math.acos(min(1.0, abs(float(fa.u @ fb.u)))))
    if ang < min_angle_deg:
        return None
    return cross_lines(fa.c, fa.c + fa.u, fb.c, fb.c + fb.u)


# ---------------------------------------------------------------- 전체: 되풀이 다듬기

def lsq_homography(px, mat):
    """화면 → 매트 호모그래피. 화면 픽셀 오차를 최소로 맞춘다(매트 → 화면으로 맞춘 뒤 뒤집음)."""
    G, _ = cv2.findHomography(np.asarray(mat, float), np.asarray(px, float), 0)
    if G is None:
        return None
    H = np.linalg.inv(G)
    return H / H[2, 2]


def px_residual(H, px, mat):
    """매트 점을 H로 화면에 돌려놓았을 때 찍힌 점과의 거리(픽셀)."""
    q = apply_h(np.linalg.inv(H), mat)
    return np.linalg.norm(q - np.asarray(px, float), axis=1)


class AutoResult:
    def __init__(self):
        self.ok = False
        self.H = None
        self.fits = {}
        self.autos = []            # [{"id","mat","px","status","inlier","used","dev_px"}]
        self.manual_bad = {}       # 손 점 id → 어긋난 거리(매트 단위)
        self.warnings = []
        self.rounds = []           # 되풀이마다 요약

    def counts(self):
        c = {"found": 0, "ambiguous": 0, "missing": 0}
        for f in self.fits.values():
            c[f.status] += 1
        return c

    def used_refs(self):
        return [a for a in self.autos if a["used"]]


def auto_find(gray, ref, manual, size, opts=None, excluded=(), seed=0):
    """손 점(manual: [{"id","mat","px"}], 4개 이상)으로 시작해 기준점을 자동으로 찾는다.
    excluded: 관리자가 뺀 자동 점 id. 같은 id의 손 점이 있으면 손 점을 쓴다(손 점 우선)."""
    P = settings(opts, ref.scan)
    res = AutoResult()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    rng = np.random.default_rng(seed)
    man = [m for m in manual if m.get("px") is not None]
    if len(man) < 4:
        res.warnings.append("손 점이 4개 이상 있어야 자동 찾기를 할 수 있습니다.")
        return res
    mpx = np.array([m["px"] for m in man], float)
    mmat = np.array([m["mat"] for m in man], float)
    # 손 점이 한 줄로 서 있으면 호모그래피가 정해지지 않는다.
    # 점이 4개면 어느 세 점도 한 줄이면 안 되고, 더 많으면 전체가 한 줄에 몰리지 않으면 된다.
    span = float(np.ptp(mmat, axis=0).max()) or 1.0
    areas = []
    for i in range(len(mmat)):
        for j in range(i + 1, len(mmat)):
            for k in range(j + 1, len(mmat)):
                a, b = mmat[j] - mmat[i], mmat[k] - mmat[i]
                areas.append(abs(a[0] * b[1] - a[1] * b[0]) / 2 / (span * span))
    flat = min(areas) if len(mmat) == 4 else max(areas)
    if flat < P["min_spread"]:
        res.warnings.append("손 점들이 거의 한 줄에 있습니다(점이 4개면 어느 세 점도 한 줄이면 안 됨). "
                            "다른 줄(깊이가 다른 곳)의 점을 섞어 찍으세요.")
        return res
    H = lsq_homography(mpx, mmat)
    if H is None:
        res.warnings.append("손 점으로 호모그래피를 구하지 못했습니다(점이 한 줄에 있는지 확인).")
        return res
    man_ids = {m["id"] for m in man}
    bad_manual = set()
    for rnd, search in enumerate(P["search"]):
        front = front_sign(np.linalg.inv(H), mmat)
        proj = Proj(H, front)
        fits = {lid: detect_line(gray, line, proj, size, search, P, rng) for lid, line in ref.lines.items()}
        autos = []
        for p in ref.points:
            fa, fb = fits[p["a"]], fits[p["b"]]
            st = {fa.status, fb.status}
            if "missing" in st or (not P["use_ambiguous"] and "ambiguous" in st):
                continue
            q = intersect_fits(fa, fb, P["min_angle_deg"])
            if q is None or not np.isfinite(q).all():
                continue
            pred, okp = proj([p["mat"]])
            sc = np.nanmedian([f.scale for f in (fa, fb) if f.scale])
            dev = float(np.linalg.norm(q - pred[0]))
            if not okp[0] or dev > search * sc * 1.5:
                continue
            autos.append({"id": p["id"], "mat": p["mat"].tolist(), "px": q.tolist(),
                          "status": "found" if st == {"found"} else "ambiguous", "dev_px": dev})
        use = [a for a in autos if a["id"] not in excluded and a["id"] not in man_ids]
        enough = len(use) >= P["min_auto_points"] and len(use) + len(man) >= P["min_total_points"]
        if not enough and res.H is not None:
            res.warnings.append(f"찾는 폭 {search}에서는 점이 {len(use)}개뿐이라 앞 회차 결과를 씁니다.")
            break
        if not enough:
            res.fits, res.autos = fits, [dict(a, inlier=False, used=False) for a in autos]
            res.warnings.append(f"자동으로 찾은 점이 {len(use)}개뿐이라 결과를 쓰지 않습니다"
                                f"(자동 {P['min_auto_points']}개, 손 점과 합쳐 {P['min_total_points']}개 필요). "
                                f"손 점이 맞는지, 찾는 폭(search)이 맞는지 보세요.")
            res.rounds.append({"search": search, "auto": len(autos), "used": len(use)})
            return res
        apx = np.array([a["px"] for a in use], float)
        amat = np.array([a["mat"] for a in use], float)
        G, mask = cv2.findHomography(amat, apx, cv2.RANSAC, P["h_ransac_px"])
        inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(use), bool)
        Ha = None if G is None else np.linalg.inv(G)
        if Ha is not None and inl.sum() >= P["min_auto_points"]:
            # 맞춘 뒤 많이 어긋난 점을 한 번 더 뺀다
            Ha = lsq_homography(apx[inl], amat[inl])
            r = px_residual(Ha, apx, amat)
            lim = max(P["h_prune_px"], 3.0 * float(np.median(r[inl])))
            inl &= r <= lim
        if (Ha is None or inl.sum() < P["min_auto_points"]) and res.H is not None:
            res.warnings.append(f"찾는 폭 {search}에서 자동 점이 맞지 않아 앞 회차 결과를 씁니다.")
            break
        if Ha is None or inl.sum() < P["min_auto_points"]:
            res.fits = fits
            res.warnings.append("자동 점들이 하나의 평면으로 맞지 않습니다. 손 점 4개가 맞는지 보세요.")
            return res
        Ha = lsq_homography(apx[inl], amat[inl])
        # 손 점 검사: 자동 점만으로 구한 결과와 크게 어긋나면 잘못 찍은 점
        e = np.linalg.norm(apply_h(Ha, mpx) - mmat, axis=1)
        bad_now = {m["id"]: float(err) for m, err in zip(man, e) if err > P["manual_warn"]}
        if len(bad_now) * 2 > len(man):
            # 손 점 대부분과 맞지 않는 자동 결과는 믿지 않는다 (처음 손 점이 틀려 엉뚱한 선을 잡았을 때)
            msg = f"찾는 폭 {search}: 자동 점이 손 점 {len(man)}개 중 {len(bad_now)}개와 맞지 않습니다."
            if res.H is not None:
                res.warnings.append(msg + " 앞 회차 결과를 씁니다.")
                break
            res.fits = fits
            res.autos = [dict(a, inlier=False, used=False) for a in autos]
            res.warnings.append(msg + " 자동 결과를 쓰지 않습니다. 손 점의 이름과 자리를 확인하세요"
                                      "(옆 칸에 찍은 점, 화면 밖 점이 있으면 지우고 다시 찍기).")
            res.rounds.append({"search": search, "auto": len(autos), "used": 0})
            return res
        bad_manual = bad_now
        good_m = [m for m in man if m["id"] not in bad_manual]
        allpx = np.r_[apx[inl], np.array([m["px"] for m in good_m]).reshape(-1, 2)]
        allmat = np.r_[amat[inl], np.array([m["mat"] for m in good_m]).reshape(-1, 2)]
        H = lsq_homography(allpx, allmat)
        res.rounds.append({"search": search, "auto": len(autos), "used": int(inl.sum()),
                           "lines": {s: sum(f.status == s for f in fits.values()) for s in ("found", "ambiguous", "missing")}})
        ids_in = {use[i]["id"] for i in np.where(inl)[0]}
        res.autos = [dict(a, inlier=a["id"] in ids_in, used=a["id"] in ids_in and a["id"] not in man_ids
                          and a["id"] not in excluded) for a in autos]
        res.fits = fits
        res.H = H
    res.manual_bad = bad_manual
    for mid, err in sorted(bad_manual.items()):
        res.warnings.append(f"손 점 {mid}가 자동으로 찾은 선과 {err:.2f}만큼 어긋납니다(옆 칸으로 밀려 찍었거나 이름이 틀린 것 같음). "
                            f"계산에서 뺐습니다. 다시 찍거나 지우세요.")
    res.H = H
    res.ok = True
    return res


def edge_report(refs, pose_err_px, size, frac=0.15):
    """화면 가장자리 점과 가운데 점의 재투영 오차(픽셀) 평균. 렌즈 왜곡이 있으면 가장자리가 커진다."""
    W, H = size
    edge, mid = [], []
    for r, e in zip(refs, pose_err_px):
        x, y = r["px"]
        border = x < W * frac or x > W * (1 - frac) or y < H * frac or y > H * (1 - frac)
        (edge if border else mid).append(float(e))
    return {"edge_mean": float(np.mean(edge)) if edge else None, "edge_n": len(edge),
            "mid_mean": float(np.mean(mid)) if mid else None, "mid_n": len(mid)}
