"""PARKIT 점유 판정 — 보정 도구 (25 문서 K1~K8, 22 문서 3.2·3.3절) — tkinter 화면

카메라마다 기준점을 찍어 호모그래피와 카메라 자세(solvePnP)를 구하고, 감시 칸을 고르고,
칸 꼭짓점을 손으로 고쳐 config\\cam_<이름>.json 에 저장한다. 확인용 이미지도 같이 저장한다.
이전 판(OpenCV 창)은 이전\\20261101_OpenCV창판\\ 에 있다.

사용법 (docs\\도구\\occupancy 에서):
    python calibrate.py                                         시작 창에서 웹캠을 보고 카메라(CAM-1, CAM-2 …)와 웹캠 번호를 고른다
    python calibrate.py --no-picker                             시작 창 없이 CAM-1, 저장된 입력(없으면 웹캠 0번)
    python calibrate.py --cam CAM-1 --source 0                 웹캠 0번 (웹캠 번호·해상도는 창에서도 바꿀 수 있다)
    python calibrate.py --cam CAM-2 --source 1                 두 번째 카메라
    python calibrate.py --cam CAM-1 --source 0 --width 1280 --height 720
    python calibrate.py --cam CAM-1 --source 녹화.mp4           영상 파일로 (Space로 멈춰 놓고 찍는다)
    python calibrate.py --cam CAM-1 --image 빈매트.jpg          사진으로
설정 파일이 이미 있으면 불러와서 이어서 고친다(K1). 단축키와 마우스는 창에서 ? 키(도움말)로 본다.
"""

import argparse, datetime, os, sys, time

import cv2
import numpy as np

from camera_input import parse_source, validate_source, LARIX_INPUTS
from parkit_config import CONFIG_DIR, DEFAULT_RUN, Params, cam_config_path, load_json, load_run, save_json
from parkit_draw import PlanView, camera_view_angle, poly, read_image, save_image, text
from parkit_geom import (REF_KINDS, CamGeom, SlotTable, apply_h, aspect_same, auto_monitored, edited_to_inner,
                         build_camera_config, estimate_hfov, ref_points)
from parkit_tk import (MONO_FONT, RES_CHOICES, UI_FONT, HelpWindow, ImageView, PlanAngleBar, ScrollFrame, check_tk,
                       is_typing, key_of, parse_res, plan_angle_setting, setup_style)
from parkit_video import FileSource, LiveSource, open_camera, source_kind

PICK_PX = 12                  # 점·꼭짓점을 잡는 거리 (화면 픽셀)
GOOD_ERR = 0.5                # 25 문서 6절: 기준점 오차 0.5cm 안
WARN_REPROJ = 5.0             # 카메라 자세 재투영 오차가 이 픽셀보다 크면 경고
NUDGE_STEPS = [0.25, 0.5, 1.0, 2.0, 5.0]
MODE_NAMES = {1: "① 기준점", 2: "② 감시 칸", 3: "③ 꼭짓점", 4: "④ 보조선"}
SNAP_PX = 14                  # 보조선 교차점에 붙는 거리 (화면 픽셀)
ZOOM_R, ZOOM_Z = 30, 5        # 확대 창: 원본에서 반지름 30픽셀을 5배로

HELP = [
    ("단계(모드)", [("1 / 2 / 3 / 4", "① 기준점 찍기 / ② 감시 칸 고르기 / ③ 칸 꼭짓점 고치기 / ④ 보조선 긋기")]),
    ("① 기준점", [
        ("평면도 클릭", "칸 고르기 → 오른쪽 목록에 그 칸의 기준점 4개가 나온다"),
        ("목록에서 점 클릭", "찍을 점 고르기 (입구 끝/안쪽 끝 × x 작은 쪽/큰 쪽)"),
        ("화면 왼쪽 클릭", "고른 점을 그 자리에 찍기 (이미 찍었으면 옮기기)"),
        ("점 끌기", "옮기기. 마우스를 놓을 때 한 번 다시 계산한다"),
        ("오른쪽 클릭", "가까운 점 지우기"),
        ("N", "다음 추천 점 고르기 (화면 안, 찍은 점들에서 먼 점)"),
        ("Delete / X", "분홍 겹원 점 지우기"),
    ]),
    ("미세 이동", [
        ("← ↑ → ↓  또는  J I L K", "①②: 분홍 겹원 점 / ③: 고른 꼭짓점 / ④: 고른 보조선을 이동 폭만큼 옮기기"),
        ("Shift + 화살표", "이동 폭의 5배만큼 옮기기"),
        (", / .", "이동 폭 줄이기 / 늘리기 (0.25 ~ 5 픽셀)"),
        ("Tab / Shift+Tab", "①②: 키보드로 움직일 점 바꾸기 (찍은 점 차례로) / ③: 꼭짓점 바꾸기 / ④: 보조선 바꾸기"),
        ("확대 창", "마우스를 움직이면 마우스 둘레, 키보드로 옮기면 그 대상 둘레를 5배로 보여 준다"),
        ("분홍 겹원", "지금 키보드로 움직일 점. 방금 찍은 점, 클릭하거나 끈 점, 목록에서 고른 점이 된다"),
    ]),
    ("② 감시 칸", [("왼쪽 클릭·끌기 (화면, 평면도)", "감시 켜기. 끌기 방식 '지나간 칸'이면 마우스가 지나간 칸, '네모'면 그린 네모 안에 가운데가 든 칸"),
                  ("오른쪽 클릭·끌기", "감시 끄기 (방식은 왼쪽과 같음)"),
                  ("A", "자동 선택으로 되돌리기 (화면 안에 다 들어온 칸)")]),
    ("평면도 각도", [("도면 각도 막대", "평면도를 0~359° 돌리기"), ("[0°] / [↻90°]", "처음 방향 / 90° 더 돌리기"),
                  ("[카메라 시점]", "카메라가 아래에서 위를 보게 돌림(카메라 화면처럼 가까운 칸이 아래). 보정이 바뀌면 따라감(각도 옆 *)")]),
    ("③ 꼭짓점", [("칸 클릭", "칸 고르기"), ("노란 꼭짓점 끌기", "빈 공간 모서리(흰 선 안쪽)를 인쇄된 선에 맞추기"),
                 ("R", "고른 칸을 호모그래피 결과로 되돌리기")]),
    ("④ 보조선", [
        ("빈 곳을 누르고 끌기", "새 직선 긋기 (호모그래피와 상관없는 화면 위 선)"),
        ("선 몸통 끌기 / 끝점 끌기", "선 전체 옮기기 / 끝점만 옮기기 (기울기 바꾸기)"),
        ("Tab / E", "다른 선 고르기 / 고른 선에서 전체 → 끝점 1 → 끝점 2 차례로 고르기"),
        ("D", "고른 선을 옆으로 복제 (나란한 기준선 만들기)"),
        ("Delete / X, 오른쪽 클릭", "고른 선 / 가까운 선 지우기"),
        ("G", "보조선 보이기/숨기기 (모든 단계)"),
        ("노란 원 (교차점)", "두 선이 만나는 점. ①·③에서 이 가까이 찍거나 놓으면 정확히 그 점에 붙는다 (SNAP)"),
    ]),
    ("계산", [("C", "다시 계산"), ("F", "화각 자동 추정 (기준점 6개 이상)"), ("[ / ]", "화각 0.5도 줄이기 / 늘리기")]),
    ("보기", [("O", "칸 선 보이기/숨기기"), ("Z", "판정 구역·칸 화면 영역·예상 접점 보이기"),
             ("H", "수평 확인선"), ("Space", "화면 멈춤/풀기")]),
    ("카메라", [("[보정할 카메라] 고르고 [바꾸기]", "CAM-1 ↔ CAM-2 바꾸기. 그 카메라의 보정 파일과 저장된 웹캠 번호로 다시 연다"),
               ("[웹캠] 번호 고르고 [열기]", "지금 카메라에 쓸 웹캠 바꾸기 (저장하면 보정 파일에 이 번호가 들어간다)"),
               ("시작 창", "python calibrate.py 만 실행하면 웹캠 미리보기에서 카메라마다 웹캠을 고른다")]),
    ("파일", [("S", "저장 (설정 json + 확인 사진)"), ("Q / Esc", "끝내기"), ("? / F1", "이 도움말")]),
]


# ---------------------------------------------------------------- 보조선 계산 (호모그래피와 상관없는 화면 위 직선)

def seg_dist(p, a, b):
    """점 p와 선분 a-b 사이 거리."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    L = float(ab @ ab)
    t = 0.0 if L == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L))
    return float(np.linalg.norm(p - (a + t * ab)))


def line_cross(l1, l2):
    """두 직선(끝점 두 개로 정한 무한 직선)이 만나는 점. 거의 나란하면 None."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6 * max(1.0, np.hypot(x1 - x2, y1 - y2) * np.hypot(x3 - x4, y3 - y4)):
        return None
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return np.array([(a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d])


def extend_line(l, W, H):
    """선을 화면 끝까지 늘린 두 끝점. 화면을 지나지 않으면 None."""
    a, b = np.array(l[:2], float), np.array(l[2:], float)
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d = d / n * (W + H) * 2
    ok, p1, p2 = cv2.clipLine((0, 0, int(W), int(H)), tuple(int(v) for v in a - d), tuple(int(v) for v in a + d))
    return (p1, p2) if ok else None


def ref_label(code):
    for c, label, *_ in REF_KINDS:
        if c == code:
            return label
    return code


def ranges(nos):
    """[1,2,3,5] → '1~3, 5'"""
    nos = sorted(nos)
    out, i = [], 0
    while i < len(nos):
        j = i
        while j + 1 < len(nos) and nos[j + 1] == nos[j] + 1:
            j += 1
        out.append(str(nos[i]) if i == j else f"{nos[i]}~{nos[j]}")
        i = j + 1
    return ", ".join(out) or "없음"


# ---------------------------------------------------------------- 입력

class CalSource:
    """사진, 웹캠/rtsp, 영상 파일에서 화면 한 장씩. 웹캠은 해상도를 바꿔 다시 열 수 있다."""

    def __init__(self, source, image, size):
        self.source, self.image, self.req = source, image, tuple(size)
        self.kind = "image" if image else ("file" if source_kind(source) == "file" else "live")
        self.src = None
        self.still = None
        self.frozen = None
        self.last = None
        self.t0 = None
        self.next_item = None
        self.msg = ""

    def open(self):
        if self.kind == "image":
            self.still = read_image(self.image)
            if self.still is None:
                self.msg = f"사진을 읽지 못함: {self.image}"
                return False
            self.msg = f"사진: {os.path.basename(self.image)}"
            return True
        if self.kind == "file":
            self.src = FileSource("cal", self.source, 1, loop=True)
        else:
            s = self.source
            self.src = LiveSource("cal", int(s) if source_kind(s) == "camera" else s, *self.req)
        if not self.src.open():
            self.msg = self.src.msg
            return False
        self.src.start()
        self.msg = self.src.msg
        self.t0 = time.time()
        return True

    def reopen(self, size, cam=None):
        """웹캠 번호·해상도 바꾸기. cam을 주면 그 웹캠으로 연다(사진·영상으로 시작했어도 웹캠으로 바뀐다).
        열지 못하면 앞의 입력으로 되돌리고 False."""
        old = (self.source, self.image, self.kind, self.req)
        self.close()
        self.req = tuple(size)
        self.frozen = None
        if cam is not None:
            self.source, self.image, self.kind = parse_source(cam), None, "live"
        if self.open():
            return True
        if self.kind == "live":
            time.sleep(1.0)                # 다른 프로그램이 막 놓은 웹캠은 풀리는 데 조금 걸린다
            if self.open():
                return True
        msg = self.msg
        self.source, self.image, self.kind, self.req = old
        self.open()
        self.msg = msg
        return False

    def get(self):
        if self.frozen is not None:
            return self.frozen
        if self.kind == "image":
            return self.still
        if self.kind == "file":
            # 실제 속도로 재생 (영상 시각이 지금 시각을 넘지 않을 때까지 꺼낸다)
            el = time.time() - self.t0
            while True:
                it = self.next_item if self.next_item is not None else self.src.get(timeout=0.001)
                self.next_item = None
                if it is None or it == "wait":
                    break
                if it[0] > el:
                    self.next_item = it
                    break
                if it[2] is not None:
                    self.last = it[2]
            return self.last
        f, t, n = self.src.latest()
        return f

    def toggle_freeze(self):
        if self.frozen is not None:
            self.frozen = None
            return False
        f = self.get()
        self.frozen = None if f is None else f.copy()
        return self.frozen is not None

    def describe(self, frame):
        size = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "?"
        if self.kind == "image":
            return f"사진 {os.path.basename(self.image)} · {size}"
        if self.kind == "file":
            return f"영상 {os.path.basename(str(self.source))} · {size}"
        fps = self.src.fps() if self.src else 0
        name = f"웹캠 {self.source}번" if source_kind(self.source) == "camera" else str(self.source)
        return f"{name} · {size} · {fps:.0f}fps"

    def close(self):
        if self.src is not None:
            self.src.close()
            self.src = None


# ---------------------------------------------------------------- 보정 상태와 계산 (화면 코드 없음)

class CalibModel:
    def __init__(self, name, source, cfg, table, params, size):
        self.name, self.source, self.cfg, self.table, self.params = name, source, cfg, table, params
        self.size = tuple(size)
        self.contact_mode = "ground"
        self.refs_all = {r["id"]: r for r in ref_points(table)}
        self.refs = []                 # [{"id", "mat", "px"}]
        self.sel_ref = None            # 다음에 클릭으로 찍을 기준점
        self.active_ref = None         # 화살표 키로 움직일 기준점 (방금 찍음 / 끌었음 / 목록에서 고름)
        self.sel_slot = None
        self.sel_corner = 0
        self.mode = 1
        self.monitored = set()
        self.monitored_auto = True
        self.edited = {}
        self.hfov = float(cfg["calib"].get("hfov_deg", 50.0))
        self.show_slots = True
        self.show_zone = False
        self.show_level = True
        self.drag = None
        self.sel_rect = None           # ② 네모로 감시 칸 고르는 중: (시작점, 지금 점, 켜기?) (보정 해상도 픽셀)
        self.lines = []                # 보조선 [[x1, y1, x2, y2], ...] (보정 해상도 픽셀)
        self.sel_line = None
        self.sel_end = None            # None = 선 전체, 0/1 = 끝점
        self.show_lines = True
        self.extend_lines = True
        self.snap = True
        self.hover = None              # 마우스 위치 (스냅 표시용)
        self.saved = True
        self.cam = self.geom = self.err = None
        self.path = cam_config_path(name)
        self.notes = []

    # ---------------- 불러오기 / 저장 (K1, K7, K8)
    def load(self):
        if not os.path.exists(self.path):
            self.notes.append(f"새 카메라 설정을 만듭니다: {os.path.basename(self.path)}")
            return
        d = load_json(self.path)
        from camera_contact import contact_mode
        self.contact_mode = contact_mode(d)
        cal = tuple(d.get("calib_size", self.size))
        if not aspect_same(cal, self.size):
            self.notes.append(f"저장된 보정 해상도 {cal[0]}x{cal[1]}와 지금 화면 {self.size[0]}x{self.size[1]}의 "
                              f"비율이 달라 기준점을 새로 찍습니다.")
            return
        s = self.size[0] / cal[0]
        dropped = []
        for r in d.get("ref_points", []):
            if r["id"] not in self.refs_all:
                dropped.append(r["id"])
                continue
            m = self.refs_all[r["id"]]["mat"]
            self.refs.append({"id": r["id"], "mat": list(m), "px": [r["px"][0] * s, r["px"][1] * s]})
        if dropped:
            self.notes.append(f"지금은 쓰지 않는 기준점(U선 가운데 CE/CB 등)을 뺐습니다: {', '.join(dropped)}")
        self.monitored = set(int(n) for n in d.get("monitored", []))
        self.monitored_auto = not self.monitored
        self.edited = edited_to_inner(self.table, {k: np.asarray(v, float) * s for k, v in (d.get("edited_corners") or {}).items()},
                                      d.get("edited_corners_kind", "cell"))
        self.hfov = float(d.get("intrinsics", {}).get("hfov_deg", self.hfov))
        self.lines = [[float(v) * s for v in l] for l in d.get("guide_lines", [])]
        self.notes.append(f"불러옴: {os.path.basename(self.path)} (기준점 {len(self.refs)}개, 감시 칸 {len(self.monitored)}개, "
                          f"고친 칸 {len(self.edited)}개, 보조선 {len(self.lines)}개, 저장 {d.get('date', '?')})")

    def save(self, frame):
        if self.cam is None:
            return None
        cam = dict(self.cam)
        cam["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam["input_note"] = "source: 웹캠 번호, 영상 파일 경로, rtsp 주소"
        cam["guide_lines"] = [[round(v, 2) for v in l] for l in self.lines]     # 보조선(보정 도구에서만 씀)
        save_json(self.path, cam)
        check = os.path.join(CONFIG_DIR, f"cam_{self.name}_check.jpg")
        im = frame.copy()
        self.draw(im, 1.0, all_slots=True)
        e = cam["homography_err_cm"]
        text(im, f"{self.name} {cam['date']}  ref err mean {e['mean']:.2f} max {e['max']:.2f} cm  "
                 f"monitored {len(cam['monitored'])}", (10, 30), 0.8, (255, 255, 255), 2)
        save_image(check, im)
        self.saved = True
        return check

    # ---------------- 계산 (K3)
    def recompute(self):
        self.cam = self.geom = self.err = None
        if len(self.refs) < 4:
            return
        try:
            mon = None if self.monitored_auto else sorted(self.monitored)
            cam = build_camera_config(self.name, self.source, self.size, self.refs, self.table, self.params,
                                      self.hfov, monitored=mon, edited=self.edited)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            return
        if self.monitored_auto:
            cam["monitored"] = auto_monitored(self.table, np.asarray(cam["homography"]), self.size,
                                              self.cfg["calib"].get("auto_margin_px", 0),
                                              ref_mats=[r["mat"] for r in self.refs])
            self.monitored = set(cam["monitored"])
        cam["contact_mode"] = self.contact_mode
        self.cam = cam
        self.err = {r["id"]: r["err_cm"] for r in cam["ref_points"]}
        self.geom = CamGeom(cam, self.table, self.size, self.params)

    def changed(self):
        self.saved = False
        self.recompute()

    def fit_fov(self):
        if len(self.refs) < 6:
            return None
        f, e = estimate_hfov([r["px"] for r in self.refs], [r["mat"] for r in self.refs], *self.size)
        old = self.hfov
        self.hfov = round(f, 2)
        self.changed()
        return old, self.hfov, e

    def reset_all(self):
        """카메라를 옮겼을 때: 화면 픽셀로 저장된 것(기준점, 고친 꼭짓점, 보조선)을 모두 지우고 감시 칸은 자동으로."""
        self.refs, self.edited, self.lines = [], {}, []
        self.active_ref = self.sel_line = self.sel_end = None
        self.monitored, self.monitored_auto = set(), True
        self.changed()

    def rescale(self, new_size):
        """해상도를 바꿨을 때. 비율이 같으면 점과 꼭짓점을 비율만큼 옮기고 True, 다르면 모두 지우고 False."""
        new_size = tuple(new_size)
        if new_size == self.size:
            return True
        same = aspect_same(self.size, new_size)
        if same:
            s = new_size[0] / self.size[0]
            for r in self.refs:
                r["px"] = [r["px"][0] * s, r["px"][1] * s]
            self.edited = {k: v * s for k, v in self.edited.items()}
            self.lines = [[v * s for v in l] for l in self.lines]
        else:
            self.refs, self.edited, self.lines = [], {}, []
            self.active_ref = self.sel_line = None
        self.size = new_size
        self.changed()
        return same

    def suggest_next(self):
        """아직 안 찍은 실측 기준점 중 화면 안에 있고 찍은 점들에서 먼 것."""
        placed = {r["id"] for r in self.refs}
        if self.geom is None:
            for r in self.refs_all.values():
                if r["measured"] and r["slot"] and r["id"] not in placed:
                    self.sel_ref, self.sel_slot = r["id"], r["slot"]
                    return
            return
        best = None
        P = np.array([r["px"] for r in self.refs])
        for r in self.refs_all.values():
            if r["id"] in placed or not r["measured"] or r["slot"] is None:
                continue
            q = self.geom.to_px([r["mat"]])[0]
            if not (30 <= q[0] <= self.size[0] - 30 and 30 <= q[1] <= self.size[1] - 30):
                continue
            d = np.min(np.linalg.norm(P - q, axis=1))
            if best is None or d > best[0]:
                best = (d, r)
        if best:
            self.sel_ref, self.sel_slot = best[1]["id"], best[1]["slot"]

    # ---------------- 점, 칸 다루기
    def ref_index(self, rid):
        for i, r in enumerate(self.refs):
            if r["id"] == rid:
                return i
        return None

    def near_ref(self, p, scale):
        best = None
        for i, r in enumerate(self.refs):
            d = np.linalg.norm(np.array(r["px"]) - p) * scale
            if d <= PICK_PX and (best is None or d < best[1]):
                best = (i, d)
        return None if best is None else best[0]

    def slot_at_px(self, p):
        if self.geom is None:
            return None
        for no, fl in self.geom.floor_px.items():
            if cv2.pointPolygonTest(fl.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0:
                return no
        return None

    def corners_of(self, no):
        """③에서 고치는 꼭짓점 = 빈 공간(inner) 네 모서리. 기준점 IE/IB와 같은 자리라 화면 칸 선과 크기가 같다."""
        if no in self.edited:
            return self.edited[no]
        if self.geom is not None and no in self.geom.inner_px:
            return self.geom.inner_px[no].copy()
        return None

    def place(self, p):
        """고른 기준점을 p에 찍는다(이미 있으면 옮긴다)."""
        if self.sel_ref is None:
            return False
        i = self.ref_index(self.sel_ref)
        if i is None:
            self.refs.append({"id": self.sel_ref, "mat": list(self.refs_all[self.sel_ref]["mat"]),
                              "px": [float(p[0]), float(p[1])]})
        else:
            self.refs[i]["px"] = [float(p[0]), float(p[1])]
        self.active_ref = self.sel_ref
        self.changed()
        return True

    def delete_ref(self, rid):
        i = self.ref_index(rid)
        if i is None:
            return False
        self.refs.pop(i)
        if self.active_ref == rid:
            self.active_ref = None
        self.changed()
        return True

    def set_active(self, rid):
        """찍은 점이면 키보드로 움직일 점으로 정한다."""
        if rid is not None and self.ref_index(rid) is not None:
            self.active_ref = rid
            return True
        return False

    def toggle_monitor(self, no):
        self.monitored_auto = False
        if no in self.monitored:
            self.monitored.discard(no)
        else:
            self.monitored.add(no)
        self.changed()

    def set_monitor(self, no, on):
        """감시 켜기(on=True)/끄기. 끄는 동안 칸을 여러 개 바꾸므로 다시 계산은 하지 않는다(끝에 changed()).
        바뀌었는지 돌려준다."""
        if no is None or (no in self.monitored) == bool(on):
            return False
        self.monitored_auto = False
        (self.monitored.add if on else self.monitored.discard)(no)
        self.saved = False
        return True

    # ---------------- 보조선
    def intersections(self):
        """보조선끼리 만나는 점(화면 안). 선을 늘려 보여 주지 않아도 직선으로 보고 구한다."""
        W, H = self.size
        out = []
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                q = line_cross(self.lines[i], self.lines[j])
                if q is not None and -5 <= q[0] <= W + 5 and -5 <= q[1] <= H + 5:
                    out.append(q)
        return out

    def snap_point(self, p, scale):
        """p 가까이에 보조선 교차점이 있으면 그 점으로 붙인다. (점, 붙였는지)"""
        if not self.snap or not self.show_lines:
            return np.asarray(p, float), False
        best = None
        for q in self.intersections():
            d = np.linalg.norm(q - np.asarray(p, float)) * scale
            if d <= SNAP_PX and (best is None or d < best[0]):
                best = (d, q)
        return (best[1].copy(), True) if best else (np.asarray(p, float), False)

    def line_hit(self, p, scale):
        """p 가까이의 보조선. (선 번호, 끝점 0/1 또는 None=선 몸통) 또는 None."""
        best = None
        for i, l in enumerate(self.lines):
            for e, q in ((0, l[:2]), (1, l[2:])):
                d = np.linalg.norm(np.array(q) - p) * scale
                if d <= PICK_PX and (best is None or d < best[0]):
                    best = (d, i, e)
        if best:
            return best[1], best[2]
        for i, l in enumerate(self.lines):
            if seg_dist(p, l[:2], l[2:]) * scale <= PICK_PX * 0.8:
                return i, None
        return None

    def delete_line(self, i=None):
        i = self.sel_line if i is None else i
        if i is None or not (0 <= i < len(self.lines)):
            return False
        self.lines.pop(i)
        self.sel_line, self.sel_end = (None, None) if not self.lines else (min(i, len(self.lines) - 1), None)
        self.saved = False
        return True

    def duplicate_line(self, gap=20.0):
        """고른 선을 옆으로 gap 픽셀 옮긴 복사본을 만든다(나란한 기준선을 쉽게 만들기)."""
        if self.sel_line is None:
            return False
        l = np.array(self.lines[self.sel_line], float)
        d = l[2:] - l[:2]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return False
        off = np.array([-d[1], d[0]]) / n * gap
        self.lines.append((l + np.r_[off, off]).tolist())
        self.sel_line, self.sel_end = len(self.lines) - 1, None
        self.saved = False
        return True

    def cycle_line(self, back=False):
        if not self.lines:
            return
        i = -1 if self.sel_line is None else self.sel_line
        self.sel_line = (i + (-1 if back else 1)) % len(self.lines)
        self.sel_end = None

    def cycle_end(self):
        if self.sel_line is not None:
            self.sel_end = {None: 0, 0: 1, 1: None}[self.sel_end]

    def focus_point(self):
        """키보드로 움직이는 대상의 위치(확대 창 가운데)."""
        if self.mode == 3 and self.sel_slot is not None:
            fl = self.corners_of(self.sel_slot)
            return None if fl is None else np.asarray(fl[self.sel_corner], float)
        if self.mode == 4:
            if self.sel_line is None:
                return None
            l = np.array(self.lines[self.sel_line], float)
            return (l[:2] + l[2:]) / 2 if self.sel_end is None else l[2 * self.sel_end:2 * self.sel_end + 2]
        i = self.ref_index(self.active_ref) if self.active_ref else None
        return None if i is None else np.asarray(self.refs[i]["px"], float)

    def start_drag(self, p, scale):
        """마우스를 누른 곳에서 끌 대상(점, 꼭짓점, 보조선)을 찾는다. ④에서 빈 곳이면 새 선을 긋기 시작한다."""
        if self.mode == 4:
            h = self.line_hit(p, scale)
            if h is not None:
                self.sel_line, self.sel_end = h
                self.drag = ("line", h[0], h[1], np.asarray(p, float), list(self.lines[h[0]]))
            else:
                self.lines.append([float(p[0]), float(p[1]), float(p[0]), float(p[1])])
                self.sel_line, self.sel_end = len(self.lines) - 1, 1
                self.drag = ("newline", len(self.lines) - 1)
            return True
        if self.mode == 1:
            i = self.near_ref(p, scale)
            if i is not None:
                self.drag = ("ref", i)
                self.sel_ref = self.active_ref = self.refs[i]["id"]
                self.sel_slot = self.refs_all[self.sel_ref]["slot"]
                return True
        elif self.mode == 3 and self.sel_slot is not None:
            fl = self.corners_of(self.sel_slot)
            if fl is not None:
                d = np.linalg.norm(fl - p, axis=1) * scale
                k = int(np.argmin(d))
                if d[k] <= PICK_PX:
                    if self.sel_slot not in self.edited:
                        self.edited[self.sel_slot] = fl.copy()
                    self.drag = ("corner", self.sel_slot, k)
                    self.sel_corner = k
                    return True
        return False

    def drag_to(self, p):
        """끄는 동안은 점만 옮기고 다시 계산하지 않는다(놓을 때 end_drag에서 계산)."""
        if self.drag is None:
            return
        kind = self.drag[0]
        if kind == "ref":
            self.refs[self.drag[1]]["px"] = [float(p[0]), float(p[1])]
        elif kind == "corner":
            _, no, k = self.drag
            self.edited[no][k] = np.asarray(p, float)
        elif kind == "newline":
            self.lines[self.drag[1]][2:] = [float(p[0]), float(p[1])]
        elif kind == "line":
            _, i, end, p0, orig = self.drag
            if end is None:
                dx, dy = p[0] - p0[0], p[1] - p0[1]
                self.lines[i] = [orig[0] + dx, orig[1] + dy, orig[2] + dx, orig[3] + dy]
            else:
                self.lines[i][2 * end:2 * end + 2] = [float(p[0]), float(p[1])]
        self.saved = False

    def end_drag(self, scale=1.0):
        """마우스를 놓을 때: 점·꼭짓점은 교차점에 붙이고 다시 계산, 너무 짧은 새 선은 지운다. 붙였는지 돌려준다."""
        if self.drag is None:
            return None
        kind = self.drag[0]
        snapped = False
        if kind == "ref":
            q, snapped = self.snap_point(self.refs[self.drag[1]]["px"], scale)
            self.refs[self.drag[1]]["px"] = [float(q[0]), float(q[1])]
        elif kind == "corner":
            _, no, k = self.drag
            q, snapped = self.snap_point(self.edited[no][k], scale)
            self.edited[no][k] = q
        elif kind == "newline":
            l = self.lines[self.drag[1]]
            if np.hypot(l[2] - l[0], l[3] - l[1]) * scale < 6:
                self.lines.pop(self.drag[1])
                self.sel_line = self.sel_end = None
            else:
                self.sel_end = None            # 다 그은 선은 화살표로 선 전체를 옮긴다
        self.drag = None
        if kind in ("ref", "corner"):
            self.changed()
        return snapped

    def nudge(self, dx, dy):
        """키보드 미세 이동: ①②는 키보드 대상 기준점(active_ref), ③은 고른 칸의 고른 꼭짓점, ④는 고른 보조선."""
        if self.mode == 4:
            if self.sel_line is None:
                return False
            l = self.lines[self.sel_line]
            for e in ((0, 1) if self.sel_end is None else (self.sel_end,)):
                l[2 * e] += dx
                l[2 * e + 1] += dy
            self.saved = False
            return True
        if self.mode == 3 and self.sel_slot is not None:
            fl = self.corners_of(self.sel_slot)
            if fl is None:
                return False
            if self.sel_slot not in self.edited:
                self.edited[self.sel_slot] = fl.copy()
            self.edited[self.sel_slot][self.sel_corner] += (dx, dy)
            self.changed()
            return True
        i = self.ref_index(self.active_ref) if self.active_ref else None
        if i is None:
            return False
        self.refs[i]["px"] = [self.refs[i]["px"][0] + dx, self.refs[i]["px"][1] + dy]
        self.changed()
        return True

    # ---------------- 그리기 (OpenCV, 화면 크기에 맞춘 이미지 위)
    def draw(self, im, s, all_slots=False):
        g = self.geom
        if g is not None and (self.show_slots or all_slots):
            for no in sorted(g.floor_px):
                mon = no in self.monitored
                if self.mode == 1 and not all_slots:
                    c, th = ((60, 220, 60), 1) if mon else ((150, 150, 150), 1)
                else:
                    c, th = ((60, 220, 60), 2) if mon else ((150, 150, 150), 1)
                if no in self.edited:
                    c = (230, 60, 230) if mon else (170, 90, 170)
                if self.mode == 3 and no == self.sel_slot:
                    continue
                poly(im, g.inner_px[no], c, th, s)
                cc = g.inner_px[no].mean(axis=0) * s
                if 0 <= cc[0] < im.shape[1] and 0 <= cc[1] < im.shape[0]:
                    text(im, str(no), (cc[0] - 8, cc[1] + 5), 0.42 if s < 1 else 0.7, c, 1)
                if self.show_zone and mon:
                    p = self.params(no)
                    sl = self.table[no]
                    y0, y1 = sl.zone_y(p["zone_margin_entrance_cm"], p["zone_margin_back_cm"])
                    z = np.array([[sl.x0, y0], [sl.x1, y0], [sl.x1, y1], [sl.x0, y1]])
                    poly(im, apply_h(np.linalg.inv(g.slot_H[no]), z), (200, 200, 0), 1, s)
                    if no in g.region:
                        poly(im, g.region[no], (255, 120, 0), 1, s)
                    e = g.expected.get(no)
                    if e is not None:
                        q = apply_h(np.linalg.inv(g.slot_H[no]), [e])[0] * s
                        cv2.drawMarker(im, tuple(q.astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
            if self.show_level and not all_slots:
                self.draw_level(im, s)
        if self.mode == 2 and self.sel_rect is not None and not all_slots:
            p0, p1, on = self.sel_rect
            a, b = (np.asarray(p0) * s).astype(int), (np.asarray(p1) * s).astype(int)
            cv2.rectangle(im, tuple(a), tuple(b), (60, 220, 60) if on else (60, 60, 230), 2)
        # ③에서 고른 칸: 끄는 중에도 바로 보이도록 꼭짓점에서 직접 그린다
        if self.mode == 3 and self.sel_slot is not None and not all_slots:
            fl = self.corners_of(self.sel_slot)
            if fl is not None:
                poly(im, fl, (0, 255, 255), 2, s)
                for k, q in enumerate(fl):
                    r = 8 if k == self.sel_corner else 5
                    cv2.circle(im, tuple((q * s).astype(int)), r, (0, 255, 255), 2)
        if self.show_lines and self.lines:
            self.draw_lines(im, s, not all_slots)
        # ①에서 고른 기준점이 있어야 할 곳 (지금 호모그래피로)
        if g is not None and self.mode == 1 and self.sel_ref and self.ref_index(self.sel_ref) is None and not all_slots:
            q = g.to_px([self.refs_all[self.sel_ref]["mat"]])[0] * s
            if np.isfinite(q).all():
                cv2.drawMarker(im, tuple(q.astype(int)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 1)
        for r in self.refs:
            q = (np.array(r["px"]) * s).astype(int)
            e = self.err.get(r["id"]) if self.err else None
            sel = r["id"] == self.sel_ref
            c = (0, 255, 255) if sel else ((60, 220, 60) if e is None or e <= GOOD_ERR else (0, 0, 255))
            cv2.circle(im, tuple(q), 6 if sel else 5, c, 2)
            cv2.drawMarker(im, tuple(q), c, cv2.MARKER_CROSS, 18, 1)
            if r["id"] == self.active_ref and not all_slots:
                # 화살표 키로 움직일 점: 분홍 겹원 + 네 방향 화살
                cv2.circle(im, tuple(q), 13, (255, 0, 255), 2)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.arrowedLine(im, (q[0] + 16 * dx, q[1] + 16 * dy), (q[0] + 26 * dx, q[1] + 26 * dy),
                                    (255, 0, 255), 2, tipLength=0.5)
            lab = r["id"] + (f" {e:.2f}" if e is not None and not (self.drag and self.drag[0] == "ref") else "")
            text(im, lab, (q[0] + 7, q[1] - 7), 0.42 if s < 1 else 0.6, c, 1)

    def draw_lines(self, im, s, live=True):
        """보조선(주황), 고른 선(하늘색), 화면 끝까지 늘린 선(가는 선), 교차점(노란 원), 스냅 표시."""
        W, H = self.size
        for i, l in enumerate(self.lines):
            sel = live and i == self.sel_line and self.mode == 4
            c = (255, 255, 0) if sel else (0, 140, 255)
            if self.extend_lines:
                e = extend_line(l, W, H)
                if e is not None:
                    cv2.line(im, tuple(int(v * s) for v in e[0]), tuple(int(v * s) for v in e[1]), c, 1, cv2.LINE_AA)
            a = (int(l[0] * s), int(l[1] * s))
            b = (int(l[2] * s), int(l[3] * s))
            cv2.line(im, a, b, c, 2, cv2.LINE_AA)
            for k, q in enumerate((a, b)):
                big = sel and self.sel_end == k
                r = 6 if big else 3
                cv2.rectangle(im, (q[0] - r, q[1] - r), (q[0] + r, q[1] + r), c, 2 if big else 1)
            if live and self.mode == 4:
                text(im, f"L{i + 1}", ((a[0] + b[0]) // 2 + 6, (a[1] + b[1]) // 2 - 6), 0.45, c, 1)
        for q in self.intersections():
            cv2.circle(im, (int(q[0] * s), int(q[1] * s)), 4, (0, 255, 255), 1, cv2.LINE_AA)
        if live and self.snap and self.hover is not None and self.mode in (1, 3):
            q, ok = self.snap_point(self.hover, s)
            if ok:
                cv2.circle(im, (int(q[0] * s), int(q[1] * s)), 10, (0, 255, 255), 2, cv2.LINE_AA)
                text(im, "SNAP", (int(q[0] * s) + 12, int(q[1] * s) + 16), 0.45, (0, 255, 255), 1)

    def draw_zoom(self, big, x0, y0, z):
        """확대 창 위에 보조선, 기준점, 고른 꼭짓점을 가늘게 겹친다."""
        def T(p):
            return (int(round((p[0] - x0) * z)), int(round((p[1] - y0) * z)))
        if self.show_lines:
            for i, l in enumerate(self.lines):
                c = (255, 255, 0) if (self.mode == 4 and i == self.sel_line) else (0, 140, 255)
                ext = extend_line(l, *self.size) if self.extend_lines else None
                a, b = ext if ext is not None else (l[:2], l[2:])
                cv2.line(big, T(a), T(b), c, 1, cv2.LINE_AA)
            for q in self.intersections():
                cv2.circle(big, T(q), 6, (0, 255, 255), 1, cv2.LINE_AA)
        for r in self.refs:
            c = (255, 0, 255) if r["id"] == self.active_ref else (60, 220, 60)
            cv2.drawMarker(big, T(r["px"]), c, cv2.MARKER_CROSS, 22, 1)
        if self.mode == 3 and self.sel_slot is not None:
            fl = self.corners_of(self.sel_slot)
            if fl is not None:
                pts = [T(q) for q in fl]
                cv2.polylines(big, [np.array(pts, np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(big, pts[self.sel_corner], 7, (0, 255, 255), 1, cv2.LINE_AA)

    def draw_level(self, im, s):
        """수평 확인선(26 문서 4절): 화면의 수평선(노랑)과, 매트 입구선(하늘색)을 화면에 그린 선."""
        g = self.geom
        W, H = self.size
        for y_mat in sorted({sl.entrance_y for sl in self.table if sl.measured}):
            xs = np.linspace(0, self.table.mat_size[0], 40)
            mats = np.c_[xs, np.full_like(xs, y_mat)]
            pts = g.to_px(mats)
            w = (np.c_[mats, np.ones(len(mats))] @ np.asarray(g.Hinv).T)[:, 2] * g.front
            ok = (w > 0) & np.isfinite(pts).all(axis=1) & (pts[:, 0] > -W) & (pts[:, 0] < 2 * W) & (pts[:, 1] > -H) & (pts[:, 1] < 2 * H)
            pts = pts[ok]
            if len(pts) < 2:
                continue
            poly(im, pts, (255, 200, 0), 1, s, closed=False)
            mid = pts[np.argmin(np.abs(pts[:, 0] - W / 2))]
            cv2.line(im, (0, int(mid[1] * s)), (int(W * s), int(mid[1] * s)), (0, 200, 255), 1, cv2.LINE_AA)

    # ---------------- 안내 글
    def guide(self):
        n = len(self.refs)
        if self.mode == 1:
            if n < 4:
                g = f"① 기준점 찍기 ({n}/6): 오른쪽 평면도에서 칸을 고르고 → 점 종류를 고른 뒤 → 화면에서 그 점을 클릭하세요."
            elif n < 6:
                g = f"① 기준점 {n}개 (계산 시작): 6~8개가 좋습니다. 화면 넓게, 앞줄·뒷줄과 입구·안쪽을 섞어 더 찍으세요."
            else:
                g = f"① 기준점 {n}개: 오차와 카메라 위치를 확인하고, 괜찮으면 ② 감시 칸으로 넘어가세요(2번 키)."
        elif self.mode == 2:
            g = "② 감시 칸: 초록 = 감시함. 왼쪽 클릭·끌기로 켜고, 오른쪽으로 끄세요. 다 되면 저장(S)."
        elif self.mode == 3:
            g = "③ 꼭짓점 고치기: 어긋난 칸만 클릭해 고르고, 노란 꼭짓점을 끌거나 화살표 키로 옮기세요."
        else:
            g = ("④ 보조선: 빈 곳을 누르고 끌어 선을 그으세요. 선이나 끝점은 끌거나 화살표로 옮깁니다. "
                 "두 선이 만나는 노란 원 가까이에 기준점을 찍으면 그 점에 붙습니다.")
        return g + ("" if self.saved else "   · 저장 안 됨")

    def warnings(self):
        w = []
        if self.cam:
            e = self.cam["homography_err_cm"]
            if e["max"] > GOOD_ERR:
                bad = [r["id"] for r in self.cam["ref_points"] if r["err_cm"] > GOOD_ERR]
                w.append(f"오차 {GOOD_ERR}cm 넘는 점: {', '.join(bad)} — 옮기거나 지우세요(빨간 점).")
            p = self.cam.get("pose")
            if p is None:
                w.append("카메라 자세를 못 구했습니다. 점 이름(입구/안쪽, x 작은/큰 쪽)을 확인하세요.")
            elif p["reproj_px"]["mean"] > WARN_REPROJ:
                w.append(f"카메라 자세 재투영 오차가 큽니다({p['reproj_px']['mean']:.1f}px). 화각 자동 추정(F)을 누르거나, "
                         f"점 이름이 맞는지 확인하세요. 렌즈 위치가 실제와 다르면 예상 접점(P2b)·가림(P5)이 틀립니다.")
            if p and abs(p["roll_deg"]) > 3:
                w.append(f"카메라가 옆으로 {p['roll_deg']:.1f}도 기울었습니다. 수평을 맞추는 것이 좋습니다(26 문서 4절).")
        if self.refs:
            measured = [r for r in self.refs if self.refs_all[r["id"]]["measured"]]
            if len(measured) < len(self.refs):
                w.append("칸 표에서 사진 추정(x_source)인 칸의 점이 섞여 있습니다(±0.3cm). 실측 칸의 점을 쓰세요.")
            rows = {self.table[self.refs_all[r['id']]['slot']].row for r in self.refs if self.refs_all[r['id']]['slot']}
            ends = {self.refs_all[r["id"]]["kind"][1] for r in self.refs if self.refs_all[r["id"]]["slot"]}
            if len(self.refs) >= 4 and (len(rows) < 2 or len(ends) < 2):
                w.append("깊이가 같은 점만 있습니다. 다른 줄, 입구 끝과 안쪽 끝을 섞어 찍으세요.")
        return w


# ---------------------------------------------------------------- 카메라 고르기 (시작 창)

MAX_WEBCAMS = 8               # 웹캠·DroidCam 가상 카메라 0~7번
THUMB_W = 220                 # 미리보기 폭 (픽셀)


def camera_names(cfg, config_dir=CONFIG_DIR):
    """고를 수 있는 카메라 이름: 저장된 보정 파일 + 실행 설정의 카메라 + CAM-1, CAM-2."""
    import glob
    names = {"CAM-1", "CAM-2", "CAM-3", "CAM-4"}
    for d in {config_dir, CONFIG_DIR}:
        for p in glob.glob(os.path.join(d, "cam_*.json")):
            names.add(os.path.basename(p)[4:-5])
    for c in cfg.get("cameras", []):
        b = os.path.basename(str(c.get("config", "")))
        if b.startswith("cam_") and b.endswith(".json"):
            names.add(b[4:-5])
    return sorted(names)


WEBCAM_MAP = os.path.join(CONFIG_DIR, "webcams.json")     # 카메라 이름 → 웹캠 번호 (보정 전 카메라도 기억)


def load_webcam_map():
    try:
        d = load_json(WEBCAM_MAP)
        return {str(k): parse_source(v) for k, v in d.get("webcams", {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def save_webcam_map(mapping):
    save_json(WEBCAM_MAP, {"_설명": "카메라 이름 → 웹캠 번호. calibrate.py 시작 창이 쓰고 읽는다. "
                                     "판정에는 보정 파일(cam_<이름>.json)의 source가 쓰이고, 이 표는 보정 전 카메라의 번호도 기억한다",
                            "webcams": {k: parse_source(v) for k, v in sorted(mapping.items())}})


def saved_camera(name, path_of=cam_config_path):
    """저장된 보정 파일의 (경로 또는 None, 웹캠 번호 또는 None, 저장 날짜)."""
    for p in dict.fromkeys([path_of(name), cam_config_path(name)]):
        if os.path.exists(p):
            try:
                d = load_json(p)
            except (OSError, ValueError):
                continue
            src = d.get("source")
            return p, (int(src) if source_kind(src) == "camera" else src), d.get("date", "?")
    return None, None, None


def file_source(path):
    """보정 파일 하나의 웹캠 번호 (웹캠이 아니면 그 값 그대로)."""
    src = load_json(path).get("source")
    return int(src) if source_kind(src) == "camera" else src


def set_saved_source(path, src):
    """보정 파일의 웹캠 번호만 바꾼다(다시 보정하지 않고 USB 순서가 바뀐 것을 맞출 때)."""
    d = load_json(path)
    d["source"] = parse_source(src)
    save_json(path, d)


class CamPicker:
    """시작 창: 연결된 웹캠을 미리보기로 보여 주고, 카메라(CAM-1, CAM-2 …)마다 쓸 웹캠을 고른 뒤 보정할 카메라를 고른다.
    run() → (카메라 이름, 웹캠 번호) 또는 None(끝내기).
    웹캠 번호는 보정 파일의 source가 먼저이고, 보정 파일이 없는 카메라는 config/webcams.json 에 기억한다."""

    def __init__(self, cfg, title="PARKIT 보정 도구 — 카메라 고르기", path_of=cam_config_path, config_dir=CONFIG_DIR):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk, self.ttk, self.mb = tk, ttk, messagebox
        self.path_of = path_of
        self.result = None
        self.thumbs = {}               # 웹캠 번호 → PhotoImage (지우지 않게 붙잡아 둔다)
        self.frames = {}               # 웹캠 번호 → 미리보기 사진 (카메라 이름을 다시 그릴 때)
        self.found = {}                # 웹캠 번호 → 해상도 글자 또는 None(없음)
        self.map = load_webcam_map()
        self.root = tk.Tk()
        self.root.title(title)
        setup_style(self.root)
        self.names = sorted(set(camera_names(cfg, config_dir)) | set(self.map))
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        # 자동 USB 탐색은 macOS 연속성 카메라를 켜 Larix 송출을 방해할 수 있다.
        self.root.after(100, self.refresh)

    def build(self):
        tk, ttk = self.tk, self.ttk
        r = self.root
        ttk.Label(r, text="연결된 웹캠 (사진 왼쪽 위 숫자 = 웹캠 번호, 아래 이름 = 지금 고른 카메라). "
                          "어느 웹캠이 어느 자리 카메라인지 보고 고르세요.",
                  style="Guide.TLabel").pack(anchor="w", padx=8, pady=(8, 4))
        self.grid = ttk.Frame(r)
        self.grid.pack(padx=8)
        self.cells = []
        for i in range(MAX_WEBCAMS):
            f = ttk.Frame(self.grid, padding=3)
            f.grid(row=i // 4, column=i % 4)
            c = tk.Canvas(f, width=THUMB_W, height=THUMB_W * 9 // 16, bg="#2d2d2d", highlightthickness=0)
            c.pack()
            lab = ttk.Label(f, text=f"웹캠 {i}번: 찾는 중…")
            lab.pack()
            self.cells.append((c, lab))
        rr = ttk.Frame(r)
        rr.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Button(rr, text="USB/연속성 웹캠 찾기 (Larix는 주소 선택)", command=self.probe_all).pack(side="left")
        ttk.Label(rr, text="USB를 다시 꽂았거나 다른 프로그램(카메라 앱, occupancy.py)을 닫았으면 누르세요.",
                  foreground="#555").pack(side="left", padx=8)
        box = ttk.LabelFrame(r, text="카메라별 입력 (Larix: 수신기를 켜고 phone1/phone2 주소 선택)", padding=6)
        box.pack(fill="x", padx=8)
        self.box = box
        self.sel = tk.StringVar(value=self.names[0])
        self.rows = {}
        for n in self.names:
            self.add_row(n)
        rr = ttk.Frame(r)
        rr.pack(fill="x", padx=8, pady=4)
        ttk.Label(rr, text="새 카메라 이름").pack(side="left")
        self.new_var = tk.StringVar(value=f"CAM-{len(self.names) + 1}")
        ttk.Entry(rr, textvariable=self.new_var, width=10).pack(side="left", padx=4)
        ttk.Button(rr, text="추가", command=self.add_camera).pack(side="left")
        self.warn = ttk.Label(r, text="", style="Warn.TLabel", wraplength=900, justify="left")
        self.warn.pack(anchor="w", padx=8)
        rr = ttk.Frame(r)
        rr.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Button(rr, text="보정 시작", style="Accent.TButton", command=self.start).pack(side="left")
        ttk.Button(rr, text="입력 번호·주소 저장", command=self.save_sources).pack(side="left", padx=6)
        ttk.Button(rr, text="끝내기", command=self.quit).pack(side="left")
        ttk.Label(rr, text="웹캠 번호만 저장 = 다시 보정하지 않고 카메라마다 웹캠 번호만 저장한다(USB 순서가 바뀌었을 때)",
                  foreground="#555").pack(side="left", padx=8)
        self.refresh()

    def stored(self, name):
        """저장된 웹캠 번호: 보정 파일의 source가 먼저, 없으면 webcams.json."""
        path, src, date = saved_camera(name, self.path_of)
        if src is None:
            src = self.map.get(name)
        return path, src, date

    def add_row(self, name):
        tk, ttk = self.tk, self.ttk
        path, src, date = self.stored(name)
        row = ttk.Frame(self.box)
        row.pack(fill="x", pady=1)
        ttk.Radiobutton(row, text=name, variable=self.sel, value=name, width=10, command=self.refresh).pack(side="left")
        ttk.Label(row, text="번호 / Larix 주소").pack(side="left")
        if src is not None:
            default = src
        else:                                       # 처음 보는 카메라: 아직 아무도 안 쓰는 가장 작은 번호
            taken = {parse_source(v["var"].get()) for v in self.rows.values()}
            default = next((i for i in range(MAX_WEBCAMS) if i not in taken), 0)
        var = tk.StringVar(value=str(default))
        cb = ttk.Combobox(row, textvariable=var, values=[str(i) for i in range(MAX_WEBCAMS)] + LARIX_INPUTS, width=38)
        cb.pack(side="left", padx=(4, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        cb.bind("<FocusOut>", lambda e: self.refresh())
        info = ttk.Label(row, text="", foreground="#555")
        info.pack(side="left")
        self.rows[name] = {"var": var, "path": path, "src": src, "date": date, "info": info}

    def add_camera(self):
        n = self.new_var.get().strip()
        if not n or any(ch in n for ch in '\\/:*?"<>| '):
            self.mb.showwarning("카메라 이름", "카메라 이름은 CAM-3 처럼 빈칸과 \\ / : * ? \" < > | 없이 적으세요.")
            return
        if n not in self.rows:
            self.add_row(n)
            self.new_var.set(f"CAM-{len(self.rows) + 1}")
        self.sel.set(n)
        self.refresh()

    def probe_all(self):
        """웹캠 0~4번을 차례로 열어 한 장씩 찍고 바로 닫는다(메인 스레드: 맥에서도 된다)."""
        for i in range(MAX_WEBCAMS):
            self.cells[i][1].configure(text=f"웹캠 {i}번: 찾는 중…")
        self.root.update_idletasks()
        for i in range(MAX_WEBCAMS):
            self.root.after(10 + 10 * i, lambda i=i: self.probe(i))
        self.root.after(20 + 10 * MAX_WEBCAMS, self.refresh)

    def probe(self, i):
        cap, f, msg = open_camera(i, 640, 360)
        if cap is not None:
            for _ in range(3):                       # 첫 장은 어둡게 나오는 웹캠이 있다
                ok, g = cap.read()
                if ok:
                    f = g
            cap.release()
        if f is None:
            self.found[i] = None
            self.frames.pop(i, None)
        else:
            self.found[i] = f"{f.shape[1]}x{f.shape[0]}"
            self.frames[i] = cv2.resize(f, (THUMB_W, int(f.shape[0] * THUMB_W / f.shape[1])))
        self.draw_cell(i)
        self.root.update_idletasks()

    def draw_cell(self, i):
        """미리보기 한 칸: 사진 + 웹캠 번호 + 이 웹캠을 쓰는 카메라 이름 (고르면 바로 바뀐다)."""
        from PIL import Image, ImageTk
        if i not in self.found:
            return                                    # 아직 찾는 중
        c, lab = self.cells[i]
        users = [n for n, s in self.picks().items() if s == i]
        sel = self.sel.get() in users
        who = ", ".join(users)
        c.delete("all")
        if self.found[i] is None:
            c.create_text(THUMB_W // 2, THUMB_W * 9 // 32, text="없음" + (f"\n({who}가 고름)" if who else ""),
                          fill="#ff8080" if who else "#888", justify="center")
            lab.configure(text=f"웹캠 {i}번: 없음" + (f" ← {who}" if who else ""))
            return
        im = self.frames[i].copy()
        text(im, f"{i}", (8, 34), 1.2, (0, 255, 255), 2)
        if who:
            color = (255, 0, 255) if sel else (0, 220, 0)
            (tw, th), _ = cv2.getTextSize(who, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(im, (0, im.shape[0] - th - 16), (tw + 16, im.shape[0]), (0, 0, 0), -1)
            text(im, who, (8, im.shape[0] - 8), 0.9, color, 2)
            cv2.rectangle(im, (1, 1), (im.shape[1] - 2, im.shape[0] - 2), color, 3)
        self.thumbs[i] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        c.configure(height=im.shape[0])
        c.create_image(0, 0, anchor="nw", image=self.thumbs[i])
        lab.configure(text=f"웹캠 {i}번" + (f" → {who}" if who else " (안 씀)"))

    def picks(self):
        return {n: parse_source(r["var"].get()) for n, r in self.rows.items()}

    def refresh(self):
        """고른 값이 바뀔 때마다: 미리보기의 카메라 이름, 줄마다 저장된 값과 다른지, 경고를 다시 그린다."""
        for n, r in self.rows.items():
            s = parse_source(r["var"].get())
            if r["path"]:
                base = f"보정 파일 {os.path.basename(os.path.dirname(r['path']))}/{os.path.basename(r['path'])} (저장 {r['date']})"
            else:
                base = "보정 파일 없음 (새로 보정)"
            if r["src"] is None:
                state, color = "저장된 입력 없음", "#555"
            elif r["src"] == s:
                state, color = f"저장된 입력 {r['src']}", "#555"
            else:
                state, color = f"저장 {r['src']} → {s} (저장 안 됨)", "#c00000"
            r["info"].configure(text=f"{base} · {state}", foreground=color)
        for i in range(MAX_WEBCAMS):
            self.draw_cell(i)
        return self.check()

    def check(self):
        """같은 웹캠을 두 카메라에 고른 것, 연결되지 않은 웹캠을 고른 것을 알린다."""
        w = []
        used = {}
        for n, s in self.picks().items():
            used.setdefault(s, []).append(n)
        for s, ns in used.items():
            if len(ns) > 1:
                w.append(f"웹캠 {s}번을 {', '.join(ns)}가 같이 씁니다. 각 장치는 서로 다른 번호를 고르세요.")
        for n, s in self.picks().items():
            if s in self.found and self.found[s] is None:
                w.append(f"{n}에 고른 웹캠 {s}번이 지금 연결돼 있지 않습니다."
                         + (" 이 카메라는 지금 보정할 수 없습니다." if n == self.sel.get() else " (번호는 저장할 수 있습니다)"))
        self.warn.configure(text="\n".join(w))
        return w

    def save_sources(self):
        """카메라마다 고른 웹캠 번호를 저장한다:
        webcams.json(모든 카메라) + 보정 파일의 source(보정한 카메라, 원본 config 와 v2 둘 다)."""
        picks = self.picks()
        try:
            picks = {n: validate_source(s) for n, s in picks.items()}
        except ValueError as exc:
            self.mb.showerror("입력 주소", str(exc))
            return
        self.map.update(picks)
        save_webcam_map(self.map)
        changed = []
        for n, s in picks.items():
            r = self.rows[n]
            # 웹캠 번호는 카메라 자리의 일이라 원본 보정(config)과 ver2 보정(config/v2)을 같이 바꾼다
            v2 = os.path.join(os.path.dirname(cam_config_path(n)), "v2", f"cam_{n}.json")
            files = [p for p in dict.fromkeys([self.path_of(n), cam_config_path(n), v2]) if os.path.exists(p)]
            for p in files:
                old = file_source(p)
                if old != s:
                    set_saved_source(p, s)
                    changed.append(f"{n}: {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} 웹캠 {old} → {s}번")
            if r["src"] != s and not files:
                changed.append(f"{n}: 웹캠 {s}번 (보정 파일 없음, 번호만 기억)")
            r["src"] = s
        self.refresh()
        print("[보정] 웹캠 번호 저장: " + (", ".join(changed) if changed else "바뀐 것 없음") + f" ({WEBCAM_MAP})")
        self.mb.showinfo("웹캠 번호", ("저장했습니다.\n" + "\n".join(changed) if changed else "바뀐 번호가 없습니다.")
                         + "\n\n판정 프로그램(occupancy.py)은 보정 파일의 웹캠 번호를 씁니다.")

    def start(self):
        n = self.sel.get()
        try:
            s = validate_source(self.picks()[n])
        except ValueError as exc:
            self.mb.showerror("입력 주소", str(exc))
            return
        if self.found.get(s, "?") is None:
            if not self.mb.askyesno("웹캠", f"웹캠 {s}번이 연결돼 있지 않습니다. 그래도 열어 볼까요?"):
                return
        # 보정 파일이 없는 카메라의 번호는 기억해 둔다(보정 파일이 있는 카메라는 보정을 저장할 때 바뀐다)
        for name, val in self.picks().items():
            if not self.rows[name]["path"]:
                self.map[name] = val
        save_webcam_map(self.map)
        self.result = (n, s)
        self.root.destroy()

    def quit(self):
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


# ---------------------------------------------------------------- 화면 (tkinter)

class CalibApp:
    def __init__(self, args, cfg, table, params):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk, self.ttk, self.mb = tk, ttk, messagebox
        self.args, self.cfg, self.table, self.params = args, cfg, table, params
        self.src = CalSource(args.source, args.image, (args.width, args.height))
        if not self.src.open():
            if self.src.kind == "live":
                time.sleep(1.0)        # 바로 앞 프로그램이 막 놓은 웹캠은 풀리는 데 조금 걸린다
            if self.src.kind != "live" or not self.src.open():
                raise SystemExit(self.src.msg)
        self.frame = self.first_frame()
        size = (self.frame.shape[1], self.frame.shape[0])
        if self.src.kind == "live" and size != (args.width, args.height):
            print(f"[보정] 요청한 해상도 {args.width}x{args.height}가 아니라 {size[0]}x{size[1]}로 열렸습니다.")
        self.m = self.make_model(args.cam, args.source if args.source is not None else 0, size)
        self.step_i = 2
        self.mouse = None
        self.zoom_from = "mouse"       # 확대 창 가운데: "mouse" 또는 "key"(키보드로 움직이는 대상)
        self.hover_hint = ""
        self.msg = ""
        self.quit_ok = False

        self.root = tk.Tk()
        self.root.title(self.title_text())
        setup_style(self.root)
        self.help = HelpWindow(self.root, "보정 도구 단축키와 마우스", HELP,
                               "단축키는 한글 입력 상태에서도 됩니다(같은 자리의 영문 키로 처리). "
                               "글자 입력 칸에 커서가 있으면 단축키가 먹지 않으니 화면을 한 번 클릭하세요.")
        self.build()
        for n in self.m.notes:
            print(f"[보정] {n}")
        self.msg = self.m.notes[-1] if self.m.notes else ""
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<Key>", self.on_key)
        self.root.after(30, self.tick)

    def first_frame(self):
        t0 = time.time()
        while time.time() - t0 < 10:
            f = self.src.get()
            if f is not None:
                return f
            time.sleep(0.05)
        self.src.close()
        raise SystemExit("화면을 받지 못했습니다.")

    # ---------------- 화면 만들기
    def build(self):
        tk, ttk = self.tk, self.ttk
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)
        # 위: 단계 안내 + 저장·도움말·끝내기 (늘 보이게)
        top = ttk.Frame(r)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.guide_var = tk.StringVar()
        ttk.Label(top, textvariable=self.guide_var, style="Guide.TLabel", anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="저장 (S)", style="Accent.TButton", command=self.save).grid(row=0, column=1, padx=(6, 2))
        ttk.Button(top, text="도움말 (?)", command=self.help.toggle).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="끝내기 (Q)", command=self.quit).grid(row=0, column=3, padx=(2, 6))
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()      # 노트북·맥북 화면에도 들어가게
        self.view = ImageView(r, width=max(480, min(1100, sw - 480)), height=max(300, min(620, sh - 300)))
        self.view.canvas.grid(row=1, column=0, sticky="nsew")
        # 아래: 마우스 안내 + 보기
        bot = ttk.Frame(r)
        bot.grid(row=2, column=0, columnspan=2, sticky="ew")
        bot.columnconfigure(0, weight=1)
        self.hint_var = tk.StringVar()
        ttk.Label(bot, textvariable=self.hint_var, style="Hint.TLabel", anchor="w").grid(row=0, column=0, sticky="ew")
        self.show_slots_var = tk.BooleanVar(value=True)
        self.show_zone_var = tk.BooleanVar(value=False)
        self.show_level_var = tk.BooleanVar(value=True)
        self.show_lines_var = tk.BooleanVar(value=True)
        for i, (txt, var) in enumerate((("칸 선 (O)", self.show_slots_var), ("보조선 (G)", self.show_lines_var),
                                        ("판정 구역 (Z)", self.show_zone_var), ("수평 확인선 (H)", self.show_level_var))):
            ttk.Checkbutton(bot, text=txt, variable=var, command=self.sync_view).grid(row=0, column=i + 1, padx=6)
        cv = self.view.canvas
        cv.bind("<ButtonPress-1>", self.on_press)
        cv.bind("<B1-Motion>", self.on_drag)
        cv.bind("<ButtonRelease-1>", self.on_release)
        for n in ("3", "2"):                                    # 맥 오른쪽 클릭은 2번 버튼
            cv.bind(f"<ButtonPress-{n}>", self.on_right)
            cv.bind(f"<B{n}-Motion>", self.on_right_drag)
            cv.bind(f"<ButtonRelease-{n}>", self.on_right_release)
        cv.bind("<Motion>", self.on_motion)
        cv.bind("<Leave>", lambda e: setattr(self, "mouse", None))

        # 오른쪽 패널: 내용이 길어지면(경고가 많을 때) 스크롤된다
        self.scroll = ScrollFrame(r, width=410)
        self.scroll.outer.grid(row=1, column=1, sticky="ns")
        pn = self.scroll.inner
        # 입력과 해상도
        f = ttk.LabelFrame(pn, text="입력", padding=6)
        f.pack(fill="x")
        rr = ttk.Frame(f)
        rr.pack(fill="x")
        ttk.Label(rr, text="보정할 카메라").pack(side="left")
        self.name_var = tk.StringVar(value=self.m.name)
        self.name_cb = ttk.Combobox(rr, textvariable=self.name_var, values=self.camera_list(), width=9)
        self.name_cb.pack(side="left", padx=4)
        self.name_cb.bind("<<ComboboxSelected>>", lambda e: self.change_camera())
        self.name_cb.bind("<Return>", lambda e: self.change_camera())
        ttk.Button(rr, text="바꾸기", command=self.change_camera).pack(side="left")
        ttk.Label(rr, text="(새 이름을 적어도 됨)", foreground="#555").pack(side="left", padx=4)
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=4)
        ttk.Label(rr, text="촬영 방식").pack(side="left")
        from camera_contact import MODE_LABELS
        self.contact_var = tk.StringVar(value=MODE_LABELS[self.m.contact_mode])
        cb = ttk.Combobox(rr, textvariable=self.contact_var, values=list(MODE_LABELS.values()),
                          width=29, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", self.change_contact_mode)
        ttk.Label(f, text="아이폰 상부: 박스 중앙 / 웹캠 사선: 기존 접점\nDroidCam 영상에 추가 렌즈 왜곡 보정을 적용하지 않습니다.",
                  foreground="#555").pack(anchor="w")
        self.src_var = tk.StringVar()
        ttk.Label(f, textvariable=self.src_var).pack(anchor="w", pady=(2, 0))
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Label(rr, text="번호 / Larix 주소").pack(side="left")
        cur = str(self.src.source) if self.src.kind == "live" else "0"
        self.cam_var = tk.StringVar(value=cur)
        ttk.Combobox(rr, textvariable=self.cam_var, values=[str(i) for i in range(MAX_WEBCAMS)] + LARIX_INPUTS, width=32).pack(side="left", padx=(4, 8))
        ttk.Label(rr, text="해상도").pack(side="left")
        self.res_var = tk.StringVar(value=f"{self.m.size[0]}x{self.m.size[1]}")
        ttk.Combobox(rr, textvariable=self.res_var, values=RES_CHOICES, width=11).pack(side="left", padx=4)
        ttk.Button(rr, text="열기", command=self.change_input).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(2, 0))
        self.freeze_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(rr, text="화면 멈춤 (Space)", variable=self.freeze_var, command=self.toggle_freeze).pack(side="left")
        ttk.Label(rr, text="웹캠 번호·해상도를 고르고 [열기]", foreground="#555").pack(side="left", padx=(10, 0))
        # 단계
        f = ttk.LabelFrame(pn, text="단계 (1, 2, 3, 4 키)", padding=4)
        f.pack(fill="x", pady=(4, 0))
        self.mode_var = tk.IntVar(value=1)
        for k in (1, 2, 3, 4):
            ttk.Radiobutton(f, text=MODE_NAMES[k], variable=self.mode_var, value=k,
                            command=lambda: self.set_mode(self.mode_var.get())).pack(side="left", padx=(0, 6))
        # 평면도
        f = ttk.LabelFrame(pn, text="매트 평면도 (칸 클릭)", padding=4)
        f.pack(fill="x", pady=(4, 0))
        self.plan_box = PlanView(self.table, px_per_cm=4.6, margin=6).size     # 평면도 칸 크기 (돌려도 그대로)
        self.plan = self.make_plan(0.0)
        self.plan_view = tk.Canvas(f, width=self.plan_box[0], height=self.plan_box[1], highlightthickness=0, bg="#2d2d2d")
        self.plan_view.pack()
        self.plan_view.bind("<ButtonPress-1>", lambda e: self.on_plan_press(e, True))
        self.plan_view.bind("<B1-Motion>", self.on_plan_drag)
        self.plan_view.bind("<ButtonRelease-1>", self.on_plan_release)
        for n in ("3", "2"):                                    # 맥 오른쪽 클릭은 2번 버튼
            self.plan_view.bind(f"<ButtonPress-{n}>", lambda e: self.on_plan_press(e, False))
            self.plan_view.bind(f"<B{n}-Motion>", self.on_plan_drag)
            self.plan_view.bind(f"<ButtonRelease-{n}>", self.on_plan_release)
        self.plan_img = None
        self.plan_off = (0, 0)
        a0, follow = plan_angle_setting(self.cfg["display"].get("plan_angle", 0))
        self.angle_bar = PlanAngleBar(f, self.cam_view_angle, cams=("",), angle=a0, follow=follow,
                                      on_note=lambda s: setattr(self, "msg", s))
        self.angle_bar.frame.pack(fill="x", pady=(4, 0))
        self.mon_op = None             # ② 감시 칸 끌기 중인 동작
        # 단계별 내용
        self.mode_box = ttk.Frame(pn)
        self.mode_box.pack(fill="x", pady=(4, 0))
        self.build_mode1()
        self.build_mode2()
        self.build_mode3()
        self.build_mode4()
        # 계산 결과
        f = ttk.LabelFrame(pn, text="계산 결과", padding=6)
        f.pack(fill="x", pady=(4, 0))
        self.res_lbl = ttk.Label(f, text="", font=(MONO_FONT, 9), justify="left")
        self.res_lbl.pack(anchor="w")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Label(rr, text="가로 화각").pack(side="left")
        ttk.Button(rr, text="−0.5", width=5, command=lambda: self.add_fov(-0.5)).pack(side="left", padx=2)
        self.fov_var = tk.StringVar()
        ttk.Label(rr, textvariable=self.fov_var, width=7, anchor="center").pack(side="left")
        ttk.Button(rr, text="+0.5", width=5, command=lambda: self.add_fov(0.5)).pack(side="left", padx=2)
        ttk.Button(rr, text="자동 추정 (F)", command=self.fit_fov).pack(side="left", padx=(6, 0))
        self.warn_lbl = ttk.Label(f, text="", style="Warn.TLabel", wraplength=380, justify="left")
        self.warn_lbl.pack(anchor="w", pady=(4, 0))
        self.msg_var = tk.StringVar()
        ttk.Label(pn, textvariable=self.msg_var, wraplength=390, foreground="#2b5797").pack(anchor="w", pady=(6, 0))
        self.set_mode(1)

    def build_mode1(self):
        tk, ttk = self.tk, self.ttk
        f = self.m1 = ttk.LabelFrame(self.mode_box, text="기준점", padding=6)
        self.slot_var = tk.StringVar()
        ttk.Label(f, textvariable=self.slot_var).pack(anchor="w")
        self.kind_var = tk.StringVar()
        self.kind_btns = []
        for code, label, *_ in REF_KINDS:
            b = ttk.Radiobutton(f, text="", variable=self.kind_var, value=code, command=self.pick_kind)
            b.pack(anchor="w")
            self.kind_btns.append((code, b))
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 2))
        ttk.Button(rr, text="다음 추천 점 (N)", command=self.next_ref).pack(side="left")
        ttk.Button(rr, text="고른 점 지우기 (Del)", command=self.delete_sel).pack(side="left", padx=4)
        ttk.Button(rr, text="모두 지우고 새로 보정", command=self.reset_all).pack(side="left")
        self.add_step_row(f)
        self.active_var = tk.StringVar()
        ttk.Label(f, textvariable=self.active_var, foreground="#b000b0", wraplength=380).pack(anchor="w", pady=(4, 0))
        ttk.Label(f, text="찍은 점 (클릭하면 키보드 대상으로 고름)").pack(anchor="w", pady=(2, 0))
        self.ref_list = tk.Listbox(f, height=5, font=(MONO_FONT, 9), activestyle="none", exportselection=False)
        self.ref_list.pack(fill="x")
        self.ref_list.bind("<<ListboxSelect>>", self.on_ref_list)

    def build_mode2(self):
        tk, ttk = self.tk, self.ttk
        f = self.m2 = ttk.LabelFrame(self.mode_box, text="감시 칸", padding=6)
        self.mon_var = tk.StringVar()
        ttk.Label(f, textvariable=self.mon_var, wraplength=380, justify="left").pack(anchor="w")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Label(rr, text="끌기 방식").pack(side="left")
        how = str(self.cfg["display"].get("monitor_drag", "paint"))
        self.drag_how_var = tk.StringVar(value=how if how in ("paint", "rect") else "paint")
        for txt, v in (("지나간 칸", "paint"), ("네모 안 칸", "rect")):
            ttk.Radiobutton(rr, text=txt, variable=self.drag_how_var, value=v,
                            command=self.view.canvas.focus_set).pack(side="left", padx=(6, 0))
        ttk.Label(f, text="왼쪽 클릭·끌기 = 감시 켜기, 오른쪽 = 끄기 (카메라 화면, 평면도 모두)", foreground="#555").pack(anchor="w")
        ttk.Button(f, text="자동 선택으로 되돌리기 (A)", command=self.auto_mon).pack(anchor="w", pady=(4, 0))
        ttk.Label(f, text="자동 선택 = cell 네 꼭짓점이 모두 화면 안에 들어온 칸 (K5)", foreground="#555").pack(anchor="w")

    def build_mode3(self):
        tk, ttk = self.tk, self.ttk
        f = self.m3 = ttk.LabelFrame(self.mode_box, text="칸 꼭짓점", padding=6)
        self.edit_var = tk.StringVar()
        ttk.Label(f, textvariable=self.edit_var, wraplength=380, justify="left").pack(anchor="w")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Label(rr, text="꼭짓점").pack(side="left")
        self.corner_var = tk.IntVar(value=0)
        for k, nm in enumerate(("왼쪽 위", "오른쪽 위", "오른쪽 아래", "왼쪽 아래")):
            ttk.Radiobutton(rr, text=str(k + 1), variable=self.corner_var, value=k,
                            command=lambda: setattr(self.m, "sel_corner", self.corner_var.get())).pack(side="left")
        ttk.Button(f, text="고른 칸 초기화 (R)", command=self.reset_slot).pack(anchor="w", pady=(4, 0))
        self.add_step_row(f, second=True)
        ttk.Label(f, text="꼭짓점 1~4 = 매트 평면도에서 본 빈 공간(흰 선 안쪽)의 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래. "
                           "기준점(IE/IB)과 같은 자리라 보조선 교차점에 붙일 수 있다.", foreground="#555", wraplength=380).pack(anchor="w")

    def build_mode4(self):
        tk, ttk = self.tk, self.ttk
        f = self.m4 = ttk.LabelFrame(self.mode_box, text="보조선 (판정 계산과 상관없음)", padding=6)
        ttk.Label(f, text="빈 곳을 누르고 끌면 선이 생깁니다. 칸 선의 가장자리를 따라 선을 그어 두면, 두 선이 만나는 "
                          "노란 원에 기준점·꼭짓점을 정확히 붙일 수 있어 칸마다 같은 기준으로 찍게 됩니다.",
                  wraplength=380, foreground="#555", justify="left").pack(anchor="w")
        self.line_var = tk.StringVar()
        ttk.Label(f, textvariable=self.line_var, foreground="#0070c0").pack(anchor="w", pady=(4, 0))
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Button(rr, text="고른 선 복제 (D)", command=self.dup_line).pack(side="left")
        ttk.Button(rr, text="고른 선 지우기 (Del)", command=self.del_line).pack(side="left", padx=4)
        ttk.Button(rr, text="모두 지우기", command=self.clear_lines).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Label(rr, text="움직일 곳 (E)").pack(side="left")
        self.end_var = tk.StringVar(value="all")
        for txt, v in (("선 전체", "all"), ("끝점 1", "0"), ("끝점 2", "1")):
            ttk.Radiobutton(rr, text=txt, variable=self.end_var, value=v, command=self.pick_end).pack(side="left")
        self.add_step_row(f)
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        self.ext_var = tk.BooleanVar(value=True)
        self.snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="화면 끝까지 늘려 보기", variable=self.ext_var, command=self.sync_lines).pack(side="left")
        ttk.Checkbutton(rr, text="교차점에 붙이기 (SNAP)", variable=self.snap_var, command=self.sync_lines).pack(side="left", padx=8)

    def add_step_row(self, parent, second=False):
        ttk = self.ttk
        rr = ttk.Frame(parent)
        rr.pack(fill="x", pady=(2, 0))
        ttk.Label(rr, text="화살표 이동 폭 (, .)").pack(side="left")
        if not hasattr(self, "step_var"):
            self.step_var = self.tk.StringVar(value=str(NUDGE_STEPS[self.step_i]))
        cb = ttk.Combobox(rr, textvariable=self.step_var, values=[str(v) for v in NUDGE_STEPS], width=5, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.set_step(NUDGE_STEPS.index(float(self.step_var.get()))))
        ttk.Label(rr, text="픽셀 (Shift: 5배)").pack(side="left")

    # ---------------- 동작
    def set_mode(self, k):
        self.m.mode = k
        self.mode_var.set(k)
        for f in (self.m1, self.m2, self.m3, self.m4):
            f.pack_forget()
        {1: self.m1, 2: self.m2, 3: self.m3, 4: self.m4}[k].pack(fill="x")
        self.view.canvas.focus_set()

    def dup_line(self):
        if self.m.duplicate_line():
            self.msg = f"선 L{self.m.sel_line + 1}을(를) 복제했습니다. 화살표로 옮기세요."
            self.zoom_from = "key"
        else:
            self.msg = "복제할 선을 먼저 고르세요(④에서 선 클릭 또는 Tab)."
        self.view.canvas.focus_set()

    def del_line(self):
        n = self.m.sel_line
        if self.m.delete_line():
            self.msg = f"선 L{n + 1}을(를) 지웠습니다."
        self.view.canvas.focus_set()

    def clear_lines(self):
        if self.m.lines and self.mb.askyesno("보조선", f"보조선 {len(self.m.lines)}개를 모두 지울까요?"):
            self.m.lines, self.m.sel_line, self.m.sel_end = [], None, None
            self.m.saved = False
        self.view.canvas.focus_set()

    def pick_end(self):
        v = self.end_var.get()
        self.m.sel_end = None if v == "all" else int(v)
        self.zoom_from = "key"
        self.view.canvas.focus_set()

    def sync_lines(self):
        self.m.extend_lines = self.ext_var.get()
        self.m.snap = self.snap_var.get()
        self.view.canvas.focus_set()

    def set_step(self, i):
        self.step_i = max(0, min(len(NUDGE_STEPS) - 1, i))
        self.step_var.set(str(NUDGE_STEPS[self.step_i]))

    def sync_view(self):
        self.m.show_slots = self.show_slots_var.get()
        self.m.show_lines = self.show_lines_var.get()
        self.m.show_zone = self.show_zone_var.get()
        self.m.show_level = self.show_level_var.get()
        self.view.canvas.focus_set()

    def toggle_freeze(self, flip=False):
        if flip:
            self.freeze_var.set(not self.freeze_var.get())
        want = self.freeze_var.get()
        if want != (self.src.frozen is not None):
            self.src.toggle_freeze()
        self.msg = "화면을 멈췄습니다. 다시 Space를 누르면 움직입니다." if want else "화면이 다시 움직입니다."
        self.view.canvas.focus_set()

    def pick_kind(self):
        if self.m.sel_slot is not None:
            self.m.sel_ref = f"{self.m.sel_slot}-{self.kind_var.get()}"
            self.m.set_active(self.m.sel_ref)
        self.view.canvas.focus_set()

    def next_ref(self):
        self.m.suggest_next()
        self.view.canvas.focus_set()

    def delete_sel(self):
        m = self.m
        rid = m.active_ref if m.active_ref else (m.sel_ref if m.sel_ref and m.ref_index(m.sel_ref) is not None else None)
        if rid and m.delete_ref(rid):
            self.msg = f"{rid} 점을 지웠습니다."

    def reset_all(self):
        m = self.m
        if not (m.refs or m.edited or m.lines):
            return
        if self.mb.askyesno("새로 보정", f"카메라를 옮겼을 때 씁니다. 찍은 기준점 {len(m.refs)}개, 고친 칸 {len(m.edited)}개, "
                            f"보조선 {len(m.lines)}개를 모두 지우고 감시 칸을 자동 선택으로 되돌립니다. "
                            f"(저장하기 전까지 설정 파일은 그대로입니다) 계속할까요?"):
            m.reset_all()
            m.suggest_next()
            self.msg = "모두 지웠습니다. 새 카메라 위치에서 기준점을 다시 찍으세요."
        self.view.canvas.focus_set()

    def cycle_active(self, back=False):
        """Tab: 찍은 점을 차례로 키보드 대상으로 고른다."""
        m = self.m
        if not m.refs:
            return
        ids = [r["id"] for r in m.refs]
        i = ids.index(m.active_ref) if m.active_ref in ids else -1
        i = (i + (-1 if back else 1)) % len(ids)
        m.active_ref = m.sel_ref = ids[i]
        m.sel_slot = m.refs_all[ids[i]]["slot"]
        self.msg = f"키보드로 움직일 점: {ids[i]}"

    def auto_mon(self):
        self.m.monitored_auto = True
        self.m.changed()
        self.msg = "감시 칸을 자동 선택으로 되돌렸습니다."

    def reset_slot(self):
        if self.m.sel_slot in self.m.edited:
            del self.m.edited[self.m.sel_slot]
            self.m.changed()
            self.msg = f"{self.m.sel_slot}번 칸을 호모그래피 결과로 되돌렸습니다."

    def add_fov(self, d):
        self.m.hfov = round(self.m.hfov + d, 2)
        self.m.changed()

    def fit_fov(self):
        r = self.m.fit_fov()
        if r is None:
            self.msg = "화각 자동 추정은 기준점 6개 이상일 때 됩니다."
        else:
            self.msg = f"화각 {r[0]:.1f}도 → {r[1]:.1f}도 (재투영 {r[2]:.2f}px)"
            print(f"[보정] {self.msg}")

    def make_model(self, name, source, size):
        """카메라 하나의 보정 상태를 만들고 저장된 보정 파일을 불러온다. (ver2는 CalibModel을 바꿔 끼워 쓴다)"""
        m = CalibModel(name, source, self.cfg, self.table, self.params, size)
        m.load()
        m.recompute()
        if m.sel_ref is None:
            m.suggest_next()
        return m

    def change_contact_mode(self, event=None):
        from camera_contact import MODE_LABELS
        self.m.contact_mode = next(k for k, v in MODE_LABELS.items() if v == self.contact_var.get())
        self.m.changed()
        self.msg = "촬영 방식을 바꿨습니다. 저장(S)하면 이 카메라에만 적용됩니다."

    def title_text(self):
        return f"PARKIT 보정 도구 — {self.m.name}"

    def camera_list(self):
        return camera_names(self.cfg, os.path.dirname(self.m.path))

    def on_model_changed(self):
        """카메라를 바꿔 새 보정 상태가 들어왔을 때 (ver2가 넓힌다)."""

    def change_camera(self):
        """보정할 카메라(CAM-1, CAM-2 …)를 바꾼다: 그 카메라의 보정 파일과 저장된 웹캠 번호로 다시 연다."""
        name = self.name_var.get().strip()
        if not name or name == self.m.name:
            self.name_var.set(self.m.name)
            return
        if any(ch in name for ch in '\\/:*?"<>| '):
            self.mb.showwarning("카메라 이름", "카메라 이름은 CAM-3 처럼 빈칸과 \\ / : * ? \" < > | 없이 적으세요.")
            self.name_var.set(self.m.name)
            return
        if not self.m.saved:
            a = self.mb.askyesnocancel("카메라 바꾸기", f"{self.m.name}에 저장하지 않은 변경이 있습니다. 저장하고 바꿀까요?")
            if a is None:
                self.name_var.set(self.m.name)
                return
            if a:
                self.save()
        path, src, _ = saved_camera(name, lambda n: os.path.join(os.path.dirname(self.m.path), f"cam_{n}.json"))
        cur = parse_source(self.src.source) if self.src.kind == "live" else None
        if src is None:
            src = load_webcam_map().get(name, cur if cur is not None else 0)
        if cur != src or self.src.kind != "live":
            self.freeze_var.set(False)
            if not self.src.reopen(self.m.size, src):
                self.mb.showerror("영상 입력", f"{name}의 입력 {src}을 열지 못했습니다. 이전 카메라를 유지합니다.\n{self.src.msg}")
                self.name_var.set(self.m.name)
                return
            self.frame = self.first_frame()
        size = (self.frame.shape[1], self.frame.shape[0])
        self.m = self.make_model(name, src, size)
        self.cam_var.set(str(src))
        self.res_var.set(f"{size[0]}x{size[1]}")
        self.name_cb.configure(values=sorted(set(self.camera_list()) | {name}))
        self.root.title(self.title_text())
        self.set_mode(1)
        self.on_model_changed()
        self.msg = (f"{name}로 바꿨습니다: 웹캠 {src}번. " +
                    (f"보정 파일을 불러왔습니다({os.path.basename(path)})." if path else "새 카메라라 기준점을 새로 찍습니다."))
        for n in self.m.notes:
            print(f"[보정] {n}")
        print(f"[보정] {self.msg}")

    def change_input(self):
        """웹캠 번호와 해상도를 바꿔 다시 연다 (capture_winmac.py와 같이 0~4번에서 고른다)."""
        try:
            size = parse_res(self.res_var.get())
        except ValueError:
            self.mb.showwarning("해상도", "해상도는 1920x1080 처럼 적으세요.")
            return
        try:
            cam = validate_source(self.cam_var.get())
        except ValueError as exc:
            self.mb.showerror("입력 주소", str(exc))
            return
        same_cam = self.src.kind == "live" and parse_source(self.src.source) == cam
        if not same_cam and self.m.refs:
            if not self.mb.askyesno("웹캠 바꾸기", f"웹캠 {cam}번으로 바꿉니다. 다른 카메라라면 찍은 기준점이 맞지 않으니 "
                                    f"다시 찍어야 합니다. 계속할까요?"):
                return
        if self.m.refs and not aspect_same(self.m.size, size):
            if not self.mb.askyesno("해상도 바꾸기", f"화면 비율이 달라져서({self.m.size[0]}x{self.m.size[1]} → "
                                    f"{size[0]}x{size[1]}) 찍은 기준점과 고친 꼭짓점을 모두 지웁니다. 계속할까요?"):
                return
        self.freeze_var.set(False)
        if not self.src.reopen(size, None if same_cam else cam):
            self.mb.showerror("웹캠", self.src.msg)
            return
        if not same_cam:
            self.m.source = cam            # 저장하면 카메라 설정의 입력이 이 웹캠 번호가 된다
        f = self.first_frame()
        got = (f.shape[1], f.shape[0])
        self.frame = f
        kept = self.m.rescale(got)
        self.msg = (f"웹캠 {cam}번을 {got[0]}x{got[1]}로 열었습니다." + ("" if got == size else f" (요청 {size[0]}x{size[1]}는 지원 안 됨)")
                    + (" 기준점을 비율만큼 옮겼습니다." if kept and self.m.refs and got != size else ""))
        self.res_var.set(f"{got[0]}x{got[1]}")
        print(f"[보정] {self.msg}")

    def save(self):
        path = self.m.save(self.frame)
        if path is None:
            self.mb.showinfo("저장", "기준점이 4개 이상 있어야 저장할 수 있습니다.")
            return
        cam = self.m.cam
        e = cam["homography_err_cm"]
        self.msg = f"저장했습니다: {os.path.basename(self.m.path)}, {os.path.basename(path)}"
        print(f"[보정] 저장함: {self.m.path}")
        print(f"        확인 사진: {path}")
        print(f"        기준점 {len(self.m.refs)}개, 오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f}cm, 감시 칸 {ranges(cam['monitored'])}")
        if cam.get("pose"):
            p = cam["pose"]
            print(f"        렌즈 위치 {p['cam_pos_cm']} (x, y, 높이 cm), 아래로 {p['pitch_deg']}도, "
                  f"옆 기울기 {p['roll_deg']}도, 재투영 {p['reproj_px']['mean']}px")

    def quit(self):
        if not self.m.saved:
            a = self.mb.askyesnocancel("끝내기", "저장하지 않은 변경이 있습니다. 저장하고 끝낼까요?")
            if a is None:
                return
            if a:
                self.save()
        self.src.close()
        if getattr(self, "after_id", None):
            self.root.after_cancel(self.after_id)     # 닫은 뒤 화면 갱신이 남지 않게
        self.root.destroy()

    # ---------------- 마우스
    def src_xy(self, e):
        p = self.view.to_src(e.x, e.y)
        return None if p is None else np.array(p)

    def on_press(self, e):
        self.view.canvas.focus_set()
        p = self.src_xy(e)
        if p is None:
            return
        m = self.m
        if m.start_drag(p, self.view.scale):
            return
        if m.mode == 1:
            if m.sel_ref is None:
                self.msg = "먼저 오른쪽 평면도에서 칸을 고르고 점 종류를 고르세요."
                return
            rid = m.sel_ref
            q, snapped = m.snap_point(p, self.view.scale)
            m.place(q)
            self.msg = f"{rid} 점을 찍었습니다." + (" (보조선 교차점에 붙임)" if snapped else "")
            m.suggest_next()
        elif m.mode == 2:
            self.mon_begin("cam", True, p)
        elif m.mode == 3:
            no = m.slot_at_px(p)
            if no is not None:
                m.sel_slot = no

    def on_drag(self, e):
        p = self.src_xy(e)
        if p is not None:
            self.mouse = p
            if self.mon_op is not None:
                self.mon_move(p)
                return
            self.m.drag_to(p)

    def on_release(self, e):
        if self.mon_op is not None:
            self.mon_end()
            return
        kind = self.m.drag[0] if self.m.drag else None
        snapped = self.m.end_drag(self.view.scale)
        if snapped:
            self.msg = "보조선 교차점에 붙였습니다."
        elif kind == "newline" and self.m.sel_line is not None:
            self.msg = f"선 L{self.m.sel_line + 1}을(를) 그었습니다. 끝점을 끌거나 화살표로 맞추세요."

    def on_right(self, e):
        p = self.src_xy(e)
        if p is None:
            return
        if self.m.mode == 2:
            self.view.canvas.focus_set()
            self.mon_begin("cam", False, p)
            return
        if self.m.mode == 4:
            h = self.m.line_hit(p, self.view.scale)
            if h is not None and self.m.delete_line(h[0]):
                self.msg = f"선 L{h[0] + 1}을(를) 지웠습니다."
            return
        if self.m.mode != 1:
            return
        i = self.m.near_ref(p, self.view.scale)
        if i is not None:
            rid = self.m.refs[i]["id"]
            self.m.delete_ref(rid)
            self.msg = f"{rid} 점을 지웠습니다."

    def on_motion(self, e):
        self.mouse = self.src_xy(e)
        self.m.hover = self.mouse
        # 손이 살짝 닿은 정도(3픽셀 이하)로는 키보드 대상 확대에서 마우스 확대로 돌아가지 않는다
        last = getattr(self, "last_motion", None)
        moved = last is None or abs(e.x - last[0]) + abs(e.y - last[1]) > 3
        self.last_motion = (e.x, e.y)
        if self.mouse is not None and (moved or self.zoom_from == "mouse"):
            self.zoom_from = "mouse"

    def on_right_drag(self, e):
        p = self.src_xy(e)
        if p is not None:
            self.mouse = p
            if self.mon_op is not None:
                self.mon_move(p)

    def on_right_release(self, e):
        if self.mon_op is not None:
            self.mon_end()

    # ---------------- ② 감시 칸 끌기 (카메라 화면, 평면도). 왼쪽 = 켜기, 오른쪽 = 끄기
    def plan_xy(self, e):
        return np.array([e.x - self.plan_off[0], e.y - self.plan_off[1]], float)

    def plan_slot(self, q):
        """평면도 그림 좌표 → 칸 번호."""
        return self.table.slot_at(self.plan.to_mat(q))

    def mon_begin(self, where, on, p):
        """where: "cam"(보정 해상도 픽셀) 또는 "plan"(평면도 그림 픽셀)."""
        how = self.drag_how_var.get()
        self.mon_op = {"where": where, "on": on, "how": how, "p0": np.asarray(p, float), "last": np.asarray(p, float),
                       "moved": False, "done": []}
        if how == "paint":
            self.mon_apply(self.mon_slot_at(p))
        self.mon_show()

    def mon_slot_at(self, p):
        return self.m.slot_at_px(p) if self.mon_op["where"] == "cam" else self.plan_slot(p)

    def mon_apply(self, no):
        op = self.mon_op
        if no is not None and self.m.set_monitor(no, op["on"]):
            op["done"].append(no)

    def mon_move(self, p):
        op = self.mon_op
        p = np.asarray(p, float)
        # 화면 픽셀로 4픽셀 넘게 움직여야 끈 것으로 본다(살짝 떨린 클릭은 클릭)
        s = self.view.scale if op["where"] == "cam" else 1.0
        if np.linalg.norm(p - op["p0"]) * s > 4:
            op["moved"] = True
        if op["how"] == "paint":
            # 빠르게 끌어도 칸을 건너뛰지 않게 지난 점 → 지금 점 사이를 화면 3픽셀 간격으로 본다
            n = max(1, int(np.linalg.norm(p - op["last"]) * s / 3))
            for t in np.linspace(0, 1, n + 1)[1:]:
                self.mon_apply(self.mon_slot_at(op["last"] + (p - op["last"]) * t))
        op["last"] = p
        self.mon_show()

    def mon_end(self):
        op, m = self.mon_op, self.m
        if op["how"] == "rect":
            if op["moved"]:
                for no in self.mon_rect_slots(op["p0"], op["last"]):
                    self.mon_apply(no)
            else:
                self.mon_apply(self.mon_slot_at(op["p0"]))
        self.mon_op = None
        m.sel_rect = None
        done = sorted(op["done"])
        if done:
            m.changed()                                    # 다 고른 뒤 한 번만 다시 계산한다
            self.msg = f"감시 {'켬' if op['on'] else '끔'}: {ranges(done)} ({len(done)}칸)"
        else:
            self.msg = f"바뀐 칸이 없습니다 (이미 감시 {'켬' if op['on'] else '끔'})."

    def mon_rect_slots(self, a, b):
        """네모(a, b 두 모서리) 안에 가운데가 든 칸들."""
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        out = []
        if self.mon_op["where"] == "cam":
            g = self.m.geom
            for no, fl in ({} if g is None else g.floor_px).items():
                c = fl.mean(axis=0)
                if (lo <= c).all() and (c <= hi).all():
                    out.append(no)
        else:
            for sl in self.table:
                c = self.plan.to_px(sl.center)
                if (lo <= c).all() and (c <= hi).all():
                    out.append(sl.no)
        return out

    def mon_show(self):
        op = self.mon_op
        self.m.sel_rect = (op["p0"], op["last"], op["on"]) if op["how"] == "rect" and op["where"] == "cam" else None

    def on_plan_press(self, e, left):
        self.view.canvas.focus_set()
        if self.m.mode == 2:
            self.mon_begin("plan", left, self.plan_xy(e))
        elif left:
            self.on_plan_click(e)

    def on_plan_drag(self, e):
        if self.mon_op is not None:
            self.mon_move(self.plan_xy(e))

    def on_plan_release(self, e):
        if self.mon_op is not None:
            self.mon_end()

    # ---------------- 평면도 각도 (docs/41)
    def make_plan(self, angle):
        """0°이면 예전과 같은 크기·배율, 돌리면 같은 칸 크기 안에 맞춘다."""
        if angle % 360 == 0:
            return PlanView(self.table, px_per_cm=4.6, margin=6)
        return PlanView(self.table, margin=6, angle=angle, fit=self.plan_box)

    def cam_view_angle(self, name=""):
        pose = (self.m.cam or {}).get("pose")
        if not pose:
            return None
        return camera_view_angle(pose["cam_pos_cm"][:2], pose.get("yaw_deg"), self.table.mat_size)

    def on_plan_click(self, e):
        no = self.plan_slot(self.plan_xy(e))
        if no is None:
            return
        if self.m.mode == 2:
            self.m.toggle_monitor(no)
            self.msg = f"{no}번 칸 감시 {'켬' if no in self.m.monitored else '끔'}"
        else:
            self.m.sel_slot = no
            if self.m.mode == 1:
                code = self.kind_var.get() or REF_KINDS[0][0]
                self.m.sel_ref = f"{no}-{code}"
                self.m.set_active(self.m.sel_ref)
        self.view.canvas.focus_set()

    def on_ref_list(self, e):
        sel = self.ref_list.curselection()
        if sel and sel[0] < len(self.m.refs):
            rid = self.m.refs[sel[0]]["id"]
            self.m.sel_ref = rid
            self.m.set_active(rid)
            self.m.sel_slot = self.m.refs_all[rid]["slot"]
            self.zoom_from = "key"
        self.view.canvas.focus_set()

    # ---------------- 키보드
    def on_key(self, e):
        if is_typing(e.widget):
            return
        k = key_of(e)
        m = self.m
        shift = bool(e.state & 0x1)
        moves = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1),
                 "j": (-1, 0), "l": (1, 0), "i": (0, -1), "k": (0, 1)}
        if k in moves:
            d = NUDGE_STEPS[self.step_i] * (5 if shift else 1)
            if m.nudge(moves[k][0] * d, moves[k][1] * d):
                self.zoom_from = "key"          # 확대 창을 옮기는 대상 둘레로
            else:
                self.msg = {3: "③에서 칸을 먼저 고르세요.",
                            4: "옮길 선이 없습니다. 선을 클릭하거나 Tab으로 고르세요."}.get(
                    m.mode, "옮길 점이 없습니다. 찍은 점을 클릭하거나, [찍은 점] 목록에서 고르거나, Tab으로 고르세요.")
            return "break"
        if k == "Tab":
            if m.mode == 3:
                m.sel_corner = (m.sel_corner + (-1 if shift else 1)) % 4
            elif m.mode == 4:
                m.cycle_line(back=shift)
            else:
                self.cycle_active(back=shift)
            self.zoom_from = "key"
            return "break"
        if m.mode == 4 and k in ("x", "Delete", "BackSpace"):
            self.del_line()
            return "break"
        if k in ("1", "2", "3", "4"):
            self.set_mode(int(k))
        elif k == "d" and m.mode == 4:
            self.dup_line()
        elif k == "e" and m.mode == 4:
            m.cycle_end()
            self.zoom_from = "key"
        elif k == "g":
            m.show_lines = not m.show_lines
            self.msg = "보조선 " + ("보임" if m.show_lines else "숨김 (G로 다시 보임)")
        elif k == "s":
            self.save()
        elif k in ("q", "Escape"):
            self.quit()
        elif k in ("?", "F1"):
            self.help.toggle()
        elif k in ("x", "Delete", "BackSpace"):
            self.delete_sel()
        elif k == "n":
            self.next_ref()
        elif k == "c":
            m.recompute()
            self.msg = "다시 계산했습니다."
        elif k == "a":
            self.auto_mon()
        elif k == "r" and m.mode == 3:
            self.reset_slot()
        elif k == "f":
            self.fit_fov()
        elif k in ("[", "]"):
            self.add_fov(-0.5 if k == "[" else 0.5)
        elif k == ",":
            self.set_step(self.step_i - 1)
        elif k == ".":
            self.set_step(self.step_i + 1)
        elif k == "o":
            self.show_slots_var.set(not self.show_slots_var.get())
            self.sync_view()
        elif k == "z":
            self.show_zone_var.set(not self.show_zone_var.get())
            self.sync_view()
        elif k == "h":
            self.show_level_var.set(not self.show_level_var.get())
            self.sync_view()
        elif k == "space":
            self.toggle_freeze(flip=True)
            return "break"

    # ---------------- 주기적으로 다시 그리기
    def tick(self):
        try:
            if isinstance(self.src.src, LiveSource):
                self.src.src.reopen_on_main()     # 맥: 끊긴 웹캠은 메인 스레드에서 다시 연다
            f = self.src.get()
            if f is not None:
                self.frame = f
            self.render()
            self.update_panel()
        finally:
            self.after_id = self.root.after(40, self.tick)

    def render(self):
        f = self.frame
        H, W = f.shape[:2]
        if (W, H) != self.m.size:
            return
        s = self.view.fit_scale(W, H)
        im = cv2.resize(f, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s != 1 else f.copy()
        self.m.draw(im, s)
        center, label = None, ""
        if self.zoom_from == "key" and self.m.focus_point() is not None:
            center, label = self.m.focus_point(), "KEY"
        elif self.mouse is not None:
            center, label = self.mouse, "MOUSE"
        if center is not None:
            self.draw_magnifier(im, f, s, center, label)
        self.view.show(im, s, (W, H))

    def draw_magnifier(self, im, frame, s, center, label):
        """확대 창: 원본 해상도에서 center 둘레를 ZOOM_Z배로. 보조선·기준점도 가늘게 겹친다."""
        r, z = ZOOM_R, ZOOM_Z
        H, W = frame.shape[:2]
        cxf, cyf = float(center[0]), float(center[1])
        x0, y0 = int(np.floor(cxf)) - r, int(np.floor(cyf)) - r
        big = np.zeros((2 * r * z, 2 * r * z, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + 2 * r), min(H, y0 + 2 * r)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        crop = cv2.resize(frame[sy0:sy1, sx0:sx1], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        big[(sy0 - y0) * z:(sy0 - y0) * z + crop.shape[0], (sx0 - x0) * z:(sx0 - x0) * z + crop.shape[1]] = crop
        self.m.draw_zoom(big, x0, y0, z)
        cx, cy = int(round((cxf - x0) * z)), int(round((cyf - y0) * z))
        cv2.line(big, (cx, 0), (cx, big.shape[0]), (0, 255, 255), 1)
        cv2.line(big, (0, cy), (big.shape[1], cy), (0, 255, 255), 1)
        text(big, f"x{z} {label} ({cxf:.1f}, {cyf:.1f})", (4, big.shape[0] - 6), 0.45, (0, 255, 255), 1)
        bh, bw = big.shape[:2]
        if bh > im.shape[0] - 16 or bw > im.shape[1] - 16:
            return
        mx = 8 if cxf * s > im.shape[1] / 2 else im.shape[1] - bw - 8
        im[8:8 + bh, mx:mx + bw] = big
        cv2.rectangle(im, (mx, 8), (mx + bw, 8 + bh), (255, 0, 255) if label == "KEY" else (0, 255, 255), 2)

    def update_panel(self):
        m = self.m
        self.guide_var.set(m.guide())
        if self.show_lines_var.get() != m.show_lines:
            self.show_lines_var.set(m.show_lines)
        from camera_contact import MODE_LABELS
        self.contact_var.set(MODE_LABELS[self.m.contact_mode])
        self.src_var.set(self.src.describe(self.frame) + ("  · 멈춤" if self.src.frozen is not None else ""))
        # 마우스 안내
        hint = ""
        if m.mode == 1:
            if m.drag:
                hint = "끄는 중: 마우스를 놓으면 다시 계산합니다."
            elif self.mouse is not None and m.near_ref(self.mouse, self.view.scale) is not None:
                hint = "누르고 끌기: 이 점 옮기기  ·  클릭: 키보드 대상으로 고르기  ·  오른쪽 클릭: 지우기"
            elif m.sel_ref:
                done = " (다시 찍으면 옮겨짐)" if m.ref_index(m.sel_ref) is not None else ""
                hint = f"왼쪽 클릭: [{m.sel_ref}] {m.sel_ref.split('-')[0]}번 칸 {ref_label(m.sel_ref.split('-', 1)[1])} 찍기{done}"
            else:
                hint = "오른쪽 평면도에서 칸을 고르세요."
            if m.active_ref and not m.drag:
                hint += f"   |   화살표·IJKL: [{m.active_ref}] 옮기기, Tab: 다른 점"
        elif m.mode == 2:
            how = "지나간 칸" if self.drag_how_var.get() == "paint" else "네모 안 칸"
            hint = f"왼쪽 클릭·끌기: 감시 켜기  ·  오른쪽: 끄기  ({how}, 초록 = 감시)"
        elif m.mode == 3:
            hint = ("노란 꼭짓점 끌기 또는 화살표: 꼭짓점 옮기기  ·  Tab: 다른 꼭짓점  ·  다른 칸 클릭: 고르기" if m.sel_slot
                    else "어긋난 칸을 클릭해 고르세요.")
        else:
            h = m.line_hit(self.mouse, self.view.scale) if self.mouse is not None else None
            if m.drag:
                hint = "끄는 중"
            elif h is not None:
                hint = (f"끌기: 선 L{h[0] + 1} " + ("전체 옮기기" if h[1] is None else f"끝점 {h[1] + 1} 옮기기")
                        + "  ·  오른쪽 클릭: 지우기")
            else:
                hint = "빈 곳을 누르고 끌기: 새 선 긋기"
            if m.sel_line is not None and not m.drag:
                tgt = "전체" if m.sel_end is None else f"끝점 {m.sel_end + 1}"
                hint += f"   |   화살표·IJKL: L{m.sel_line + 1} {tgt} 옮기기, E: 움직일 곳, Tab: 다른 선, D: 복제"
        pos = "" if self.mouse is None else f"   ({self.mouse[0]:.0f}, {self.mouse[1]:.0f})"
        self.hint_var.set(f"{hint}{pos}      이동 폭 {NUDGE_STEPS[self.step_i]}px   ·   도움말: ? 키")
        # 평면도
        st = {no: ("EMPTY" if no in m.monitored else None) for no in self.table.slots}
        pts = [(r["mat"], (60, 220, 60)) for r in m.refs]
        if m.sel_ref:
            pts.append((m.refs_all[m.sel_ref]["mat"], (0, 255, 255)))
        cams = []
        if m.cam and m.cam.get("pose"):
            p = m.cam["pose"]["cam_pos_cm"]
            cams = [(m.name, p[0], p[1])]
        a = self.angle_bar.poll()
        if a != self.plan.angle:
            self.plan = self.make_plan(a)
        pim = self.plan.render(st, pts, selected=m.sel_slot, cams=cams)
        from PIL import Image, ImageTk
        self.plan_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(pim, cv2.COLOR_BGR2RGB)))
        self.plan_view.delete("all")
        self.plan_off = (0, 0)
        self.plan_view.create_image(0, 0, anchor="nw", image=self.plan_img)
        op = self.mon_op
        if op is not None and op["where"] == "plan" and op["how"] == "rect":
            (x0, y0), (x1, y1) = op["p0"] + self.plan_off, op["last"] + self.plan_off
            self.plan_view.create_rectangle(x0, y0, x1, y1, outline="#3cdc3c" if op["on"] else "#e63c3c", width=2)
        # ① 기준점
        if m.sel_slot is not None:
            sl = self.table[m.sel_slot]
            self.slot_var.set(f"고른 칸: {sl.no}번 ({sl.row}번 줄, {'실측' if sl.measured else '사진 추정 ±0.3cm'})")
            placed = {r["id"]: r for r in m.refs}
            for code, b in self.kind_btns:
                rid = f"{sl.no}-{code}"
                mat = m.refs_all[rid]["mat"]
                e = m.err.get(rid) if m.err and rid in placed else None
                mark = "✔" if rid in placed else "  "
                b.configure(text=f"{mark} {code}  {ref_label(code)}  ({mat[0]:.2f}, {mat[1]:.2f})" +
                            (f"  오차 {e:.2f}cm" if e is not None else ""))
            if m.sel_ref and m.sel_ref.startswith(f"{sl.no}-"):
                self.kind_var.set(m.sel_ref.split("-", 1)[1])
        else:
            self.slot_var.set("평면도에서 칸을 고르세요.")
        self.active_var.set(f"키보드로 움직일 점: {m.active_ref}  (분홍 겹원 · 화살표/IJKL · Tab으로 바꿈)" if m.active_ref
                            else "키보드로 움직일 점: 없음 (점을 찍거나, 클릭하거나, 목록에서 고르세요)")
        rows = []
        for r in m.refs:
            e = m.err.get(r["id"]) if m.err else None
            rows.append(f"{r['id']:7s} ({r['px'][0]:7.1f},{r['px'][1]:7.1f}) " + ("  -  " if e is None else f"{e:.3f}cm")
                        + ("  ✗" if e is not None and e > GOOD_ERR else ""))
        if list(self.ref_list.get(0, "end")) != rows:
            self.ref_list.delete(0, "end")
            for i, t in enumerate(rows):
                self.ref_list.insert("end", t)
                e = m.err.get(m.refs[i]["id"]) if m.err else None
                if e is not None and e > GOOD_ERR:
                    self.ref_list.itemconfig(i, foreground="#c00000")
        ids = [r["id"] for r in m.refs]
        if m.active_ref in ids:
            i = ids.index(m.active_ref)
            if self.ref_list.curselection() != (i,):
                self.ref_list.selection_clear(0, "end")
                self.ref_list.selection_set(i)
                self.ref_list.see(i)
        # ② ③
        if m.lines:
            tgt = "" if m.sel_line is None else (" · 고른 선 L" + str(m.sel_line + 1) +
                                                  (" 전체" if m.sel_end is None else f" 끝점 {m.sel_end + 1}"))
            self.line_var.set(f"보조선 {len(m.lines)}개, 교차점 {len(m.intersections())}개{tgt}"
                              + ("" if m.show_lines else "  (숨김: G)"))
        else:
            self.line_var.set("보조선 없음")
        self.end_var.set("all" if m.sel_end is None else str(m.sel_end))
        self.mon_var.set(f"감시 칸 {len(m.monitored)}개{' (자동 선택)' if m.monitored_auto else ''}: {ranges(m.monitored)}")
        if m.sel_slot is not None:
            self.edit_var.set(f"고른 칸: {m.sel_slot}번 " + ("(고침, 보라색)" if m.sel_slot in m.edited else "(호모그래피 그대로)")
                              + f"   ·   고친 칸: {ranges(m.edited)}")
        else:
            self.edit_var.set(f"고친 칸: {ranges(m.edited)}")
        self.corner_var.set(m.sel_corner)
        # 계산 결과
        if m.cam:
            e = m.cam["homography_err_cm"]
            lines = [f"기준점 {len(m.refs)}개  오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f} cm (목표 < {GOOD_ERR})"]
            p = m.cam.get("pose")
            if p:
                c = p["cam_pos_cm"]
                lines.append(f"렌즈 위치 x {c[0]:.1f}  y {c[1]:.1f}  높이 {c[2]:.1f} cm")
                lines.append(f"아래로 {p['pitch_deg']:.1f}도  옆 기울기 {p['roll_deg']:.1f}도  재투영 {p['reproj_px']['mean']:.2f}px")
            self.res_lbl.configure(text="\n".join(lines))
        else:
            self.res_lbl.configure(text=f"기준점 {len(m.refs)}개 — 4개부터 계산합니다.")
        self.fov_var.set(f"{m.hfov:.1f}도")
        self.warn_lbl.configure(text="\n".join("· " + w for w in m.warnings()))
        self.msg_var.set(self.msg)


def main():
    ap = argparse.ArgumentParser(description="PARKIT 보정 도구")
    ap.add_argument("--cam", default=None, help="카메라 이름 (기본 CAM-1). config/cam_<이름>.json 에 저장")
    ap.add_argument("--source", default=None, help="웹캠 번호, 영상 파일, rtsp 주소 (없으면 저장된 값 또는 0)")
    ap.add_argument("--image", default=None, help="웹캠 대신 사진으로 보정")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--run", default=DEFAULT_RUN, help="실행 설정 (칸 표 경로, 판정 값, 첫 화각)")
    ap.add_argument("--no-picker", action="store_true", help="시작 창(웹캠 고르기) 없이 바로 CAM-1로")
    args = ap.parse_args()
    check_tk()
    cfg = load_run(args.run)
    table = SlotTable(cfg["slot_table"])
    params = Params(cfg, table)
    if not args.cam and args.source is None and not args.image and not args.no_picker:
        # 아무것도 주지 않으면 시작 창에서 웹캠을 보고 카메라를 고른다
        picked = CamPicker(cfg).run()
        if picked is None:
            print("[보정] 끝냈습니다.")
            return 0
        args.cam, args.source = picked
        print(f"[보정] 고른 카메라: {args.cam}, 웹캠 {args.source}번")
        time.sleep(0.3)                    # 미리보기로 열었다 닫은 웹캠이 풀릴 때까지 조금 기다린다
    if not args.cam:
        args.cam = "CAM-1"
        print("[보정] 카메라 이름을 주지 않아 CAM-1로 합니다. 다른 카메라는 --cam CAM-2 처럼 줍니다.")
    if args.source is None and not args.image:
        p = cam_config_path(args.cam)
        args.source = load_json(p).get("source", 0) if os.path.exists(p) else 0
        print(f"[보정] 입력: {args.source}")
    app = CalibApp(args, cfg, table, params)
    print(f"[보정] {app.src.msg}. 단축키는 창에서 ? 키로 봅니다.")
    app.root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
