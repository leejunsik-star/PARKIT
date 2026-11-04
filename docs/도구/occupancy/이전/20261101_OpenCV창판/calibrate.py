"""PARKIT 점유 판정 — 보정 도구 (25 문서 K1~K8, 22 문서 3.2·3.3절)

카메라마다 기준점을 찍어 호모그래피와 카메라 자세(solvePnP)를 구하고, 감시 칸을 고르고,
칸 꼭짓점을 손으로 고쳐 config\\cam_<이름>.json 에 저장한다. 확인용 이미지도 같이 저장한다.

사용법 (docs\\도구\\occupancy 에서):
    python calibrate.py --cam CAM-1 --source 0                 웹캠 0번, 1920x1080
    python calibrate.py --cam CAM-1 --source 0 --width 1280 --height 720
    python calibrate.py --cam CAM-1 --image 빈매트.jpg          사진으로 (source는 저장만)
    python calibrate.py --cam CAM-1 --source 녹화.mp4           영상 파일로
설정 파일이 이미 있으면 불러와서 이어서 고친다(K1).

창 안 단축키 (영문 입력 상태에서)
    1 기준점  2 감시 칸  3 꼭짓점 고치기       a 감시 칸 자동 선택   r (3번 모드) 고른 칸 초기화
    x 고른 기준점 지우기   n 다음 추천 기준점   c 다시 계산   f 화각 추정   [ ] 화각 -/+0.5도
    h 수평 확인선   z 판정 구역/칸 화면 영역 보기   space 화면 멈춤/풀기   s 저장   q 끝내기
마우스
    1번 모드: 왼쪽 클릭 = 고른 기준점을 그 자리에 찍기, 점 끌기 = 옮기기, 오른쪽 클릭 = 점 지우기
    2번 모드: 칸 클릭 = 감시 켜기/끄기     3번 모드: 칸 클릭 = 고르기, 꼭짓점 끌기 = 고치기
    오른쪽 평면도 클릭 = 칸 고르기(1번 모드는 그 칸의 기준점 목록이 나옴)
"""

import argparse, datetime, os, sys, time

import cv2
import numpy as np

from parkit_config import CONFIG_DIR, DEFAULT_RUN, Params, cam_config_path, load_json, load_run, save_json
from parkit_draw import FONT, PlanView, poly, read_image, save_image, text
from parkit_geom import (REF_KINDS, CamGeom, SlotTable, apply_h, aspect_same, auto_monitored,
                         build_camera_config, estimate_hfov, ref_points)
from parkit_video import FileSource, LiveSource, source_kind

WIN = "PARKIT calibrate"
PANEL_W = 430
PICK_PX = 12                  # 점·꼭짓점을 잡는 거리 (화면 픽셀)
GOOD_ERR = 0.5                # 25 문서 6절: 기준점 오차 0.5cm 안

HELP = [
    "MODE 1 REF  2 MONITOR  3 CORNER  | s SAVE  q QUIT",
    "1: Lclick=place picked ref, drag=move, Rclick=del",
    "   x del picked  n next suggested  c recalc",
    "2: click slot=on/off  a=auto select",
    "3: click slot=select, drag corner  r=reset slot",
    "f fit FOV  [ ] FOV-/+  h level line  z zones  spc freeze",
]
SUGGEST = "Use 6-8 refs, wide + mixed depth: 1-14 ent(19.7)/back(26.4), 28-41 back(31)/ent(37.6)"


class Frames:
    """사진, 웹캠/rtsp, 영상 파일에서 화면 한 장씩."""

    def __init__(self, args, size):
        self.still = None
        self.src = None
        self.frozen = None
        if args.image:
            self.still = read_image(args.image)
            if self.still is None:
                raise SystemExit(f"사진을 읽지 못함: {args.image}")
            self.msg = f"사진: {args.image}"
        elif source_kind(args.source) == "file":
            self.src = FileSource("cal", args.source, 10.0, loop=True)
            if not self.src.open():
                raise SystemExit(self.src.msg)
            self.src.start()
            self.msg = self.src.msg
            self.last = None
        else:
            self.src = LiveSource("cal", int(args.source) if source_kind(args.source) == "camera" else args.source,
                                  size[0], size[1])
            if not self.src.open():
                raise SystemExit(self.src.msg)
            self.src.start()
            self.msg = self.src.msg

    def get(self):
        if self.frozen is not None:
            return self.frozen
        if self.still is not None:
            return self.still
        if isinstance(self.src, FileSource):
            it = self.src.get(timeout=1.0)
            if it is not None:
                self.last = it[2]
            return self.last
        f, t, n = self.src.latest()
        return f

    def toggle_freeze(self):
        self.frozen = None if self.frozen is not None else (self.get().copy() if self.get() is not None else None)
        return self.frozen is not None

    def close(self):
        if self.src is not None:
            self.src.close()


class Calibrator:
    def __init__(self, args, cfg, table, params, frame0):
        self.args, self.cfg, self.table, self.params = args, cfg, table, params
        self.name = args.cam
        self.size = (frame0.shape[1], frame0.shape[0])
        self.ds = min(1.0, args.disp / self.size[0])
        self.disp = (int(round(self.size[0] * self.ds)), int(round(self.size[1] * self.ds)))
        self.refs_all = {r["id"]: r for r in ref_points(table)}
        self.refs = []                 # [{"id", "mat", "px"}]
        self.sel_ref = None
        self.sel_slot = None           # 1번 모드에서 기준점 목록을 볼 칸 / 3번 모드에서 고친 칸
        self.mode = 1
        self.monitored = set()
        self.monitored_auto = True
        self.edited = {}
        self.hfov = float(cfg["calib"].get("hfov_deg", 50.0))
        self.show_level = True
        self.show_zone = False
        self.drag = None
        self.mouse = None
        self.saved = True
        self.msg = ""
        self.cam = None
        self.geom = None
        self.err = None
        self.source = args.source if args.source is not None else 0
        self.path = cam_config_path(self.name)
        self.load()
        self.plan = PlanView(table, px_per_cm=(PANEL_W - 16) / table.mat_size[0], margin=8)
        self.canvas_h = max(self.disp[1], 880)
        self.y_plan = 8 + 16 * len(HELP) + 6
        self.y_kinds = self.y_plan + self.plan.size[1] + 6
        self.y_list = self.y_kinds + 18 * (len(REF_KINDS) + 1) + 4
        self.recompute()

    # ---------------------------------------------------------------- 불러오기 / 저장 (K1, K7, K8)
    def load(self):
        if not os.path.exists(self.path):
            print(f"[보정] 새 카메라 설정을 만듭니다: {self.path}")
            return
        d = load_json(self.path)
        cal = tuple(d.get("calib_size", self.size))
        if not aspect_same(cal, self.size):
            print(f"[보정] 저장된 보정 해상도 {cal}와 지금 화면 {self.size}의 비율이 달라 기준점을 새로 찍습니다.")
            return
        s = self.size[0] / cal[0]
        for r in d.get("ref_points", []):
            m = self.refs_all.get(r["id"], {}).get("mat", r["mat"])
            self.refs.append({"id": r["id"], "mat": list(m), "px": [r["px"][0] * s, r["px"][1] * s]})
        self.monitored = set(int(n) for n in d.get("monitored", []))
        self.monitored_auto = not self.monitored
        self.edited = {int(k): np.asarray(v, float) * s for k, v in (d.get("edited_corners") or {}).items()}
        self.hfov = float(d.get("intrinsics", {}).get("hfov_deg", self.hfov))
        if self.args.source is None and d.get("source") is not None:
            self.source = d["source"]
        print(f"[보정] 불러옴: {self.path} (기준점 {len(self.refs)}개, 감시 칸 {len(self.monitored)}개, "
              f"고친 칸 {len(self.edited)}개, 저장 {d.get('date', '?')})")

    def save(self, frame):
        if self.cam is None:
            self.msg = "need >= 4 refs"
            print("[보정] 기준점이 4개 이상 있어야 저장할 수 있습니다.")
            return
        cam = dict(self.cam)
        cam["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam["input_note"] = "source: 웹캠 번호, 영상 파일 경로, rtsp 주소"
        save_json(self.path, cam)
        check = os.path.join(CONFIG_DIR, f"cam_{self.name}_check.jpg")
        im = frame.copy()
        self.draw_overlay(im, 1.0, all_slots=True)
        e = cam["homography_err_cm"]
        text(im, f"{self.name} {cam['date']}  ref err mean {e['mean']:.2f} max {e['max']:.2f} cm  "
                 f"monitored {len(cam['monitored'])}", (10, 30), 0.8, (255, 255, 255), 2)
        save_image(check, im)
        self.saved = True
        self.msg = "saved"
        print(f"[보정] 저장함: {self.path}")
        print(f"        확인 사진: {check}")
        print(f"        기준점 {len(self.refs)}개, 오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f}cm, "
              f"감시 칸 {cam['monitored']}")
        if cam.get("pose"):
            p = cam["pose"]
            print(f"        렌즈 위치 {p['cam_pos_cm']} (x, y, 높이 cm), 아래로 {p['pitch_deg']}도, "
                  f"옆 기울기 {p['roll_deg']}도, 재투영 {p['reproj_px']['mean']}px")

    # ---------------------------------------------------------------- 계산 (K3)
    def recompute(self):
        self.cam = self.geom = self.err = None
        if len(self.refs) < 4:
            return
        try:
            mon = None if self.monitored_auto else sorted(self.monitored)
            cam = build_camera_config(self.name, self.source, self.size, self.refs, self.table, self.params,
                                      self.hfov, monitored=mon, edited=self.edited)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as e:
            self.msg = f"calc error: {e}"
            return
        if self.monitored_auto:
            cam["monitored"] = auto_monitored(self.table, np.asarray(cam["homography"]), self.size,
                                              self.cfg["calib"].get("auto_margin_px", 0))
            self.monitored = set(cam["monitored"])
        self.cam = cam
        self.err = {r["id"]: r["err_cm"] for r in cam["ref_points"]}
        self.geom = CamGeom(cam, self.table, self.size, self.params)

    def changed(self):
        self.saved = False
        self.recompute()

    def fit_fov(self):
        if len(self.refs) < 6:
            self.msg = "need >= 6 refs for FOV fit"
            print("[보정] 화각 추정은 기준점 6개 이상일 때 합니다.")
            return
        f, e = estimate_hfov([r["px"] for r in self.refs], [r["mat"] for r in self.refs], *self.size)
        print(f"[보정] 화각 추정: {self.hfov:.1f}도 → {f:.1f}도 (재투영 {e:.2f}px)")
        self.hfov = round(f, 2)
        self.changed()

    def suggest_next(self):
        """아직 안 찍은 실측 기준점 중 화면 안에 있고 찍은 점에서 먼 것."""
        placed = {r["id"] for r in self.refs}
        if self.geom is None:
            order = [i for i in self.refs_all.values() if i["measured"] and i["slot"] and i["kind"].startswith("I")]
            for r in order:
                if r["id"] not in placed:
                    self.sel_ref, self.sel_slot = r["id"], r["slot"]
                    return
            return
        best = None
        P = np.array([r["px"] for r in self.refs])
        for r in self.refs_all.values():
            if r["id"] in placed or not r["measured"] or r["slot"] is None or not r["kind"].startswith("I"):
                continue
            q = self.geom.to_px([r["mat"]])[0]
            if not (30 <= q[0] <= self.size[0] - 30 and 30 <= q[1] <= self.size[1] - 30):
                continue
            d = np.min(np.linalg.norm(P - q, axis=1))
            if best is None or d > best[0]:
                best = (d, r)
        if best:
            self.sel_ref, self.sel_slot = best[1]["id"], best[1]["slot"]

    # ---------------------------------------------------------------- 마우스
    def on_mouse(self, ev, x, y, flags, _):
        if x >= self.disp[0]:
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.click_panel(x - self.disp[0], y)
            self.mouse = None
            return
        if y >= self.disp[1]:
            return
        p = np.array([x / self.ds, y / self.ds])
        self.mouse = p
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.press(p)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.move(p)
        elif ev == cv2.EVENT_LBUTTONUP:
            if self.drag is not None:
                self.drag = None
                self.changed()
        elif ev == cv2.EVENT_RBUTTONDOWN and self.mode == 1:
            i = self.near_ref(p)
            if i is not None:
                rid = self.refs.pop(i)["id"]
                self.msg = f"deleted {rid}"
                self.changed()

    def near_ref(self, p):
        best = None
        for i, r in enumerate(self.refs):
            d = np.linalg.norm(np.array(r["px"]) - p) * self.ds
            if d <= PICK_PX and (best is None or d < best[1]):
                best = (i, d)
        return None if best is None else best[0]

    def slot_at_px(self, p):
        if self.geom is None:
            return None
        hits = []
        for no, fl in self.geom.floor_px.items():
            if cv2.pointPolygonTest(fl.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0:
                hits.append(no)
        return hits[0] if hits else None

    def press(self, p):
        if self.mode == 1:
            i = self.near_ref(p)
            if i is not None:
                self.drag = ("ref", i)
                self.sel_ref = self.refs[i]["id"]
                return
            if self.sel_ref is None:
                self.msg = "pick a ref in the panel first"
                return
            m = self.refs_all[self.sel_ref]["mat"]
            for r in self.refs:
                if r["id"] == self.sel_ref:
                    r["px"] = [float(p[0]), float(p[1])]
                    break
            else:
                self.refs.append({"id": self.sel_ref, "mat": list(m), "px": [float(p[0]), float(p[1])]})
            self.msg = f"placed {self.sel_ref}"
            self.changed()
            self.suggest_next()
        elif self.mode == 2:
            no = self.slot_at_px(p)
            if no is not None:
                self.toggle_monitor(no)
        elif self.mode == 3:
            if self.sel_slot is not None and self.geom is not None and self.sel_slot in self.geom.floor_px:
                fl = self.geom.floor_px[self.sel_slot]
                d = np.linalg.norm(fl - p, axis=1) * self.ds
                k = int(np.argmin(d))
                if d[k] <= PICK_PX:
                    if self.sel_slot not in self.edited:
                        self.edited[self.sel_slot] = fl.copy()
                    self.drag = ("corner", self.sel_slot, k)
                    return
            no = self.slot_at_px(p)
            if no is not None:
                self.sel_slot = no

    def move(self, p):
        if self.drag[0] == "ref":
            self.refs[self.drag[1]]["px"] = [float(p[0]), float(p[1])]
        else:
            _, no, k = self.drag
            self.edited[no][k] = p
        self.changed()          # 끌면 바로 다시 계산한다(K4)

    def toggle_monitor(self, no):
        self.monitored_auto = False
        if no in self.monitored:
            self.monitored.discard(no)
        else:
            self.monitored.add(no)
        self.msg = f"slot {no} {'ON' if no in self.monitored else 'OFF'}"
        self.changed()

    def click_panel(self, x, y):
        if self.y_plan <= y < self.y_plan + self.plan.size[1]:
            no = self.table.slot_at(self.plan.to_mat((x, y - self.y_plan)))
            if no is None:
                return
            if self.mode == 2:
                self.toggle_monitor(no)
            else:
                self.sel_slot = no
            return
        if self.y_kinds + 18 <= y < self.y_kinds + 18 * (len(REF_KINDS) + 1) and self.sel_slot is not None:
            k = (y - self.y_kinds - 18) // 18
            if 0 <= k < len(REF_KINDS):
                self.sel_ref = f"{self.sel_slot}-{REF_KINDS[k][0]}"
                self.mode = 1
            return
        if y >= self.y_list + 18:
            k = (y - self.y_list - 18) // 16
            if 0 <= k < len(self.refs):
                self.sel_ref = self.refs[k]["id"]
                self.sel_slot = self.refs_all.get(self.sel_ref, {}).get("slot")

    # ---------------------------------------------------------------- 그리기
    def draw_overlay(self, im, s, all_slots=False):
        g = self.geom
        if g is not None:
            for no in sorted(g.floor_px):
                mon = no in self.monitored
                if not mon and not all_slots and self.mode == 1:
                    c, th = (140, 140, 140), 1
                else:
                    c, th = ((60, 220, 60), 2) if mon else ((140, 140, 140), 1)
                if no in self.edited:
                    c = (230, 60, 230) if mon else (170, 90, 170)
                if self.mode == 3 and no == self.sel_slot:
                    c, th = (0, 255, 255), 2
                poly(im, g.inner_px[no], c, th, s)
                poly(im, g.floor_px[no], tuple(int(v * 0.6) for v in c), 1, s)
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
            if self.mode == 3 and self.sel_slot in g.floor_px:
                for q in g.floor_px[self.sel_slot]:
                    cv2.circle(im, tuple((q * s).astype(int)), 6, (0, 255, 255), 2)
            if self.show_level:
                self.draw_level(im, s)
            # 고른 기준점이 있어야 할 곳 (계산된 호모그래피로)
            if self.sel_ref and self.mode == 1 and self.sel_ref not in {r["id"] for r in self.refs}:
                q = g.to_px([self.refs_all[self.sel_ref]["mat"]])[0] * s
                cv2.drawMarker(im, tuple(q.astype(int)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 1)
        for r in self.refs:
            q = (np.array(r["px"]) * s).astype(int)
            e = self.err.get(r["id"]) if self.err else None
            c = (0, 255, 255) if r["id"] == self.sel_ref else ((60, 220, 60) if e is None or e <= GOOD_ERR else (0, 0, 255))
            cv2.circle(im, tuple(q), 5, c, 2)
            cv2.drawMarker(im, tuple(q), c, cv2.MARKER_CROSS, 18, 1)
            lab = r["id"] + (f" {e:.2f}" if e is not None else "")
            text(im, lab, (q[0] + 7, q[1] - 7), 0.42 if s < 1 else 0.6, c, 1)

    def draw_level(self, im, s):
        """수평 확인선(26 문서 4절): 화면의 수평선과, 매트의 가로줄(1~14번 입구 y 19.7)을 화면에 그린 선."""
        g = self.geom
        W, H = self.size
        for y_mat in sorted({sl.entrance_y for sl in self.table if sl.measured}):
            xs = np.linspace(0, self.table.mat_size[0], 40)
            pts = g.to_px(np.c_[xs, np.full_like(xs, y_mat)])
            ok = (pts[:, 0] > -W) & (pts[:, 0] < 2 * W) & (pts[:, 1] > -H) & (pts[:, 1] < 2 * H)
            pts = pts[ok]
            if len(pts) < 2:
                continue
            poly(im, pts, (255, 200, 0), 1, s, closed=False)
            mid = pts[np.argmin(np.abs(pts[:, 0] - W / 2))]
            cv2.line(im, (0, int(mid[1] * s)), (int(W * s), int(mid[1] * s)), (0, 200, 255), 1, cv2.LINE_AA)

    def render(self, frame):
        im = cv2.resize(frame, self.disp, interpolation=cv2.INTER_AREA) if self.ds != 1 else frame.copy()
        self.draw_overlay(im, self.ds)
        if self.mouse is not None:
            self.draw_magnifier(im, frame)
        canvas = np.full((self.canvas_h, self.disp[0] + PANEL_W, 3), 30, np.uint8)
        canvas[:self.disp[1], :self.disp[0]] = im
        self.draw_panel(canvas[:, self.disp[0]:])
        return canvas

    def draw_magnifier(self, im, frame):
        """마우스 둘레 4배 확대 (원본 해상도에서)."""
        r, z = 30, 4
        x, y = int(self.mouse[0]), int(self.mouse[1])
        H, W = frame.shape[:2]
        x0, y0 = max(0, x - r), max(0, y - r)
        crop = frame[y0:min(H, y + r), x0:min(W, x + r)]
        if crop.size == 0:
            return
        big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        cx, cy = (x - x0) * z, (y - y0) * z
        cv2.line(big, (cx, 0), (cx, big.shape[0]), (0, 255, 255), 1)
        cv2.line(big, (0, cy), (big.shape[1], cy), (0, 255, 255), 1)
        bh, bw = big.shape[:2]
        mx = 8 if self.mouse[0] * self.ds > self.disp[0] / 2 else self.disp[0] - bw - 8
        my = 8
        im[my:my + bh, mx:mx + bw] = big
        cv2.rectangle(im, (mx, my), (mx + bw, my + bh), (0, 255, 255), 1)
        text(im, f"({x},{y})", (mx + 4, my + bh - 6), 0.4)

    def draw_panel(self, pn):
        y = 8
        for i, s in enumerate(HELP):
            text(pn, s, (6, y + 10), 0.4, (255, 255, 255) if i == 0 else (200, 200, 200), 1, None)
            y += 16
        mode_name = {1: "1 REF POINTS", 2: "2 MONITOR SLOTS", 3: "3 EDIT CORNERS"}[self.mode]
        text(pn, mode_name, (PANEL_W - 150, 18), 0.45, (0, 255, 255), 1)
        # 평면도: 감시 칸 초록, 찍은 기준점, 고른 기준점
        st = {no: ("EMPTY" if no in self.monitored else None) for no in self.table.slots}
        pts = [(r["mat"], (60, 220, 60)) for r in self.refs]
        if self.sel_ref:
            pts.append((self.refs_all[self.sel_ref]["mat"], (0, 255, 255)))
        cams = []
        if self.cam and self.cam.get("pose"):
            p = self.cam["pose"]["cam_pos_cm"]
            cams = [(self.name, p[0], p[1])]
        im = self.plan.render(st, pts, selected=self.sel_slot, cams=cams)
        pn[self.y_plan:self.y_plan + im.shape[0], :im.shape[1]] = im
        # 고른 칸의 기준점 목록
        y = self.y_kinds
        if self.sel_slot is not None:
            sl = self.table[self.sel_slot]
            text(pn, f"slot {sl.no} row {sl.row} ({'measured' if sl.measured else 'estimated +-0.3'})",
                 (6, y + 12), 0.42, (255, 255, 255) if sl.measured else (0, 160, 255), 1, None)
            placed = {r["id"]: r for r in self.refs}
            for k, (code, _, which, end, side) in enumerate(REF_KINDS):
                rid = f"{sl.no}-{code}"
                m = self.refs_all[rid]["mat"]
                yy = y + 18 * (k + 1)
                if rid == self.sel_ref:
                    cv2.rectangle(pn, (2, yy), (PANEL_W - 2, yy + 17), (70, 70, 0), -1)
                e = self.err.get(rid) if self.err and rid in placed else None
                mark = "*" if rid in placed else " "
                s = f"{mark} {code:4s} {which:5s} {end:4s} x{'-+'[side]}  ({m[0]:.2f}, {m[1]:.2f})"
                if e is not None:
                    s += f"  {e:.2f}cm"
                text(pn, s, (6, yy + 13), 0.4, (255, 255, 255), 1, None)
        else:
            text(pn, "click a slot on the plan to list its ref points", (6, y + 12), 0.4, (200, 200, 200), 1, None)
        # 찍은 기준점과 오차
        y = self.y_list
        text(pn, f"refs {len(self.refs)}  (n = next suggestion)", (6, y + 12), 0.42, (255, 255, 255), 1, None)
        for k, r in enumerate(self.refs):
            yy = y + 18 + 16 * k
            if yy + 16 > self.canvas_h - 84:
                text(pn, f"... +{len(self.refs) - k} more", (6, yy + 12), 0.4, (200, 200, 200), 1, None)
                break
            e = self.err.get(r["id"]) if self.err else None
            c = (60, 220, 60) if e is None or e <= GOOD_ERR else (0, 0, 255)
            if r["id"] == self.sel_ref:
                c = (0, 255, 255)
            text(pn, f"{r['id']:8s} px ({r['px'][0]:7.1f},{r['px'][1]:7.1f})  err " +
                 ("-" if e is None else f"{e:.3f} cm"), (6, yy + 12), 0.4, c, 1, None)
        # 계산 결과
        y = self.canvas_h - 66
        if self.cam:
            e = self.cam["homography_err_cm"]
            text(pn, f"H err mean {e['mean']:.3f} max {e['max']:.3f} cm  (goal < {GOOD_ERR})", (6, y), 0.42,
                 (60, 220, 60) if e["max"] <= GOOD_ERR else (0, 0, 255), 1, None)
            p = self.cam.get("pose")
            if p:
                text(pn, f"cam x {p['cam_pos_cm'][0]:.1f} y {p['cam_pos_cm'][1]:.1f} h {p['cam_pos_cm'][2]:.1f} "
                         f"pitch {p['pitch_deg']:.1f} roll {p['roll_deg']:.1f}", (6, y + 16), 0.42, (255, 255, 255), 1, None)
                text(pn, f"FOV {self.hfov:.1f} deg  reproj {p['reproj_px']['mean']:.2f}px  "
                         f"monitored {len(self.monitored)}{' (auto)' if self.monitored_auto else ''}",
                     (6, y + 32), 0.42, (255, 255, 255), 1, None)
        else:
            text(pn, SUGGEST[:60], (6, y), 0.38, (0, 200, 255), 1, None)
            text(pn, SUGGEST[60:], (6, y + 16), 0.38, (0, 200, 255), 1, None)
            text(pn, f"FOV {self.hfov:.1f} deg  - need >= 4 refs", (6, y + 32), 0.42, (255, 255, 255), 1, None)
        text(pn, ("" if self.saved else "[unsaved] ") + self.msg, (6, y + 50), 0.42,
             (0, 200, 255) if not self.saved else (200, 200, 200), 1, None)


def main():
    ap = argparse.ArgumentParser(description="PARKIT 보정 도구")
    ap.add_argument("--cam", required=True, help="카메라 이름 (예: CAM-1). config\\cam_<이름>.json 에 저장")
    ap.add_argument("--source", default=None, help="웹캠 번호, 영상 파일, rtsp 주소 (없으면 저장된 값 또는 0)")
    ap.add_argument("--image", default=None, help="웹캠 대신 사진으로 보정")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--run", default=DEFAULT_RUN, help="실행 설정 (칸 표 경로, 판정 값, 첫 화각)")
    ap.add_argument("--disp", type=int, default=1280, help="창에 보일 카메라 화면 폭")
    args = ap.parse_args()

    cfg = load_run(args.run)
    table = SlotTable(cfg["slot_table"])
    params = Params(cfg, table)
    if args.source is None and not args.image:
        p = cam_config_path(args.cam)
        args.source = load_json(p).get("source", 0) if os.path.exists(p) else 0
        print(f"[보정] 입력: {args.source}")
    frames = Frames(args, (args.width, args.height))
    print(f"[보정] {frames.msg}")
    f0 = None
    t0 = time.time()
    while f0 is None and time.time() - t0 < 10:
        f0 = frames.get()
        if f0 is None:
            time.sleep(0.1)
    if f0 is None:
        frames.close()
        raise SystemExit("화면을 받지 못했습니다.")
    if not args.image and (f0.shape[1], f0.shape[0]) != (args.width, args.height):
        print(f"[보정] 요청한 해상도 {args.width}x{args.height}가 아니라 {f0.shape[1]}x{f0.shape[0]}로 열렸습니다. "
              f"이 해상도로 보정합니다.")
    cal = Calibrator(args, cfg, table, params, f0)
    if cal.sel_ref is None:
        cal.suggest_next()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, cal.on_mouse)
    print("[보정] 창에서 단축키를 쓰려면 한/영 키로 영문 입력 상태로 두세요. q로 끝냅니다.")
    quit_warn = False
    frame = f0
    while True:
        f = frames.get()
        if f is not None:
            frame = f
        cv2.imshow(WIN, cal.render(frame))
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        ch = chr(k).lower() if k < 128 else ""
        if ch in "123":
            cal.mode = int(ch)
            cal.msg = f"mode {ch}"
        elif ch == "s":
            cal.save(frame)
        elif ch == "q" or k == 27:
            if not cal.saved and not quit_warn:
                cal.msg = "unsaved! press q again to quit without saving"
                print("[보정] 저장하지 않은 변경이 있습니다. 그래도 끝내려면 q를 한 번 더 누르세요.")
                quit_warn = True
                continue
            break
        elif ch == "x" and cal.sel_ref:
            n = len(cal.refs)
            cal.refs = [r for r in cal.refs if r["id"] != cal.sel_ref]
            if len(cal.refs) != n:
                cal.msg = f"deleted {cal.sel_ref}"
                cal.changed()
        elif ch == "n":
            cal.suggest_next()
        elif ch == "c":
            cal.recompute()
            cal.msg = "recalculated"
        elif ch == "a":
            cal.monitored_auto = True
            cal.changed()
            cal.msg = "auto monitored"
        elif ch == "r" and cal.mode == 3 and cal.sel_slot in cal.edited:
            del cal.edited[cal.sel_slot]
            cal.msg = f"slot {cal.sel_slot} reset"
            cal.changed()
        elif ch == "f":
            cal.fit_fov()
        elif ch in "[]":
            cal.hfov += -0.5 if ch == "[" else 0.5
            cal.changed()
        elif ch == "h":
            cal.show_level = not cal.show_level
        elif ch == "z":
            cal.show_zone = not cal.show_zone
        elif ch == " ":
            cal.msg = "frozen" if frames.toggle_freeze() else "live"
        quit_warn = False
    frames.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
