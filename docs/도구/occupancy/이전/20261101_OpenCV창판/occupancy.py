"""PARKIT 점유 판정 프로그램 (25 문서 P1~P11, D1~D6, L1~L4)

카메라마다 M8로 차를 찾고(P1), 바닥 접점으로 칸을 정하고(P2~P5, parkit_judge),
시간 거르기와 카메라 합치기(P6~P10, parkit_state)를 거쳐 칸 상태를 화면과 기록 파일에 낸다.
카메라 수와 감시 칸은 설정 파일로만 정한다(config\\run.json 의 cameras, config\\cam_*.json 의 monitored).

사용법 (docs\\도구\\occupancy 에서):
    python occupancy.py                                 config\\run.json 대로
    python occupancy.py --preset demo                   시연용 짧은 시간 값(1초/2초)
    python occupancy.py --source CAM-1=녹화.mp4          CAM-1 입력만 영상 파일로 바꿔서
    python occupancy.py --run config\\run_2대.json       다른 실행 설정
    python occupancy.py --speed 0                       영상 파일을 기다리지 않고 최대한 빨리 (결과는 같다)
    python occupancy.py --no-window --speed 0           창 없이 (시험용)

창 안 단축키 (영문 입력 상태에서)
    s 화면 저장(L3)  r 녹화 켜기/끄기(L4)  l 프레임 기록 켜기/끄기(L2)  v 카메라 화면 바꾸기(D5)
    d 시연 모드(D6)  o 판정 구역/칸 화면 영역 보기  p 일시정지(영상 파일)  q 끝내기
    평면도나 카메라 화면의 칸 클릭 = 칸 정보(D3)

시간: 입력이 모두 영상 파일이면 영상 시각을 쓴다. 그래서 같은 파일은 PC 속도와 상관없이 같은 결과가 나온다(C2).
웹캠이나 rtsp가 하나라도 있으면 실제 시각을 쓴다.
"""

import argparse, csv, datetime, json, os, sys, time

import cv2
import numpy as np

from parkit_config import DEFAULT_RUN, Params, load_json, load_run
from parkit_draw import COLORS, SHORT, PlanView, draw_boxes, draw_slots, save_image, text
from parkit_geom import CamGeom, ResolutionMismatch, SlotTable
from parkit_judge import judge_frame
from parkit_state import DYNAMIC_BLIND, EMPTY, OCCUPIED, STATIC_BLIND, Tracker
from parkit_video import FileSource, make_source, source_kind

WIN = "PARKIT occupancy"


# ---------------------------------------------------------------- 탐지기 (P1)

class ModelDetector:
    """모델 하나를 모든 카메라가 같이 쓴다(P11)."""

    def __init__(self, path, conf, imgsz):
        from ultralytics import YOLO
        if not os.path.exists(path):
            raise SystemExit(f"모델 파일이 없습니다: {path}")
        self.model = YOLO(path)
        self.conf, self.imgsz = float(conf), int(imgsz)
        self.names = self.model.names
        print(f"[판정] 모델: {path} (클래스 {self.names}), 신뢰도 {self.conf}, imgsz {self.imgsz}")

    def __call__(self, cam, frame, idx):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        out = []
        for b, c, k in zip(xyxy, cf, cls):
            if str(self.names.get(k, k)).lower() == "car" or len(self.names) == 1:
                out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        return out


class GTDetector:
    """시험용: 영상 옆의 <영상>.gt.json 에 적힌 상자를 그대로 낸다(test_pipeline.py). 실제 시험에는 쓰지 않는다."""

    def __init__(self, cams):
        self.gt = {}
        for c in cams:
            p = str(c.src) + ".gt.json"
            if not os.path.exists(p):
                raise SystemExit(f"--detector gt 에는 {p} 가 있어야 합니다.")
            self.gt[c.name] = load_json(p)["frames"]

    def __call__(self, cam, frame, idx):
        return [tuple(b) for b in self.gt[cam.name].get(str(idx), [])]


# ---------------------------------------------------------------- 카메라 하나

class Cam:
    def __init__(self, entry, cfg, table, params):
        self.cfg_path = entry["config"]
        if not os.path.exists(self.cfg_path):
            raise SystemExit(f"카메라 설정이 없습니다: {self.cfg_path}  (calibrate.py로 먼저 보정하세요)")
        self.cal = load_json(self.cfg_path)
        self.name = entry.get("name") or self.cal.get("name") or os.path.splitext(os.path.basename(self.cfg_path))[0]
        self.src = entry.get("source") if entry.get("source") is not None else self.cal.get("source", 0)
        self.req = (int(entry.get("width", 1920)), int(entry.get("height", 1080)))
        self.monitored = sorted(int(n) for n in self.cal.get("monitored", []))
        self.table, self.params = table, params
        self.source = make_source(self.name, self.src, self.req[0], self.req[1], cfg["analyze_fps"], entry.get("loop", False))
        self.state = "offline"          # ok / offline / stopped
        self.geom = None
        self.size = None
        self.last_t = None              # 마지막 프레임 받은 시각(판정 시계)
        self.last_n = 0
        self.frame = None
        self.idx = 0
        self.res = None
        self.ms = 0.0
        self.ended = False
        self.note = ""

    def open(self, t0):
        ok = self.source.open()
        print(f"[{self.name}] {self.source.msg}")
        self.last_t = t0
        if ok:
            self.source.start()
            self.size = self.source.size
            self.setup_geom()
        elif self.source.kind == "file":
            self.state = "stopped"
            self.note = "open failed"
        else:
            self.source.start()          # 웹캠/rtsp는 스레드가 계속 다시 열어 본다

    def setup_geom(self):
        """C6: 실행 해상도가 보정과 비율이 같으면 좌표를 바꿔 쓰고, 다르면 멈춘다."""
        if self.geom is not None or self.size is None:
            return
        if tuple(self.size) != self.req and self.source.kind != "file":
            print(f"[{self.name}] 요청 해상도 {self.req[0]}x{self.req[1]}가 아니라 {self.size[0]}x{self.size[1]}로 열렸습니다(C1).")
        try:
            self.geom = CamGeom(self.cal, self.table, self.size, self.params)
            cal = self.cal["calib_size"]
            if tuple(cal) != tuple(self.size):
                print(f"[{self.name}] 보정 {cal[0]}x{cal[1]} → 실행 {self.size[0]}x{self.size[1]}: 좌표를 {self.geom.scale:.3f}배로 바꿔 씁니다.")
            if not self.geom.has_pose:
                print(f"[{self.name}] 카메라 자세가 없어 P2b 예상 접점과 P5 칸 화면 영역을 쓸 수 없습니다. 다시 보정하세요.")
            self.state = "ok"
        except ResolutionMismatch as e:
            print(f"[{self.name}] {e}")
            self.state = "stopped"
            self.note = "RECALIBRATE"
            self.source.close()

    def close(self):
        self.source.close()


# ---------------------------------------------------------------- 기록 (L1, L2)

class Logs:
    def __init__(self, base, frames_on):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base, stamp)
        n = 1
        while os.path.exists(self.dir):
            n += 1
            self.dir = os.path.join(base, f"{stamp}_{n}")
        os.makedirs(self.dir)
        self.f_state = open(os.path.join(self.dir, "state_changes.csv"), "w", newline="", encoding="utf-8-sig")
        self.w_state = csv.writer(self.f_state)
        self.w_state.writerow(["t_s", "wall", "slot", "prev", "status", "reason", "last_known", "prev_reason", "cams", "why"])
        self.f_frames = None
        self.w_frames = None
        if frames_on:
            self.toggle_frames()

    def state(self, ev):
        self.w_state.writerow([f"{ev['time']:.3f}", datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], ev["slot"],
                               ev["prev"] or "", ev["status"], ev["reason"] or "", ev["last_known"] or "",
                               ev["prev_reason"] or "", ev["cams"], ev["why"]])
        self.f_state.flush()

    def toggle_frames(self):
        if self.f_frames is None:
            new = not os.path.exists(os.path.join(self.dir, "frames.csv"))
            self.f_frames = open(os.path.join(self.dir, "frames.csv"), "a", newline="", encoding="utf-8-sig")
            self.w_frames = csv.writer(self.f_frames)
            if new:
                self.w_frames.writerow(["t_s", "cam", "frame", "row_type", "box", "x1", "y1", "x2", "y2", "conf",
                                        "contact_u", "contact_v", "mat_x", "mat_y", "slot", "slot_mat_x", "slot_mat_y",
                                        "dist_cm", "kind", "straddle", "p2a", "p2a_occluded", "p5_occluded", "obs"])
            return True
        self.f_frames.close()
        self.f_frames = self.w_frames = None
        return False

    def frame(self, t, cam, idx, res):
        if self.w_frames is None:
            return
        for b in res.boxes:
            sm = b.slot_mat if b.slot_mat is not None else (None, None)
            self.w_frames.writerow([f"{t:.3f}", cam, idx, "box", b.i] + [f"{v:.1f}" for v in b.box] +
                                   [f"{b.conf:.3f}", f"{b.contact_px[0]:.1f}", f"{b.contact_px[1]:.1f}",
                                    f"{b.contact_mat[0]:.2f}", f"{b.contact_mat[1]:.2f}",
                                    "" if b.slot is None else b.slot,
                                    "" if sm[0] is None else f"{sm[0]:.2f}", "" if sm[1] is None else f"{sm[1]:.2f}",
                                    "" if b.dist is None else f"{b.dist:.2f}", b.kind,
                                    "+".join(map(str, b.straddle)), b.flags(),
                                    "+".join(map(str, b.occluded_slots)), "", ""])
        p5 = ";".join(f"{n}:{c:.2f}" for n, c in sorted(res.cover.items()) if c > 0)
        obs = ";".join(f"{n}:{(o or '-')[0]}" for n, o in sorted(res.obs.items()))
        self.w_frames.writerow([f"{t:.3f}", cam, idx, "frame"] + [""] * 18 + [p5, obs])
        self.f_frames.flush()

    def info(self, d):
        with open(os.path.join(self.dir, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    def close(self):
        self.f_state.close()
        if self.f_frames:
            self.f_frames.close()


# ---------------------------------------------------------------- 프로그램

class App:
    def __init__(self, args):
        self.args = args
        self.cfg = cfg = load_run(args.run, args.preset)
        for s in args.source or []:
            name, _, src = s.partition("=")
            hit = [c for c in cfg["cameras"] if (c.get("name") or "") == name or
                   os.path.basename(c["config"]) == f"cam_{name}.json"]
            if not hit:
                raise SystemExit(f"--source {s}: 실행 설정에 {name} 카메라가 없습니다.")
            hit[0]["source"] = src if source_kind(src) != "file" else os.path.abspath(src)
        self.table = SlotTable(cfg["slot_table"])
        self.params = Params(cfg, self.table)
        entries = [c for c in cfg["cameras"] if c.get("enabled", True)]
        if not entries:
            raise SystemExit(f"실행 설정에 카메라가 없습니다: {cfg['_path']}")
        self.cams = [Cam(e, cfg, self.table, self.params) for e in entries]
        names = [c.name for c in self.cams]
        if len(set(names)) != len(names):
            raise SystemExit(f"카메라 이름이 겹칩니다: {names}")
        cover = {}
        for c in self.cams:
            for no in c.monitored:
                cover.setdefault(no, []).append(c.name)
        self.tracker = Tracker(self.table.slots.keys(), cover, self.params)
        self.logs = Logs(cfg["log"]["dir"], args.log_frames or cfg["log"].get("frames", False))
        self.tracker.on_change(self.on_status_change)
        self.listeners = []           # 나중에 Firebase 등을 붙이는 곳 (add_listener)
        self.all_files = all(c.source.kind == "file" for c in self.cams)
        if args.detector == "gt":
            self.detector = GTDetector(self.cams)
        else:
            self.detector = ModelDetector(cfg["model"], args.conf if args.conf is not None else cfg["conf"], cfg["imgsz"])
        disp = cfg["display"]
        self.plan = PlanView(self.table, disp.get("plan_px_per_cm", 10))
        self.cam_h = int(disp.get("cam_height", 540))
        self.overlay_s = float(disp.get("startup_overlay_s", 5.0))
        self.view = -1 if len(self.cams) > 1 else 0
        self.demo = False
        self.show_zone = False
        self.paused = False
        self.selected = None
        self.layout = []
        self.t = 0.0
        self.wall0 = None
        self.quiet = args.quiet

    def add_listener(self, fn):
        self.listeners.append(fn)

    def on_status_change(self, ev):
        """상태 변화를 내보내는 한 곳. L1 기록을 쓰고, 등록한 함수(나중에 Firebase spots 쓰기)를 부른다."""
        self.logs.state(ev)
        if ev["prev"] is not None and not self.quiet:
            lk = f" (직전 {ev['last_known']})" if ev["last_known"] else ""
            print(f"[{ev['time']:8.2f}s] {ev['slot']:2d}번 {ev['prev']} → {ev['status']}"
                  f"{' ' + ev['reason'] if ev['reason'] else ''}{lk}")
        for fn in self.listeners:
            fn(ev)

    # ---------------------------------------------------------------- 실행
    def run(self):
        afps = float(self.cfg["analyze_fps"])
        for c in self.cams:
            c.open(0.0)
        if self.cfg["log"].get("record"):
            self.toggle_record()
        self.logs.info({"run": self.cfg["_path"], "preset": self.args.preset, "detector": self.args.detector,
                        "clock": "video" if self.all_files else "wall", "params": self.params.base,
                        "overrides": self.cfg["overrides"],
                        "cameras": [{"name": c.name, "source": str(c.src), "requested": c.req, "opened": c.size,
                                     "calib_size": c.cal.get("calib_size"), "state": c.state,
                                     "monitored": c.monitored} for c in self.cams]})
        print(f"[판정] 기록 폴더: {self.logs.dir}")
        print(f"[판정] 시계: {'영상 시각' if self.all_files else '실제 시각'}, 초당 {afps:g}장 분석")
        if not self.args.no_window:
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WIN, self.on_mouse)
        self.wall0 = time.time()
        self.tracker.start(0.0)
        k = 0
        try:
            while True:
                if not self.paused:
                    if self.all_files:
                        if not self.step_files():
                            break
                    else:
                        self.t = time.time() - self.wall0
                        self.step_live()
                    k += 1
                if self.args.duration and self.t >= self.args.duration:
                    break
                if not self.args.no_window:
                    if not self.ui(afps):
                        break
                elif not self.all_files:
                    time.sleep(max(0.0, self.wall0 + k / afps - time.time()))
        finally:
            self.close()

    def step_files(self):
        """영상 파일만 있을 때: 카메라마다 다음 분석 프레임을 받아 같은 영상 시각으로 판정."""
        items = {}
        for c in self.cams:
            if c.state == "stopped" or c.ended:
                continue
            it = c.source.get(timeout=30.0)
            if it is None:
                c.ended = True
                print(f"[{c.name}] 영상이 끝났습니다.")
                continue
            items[c.name] = it
        if not items:
            return False
        t = max(it[0] for it in items.values())
        self.t = t
        sp = self.args.speed
        if sp and sp > 0:
            time.sleep(max(0.0, self.wall0 + t / sp - time.time()))
        for c in self.cams:
            if c.name in items:
                tv, idx, frame = items[c.name]
                self.process(c, frame, idx)
        self.finish_tick()
        return True

    def step_live(self):
        for c in self.cams:
            if c.state == "stopped":
                continue
            if isinstance(c.source, FileSource):
                it = None if c.ended else c.source.get(timeout=0.01)
                if it is None:
                    continue
                frame, idx = it[2], it[1]
            else:
                frame, tw, n = c.source.latest()
                if frame is None or n == c.last_n:
                    continue
                c.last_n, idx = n, n
                if c.size is None:
                    c.size = (frame.shape[1], frame.shape[0])
                    c.setup_geom()
                    if c.state == "stopped":
                        continue
            self.process(c, frame, idx)
        self.finish_tick()

    def process(self, c, frame, idx):
        if c.geom is None:
            c.size = (frame.shape[1], frame.shape[0])
            c.setup_geom()
            if c.geom is None:
                return
        if (frame.shape[1], frame.shape[0]) != tuple(c.geom.size):
            frame = cv2.resize(frame, tuple(c.geom.size))
        t0 = time.time()
        dets = self.detector(c, frame, idx)
        c.ms = (time.time() - t0) * 1000
        c.res = judge_frame(c.geom, self.params, dets)
        c.frame, c.idx, c.last_t = frame, idx, self.t
        c.state = "ok"
        self.logs.frame(self.t, c.name, idx, c.res)

    def finish_tick(self):
        t_off = self.params(None)["t_offline_s"]
        t_stale = self.params(None)["t_stale_s"]
        obs, state, why = {}, {}, {}
        for c in self.cams:
            if c.state != "stopped" and c.last_t is not None and self.t - c.last_t > t_off:
                if c.state != "offline":
                    print(f"[{c.name}] {t_off:g}초 넘게 프레임이 없습니다(P10).")
                c.state = "offline"
            state[c.name] = c.state
            if c.state == "ok" and c.last_t is not None and self.t - c.last_t > t_stale:
                state[c.name] = "stale"
            if c.res is not None:
                obs[c.name] = c.res.obs
                why[c.name] = c.res.why
        self.tracker.set_why(why)
        self.tracker.update(self.t, obs, state)

    # ---------------------------------------------------------------- 화면 (D1~D6)
    def status_map(self):
        return {no: s.status for no, s in self.tracker.slots.items()}

    def render(self):
        st = self.status_map()
        pts = []
        for c in self.cams:
            if c.res is None or c.state != "ok":
                continue
            for b in c.res.boxes:
                pts.append((b.contact_mat, (40, 40, 230) if b.slot is not None else (255, 160, 0)))
        cams_pos = []
        for c in self.cams:
            p = (c.cal.get("pose") or {}).get("cam_pos_cm")
            if p:
                cams_pos.append((c.name, p[0], p[1]))
        n = self.tracker.counts()
        if self.demo:
            k = min(1600 / self.plan.size[0], 860 / self.plan.size[1])
            big = PlanView(self.table, self.plan.k * k)
            im = big.render(st, pts, self.selected, cams_pos)
            bar = np.full((60, im.shape[1], 3), 30, np.uint8)
            x = 10
            for key, lab in ((EMPTY, "EMPTY"), (OCCUPIED, "OCCUPIED"), (DYNAMIC_BLIND, "UNKNOWN"), (STATIC_BLIND, "BLIND")):
                cv2.rectangle(bar, (x, 15), (x + 30, 45), COLORS[key], -1)
                x += 40 + text(bar, f"{lab} {n[key]}", (x + 38, 40), 0.9, (255, 255, 255), 2, None) + 40
            self.layout = [("plan", 0, 0, big)]
            return np.vstack([im, bar])
        # 카메라 화면
        views = []
        show = self.cams if self.view < 0 else [self.cams[self.view]]
        plan = self.plan.render(st, pts, self.selected, cams_pos)
        if len(show) == 1:
            each_w = int(self.cam_h * 16 / 9)
        else:                          # 여러 대는 위아래로 쌓는다 (D5, v로 하나씩 크게)
            each_w = int(max(self.cam_h, plan.shape[0]) / len(show) * 16 / 9)
        col_w = each_w
        startup = self.t < self.overlay_s
        for c in show:
            if c.frame is None or c.geom is None:
                v = np.full((int(each_w * 9 / 16), each_w, 3), 20, np.uint8)
                text(v, f"{c.name}: {c.state} {c.note}", (10, 30), 0.6, (0, 200, 255), 1)
                views.append((c, v, 0))
                continue
            s = each_w / c.frame.shape[1]
            v = cv2.resize(c.frame, (each_w, int(round(c.frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
            draw_slots(v, c.geom, st, s, thick=4 if startup else 2, show_zone=self.show_zone,
                       show_region=self.show_zone, params=self.params)
            if c.res is not None:
                draw_boxes(v, c.res, s)
            if startup:
                text(v, f"CHECK SLOT LINES ({self.overlay_s - self.t:.0f}s)", (10, v.shape[0] - 14), 0.8, (0, 255, 255), 2)
            stt = c.state if c.state != "ok" else f"{c.size[0]}x{c.size[1]} {c.ms:.0f}ms"
            text(v, f"{c.name} {stt}{' REC' if c.source.recording() else ''}", (8, 20), 0.55,
                 (0, 0, 255) if c.source.recording() else (255, 255, 255), 1)
            views.append((c, v, s))
        cam_col = np.full((sum(v.shape[0] for _, v, _ in views), col_w, 3), 20, np.uint8)
        y = 0
        self.layout = []
        for c, v, s in views:
            cam_col[y:y + v.shape[0], :v.shape[1]] = v
            self.layout.append(("cam", 0, y, (c, s)))
            y += v.shape[0]
        H = max(cam_col.shape[0], plan.shape[0]) + 150
        canvas = np.full((H, col_w + plan.shape[1], 3), 30, np.uint8)
        canvas[:cam_col.shape[0], :col_w] = cam_col
        canvas[:plan.shape[0], col_w:col_w + plan.shape[1]] = plan
        self.layout.append(("plan", col_w, 0, self.plan))
        self.draw_info(canvas, max(cam_col.shape[0], plan.shape[0]) + 8, n)
        return canvas

    def draw_info(self, im, y, n):
        clock = "video" if self.all_files else "wall"
        text(im, f"t {self.t:7.1f}s ({clock})  EMP {n[EMPTY]}  OCC {n[OCCUPIED]}  DYN {n[DYNAMIC_BLIND]}  "
                 f"STA {n[STATIC_BLIND]}  {'PAUSED' if self.paused else ''}  log: {os.path.basename(self.logs.dir)}"
                 f"{'  frames.csv ON' if self.logs.w_frames else ''}", (8, y + 14), 0.5, (255, 255, 255), 1, None)
        text(im, "s shot  r rec  l frame log  v view  d demo  o zones  p pause  q quit  | click slot = info",
             (8, y + 34), 0.45, (200, 200, 200), 1, None)
        if self.selected is not None:
            s = self.tracker.slots[self.selected]
            p = self.params(self.selected)
            rem = s.remaining(self.t, p)
            line1 = (f"slot {s.no}: {s.status}{' ' + s.reason if s.reason else ''}"
                     f"{'  last ' + s.last_known if s.last_known else ''}  frame obs {s.obs or '-'}"
                     f"  next change in {'-' if rem is None else f'{rem:.1f}s'}")
            cams = ", ".join(f"{c}:{o or '-'}" for c, o in s.obs_cams.items()) or "no camera (STATIC_BLIND)"
            text(im, line1, (8, y + 60), 0.55, COLORS.get(s.status, (255, 255, 255)), 1, None)
            text(im, f"cams {cams}", (8, y + 82), 0.5, (255, 255, 255), 1, None)
            if s.why:
                text(im, f"why {s.why}"[:150], (8, y + 104), 0.45, (200, 200, 200), 1, None)
        else:
            text(im, "click a slot on the plan or camera view", (8, y + 60), 0.5, (150, 150, 150), 1, None)

    def on_mouse(self, ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        for kind, x0, y0, obj in self.layout:
            if kind == "plan":
                p = obj.to_mat((x - x0, y - y0))
                w, h = self.table.mat_size
                if -1 <= p[0] <= w + 1 and -1 <= p[1] <= h + 1:
                    no = self.table.slot_at(p)
                    if no is not None:
                        self.selected = no
                        return
            else:
                c, s = obj
                if c.geom is None or c.frame is None or s == 0:
                    continue
                u, v = (x - x0) / s, (y - y0) / s
                if 0 <= u < c.geom.size[0] and 0 <= v < c.geom.size[1] and x - x0 < c.frame.shape[1] * s:
                    for no, fl in c.geom.floor_px.items():
                        if cv2.pointPolygonTest(fl.astype(np.float32), (float(u), float(v)), False) >= 0:
                            self.selected = no
                            return

    def ui(self, afps):
        canvas = self.render()
        cv2.imshow(WIN, canvas)
        wait = 1
        if not self.all_files:
            wait = max(1, int((self.wall0 + (int(self.t * afps) + 1) / afps - time.time()) * 1000))
        k = cv2.waitKey(wait) & 0xFF
        if k == 255:
            return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
        ch = chr(k).lower() if k < 128 else ""
        if ch == "q" or k == 27:
            return False
        if ch == "s":
            self.screenshot(canvas)
        elif ch == "r":
            self.toggle_record()
        elif ch == "l":
            on = self.logs.toggle_frames()
            print(f"[판정] 프레임 기록(frames.csv) {'켬' if on else '끔'}")
        elif ch == "v":
            self.view = (self.view + 2) % (len(self.cams) + 1) - 1 if len(self.cams) > 1 else 0
        elif ch == "d":
            self.demo = not self.demo
        elif ch == "o":
            self.show_zone = not self.show_zone
        elif ch == "p" and self.all_files:
            self.paused = not self.paused
            if not self.paused:
                self.wall0 = time.time() - self.t / (self.args.speed or 1e9)
        return True

    def screenshot(self, canvas):
        """L3: 지금 화면, 카메라 원본, 평면도."""
        stamp = datetime.datetime.now().strftime("%H%M%S")
        save_image(os.path.join(self.logs.dir, f"shot_{stamp}_screen.jpg"), canvas)
        save_image(os.path.join(self.logs.dir, f"shot_{stamp}_plan.png"), self.plan.render(self.status_map()))
        for c in self.cams:
            if c.frame is not None:
                im = c.frame.copy()
                draw_slots(im, c.geom, self.status_map(), 1.0, 2)
                if c.res is not None:
                    draw_boxes(im, c.res)
                save_image(os.path.join(self.logs.dir, f"shot_{stamp}_{c.name}.jpg"), im)
        print(f"[판정] 화면 저장: {self.logs.dir}\\shot_{stamp}_*")

    def toggle_record(self):
        """L4: 카메라 원본 녹화. 영상 파일 입력은 녹화하지 않는다."""
        live = [c for c in self.cams if c.source.kind != "file"]
        if not live:
            print("[판정] 영상 파일 입력은 녹화하지 않습니다.")
            return
        if any(c.source.recording() for c in live):
            for c in live:
                p = c.source.stop_record()
                if p:
                    print(f"[{c.name}] 녹화 끝: {p}")
            return
        stamp = datetime.datetime.now().strftime("%H%M%S")
        for c in live:
            p = os.path.join(self.logs.dir, f"rec_{c.name}_{stamp}.mp4")
            if c.source.start_record(p):
                print(f"[{c.name}] 녹화 시작: {p}")
            else:
                print(f"[{c.name}] 녹화를 시작하지 못했습니다(아직 프레임이 없거나 저장 실패).")

    def close(self):
        for c in self.cams:
            p = c.source.stop_record()
            if p:
                print(f"[{c.name}] 녹화 끝: {p}")
            c.close()
        n = self.tracker.counts()
        print(f"[판정] 끝 (t {self.t:.1f}s). EMPTY {n[EMPTY]}, OCCUPIED {n[OCCUPIED]}, "
              f"DYNAMIC_BLIND {n[DYNAMIC_BLIND]}, STATIC_BLIND {n[STATIC_BLIND]}. 기록: {self.logs.dir}")
        self.logs.close()
        if not self.args.no_window:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="PARKIT 점유 판정 프로그램")
    ap.add_argument("--run", default=DEFAULT_RUN, help="실행 설정 json")
    ap.add_argument("--preset", default=None, help="실행 설정의 presets 중 하나 (예: demo)")
    ap.add_argument("--source", action="append", help="카메라 입력 바꾸기: 이름=웹캠번호|영상|rtsp (여러 번 가능)")
    ap.add_argument("--conf", type=float, default=None, help="신뢰도 (기본: 실행 설정, 0.25)")
    ap.add_argument("--detector", choices=["model", "gt"], default="model", help="gt = 시험용 정답 상자")
    ap.add_argument("--speed", type=float, default=1.0, help="영상 파일 재생 속도 (1 = 실제 속도, 0 = 최대한 빨리)")
    ap.add_argument("--duration", type=float, default=0, help="이 시간(초)이 지나면 끝")
    ap.add_argument("--log-frames", action="store_true", help="처음부터 프레임 기록(frames.csv) 켜기")
    ap.add_argument("--no-window", action="store_true", help="창 없이 실행 (시험용)")
    ap.add_argument("--quiet", action="store_true", help="상태 변화를 콘솔에 찍지 않음")
    args = ap.parse_args()
    App(args).run()


if __name__ == "__main__":
    sys.exit(main())
