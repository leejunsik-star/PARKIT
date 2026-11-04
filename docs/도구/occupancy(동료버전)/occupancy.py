"""PARKIT 점유 판정 프로그램 (25 문서 P1~P11, D1~D6, L1~L4) — tkinter 화면

카메라마다 M8로 차를 찾고(P1), 바닥 접점으로 칸을 정하고(P2~P5, parkit_judge),
시간 거르기와 카메라 합치기(P6~P10, parkit_state)를 거쳐 칸 상태를 화면과 기록 파일에 낸다.
카메라 수와 감시 칸은 설정 파일로만 정한다(config\\run.json 의 cameras, config\\cam_*.json 의 monitored).
이전 판(OpenCV 창)은 이전\\20261101_OpenCV창판\\ 에 있다.

사용법 (docs\\도구\\occupancy 에서):
    python occupancy.py                                 config\\run.json 대로
    python occupancy.py --preset demo                   시연용 짧은 시간 값(1초/2초)
    python occupancy.py --source CAM-1=녹화.mp4          CAM-1 입력만 영상 파일로 바꿔서
    python occupancy.py --run config\\run_2대.json       카메라 2대
    python occupancy.py --every 10                      10프레임마다 분석 (창에서도 바꿀 수 있다)
    python occupancy.py --model models/M7.pt             다른 모델로 (창에서도 고를 수 있다)
    python occupancy.py --no-window --speed 0           창 없이 최대한 빨리 (시험용)
단축키와 마우스는 창에서 ? 키(도움말)로 본다.

구조
    Engine: 프레임 받기 → N프레임마다 분석 → 상태 판정 → 기록. 별도 스레드에서 돈다(창 없이도 같은 코드).
    OccApp: tkinter 화면. Engine의 결과를 주기적으로 읽어 그리고, 설정 변경은 Engine에 명령으로 넘긴다.
    웹캠 열기(시작, 웹캠·해상도 바꾸기, 끊긴 뒤 다시 열기)는 메인 스레드에서 한다(main_thread_work).
    맥의 OpenCV(AVFoundation)는 메인 스레드에서 열어야 카메라 권한을 물을 수 있기 때문이다(13 문서).
시간: 입력이 모두 영상 파일이면 영상 시각을 쓴다. 그래서 같은 파일, 같은 분석 간격이면 PC 속도와 상관없이
같은 결과가 나온다(C2). 웹캠이나 rtsp가 하나라도 있으면 실제 시각을 쓴다.
"""

import argparse, collections, csv, datetime, glob, json, os, queue, sys, threading, time

import cv2
import numpy as np

from parkit_config import DEFAULT_RUN, HERE, MODEL_DIR, Params, load_json, load_run
from parkit_draw import PlanView, draw_boxes, draw_slots, save_image, text
from parkit_geom import CamGeom, ResolutionMismatch, SlotTable
from parkit_judge import judge_frame
from parkit_state import DYNAMIC_BLIND, EMPTY, OCCUPIED, STATIC_BLIND, Tracker
from parkit_video import FileSource, LiveSource, make_source, source_kind

TOOLS_DIR = os.path.dirname(HERE)          # 저장소 안이면 docs/도구 (모델 파일이 더 있는 곳)
CONF_CHOICES = ["0.25", "0.3", "0.35", "0.4"]   # 25 문서 P1: 0.25 기본, 0.4까지
SPEEDS = [("0.5배", 0.5), ("1배 (실제 속도)", 1.0), ("2배", 2.0), ("4배", 4.0), ("최대", 0.0)]
TICK_S = 0.5            # 분석이 없어도 이 간격으로 상태를 다시 본다(끊김 판정)

HELP = [
    ("보기", [("F", "모든 프레임 보기 ↔ 분석한 프레임만 보기"),
             ("V", "카메라 화면 바꾸기 (모두 → 1번 → 2번 …)"),
             ("O", "판정 구역(하늘색)·칸 화면 영역(파랑) 보이기"),
             ("D", "시연 화면 열기/닫기 (큰 평면도와 칸 수. 시연 화면에서 F11 = 전체 화면)")]),
    ("분석", [("[ / ]", "분석 간격 줄이기 / 늘리기 (N프레임마다 한 장 분석)"),
             ("Space", "일시정지 / 계속 (영상 파일)")]),
    ("모델", [("모델 목록", "occupancy/models (와 저장소 안이면 docs/도구/models, docs/도구)의 .pt 파일 중에서 고르기. "
                         "YOLO 모델과 동료가 만든 PARKIT Tiny(mymodel_*.pt, model.py 필요) 모두 된다. "
                         "불러오는 동안은 앞 모델로 계속 판정"),
             ("[파일 찾기…]", "다른 위치의 .pt 모델 파일 고르기"),
             ("신뢰도 목록", "0.25 ~ 0.4 (25 문서 P1)")]),
    ("기록", [("R", "녹화 켜기/끄기 (웹캠·rtsp 원본, L4)"),
             ("S", "화면 저장 (평면도, 카메라 화면, L3)"),
             ("L", "프레임 기록 frames.csv 켜기/끄기 (L2)")]),
    ("마우스", [("평면도나 카메라 화면의 칸 클릭", "칸 정보: 상태, 이번 판정, 바뀌기까지 남은 시간, 본 카메라 (D3)"),
               ("카메라 줄의 웹캠 번호·해상도 + [적용]", "웹캠 바꾸기, 해상도 바꾸기 (이번 실행에만. 보정 때와 화면 비율이 같아야 한다)")]),
    ("기타", [("? / F1", "이 도움말"), ("Q / Esc", "끝내기")]),
]


# ---------------------------------------------------------------- 탐지기 (P1)

def list_models(extra=()):
    """고를 수 있는 모델 파일. (화면에 보일 이름, 절대 경로) 목록.
    occupancy/models/*.pt (폴더만 보내도 있는 것), 저장소 안이면 docs/도구/models/*.pt, docs/도구/*.pt 도."""
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if os.path.isdir(os.path.join(TOOLS_DIR, "models")):          # 저장소 안 (docs/도구)
        paths += sorted(glob.glob(os.path.join(TOOLS_DIR, "models", "*.pt"))) + sorted(glob.glob(os.path.join(TOOLS_DIR, "*.pt")))
    # YOLOE(문장 프롬프트가 필요한 모델)는 뺀다. 동료가 만든 PARKIT Tiny(mymodel*)는 TinyDetector가 돌린다
    paths = [q for q in paths if not os.path.basename(q).lower().startswith("yoloe")]
    for p in extra:
        if p and os.path.abspath(p) not in [os.path.abspath(q) for q in paths]:
            paths.insert(0, p)
    local = {os.path.basename(q) for q in glob.glob(os.path.join(MODEL_DIR, "*.pt"))}
    out, seen = [], set()
    for p in paths:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.dirname(p) == os.path.abspath(os.path.join(TOOLS_DIR, "models")) and os.path.basename(p) in local:
            continue                               # occupancy/models 에 같은 이름이 있으면 그것만 보인다
        try:
            rel = os.path.relpath(p, HERE).replace(os.sep, "/")
            if rel.startswith(".."):
                rel = "도구/" + os.path.relpath(p, TOOLS_DIR).replace(os.sep, "/")
        except ValueError:                         # 다른 드라이브
            rel = p
        out.append((rel if not rel.startswith("도구/..") else p, p))
    return out


class ModelDetector:
    """모델 하나를 모든 카메라가 같이 쓴다(P11)."""

    def __init__(self, path, conf, imgsz):
        from ultralytics import YOLO
        if not os.path.exists(path):
            raise SystemExit(f"모델 파일이 없습니다: {path}")
        self.path = os.path.abspath(path)
        self.model = YOLO(path)
        self.conf, self.imgsz = float(conf), int(imgsz)
        self.names = self.model.names
        self.model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False)   # 첫 추론 준비
        print(f"[판정] 모델: {path} (클래스 {self.classes()}), 신뢰도 {self.conf}, imgsz {self.imgsz}")

    def classes(self):
        """클래스 설명: 적으면 이름 그대로, 많으면(COCO 80개 등) 개수와 car만 쓴다는 말."""
        if len(self.names) <= 5:
            return str(self.names)
        return f"{len(self.names)}개 중 car만 씀"

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


def parkit_checkpoint(path):
    """동료가 직접 만든 PARKIT Tiny 체크포인트({"model": 가중치, "width", "size", ...})면 그 내용을, 아니면 None.
    live_detect_winmac.py와 같은 방법: weights_only=True로 읽어 가중치 말고 다른 코드는 실행하지 않는다.
    Ultralytics 파일은 여기서 실패해 None이 된다."""
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict) and "stem.0.weight" in ck["model"]:
        return ck
    return None


class TinyDetector:
    """동료가 처음부터 만든 PARKIT Tiny 모델(model.py + mymodel_*.pt)을 돌린다(live_detect_winmac.py의 ParkitTiny와 같은 방법).
    모델 구조는 model.py에서 가져온다. 찾는 순서: 가중치 파일 옆 → occupancy 폴더 → docs/도구.
    입력: 가운데 letterbox(회색 114), RGB, 0~1. 클래스는 car 하나."""
    names = {0: "car"}

    def __init__(self, path, ck, conf, imgsz):
        import importlib.util
        import torch
        for p in (os.path.join(os.path.dirname(os.path.abspath(path)), "model.py"), os.path.join(HERE, "model.py"),
                  os.path.join(TOOLS_DIR, "model.py")):
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("parkit_tiny_model", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.model_py = p
                break
        else:
            raise SystemExit(f"PARKIT Tiny 모델 구조 파일 model.py가 없습니다. 가중치 파일({os.path.basename(path)}) 옆에 두세요.")
        self.torch, self.mod = torch, mod
        self.net = mod.ParkitTiny(ck.get("width", 16))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.path = os.path.abspath(path)
        self.conf, self.imgsz = float(conf), int(imgsz)
        with torch.no_grad():
            self(None, np.zeros((360, 640, 3), np.uint8), 0)               # 첫 추론 준비
        print(f"[판정] 모델: {path} (PARKIT Tiny, 너비 {ck.get('width', 16)}, 학습 크기 {ck.get('size', 640)}, "
              f"구조 {self.model_py}), 신뢰도 {self.conf}, imgsz {self.imgsz}")

    def classes(self):
        return "PARKIT Tiny, car"

    def __call__(self, cam, frame, idx):
        size = max(32, self.imgsz // 16 * 16)          # 출력 칸이 8, 16 간격이라 16의 배수로 맞춘다
        h, w = frame.shape[:2]
        s = size / max(h, w)
        rs = cv2.resize(frame, (int(round(w * s)), int(round(h * s))))
        pad = np.full((size, size, 3), 114, np.uint8)
        dy, dx = (size - rs.shape[0]) // 2, (size - rs.shape[1]) // 2
        pad[dy:dy + rs.shape[0], dx:dx + rs.shape[1]] = rs
        x = self.torch.from_numpy(np.ascontiguousarray(pad[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255.0
        with self.torch.no_grad():
            boxes, scores = self.mod.predict(self.net, x, conf=self.conf)[0]
        out = []
        for (x1, y1, x2, y2), p in zip(boxes.tolist(), scores.tolist()):
            out.append(((x1 - dx) / s, (y1 - dy) / s, (x2 - dx) / s, (y2 - dy) / s, float(p)))
        return out


def make_detector(path, conf, imgsz):
    """모델 파일을 보고 PARKIT Tiny면 TinyDetector, 아니면 ultralytics(YOLO 등) ModelDetector."""
    if not os.path.exists(path):
        raise SystemExit(f"모델 파일이 없습니다: {path}")
    ck = parkit_checkpoint(path)
    if ck is not None:
        return TinyDetector(path, ck, conf, imgsz)
    return ModelDetector(path, conf, imgsz)


class GTDetector:
    """시험용: 영상 옆의 <영상>.gt.json 에 적힌 상자를 그대로 낸다(test_pipeline.py). 실제 시험에는 쓰지 않는다."""

    def __init__(self, cams):
        self.path = "gt"
        self.conf = 1.0
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
        self.loop = entry.get("loop", False)
        self.monitored = sorted(int(n) for n in self.cal.get("monitored", []))
        self.table, self.params = table, params
        self.source = None
        self.kind = source_kind(self.src)
        self.state = "offline"          # ok / offline / stopped
        self.geom = None
        self.size = None
        self.last_t = None              # 마지막 프레임을 분석한 시각(판정 시계)
        self.last_n = 0
        self.since = 0                  # 마지막 분석 뒤 받은 프레임 수
        self.frame = None               # 마지막으로 분석한 프레임
        self.display = None             # 가장 최근 프레임 (모든 프레임 보기)
        self.idx = 0
        self.res = None
        self.ms = 0.0
        self.ended = False
        self.pending = None
        self.note = ""

    def describe(self):
        if self.kind == "camera":
            return f"웹캠 {self.src}번"
        return os.path.basename(str(self.src)) if self.kind == "file" else str(self.src)

    def fps(self):
        if self.source is None:
            return 0.0
        return self.source.fps() if isinstance(self.source, LiveSource) else self.source.fps

    def make(self, every_n, need_all, src=None, size=None):
        """입력을 만들어 연다(메인 스레드에서). (입력, 열렸는지)"""
        src = self.src if src is None else src
        size = self.req if size is None else tuple(size)
        source = make_source(self.name, src, size[0], size[1], every_n, self.loop, need_all)
        ok = source.open()
        if not ok and self.kind == "camera":
            time.sleep(1.0)                    # 막 놓은 웹캠은 풀리는 데 조금 걸린다
            ok = source.open()
        print(f"[{self.name}] {source.msg}")
        return source, ok

    def open(self, every_n, need_all, t0, size=None):
        source, ok = self.make(every_n, need_all, size=size)
        self.install(source, ok, t0, size=size)

    def install(self, source, ok, t0, src=None, size=None):
        """연 입력을 이 카메라에 끼운다(엔진 잠금 안에서)."""
        if src is not None:
            self.src = src
        if size is not None:
            self.req = tuple(size)
        self.source = source
        self.geom, self.size, self.note = None, None, ""
        self.res = self.frame = self.display = None
        self.state = "offline"
        self.last_n = self.since = 0
        self.last_t = t0
        if ok:
            self.source.start()
            self.size = self.source.size
            self.setup_geom()
        elif self.kind == "file":
            self.state = "stopped"
            self.note = "영상을 열지 못함"
        else:
            self.state = "offline"
            self.source.start()          # 웹캠/rtsp는 스레드가 계속 다시 열어 본다

    def setup_geom(self):
        """C6: 실행 해상도가 보정과 비율이 같으면 좌표를 바꿔 쓰고, 다르면 멈춘다."""
        if self.geom is not None or self.size is None:
            return
        if tuple(self.size) != self.req and self.kind != "file":
            print(f"[{self.name}] 요청 해상도 {self.req[0]}x{self.req[1]}가 아니라 {self.size[0]}x{self.size[1]}로 열렸습니다(C1).")
        try:
            self.geom = CamGeom(self.cal, self.table, self.size, self.params)
            cal = self.cal["calib_size"]
            if tuple(cal) != tuple(self.size):
                print(f"[{self.name}] 보정 {cal[0]}x{cal[1]} → 실행 {self.size[0]}x{self.size[1]}: 좌표를 {self.geom.scale:.3f}배로 바꿔 씁니다.")
            if not self.geom.has_pose:
                print(f"[{self.name}] 카메라 자세가 없어 P2b 예상 접점과 P5 칸 화면 영역을 쓸 수 없습니다. 다시 보정하세요.")
            self.state = "ok"
            self.note = ""
        except ResolutionMismatch as e:
            print(f"[{self.name}] {e}")
            self.state = "stopped"
            self.note = f"화면 비율이 보정({self.cal['calib_size'][0]}x{self.cal['calib_size'][1]})과 달라 멈춤 — 다시 보정하세요"

    def close(self):
        if self.source is not None:
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
        self.lock = threading.Lock()
        if frames_on:
            self.toggle_frames()

    def state(self, ev):
        self.w_state.writerow([f"{ev['time']:.3f}", datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], ev["slot"],
                               ev["prev"] or "", ev["status"], ev["reason"] or "", ev["last_known"] or "",
                               ev["prev_reason"] or "", ev["cams"], ev["why"]])
        self.f_state.flush()

    def toggle_frames(self):
        with self.lock:
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
        with self.lock:
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


# ---------------------------------------------------------------- 판정 엔진 (화면 코드 없음)

class Engine:
    def __init__(self, args, headless=False):
        self.args = args
        self.headless = headless
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
        self.listeners = []              # 나중에 Firebase 등을 붙이는 곳 (add_listener)
        self.tracker.on_change(self.on_status_change)
        self.all_files = all(c.kind == "file" for c in self.cams)
        self.every_n = int(args.every or cfg["analyze_every_frames"])
        self.view_all = bool(cfg.get("view_all_frames", True)) and not headless
        self.speed = float(args.speed)
        self.paused = False
        self.running = False
        self.finished = False
        self.t = 0.0
        self.wall0 = None
        self.pause_t0 = None
        self.last_tick = -1e9
        self.detector = None
        self.model_path = os.path.abspath(args.model) if getattr(args, "model", None) else cfg["model"]
        self.conf = float(args.conf if args.conf is not None else cfg["conf"])
        self.cmds = queue.Queue()          # 엔진 스레드에서 처리할 명령
        self.main_cmds = queue.Queue()     # 메인 스레드에서 처리할 명령 (웹캠 열기)
        self.lock = threading.RLock()      # 엔진 한 걸음과 메인 스레드의 입력 바꾸기가 겹치지 않게
        self.opened = False
        self.events = collections.deque(maxlen=300)
        self.n_events = 0
        self.setting_log = []
        self.quiet = args.quiet
        # Optional extension: the original slot Tracker remains untouched.
        self.vehicle_tracking = None
        self.api_server = None
        if getattr(args, "tracking", False) or getattr(args, "web", False):
            from tracking_bridge import TrackingBridge
            tracking_cfg = (load_json(args.tracking_config) if getattr(args, "tracking_config", None)
                            else cfg.get("vehicle_tracking", {}))
            self.vehicle_tracking = TrackingBridge(self.table, tracking_cfg,
                                                  "video" if self.all_files else "wall", names)

    # ---------------- 준비
    def load_detector(self):
        if self.args.detector == "gt":
            self.detector = GTDetector(self.cams)
        else:
            self.detector = make_detector(self.model_path, self.conf, self.cfg["imgsz"])

    def add_listener(self, fn):
        self.listeners.append(fn)

    def on_status_change(self, ev):
        """상태 변화를 내보내는 한 곳. L1 기록을 쓰고, 등록한 함수(나중에 Firebase spots 쓰기)를 부른다."""
        self.logs.state(ev)
        if ev["prev"] is not None:
            self.events.append(ev)
            self.n_events += 1
            if not self.quiet:
                lk = f" (직전 {ev['last_known']})" if ev["last_known"] else ""
                print(f"[{ev['time']:8.2f}s] {ev['slot']:2d}번 {ev['prev']} → {ev['status']}"
                      f"{' ' + ev['reason'] if ev['reason'] else ''}{lk}")
        for fn in self.listeners:
            fn(ev)

    def open_sources(self):
        """입력을 연다. 창이 있으면 메인 스레드에서 먼저 부른다(맥 웹캠 권한)."""
        if not self.opened:
            for c in self.cams:
                c.open(self.every_n, self.view_all, 0.0)
            self.opened = True

    def start(self):
        self.open_sources()
        for c in self.cams:
            c.last_t = 0.0
        if self.cfg["log"].get("record"):
            self.toggle_record()
        self.write_info()
        print(f"[판정] 기록 폴더: {self.logs.dir}")
        print(f"[판정] 시계: {'영상 시각' if self.all_files else '실제 시각'}, {self.every_n}프레임마다 분석")
        self.wall0 = time.time()
        self.tracker.start(0.0)
        if self.vehicle_tracking is not None:
            self.publish_tracking()
        if getattr(self.args, "web", False):
            from parking_api import ApiServer
            self.api_server = ApiServer(self.vehicle_tracking.store, self.args.web_host, self.args.web_port)
            self.api_server.start()
        self.running = True

    def write_info(self):
        self.logs.info({"run": self.cfg["_path"], "preset": self.args.preset, "detector": self.args.detector,
                        "model": self.detector.path if self.detector else self.model_path,
                        "conf": getattr(self.detector, "conf", self.conf),
                        "clock": "video" if self.all_files else "wall", "analyze_every_frames": self.every_n,
                        "setting_changes": self.setting_log, "params": self.params.base,
                        "overrides": self.cfg["overrides"],
                        "cameras": [{"name": c.name, "source": str(c.src), "requested": c.req, "opened": c.size,
                                     "calib_size": c.cal.get("calib_size"), "state": c.state,
                                     "monitored": c.monitored} for c in self.cams]})

    # ---------------- 실행 중 바꾸기 (화면에서 부른다. 실제 적용은 엔진 스레드에서)
    def command(self, name, *a):
        if name == "resolution":
            self.main_cmds.put((name, a))   # 웹캠 열기는 메인 스레드에서
        else:
            self.cmds.put((name, a))

    def main_thread_work(self):
        """메인 스레드에서 자주 부른다: 웹캠·해상도 바꾸기, (맥) 끊긴 웹캠 다시 열기."""
        while not self.main_cmds.empty():
            name, a = self.main_cmds.get()
            if name != "resolution":
                continue
            # a = (카메라 이름, (폭, 높이), 웹캠 번호 또는 None)
            c = self.cam_by_name(a[0])
            if c is None or c.kind == "file":
                continue
            src = int(a[2]) if len(a) > 2 and a[2] is not None and c.kind == "camera" else c.src
            with self.lock:
                if c.source is not None:
                    c.source.close()           # 같은 웹캠을 다시 열려면 먼저 놓아야 한다
                c.source = None
                c.state = "offline"
            source, ok = c.make(self.every_n, self.view_all, src=src, size=a[1])
            with self.lock:
                c.install(source, ok, self.t, src=src, size=a[1])   # 웹캠 번호는 이번 실행에만 바뀐다
                self.setting_log.append({"t": round(self.t, 3), "camera": c.name, "source": str(c.src),
                                         "resolution": list(a[1])})
                self.write_info()
        for c in self.cams:
            if isinstance(c.source, LiveSource):
                c.source.reopen_on_main()

    def _apply(self, name, a):
        if name == "every":
            n = max(1, min(600, int(a[0])))
            if n != self.every_n:
                self.every_n = n
                for c in self.cams:
                    if isinstance(c.source, FileSource):
                        c.source.every_n = n
                self.setting_log.append({"t": round(self.t, 3), "analyze_every_frames": n})
                print(f"[판정] {self.t:.1f}초: 분석 간격 {n}프레임")
                self.write_info()
        elif name == "view_all":
            self.view_all = bool(a[0])
            for c in self.cams:
                if isinstance(c.source, FileSource):
                    c.source.need_all = self.view_all
        elif name == "detector":
            # 화면이 뒤에서 불러 둔 새 모델로 바꿔 끼운다
            self.detector = a[0]
            self.model_path = self.detector.path
            self.setting_log.append({"t": round(self.t, 3), "model": self.detector.path, "conf": self.detector.conf})
            print(f"[판정] {self.t:.1f}초: 모델 {self.detector.path} (신뢰도 {self.detector.conf})")
            self.write_info()
        elif name == "conf":
            if isinstance(self.detector, (ModelDetector, TinyDetector)):
                self.detector.conf = self.conf = float(a[0])
                self.setting_log.append({"t": round(self.t, 3), "conf": self.conf})
                print(f"[판정] {self.t:.1f}초: 신뢰도 {self.conf}")
                self.write_info()
        elif name == "record":
            self.toggle_record()
        elif name == "frames":
            on = self.logs.toggle_frames()
            print(f"[판정] 프레임 기록(frames.csv) {'켬' if on else '끔'}")

    def cam_by_name(self, name):
        for c in self.cams:
            if c.name == name:
                return c
        return None

    def set_pause(self, on):
        if not self.all_files or on == self.paused:
            return
        if on:
            self.pause_t0 = time.time()
        else:
            self.wall0 += time.time() - (self.pause_t0 or time.time())
        self.paused = on

    def set_speed(self, sp):
        sp = float(sp)
        if self.all_files and sp > 0:
            self.wall0 = time.time() - self.t / sp
        self.speed = sp

    # ---------------- 돌리기
    def run(self):
        """엔진 루프. 창이 있으면 별도 스레드에서, 없으면 그대로 부른다."""
        try:
            while self.running:
                if self.headless:
                    self.main_thread_work()        # 창이 없으면 이 루프가 메인 스레드다
                idle = False
                with self.lock:
                    while not self.cmds.empty():
                        self._apply(*self.cmds.get())
                    if self.paused:
                        if self.vehicle_tracking is not None:
                            self.publish_tracking()
                        idle = True
                    elif self.all_files:
                        if not self.step_files():
                            break
                    else:
                        idle = not self.step_live()
                if idle:
                    time.sleep(0.05 if self.paused else 0.005)
                if self.args.duration and self.t >= self.args.duration:
                    break
        finally:
            self.finished = True
            if self.vehicle_tracking is not None:
                self.publish_tracking()

    def step_files(self):
        """영상 파일만 있을 때: 영상 시각이 가장 이른 프레임부터 차례로. N프레임마다 하나를 분석."""
        for c in self.cams:
            if c.state == "stopped" or c.ended or c.pending is not None:
                continue
            it = c.source.get(timeout=30.0)
            if it is None:
                c.ended = True
                print(f"[{c.name}] 영상이 끝났습니다.")
            elif it != "wait":
                c.pending = it
        waiting = [c for c in self.cams if c.pending is not None]
        if not waiting:
            if any(not c.ended and c.state != "stopped" for c in self.cams):
                time.sleep(0.01)           # 디코딩을 기다린다
                return True
            return False
        t = min(c.pending[0] for c in waiting)
        if self.speed > 0:
            target = self.wall0 + t / self.speed
            while self.running and not self.paused and time.time() < target - 0.002:
                time.sleep(min(0.02, target - time.time()))
        analyzed = False
        self.t = t
        for c in waiting:
            if c.pending[0] > t + 1e-6:
                continue
            tv, idx, frame = c.pending
            c.pending = None
            if frame is not None:
                c.display = frame
            if idx % self.every_n == 0 and frame is not None:
                self.analyze(c, frame, idx, observed_at=tv)
                analyzed = True
        if analyzed or self.t - self.last_tick >= TICK_S:
            self.finish_tick()
        return True

    def step_live(self):
        self.t = time.time() - self.wall0
        new, analyzed = False, False
        for c in self.cams:
            if c.state == "stopped" or c.source is None:
                continue
            if isinstance(c.source, FileSource):        # 웹캠과 섞인 영상 파일: 실제 시각에 맞춰 꺼낸다
                while True:
                    it = c.pending if c.pending is not None else c.source.get(timeout=0.0005)
                    c.pending = None
                    if it is None:
                        c.ended = True
                        break
                    if it == "wait":
                        break
                    if it[0] > self.t:
                        c.pending = it
                        break
                    tv, idx, frame = it
                    if frame is not None:
                        c.display = frame
                        new = True
                    if idx % self.every_n == 0 and frame is not None:
                        self.analyze(c, frame, idx, observed_at=tv)
                        analyzed = True
                continue
            frame, tw, n = c.source.latest()
            if frame is None or n == c.last_n:
                continue
            c.since += n - c.last_n
            c.last_n = n
            c.display = frame
            new = True
            if c.geom is None:
                c.size = (frame.shape[1], frame.shape[0])
                c.setup_geom()
                if c.state == "stopped":
                    continue
            if c.since >= self.every_n:
                c.since = 0
                self.analyze(c, frame, n, observed_at=max(0.0, tw - self.wall0))
                analyzed = True
        if analyzed or self.t - self.last_tick >= TICK_S:
            self.finish_tick()
        return new

    def analyze(self, c, frame, idx, observed_at=None):
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
        if self.vehicle_tracking is not None:
            self.vehicle_tracking.submit(c.name, self.t if observed_at is None else observed_at, c.res)

    def finish_tick(self):
        self.last_tick = self.t
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
        if self.vehicle_tracking is not None:
            self.publish_tracking(state)

    def publish_tracking(self, camera_states=None):
        """Called only by the analysis thread (or before it starts/after it stops)."""
        now = self.t if self.all_files else max(self.t, time.time() - (self.wall0 or time.time()))
        state = "finished" if self.finished else "paused" if self.paused else "running"
        self.vehicle_tracking.publish(now, self.tracker.slots,
                                      camera_states or {c.name: c.state for c in self.cams}, state)

    def toggle_record(self):
        """L4: 카메라 원본 녹화. 영상 파일 입력은 녹화하지 않는다."""
        live = [c for c in self.cams if c.kind != "file" and c.source is not None]
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

    def recording(self):
        return any(c.source is not None and c.source.recording() for c in self.cams)

    def status_map(self):
        return {no: s.status for no, s in self.tracker.slots.items()}

    def close(self):
        self.running = False
        for c in self.cams:
            if c.source is not None:
                p = c.source.stop_record()
                if p:
                    print(f"[{c.name}] 녹화 끝: {p}")
            c.close()
        n = self.tracker.counts()
        self.write_info()
        print(f"[판정] 끝 (t {self.t:.1f}s). EMPTY {n[EMPTY]}, OCCUPIED {n[OCCUPIED]}, "
              f"DYNAMIC_BLIND {n[DYNAMIC_BLIND]}, STATIC_BLIND {n[STATIC_BLIND]}. 기록: {self.logs.dir}")
        self.logs.close()
        if self.api_server is not None:
            self.api_server.close()


# ---------------------------------------------------------------- 화면 (tkinter, D1~D6)

class OccApp:
    def __init__(self, engine):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from parkit_tk import HelpWindow, ImageView, setup_style
        self.tk, self.ttk, self.mb = tk, ttk, messagebox
        self.ImageView = ImageView
        self.e = engine
        self.root = tk.Tk()
        self.root.title("PARKIT 점유 판정")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1500, sw - 40)}x{min(930, sh - 90)}+10+10")   # 노트북·맥북 화면에도 들어가게
        setup_style(self.root)
        self.help = HelpWindow(self.root, "판정 프로그램 단축키와 마우스", HELP,
                               "단축키는 한글 입력 상태에서도 됩니다(같은 자리의 영문 키로 처리). "
                               "글자 입력 칸에 커서가 있으면 단축키가 먹지 않으니 화면을 한 번 클릭하세요.")
        self.selected = None
        self.view_idx = -1 if len(engine.cams) > 1 else 0
        self.show_zone = False
        self.layout = []
        self.plan = None
        self.plan_k = None
        self.demo = None
        self.shown_events = 0
        self.flash_msg = None
        self.ready = False
        self.error = None
        self.end_noted = False
        self.model_loading = False
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<Key>", self.on_key)
        # 웹캠은 메인 스레드에서 연다(맥 카메라 권한). 모델 불러오기와 판정은 스레드에서
        self.banner.configure(text="카메라를 여는 중입니다…")
        self.root.update()
        engine.open_sources()
        self.banner.configure(text="모델을 불러오는 중입니다… (처음 몇 초 걸립니다)")
        threading.Thread(target=self.prepare, daemon=True).start()
        self.after_id = self.root.after(40, self.tick)

    # ---------------- 준비 (모델 불러오기는 오래 걸려 스레드에서)
    def prepare(self):
        try:
            self.e.load_detector()
            self.e.start()
            self.ready = True
            self.e.run()
        except SystemExit as ex:
            self.error = str(ex)
        except Exception as ex:      # 화면에 알리고 콘솔에도 남긴다
            import traceback
            traceback.print_exc()
            self.error = f"{type(ex).__name__}: {ex}"

    # ---------------- 화면 만들기
    def build(self):
        from parkit_tk import RES_CHOICES, STATUS_HEX, STATUS_KO, UI_FONT
        tk, ttk = self.tk, self.ttk
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(2, weight=1)
        e = self.e
        # 도구 막대 1: 재생, 보기, 분석
        bar = ttk.Frame(r, padding=(6, 4))
        bar.grid(row=0, column=0, sticky="ew")
        self.pause_btn = ttk.Button(bar, text="⏸ 일시정지 (Space)", command=self.toggle_pause)
        self.speed_var = tk.StringVar(value=next((n for n, v in SPEEDS if v == e.speed), "1배 (실제 속도)"))
        if e.all_files:
            self.pause_btn.pack(side="left")
            ttk.Label(bar, text="  재생 속도").pack(side="left")
            sp = ttk.Combobox(bar, textvariable=self.speed_var, values=[n for n, _ in SPEEDS], width=13, state="readonly")
            sp.pack(side="left", padx=(4, 12))
            sp.bind("<<ComboboxSelected>>", lambda _: self.set_speed())
        ttk.Label(bar, text="화면에").pack(side="left")
        self.view_all_var = tk.BooleanVar(value=e.view_all)
        ttk.Radiobutton(bar, text="모든 프레임", variable=self.view_all_var, value=True,
                        command=self.set_view).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(bar, text="분석한 프레임만 (F)", variable=self.view_all_var, value=False,
                        command=self.set_view).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="분석 간격 ([ ])").pack(side="left")
        self.every_var = tk.StringVar(value=str(e.every_n))
        sb = ttk.Spinbox(bar, from_=1, to=120, textvariable=self.every_var, width=5, command=self.set_every)
        sb.pack(side="left", padx=4)
        sb.bind("<Return>", lambda _: self.set_every())
        sb.bind("<FocusOut>", lambda _: self.set_every())
        self.every_lbl = ttk.Label(bar, text="")
        self.every_lbl.pack(side="left", padx=(0, 12))
        if len(e.cams) > 1:
            ttk.Label(bar, text="카메라 화면 (V)").pack(side="left")
            self.camview_var = tk.StringVar(value="모두")
            cv = ttk.Combobox(bar, textvariable=self.camview_var, values=["모두"] + [c.name for c in e.cams],
                              width=8, state="readonly")
            cv.pack(side="left", padx=4)
            cv.bind("<<ComboboxSelected>>", lambda _: self.set_camview())
        # 도구 막대 2: 기록, 기타
        bar2 = ttk.Frame(r, padding=(6, 0, 6, 4))
        bar2.grid(row=1, column=0, sticky="ew")
        self.rec_btn = ttk.Button(bar2, text="● 녹화 (R)", command=lambda: e.command("record"))
        self.rec_btn.pack(side="left")
        if all(c.kind == "file" for c in e.cams):
            self.rec_btn.configure(state="disabled")
        ttk.Button(bar2, text="화면 저장 (S)", command=self.screenshot).pack(side="left", padx=4)
        self.frames_var = tk.BooleanVar(value=e.logs.w_frames is not None)
        ttk.Checkbutton(bar2, text="프레임 기록 (L)", variable=self.frames_var,
                        command=lambda: e.command("frames")).pack(side="left", padx=4)
        self.zone_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar2, text="판정 구역 보기 (O)", variable=self.zone_var, command=self.sync_zone).pack(side="left", padx=4)
        ttk.Button(bar2, text="시연 화면 (D)", command=self.toggle_demo).pack(side="left", padx=4)
        # 모델과 신뢰도 (P1)
        ttk.Label(bar2, text="   모델").pack(side="left")
        self.models = list_models([e.model_path])
        cur = next((n for n, p in self.models if os.path.abspath(p) == os.path.abspath(e.model_path)), e.model_path)
        self.model_var = tk.StringVar(value=cur)
        self.model_cb = ttk.Combobox(bar2, textvariable=self.model_var, values=[n for n, _ in self.models],
                                     width=18, state="readonly")
        self.model_cb.pack(side="left", padx=4)
        self.model_cb.bind("<<ComboboxSelected>>", lambda _: self.pick_model())
        self.find_btn = ttk.Button(bar2, text="파일 찾기…", command=self.find_model)
        self.find_btn.pack(side="left")
        ttk.Label(bar2, text="  신뢰도").pack(side="left")
        self.conf_var = tk.StringVar(value=f"{e.conf:g}")
        self.conf_cb = ttk.Combobox(bar2, textvariable=self.conf_var, values=CONF_CHOICES, width=5)
        self.conf_cb.pack(side="left", padx=4)
        self.conf_cb.bind("<<ComboboxSelected>>", lambda _: self.set_conf())
        self.conf_cb.bind("<Return>", lambda _: self.set_conf())
        if e.args.detector == "gt":
            for w in (self.model_cb, self.find_btn, self.conf_cb):
                w.configure(state="disabled")
        ttk.Button(bar2, text="끝내기 (Q)", command=self.quit).pack(side="right")
        ttk.Button(bar2, text="도움말 (?)", command=self.help.toggle).pack(side="right", padx=4)
        # 가운데: 카메라 화면 | 평면도와 정보
        mid = ttk.Panedwindow(r, orient="horizontal")
        mid.grid(row=2, column=0, sticky="nsew")
        left = ttk.Frame(mid)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        self.banner = tk.Label(left, text="모델을 불러오는 중입니다… (처음 몇 초 걸립니다)", bg="#2b5797", fg="white",
                               font=(UI_FONT, 11, "bold"), anchor="w", padx=8, pady=4, wraplength=880, justify="left")
        self.banner.grid(row=0, column=0, sticky="ew")
        self.camv = self.ImageView(left, width=900, height=520)
        self.camv.canvas.grid(row=1, column=0, sticky="nsew")
        self.camv.canvas.bind("<ButtonPress-1>", self.on_cam_click)
        mid.add(left, weight=3)
        right = ttk.Frame(mid, padding=(6, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.planv = self.ImageView(right, width=560, height=400, bg="#2d2d2d")
        self.planv.canvas.grid(row=0, column=0, sticky="nsew")
        self.planv.canvas.bind("<ButtonPress-1>", self.on_plan_click)
        leg = ttk.Frame(right)
        leg.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.count_lbls = {}
        for st in (EMPTY, OCCUPIED, DYNAMIC_BLIND, STATIC_BLIND):
            tk.Label(leg, bg=STATUS_HEX[st], width=2).pack(side="left", padx=(6, 2))
            lb = ttk.Label(leg, text=STATUS_KO[st])
            lb.pack(side="left", padx=(0, 6))
            self.count_lbls[st] = lb
        info = ttk.LabelFrame(right, text="칸 정보 (평면도나 카메라 화면의 칸을 클릭)", padding=6)
        info.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.info_title = tk.Label(info, text="칸을 클릭하세요", font=(UI_FONT, 13, "bold"), anchor="w")
        self.info_title.pack(fill="x")
        self.info_text = ttk.Label(info, text="", justify="left", wraplength=540)
        self.info_text.pack(fill="x")
        evf = ttk.LabelFrame(right, text="최근 상태 변화 (위가 최신)", padding=4)
        evf.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        self.ev_list = tk.Listbox(evf, height=7, font=(UI_FONT, 10), activestyle="none")
        self.ev_list.pack(fill="both", expand=True)
        mid.add(right, weight=2)
        # 아래: 카메라 줄
        camf = ttk.LabelFrame(r, text="카메라", padding=(6, 2))
        camf.grid(row=3, column=0, sticky="ew", padx=6)
        self.cam_rows = {}
        for c in e.cams:
            row = ttk.Frame(camf)
            row.pack(fill="x", pady=1)
            rv = tk.StringVar(value=f"{c.req[0]}x{c.req[1]}")
            cv = tk.StringVar(value=str(c.src) if c.kind == "camera" else "")
            if c.kind == "camera":
                ttk.Label(row, text="웹캠").pack(side="left")
                ttk.Combobox(row, textvariable=cv, values=[str(i) for i in range(5)], width=3,
                             state="readonly").pack(side="left", padx=(4, 8))
            if c.kind != "file":
                ttk.Label(row, text="해상도").pack(side="left")
                ttk.Combobox(row, textvariable=rv, values=RES_CHOICES, width=11).pack(side="left", padx=4)
                ttk.Button(row, text="적용", command=lambda c=c, rv=rv, cv=cv: self.change_res(c, rv, cv)).pack(side="left", padx=(0, 8))
            lb = ttk.Label(row, text=c.name)
            lb.pack(side="left")
            self.cam_rows[c.name] = (lb, rv)
        # 맨 아래 상태 줄
        self.status_var = tk.StringVar()
        ttk.Label(r, textvariable=self.status_var, style="Hint.TLabel", anchor="w").grid(row=4, column=0, sticky="ew")

    # ---------------- 모델 고르기
    def pick_model(self):
        path = dict(self.models).get(self.model_var.get())
        if path:
            self.load_model(path)
        self.focus()

    def find_model(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(title="모델 파일 고르기", initialdir=MODEL_DIR,
                                       filetypes=[("PyTorch 모델", "*.pt"), ("모든 파일", "*.*")])
        if p:
            self.models = list_models([p])
            self.model_cb.configure(values=[n for n, _ in self.models])
            self.model_var.set(next(n for n, q in self.models if os.path.abspath(q) == os.path.abspath(p)))
            self.load_model(p)
        self.focus()

    def load_model(self, path):
        """새 모델을 뒤에서 불러오고, 다 되면 엔진에 바꿔 끼운다. 그동안은 앞 모델로 계속 판정한다."""
        if self.model_loading or not self.ready:
            self.flash("모델을 불러오는 중입니다. 잠시 뒤에 다시 고르세요.")
            return
        if self.e.detector is not None and os.path.abspath(path) == self.e.detector.path:
            return
        self.model_loading = True
        self.flash(f"모델 {os.path.basename(path)} 불러오는 중… (그동안 앞 모델로 판정)")
        conf = self.e.conf

        def work():
            try:
                det = make_detector(path, conf, self.e.cfg["imgsz"])
                self.e.command("detector", det)
                self.flash(f"모델을 {os.path.basename(path)}(으)로 바꿨습니다. 클래스 {det.classes()}")
            except BaseException as ex:            # SystemExit(파일 없음)도 화면에 알린다
                self.flash(f"모델을 불러오지 못했습니다: {ex}")
            finally:
                self.model_loading = False
        threading.Thread(target=work, daemon=True).start()

    def set_conf(self):
        try:
            v = float(self.conf_var.get())
        except ValueError:
            v = self.e.conf
        v = max(0.05, min(0.95, v))
        self.conf_var.set(f"{v:g}")
        self.e.command("conf", v)
        self.focus()

    # ---------------- 동작
    def focus(self):
        self.camv.canvas.focus_set()

    def toggle_pause(self):
        self.e.set_pause(not self.e.paused)
        self.pause_btn.configure(text="▶ 계속 (Space)" if self.e.paused else "⏸ 일시정지 (Space)")

    def set_speed(self):
        self.e.set_speed(dict(SPEEDS)[self.speed_var.get()])
        self.focus()

    def set_view(self):
        self.e.command("view_all", self.view_all_var.get())
        self.focus()

    def set_every(self, delta=0):
        try:
            n = int(self.every_var.get()) + delta
        except ValueError:
            n = self.e.every_n
        n = max(1, min(120, n))
        self.every_var.set(str(n))
        self.e.command("every", n)

    def set_camview(self):
        v = self.camview_var.get()
        self.view_idx = -1 if v == "모두" else [c.name for c in self.e.cams].index(v)
        self.focus()

    def cycle_camview(self):
        n = len(self.e.cams)
        if n > 1:
            self.view_idx = (self.view_idx + 2) % (n + 1) - 1
            self.camview_var.set("모두" if self.view_idx < 0 else self.e.cams[self.view_idx].name)

    def sync_zone(self):
        self.show_zone = self.zone_var.get()
        self.focus()

    def change_res(self, c, rv, cv=None):
        """웹캠 번호·해상도를 바꿔 그 카메라를 다시 연다(이번 실행에만, 설정 파일은 그대로)."""
        from parkit_tk import parse_res
        try:
            size = parse_res(rv.get())
        except ValueError:
            self.mb.showwarning("해상도", "해상도는 1920x1080 처럼 적으세요.")
            return
        cam = int(cv.get()) if cv is not None and cv.get() != "" else None
        if cam is not None and c.kind == "camera" and cam != int(c.src):
            if not self.mb.askyesno("웹캠 바꾸기", f"{c.name}의 입력을 웹캠 {cam}번으로 바꿉니다. {c.name} 보정은 "
                                    f"웹캠 {c.src}번 자리에서 한 것이라, 다른 카메라면 칸 선이 맞지 않습니다. 계속할까요?"):
                cv.set(str(c.src))
                return
        cal = tuple(c.cal["calib_size"])
        if abs(size[0] / size[1] - cal[0] / cal[1]) > 0.01 * cal[0] / cal[1]:
            if not self.mb.askyesno("해상도", f"{c.name}은 {cal[0]}x{cal[1]}로 보정했습니다. 화면 비율이 다른 해상도로 바꾸면 "
                                    f"다시 보정할 때까지 이 카메라는 멈춥니다. 그래도 바꿀까요?"):
                return
        self.e.command("resolution", c.name, size, cam)
        self.flash(f"{c.name}: " + (f"웹캠 {cam}번, " if cam is not None else "") + f"{size[0]}x{size[1]}로 다시 여는 중…")
        self.focus()

    def screenshot(self):
        """L3: 평면도, 카메라 화면(칸 선·상자 포함)."""
        e = self.e
        stamp = datetime.datetime.now().strftime("%H%M%S")
        st = e.status_map()
        save_image(os.path.join(e.logs.dir, f"shot_{stamp}_plan.png"), PlanView(e.table, 12).render(st, self.plan_points()))
        for c in e.cams:
            f = c.display if e.view_all and c.display is not None else c.frame
            if f is not None and c.geom is not None and (f.shape[1], f.shape[0]) == tuple(c.geom.size):
                im = f.copy()
                draw_slots(im, c.geom, st, 1.0, 2)
                if c.res is not None:
                    draw_boxes(im, c.res)
                save_image(os.path.join(e.logs.dir, f"shot_{stamp}_{c.name}.jpg"), im)
        shot = os.path.join(e.logs.dir, f"shot_{stamp}_*")
        self.flash(f"화면을 저장했습니다: {os.path.join(os.path.basename(e.logs.dir), os.path.basename(shot))}")
        print(f"[판정] 화면 저장: {shot}")

    def flash(self, msg):
        self.flash_msg = (msg, time.time())

    def toggle_demo(self):
        from parkit_tk import STATUS_HEX, STATUS_KO, UI_FONT
        tk = self.tk
        if self.demo is not None and self.demo["win"].winfo_exists():
            self.demo["win"].destroy()
            self.demo = None
            return
        w = tk.Toplevel(self.root)
        w.title("PARKIT 주차 현황")
        w.geometry("1280x820")
        top = tk.Frame(w, bg="#1e1e1e")
        top.pack(fill="x")
        lbls = {}
        for st in (EMPTY, OCCUPIED, DYNAMIC_BLIND, STATIC_BLIND):
            tk.Label(top, bg=STATUS_HEX[st], width=3).pack(side="left", padx=(16, 6), pady=10)
            lb = tk.Label(top, text="", fg="white", bg="#1e1e1e", font=(UI_FONT, 20, "bold"))
            lb.pack(side="left")
            lbls[st] = lb
        tk.Label(top, text="F11 전체 화면 · Esc 닫기", fg="#aaaaaa", bg="#1e1e1e", font=(UI_FONT, 10)).pack(side="right", padx=10)
        view = self.ImageView(w, width=1280, height=760, bg="#1e1e1e")
        view.canvas.pack(fill="both", expand=True)
        w.bind("<F11>", lambda _: w.attributes("-fullscreen", not w.attributes("-fullscreen")))
        w.bind("<Escape>", lambda _: self.toggle_demo())
        w.protocol("WM_DELETE_WINDOW", self.toggle_demo)
        self.demo = {"win": w, "view": view, "lbls": lbls}

    def quit(self):
        self.e.running = False
        time.sleep(0.15)
        if self.ready:
            self.e.close()
        if getattr(self, "after_id", None):
            self.root.after_cancel(self.after_id)     # 닫은 뒤 화면 갱신이 남지 않게
        self.root.destroy()

    # ---------------- 마우스, 키보드
    def on_plan_click(self, ev):
        self.focus()
        if self.plan is None:
            return
        p = self.planv.to_src(ev.x, ev.y)
        if p is None:
            return
        no = self.e.table.slot_at(self.plan.to_mat(p))
        if no is not None:
            self.selected = no

    def on_cam_click(self, ev):
        self.focus()
        p = self.camv.to_src(ev.x, ev.y)
        if p is None:
            return
        for c, x0, y0, s, w, h in self.layout:
            if x0 <= p[0] < x0 + w and y0 <= p[1] < y0 + h and c.geom is not None:
                u, v = (p[0] - x0) / s, (p[1] - y0) / s
                for no, fl in c.geom.floor_px.items():
                    if cv2.pointPolygonTest(fl.astype(np.float32), (float(u), float(v)), False) >= 0:
                        self.selected = no
                        return

    def on_key(self, ev):
        from parkit_tk import is_typing, key_of
        if is_typing(ev.widget):
            return
        k = key_of(ev)
        e = self.e
        if k in ("q", "Escape"):
            self.quit()
        elif k in ("?", "F1"):
            self.help.toggle()
        elif k == "space":
            self.toggle_pause()
            return "break"
        elif k == "f":
            self.view_all_var.set(not self.view_all_var.get())
            self.set_view()
        elif k in ("[", "]"):
            self.set_every(-1 if k == "[" else 1)
        elif k == "v":
            self.cycle_camview()
        elif k == "r":
            e.command("record")
        elif k == "s":
            self.screenshot()
        elif k == "l":
            self.frames_var.set(not self.frames_var.get())
            e.command("frames")
        elif k == "o":
            self.zone_var.set(not self.zone_var.get())
            self.sync_zone()
        elif k == "d":
            self.toggle_demo()

    # ---------------- 그리기
    def plan_points(self):
        pts = []
        for c in self.e.cams:
            if c.res is None or c.state != "ok":
                continue
            for b in c.res.boxes:
                pts.append((b.contact_mat, (40, 40, 230) if b.slot is not None else (255, 160, 0)))
        return pts

    def tick(self):
        try:
            self.e.main_thread_work()          # 웹캠 바꾸기, (맥) 끊긴 웹캠 다시 열기
            self.draw()
        except Exception:
            import traceback
            traceback.print_exc()
        if self.e.finished and self.ready and not self.end_noted:
            self.end_noted = True
            self.flash("영상이 끝났습니다. 결과를 보고 끝내기(Q)를 누르세요.")
        self.after_id = self.root.after(40, self.tick)

    def draw(self):
        from parkit_tk import OBS_KO, REASON_KO, STATUS_HEX, STATUS_KO
        e = self.e
        if self.error:
            self.banner.configure(text=f"시작하지 못했습니다: {self.error}", bg="#c00000")
            return
        if not self.ready:
            return
        ov = float(e.cfg["display"].get("startup_overlay_s", 5.0))
        startup = e.t < ov
        fm = self.flash_msg
        if fm and time.time() - fm[1] < 4:
            self.banner.configure(text=fm[0], bg="#2b5797")
        elif startup:
            self.banner.configure(text=f"켤 때 칸 선 확인 ({max(0.0, ov - e.t):.0f}초): 굵은 칸 선이 매트에 인쇄된 선과 맞는지 "
                                       f"보세요. 어긋나면 카메라가 밀린 것이니 다시 보정하세요.", bg="#b8860b")
        elif e.paused:
            self.banner.configure(text="일시정지 중입니다. Space 또는 [▶ 계속]을 누르세요.", bg="#555555")
        elif e.finished:
            self.banner.configure(text="영상이 끝났습니다. 결과를 보고 끝내기(Q)를 누르세요.", bg="#555555")
        else:
            self.banner.configure(text="판정 중  ·  칸을 클릭하면 자세한 정보가 나옵니다  ·  도움말: ? 키", bg="#2b5797")
        st = e.status_map()
        # 카메라 화면 (D1, D5)
        show = e.cams if self.view_idx < 0 else [e.cams[self.view_idx]]
        W, H = self.camv.avail()
        cell_h = max(60, H // len(show))
        tiles, self.layout, y = [], [], 0
        for c in show:
            f = c.display if (e.view_all and c.display is not None) else c.frame
            if f is None or c.geom is None:
                tw_ = min(W, int(cell_h * 16 / 9))
                tile = np.full((min(cell_h, int(tw_ * 9 / 16)), tw_, 3), 25, np.uint8)
                msg = {"stopped": f"{c.name}: STOPPED", "offline": f"{c.name}: NO FRAME"}.get(c.state, f"{c.name}: waiting")
                text(tile, msg, (12, 30), 0.7, (0, 200, 255), 2)
                self.layout.append((c, 0, y, 1.0, 0, 0))
            else:
                fh, fw = f.shape[:2]
                s = min(W / fw, cell_h / fh)
                tile = cv2.resize(f, (max(1, int(fw * s)), max(1, int(fh * s))), interpolation=cv2.INTER_AREA)
                if (fw, fh) == tuple(c.geom.size):
                    draw_slots(tile, c.geom, st, s, thick=4 if startup else 2, show_zone=self.show_zone,
                               show_region=self.show_zone, params=e.params)
                    if c.res is not None:
                        draw_boxes(tile, c.res, s)
                text(tile, c.name, (8, 22), 0.6, (255, 255, 255), 1)
                if c.source is not None and c.source.recording():
                    cv2.circle(tile, (tile.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                self.layout.append((c, 0, y, s, tile.shape[1], tile.shape[0]))
            tiles.append(tile)
            y += tile.shape[0]
        tw = max(t.shape[1] for t in tiles)
        canvas = np.full((sum(t.shape[0] for t in tiles), tw, 3), 25, np.uint8)
        y = 0
        for t in tiles:
            canvas[y:y + t.shape[0], :t.shape[1]] = t
            y += t.shape[0]
        self.camv.show(canvas, 1.0, (canvas.shape[1], canvas.shape[0]))
        # 평면도 (D2, 창 크기에 맞춰 다시 만든다)
        PW, PH = self.planv.avail()
        mw, mh = e.table.mat_size
        k = max(3.0, min((PW - 16) / mw, (PH - 16) / mh))
        if self.plan is None or abs(k - self.plan_k) > 0.05:
            self.plan, self.plan_k = PlanView(e.table, k, margin=8), k
        cams_pos = []
        for c in e.cams:
            p = (c.cal.get("pose") or {}).get("cam_pos_cm")
            if p:
                cams_pos.append((c.name, p[0], p[1]))
        pim = self.plan.render(st, self.plan_points(), self.selected, cams_pos)
        self.planv.show(pim, 1.0, (pim.shape[1], pim.shape[0]))
        cnt = e.tracker.counts()
        for s_, lb in self.count_lbls.items():
            lb.configure(text=f"{STATUS_KO[s_]} {cnt[s_]}")
        # 칸 정보 (D3)
        if self.selected is not None:
            s = e.tracker.slots[self.selected]
            p = e.params(self.selected)
            rem = s.remaining(e.t, p)
            self.info_title.configure(text=f"{s.no}번 칸 · {STATUS_KO[s.status]}",
                                      fg="#b8860b" if s.status == DYNAMIC_BLIND else STATUS_HEX[s.status])
            lines = []
            if s.reason:
                lines.append(f"이유: {REASON_KO.get(s.reason, s.reason)}")
            if s.last_known:
                lines.append(f"확인 불가가 되기 전 상태: {STATUS_KO[s.last_known]}")
            if s.cams:
                lines.append(f"이번 판정: {OBS_KO.get(s.obs, s.obs)}")
                if rem is not None:
                    tgt = {"present": "주차됨", "absent": "빈 칸", "occluded": "확인 불가"}.get(s.obs, "")
                    lines.append(f"이대로 {rem:.1f}초 더 이어지면 → {tgt}")
                lines.append("본 카메라: " + (", ".join(f"{cn} {OBS_KO.get(o, o)}" for cn, o in s.obs_cams.items()) or "아직 없음"))
                if s.why:
                    lines.append(f"판정 근거: {s.why}")
            else:
                lines.append("어느 카메라도 감시하지 않는 칸입니다(감시 칸은 보정 도구에서 정함).")
            self.info_text.configure(text="\n".join(lines))
        # 최근 상태 변화
        if e.n_events != self.shown_events:
            new = min(e.n_events - self.shown_events, len(e.events))
            for ev in list(e.events)[-new:]:
                line = f"{ev['time']:7.1f}초  {ev['slot']:2d}번  {STATUS_KO.get(ev['prev'], ev['prev'])} → {STATUS_KO[ev['status']]}"
                if ev["reason"]:
                    line += f"  ({REASON_KO.get(ev['reason'], ev['reason'])})"
                self.ev_list.insert(0, line)
                self.ev_list.itemconfig(0, foreground={OCCUPIED: "#c00000", EMPTY: "#1a7f1a",
                                                       DYNAMIC_BLIND: "#9a7400"}.get(ev["status"], "#555555"))
            self.shown_events = e.n_events
            if self.ev_list.size() > 300:
                self.ev_list.delete(300, "end")
        # 카메라 줄
        for c in e.cams:
            lb, rv = self.cam_rows[c.name]
            size = f"{c.size[0]}x{c.size[1]}" if c.size else "-"
            state = {"ok": "정상", "offline": "프레임 없음", "stopped": "멈춤"}.get(c.state, c.state)
            parts = [c.name, c.describe(), size, f"{c.fps():.0f}fps", f"추론 {c.ms:.0f}ms", state]
            if c.note:
                parts.append(c.note)
            if c.source is not None and c.source.recording():
                parts.append("● 녹화 중")
            lb.configure(text="  ·  ".join(parts), foreground="#c00000" if c.state != "ok" else "#000000")
        # 도구 막대와 상태 줄
        fps = max([c.fps() for c in e.cams] + [1.0])
        self.every_lbl.configure(text=f"프레임마다 분석  (약 초당 {fps / e.every_n:.1f}장)")
        self.rec_btn.configure(text="■ 녹화 멈추기 (R)" if e.recording() else "● 녹화 (R)")
        clock = "영상 시각" if e.all_files else "실제 시각"
        det = e.detector
        model = "정답 상자(시험)" if det is not None and det.path == "gt" else \
            f"{os.path.basename(det.path if det else e.model_path)} · 신뢰도 {getattr(det, 'conf', e.conf):g}"
        self.status_var.set(f"시각 {e.t:7.1f}초 ({clock})   ·   모델 {model}{' (바꾸는 중)' if self.model_loading else ''}"
                            f"   ·   기록 폴더: {e.logs.dir}"
                            f"{'   ·   frames.csv 기록 중' if e.logs.w_frames is not None else ''}   ·   도움말: ? 키")
        # 시연 화면 (D6)
        if self.demo is not None and self.demo["win"].winfo_exists():
            dv = self.demo["view"]
            DW, DH = dv.avail()
            big = PlanView(e.table, max(3.0, min((DW - 20) / mw, (DH - 20) / mh)), margin=10)
            im = big.render(st, [], None, cams_pos)
            dv.show(im, 1.0, (im.shape[1], im.shape[0]))
            for s_, lb in self.demo["lbls"].items():
                lb.configure(text=f"{STATUS_KO[s_]} {cnt[s_]}")


def main():
    ap = argparse.ArgumentParser(description="PARKIT 점유 판정 프로그램")
    ap.add_argument("--run", default=DEFAULT_RUN, help="실행 설정 json")
    ap.add_argument("--preset", default=None, help="실행 설정의 presets 중 하나 (예: demo)")
    ap.add_argument("--source", action="append", help="카메라 입력 바꾸기: 이름=웹캠번호|영상|rtsp (여러 번 가능)")
    ap.add_argument("--model", default=None, help="모델 파일 (기본: 실행 설정의 model, M8.pt)")
    ap.add_argument("--conf", type=float, default=None, help="신뢰도 (기본: 실행 설정, 0.25)")
    ap.add_argument("--every", type=int, default=None, help="N프레임마다 분석 (기본: 실행 설정 analyze_every_frames)")
    ap.add_argument("--detector", choices=["model", "gt"], default="model", help="gt = 시험용 정답 상자")
    ap.add_argument("--speed", type=float, default=1.0, help="영상 파일 재생 속도 (1 = 실제 속도, 0 = 최대한 빨리)")
    ap.add_argument("--duration", type=float, default=0, help="이 시간(초)이 지나면 끝")
    ap.add_argument("--log-frames", action="store_true", help="처음부터 프레임 기록(frames.csv) 켜기")
    ap.add_argument("--no-window", action="store_true", help="창 없이 실행 (시험용)")
    ap.add_argument("--quiet", action="store_true", help="상태 변화를 콘솔에 찍지 않음")
    ap.add_argument("--tracking", action="store_true", help="매트 좌표 기반 차량 ID 추적 (기존 점유 판정과 분리)")
    ap.add_argument("--web", action="store_true", help="차량 추적과 2D 웹 API 켜기")
    ap.add_argument("--web-host", default="127.0.0.1", help="API 바인드 주소")
    ap.add_argument("--web-port", type=int, default=8765, help="API 포트")
    ap.add_argument("--tracking-config", default=None, help="선택적 차량 추적 파라미터 JSON")
    args = ap.parse_args()
    if args.no_window:
        e = Engine(args, headless=True)
        try:
            e.load_detector()
            e.start()
            e.run()
        finally:
            e.close()
        return
    from parkit_tk import check_tk
    check_tk()
    app = OccApp(Engine(args))
    app.root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
