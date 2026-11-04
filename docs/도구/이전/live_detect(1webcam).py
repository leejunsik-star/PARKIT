"""실시간 차량 탐지 모델 확인 프로그램 (PARKIT 테스트베드용)

웹캠 화면을 받아 고른 모델로 계속 분석하고, 결과를 화면(상자)과 결과 창(글)에 바로 낸다.
모델, 입력 크기, 신뢰도 기준, 웹캠/영상 선택 등 모든 옵션은 창 오른쪽 패널에서 마우스로 바꾼다.

설치(한 번만):
    pip install ultralytics
    pip install "opencv-python==4.10.0.84"      (OpenCV 5.0은 이 웹캠을 못 열었음)
    pip install effdet                           (EfficientDet을 쓸 때만)

사용법:
    python live_detect.py                       창이 뜨고 웹캠 0번, YOLO26n으로 시작
    python live_detect.py --cam 1               시작 웹캠 번호
    python live_detect.py --model rtdetr        시작 모델 (아래 목록의 키 이름)
    python live_detect.py --source 영상.mp4      웹캠 대신 녹화 영상으로 시작
    python live_detect.py --rate 1 --threads 2  초당 1회, CPU 코어 2개로 분석 (CPU 사용 줄이기)
    python live_detect.py --nogui               창 없이 콘솔 출력만 (영상/웹캠 자동 분석)

모델 파일은 처음 고를 때 자동으로 내려받는다(인터넷 필요, 실행한 폴더에 저장).
"""
import argparse, csv, os, platform, queue, sys, threading, time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 키, 화면 이름, 종류, 가중치 파일, 기본 입력 크기
MODELS = [
    ("1", "YOLO26n",          "coco", "yolo26n.pt",       640),
    ("2", "YOLO26s",          "coco", "yolo26s.pt",       1280),
    ("3", "YOLO26m",          "coco", "yolo26m.pt",       1280),
    ("4", "YOLO11n",          "coco", "yolo11n.pt",       1280),
    ("5", "YOLO11s",          "coco", "yolo11s.pt",       1280),
    ("6", "YOLO11m",          "coco", "yolo11m.pt",       1280),
    ("7", "YOLOv8n",          "coco", "yolov8n.pt",       640),
    ("8", "YOLOv8s",          "coco", "yolov8s.pt",       1280),
    ("9", "YOLOv8m",          "coco", "yolov8m.pt",       1280),
    ("0", "RT-DETR-L",        "rtdetr", "rtdetr-l.pt",    1280),
    ("a", "OpenImages-v8m",   "oiv7", "yolov8m-oiv7.pt",  640),
    ("b", "YOLOE-11s",        "yoloe", "yoloe-11s-seg.pt", 640),
    ("c", "YOLOE-11l",        "yoloe", "yoloe-11l-seg.pt", 1280),
    ("d", "EfficientDet-D2",  "effdet", "tf_efficientdet_d2", 768),
]
ALIASES = {"yolo26n": "1", "yolo26s": "2", "yolo26m": "3", "yolo11n": "4", "yolo11s": "5",
           "yolo11m": "6", "yolov8n": "7", "yolov8s": "8", "yolov8m": "9", "rtdetr": "0",
           "oiv7": "a", "openimages": "a", "yoloe-s": "b", "yoloe-l": "c", "effdet": "d"}
MODEL_NOTE = {"1": "가장 빠름", "0": "가장 정확, 무거움", "a": "검정 차에 강함", "b": "글 프롬프트",
              "c": "글 프롬프트", "d": "effdet 설치 필요"}
SIZES = [320, 480, 640, 768, 960, 1280]
# 분석 빈도(초당 횟수). 0은 제한 없이 끝나는 대로 바로 다음 분석
RATES = [("제한 없음", 0), ("초당 10회", 10), ("초당 5회", 5), ("초당 2회", 2), ("초당 1회", 1),
         ("2초에 1회", 0.5), ("5초에 1회", 0.2)]
CPU_COUNT = os.cpu_count() or 4

# 차량으로 셀 클래스 이름 (모델 종류별)
VEHICLE = {
    "coco": {"car", "bus", "truck"},
    "rtdetr": {"car", "bus", "truck"},
    "oiv7": {"Car", "Land vehicle", "Vehicle", "Toy", "Van", "Truck"},
    "effdet": {"car", "bus", "truck"},
}
COCO91 = {1: "person", 3: "car", 6: "bus", 8: "truck"}  # EfficientDet은 91개 COCO 번호를 씀


def iou(a, b):
    x1, y1 = max(a[2], b[2]), max(a[3], b[3])
    x2, y2 = min(a[4], b[4]), min(a[5], b[5])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda d: (d[4] - d[2]) * (d[5] - d[3])
    return inter / (area(a) + area(b) - inter + 1e-6)


def merge_vehicles(dets, is_vehicle):
    """한 차에 car, truck 상자가 겹쳐 나오면 신뢰도가 높은 하나만 남긴다."""
    veh = sorted((d for d in dets if is_vehicle(d[0])), key=lambda d: -d[1])
    keep = []
    for d in veh:
        if all(iou(d, k) < 0.6 for k in keep):
            keep.append(d)
    return keep + [d for d in dets if not is_vehicle(d[0])]


def device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", platform.processor() or "CPU"


class Detector:
    """모델 하나를 감싸서 run(frame) -> [(이름, 신뢰도, x1, y1, x2, y2)] 형태로 통일한다."""

    def __init__(self, kind, weight, prompt):
        self.kind = kind
        self.prompt = list(prompt)
        dev, _ = device_name()
        self.dev = dev
        if kind == "effdet":
            import torch
            from effdet import create_model
            self.torch = torch
            self.m = create_model(weight, bench_task="predict", pretrained=True).eval().to(dev)
            self.size = 768
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
            return
        path = os.path.join(HERE, weight)
        src = path if os.path.exists(path) else weight
        if kind == "rtdetr":
            from ultralytics import RTDETR
            self.m = RTDETR(src)
        elif kind == "yoloe":
            from ultralytics import YOLOE
            self.m = YOLOE(src)
            self.set_prompt(self.prompt)
        else:
            from ultralytics import YOLO
            self.m = YOLO(src)

    def set_prompt(self, prompt):
        self.prompt = list(prompt)
        self.m.set_classes(self.prompt, self.m.get_text_pe(self.prompt))

    def is_vehicle(self, name):
        if self.kind == "yoloe":
            return True  # 프롬프트로 고른 대상이 곧 찾는 물체
        return name in VEHICLE[self.kind]

    def run(self, im, imgsz, conf):
        if self.kind == "effdet":
            return self._run_effdet(im, conf)
        r = self.m.predict(im, imgsz=imgsz, conf=conf, verbose=False)[0]
        names = self.prompt if self.kind == "yoloe" else r.names
        out = []
        for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            out.append((names[int(c)], float(p), *b))
        return out

    def _run_effdet(self, im, conf):
        h, w = im.shape[:2]
        s = self.size / max(h, w)
        rs = cv2.resize(im, (int(w * s), int(h * s)))
        pad = np.zeros((self.size, self.size, 3), np.uint8)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        x = (pad[:, :, ::-1] / 255.0 - self.mean) / self.std
        x = self.torch.tensor(x.transpose(2, 0, 1), dtype=self.torch.float32)[None].to(self.dev)
        with self.torch.no_grad():
            d = self.m(x)[0].cpu().numpy()
        out = []
        for x1, y1, x2, y2, p, c in d:
            if p < conf:
                continue
            out.append((COCO91.get(int(c), f"coco{int(c)}"), float(p), x1 / s, y1 / s, x2 / s, y2 / s))
        return out


class Camera(threading.Thread):
    """항상 가장 최근 프레임만 들고 있는다. 무거운 모델을 써도 화면이 밀리지 않게 하기 위함.
    영상 파일이면 재생/정지, 위치 이동(seek), 반복 재생을 지원한다."""

    def __init__(self, cap, is_file, fps):
        super().__init__(daemon=True)
        self.cap, self.is_file = cap, is_file
        self.fps = fps if fps and fps > 0 else 25
        self.delay = 1.0 / self.fps if is_file else 0
        self.total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if is_file else 0
        self.pos = -1           # 화면에 보이는 프레임 번호 (영상 파일)
        self.playing = True
        self.loop = True
        self.seek_to = None
        self.frame, self.n, self.alive = None, 0, True
        self.lock = threading.Lock()

    def _store(self, f, pos):
        with self.lock:
            self.frame, self.n, self.pos = f, self.n + 1, pos

    def seek(self, frame_no):
        if self.is_file:
            with self.lock:
                self.seek_to = int(frame_no)

    def where(self):
        """이동 중이면 이동할 위치, 아니면 지금 위치. 버튼을 빠르게 여러 번 눌러도 누적되게 하기 위함."""
        with self.lock:
            return self.pos if self.seek_to is None else self.seek_to

    def step(self, delta):
        """정지 상태에서 프레임 단위로 앞뒤 이동."""
        self.playing = False
        self.seek(self.where() + delta)

    def toggle(self):
        if not self.playing and self.pos >= self.total - 1:
            self.seek(0)  # 끝에서 다시 누르면 처음부터
        self.playing = not self.playing

    def run(self):
        t_next = time.time()
        while self.alive:
            if self.is_file:
                with self.lock:
                    target, self.seek_to = self.seek_to, None
                if target is not None:
                    target = min(max(0, target), self.total - 1)
                    if target != self.pos + 1:  # 바로 다음 프레임이면 그냥 읽는 것이 빠름
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ok, f = self.cap.read()
                    if ok:
                        self._store(f, target)
                    t_next = time.time() + self.delay
                    continue
                if not self.playing:
                    time.sleep(0.02)
                    continue
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    if self.loop:  # 영상이 끝나면 처음부터 다시
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.pos = -1
                    else:
                        self.playing = False
                    continue
                time.sleep(0.01)
                continue
            self._store(f, self.pos + 1 if self.is_file else 0)
            if self.delay:  # 영상 원래 속도에 맞춰 재생
                t_next += self.delay
                wait = t_next - time.time()
                if wait > 0:
                    time.sleep(wait)
                else:
                    t_next = time.time()
        self.cap.release()

    def latest(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame.copy(), self.n)


def open_camera(idx, w, h):
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)] \
        if platform.system() == "Windows" else [("ANY", cv2.CAP_ANY)]
    for name, be in backends:
        cap = cv2.VideoCapture(idx, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, f = cap.read()
        if ok:
            return cap, f"카메라 {idx}번 열림 ({name}), {f.shape[1]}x{f.shape[0]}"
        cap.release()
    return None, f"카메라 {idx}번을 열지 못함. 카메라 앱 등 다른 프로그램을 닫거나 번호를 바꿔 보세요."


def open_source(source, cam_idx, w, h):
    """영상 파일 또는 웹캠을 열어 Camera 스레드를 돌려준다. 실패하면 (None, 메시지)."""
    if source:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None, f"영상을 열지 못함: {source}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cam = Camera(cap, True, fps)
        msg = f"영상 열림: {os.path.basename(source)}"
    else:
        cap, msg = open_camera(cam_idx, w, h)
        if cap is None:
            return None, msg
        cam = Camera(cap, False, 0)
    cam.start()
    return cam, msg


def draw(im, dets, det_obj, show_all):
    for name, p, x1, y1, x2, y2 in dets:
        veh = det_obj.is_vehicle(name)
        if not veh and not show_all:
            continue
        col = (0, 220, 0) if veh else (0, 160, 255)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(im, p1, p2, col, 3)
        txt = f"{name} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y = max(th + 6, p1[1])
        cv2.rectangle(im, (p1[0], y - th - 6), (p1[0] + tw + 6, y), col, -1)
        cv2.putText(im, txt, (p1[0] + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


class Engine:
    """카메라와 모델 분석을 돌리는 부분. 화면(GUI)과 분리되어 있어 옵션은 state로만 바꾼다."""

    def __init__(self, keymap, start_key, conf, prompt, on_result, on_message, rate=2, threads=0):
        self.keymap = keymap
        self.state = {"key": start_key, "imgsz": keymap[start_key][3], "conf": conf,
                      "show_all": False, "prompt": list(prompt), "paused": False,
                      "rate": rate, "threads": threads or max(1, CPU_COUNT // 2)}
        self.applied_threads = None
        self.on_result, self.on_message = on_result, on_message
        self.cache = {}
        self.cam = None
        self.lock = threading.Lock()
        self.shared = {"dets": [], "ms": 0.0, "det": None, "fps": 0.0, "loading": "", "n": 0}
        self.stop = threading.Event()
        self.writer = self.logf = None
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def set_source(self, source, cam_idx, w, h):
        old, self.cam = self.cam, None
        if old:
            old.alive = False
        cam, msg = open_source(source, cam_idx, w, h)
        self.cam = cam
        with self.lock:
            self.shared.update(dets=[], det=None)
        self.on_message(msg)
        return cam is not None

    def set_log(self, on):
        if on and not self.writer:
            path = os.path.join(HERE, f"live_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
            self.logf = open(path, "w", newline="", encoding="utf-8-sig")
            self.writer = csv.writer(self.logf)
            self.writer.writerow(["시각", "모델", "입력크기", "신뢰도기준", "추론ms", "차량수", "검출(이름:신뢰도)"])
            self.on_message(f"기록 시작: {path}")
        elif not on and self.writer:
            self.writer = None
            self.logf.close()
            self.on_message("기록 끝")

    def _load(self, key):
        name, kind, w, _ = self.keymap[key]
        det = self.cache.get(key)
        if det is None:
            with self.lock:
                self.shared["loading"] = name
            self.on_message(f"모델 불러오는 중: {name} (처음이면 내려받기 때문에 시간이 걸림)")
            try:
                det = Detector(kind, w, self.state["prompt"])
            except Exception as e:
                self.on_message(f"{name} 불러오기 실패: {e}")
                return None
            finally:
                with self.lock:
                    self.shared["loading"] = ""
            self.cache[key] = det
            self.on_message(f"준비됨: {name}")
        if det.kind == "yoloe" and det.prompt != self.state["prompt"]:
            det.set_prompt(self.state["prompt"])
            self.on_message(f"YOLOE 프롬프트 적용: {', '.join(det.prompt)}")
        return det

    def _apply_threads(self, n):
        """모델 계산에 쓸 CPU 코어 수. 줄이면 CPU 사용률이 내려가는 대신 한 번 분석이 느려진다."""
        try:
            import torch
            # 모델을 불러올 때 라이브러리가 값을 되돌릴 수 있어 분석 직전마다 확인한다
            if torch.get_num_threads() != n:
                torch.set_num_threads(n)
        except Exception as e:
            if n != self.applied_threads:
                self.on_message(f"코어 수 설정 실패: {e}")
            self.applied_threads = n
            return
        if n != self.applied_threads:
            self.applied_threads = n
            self.on_message(f"CPU 코어 {n}개 사용")

    def _wait_until(self, t_due):
        """다음 분석 시각까지 쉬되, 빈도를 바꾸거나 끝낼 때는 바로 반응한다."""
        rate = self.state["rate"]
        while not self.stop.is_set() and time.time() < t_due:
            if self.state["rate"] != rate or self.state["paused"]:
                return
            time.sleep(min(0.05, max(0.0, t_due - time.time())))

    def _worker(self):
        last_n, t_prev, t_start = -1, time.time(), 0.0
        while not self.stop.is_set():
            st = dict(self.state)
            cam = self.cam
            if st["paused"] or cam is None:
                time.sleep(0.05)
                continue
            self._apply_threads(st["threads"])
            if st["rate"] > 0:
                self._wait_until(t_start + 1.0 / st["rate"])
                if self.stop.is_set():
                    break
                st = dict(self.state)
            det = self._load(st["key"])
            if det is None:  # 실패하면 YOLO26n으로 돌아감
                self.state["key"] = "1"
                self.state["imgsz"] = self.keymap["1"][3]
                self.on_message("YOLO26n으로 돌아감")
                continue
            im, n = cam.latest()
            if im is None or n == last_n:
                time.sleep(0.005)
                continue
            last_n = n
            self._apply_threads(st["threads"])
            t = t_start = time.time()
            try:
                dets = merge_vehicles(det.run(im, st["imgsz"], st["conf"]), det.is_vehicle)
            except Exception as e:
                self.on_message(f"분석 오류: {e}")
                time.sleep(0.5)
                continue
            ms = (time.time() - t) * 1000
            now = time.time()
            inf_fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            veh = [d for d in dets if det.is_vehicle(d[0])]
            with self.lock:
                self.shared.update(dets=dets, ms=ms, det=det, fps=inf_fps, n=len(veh))
            name = self.keymap[st["key"]][0]
            items = ", ".join(f"{d[0]}:{d[1]:.2f}" for d in veh)
            others = [d for d in dets if not det.is_vehicle(d[0])]
            extra = f"  (기타: {', '.join(f'{d[0]}:{d[1]:.2f}' for d in others[:3])})" if others else ""
            line = f"{datetime.now():%H:%M:%S} [{name} {st['imgsz']}] {ms:6.0f}ms  차량 {len(veh)}대  {items}{extra}"
            self.on_result(line)
            if self.writer:
                self.writer.writerow([datetime.now().isoformat(timespec="milliseconds"), name, st["imgsz"],
                                      f"{st['conf']:.2f}", round(ms), len(veh),
                                      "; ".join(f"{d[0]}:{d[1]:.2f}" for d in dets)])

    def frame(self):
        """그릴 프레임과 현재 결과를 돌려준다."""
        cam = self.cam
        im, _ = cam.latest() if cam else (None, 0)
        with self.lock:
            sh = dict(self.shared)
        if im is not None and sh["det"] is not None:
            draw(im, sh["dets"], sh["det"], self.state["show_all"])
        return im, sh

    def close(self):
        self.stop.set()
        if self.cam:
            self.cam.alive = False
        self.th.join(timeout=3)
        self.set_log(False)


def run_gui(a, keymap, start, prompt):
    import tkinter as tk
    from tkinter import ttk, filedialog
    from PIL import Image, ImageTk

    msgs = queue.Queue()
    root = tk.Tk()
    root.title("PARKIT 실시간 차량 탐지 확인")
    root.geometry("1360x780")

    def post(line):
        print(line)
        msgs.put(line)

    eng = Engine(keymap, start, a.conf, prompt, post, post, a.rate, a.threads)

    # ---- 오른쪽: 옵션 패널 (먼저 배치해야 폭이 줄지 않음) ----
    panel = ttk.Frame(root, width=360)
    panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
    panel.pack_propagate(False)

    # ---- 왼쪽: 영상 ----
    left = ttk.Frame(root)
    left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    stat = tk.StringVar(value="")
    stat_lbl = ttk.Label(left, textvariable=stat, font=("맑은 고딕", 11, "bold"))
    stat_lbl.pack(side="bottom", anchor="w", pady=(6, 0))

    # 영상 파일일 때만 보이는 재생 막대
    vbar = ttk.Frame(left)
    play_txt = tk.StringVar(value="정지")
    pos_var = tk.DoubleVar(value=0)
    time_txt = tk.StringVar(value="")
    loop_var = tk.BooleanVar(value=True)
    drag = {"on": False}

    def vcam():
        c = eng.cam
        return c if c is not None and c.is_file else None

    def fmt(sec):
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

    def toggle_play():
        c = vcam()
        if c:
            c.toggle()

    def seek_rel(delta_frames):
        c = vcam()
        if c:
            c.seek(c.where() + delta_frames)

    def step(delta):
        c = vcam()
        if c:
            c.step(delta)

    def on_drag_end(_):
        drag["on"] = False
        c = vcam()
        if c:
            c.seek(round(pos_var.get()))

    def on_drag_move(v):
        c = vcam()
        if c and drag["on"]:
            time_txt.set(f"{fmt(float(v) / c.fps)} / {fmt(c.total / c.fps)}")

    ttk.Button(vbar, textvariable=play_txt, width=6, command=toggle_play).pack(side="left")
    ttk.Button(vbar, text="처음", width=5, command=lambda: vcam() and vcam().seek(0)).pack(side="left", padx=(4, 0))
    ttk.Button(vbar, text="-5초", width=5, command=lambda: vcam() and seek_rel(-int(5 * vcam().fps))).pack(side="left", padx=(4, 0))
    ttk.Button(vbar, text="◀ 1프레임", width=9, command=lambda: step(-1)).pack(side="left", padx=(4, 0))
    ttk.Button(vbar, text="1프레임 ▶", width=9, command=lambda: step(1)).pack(side="left", padx=(4, 0))
    ttk.Button(vbar, text="+5초", width=5, command=lambda: vcam() and seek_rel(int(5 * vcam().fps))).pack(side="left", padx=(4, 0))
    ttk.Checkbutton(vbar, text="반복", variable=loop_var,
                    command=lambda: vcam() and setattr(vcam(), "loop", loop_var.get())).pack(side="right")
    ttk.Label(vbar, textvariable=time_txt, width=13, anchor="e").pack(side="right", padx=(6, 6))
    seek_bar = ttk.Scale(vbar, from_=0, to=1, variable=pos_var, orient="horizontal", command=on_drag_move)
    seek_bar.pack(side="left", fill="x", expand=True, padx=(8, 0))
    seek_bar.bind("<ButtonPress-1>", lambda _: drag.update(on=True))
    seek_bar.bind("<ButtonRelease-1>", on_drag_end)
    holder = tk.Frame(left, bg="#202020")  # 영상 크기가 창 크기를 밀어내지 않도록 고정 틀에 넣음
    holder.pack(fill="both", expand=True)
    holder.pack_propagate(False)
    video = tk.Label(holder, bg="#202020", text="영상을 여는 중...", fg="white")
    video.pack(fill="both", expand=True)

    def section(title):
        f = ttk.LabelFrame(panel, text=title, padding=6)
        f.pack(fill="x", pady=4)
        return f

    # 입력 장치
    src = section("입력")
    src_mode = tk.StringVar(value="file" if a.source else "cam")
    file_path = tk.StringVar(value=a.source or "")
    cam_var = tk.StringVar(value=str(a.cam))
    res_var = tk.StringVar(value=f"{a.width}x{a.height}")
    r1 = ttk.Frame(src)
    r1.pack(fill="x")
    ttk.Radiobutton(r1, text="웹캠", variable=src_mode, value="cam").pack(side="left")
    ttk.Combobox(r1, textvariable=cam_var, values=[str(i) for i in range(5)], width=3,
                 state="readonly").pack(side="left", padx=4)
    ttk.Combobox(r1, textvariable=res_var, values=["640x480", "1280x720", "1920x1080"], width=10,
                 state="readonly").pack(side="left", padx=4)
    r2 = ttk.Frame(src)
    r2.pack(fill="x", pady=(4, 0))
    ttk.Radiobutton(r2, text="영상 파일", variable=src_mode, value="file").pack(side="left")

    def pick_file():
        p = filedialog.askopenfilename(filetypes=[("영상", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")])
        if p:
            file_path.set(p)
            src_mode.set("file")

    ttk.Button(r2, text="찾기...", command=pick_file).pack(side="left", padx=4)
    ttk.Label(src, textvariable=file_path, wraplength=330, foreground="#555").pack(fill="x")

    def apply_source():
        w, h = (int(v) for v in res_var.get().split("x"))
        if src_mode.get() == "file":
            if not file_path.get():
                post("영상 파일을 먼저 고르세요")
                return
            eng.set_source(file_path.get(), 0, w, h)
        else:
            eng.set_source(None, int(cam_var.get()), w, h)
        c = vcam()
        if c:  # 영상이면 재생 막대를 보이고, 웹캠이면 숨긴다
            c.loop = loop_var.get()
            seek_bar.configure(to=max(1, c.total - 1))
            vbar.pack(side="bottom", fill="x", pady=(6, 0), after=stat_lbl)
        else:
            vbar.pack_forget()

    ttk.Button(src, text="이 입력으로 열기", command=apply_source).pack(fill="x", pady=(4, 0))

    # 모델
    mf = section("모델")
    labels = [f"{name}  ({sz})" + (f" - {MODEL_NOTE[k]}" if k in MODEL_NOTE else "")
              for k, name, kind, w, sz in MODELS]
    key_of = {lab: m[0] for lab, m in zip(labels, MODELS)}
    model_var = tk.StringVar(value=labels[[m[0] for m in MODELS].index(start)])
    size_var = tk.StringVar(value=str(eng.state["imgsz"]))

    def on_model(_=None):
        k = key_of[model_var.get()]
        eng.state["key"] = k
        eng.state["imgsz"] = keymap[k][3]
        size_var.set(str(keymap[k][3]))
        prompt_row_state()

    cb = ttk.Combobox(mf, textvariable=model_var, values=labels, state="readonly", height=16)
    cb.pack(fill="x")
    cb.bind("<<ComboboxSelected>>", on_model)
    r3 = ttk.Frame(mf)
    r3.pack(fill="x", pady=(6, 0))
    ttk.Label(r3, text="입력 크기").pack(side="left")
    sz_cb = ttk.Combobox(r3, textvariable=size_var, values=[str(s) for s in SIZES], width=6, state="readonly")
    sz_cb.pack(side="left", padx=6)
    sz_cb.bind("<<ComboboxSelected>>", lambda _: eng.state.update(imgsz=int(size_var.get())))
    ttk.Label(r3, text="(클수록 작은 차에 유리, 느려짐)", foreground="#666").pack(side="left")

    # YOLOE 프롬프트
    pf = section("YOLOE 프롬프트 (쉼표로 구분)")
    prompt_var = tk.StringVar(value=", ".join(prompt))
    pe = ttk.Entry(pf, textvariable=prompt_var)
    pe.pack(fill="x")

    def apply_prompt():
        p = [s.strip() for s in prompt_var.get().split(",") if s.strip()]
        if p:
            eng.state["prompt"] = p
            post(f"프롬프트 설정: {', '.join(p)}")

    pb = ttk.Button(pf, text="프롬프트 적용", command=apply_prompt)
    pb.pack(fill="x", pady=(4, 0))

    def prompt_row_state():
        on = keymap[eng.state["key"]][1] == "yoloe"
        for wdg in (pe, pb):
            wdg.state(["!disabled"] if on else ["disabled"])

    prompt_row_state()

    # 탐지 기준
    tf = section("탐지 기준")
    conf_var = tk.DoubleVar(value=a.conf)
    conf_lbl = tk.StringVar(value=f"신뢰도 기준 {a.conf:.2f}")
    ttk.Label(tf, textvariable=conf_lbl).pack(anchor="w")

    def on_conf(v):
        c = round(float(v) / 0.05) * 0.05
        eng.state["conf"] = c
        conf_lbl.set(f"신뢰도 기준 {c:.2f}")

    ttk.Scale(tf, from_=0.05, to=0.95, variable=conf_var, command=on_conf).pack(fill="x")
    show_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(tf, text="차량이 아닌 물체도 보기 (주황색)", variable=show_var,
                    command=lambda: eng.state.update(show_all=show_var.get())).pack(anchor="w", pady=(4, 0))

    # 실행
    rf = section("실행")
    r4 = ttk.Frame(rf)
    r4.pack(fill="x", pady=(0, 4))
    ttk.Label(r4, text="분석 빈도").pack(side="left")
    rate_labels = [lab for lab, _ in RATES]
    rate_var = tk.StringVar(value=next((lab for lab, v in RATES if v == eng.state["rate"]), rate_labels[0]))
    rate_cb = ttk.Combobox(r4, textvariable=rate_var, values=rate_labels, width=10, state="readonly")
    rate_cb.pack(side="left", padx=6)
    rate_cb.bind("<<ComboboxSelected>>", lambda _: eng.state.update(rate=dict(RATES)[rate_var.get()]))
    ttk.Label(r4, text="CPU 코어").pack(side="left", padx=(6, 0))
    core_vals = sorted({1, 2, 4, CPU_COUNT // 2, CPU_COUNT} - {0})
    core_vals = [c for c in core_vals if c <= CPU_COUNT]
    core_var = tk.StringVar(value=str(eng.state["threads"]))
    core_cb = ttk.Combobox(r4, textvariable=core_var, values=[str(c) for c in core_vals], width=3, state="readonly")
    core_cb.pack(side="left", padx=6)
    core_cb.bind("<<ComboboxSelected>>", lambda _: eng.state.update(threads=int(core_var.get())))
    ttk.Label(rf, text="줄일수록 CPU 사용률이 내려감 (전체 %d코어)" % CPU_COUNT,
              foreground="#666").pack(anchor="w")
    pause_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(rf, text="분석 일시정지", variable=pause_var,
                    command=lambda: eng.state.update(paused=pause_var.get())).pack(anchor="w")
    log_var = tk.BooleanVar(value=a.log)
    ttk.Checkbutton(rf, text="결과를 CSV로 기록", variable=log_var,
                    command=lambda: eng.set_log(log_var.get())).pack(anchor="w")
    last_img = {"im": None}

    def snapshot():
        im = last_img["im"]
        if im is None:
            return
        name = keymap[eng.state["key"]][0]
        p = os.path.join(HERE, f"live_{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(p, im)
        post(f"화면 저장: {p}")

    ttk.Button(rf, text="현재 화면 저장", command=snapshot).pack(fill="x", pady=(4, 0))

    # 결과 출력
    of = section("결과")
    log = tk.Text(of, height=14, width=44, font=("Consolas", 9), wrap="word")
    log.pack(fill="both", expand=True)

    if a.log:
        eng.set_log(True)
    dev, devname = device_name()
    post(f"계산 장치: {dev} ({devname})")

    def tick():
        c = vcam()
        if c:
            play_txt.set("정지" if c.playing else "재생")
            if not drag["on"]:
                pos_var.set(max(0, c.pos))
                time_txt.set(f"{fmt(max(0, c.pos) / c.fps)} / {fmt(c.total / c.fps)}")
        im, sh = eng.frame()
        if im is not None:
            last_img["im"] = im
            vw, vh = max(holder.winfo_width(), 100), max(holder.winfo_height(), 100)
            h, w = im.shape[:2]
            s = min(vw / w, vh / h)
            disp = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            video.configure(image=photo, text="")
            video.image = photo
        name = keymap[eng.state["key"]][0]
        if sh["loading"]:
            stat.set(f"{sh['loading']} 불러오는 중...")
        elif eng.state["paused"]:
            stat.set("분석 일시정지")
        else:
            stat.set(f"{name}  |  입력 {eng.state['imgsz']}  |  분석 1회 {sh['ms']:.0f} ms  |  "
                     f"실제 {sh['fps']:.1f}회/초  |  차량 {sh['n']}대")
        changed = False
        while not msgs.empty():
            log.insert("end", msgs.get() + "\n")
            changed = True
        if changed:
            n = int(log.index("end-1c").split(".")[0])
            if n > 400:
                log.delete("1.0", f"{n - 400}.0")
            log.see("end")
        root.after(33, tick)

    def on_close():
        eng.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, apply_source)
    root.after(200, tick)
    if a.max_seconds:
        root.after(int(a.max_seconds * 1000), on_close)
    root.mainloop()


def run_console(a, keymap, start, prompt):
    eng = Engine(keymap, start, a.conf, prompt, print, print, a.rate, a.threads)
    dev, devname = device_name()
    print(f"계산 장치: {dev} ({devname})")
    if a.log:
        eng.set_log(True)
    if not eng.set_source(a.source, a.cam, a.width, a.height):
        eng.close()
        sys.exit(1)
    t0 = time.time()
    try:
        while not a.max_seconds or time.time() - t0 < a.max_seconds:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    eng.close()


def main():
    ap = argparse.ArgumentParser(description="실시간 차량 탐지 모델 확인")
    ap.add_argument("--cam", type=int, default=0, help="시작 웹캠 번호")
    ap.add_argument("--source", help="웹캠 대신 쓸 영상 파일")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="1", help="시작 모델 (키 또는 이름, 예: 1, rtdetr, oiv7)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--rate", type=float, default=2, help="초당 분석 횟수 (0이면 제한 없음, 기본 2)")
    ap.add_argument("--threads", type=int, default=0, help="모델 계산에 쓸 CPU 코어 수 (기본: 전체의 절반)")
    ap.add_argument("--prompt", default="toy car,car,model car,vehicle", help="YOLOE 글 프롬프트, 쉼표로 구분")
    ap.add_argument("--log", action="store_true", help="처음부터 결과를 csv로 저장")
    ap.add_argument("--nogui", action="store_true", help="창 없이 콘솔 출력만")
    ap.add_argument("--max-seconds", type=float, default=0, help="정해진 시간 뒤 자동 종료 (0이면 계속)")
    a = ap.parse_args()

    prompt = [s.strip() for s in a.prompt.split(",") if s.strip()]
    keymap = {k: (name, kind, w, sz) for k, name, kind, w, sz in MODELS}
    start = ALIASES.get(a.model.lower(), a.model.lower())
    if start not in keymap:
        sys.exit(f"모르는 모델: {a.model}")
    if a.nogui:
        run_console(a, keymap, start, prompt)
    else:
        run_gui(a, keymap, start, prompt)


if __name__ == "__main__":
    main()
