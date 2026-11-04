"""실시간 차량 탐지 모델 확인 프로그램 (PARKIT 테스트베드용)

웹캠 화면을 받아 고른 모델로 계속 분석하고, 결과를 화면(상자)과 결과 창(글)에 바로 낸다.
모델, 입력 크기, 신뢰도 기준, 웹캠/영상 선택 등 모든 옵션은 창 오른쪽 패널에서 마우스로 바꾼다.
"카메라 2개"를 고르면 웹캠 2대를 위아래로 보여 주고, 카메라마다 모델과 옵션을 따로 고른다.
"모델 비교"를 고르면 웹캠 1대의 같은 프레임을 모델 A와 모델 B로 각각 분석해 위아래로 보여 준다.

설치(한 번만):
    pip install ultralytics
    pip install "opencv-python==4.10.0.84"      (OpenCV 5.0은 이 웹캠을 못 열었음)
    pip install effdet                           (EfficientDet을 쓸 때만)

사용법:
    python live_detect.py                       창이 뜨고 웹캠 0번, YOLO26n으로 시작
    python live_detect.py --cam 1               시작 웹캠 번호
    python live_detect.py --model rtdetr        시작 모델 (아래 목록의 키 이름)
    python live_detect.py --source 영상.mp4      웹캠 대신 녹화 영상으로 시작
    python live_detect.py --model best          우리가 학습한 모델(이 폴더의 best.pt)로 시작
    python live_detect.py --weights 경로\best.pt  다른 위치의 모델 파일로 시작
    python live_detect.py --model best-onnx     PyTorch 없이 이 폴더의 best.onnx로 시작 (PyTorch가 막힌 PC용)
    python live_detect.py --rate 1 --threads 2  초당 1회, CPU 코어 2개로 분석 (CPU 사용 줄이기)
    python live_detect.py --dual                카메라 2개(웹캠 0번, 1번)로 시작
    python live_detect.py --dual --model best --model2 yolo26n   카메라마다 다른 모델로 시작
    python live_detect.py --compare --model yolo26n --model2 best   웹캠 1대로 두 모델 비교 (위 A, 아래 B)
    python live_detect.py --nogui               창 없이 콘솔 출력만 (영상/웹캠 자동 분석)

모델 파일은 처음 고를 때 자동으로 내려받는다(인터넷 필요, 실행한 폴더에 저장).
학습한 모델: Colab에서 받은 best.pt를 이 파일과 같은 폴더에 두면 목록의 "학습 모델(best.pt)"로 쓸 수 있다.
다른 위치의 .pt 파일은 창 오른쪽 "모델 파일 불러오기..." 버튼이나 --weights로 불러온다.
직접 만든 PARKIT Tiny 모델(mymodel.pt)도 같은 방법으로 불러온다. 구조 파일 model.py를 가중치 파일과 같은 폴더에 둔다.
PyTorch가 막힌 PC: Colab에서 best.pt를 best.onnx로 바꿔(yolo export ... format=onnx) 이 폴더에 두고
"학습 모델 ONNX(best.onnx)"를 고른다. ONNX 모델은 OpenCV만으로 돌아가 PyTorch가 필요 없다.
"""
import argparse, csv, os, platform, queue, sys, threading, time, types
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
    ("e", "학습 모델(best.pt)", "custom", "best.pt",          640),
    ("g", "학습 모델 ONNX(best.onnx)", "onnx", "best.onnx",   640),
]
ALIASES = {"yolo26n": "1", "yolo26s": "2", "yolo26m": "3", "yolo11n": "4", "yolo11s": "5",
           "yolo11m": "6", "yolov8n": "7", "yolov8s": "8", "yolov8m": "9", "rtdetr": "0",
           "oiv7": "a", "openimages": "a", "yoloe-s": "b", "yoloe-l": "c", "effdet": "d",
           "best": "e", "custom": "e", "best-onnx": "g", "onnx": "g"}
MODEL_NOTE = {"1": "가장 빠름", "0": "가장 정확, 무거움", "a": "검정 차에 강함", "b": "글 프롬프트",
              "c": "글 프롬프트", "d": "effdet 설치 필요", "e": "이 폴더의 best.pt",
              "g": "이 폴더의 best.onnx, PyTorch 없이 실행"}
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
CUSTOM_VEHICLE = {"car", "bus", "truck", "vehicle"}  # 불러온 모델에서 차량으로 셀 이름


def add_model_file(keymap, path, imgsz=640):
    """파일 위치의 모델을 목록(keymap)에 더하고 키를 돌려준다. 이미 있으면 그 키를 돌려준다."""
    path = os.path.abspath(path)
    for k, (_, kind, w, _) in keymap.items():
        if kind in ("custom", "onnx") and os.path.abspath(os.path.join(HERE, w)) == path:
            return k
    # 이름은 화면 표시, 기록, 저장 파일 이름에 쓰인다. runs/이름/weights/best.pt면 "이름_best"
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    if parent == "weights":
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    name = f"{parent}_{stem}" if parent else stem
    n = sum(1 for k in keymap if k.startswith("f")) + 1
    key = f"f{n}"
    kind = "onnx" if path.lower().endswith(".onnx") else "custom"
    keymap[key] = (name, kind, path, imgsz)
    return key


def onnx_names(path):
    """Ultralytics가 ONNX 파일에 적어 둔 클래스 이름({0: 'car'})을 읽는다. onnx 패키지 없이 파일에서 직접 찾는다."""
    import ast
    data = open(path, "rb").read()
    i = data.rfind(b"\n\x05names\x12")  # 메타데이터 항목: key="names", value=길이+문자열
    if i < 0:
        return None
    j, n, shift = i + 8, 0, 0
    while True:  # 길이는 protobuf varint로 적혀 있다
        b = data[j]
        n |= (b & 0x7F) << shift
        j += 1
        shift += 7
        if b < 0x80:
            break
    try:
        return {int(k): str(v) for k, v in ast.literal_eval(data[j:j + n].decode("utf-8")).items()}
    except Exception:
        return None


_TORCH = {}


def get_torch():
    """torch를 한 번만 불러 본다. 보안 정책 등으로 막히면 None (매번 다시 불러 보면 느려짐)."""
    if "m" not in _TORCH:
        try:
            import torch
            _TORCH["m"], _TORCH["err"] = torch, None
        except Exception as e:
            _TORCH["m"], _TORCH["err"] = None, e
    return _TORCH["m"]


class OnnxYolo:
    """Ultralytics에서 내보낸 YOLO ONNX 모델을 OpenCV DNN으로 돌린다. PyTorch가 필요 없다.
    출력 (1, 4+클래스수, 8400) 형식만 지원한다(yolo export 기본값). nms=True로 내보낸 파일은 OpenCV가 못 읽는다."""

    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.names = onnx_names(path)
        self.size = 640

    def predict(self, im, conf):
        h, w = im.shape[:2]
        s = self.size / max(h, w)
        rs = cv2.resize(im, (int(round(w * s)), int(round(h * s))))
        pad = np.full((self.size, self.size, 3), 114, np.uint8)  # Ultralytics처럼 가운데 두고 회색으로 채움
        dy, dx = (self.size - rs.shape[0]) // 2, (self.size - rs.shape[1]) // 2
        pad[dy:dy + rs.shape[0], dx:dx + rs.shape[1]] = rs
        self.net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (self.size, self.size), swapRB=True))
        out = self.net.forward()[0]
        if out.ndim != 2 or out.shape[0] < 5:
            raise RuntimeError(f"지원하지 않는 ONNX 출력 형식 {out.shape}. format=onnx 기본값으로 다시 내보내세요")
        out = out.T  # (8400, 4+클래스수)
        if self.names is None:
            self.names = {i: f"class{i}" for i in range(out.shape[1] - 4)}
        cls = out[:, 4:].argmax(1)
        score = out[np.arange(len(out)), 4 + cls]
        keep = score >= conf
        box, cls, score = out[keep, :4], cls[keep], score[keep]
        xywh = np.stack([box[:, 0] - box[:, 2] / 2 - dx, box[:, 1] - box[:, 3] / 2 - dy, box[:, 2], box[:, 3]], 1) / s
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, 0.7) if len(score) else []
        res = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            res.append((self.names.get(int(cls[i]), str(cls[i])), float(score[i]), x, y, x + bw, y + bh))
        return res


def parkit_checkpoint(path):
    """직접 만든 PARKIT Tiny 체크포인트({"model": 가중치, "width", "size", ...})면 그 내용을, 아니면 None.
    weights_only=True로 읽어 가중치 말고 다른 코드는 실행하지 않는다. Ultralytics 파일은 여기서 실패해 None이 된다."""
    torch = get_torch()
    if torch is None:
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict) and "stem.0.weight" in ck["model"]:
        return ck
    return None


class ParkitTiny:
    """동료가 처음부터 만든 PARKIT Tiny 모델(model.py + mymodel.pt)을 돌린다. Ultralytics 형식이 아니라 따로 처리한다.
    모델 구조는 가중치 파일 옆(없으면 이 파일 옆)의 model.py에서 가져온다. 구조를 바꿔 다시 학습해도 model.py만 같이 두면 된다.
    입력: 가운데 letterbox(회색 114), RGB, 0~1. test 사진으로 전처리 방식을 비교해 정답 상자와 가장 잘 맞는 방식을 골랐다."""
    names = {0: "car"}

    def __init__(self, path, ck):
        import importlib.util
        for p in (os.path.join(os.path.dirname(path), "model.py"), os.path.join(HERE, "model.py")):
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("parkit_model", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                break
        else:
            raise FileNotFoundError("PARKIT Tiny 모델 구조 파일 model.py가 없음. 가중치 파일이나 live_detect.py와 같은 폴더에 두세요")
        self.torch, self.mod = get_torch(), mod
        self.net = mod.ParkitTiny(ck.get("width", 16))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.size = int(ck.get("size", 640))

    def predict(self, im, imgsz, conf):
        size = max(32, int(imgsz) // 16 * 16)  # 출력 칸이 8, 16 간격이라 16의 배수로 맞춘다
        h, w = im.shape[:2]
        s = size / max(h, w)
        rs = cv2.resize(im, (int(round(w * s)), int(round(h * s))))
        pad = np.full((size, size, 3), 114, np.uint8)
        dy, dx = (size - rs.shape[0]) // 2, (size - rs.shape[1]) // 2
        pad[dy:dy + rs.shape[0], dx:dx + rs.shape[1]] = rs
        x = self.torch.from_numpy(np.ascontiguousarray(pad[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255.0
        boxes, scores = self.mod.predict(self.net, x, conf=conf)[0]
        out = []
        for (x1, y1, x2, y2), p in zip(boxes.tolist(), scores.tolist()):
            out.append(("car", float(p), (x1 - dx) / s, (y1 - dy) / s, (x2 - dx) / s, (y2 - dy) / s))
        return out


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
        torch = get_torch()
        if torch is not None and torch.cuda.is_available():
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
        if kind in ("custom", "onnx") and not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 없음: {path} "
                                    f"({os.path.basename(path)}를 live_detect.py 폴더에 넣거나 '모델 파일 불러오기...'로 고르세요)")
        if kind == "onnx":
            self.m = OnnxYolo(path)
            names = set(self.m.names.values()) if self.m.names else {"car"}
            self.vehicle = (names & CUSTOM_VEHICLE) or names
            return
        if kind == "custom":
            ck = parkit_checkpoint(path)
            if ck is not None:  # Ultralytics 형식이 아닌, 직접 만든 모델
                self.m = ParkitTiny(path, ck)
                self.vehicle = {"car"}
                return
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
        if kind == "custom":
            # 우리 모델은 car 하나뿐이다. 차량 이름이 하나도 없는 모델이면 모든 클래스를 차량으로 센다
            names = set(self.m.names.values())
            self.vehicle = (names & CUSTOM_VEHICLE) or names

    def set_prompt(self, prompt):
        self.prompt = list(prompt)
        self.m.set_classes(self.prompt, self.m.get_text_pe(self.prompt))

    def is_vehicle(self, name):
        if self.kind == "yoloe":
            return True  # 프롬프트로 고른 대상이 곧 찾는 물체
        if self.kind in ("custom", "onnx"):
            return name in self.vehicle
        return name in VEHICLE[self.kind]

    def run(self, im, imgsz, conf):
        if self.kind == "effdet":
            return self._run_effdet(im, conf)
        if self.kind == "onnx":
            return self.m.predict(im, conf)  # 입력 크기는 내보낼 때 정한 640으로 고정
        if isinstance(self.m, ParkitTiny):
            return self.m.predict(im, imgsz, conf)
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

    def __init__(self, keymap, start_key, conf, prompt, on_result, on_message, rate=2, threads=0, tag=""):
        self.keymap = keymap
        self.tag = tag  # 카메라 이름. 기록 파일과 화면 저장 파일 이름에 붙는다
        self.cam_idx = None
        self.state = {"key": start_key, "imgsz": keymap[start_key][3], "conf": conf,
                      "show_all": False, "prompt": list(prompt), "paused": False,
                      "rate": rate, "threads": threads or max(1, CPU_COUNT // 2)}
        self.applied_threads = None
        self.on_result, self.on_message = on_result, on_message
        self.cache = {}
        self.failed = set()  # 불러오기에 실패한 모델 키. 목록에서 다시 고를 때까지 재시도하지 않는다
        self.cam = None
        # 모델 비교 모드: B(follower)는 자기 카메라 없이 A(leader)가 분석한 프레임을 그대로 받아 분석하고,
        # A는 B가 그 프레임을 끝낼 때까지 다음 프레임을 잡지 않는다. 두 모델이 같은 그림을 보고, 번갈아 돌아 속도도 공정하게 잰다
        self.leader = self.follower = None
        self.in_compare = False  # A: 비교 모드면 True. 화면의 상자를 B가 끝낸 프레임 결과에 맞춘다
        self.pub = 0  # 분석을 끝내 내놓은 횟수. 입력을 바꿔도 되돌리지 않는다
        self.prev = None  # A: 바로 전에 내놓은 결과. B가 따라올 때까지 화면에는 이것을 보여 준다
        self.done_pub = 0  # B: 마지막으로 분석을 끝낸 A의 pub
        self.lock = threading.Lock()
        self.shared = {"dets": [], "ms": 0.0, "det": None, "fps": 0.0, "loading": "", "n": 0, "im": None, "fn": 0}
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
        self.cam_idx = None if source else cam_idx
        with self.lock:
            self.shared.update(dets=[], det=None, im=None, fn=0)
        self.on_message(msg)
        return cam is not None

    def close_source(self):
        """입력을 닫는다(카메라 2대 모드를 끌 때)."""
        old, self.cam, self.cam_idx = self.cam, None, None
        if old:
            old.alive = False
        with self.lock:
            self.shared.update(dets=[], det=None, n=0, im=None, fn=0)

    def set_log(self, on):
        if on and not self.writer:
            tag = f"{self.tag}_" if self.tag else ""
            path = os.path.join(HERE, f"live_log_{tag}{datetime.now():%Y%m%d_%H%M%S}.csv")
            self.logf = open(path, "w", newline="", encoding="utf-8-sig")
            self.writer = csv.writer(self.logf)
            self.writer.writerow(["시각", "모델", "입력크기", "신뢰도기준", "추론ms", "차량수", "검출(이름:신뢰도)",
                                  "프레임번호"])
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
        cv2.setNumThreads(n)  # ONNX 모델은 OpenCV가 계산한다
        try:
            torch = get_torch()
            # 모델을 불러올 때 라이브러리가 값을 되돌릴 수 있어 분석 직전마다 확인한다
            if torch is not None and torch.get_num_threads() != n:  # PyTorch가 막힌 PC면 OpenCV만 설정
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

    def _latest(self):
        """분석할 프레임, 프레임 번호, 새 프레임인지 가릴 값. 비교 모드의 B는 A가 방금 분석한 프레임을 받는다."""
        ld = self.leader
        if ld is not None:
            with ld.lock:
                im, n, pub = ld.shared["im"], ld.shared["fn"], ld.pub
            return (None, 0, 0) if im is None else (im.copy(), n, ("B", pub))
        cam = self.cam
        im, n = cam.latest() if cam else (None, 0)
        return im, n, n

    def _follower_busy(self):
        """비교 모드에서 B가 A의 마지막 프레임을 아직 분석하는 중인지. B가 쉬거나 실패했으면 기다리지 않는다."""
        f = self.follower
        if f is None or f.state["paused"] or f.state["key"] in f.failed:
            return False
        with f.lock:
            if f.shared["loading"]:
                return False
        with self.lock:
            if self.shared["im"] is None:  # 입력을 바꿔 내놓은 프레임이 없으면 기다릴 것도 없다
                return False
            pub = self.pub
        return f.done_pub < pub

    def _done(self, key):
        if isinstance(key, tuple):  # B가 A의 프레임을 끝냄
            self.done_pub = key[1]

    def _worker(self):
        last_n, t_prev, t_start = -1, time.time(), 0.0
        while not self.stop.is_set():
            st = dict(self.state)
            cam = self.cam
            if st["paused"] or (cam is None and self.leader is None):
                time.sleep(0.05)
                continue
            self._apply_threads(st["threads"])
            if st["rate"] > 0 and self.leader is None:  # B는 A의 속도를 따라간다
                self._wait_until(t_start + 1.0 / st["rate"])
                if self.stop.is_set():
                    break
                st = dict(self.state)
            if st["key"] in self.failed:  # 실패한 모델은 다시 고를 때까지 쉬기만 한다
                time.sleep(0.2)
                continue
            det = self._load(st["key"])
            if det is None:
                self.failed.add(st["key"])
                if "1" not in self.failed:  # 실패하면 YOLO26n으로 돌아감
                    self.state["key"] = "1"
                    self.state["imgsz"] = self.keymap["1"][3]
                    self.on_message("YOLO26n으로 돌아감")
                else:
                    self.on_message("YOLO26n도 불러오지 못해 분석을 멈춤. 위 오류를 해결한 뒤 모델을 다시 고르세요")
                continue
            if self._follower_busy():
                time.sleep(0.005)
                continue
            im, n, key = self._latest()
            if im is None or key == last_n:
                time.sleep(0.005)
                continue
            last_n = key
            self._apply_threads(st["threads"])
            t = t_start = time.time()
            try:
                dets = merge_vehicles(det.run(im, st["imgsz"], st["conf"]), det.is_vehicle)
            except Exception as e:
                self._done(key)
                self.on_message(f"분석 오류: {e}")
                time.sleep(0.5)
                continue
            ms = (time.time() - t) * 1000
            now = time.time()
            inf_fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            veh = [d for d in dets if det.is_vehicle(d[0])]
            with self.lock:
                self.prev = dict(self.shared)
                self.shared.update(dets=dets, ms=ms, det=det, fps=inf_fps, n=len(veh), im=im, fn=n)
                self.pub += 1
            self._done(key)
            name = self.keymap[st["key"]][0]
            items = ", ".join(f"{d[0]}:{d[1]:.2f}" for d in veh)
            others = [d for d in dets if not det.is_vehicle(d[0])]
            extra = f"  (기타: {', '.join(f'{d[0]}:{d[1]:.2f}' for d in others[:3])})" if others else ""
            line = f"{datetime.now():%H:%M:%S} [{name} {st['imgsz']}] {ms:6.0f}ms  차량 {len(veh)}대  {items}{extra}"
            self.on_result(line)
            if self.writer:
                self.writer.writerow([datetime.now().isoformat(timespec="milliseconds"), name, st["imgsz"],
                                      f"{st['conf']:.2f}", round(ms), len(veh),
                                      "; ".join(f"{d[0]}:{d[1]:.2f}" for d in dets), n])

    def frame(self, analyzed=False):
        """그릴 프레임과 현재 결과를 돌려준다.
        화면은 실시간 영상에 마지막 결과 상자를 겹친다. analyzed=True면 결과를 낸 그 프레임에 그린다(비교 화면 저장)."""
        lag = self.in_compare and self._follower_busy()
        with self.lock:
            sh = dict(self.prev if lag and self.prev else self.shared)  # 비교 모드: A의 상자를 B와 같은 프레임 결과로 맞춤
        if analyzed:
            im = sh["im"]
            if im is None and self.leader is not None:  # B가 아직 결과가 없으면 A의 프레임만
                with self.leader.lock:
                    im = self.leader.shared["im"]
                sh["det"] = None
            im = None if im is None else im.copy()
        else:
            cam = self.cam if self.leader is None else self.leader.cam  # 비교 모드의 B는 A의 카메라를 같이 본다
            im, _ = cam.latest() if cam else (None, 0)
        if im is not None and sh["det"] is not None:
            draw(im, sh["dets"], sh["det"], self.state["show_all"])
        return im, sh

    def close(self):
        self.stop.set()
        if self.cam:
            self.cam.alive = False
        self.th.join(timeout=3)
        self.set_log(False)


def run_gui(a, keymap, starts, prompt):
    import tkinter as tk
    from tkinter import ttk, filedialog
    from PIL import Image, ImageTk

    msgs = queue.Queue()
    root = tk.Tk()
    root.title("PARKIT 실시간 차량 탐지 확인")
    root.geometry("1360x780")
    # 보기 방식: single(카메라 1개), dual(카메라 2개), compare(카메라 1개를 모델 A/B로 비교)
    # 분석 스레드에서도 읽으므로 tk 변수 대신 일반 dict에 둔다
    mode = {"view": "compare" if a.compare else "dual" if a.dual else "single"}
    TAGS = {"dual": ("카메라1", "카메라2"), "compare": ("A", "B")}

    def post(line):
        print(line)
        msgs.put(line)

    def poster(i):
        """카메라 2대·모델 비교 모드에서는 어느 쪽의 줄인지 앞에 붙인다."""
        return lambda line: post(f"[{TAGS[mode['view']][i]}] {line}" if mode["view"] in TAGS else line)

    threads = a.threads or max(1, CPU_COUNT // 2)
    engs = [Engine(keymap, starts[i], a.conf, prompt, poster(i), poster(i),
                   a.rate, threads, tag=f"cam{i + 1}") for i in range(2)]

    # ---- 오른쪽: 옵션 패널 (먼저 배치해야 폭이 줄지 않음) ----
    panel = ttk.Frame(root, width=360)
    panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
    panel.pack_propagate(False)

    # ---- 왼쪽: 영상 (카메라 2대·모델 비교 모드면 위아래로 둘) ----
    left = ttk.Frame(root)
    left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

    def fmt(sec):
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

    def build_view(eng, title):
        """카메라 하나의 화면, 상태 줄, 재생 막대."""
        v = types.SimpleNamespace(eng=eng, last=None)
        v.frame = ttk.Frame(left)
        v.title = ttk.Label(v.frame, text=title, font=("맑은 고딕", 10, "bold"))
        v.stat = tk.StringVar(value="")
        v.stat_lbl = ttk.Label(v.frame, textvariable=v.stat, font=("맑은 고딕", 11, "bold"))
        v.stat_lbl.pack(side="bottom", anchor="w", pady=(6, 0))

        # 영상 파일일 때만 보이는 재생 막대
        v.vbar = ttk.Frame(v.frame)
        v.play_txt = tk.StringVar(value="정지")
        v.pos_var = tk.DoubleVar(value=0)
        v.time_txt = tk.StringVar(value="")
        v.loop_var = tk.BooleanVar(value=True)
        drag = {"on": False}
        v.drag = drag

        def vcam():
            c = eng.cam
            return c if c is not None and c.is_file else None

        v.vcam = vcam

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
                c.seek(round(v.pos_var.get()))

        def on_drag_move(val):
            c = vcam()
            if c and drag["on"]:
                v.time_txt.set(f"{fmt(float(val) / c.fps)} / {fmt(c.total / c.fps)}")

        vbar = v.vbar
        ttk.Button(vbar, textvariable=v.play_txt, width=6, command=toggle_play).pack(side="left")
        ttk.Button(vbar, text="처음", width=5, command=lambda: vcam() and vcam().seek(0)).pack(side="left", padx=(4, 0))
        ttk.Button(vbar, text="-5초", width=5, command=lambda: vcam() and seek_rel(-int(5 * vcam().fps))).pack(side="left", padx=(4, 0))
        ttk.Button(vbar, text="◀ 1프레임", width=9, command=lambda: step(-1)).pack(side="left", padx=(4, 0))
        ttk.Button(vbar, text="1프레임 ▶", width=9, command=lambda: step(1)).pack(side="left", padx=(4, 0))
        ttk.Button(vbar, text="+5초", width=5, command=lambda: vcam() and seek_rel(int(5 * vcam().fps))).pack(side="left", padx=(4, 0))
        ttk.Checkbutton(vbar, text="반복", variable=v.loop_var,
                        command=lambda: vcam() and setattr(vcam(), "loop", v.loop_var.get())).pack(side="right")
        ttk.Label(vbar, textvariable=v.time_txt, width=13, anchor="e").pack(side="right", padx=(6, 6))
        v.seek_bar = ttk.Scale(vbar, from_=0, to=1, variable=v.pos_var, orient="horizontal", command=on_drag_move)
        v.seek_bar.pack(side="left", fill="x", expand=True, padx=(8, 0))
        v.seek_bar.bind("<ButtonPress-1>", lambda _: drag.update(on=True))
        v.seek_bar.bind("<ButtonRelease-1>", on_drag_end)
        holder = tk.Frame(v.frame, bg="#202020")  # 영상 크기가 창 크기를 밀어내지 않도록 고정 틀에 넣음
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)
        v.holder = holder
        v.video = tk.Label(holder, bg="#202020", text="영상을 여는 중...", fg="white")
        v.video.pack(fill="both", expand=True)

        def after_source():
            c = vcam()
            if c:  # 영상이면 재생 막대를 보이고, 웹캠이면 숨긴다
                c.loop = v.loop_var.get()
                v.seek_bar.configure(to=max(1, c.total - 1))
                vbar.pack(side="bottom", fill="x", pady=(6, 0), after=v.stat_lbl)
            else:
                vbar.pack_forget()

        v.after_source = after_source
        return v

    views = [build_view(engs[i], f"카메라 {i + 1}") for i in range(2)]

    def section(parent, title):
        f = ttk.LabelFrame(parent, text=title, padding=6)
        f.pack(fill="x", pady=4)
        return f

    # ---- 공통 설정 ----
    cf = section(panel, "공통")
    view_var = tk.StringVar(value=mode["view"])
    rv = ttk.Frame(cf)
    rv.pack(fill="x")
    ttk.Label(rv, text="보기").pack(side="left")
    for val, txt in (("single", "카메라 1개"), ("dual", "카메라 2개"), ("compare", "모델 비교")):
        ttk.Radiobutton(rv, text=txt, variable=view_var, value=val).pack(side="left", padx=(8, 0))
    rc = ttk.Frame(cf)
    rc.pack(fill="x", pady=(4, 0))
    ttk.Label(rc, text="CPU 코어(전체 %d)" % CPU_COUNT).pack(side="left")
    core_vals = sorted({1, 2, 4, CPU_COUNT // 2, CPU_COUNT} - {0})
    core_vals = [c for c in core_vals if c <= CPU_COUNT]
    core_var = tk.StringVar(value=str(threads))
    core_cb = ttk.Combobox(rc, textvariable=core_var, values=[str(c) for c in core_vals], width=3, state="readonly")
    core_cb.pack(side="left", padx=6)

    def on_cores(_=None):
        for e in engs:  # 코어 수는 프로그램 전체에 한 값이라 두 카메라에 같이 적용
            e.state.update(threads=int(core_var.get()))

    core_cb.bind("<<ComboboxSelected>>", on_cores)
    ttk.Label(rc, text="줄이면 CPU 사용률이 내려감", foreground="#666").pack(side="left")

    # ---- 카메라별 설정 탭 ----
    nb = ttk.Notebook(panel)
    nb.pack(fill="x", pady=4)
    ctrls = []

    def build_controls(i):
        eng, v = engs[i], views[i]
        tab = ttk.Frame(nb, padding=2)
        nb.add(tab, text=f"카메라 {i + 1}")
        c = types.SimpleNamespace()

        # 입력 장치
        src = section(tab, "입력")
        c.src = src
        first_src = (a.source, a.source2)[i]
        src_mode = tk.StringVar(value="file" if first_src else "cam")
        file_path = tk.StringVar(value=first_src or "")
        cam_var = tk.StringVar(value=str((a.cam, a.cam2)[i]))
        res_var = tk.StringVar(value=f"{a.width}x{a.height}")
        r1 = ttk.Frame(src)
        r1.pack(fill="x")
        ttk.Radiobutton(r1, text="웹캠", variable=src_mode, value="cam").pack(side="left")
        ttk.Combobox(r1, textvariable=cam_var, values=[str(n) for n in range(5)], width=3,
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
            w, h = (int(n) for n in res_var.get().split("x"))
            if src_mode.get() == "file":
                if not file_path.get():
                    eng.on_message("영상 파일을 먼저 고르세요")
                    return
                eng.set_source(file_path.get(), 0, w, h)
            else:
                other = engs[1 - i]
                if mode["view"] == "dual" and other.cam is not None and not other.cam.is_file \
                        and other.cam_idx == int(cam_var.get()):
                    eng.on_message(f"웹캠 {cam_var.get()}번은 다른 카메라 탭에서 쓰는 중. 번호를 바꾸세요")
                    return
                eng.set_source(None, int(cam_var.get()), w, h)
            v.after_source()

        c.apply_source = apply_source
        ttk.Button(src, text="이 입력으로 열기", command=apply_source).pack(fill="x", pady=(4, 0))

        # 모델
        mf = section(tab, "모델")
        key_of = {}

        def label_of(k):
            name, kind, w, sz = keymap[k]
            note = MODEL_NOTE.get(k) or (os.path.abspath(w) if k.startswith("f") else "")
            return f"{name}  ({sz})" + (f" - {note}" if note else "")

        model_var = tk.StringVar(value=label_of(eng.state["key"]))
        size_var = tk.StringVar(value=str(eng.state["imgsz"]))

        def refresh_labels():
            key_of.clear()
            key_of.update({label_of(k): k for k in keymap})
            cb.configure(values=list(key_of))

        c.refresh_labels = refresh_labels

        def on_model(_=None):
            k = key_of[model_var.get()]
            eng.failed.discard(k)  # 다시 고르면 한 번 더 불러와 본다
            eng.state["key"] = k
            eng.state["imgsz"] = keymap[k][3]
            size_var.set(str(keymap[k][3]))
            prompt_row_state()

        cb = ttk.Combobox(mf, textvariable=model_var, state="readonly", height=16)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", on_model)

        def pick_model():
            p = filedialog.askopenfilename(title="모델 파일 고르기", initialdir=HERE,
                                           filetypes=[("모델", "*.pt *.onnx"), ("모든 파일", "*.*")])
            if not p:
                return
            k = add_model_file(keymap, p)
            for o in ctrls:  # 다른 카메라 탭 목록에도 더한다
                o.refresh_labels()
            model_var.set(label_of(k))
            on_model()
            eng.on_message(f"모델 파일 선택: {p}")

        ttk.Button(mf, text="모델 파일 불러오기...", command=pick_model).pack(fill="x", pady=(4, 0))
        r3 = ttk.Frame(mf)
        r3.pack(fill="x", pady=(6, 0))
        ttk.Label(r3, text="입력 크기").pack(side="left")
        sz_cb = ttk.Combobox(r3, textvariable=size_var, values=[str(s) for s in SIZES], width=6, state="readonly")
        sz_cb.pack(side="left", padx=6)
        sz_cb.bind("<<ComboboxSelected>>", lambda _: eng.state.update(imgsz=int(size_var.get())))
        ttk.Label(r3, text="(클수록 작은 차에 유리, 느려짐)", foreground="#666").pack(side="left")

        # YOLOE 프롬프트 (YOLOE를 골랐을 때만 켜짐)
        pf = ttk.Frame(mf)
        pf.pack(fill="x", pady=(6, 0))
        ttk.Label(pf, text="YOLOE 프롬프트").pack(side="left")
        prompt_var = tk.StringVar(value=", ".join(prompt))
        pe = ttk.Entry(pf, textvariable=prompt_var, width=18)
        pe.pack(side="left", fill="x", expand=True, padx=4)

        def apply_prompt():
            p = [s.strip() for s in prompt_var.get().split(",") if s.strip()]
            if p:
                eng.state["prompt"] = p
                eng.on_message(f"프롬프트 설정: {', '.join(p)}")

        pb = ttk.Button(pf, text="적용", width=5, command=apply_prompt)
        pb.pack(side="left")

        def prompt_row_state():
            on = keymap[eng.state["key"]][1] == "yoloe"
            for wdg in (pe, pb):
                wdg.state(["!disabled"] if on else ["disabled"])

        # 탐지 기준
        tf = section(tab, "탐지 기준")
        conf_var = tk.DoubleVar(value=a.conf)
        conf_lbl = tk.StringVar(value=f"신뢰도 기준 {a.conf:.2f}")
        ttk.Label(tf, textvariable=conf_lbl).pack(anchor="w")

        def on_conf(val):
            cf_ = round(float(val) / 0.05) * 0.05
            eng.state["conf"] = cf_
            conf_lbl.set(f"신뢰도 기준 {cf_:.2f}")

        conf_sc = ttk.Scale(tf, from_=0.05, to=0.95, variable=conf_var, command=on_conf)
        conf_sc.pack(fill="x")

        def conf_follow(e):
            """누르거나 끄는 위치로 값을 바로 옮긴다. 기본 동작은 막대 빈 곳을 누르면 양 끝으로 튄다."""
            x_lo, x_hi = conf_sc.coords(0.05)[0], conf_sc.coords(0.95)[0]  # 양 끝 값의 손잡이 위치
            val = 0.05 + 0.9 * min(1.0, max(0.0, (e.x - x_lo) / max(1, x_hi - x_lo)))
            conf_var.set(val)
            on_conf(val)
            return "break"

        conf_sc.bind("<ButtonPress-1>", conf_follow)
        conf_sc.bind("<B1-Motion>", conf_follow)
        show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tf, text="차량이 아닌 물체도 보기 (주황색)", variable=show_var,
                        command=lambda: eng.state.update(show_all=show_var.get())).pack(anchor="w", pady=(4, 0))

        # 실행
        rf = section(tab, "실행")
        r4 = ttk.Frame(rf)
        r4.pack(fill="x", pady=(0, 4))
        ttk.Label(r4, text="분석 빈도").pack(side="left")
        rate_labels = [lab for lab, _ in RATES]
        rate_var = tk.StringVar(value=next((lab for lab, val in RATES if val == eng.state["rate"]), rate_labels[0]))
        rate_cb = ttk.Combobox(r4, textvariable=rate_var, values=rate_labels, width=10, state="readonly")
        rate_cb.pack(side="left", padx=6)
        c.rate_cb = rate_cb
        rate_cb.bind("<<ComboboxSelected>>", lambda _: eng.state.update(rate=dict(RATES)[rate_var.get()]))
        pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r4, text="분석 일시정지", variable=pause_var,
                        command=lambda: eng.state.update(paused=pause_var.get())).pack(side="left", padx=(6, 0))
        r5 = ttk.Frame(rf)
        r5.pack(fill="x")
        log_var = tk.BooleanVar(value=a.log)
        ttk.Checkbutton(r5, text="결과를 CSV로 기록", variable=log_var,
                        command=lambda: eng.set_log(log_var.get())).pack(side="left")

        def snapshot():
            if mode["view"] == "compare":  # 비교 모드면 A와 B를 한 장에 나란히 저장
                save_compare()
                return
            im = v.last
            if im is None:
                return
            name = keymap[eng.state["key"]][0]
            p = os.path.join(HERE, f"live_{eng.tag}_{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(p, im)
            eng.on_message(f"화면 저장: {p}")

        ttk.Button(r5, text="현재 화면 저장", command=snapshot).pack(side="right")

        def sync():
            if key_of.get(model_var.get()) != eng.state["key"]:  # 불러오기에 실패해 YOLO26n으로 돌아간 경우
                model_var.set(label_of(eng.state["key"]))
                size_var.set(str(eng.state["imgsz"]))

        c.sync = sync
        refresh_labels()
        prompt_row_state()
        if a.log:
            eng.set_log(True)
        return c

    for i in range(2):
        ctrls.append(build_controls(i))

    # 결과 출력 (두 카메라 공통)
    of = ttk.LabelFrame(panel, text="결과", padding=6)
    of.pack(fill="both", expand=True, pady=4)
    log = tk.Text(of, height=6, width=44, font=("Consolas", 9), wrap="word")
    log.pack(fill="both", expand=True)

    def set_enabled(widget, on):
        for w in widget.winfo_children():
            if isinstance(w, ttk.Widget):
                w.state(["!disabled"] if on else ["disabled"])
            set_enabled(w, on)

    def save_compare():
        """모델 비교 화면(A | B)을 한 장으로 저장. 실시간 화면이 아니라 두 모델이 분석한 그 프레임에 상자를 그린다."""
        ims = [e.frame(analyzed=True)[0] for e in engs]
        if any(im is None for im in ims):
            return
        h = min(im.shape[0] for im in ims)
        ims = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in ims]
        names = [keymap[e.state["key"]][0] for e in engs]
        gap = np.full((h, 12, 3), 255, np.uint8)
        out = np.hstack([ims[0], gap, ims[1]])
        p = os.path.join(HERE, f"live_compare_{names[0]}_vs_{names[1]}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(p, out)
        post(f"비교 화면 저장: {p}")

    def set_view(_=None):
        new = view_var.get()
        mode["view"] = new
        e0, e1 = engs
        # 비교 모드 연결은 먼저 끊고, 필요할 때 다시 잇는다
        e0.follower = e1.leader = None
        e0.in_compare = False
        for v in views:
            v.title.pack_forget()
            v.frame.pack_forget()
        e1.close_source()  # 아래 화면은 새 모드에서 처음부터 다시 채운다
        views[1].last = None
        views[1].video.configure(image="", text="영상을 여는 중...")
        views[1].video.image = None
        if new == "single":
            e1.set_log(False)
        e0.tag, e1.tag = ("A", "B") if new == "compare" else ("cam1", "cam2")
        if new == "compare":
            views[0].title.configure(text="모델 A")
            views[1].title.configure(text="모델 B (같은 프레임)")
            e1.leader, e0.follower, e0.in_compare = e0, e1, True
        elif new == "dual":
            views[0].title.configure(text="카메라 1")
            views[1].title.configure(text="카메라 2")
        if new != "single":  # 위아래로 둘 (16:9 영상은 좌우보다 위아래가 크게 보임)
            for v, pad in ((views[0], 0), (views[1], 8)):
                v.title.pack(side="top", anchor="w", before=v.stat_lbl)
                v.frame.pack(side="top", fill="both", expand=True, pady=(pad, 0))
        else:
            views[0].frame.pack(side="top", fill="both", expand=True)
        cmp_ = new == "compare"
        nb.tab(0, text="모델 A (입력)" if cmp_ else "카메라 1")
        nb.tab(1, text="모델 B" if cmp_ else "카메라 2", state="disabled" if new == "single" else "normal")
        set_enabled(ctrls[1].src, not cmp_)  # B는 A의 입력을 같이 쓴다
        ctrls[1].rate_cb.state(["disabled"] if cmp_ else ["!disabled"])  # B는 A의 분석 빈도를 따른다
        if new == "single":
            nb.select(0)
        if new == "dual":
            ctrls[1].apply_source()

    view_var.trace_add("write", lambda *_: set_view())

    dev, devname = device_name()
    post(f"계산 장치: {dev} ({devname})")

    def show(v):
        eng = v.eng
        c = v.vcam()
        if c:
            v.play_txt.set("정지" if c.playing else "재생")
            if not v.drag["on"]:
                v.pos_var.set(max(0, c.pos))
                v.time_txt.set(f"{fmt(max(0, c.pos) / c.fps)} / {fmt(c.total / c.fps)}")
        im, sh = eng.frame()
        if im is not None:
            v.last = im
            vw, vh = max(v.holder.winfo_width(), 100), max(v.holder.winfo_height(), 100)
            h, w = im.shape[:2]
            s = min(vw / w, vh / h)
            disp = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            v.video.configure(image=photo, text="")
            v.video.image = photo
        name = keymap[eng.state["key"]][0]
        if sh["loading"]:
            v.stat.set(f"{sh['loading']} 불러오는 중...")
        elif eng.state["key"] in eng.failed:
            v.stat.set(f"{name} 불러오기 실패 (결과 칸의 오류 확인)")
        elif eng.state["paused"]:
            v.stat.set("분석 일시정지")
        else:
            frame_no = f"  |  프레임 {sh['fn']}" if mode["view"] == "compare" else ""
            v.stat.set(f"{name}  |  입력 {eng.state['imgsz']}  |  분석 1회 {sh['ms']:.0f} ms  |  "
                       f"실제 {sh['fps']:.1f}회/초  |  차량 {sh['n']}대{frame_no}")

    def tick():
        for i in range(1 if mode["view"] == "single" else 2):
            ctrls[i].sync()
            show(views[i])
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
        for e in engs:
            e.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    set_view()  # 카메라 2대 모드면 카메라 2 입력도 이때 연다
    root.after(100, ctrls[0].apply_source)
    root.after(200, tick)
    if a.max_seconds:
        root.after(int(a.max_seconds * 1000), on_close)
    root.mainloop()


def run_console(a, keymap, starts, prompt):
    n = 2 if a.dual or a.compare else 1
    tags = ("A", "B") if a.compare else ("카메라1", "카메라2")
    engs = []
    dev, devname = device_name()
    print(f"계산 장치: {dev} ({devname})")
    for i in range(n):
        out = (lambda t: lambda line: print(f"[{t}] {line}"))(tags[i]) if n == 2 else print
        tag = ("A", "B")[i] if a.compare else f"cam{i + 1}" if a.dual else ""
        eng = Engine(keymap, starts[i], a.conf, prompt, out, out, a.rate, a.threads, tag=tag)
        engs.append(eng)
        if a.log:
            eng.set_log(True)
        if a.compare and i == 1:  # 모델 B는 A가 분석한 프레임을 받는다
            eng.leader, engs[0].follower = engs[0], eng
            continue
        if not eng.set_source((a.source, a.source2)[i], (a.cam, a.cam2)[i], a.width, a.height):
            for e in engs:
                e.close()
            sys.exit(1)
    t0 = time.time()
    try:
        while not a.max_seconds or time.time() - t0 < a.max_seconds:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    for e in engs:
        e.close()


def main():
    ap = argparse.ArgumentParser(description="실시간 차량 탐지 모델 확인")
    ap.add_argument("--cam", type=int, default=0, help="시작 웹캠 번호")
    ap.add_argument("--source", help="웹캠 대신 쓸 영상 파일")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="1", help="시작 모델 (키 또는 이름, 예: 1, rtdetr, oiv7, best)")
    ap.add_argument("--weights", help="시작할 모델 파일 위치 (예: runs\\pilot_v1\\weights\\best.pt)")
    ap.add_argument("--dual", action="store_true", help="카메라 2개로 시작")
    ap.add_argument("--cam2", type=int, default=1, help="카메라 2의 웹캠 번호 (기본 1)")
    ap.add_argument("--source2", help="카메라 2에 웹캠 대신 쓸 영상 파일")
    ap.add_argument("--model2", help="카메라 2(비교 모드면 모델 B)의 시작 모델 (기본: 카메라 1과 같음)")
    ap.add_argument("--weights2", help="카메라 2의 시작 모델 파일 위치")
    ap.add_argument("--compare", action="store_true",
                    help="모델 비교: 웹캠 1대를 --model(A)과 --model2(B)로 같이 분석")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--rate", type=float, default=2, help="초당 분석 횟수 (0이면 제한 없음, 기본 2)")
    ap.add_argument("--threads", type=int, default=0, help="모델 계산에 쓸 CPU 코어 수 (기본: 전체의 절반)")
    ap.add_argument("--prompt", default="toy car,car,model car,vehicle", help="YOLOE 글 프롬프트, 쉼표로 구분")
    ap.add_argument("--log", action="store_true", help="처음부터 결과를 csv로 저장")
    ap.add_argument("--nogui", action="store_true", help="창 없이 콘솔 출력만")
    ap.add_argument("--max-seconds", type=float, default=0, help="정해진 시간 뒤 자동 종료 (0이면 계속)")
    a = ap.parse_args()
    if a.dual and a.compare:
        sys.exit("--dual과 --compare는 함께 쓸 수 없음")

    prompt = [s.strip() for s in a.prompt.split(",") if s.strip()]
    keymap = {k: (name, kind, w, sz) for k, name, kind, w, sz in MODELS}

    def resolve(model, weights):
        if weights:
            if not os.path.exists(weights):
                sys.exit(f"모델 파일이 없음: {weights}")
            return add_model_file(keymap, weights)
        k = ALIASES.get(model.lower(), model.lower())
        if k not in keymap:
            sys.exit(f"모르는 모델: {model}")
        return k

    start = resolve(a.model, a.weights)
    start2 = resolve(a.model2, a.weights2) if (a.model2 or a.weights2) else start
    if a.nogui:
        run_console(a, keymap, (start, start2), prompt)
    else:
        run_gui(a, keymap, (start, start2), prompt)


if __name__ == "__main__":
    main()
