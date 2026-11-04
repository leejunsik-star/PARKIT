"""PARKIT 점유 판정 — 입력 열기와 프레임 받기 (25 문서 C1, C2, P11, L4)

입력 세 가지
    웹캠 번호(0, 1, ...)   Windows는 DSHOW → MSMF → ANY 순서로 열고 MJPG로 받는다(live_detect_winmac.py와 같은 방법).
    rtsp/http 주소         FFMPEG로 연다. 끊기면 몇 초마다 다시 연다.
    영상 파일(mp4 등)      영상 시각을 그대로 쓴다. 분석할 프레임(초당 analyze_fps장)만 디코딩해 차례로 넘긴다.
                           그래서 PC가 느리든 빠르든 같은 파일은 늘 같은 결과가 나온다(C2).

카메라마다 프레임을 받는 스레드를 따로 둔다(P11). 모델은 판정 프로그램이 하나만 만들어 같이 쓴다.
"""

import os, platform, queue, threading, time

import cv2


def source_kind(src):
    if isinstance(src, int) or (isinstance(src, str) and src.strip().isdigit()):
        return "camera"
    if isinstance(src, str) and "://" in src:
        return "stream"
    return "file"


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
            return cap, f, f"웹캠 {idx}번 열림 ({name}), {f.shape[1]}x{f.shape[0]}"
        cap.release()
    return None, None, f"웹캠 {idx}번을 열지 못함. 카메라 앱 등 다른 프로그램을 닫거나 번호를 바꿔 보세요."


class Recorder:
    """원본 녹화(L4). 실제 받은 시각에 맞춰 일정한 fps로 쓴다(프레임이 늦으면 앞 프레임을 한 번 더 쓴다).
    그래서 녹화한 파일을 다시 돌리면 영상 시각이 실제 시각과 같다."""

    def __init__(self, path, size, fps=30.0):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.fps = fps
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
        self.t0 = None
        self.n = 0
        self.size = tuple(size)

    def ok(self):
        return self.w.isOpened()

    def write(self, frame, t):
        if self.t0 is None:
            self.t0 = t
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        want = int((t - self.t0) * self.fps) + 1
        while self.n < want:
            self.w.write(frame)
            self.n += 1

    def close(self):
        self.w.release()


class LiveSource(threading.Thread):
    """웹캠 / rtsp. 가장 최근 프레임 하나만 들고 있다."""

    def __init__(self, name, src, width, height):
        super().__init__(daemon=True)
        self.name, self.src, self.req = name, src, (int(width), int(height))
        self.kind = source_kind(src)
        self.lock = threading.Lock()
        self.frame, self.t, self.n = None, None, 0
        self.size = None
        self.msg = ""
        self.running = True
        self.cap = None
        self.rec = None
        self.rec_lock = threading.Lock()

    def open(self):
        if self.kind == "camera":
            self.cap, f, self.msg = open_camera(int(self.src), *self.req)
        else:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            ok, f = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                self.cap.release()
                self.cap, f = None, None
                self.msg = f"주소를 열지 못함: {self.src}"
            else:
                self.msg = f"주소 열림: {self.src}, {f.shape[1]}x{f.shape[0]}"
        if self.cap is not None:
            self.size = (f.shape[1], f.shape[0])
            self._store(f)
        return self.cap is not None

    def _store(self, f):
        t = time.time()
        with self.lock:
            self.frame, self.t, self.n = f, t, self.n + 1
        with self.rec_lock:
            if self.rec is not None:
                self.rec.write(f, t)

    def run(self):
        fails = 0
        while self.running:
            if self.cap is None:
                time.sleep(2.0)                # 끊긴 뒤 다시 열기
                if self.running and self.open():
                    print(f"[{self.name}] 다시 연결됨: {self.msg}")
                continue
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails >= 30:
                    print(f"[{self.name}] 프레임이 안 옵니다. 다시 열어 봅니다.")
                    self.cap.release()
                    self.cap = None
                    fails = 0
                else:
                    time.sleep(0.05)
                continue
            fails = 0
            self._store(f)

    def latest(self):
        with self.lock:
            return (None, None, 0) if self.frame is None else (self.frame, self.t, self.n)

    def start_record(self, path):
        with self.rec_lock:
            if self.rec is None and self.size:
                self.rec = Recorder(path, self.size)
                if not self.rec.ok():
                    self.rec = None
                    return False
        return True

    def stop_record(self):
        with self.rec_lock:
            if self.rec is not None:
                self.rec.close()
                p, self.rec = self.rec.path, None
                return p
        return None

    def recording(self):
        return self.rec is not None

    def close(self):
        self.running = False
        self.stop_record()
        if self.cap is not None:
            time.sleep(0.1)
            self.cap.release()


class FileSource(threading.Thread):
    """영상 파일. 영상 시각 k / analyze_fps 에 해당하는 프레임만 디코딩해서 큐에 넣는다."""

    def __init__(self, name, path, analyze_fps, loop=False):
        super().__init__(daemon=True)
        self.name, self.src = name, path
        self.kind = "file"
        self.analyze_fps = float(analyze_fps)
        self.loop = loop
        self.q = queue.Queue(maxsize=4)
        self.running = True
        self.cap = None
        self.size = None
        self.fps = 30.0
        self.n_frames = 0
        self.msg = ""
        self.ended = False
        self.rec = None

    def open(self):
        if not os.path.exists(self.src):
            self.msg = f"영상 파일이 없음: {self.src}"
            return False
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            self.msg = f"영상을 열지 못함: {self.src}"
            self.cap = None
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.msg = (f"영상 열림: {os.path.basename(self.src)}, {self.size[0]}x{self.size[1]}, "
                    f"{self.fps:.1f}fps, {self.n_frames / self.fps:.1f}초")
        return True

    def run(self):
        k, idx, base = 0, -1, 0.0
        while self.running:
            t = k / self.analyze_fps
            target = int(round(t * self.fps))
            got = None
            while idx < target:
                ok = self.cap.grab()
                if not ok:
                    break
                idx += 1
                if idx == target:
                    ok, got = self.cap.retrieve()
                    if not ok:
                        got = None
            if got is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    base += idx / self.fps
                    idx, k = -1, 0
                    continue
                break
            self._put((base + t, target, got))
            k += 1
        self._put(None)
        self.ended = True

    def _put(self, item):
        while self.running:
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def get(self, timeout=10.0):
        """(영상 시각, 프레임 번호, 프레임) 또는 끝나면 None."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def start_record(self, path):
        return False

    def stop_record(self):
        return None

    def recording(self):
        return False

    def close(self):
        self.running = False
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self.cap is not None:
            time.sleep(0.05)
            self.cap.release()


def make_source(name, src, width, height, analyze_fps, loop=False):
    if source_kind(src) == "file":
        return FileSource(name, src, analyze_fps, loop)
    return LiveSource(name, int(src) if source_kind(src) == "camera" else src, width, height)
