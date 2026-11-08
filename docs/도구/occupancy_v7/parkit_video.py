"""PARKIT 점유 판정 — 입력 열기와 프레임 받기 (25 문서 C1, C2, P11, L4)

입력 세 가지
    웹캠 번호(0, 1, ...)   Windows는 DSHOW → MSMF → ANY 순서로 열고 MJPG로 받는다(live_detect_winmac.py와 같은 방법).
    rtsp/http 주소         FFMPEG로 연다. 끊기면 몇 초마다 다시 연다.
    영상 파일(mp4 등)      영상 시각을 그대로 쓴다. 모든 프레임을 차례로 넘기고, N프레임마다 하나를 분석한다.
                           그래서 PC가 느리든 빠르든 같은 파일, 같은 N이면 늘 같은 결과가 나온다(C2).

카메라마다 프레임을 받는 스레드를 따로 둔다(P11). 모델은 판정 프로그램이 하나만 만들어 같이 쓴다.
"""

import os, platform, queue, sys, threading, time

import cv2


# 맥의 OpenCV(AVFoundation)는 웹캠을 메인 스레드에서 열어야 카메라 권한을 물을 수 있다.
# 그래서 맥에서는 끊긴 웹캠을 프레임 스레드가 직접 다시 열지 않고, 메인 스레드가 reopen_on_main()으로 연다.
MAIN_THREAD_OPEN = sys.platform == "darwin"


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
        self.times = []                 # 최근 프레임 받은 시각 (fps 계산)
        self.want_reopen = False        # 맥: 메인 스레드가 다시 열어 주기를 기다림
        self.last_try = 0.0

    def fps(self):
        with self.lock:
            ts = list(self.times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

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
            self.times.append(t)
            if len(self.times) > 30:
                del self.times[0]
        with self.rec_lock:
            if self.rec is not None:
                self.rec.write(f, t)

    def run(self):
        fails = 0
        while self.running:
            if self.cap is None:
                if MAIN_THREAD_OPEN and self.kind == "camera":
                    self.want_reopen = True    # 맥: 메인 스레드가 reopen_on_main()으로 연다
                    time.sleep(0.2)
                    continue
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

    def reopen_on_main(self):
        """메인 스레드에서 부른다. 맥에서 끊긴 웹캠을 2초마다 다시 열어 본다(다른 운영체제는 할 일 없음)."""
        if not (self.want_reopen and self.running) or time.time() - self.last_try < 2.0:
            return
        self.last_try = time.time()
        if self.open():
            self.want_reopen = False
            print(f"[{self.name}] 다시 연결됨: {self.msg}")

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
    """영상 파일. 모든 프레임을 차례로 넘긴다: (영상 시각, 프레임 번호, 프레임 또는 None).
    every_n 프레임마다 하나(분석할 프레임)는 반드시 디코딩한다. need_all이 False면 나머지는 디코딩하지 않고
    건너뛰어(None) 빠르게 돈다(창 없이 돌릴 때). 두 값은 돌아가는 중에 바꿔도 된다."""

    def __init__(self, name, path, every_n=15, loop=False, need_all=True):
        super().__init__(daemon=True)
        self.name, self.src = name, path
        self.kind = "file"
        self.every_n = max(1, int(every_n))
        self.need_all = need_all
        self.loop = loop
        self.q = queue.Queue(maxsize=8)
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
        idx, base = -1, 0.0
        while self.running:
            ok = self.cap.grab()
            if not ok:
                if self.loop and idx >= 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    base += (idx + 1) / self.fps
                    idx = -1
                    continue
                break
            idx += 1
            frame = None
            if self.need_all or idx % self.every_n == 0:
                ok, frame = self.cap.retrieve()
                if not ok:
                    frame = None
            self._put((base + idx / self.fps, idx, frame))
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
        """(영상 시각, 프레임 번호, 프레임 또는 None) 또는 끝나면 None. 시간이 지나도 없으면 "wait"."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return "wait"

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


def make_source(name, src, width, height, every_n=15, loop=False, need_all=True):
    if source_kind(src) == "file":
        return FileSource(name, src, every_n, loop, need_all)
    return LiveSource(name, int(src) if source_kind(src) == "camera" else src, width, height)
