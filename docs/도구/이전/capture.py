"""학습용 사진·영상 촬영 프로그램 (PARKIT 테스트베드용)

웹캠 화면을 띄워 두고 스페이스바를 누를 때마다 원본 사진(상자 없음)을 번호를 매겨 저장한다.
녹화 버튼(또는 R 키)으로 영상도 저장한다. 실시간 모델 확인 프로그램(live_detect.py)과 같은 방식
(OpenCV, MJPEG, 같은 해상도)으로 웹캠을 열기 때문에, 시연 때 모델이 보는 화면과 같은 사진이 모인다.

설치(한 번만):
    pip install "opencv-python==4.10.0.84" pillow
    (live_detect.py용 설치를 이미 했다면 따로 할 것 없음)

사용법:
    python capture.py                         웹캠 0번, 1280x720으로 시작
    python capture.py --cam 1 --width 1920 --height 1080
    python capture.py --prefix A              파일 이름 앞부분 (사람·묶음별로 다르게 하면 합칠 때 안 겹침)
    python capture.py --out D:\\photos         저장 폴더 (기본: Parkit\\dataset)
    python capture.py --dual --cam 0 --cam2 1  웹캠 2개로 동시에 찍기 (A_cam1_0001.jpg, A_cam2_0001.jpg처럼 짝으로 저장)

키:
    스페이스바   사진 저장 (카메라 2개면 두 카메라 모두)
    R           녹화 시작/정지 (카메라 2개면 두 카메라 모두)
    Esc         종료
"""
import argparse, os, platform, re, threading, time, types
from collections import deque
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "dataset"))  # docs\도구 → Parkit\dataset (영상은 dataset\video)
RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]


def open_camera(idx, w, h):
    """live_detect.py와 같은 순서로 웹캠을 연다(Windows는 DSHOW → MSMF → 기본)."""
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


def save_jpg(path, im):
    """한글이 들어간 경로에서도 저장되도록 imencode + tofile을 쓴다(cv2.imwrite는 한글 경로에서 실패)."""
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


class Camera(threading.Thread):
    """웹캠을 계속 읽어 최신 프레임을 들고 있고, 녹화 중이면 읽은 프레임을 모두 영상 파일에 쓴다."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame, self.n, self.alive = None, 0, True
        self.times = deque(maxlen=60)
        self.writer, self.rec_frames, self.rec_start = None, 0, 0.0
        self.lock = threading.Lock()

    def run(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.n = f, self.n + 1
                self.times.append(time.time())
                if self.writer is not None:
                    self.writer.write(f)
                    self.rec_frames += 1
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        self.cap.release()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def fps(self):
        with self.lock:
            t = list(self.times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 10 and t[-1] > t[0] else 0.0

    def start_record(self, path):
        """녹화를 시작한다. 실제 웹캠 속도로 기록해야 재생 속도가 맞으므로 측정한 fps를 쓴다."""
        f = self.latest()
        if f is None:
            return None, "아직 카메라 화면이 없음"
        fps = round(self.fps(), 1) or 30.0
        size = (f.shape[1], f.shape[0])
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not w.isOpened():  # mp4를 못 쓰는 환경이면 avi(MJPG)로
            path = os.path.splitext(path)[0] + ".avi"
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
            if not w.isOpened():
                return None, "영상 파일을 만들지 못함. 저장 폴더 경로에 한글이 있으면 영어 경로로 바꿔 보세요."
        with self.lock:
            self.writer, self.rec_frames, self.rec_start = w, 0, time.time()
        return path, f"녹화 시작: {os.path.basename(path)} ({size[0]}x{size[1]}, {fps}fps)"

    def stop_record(self):
        with self.lock:
            w, n, t = self.writer, self.rec_frames, time.time() - self.rec_start
            self.writer = None
        if w is not None:
            w.release()
        return n, t

    @property
    def recording(self):
        return self.writer is not None


def clean_prefix(s):
    """파일 이름 앞부분은 영문·숫자·_·-만 쓴다(OpenCV 영상 저장이 한글 이름에서 실패하기 때문)."""
    return re.sub(r"[^A-Za-z0-9_-]", "", s) or "img"


def next_number(folder, prefix):
    """폴더에 있는 prefix_0001.jpg 같은 파일 중 가장 큰 번호 + 1."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.jpg$", re.I)
    nums = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))] if os.path.isdir(folder) else []
    return max(nums, default=0) + 1


def count_jpg(folder):
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(".jpg")) if os.path.isdir(folder) else 0


def run_gui(args):
    import tkinter as tk
    from tkinter import ttk, filedialog
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("PARKIT 학습용 촬영")
    root.geometry("1280x760")
    root.minsize(980, 600)

    # cams: 카메라 1, 2의 Camera. 카메라 2개 모드가 아니면 두 번째는 None
    st = {"cams": [None, None], "out": args.out, "shots": 0, "flash": 0.0, "rec_paths": [None, None]}

    # ---- 오른쪽: 옵션 패널 (먼저 붙여야 영상이 공간을 다 차지하지 않음) ----
    panel = ttk.Frame(root, width=340)
    panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
    panel.pack_propagate(False)

    # ---- 왼쪽: 영상 (카메라 2개면 위아래로 둘) ----
    left = ttk.Frame(root)
    left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    stat = tk.StringVar(value="")
    ttk.Label(left, textvariable=stat, font=("맑은 고딕", 11, "bold")).pack(side="bottom", anchor="w", pady=(6, 0))
    views = []
    for i in range(2):
        v = types.SimpleNamespace()
        v.frame = ttk.Frame(left)
        v.title = ttk.Label(v.frame, text=f"카메라 {i + 1}", font=("맑은 고딕", 10, "bold"))
        v.holder = tk.Frame(v.frame, bg="#202020")
        v.holder.pack(side="bottom", fill="both", expand=True)
        v.holder.pack_propagate(False)
        v.video = tk.Label(v.holder, bg="#202020", text="카메라를 여는 중...", fg="white")
        v.video.pack(fill="both", expand=True)
        views.append(v)

    def post(msg):
        print(msg, flush=True)
        log.insert("end", f"{datetime.now():%H:%M:%S} {msg}\n")
        log.see("end")
        if int(log.index("end-1c").split(".")[0]) > 300:
            log.delete("1.0", "50.0")

    # 카메라
    cf = ttk.LabelFrame(panel, text="카메라", padding=6)
    cf.pack(fill="x", pady=(0, 6))
    cam_vars = [tk.StringVar(value=str(args.cam)), tk.StringVar(value=str(args.cam2))]
    res_var = tk.StringVar(value=f"{args.width}x{args.height}")
    dual_var = tk.BooleanVar(value=args.dual)
    row = ttk.Frame(cf)
    row.pack(fill="x")
    ttk.Label(row, text="번호").pack(side="left")
    cb_cam = ttk.Combobox(row, textvariable=cam_vars[0], values=[str(i) for i in range(5)], width=3, state="readonly")
    cb_cam.pack(side="left", padx=(4, 10))
    ttk.Label(row, text="해상도").pack(side="left")
    cb_res = ttk.Combobox(row, textvariable=res_var, width=10, state="readonly",
                          values=sorted(set(RESOLUTIONS + [res_var.get()]), key=lambda r: int(r.split("x")[0])))
    cb_res.pack(side="left", padx=(4, 0))
    row = ttk.Frame(cf)
    row.pack(fill="x", pady=(4, 0))
    ttk.Checkbutton(row, text="카메라 2개 동시 촬영", variable=dual_var,
                    command=lambda: (set_layout(), open_cam())).pack(side="left")
    ttk.Label(row, text="2번 카메라").pack(side="left", padx=(10, 0))
    cb_cam2 = ttk.Combobox(row, textvariable=cam_vars[1], values=[str(i) for i in range(5)], width=3, state="readonly")
    cb_cam2.pack(side="left", padx=(4, 0))
    ttk.Label(cf, text="실시간 프로그램과 같은 해상도로 찍는 것을 권장", foreground="#666").pack(anchor="w", pady=(4, 0))
    ttk.Button(cf, text="이 카메라로 열기", command=lambda: open_cam()).pack(fill="x", pady=(4, 0))

    # 저장 위치
    sf = ttk.LabelFrame(panel, text="저장 위치", padding=6)
    sf.pack(fill="x", pady=(0, 6))
    out_txt = tk.StringVar()
    ttk.Label(sf, textvariable=out_txt, wraplength=310, foreground="#333").pack(anchor="w")
    ttk.Button(sf, text="폴더 바꾸기...", command=lambda: choose_folder()).pack(anchor="w", pady=(4, 6))
    row = ttk.Frame(sf)
    row.pack(fill="x")
    ttk.Label(row, text="파일 이름 앞부분").pack(side="left")
    prefix_var = tk.StringVar(value=clean_prefix(args.prefix))
    ent = ttk.Entry(row, textvariable=prefix_var, width=10)
    ent.pack(side="left", padx=(4, 0))
    next_txt = tk.StringVar()
    ttk.Label(sf, textvariable=next_txt, foreground="#666").pack(anchor="w", pady=(4, 0))
    ttk.Label(sf, text="영문·숫자만. 사람이나 묶음마다 다르게 정하면\n합칠 때 이름이 겹치지 않음 (예: A, B, C)",
              foreground="#666").pack(anchor="w", pady=(2, 0))

    # 사진
    pf = ttk.LabelFrame(panel, text="사진", padding=6)
    pf.pack(fill="x", pady=(0, 6))
    ttk.Button(pf, text="사진 찍기  (스페이스바)", command=lambda: shoot()).pack(fill="x", ipady=6)
    shot_txt = tk.StringVar()
    ttk.Label(pf, textvariable=shot_txt).pack(anchor="w", pady=(4, 0))

    # 영상
    vf = ttk.LabelFrame(panel, text="영상", padding=6)
    vf.pack(fill="x", pady=(0, 6))
    rec_btn_txt = tk.StringVar(value="녹화 시작  (R)")
    ttk.Button(vf, textvariable=rec_btn_txt, command=lambda: toggle_rec()).pack(fill="x", ipady=6)
    rec_txt = tk.StringVar(value="영상은 저장 폴더 안의 video 폴더에 저장")
    ttk.Label(vf, textvariable=rec_txt).pack(anchor="w", pady=(4, 0))

    # 기록
    lf = ttk.LabelFrame(panel, text="기록", padding=6)
    lf.pack(fill="both", expand=True)
    log = tk.Text(lf, height=8, width=40, font=("Consolas", 9), wrap="word")
    log.pack(fill="both", expand=True)

    def prefix():
        p = clean_prefix(prefix_var.get())
        if p != prefix_var.get():
            prefix_var.set(p)
        return p

    def names():
        """카메라마다 붙일 파일 이름 앞부분. 2개 모드면 A_cam1, A_cam2처럼 카메라를 붙인다."""
        if dual_var.get():
            return [f"{prefix()}_cam1", f"{prefix()}_cam2"]
        return [prefix()]

    def next_no():
        """2개 모드에서는 두 카메라가 같은 번호를 쓰도록 둘 중 큰 번호를 고른다(같은 순간에 찍은 짝이 번호로 맞음)."""
        return max(next_number(st["out"], p) for p in names())

    def refresh_names(*_):
        out_txt.set(st["out"])
        n = next_no()
        next_txt.set("다음 파일: " + ", ".join(f"{p}_{n:04d}.jpg" for p in names()))
        shot_txt.set(f"이번에 찍은 사진 {st['shots']}장 / 폴더 전체 {count_jpg(st['out'])}장")

    def choose_folder():
        d = filedialog.askdirectory(initialdir=st["out"] if os.path.isdir(st["out"]) else HERE, title="저장 폴더 고르기")
        if d:
            st["out"] = os.path.normpath(d)
            refresh_names()
            post(f"저장 폴더: {st['out']}")

    def recording():
        return any(c is not None and c.recording for c in st["cams"])

    def close_cams():
        if recording():
            toggle_rec()
        for i, c in enumerate(st["cams"]):
            if c is not None:
                c.alive = False
                c.join(timeout=2)
            st["cams"][i] = None

    def set_layout():
        for v in views:
            v.frame.pack_forget()
            v.title.pack_forget()
        if dual_var.get():
            for v, pad in ((views[0], 0), (views[1], 8)):
                v.title.pack(side="top", anchor="w")
                v.frame.pack(side="top", fill="both", expand=True, pady=(pad, 0))
        else:
            views[0].frame.pack(side="top", fill="both", expand=True)
        cb_cam2.state(["!disabled"] if dual_var.get() else ["disabled"])
        refresh_names()

    def open_cam():
        close_cams()
        w, h = (int(v) for v in res_var.get().split("x"))
        n_cam = 2 if dual_var.get() else 1
        idx = [int(cam_vars[i].get()) for i in range(n_cam)]
        if n_cam == 2 and idx[0] == idx[1]:
            post("카메라 1과 2의 번호가 같음. 2번 카메라 번호를 바꾸세요")
            n_cam = 1
        for i in range(n_cam):
            views[i].video.configure(image="", text="카메라를 여는 중...")
            views[i].video.image = None
        root.update_idletasks()
        for i in range(n_cam):
            cap, msg = open_camera(idx[i], w, h)
            post(f"[카메라{i + 1}] {msg}" if dual_var.get() else msg)
            if cap is not None:
                st["cams"][i] = Camera(cap)
                st["cams"][i].start()
            else:
                views[i].video.configure(text=msg)
        root.focus_set()

    def shoot():
        n_cam = 2 if dual_var.get() else 1
        cams = st["cams"][:n_cam]
        frames = [c.latest() if c else None for c in cams]  # 두 카메라의 최신 프레임을 바로 이어서 잡는다
        if any(f is None for f in frames):
            missing = [str(i + 1) for i, f in enumerate(frames) if f is None]
            post(f"카메라 {', '.join(missing)}의 화면이 없어 저장하지 않음")
            return
        os.makedirs(st["out"], exist_ok=True)
        n = next_no()
        saved = []
        for p, f in zip(names(), frames):
            path = os.path.join(st["out"], f"{p}_{n:04d}.jpg")
            if save_jpg(path, f):
                saved.append(f"{os.path.basename(path)} ({f.shape[1]}x{f.shape[0]})")
            else:
                post(f"사진 저장 실패: {path}")
        if saved:
            st["shots"] += 1
            st["flash"] = time.time()
            post("사진 저장: " + ", ".join(saved))
        refresh_names()

    def toggle_rec():
        cams = [c for c in st["cams"] if c is not None]
        if not cams:
            post("카메라가 열려 있지 않음")
            return
        if recording():
            for i, c in enumerate(st["cams"]):
                if c is not None and c.recording:
                    n, t = c.stop_record()
                    post(f"녹화 끝: {os.path.basename(st['rec_paths'][i])} ({t:.1f}초, {n}프레임)")
            rec_btn_txt.set("녹화 시작  (R)")
            rec_txt.set("영상은 저장 폴더 안의 video 폴더에 저장")
            return
        vdir = os.path.join(st["out"], "video")
        os.makedirs(vdir, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        for i, (c, p) in enumerate(zip(st["cams"], names())):
            if c is None:
                continue
            path, msg = c.start_record(os.path.join(vdir, f"{p}_{stamp}.mp4"))
            post(msg)
            if path:
                st["rec_paths"][i] = path
        if recording():
            rec_btn_txt.set("녹화 정지  (R)")

    # 키: 글자 입력 칸에 있을 때는 동작하지 않게 한다
    def typing():
        w = root.focus_get()
        return w is not None and w.winfo_class() in ("TEntry", "Entry")

    def on_space(_):
        if typing():
            return None
        shoot()
        return "break"  # 버튼에 포커스가 있어도 버튼이 눌리지 않게

    def on_r(_):
        if typing():
            return None
        toggle_rec()
        return "break"

    root.bind_all("<KeyPress-space>", on_space)
    for k in ("r", "R"):
        root.bind_all(f"<KeyPress-{k}>", on_r)
    root.bind_all("<Key>", lambda e: on_r(e) if e.char == "ㄱ" else None, add="+")  # 한글 입력 상태에서 R
    root.bind_all("<Escape>", lambda _: on_close())
    for cb in (cb_cam, cb_cam2, cb_res):
        cb.bind("<<ComboboxSelected>>", lambda _: root.focus_set())
    ent.bind("<Return>", lambda _: (refresh_names(), root.focus_set()))
    ent.bind("<FocusOut>", refresh_names)
    for v in views:
        v.video.bind("<Button-1>", lambda _: root.focus_set())

    def show(v, c):
        f = c.latest() if c else None
        if f is None:
            return None
        hw, hh = v.holder.winfo_width(), v.holder.winfo_height()
        if hw > 10 and hh > 10:
            s = min(hw / f.shape[1], hh / f.shape[0])
            disp = cv2.resize(f, (max(1, int(f.shape[1] * s)), max(1, int(f.shape[0] * s))))
            # 아래 표시는 화면에만 그린다. 저장되는 사진·영상에는 들어가지 않는다.
            if time.time() - st["flash"] < 0.15:
                cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), (255, 255, 255), 12)
            if c.recording:
                cv2.circle(disp, (24, 24), 10, (0, 0, 255), -1)
                cv2.putText(disp, "REC", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
            v.video.configure(image=photo, text="")
            v.video.image = photo
        return f

    def tick():
        n_cam = 2 if dual_var.get() else 1
        parts = []
        for i in range(n_cam):
            c = st["cams"][i]
            f = show(views[i], c)
            if f is not None:
                tag = f"카메라{i + 1} " if n_cam == 2 else ""
                parts.append(f"{tag}{f.shape[1]}x{f.shape[0]} {c.fps():.1f}fps")
        rec = [c for c in st["cams"][:n_cam] if c is not None and c.recording]
        if rec:
            c = rec[0]
            t = time.time() - c.rec_start
            rec_txt.set(f"● 녹화 중 {int(t // 60):02d}:{int(t % 60):02d} ({c.rec_frames}프레임)")
            parts.append("녹화 중")
        if parts:
            stat.set("  |  ".join(parts + [f"사진 {st['shots']}번"]))
        root.after(33, tick)

    def on_close():
        close_cams()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    set_layout()
    post(f"저장 폴더: {st['out']}")
    root.after(100, open_cam)
    root.after(200, tick)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="PARKIT 학습용 사진·영상 촬영")
    ap.add_argument("--cam", type=int, default=0, help="웹캠 번호 (기본 0)")
    ap.add_argument("--dual", action="store_true", help="웹캠 2개로 동시에 찍기")
    ap.add_argument("--cam2", type=int, default=1, help="2번 카메라의 웹캠 번호 (기본 1)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=DEFAULT_OUT, help="저장 폴더 (기본: Parkit\\dataset)")
    ap.add_argument("--prefix", default="img", help="파일 이름 앞부분 (영문·숫자)")
    run_gui(ap.parse_args())


if __name__ == "__main__":
    main()
