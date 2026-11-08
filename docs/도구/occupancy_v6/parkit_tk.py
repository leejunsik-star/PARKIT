"""PARKIT 점유 판정 — tkinter 화면 공통 도구 (calibrate.py, occupancy.py가 같이 쓴다)

13 문서 3절 규칙을 따른다.
    - 글꼴은 여기서 한 번 정한다(UI_FONT, MONO_FONT).
    - 맥의 오래된 Tk(8.5)는 시작 전에 확인하고 안내한다(check_tk).
    - 단축키는 한글 입력기에 기대지 않는다: 한글 자모가 들어와도 같은 자리의 영문 키로 바꿔 처리하고,
      모든 단축키는 버튼으로도 할 수 있게 둔다.
    - Tk 표준 위젯(tk, ttk)과 표준 명령만 쓴다.
이미지 표시는 Pillow(ImageTk)를 쓴다. Pillow는 ultralytics가 같이 설치한다(따로 설치할 것 없음).
"""

import sys

import cv2

if sys.platform == "win32":
    UI_FONT, MONO_FONT = "맑은 고딕", "Consolas"
elif sys.platform == "darwin":
    UI_FONT, MONO_FONT = "Apple SD Gothic Neo", "Menlo"
else:
    UI_FONT, MONO_FONT = "Noto Sans CJK KR", "DejaVu Sans Mono"

RES_CHOICES = ["640x480", "800x600", "1280x720", "1600x900", "1920x1080"]

# 상태 이름(한글)과 색 (25 문서 D1)
STATUS_KO = {"EMPTY": "빈 칸", "OCCUPIED": "주차됨", "DYNAMIC_BLIND": "확인 불가", "STATIC_BLIND": "사각지대", None: "-"}
STATUS_HEX = {"EMPTY": "#3cc83c", "OCCUPIED": "#e62828", "DYNAMIC_BLIND": "#ffd700", "STATIC_BLIND": "#969696", None: "#dddddd"}
REASON_KO = {"INITIALIZING": "켠 직후 첫 판정 기다림", "OCCLUDED": "다른 차에 가려짐", "CAMERA_OFFLINE": "카메라 프레임이 안 옴",
             "CAMERA_STOPPED": "해상도 비율이 달라 카메라 멈춤(다시 보정)", "NO_COVERAGE": "감시하는 카메라 없음",
             None: "", "": ""}
OBS_KO = {"present": "차 있음", "absent": "차 없음", "occluded": "가려짐", "offline": "끊김", None: "모름"}

# 두벌식 한글 자모 → 같은 자리의 영문 키
_HANGUL = dict(zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ", "qwertyuiopasdfghjklzxcvbnm"))
_HANGUL.update(zip("ㅃㅉㄸㄲㅆㅒㅖ", "qwertop"))


def check_tk():
    """맥의 시스템 파이썬(Tk 8.5)은 macOS에서 창을 만들자마자 꺼진다. 창을 만들기 전에 확인한다."""
    import tkinter
    if sys.platform == "darwin" and tkinter.TkVersion < 8.6:
        sys.exit(f"지금 파이썬({sys.executable})은 Tk {tkinter.TkVersion} 버전이라 macOS에서 창을 열다 꺼집니다.\n"
                 "python.org의 Python 3.12로 가상환경을 만들어 실행하세요(13 문서).")


def key_of(event):
    """키 이벤트 → 'a'~'z', '0'~'9', 'Left', 'space', 'F1', '?' 같은 이름. 한글 자모는 영문 키로 바꾼다."""
    ks = event.keysym
    if ks in ("Left", "Right", "Up", "Down", "space", "Delete", "BackSpace", "Escape", "Return", "Tab",
              "F1", "F11", "Prior", "Next"):
        return ks
    ch = event.char or ""
    if ch in _HANGUL:
        return _HANGUL[ch]
    if ch == "?" or ks == "question":
        return "?"
    if len(ch) == 1 and ch.isprintable():
        return ch.lower()
    return ks


def is_typing(widget):
    """글자를 입력하는 위젯에 포커스가 있으면 단축키를 쓰지 않는다."""
    try:
        cls = widget.winfo_class()
    except Exception:
        return False
    return cls in ("Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox", "Text")


def parse_res(s):
    w, h = str(s).lower().replace(" ", "").split("x")
    return int(w), int(h)


class ImageView:
    """Canvas 하나에 이미지를 비율 그대로 맞춰 그린다. 원본 좌표 ↔ 화면 좌표를 바꿔 준다.
    쓰는 쪽은 fit_scale(원본 크기)로 비율을 받아, 그 크기로 줄인 이미지에 겹쳐 그린 뒤 show()로 넘긴다."""

    def __init__(self, parent, bg="#202020", width=960, height=540):
        import tkinter as tk
        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0, width=width, height=height)
        self.item = None
        self.photo = None
        self.scale = 1.0
        self.off = (0, 0)
        self.src_size = None

    def avail(self):
        self.canvas.update_idletasks()
        return max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())

    def fit_scale(self, src_w, src_h, upscale=False):
        W, H = self.avail()
        s = min(W / src_w, H / src_h)
        if not upscale:
            s = min(s, 1.0)
        return s

    def show(self, bgr, scale, src_size):
        from PIL import Image, ImageTk
        W, H = self.avail()
        h, w = bgr.shape[:2]
        self.scale = scale
        self.src_size = src_size
        self.off = ((W - w) // 2, (H - h) // 2)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if self.item is None:
            self.item = self.canvas.create_image(self.off[0], self.off[1], anchor="nw", image=self.photo)
        else:
            self.canvas.itemconfig(self.item, image=self.photo)
            self.canvas.coords(self.item, self.off[0], self.off[1])

    def to_src(self, x, y):
        """화면(캔버스) 좌표 → 원본 이미지 좌표. 이미지 밖이면 None."""
        if self.src_size is None or self.scale <= 0:
            return None
        u = (x - self.off[0]) / self.scale
        v = (y - self.off[1]) / self.scale
        if 0 <= u < self.src_size[0] and 0 <= v < self.src_size[1]:
            return u, v
        return None


def plan_angle_setting(v):
    """run.json display.plan_angle 값 → (각도, 따라갈 카메라). 숫자면 그 각도, "camera"면 카메라 시점(첫 카메라)."""
    if isinstance(v, str) and v.strip().lower() == "camera":
        return 0.0, ""
    try:
        return float(v) % 360.0, None
    except (TypeError, ValueError):
        return 0.0, None


class PlanAngleBar:
    """평면도 각도 막대 (docs/41): 슬라이더(0~359°), [0°], [↻90°], 카메라마다 [<이름> 시점].

    [<이름> 시점]을 누르면 그 카메라가 아래에서 위를 보게 돌리고, 그 카메라를 따라간다(보정을 고쳐 카메라 자세가
    바뀌면 각도도 따라 바뀜). 슬라이더, [0°], [↻90°]를 쓰면 따라가기를 끈다.
    angle_of(이름) -> 그 카메라 시점 각도(도) 또는 None(카메라 자세가 아직 없음). 이름 ""는 첫 카메라.
    쓰는 쪽은 그릴 때마다 poll()로 각도를 받는다."""

    def __init__(self, parent, angle_of, cams=("",), angle=0.0, follow=None, on_note=None):
        import tkinter as tk
        from tkinter import ttk
        self.angle_of = angle_of
        self.on_note = on_note or (lambda s: None)
        self.angle = float(angle) % 360.0
        self.follow = follow                       # 따라가는 카메라 이름 (None이면 손으로 정한 각도)
        self.frame = ttk.Frame(parent)
        r1 = ttk.Frame(self.frame)
        r1.pack(fill="x")
        ttk.Label(r1, text="도면 각도").pack(side="left")
        self.var = tk.DoubleVar(value=self.angle)
        ttk.Scale(r1, from_=0, to=359, orient="horizontal", length=170, variable=self.var,
                  command=self._on_scale).pack(side="left", padx=4, fill="x", expand=True)
        self.lbl = ttk.Label(r1, width=5, anchor="e")
        self.lbl.pack(side="left")
        r2 = ttk.Frame(self.frame)
        r2.pack(fill="x", pady=(2, 0))
        ttk.Button(r2, text="0°", width=4, command=lambda: self.set(0.0)).pack(side="left")
        ttk.Button(r2, text="↻90°", width=5, command=lambda: self.set(self.angle + 90.0)).pack(side="left", padx=2)
        for name in cams:
            ttk.Button(r2, text=f"{name or '카메라'} 시점", command=lambda n=name: self.set_camera(n)).pack(side="left", padx=2)
        self._show()

    def _show(self):
        self.lbl.configure(text=f"{self.angle:.0f}°" + ("*" if self.follow is not None else ""))

    def _on_scale(self, v):
        self.angle = round(float(v)) % 360.0
        self.follow = None
        self._show()

    def set(self, angle):
        """손으로 정한 각도 (따라가기 끔)."""
        self.angle = float(angle) % 360.0
        self.follow = None
        self.var.set(self.angle)
        self._show()

    def set_camera(self, name):
        a = self.angle_of(name)
        if a is None:
            self.on_note(f"{name or '카메라'} 자세가 아직 없어 카메라 시점으로 돌릴 수 없습니다(기준점 4개 이상 찍고 계산).")
            return
        self.follow = name
        self.angle = float(a) % 360.0
        self.var.set(self.angle)
        self._show()

    def poll(self):
        """지금 각도. 카메라를 따라가는 중이면 그 카메라 자세에서 다시 구한다."""
        if self.follow is not None:
            a = self.angle_of(self.follow)
            if a is not None and abs(((a - self.angle) + 180) % 360 - 180) > 0.5:
                self.angle = float(a) % 360.0
                self.var.set(self.angle)
                self._show()
        return self.angle


class ScrollFrame:
    """세로 스크롤되는 틀. 내용은 .inner 에 넣고, .outer 를 배치한다.
    마우스가 틀 위에 있을 때만 휠로 굴린다(Windows·맥은 MouseWheel, 리눅스는 Button-4/5)."""

    def __init__(self, parent, width=400):
        import tkinter as tk
        from tkinter import ttk
        self.outer = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, width=width)
        self.bar = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner = ttk.Frame(self.canvas, padding=(8, 4))
        self.win = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.outer.bind("<Enter>", lambda e: self._wheel(True))
        self.outer.bind("<Leave>", lambda e: self._wheel(False))

    def _on_inner(self, e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        w = self.inner.winfo_reqwidth()
        if int(self.canvas.cget("width")) != w:
            self.canvas.configure(width=w)

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self.win, width=max(e.width, self.inner.winfo_reqwidth()))

    def _wheel(self, on):
        c = self.canvas
        if on:
            c.bind_all("<MouseWheel>", self._scroll)
            c.bind_all("<Button-4>", lambda e: c.yview_scroll(-3, "units"))
            c.bind_all("<Button-5>", lambda e: c.yview_scroll(3, "units"))
        else:
            for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                c.unbind_all(s)

    def _scroll(self, e):
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
            return
        d = e.delta if sys.platform == "darwin" else e.delta / 120
        self.canvas.yview_scroll(int(-d * 3) or (-1 if e.delta > 0 else 1), "units")


class HelpWindow:
    """단축키와 마우스 사용법 표. 같은 창이 이미 열려 있으면 닫는다(토글)."""

    def __init__(self, root, title, sections, note=""):
        self.root, self.title, self.sections, self.note = root, title, sections, note
        self.win = None

    def toggle(self):
        import tkinter as tk
        from tkinter import ttk
        if self.win is not None and self.win.winfo_exists():
            self.win.destroy()
            self.win = None
            return
        w = tk.Toplevel(self.root)
        w.title(self.title)
        w.transient(self.root)
        f = ttk.Frame(w, padding=10)
        f.pack(fill="both", expand=True)
        tv = ttk.Treeview(f, columns=("key", "desc"), show="tree headings", height=24)
        tv.heading("#0", text="구분")
        tv.heading("key", text="키 / 마우스")
        tv.heading("desc", text="하는 일")
        tv.column("#0", width=150, stretch=False)
        tv.column("key", width=170, stretch=False)
        tv.column("desc", width=430)
        for sec, rows in self.sections:
            p = tv.insert("", "end", text=sec, open=True)
            for k, d in rows:
                tv.insert(p, "end", text="", values=(k, d))
        sb = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        if self.note:
            ttk.Label(f, text=self.note, wraplength=740, foreground="#555").grid(row=1, column=0, columnspan=2,
                                                                                sticky="w", pady=(8, 0))
        ttk.Button(f, text="닫기 (Esc, ?)", command=self.toggle).grid(row=2, column=0, columnspan=2, pady=(8, 0))
        w.bind("<Escape>", lambda e: self.toggle())
        w.bind("<Key>", lambda e: self.toggle() if key_of(e) in ("?", "F1") else None)
        w.geometry("800x620")
        self.win = w


def setup_style(root):
    from tkinter import ttk
    st = ttk.Style(root)
    for name in ("TLabel", "TButton", "TCheckbutton", "TRadiobutton", "TLabelframe.Label", "TCombobox", "TSpinbox"):
        st.configure(name, font=(UI_FONT, 10))
    st.configure("Treeview", font=(UI_FONT, 10), rowheight=24)
    st.configure("Treeview.Heading", font=(UI_FONT, 10, "bold"))
    st.configure("Big.TLabel", font=(UI_FONT, 12, "bold"))
    st.configure("Guide.TLabel", font=(UI_FONT, 12, "bold"), foreground="#ffffff", background="#2b5797", padding=8)
    st.configure("Warn.TLabel", font=(UI_FONT, 10, "bold"), foreground="#c00000")
    st.configure("Hint.TLabel", font=(UI_FONT, 10), foreground="#333333", background="#f0f0d8", padding=4)
    st.configure("Accent.TButton", font=(UI_FONT, 10, "bold"))
    return st
