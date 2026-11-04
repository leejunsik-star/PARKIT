"""웹캠 점검 스크립트 (PARKIT 테스트베드용)

웹캠이 MJPEG를 지원하는지, 해상도별로 실제 몇 fps가 나오는지 확인한다.
사용법:  python webcam_check.py          (카메라 번호 0)
         python webcam_check.py 1        (카메라 번호 1)
결과는 화면에 출력되고, 같은 폴더의 webcam_check_result.txt와 스냅샷 jpg로 저장된다.
"""
import os, sys, time, shutil, subprocess, platform

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "webcam_check_result.txt")
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def fourcc_str(v):
    v = int(v)
    s = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))
    return s if s.isprintable() and s.strip() else f"(코드 {v})"

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
is_win = platform.system() == "Windows"
log(f"OS: {platform.platform()}  /  Python {platform.python_version()}  /  카메라 번호 {idx}")

# 1) 윈도우에 등록된 카메라 이름
names = []
if is_win:
    try:
        ps = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in 'Camera','Image' } "
              "| ForEach-Object { $_.Name }")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20)
        names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
    except Exception as e:
        log(f"카메라 이름 조회 실패: {e}")
log("\n[1] 연결된 카메라: " + (", ".join(names) if names else "(이름을 못 읽음)"))

# 2) ffmpeg가 있으면 카메라가 직접 알려주는 지원 형식 목록
log("\n[2] 카메라가 알려주는 지원 형식 (ffmpeg)")
if is_win and shutil.which("ffmpeg") and names:
    for n in names:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-list_options", "true", "-f", "dshow", "-i", f"video={n}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        log(f"--- {n}")
        for l in r.stderr.splitlines():
            if "pixel_format" in l or "vcodec" in l:
                log(l.split("]")[-1].strip())
else:
    log("ffmpeg가 없어 건너뜀 (아래 OpenCV 시험만으로도 판단 가능)")

# 3) OpenCV로 형식·해상도별 실제 동작 시험
import cv2
log(f"\n[3] OpenCV {cv2.__version__} 실측 (형식을 요청한 뒤 실제로 받은 값과 fps)")
backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)] if is_win else [("ANY", cv2.CAP_ANY)]
backend = None
for bname, b in backends:
    for i in [idx] + [k for k in range(4) if k != idx]:
        c = cv2.VideoCapture(i, b)
        ok = c.isOpened() and c.read()[0]
        c.release()
        log(f"열기 시도: 번호 {i}, {bname} -> {'성공' if ok else '실패'}")
        if ok:
            backend, idx = b, i
            break
    if backend is not None:
        break
if backend is None:
    log("어떤 번호·방식으로도 카메라를 열지 못함")
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    sys.exit(1)
log(f"사용: 번호 {idx}, {bname}")
tests = []
for fcc in ("MJPG", "YUY2"):
    for w, h in ((2592, 1944), (2560, 1440), (1920, 1080), (1280, 720), (640, 480)):
        tests.append((fcc, w, h, 30))
tests.append(("MJPG", 1280, 720, 15))  # 프로젝트 기본 설정

best = None
log(f"{'요청':<22}{'실제 해상도':<14}{'실제 형식':<12}{'측정 fps':>8}")
for fcc, w, h, fps in tests:
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        log("카메라를 열 수 없음. 번호를 바꿔 보거나 다른 프로그램(카메라 앱, 줌 등)을 닫고 다시 실행")
        break
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):
        cap.read()
    n, t0, frame = 0, time.time(), None
    while time.time() - t0 < 3:
        ok, f = cap.read()
        if ok:
            n += 1
            frame = f
    dt = time.time() - t0
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    mfps = n / dt if dt else 0
    log(f"{fcc} {w}x{h} @{fps:<6}{f'{aw}x{ah}':<14}{real_fcc:<12}{mfps:>8.1f}")
    if frame is not None and fcc == "MJPG" and (best is None or aw * ah > best[0]):
        best = (aw * ah, frame, f"{aw}x{ah}")
    if frame is not None and (fcc, w, h, fps) == ("MJPG", 1280, 720, 15):
        cv2.imwrite(os.path.join(HERE, "webcam_snapshot_720p.jpg"), frame)
    cap.release()

if best:
    cv2.imwrite(os.path.join(HERE, f"webcam_snapshot_max_{best[2]}.jpg"), best[1])
    log(f"\n스냅샷 저장: webcam_snapshot_max_{best[2]}.jpg, webcam_snapshot_720p.jpg")

with open(OUT, "w", encoding="utf-8") as fp:
    fp.write("\n".join(lines))
log(f"\n결과 파일: {OUT}")
