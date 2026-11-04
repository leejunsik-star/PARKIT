"""PARKIT 보정 도구 ver2 — 기준점 자동 찾기 (docs/31 프롬프트, docs/32 사용법)

원본 calibrate.py를 고치지 않고 불러와 넓힌다. 손으로 기준점을 4개쯤 넓게 찍고 Enter를 누르면
기준선 목록(data/기준선_매트.json)의 선을 화면에서 찾아 그 교점을 자동 기준점으로 더한 뒤 다시 계산한다.
저장은 config\\v2\\cam_<이름>.json (+ 확인 사진). 원본 occupancy.py가 그대로 읽는다(키를 더했을 뿐).

    손 점   : 동그라미 + 십자 (v1과 같음). 같은 이름의 자동 점보다 늘 우선한다
    자동 점 : 마름모. 초록 = 두 선 모두 찾음, 노랑 = 애매한 선에서 나옴, 회색 작은 마름모 = 맞지 않아 안 씀, 회색 X = 뺌
    기준선  : 초록 찾음 / 노랑 애매함 / 회색 못 찾음. 가는 선 = 지금 계산으로 예상한 자리, 굵은 선 = 찾은 선
    빨간 겹원 손 점 : 자동으로 찾은 선과 크게 어긋난 손 점. 계산에서 뺀다(옆 칸에 찍었거나 이름이 틀렸을 때)

사용법 (docs\\도구\\occupancy 에서):
    python calibrate_v2.py                                 CAM-1, config/run_v2.json
    python calibrate_v2.py --cam CAM-1 --image 빈매트.jpg   사진으로
    python calibrate_v2.py --cam CAM-1 --source 0 --width 1920 --height 1080
    python occupancy.py --run config/run_v2.json           ver2 보정 결과로 판정 (원본 판정 프로그램 그대로)
렌즈 왜곡은 보정하지 않는다. 화면 가장자리와 가운데의 재투영 오차만 비교해 보여 준다.
"""

import argparse, datetime, os, sys, time

import cv2
import numpy as np

import calibrate as v1
from calibrate import PICK_PX, CalibApp, CalibModel, ranges
from parkit_autoref import RefLines, auto_find, edge_report, lsq_homography, settings, unknown_settings
from parkit_config import CONFIG_DIR, Params, cam_config_path, load_json, load_run, resolve, save_json
from parkit_draw import save_image, text
from parkit_geom import SlotTable, apply_h, aspect_same, estimate_hfov, to3d
from parkit_tk import MONO_FONT, check_tk, is_typing, key_of

V2_DIR = os.path.join(CONFIG_DIR, "v2")
DEFAULT_RUN_V2 = os.path.join(CONFIG_DIR, "run_v2.json")
DEFAULT_REF_LINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "기준선_매트.json")

LINE_COLORS = {"found": (60, 200, 60), "ambiguous": (0, 210, 255), "missing": (140, 140, 140)}
AUTO_COLORS = {"found": (60, 220, 60), "ambiguous": (0, 210, 255)}
DRAG_START_PX = 3             # 자동 점을 이만큼(화면 픽셀) 넘게 끌면 손 점으로 바뀐다

HELP_V2 = ("ver2 자동 찾기", [
    ("Enter", "자동 찾기 (손 점 4개 이상. 웹캠이면 화면을 멈추고 그 화면에서 찾는다)"),
    ("B", "기준선 보이기/숨기기 (초록 찾음 / 노랑 애매함 / 회색 못 찾음, 가는 선 = 예상 자리)"),
    ("U", "자동 점 쓰기 켜기/끄기 (끄면 손 점만으로 계산 = v1과 같은 계산)"),
    ("마름모(자동 점) 클릭", "고르기 → Delete / X: 빼기·되살리기.  오른쪽 클릭: 바로 빼기·되살리기"),
    ("마름모 끌기 / 고른 뒤 화살표", "같은 이름의 손 점으로 바뀌어 옮겨진다 (손 점이 자동 점보다 우선)"),
    ("빨간 겹원 손 점", "자동으로 찾은 선과 크게 어긋난 손 점. 계산에서 빠져 있다. 다시 찍거나 지운다"),
    ("[v1 기준점 가져오기]", "원본 보정 파일(config/cam_<이름>.json)의 손 점을 가져온다"),
])


def v2_config_path(name):
    return os.path.join(V2_DIR, f"cam_{name}.json")


def load_ref_lines(cfg, table):
    """실행 설정의 ref_lines를 읽는다. 이 매트의 기본 목록이 없으면 칸 표에서 새로 만든다(make_ref_lines.py와 같음)."""
    path = resolve(cfg["_path"], cfg.get("ref_lines")) if cfg.get("ref_lines") else DEFAULT_REF_LINES
    note = None
    if not os.path.exists(path):
        if os.path.normcase(os.path.abspath(path)) != os.path.normcase(DEFAULT_REF_LINES):
            raise SystemExit(f"기준선 목록이 없습니다: {path}")
        import make_ref_lines
        lines, points = make_ref_lines.build(table)
        save_json(path, {"_설명": {"무엇": "calibrate_v2.py가 칸 표에서 새로 만든 기준선 목록 (make_ref_lines.py와 같음)",
                                   "모양": make_ref_lines.MAT_SHAPE}, "lines": lines, "points": points})
        note = f"기준선 목록이 없어 칸 표에서 새로 만들었습니다: {os.path.basename(path)}"
    try:
        return RefLines.load(path), note
    except ValueError as e:
        raise SystemExit(f"{path}\n{e}")


# ---------------------------------------------------------------- 보정 상태와 계산 (화면 코드 없음)

class CalibModelV2(CalibModel):
    """v1 CalibModel + 자동 점. self.refs는 손 점만 담고, 계산할 때만 손 점 + 쓰는 자동 점을 합친다."""

    def __init__(self, name, source, cfg, table, params, size):
        super().__init__(name, source, cfg, table, params, size)
        self.path = v2_config_path(name)
        self.ref, note = load_ref_lines(cfg, table)
        if note:
            self.notes.append(note)
        self.opts = dict(cfg.get("autoref") or {})
        if unknown_settings(self.opts):
            self.notes.append(f"실행 설정 autoref에 모르는 이름이 있어 무시합니다: {unknown_settings(self.opts)}")
        self.P = settings(self.opts, self.ref.scan)
        # 일반 주차장: 칸 표에 없는 기준점 이름도 손으로 찍을 수 있게 목록에 더한다
        for p in self.ref.points:
            if p["id"] not in self.refs_all:
                self.refs_all[p["id"]] = {"id": p["id"], "slot": None, "kind": "REF", "label": f"기준점 {p['id']}",
                                          "mat": tuple(float(v) for v in p["mat"]), "measured": True}
        self.autos = []                # [{"id","mat","px","status","inlier","used","dev_px"}]
        self.excluded = set()          # 관리자가 뺀 자동 점 id
        self.fits = {}                 # 기준선 id → LineFit (마지막 자동 찾기)
        self.auto_H = None             # 자동 점만으로 구한 호모그래피 (손 점 검사용)
        self.manual_bad = {}           # 손 점 id → 어긋난 거리 (매트 단위)
        self.run_warnings = []
        self.last_run = None           # 마지막 자동 찾기 요약
        self.used = []                 # 계산에 쓴 점 (손 + 자동)
        self.edge = None               # 가장자리/가운데 재투영
        self.sel_auto = None
        self.use_auto = True
        self.show_reflines = True

    # ---------------- 불러오기 / 저장
    def load(self):
        super().load()
        if not os.path.exists(self.path):
            if os.path.exists(cam_config_path(self.name)):
                self.notes.append(f"원본 보정 파일({os.path.basename(cam_config_path(self.name))})이 있습니다. "
                                  f"[v1 기준점 가져오기]로 손 점을 가져올 수 있습니다.")
            return
        d = load_json(self.path)
        cal = tuple(d.get("calib_size", self.size))
        if not aspect_same(cal, self.size):
            return
        s = self.size[0] / cal[0]
        ar = d.get("auto_ref") or {}
        if "manual_points" in ar:
            self.refs = [{"id": r["id"], "mat": list(self.refs_all[r["id"]]["mat"]), "px": [r["px"][0] * s, r["px"][1] * s]}
                         for r in ar["manual_points"] if r["id"] in self.refs_all]
        else:
            auto_ids = {r["id"] for r in d.get("ref_points", []) if r.get("auto")}
            self.refs = [r for r in self.refs if r["id"] not in auto_ids]
        self.autos = [{"id": a["id"], "mat": list(a["mat"]), "px": [a["px"][0] * s, a["px"][1] * s],
                       "status": a.get("status", "found"), "inlier": bool(a.get("inlier", True)),
                       "used": bool(a.get("used", False)), "dev_px": a.get("dev_px", 0.0)} for a in ar.get("autos", [])]
        self.excluded = set(ar.get("excluded", []))
        self.use_auto = bool(ar.get("use_auto", True))
        self.auto_H = self._auto_h()
        self.notes.append(f"ver2: 손 점 {len(self.refs)}개, 자동 점 {sum(a['inlier'] for a in self.autos)}개, "
                          f"뺀 자동 점 {len(self.excluded)}개")

    def import_v1(self):
        """원본 보정 파일의 손 점(과 화각)을 가져온다. (가져온 수, 알림) 또는 (None, 이유)."""
        p = cam_config_path(self.name)
        if not os.path.exists(p):
            return None, f"원본 보정 파일이 없습니다: {p}"
        d = load_json(p)
        cal = tuple(d.get("calib_size", self.size))
        if not aspect_same(cal, self.size):
            return None, f"원본 보정 해상도 {cal[0]}x{cal[1]}와 지금 화면 {self.size[0]}x{self.size[1]}의 비율이 다릅니다."
        s = self.size[0] / cal[0]
        refs = [{"id": r["id"], "mat": list(self.refs_all[r["id"]]["mat"]), "px": [r["px"][0] * s, r["px"][1] * s]}
                for r in d.get("ref_points", []) if r["id"] in self.refs_all and not r.get("auto")]
        if not refs:
            return None, "원본 보정 파일에 가져올 기준점이 없습니다."
        self.refs = refs
        self.active_ref = None
        self.hfov = float(d.get("intrinsics", {}).get("hfov_deg", self.hfov))
        self.changed()
        return len(refs), f"원본 {os.path.basename(p)}에서 손 점 {len(refs)}개와 화각 {self.hfov:.1f}도를 가져왔습니다."

    def auto_ref_block(self):
        """설정 json에 더할 ver2 기록. 원본 occupancy.py는 이 키를 읽지 않는다."""
        try:
            ref_path = os.path.relpath(self.ref.path, V2_DIR).replace("\\", "/")
        except (TypeError, ValueError):           # 드라이브가 다르면 절대 경로
            ref_path = self.ref.path
        return {
            "version": 2,
            "ref_lines": ref_path,
            "settings": self.opts,
            "use_auto": self.use_auto,
            "manual_points": [{"id": r["id"], "mat": list(r["mat"]), "px": [round(float(v), 2) for v in r["px"]]}
                              for r in self.refs],
            "manual_bad": {k: round(v, 3) for k, v in self.manual_bad.items()},
            "excluded": sorted(self.excluded),
            "autos": [{"id": a["id"], "mat": list(a["mat"]), "px": [round(float(v), 2) for v in a["px"]],
                       "status": a["status"], "inlier": bool(a["inlier"]), "used": bool(a["used"]),
                       "dev_px": round(float(a.get("dev_px", 0.0)), 2)} for a in self.autos],
            "counts": {"manual": len(self.refs), "auto_used": sum(a["used"] for a in self.autos),
                       "manual_used": len(self.used) - sum(a["used"] for a in self.autos)},
            "last_run": self.last_run,
            "edge_report": self.edge,
        }

    def save(self, frame):
        if self.cam is None:
            return None
        self.recompute()
        cam = dict(self.cam)
        cam["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam["input_note"] = "source: 웹캠 번호, 영상 파일 경로, rtsp 주소"
        cam["guide_lines"] = [[round(v, 2) for v in l] for l in self.lines]
        cam["auto_ref"] = self.auto_ref_block()
        save_json(self.path, cam)
        check = os.path.join(V2_DIR, f"cam_{self.name}_check.jpg")
        im = frame.copy()
        self.draw(im, 1.0, all_slots=True)
        e = cam["homography_err_cm"]
        c = cam["auto_ref"]["counts"]
        text(im, f"{self.name} v2 {cam['date']}  refs manual {c['manual_used']} + auto {c['auto_used']}  "
                 f"err mean {e['mean']:.2f} max {e['max']:.2f} cm  monitored {len(cam['monitored'])}",
             (10, 30), 0.8, (255, 255, 255), 2)
        save_image(check, im)
        self.saved = True
        return check

    # ---------------- 자동 찾기와 계산
    def _auto_h(self):
        pts = [a for a in self.autos if a["inlier"] and a["id"] not in self.excluded]
        if len(pts) < 4:
            return None
        return lsq_homography(np.array([a["px"] for a in pts], float), np.array([a["mat"] for a in pts], float))

    def run_auto(self, frame):
        res = auto_find(frame, self.ref, self.refs, self.size, self.opts, self.excluded)
        self.fits = res.fits
        self.autos = [dict(a, inlier=bool(a.get("inlier", False))) for a in res.autos]
        self.auto_H = self._auto_h() if res.ok else None
        bad_msgs = tuple(f"손 점 {mid}가" for mid in res.manual_bad)
        self.run_warnings = [w for w in res.warnings if not w.startswith(bad_msgs)]     # 손 점 경고는 recompute가 새로 낸다
        self.last_run = {"ok": res.ok, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "lines": res.counts(), "rounds": res.rounds}
        self.sel_auto = None
        self.changed()
        return res

    def toggle_exclude(self, aid):
        if aid in self.excluded:
            self.excluded.discard(aid)
        else:
            self.excluded.add(aid)
        self.auto_H = self._auto_h()
        self.changed()
        return aid in self.excluded

    def auto_of(self, aid):
        for a in self.autos:
            if a["id"] == aid:
                return a
        return None

    def near_auto(self, p, scale):
        best = None
        for a in self.autos:
            d = np.linalg.norm(np.array(a["px"]) - p) * scale
            if d <= PICK_PX and (best is None or d < best[1]):
                best = (a["id"], d)
        return None if best is None else best[0]

    def auto_to_manual(self, aid):
        """자동 점을 같은 이름의 손 점으로 바꾼다(그 자리 그대로). 손 점 번호를 돌려준다. 다시 계산은 부른 쪽이."""
        a = self.auto_of(aid)
        if a is None or aid not in self.refs_all:
            return None
        i = self.ref_index(aid)
        if i is None:
            self.refs.append({"id": aid, "mat": list(self.refs_all[aid]["mat"]), "px": [float(v) for v in a["px"]]})
            i = len(self.refs) - 1
        self.sel_ref = self.active_ref = aid
        self.sel_slot = self.refs_all[aid]["slot"]
        self.sel_auto = None
        return i

    def recompute(self):
        manual = self.refs
        man_ids = {r["id"] for r in manual}
        # 손 점 검사: 자동 점만으로 구한 호모그래피와 크게 어긋나는 손 점
        self.manual_bad = {}
        if self.auto_H is not None:
            for r in manual:
                e = float(np.linalg.norm(apply_h(self.auto_H, [r["px"]])[0] - np.asarray(r["mat"], float)))
                if e > self.P["manual_warn"]:
                    self.manual_bad[r["id"]] = e
        for a in self.autos:
            a["used"] = bool(self.use_auto and a["inlier"] and a["id"] not in self.excluded and a["id"] not in man_ids)
        use_m = [r for r in manual if not (self.use_auto and r["id"] in self.manual_bad)]
        self.used = use_m + [{"id": a["id"], "mat": list(a["mat"]), "px": list(a["px"]), "auto": True}
                             for a in self.autos if a["used"]]
        self.refs = self.used
        try:
            super().recompute()
        finally:
            self.refs = manual
        self.edge = None
        if self.cam and self.cam.get("pose"):
            p, it = self.cam["pose"], self.cam["intrinsics"]
            K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1.0]])
            q, _ = cv2.projectPoints(to3d([r["mat"] for r in self.used]), np.array(p["rvec"], float),
                                     np.array(p["tvec"], float), K, None)
            perr = np.linalg.norm(q.reshape(-1, 2) - np.array([r["px"] for r in self.used], float), axis=1)
            self.edge = {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in edge_report(self.used, perr, self.size, self.P["border_frac"]).items()}

    def fit_fov(self):
        refs = self.used
        if len(refs) < 6:
            return None
        f, e = estimate_hfov([r["px"] for r in refs], [r["mat"] for r in refs], *self.size)
        old = self.hfov
        self.hfov = round(f, 2)
        self.changed()
        return old, self.hfov, e

    def reset_all(self):
        self.autos, self.excluded, self.fits = [], set(), {}
        self.auto_H, self.sel_auto, self.last_run, self.run_warnings = None, None, None, []
        super().reset_all()

    def rescale(self, new_size):
        """해상도를 바꿨을 때: 자동 점도 손 점처럼 비율만큼 옮긴다(비율이 다르면 지운다). 찾은 선은 다시 찾는다."""
        new_size = tuple(new_size)
        if new_size != self.size:
            self.fits = {}
            if aspect_same(self.size, new_size):
                s = new_size[0] / self.size[0]
                for a in self.autos:
                    a["px"] = [a["px"][0] * s, a["px"][1] * s]
            else:
                self.autos, self.excluded, self.sel_auto = [], set(), None
            self.auto_H = self._auto_h()
        return super().rescale(new_size)

    # ---------------- 그리기
    def draw(self, im, s, all_slots=False):
        super().draw(im, s, all_slots)
        if self.show_reflines or all_slots:
            self.draw_reflines(im, s)
        self.draw_autos(im, s, all_slots)
        for r in self.refs:
            if self.use_auto and r["id"] in self.manual_bad:
                q = (np.array(r["px"]) * s).astype(int)
                cv2.circle(im, tuple(q), 10, (0, 0, 255), 2)
                cv2.circle(im, tuple(q), 15, (0, 0, 255), 2)
                text(im, f"{r['id']} ! {self.manual_bad[r['id']]:.2f}", (q[0] + 12, q[1] + 18), 0.45 if s < 1 else 0.6,
                     (0, 0, 255), 1)

    def fit_segment(self, f):
        """찾은 선을 안쪽 점이 덮는 길이만큼의 선분으로."""
        if f.c is None or f.u is None or not len(f.pts):
            return None
        pts = f.pts[f.inl] if f.inl is not None and f.inl.any() else f.pts
        t = (pts - f.c) @ f.u
        return f.c + f.u * t.min(), f.c + f.u * t.max()

    def draw_reflines(self, im, s):
        for f in self.fits.values():
            c = LINE_COLORS[f.status]
            if f.expected is not None and np.isfinite(f.expected).all():
                a, b = (np.asarray(f.expected, float) * s).astype(int)
                cv2.line(im, tuple(a), tuple(b), c, 1, cv2.LINE_AA)
            if f.status != "missing":
                seg = self.fit_segment(f)
                if seg is not None:
                    cv2.line(im, tuple((seg[0] * s).astype(int)), tuple((seg[1] * s).astype(int)), c, 2, cv2.LINE_AA)

    def draw_autos(self, im, s, all_slots=False):
        man_ids = {r["id"] for r in self.refs}
        for a in self.autos:
            q = tuple((np.array(a["px"]) * s).astype(int))
            if a["id"] in self.excluded:
                cv2.drawMarker(im, q, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 14, 2)
            elif a["used"]:
                cv2.drawMarker(im, q, AUTO_COLORS.get(a["status"], (60, 220, 60)), cv2.MARKER_DIAMOND, 14, 2)
            elif a["id"] in man_ids:
                cv2.drawMarker(im, q, (200, 200, 200), cv2.MARKER_DIAMOND, 9, 1)
            else:
                cv2.drawMarker(im, q, (130, 130, 130), cv2.MARKER_DIAMOND, 9, 1)
            if a["id"] == self.sel_auto and not all_slots:
                cv2.rectangle(im, (q[0] - 12, q[1] - 12), (q[0] + 12, q[1] + 12), (255, 0, 255), 2)
                text(im, f"{a['id']} auto", (q[0] + 14, q[1] - 10), 0.45 if s < 1 else 0.6, (255, 0, 255), 1)

    def draw_zoom(self, big, x0, y0, z):
        super().draw_zoom(big, x0, y0, z)

        def T(p):
            return (int(round((p[0] - x0) * z)), int(round((p[1] - y0) * z)))
        if self.show_reflines:
            for f in self.fits.values():
                if f.status == "missing":
                    continue
                seg = self.fit_segment(f)
                if seg is not None:
                    cv2.line(big, T(seg[0]), T(seg[1]), LINE_COLORS[f.status], 1, cv2.LINE_AA)
        for a in self.autos:
            if a["used"] or a["id"] == self.sel_auto:
                c = (255, 0, 255) if a["id"] == self.sel_auto else AUTO_COLORS.get(a["status"], (60, 220, 60))
                cv2.drawMarker(big, T(a["px"]), c, cv2.MARKER_DIAMOND, 22, 1)

    # ---------------- 안내 글
    def guide(self):
        if self.mode != 1:
            return super().guide()
        n = len(self.refs)
        na = sum(a["used"] for a in self.autos)
        if n < 4:
            g = (f"① 손 점 {n}/4: 화면 네 귀퉁이 쪽 칸의 IE/IB 점을 넓게 찍으세요(앞줄·뒷줄, 입구·안쪽 섞기). "
                 f"4개가 되면 Enter = 자동 찾기.")
        elif not self.autos:
            g = f"① 손 점 {n}개: Enter를 누르면 기준선을 찾아 기준점을 자동으로 더합니다."
        else:
            g = (f"① 손 점 {len(self.used) - na} + 자동 점 {na}: 노랑(애매)·빨간 겹원 점을 확인하세요. "
                 f"틀린 자동 점은 Delete로 빼거나 끌어서 고칩니다. 괜찮으면 ② 감시 칸(2).")
        return g + ("" if self.saved else "   · 저장 안 됨")

    def warnings(self):
        w = []
        if self.use_auto:
            for mid, e in sorted(self.manual_bad.items()):
                w.append(f"손 점 {mid}가 자동으로 찾은 선과 {e:.2f}만큼 어긋납니다(옆 칸에 찍었거나 이름이 틀림). "
                         f"계산에서 뺐습니다. 다시 찍거나 지우세요(빨간 겹원).")
        w += self.run_warnings
        if self.edge and self.edge["edge_mean"] is not None and self.edge["mid_mean"] is not None:
            if self.edge["edge_mean"] > max(2.0, 2.0 * self.edge["mid_mean"]):
                w.append(f"화면 가장자리 점의 재투영 오차({self.edge['edge_mean']:.2f}px)가 가운데({self.edge['mid_mean']:.2f}px)보다 "
                         f"큽니다. 렌즈 왜곡일 수 있습니다(ver2는 왜곡을 보정하지 않음). 가장자리 칸은 꼭짓점을 확인하세요.")
        manual = self.refs
        self.refs = self.used if self.used else manual
        try:
            w += super().warnings()
        finally:
            self.refs = manual
        return w


# ---------------------------------------------------------------- 화면 (tkinter)

class CalibAppV2(CalibApp):
    def __init__(self, args, cfg, table, params):
        self.pending_auto = None
        super().__init__(args, cfg, table, params)
        self.root.title(f"PARKIT 보정 도구 ver2 (기준점 자동 찾기) — {args.cam}")

    def build(self):
        super().build()
        tk, ttk = self.tk, self.ttk
        pn = self.scroll.inner
        first = pn.winfo_children()[0]          # "입력" 틀 바로 뒤에 넣는다
        f = ttk.LabelFrame(pn, text="자동 찾기 (ver2)", padding=6)
        f.pack(fill="x", pady=(4, 0), after=first)
        ref = self.m.ref
        ttk.Label(f, text=f"기준선 목록: {os.path.basename(ref.path or '?')}  (선 {len(ref.lines)}개, 기준점 {len(ref.points)}개)",
                  foreground="#555").pack(anchor="w")
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(4, 0))
        ttk.Button(rr, text="자동 찾기 (Enter)", style="Accent.TButton", command=self.run_auto).pack(side="left")
        ttk.Button(rr, text="v1 기준점 가져오기", command=self.import_v1).pack(side="left", padx=4)
        rr = ttk.Frame(f)
        rr.pack(fill="x", pady=(2, 0))
        self.show_ref_var = tk.BooleanVar(value=True)
        self.use_auto_var = tk.BooleanVar(value=self.m.use_auto)
        ttk.Checkbutton(rr, text="기준선 보이기 (B)", variable=self.show_ref_var, command=self.sync_v2).pack(side="left")
        ttk.Checkbutton(rr, text="자동 점 쓰기 (U)", variable=self.use_auto_var, command=self.sync_v2).pack(side="left", padx=8)
        self.ar_sum = ttk.Label(f, text="", font=(MONO_FONT, 9), justify="left")
        self.ar_sum.pack(anchor="w", pady=(4, 0))
        self.ar_sel_var = tk.StringVar()
        ttk.Label(f, textvariable=self.ar_sel_var, foreground="#b000b0", wraplength=380).pack(anchor="w")
        ttk.Label(f, text="◇ 자동 점: 초록 찾음 · 노랑 애매 · 회색 안 씀 · X 뺌.  ○ 손 점이 늘 우선. "
                          "빨간 겹원 = 자동 결과와 어긋난 손 점(계산에서 뺌).", foreground="#555", wraplength=380,
                  justify="left").pack(anchor="w", pady=(2, 0))

    # ---------------- 동작
    def sync_v2(self):
        self.m.show_reflines = self.show_ref_var.get()
        if self.m.use_auto != self.use_auto_var.get():
            self.m.use_auto = self.use_auto_var.get()
            self.m.changed()
            self.msg = "자동 점을 씁니다." if self.m.use_auto else "자동 점을 쓰지 않습니다(손 점만 = v1과 같은 계산)."
        self.view.canvas.focus_set()

    def run_auto(self):
        m = self.m
        if len(m.refs) < 4:
            self.msg = f"손 점이 4개 이상 있어야 자동 찾기를 합니다(지금 {len(m.refs)}개)."
            return
        if self.src.kind != "image" and self.src.frozen is None:
            self.freeze_var.set(True)
            self.toggle_freeze()                # 찾은 화면과 보이는 화면이 같도록 멈춘다
        f = self.src.get()
        f = self.frame if f is None else f
        if (f.shape[1], f.shape[0]) != m.size:
            self.msg = "화면 크기가 보정 해상도와 달라 자동 찾기를 못 합니다."
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        t0 = time.time()
        try:
            res = m.run_auto(f)
        finally:
            self.root.configure(cursor="")
        c = res.counts()
        na = sum(a["used"] for a in m.autos)
        if res.ok:
            e = m.cam["homography_err_cm"] if m.cam else None
            self.msg = (f"자동 찾기 {time.time() - t0:.1f}초: 선 찾음 {c['found']} / 애매 {c['ambiguous']} / 못 찾음 {c['missing']}, "
                        f"자동 점 {na}개를 더했습니다." + (f" 오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f}cm" if e else ""))
        else:
            self.msg = "자동 찾기 실패: " + (res.warnings[-1] if res.warnings else "이유 모름")
        print(f"[보정 v2] {self.msg}")
        for w in res.warnings:
            print(f"          {w}")

    def import_v1(self):
        m = self.m
        if m.refs and not self.mb.askyesno("v1 기준점 가져오기", f"지금 손 점 {len(m.refs)}개를 원본 보정 파일의 점으로 바꿉니다. 계속할까요?"):
            return
        n, msg = m.import_v1()
        self.msg = msg
        if n is None:
            self.mb.showinfo("v1 기준점 가져오기", msg)
        print(f"[보정 v2] {msg}")
        self.view.canvas.focus_set()

    def toggle_exclude(self, aid):
        m = self.m
        out = m.toggle_exclude(aid)
        a = m.auto_of(aid)
        if out:
            self.msg = f"자동 점 {aid}를 뺐습니다. 다시 Delete를 누르면 되살립니다."
        elif a is not None and not a["inlier"]:
            self.msg = f"자동 점 {aid}를 되살렸습니다. 지난 찾기에서 맞지 않아 빠졌던 점이라 자동 찾기(Enter)를 다시 해야 쓰입니다."
        else:
            self.msg = f"자동 점 {aid}를 되살렸습니다."

    def save(self):
        path = self.m.save(self.frame)
        if path is None:
            self.mb.showinfo("저장", "기준점이 4개 이상 있어야 저장할 수 있습니다.")
            return
        cam = self.m.cam
        e = cam["homography_err_cm"]
        c = self.m.auto_ref_block()["counts"]
        self.msg = f"저장했습니다: v2/{os.path.basename(self.m.path)}, {os.path.basename(path)}"
        print(f"[보정 v2] 저장함: {self.m.path}")
        print(f"          확인 사진: {path}")
        print(f"          기준점 손 {c['manual_used']} + 자동 {c['auto_used']}개, 오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f}cm, "
              f"감시 칸 {ranges(cam['monitored'])}")
        if cam.get("pose"):
            p = cam["pose"]
            print(f"          렌즈 위치 {p['cam_pos_cm']} (x, y, 높이 cm), 아래로 {p['pitch_deg']}도, "
                  f"옆 기울기 {p['roll_deg']}도, 재투영 {p['reproj_px']['mean']}px")
        if self.m.edge:
            print(f"          재투영 가장자리 {self.m.edge['edge_mean']}px ({self.m.edge['edge_n']}점) / "
                  f"가운데 {self.m.edge['mid_mean']}px ({self.m.edge['mid_n']}점)")

    # ---------------- 마우스
    def on_press(self, e):
        m = self.m
        self.pending_auto = None
        if m.mode == 1:
            p = self.src_xy(e)
            if p is not None and m.near_ref(p, self.view.scale) is None:
                aid = m.near_auto(p, self.view.scale)
                if aid is not None:
                    self.view.canvas.focus_set()
                    m.sel_auto = aid
                    self.pending_auto = (aid, e.x, e.y)
                    a = m.auto_of(aid)
                    state = "뺌" if aid in m.excluded else ("씀" if a["used"] else "안 씀")
                    self.msg = f"자동 점 {aid} ({state}) 고름: Delete로 빼기/되살리기, 끌거나 화살표로 옮기면 손 점이 됩니다."
                    return
        m.sel_auto = None
        super().on_press(e)

    def on_drag(self, e):
        pa = self.pending_auto
        if pa is not None:
            if abs(e.x - pa[1]) + abs(e.y - pa[2]) <= DRAG_START_PX:
                return
            self.pending_auto = None
            i = self.m.auto_to_manual(pa[0])
            if i is None:
                return
            self.m.drag = ("ref", i)
            self.msg = f"자동 점 {pa[0]}를 손 점으로 바꿔 옮깁니다(손 점이 우선)."
        super().on_drag(e)

    def on_release(self, e):
        if self.pending_auto is not None:       # 끌지 않은 클릭: 고르기만
            self.pending_auto = None
            return
        super().on_release(e)

    def on_right(self, e):
        m = self.m
        if m.mode == 1:
            p = self.src_xy(e)
            if p is not None and m.near_ref(p, self.view.scale) is None:
                aid = m.near_auto(p, self.view.scale)
                if aid is not None:
                    m.sel_auto = aid
                    self.toggle_exclude(aid)
                    return
        super().on_right(e)

    # ---------------- 키보드
    def on_key(self, e):
        if is_typing(e.widget):
            return
        k = key_of(e)
        m = self.m
        if k == "Return":
            self.run_auto()
            return "break"
        if k == "b":
            self.show_ref_var.set(not self.show_ref_var.get())
            self.sync_v2()
            return "break"
        if k == "u":
            self.use_auto_var.set(not self.use_auto_var.get())
            self.sync_v2()
            return "break"
        if m.mode == 1 and m.sel_auto is not None:
            if k in ("x", "Delete", "BackSpace"):
                self.toggle_exclude(m.sel_auto)
                return "break"
            if k in ("Left", "Right", "Up", "Down", "j", "l", "i", "k"):
                aid = m.sel_auto
                if m.auto_to_manual(aid) is not None:
                    self.msg = f"자동 점 {aid}를 손 점으로 바꿨습니다. 화살표로 옮깁니다."
            elif k == "Tab":
                m.sel_auto = None
        return super().on_key(e)

    # ---------------- 패널
    def update_panel(self):
        super().update_panel()
        m = self.m
        if self.show_ref_var.get() != m.show_reflines:
            self.show_ref_var.set(m.show_reflines)
        # 계산 결과: 손 점 + 자동 점
        if m.cam:
            e = m.cam["homography_err_cm"]
            na = sum(a["used"] for a in m.autos)
            lines = [f"기준점 {len(m.used)}개 (손 {len(m.used) - na} + 자동 {na})",
                     f"오차 평균 {e['mean']:.3f} / 최대 {e['max']:.3f} cm (목표 < {v1.GOOD_ERR})"]
            p = m.cam.get("pose")
            if p:
                c = p["cam_pos_cm"]
                lines.append(f"렌즈 위치 x {c[0]:.1f}  y {c[1]:.1f}  높이 {c[2]:.1f} cm")
                lines.append(f"아래로 {p['pitch_deg']:.1f}도  옆 기울기 {p['roll_deg']:.1f}도  재투영 {p['reproj_px']['mean']:.2f}px")
            if m.edge and m.edge["edge_mean"] is not None:
                mid = "-" if m.edge["mid_mean"] is None else f"{m.edge['mid_mean']:.2f}"
                lines.append(f"재투영 가장자리 {m.edge['edge_mean']:.2f}px ({m.edge['edge_n']}점) / 가운데 {mid}px ({m.edge['mid_n']}점)")
            self.res_lbl.configure(text="\n".join(lines))
        # 자동 찾기 요약
        if m.fits:
            c = {"found": 0, "ambiguous": 0, "missing": 0}
            for f in m.fits.values():
                c[f.status] += 1
            seen = c["found"] + c["ambiguous"]
            s1 = f"기준선: 찾음 {c['found']} · 애매 {c['ambiguous']} · 못 찾음 {c['missing']} (보이는 곳 {seen}개)"
        elif m.autos:
            s1 = "기준선: 저장된 결과를 불러옴 (선을 보려면 Enter)"
        else:
            s1 = "기준선: 아직 찾지 않음 (손 점 4개 → Enter)"
        used = [a for a in m.autos if a["used"]]
        s2 = (f"자동 점: 씀 {len(used)} (찾음 {sum(a['status'] == 'found' for a in used)} · "
              f"애매 {sum(a['status'] == 'ambiguous' for a in used)}) · 안 씀 "
              f"{sum(not a['used'] and a['id'] not in m.excluded for a in m.autos)} · 뺌 {len(m.excluded)}")
        rounds = (m.last_run or {}).get("rounds") or []
        s3 = "찾는 폭: " + (" → ".join(f"{r['search']}({r['used']}점)" for r in rounds) if rounds else "-")
        s4 = "" if m.use_auto else "\n자동 점 쓰기 꺼짐 (U): 손 점만으로 계산"
        txt = f"{s1}\n{s2}\n{s3}{s4}"
        if self.ar_sum.cget("text") != txt:
            self.ar_sum.configure(text=txt)
        if m.sel_auto is not None and m.auto_of(m.sel_auto) is not None:
            a = m.auto_of(m.sel_auto)
            state = "뺌" if a["id"] in m.excluded else ("씀" if a["used"] else "안 씀")
            err = m.err.get(a["id"]) if m.err else None
            self.ar_sel_var.set(f"고른 자동 점: {a['id']} ({state}, {'찾음' if a['status'] == 'found' else '애매'}, "
                                f"예상 자리와 {a.get('dev_px', 0):.1f}px" + (f", 오차 {err:.3f}cm" if err is not None else "")
                                + ")  Delete: 빼기/되살리기")
        else:
            self.ar_sel_var.set("")


def main():
    ap = argparse.ArgumentParser(description="PARKIT 보정 도구 ver2 (기준점 자동 찾기)")
    ap.add_argument("--cam", default=None, help="카메라 이름 (기본 CAM-1). config/v2/cam_<이름>.json 에 저장")
    ap.add_argument("--source", default=None, help="웹캠 번호, 영상 파일, rtsp 주소 (없으면 저장된 값 또는 0)")
    ap.add_argument("--image", default=None, help="웹캠 대신 사진으로 보정")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--run", default=DEFAULT_RUN_V2, help="실행 설정 (칸 표, 기준선 목록 ref_lines, 자동 찾기 설정 autoref)")
    args = ap.parse_args()
    if not args.cam:
        args.cam = "CAM-1"
        print("[보정 v2] 카메라 이름을 주지 않아 CAM-1로 합니다. 다른 카메라는 --cam CAM-2 처럼 줍니다.")
    check_tk()
    cfg = load_run(args.run)
    table = SlotTable(cfg["slot_table"])
    params = Params(cfg, table)
    if args.source is None and not args.image:
        for p in (v2_config_path(args.cam), cam_config_path(args.cam)):
            if os.path.exists(p):
                args.source = load_json(p).get("source", 0)
                break
        else:
            args.source = 0
        print(f"[보정 v2] 입력: {args.source}")
    # 원본 CalibApp이 만드는 모델과 도움말을 ver2 것으로 바꿔 끼운다 (원본 파일은 그대로)
    v1.CalibModel = CalibModelV2
    v1.HELP = v1.HELP[:1] + [HELP_V2] + v1.HELP[1:]
    app = CalibAppV2(args, cfg, table, params)
    print(f"[보정 v2] {app.src.msg}. 단축키는 창에서 ? 키로 봅니다.")
    app.root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
