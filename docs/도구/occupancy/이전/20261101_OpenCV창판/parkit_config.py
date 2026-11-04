"""PARKIT 점유 판정 — 설정 파일 읽기/쓰기 (25 문서 5절, P8)

실행 설정(run.json): 쓸 카메라 목록, 모델, 신뢰도, 판정 값. 사람이 고친다.
카메라 설정(cam_*.json): 보정 도구가 만든다.
설정 파일 안의 상대 경로는 그 설정 파일이 있는 폴더 기준이다.
"""

import copy, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "config")
DEFAULT_RUN = os.path.join(CONFIG_DIR, "run.json")
LOG_DIR = os.path.join(HERE, "logs")

# 판정 값 첫 값 (22·26 문서). run.json의 params, overrides로 바꾼다.
DEFAULT_PARAMS = {
    "zone_margin_entrance_cm": 1.0,   # P2 판정 구역: cell을 입구 쪽으로 늘리는 길이
    "zone_margin_back_cm": 1.0,       # P2 판정 구역: cell을 안쪽(카스토퍼 쪽)으로 늘리는 길이
    "expect_max_dist_cm": 2.0,        # P2b 예상 접점과 줄 방향(x) 거리 한계. 깊이는 판정 구역으로 본다
    "car_width_cm": 3.2,              # P2b 예상 접점, P3 차 범위
    "car_length_cm": 6.5,             # P2b 예상 접점, P2a 주차된 차의 뒤 끝
    "car_height_cm": 2.0,             # P2b 예상 접점
    "straddle_min_cm": 1.0,           # P3 옆 칸 inner에 들어간 길이 기준
    "cut_overlap_ratio": 0.5,         # P2a 가로 겹침이 상자 폭의 이 비율 이상이면 아래가 잘린 상자
    "side_overlap_ratio": 0.1,        # P2a 가로 겹침이 이 비율 미만이면 옆이 가려진 것으로도 보지 않음 (스치기만 한 상자)
    "cut_touch_ratio": 0.01,         # P2a 아래 변이 가리는 상자 위 끝보다 이만큼(화면 높이 비율) 위여도 안에 든 것으로 봄
    "occluder_depth_cm": 3.2,         # P2a 통로 차의 깊이 방향 길이 (접점 → 뒤 끝)
    "occlusion_ratio": 0.30,          # P5 칸 화면 영역을 덮는 비율 기준
    "occlusion_height_cm": 2.0,       # P5 칸을 올리는 높이
    "t_occupied_s": 3.0,              # P6 차 있음이 이만큼 이어지면 OCCUPIED
    "t_empty_s": 5.0,                 # P6 차 없음이 이만큼 이어지면 EMPTY
    "t_blind_s": 1.0,                 # P5 가려짐이 이만큼 넘게 이어지면 DYNAMIC_BLIND
    "t_stale_s": 2.0,                 # P9 프레임이 이만큼 안 온 카메라의 마지막 판정은 합칠 때 쓰지 않음
    "t_offline_s": 30.0,              # P10 프레임이 이만큼 안 오면 카메라 끊김
}

DEFAULT_RUN_CONFIG = {
    "slot_table": "../../../칸표_매트좌표_초안.json",
    "model": "../../models/M8.pt",
    "conf": 0.25,
    "imgsz": 640,
    "analyze_fps": 2.0,
    "calib": {"hfov_deg": 50.0, "auto_margin_px": 0},
    "cameras": [],
    "params": {},
    "overrides": {"rows": {}, "slots": {}},
    "presets": {"demo": {"t_occupied_s": 1.0, "t_empty_s": 2.0}},
    "display": {"cam_height": 540, "plan_px_per_cm": 10, "startup_overlay_s": 5.0},
    "log": {"frames": False, "record": False, "dir": "../logs"},
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def resolve(base_file, p):
    """설정 파일 기준 상대 경로를 절대 경로로. 웹캠 번호, rtsp 주소는 그대로 둔다."""
    if p is None or isinstance(p, int):
        return p
    s = str(p)
    if s.isdigit() or "://" in s:
        return s
    if os.path.isabs(s):
        return s
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), s))


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_run(path=DEFAULT_RUN, preset=None):
    """실행 설정을 읽어 기본값과 합친다. 경로는 절대 경로로 바꿔 둔다."""
    raw = load_json(path) if os.path.exists(path) else {}
    cfg = merge(DEFAULT_RUN_CONFIG, raw)
    cfg["_path"] = os.path.abspath(path)
    cfg["slot_table"] = resolve(path, cfg["slot_table"])
    cfg["model"] = resolve(path, cfg["model"])
    cfg["log"]["dir"] = resolve(path, cfg["log"]["dir"])
    for c in cfg["cameras"]:
        c["config"] = resolve(path, c["config"])
        if c.get("source") is not None:
            c["source"] = resolve(path, c["source"])
    if preset:
        if preset not in cfg["presets"]:
            raise SystemExit(f"실행 설정에 '{preset}' 프리셋이 없습니다. 있는 것: {list(cfg['presets'])}")
        cfg["params"] = merge(cfg["params"], cfg["presets"][preset])
    return cfg


class Params:
    """칸별 판정 값(P8). 우선순위: 칸별 > 줄별 > params > 기본값.
    p = Params(cfg, table); p(4)["car_width_cm"], p(None)은 공통 값."""

    def __init__(self, cfg=None, table=None):
        cfg = cfg or {}
        self.base = merge(DEFAULT_PARAMS, cfg.get("params", {}))
        ov = cfg.get("overrides", {}) or {}
        self.rows = ov.get("rows", {}) or {}
        self.slots = {str(k): v for k, v in (ov.get("slots", {}) or {}).items()}
        self.table = table
        self.cache = {}
        unknown = set(self.base) - set(DEFAULT_PARAMS)
        for d in list(self.rows.values()) + list(self.slots.values()):
            unknown |= set(d) - set(DEFAULT_PARAMS)
        if unknown:
            print(f"[설정] 모르는 판정 값 이름이 있습니다(무시): {sorted(unknown)}")

    def __call__(self, no):
        if no in self.cache:
            return self.cache[no]
        d = dict(self.base)
        if no is not None:
            if self.table is not None and int(no) in self.table.slots:
                d.update(self.rows.get(self.table[no].row, {}))
            d.update(self.slots.get(str(no), {}))
        self.cache[no] = d
        return d


def cam_config_path(name):
    return os.path.join(CONFIG_DIR, f"cam_{name}.json")
