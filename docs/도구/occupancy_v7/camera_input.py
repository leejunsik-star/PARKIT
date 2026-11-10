"""웹캠 번호와 Larix 수신 주소를 UI/설정에서 동일하게 처리한다."""
from urllib.parse import urlsplit

LARIX_INPUTS = [f"rtsp://127.0.0.1:8554/phone{i}" for i in range(1, 5)]


def parse_source(value):
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def validate_source(value, allow_file=False):
    source = parse_source(value)
    if isinstance(source, int):
        return source
    if not source:
        raise ValueError("웹캠 번호 또는 수신 영상 주소를 입력하세요.")
    if "://" not in source:
        if allow_file:
            return source
        raise ValueError("입력 예: 0 또는 rtsp://127.0.0.1:8554/phone1")
    url = urlsplit(source)
    if url.scheme not in ("rtsp", "rtsps", "http", "https", "rtmp", "rtmps") or not url.hostname:
        raise ValueError("RTSP/HTTP/RTMP 수신 주소를 사용하세요. Larix는 RTSP 수신 프로그램으로 연결합니다.")
    try:
        url.port
    except ValueError as exc:
        raise ValueError("영상 주소의 포트 번호를 확인하세요.") from exc
    return source
