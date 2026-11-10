"""Larix RTSP 영상을 받는 MediaMTX 실행기 (Mac/Windows/Linux)."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import platform
import socket
import subprocess
import tarfile
import urllib.request
import zipfile

VERSION = "1.21.3"
ASSETS = {
    ("Darwin", "arm64"): ("darwin_arm64.tar.gz", "bc1a0762fd18ffb413bcd81dcb9a072ac159fbd1f27e79ac6f14598aaac5ed3f"),
    ("Darwin", "x86_64"): ("darwin_amd64.tar.gz", "8a8416ee2ff5d17d9366498b6d88b76ab6d3afcda50b6d7321ccddb349bdbaf2"),
    ("Windows", "amd64"): ("windows_amd64.zip", "b2f5246b07cb72961df03b8d5e8cad43dc657bcf3f49a87bb031bdf22721af74"),
    ("Linux", "x86_64"): ("linux_amd64.tar.gz", "0a2ea7eba96023631efa1bea7804823a9145b043b6646a8a05798d4ac24ed635"),
    ("Linux", "aarch64"): ("linux_arm64.tar.gz", "27f038237cb5f77a724f6afc58eca2798f2bc4edf9989895631a80eeb41a4fe2"),
}
BASE = Path(__file__).resolve().parent


def binary_path():
    key = (platform.system(), platform.machine().lower())
    if key not in ASSETS:
        raise RuntimeError(f"자동 설치 미지원 환경: {key}. MediaMTX를 별도로 설치하세요.")
    asset, digest = ASSETS[key]
    filename = "mediamtx.exe" if key[0] == "Windows" else "mediamtx"
    return BASE / ".runtime" / f"mediamtx-{VERSION}-{key[0]}-{key[1]}" / filename, asset, digest


def install():
    dest, asset, digest = binary_path()
    url = f"https://github.com/bluenviron/mediamtx/releases/download/v{VERSION}/mediamtx_v{VERSION}_{asset}"
    print(f"MediaMTX {VERSION} 내려받기…", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("다운로드 검증 실패: 파일을 설치하지 않았습니다.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 지정된 실행 파일과 라이선스만 읽어 압축 경로가 폴더 밖에 쓰이지 않게 한다.
    archive = zipfile.ZipFile(io.BytesIO(data)) if asset.endswith(".zip") else tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    with archive:
        for name in (dest.name, "LICENSE"):
            if asset.endswith(".zip"):
                payload = archive.read(name)
            else:
                with archive.extractfile(name) as handle:
                    payload = handle.read()
            (dest.parent / name).write_bytes(payload)
    dest.chmod(0o755)
    print(f"설치 완료: {dest}")
    return dest


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.168.0.1", 9))  # 패킷을 보내지 않고 인터페이스 주소 조회
            return sock.getsockname()[0]
    except OSError:
        return "컴퓨터의-LAN-IP"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="수신기 설치 후 실행")
    parser.add_argument("--install-only", action="store_true", help="수신기 설치만")
    parser.add_argument("--host-ip", help="휴대폰에 입력할 PC의 LAN IP (안내 표시용)")
    args = parser.parse_args()
    dest, _, _ = binary_path()
    if args.install or args.install_only:
        dest = install()
    if args.install_only:
        return
    if not dest.exists():
        parser.error("수신기가 없습니다. 이 명령에 --install 을 붙여 처음 한 번 설치하세요.")
    host = args.host_ip or lan_ip()
    print(f"휴대폰 1 Larix 송출 주소: rtsp://{host}:8554/phone1\n"
          f"휴대폰 2 Larix 송출 주소: rtsp://{host}:8554/phone2\n"
          "PARKIT 입력: rtsp://127.0.0.1:8554/phone1 또는 /phone2\n"
          "이 창을 켜 두세요. 종료: Ctrl+C", flush=True)
    child = subprocess.Popen([str(dest), str(BASE / "larix_receiver.yml")], cwd=dest.parent)
    try:
        code = child.wait()
    except KeyboardInterrupt:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
