# Larix 휴대폰 + 웹캠 혼합 입력

Larix Broadcaster의 RTSP 송출을 사용한다. 가상 웹캠 번호가 아니라 주소로 연결한다.

```text
휴대폰 1 Larix ─ /phone1 ─┐
휴대폰 2 Larix ─ /phone2 ─┴─ MediaMTX 수신기(컴퓨터) ─ RTSP ─ PARKIT
웹캠 1·2 ───────────────────────────────────────────── USB ─ PARKIT
```

## 1. 컴퓨터에서 수신기 실행

PARKIT 폴더에서 실행한다. 현재 Mac에는 수신기를 설치해 두었다.

```sh
.venv-gui/bin/python docs/도구/occupancy_v7/larix_receiver.py
```

Windows는 프로젝트 가상환경을 활성화한 터미널에서 다음을 실행한다. 처음 한 번 `--install`을 붙여 Windows용 수신기를 설치한다. 다음부터는 빼도 된다.

```powershell
python docs/도구/occupancy_v7/larix_receiver.py --install
```

수신기는 이 창에서 계속 실행해야 한다. 종료는 Ctrl+C. Mac의 다른 컴퓨터에서도 최초에는 `--install`이 필요하다. 공식 MediaMTX 1.21.3 실행 파일을 SHA-256 검증 후 `.runtime/`에 설치한다. Git에는 실행 파일을 넣지 않으며 Docker는 필요 없다.

## 2. 휴대폰에서 송출

컴퓨터와 휴대폰을 같은 LAN/Wi-Fi에 연결한다. Larix **Broadcaster**의 연결 설정에서 RTSP 송출 연결을 추가한다. 이름은 자유롭게 정한다.

- 휴대폰 1: 수신기 창에 표시된 `rtsp://컴퓨터IP:8554/phone1`
- 휴대폰 2: `rtsp://컴퓨터IP:8554/phone2`
- 필요하면 phone3, phone4도 지원한다.
- 전송 방식 선택이 있으면 **TCP**, 영상 코덱 **H.264**, 우선 **1280×720 / 30fps**, 오디오는 끈다.
- 원하는 광각 렌즈를 선택하고 휴대폰 방향·위치를 고정한 다음 송출을 시작한다.

**휴대폰에는 127.0.0.1을 입력하면 안 된다.** 그것은 휴대폰 자기 자신이다. 수신기 창에 표시된 컴퓨터 IP를 사용한다. VPN/여러 네트워크 때문에 안내 주소가 다르면 컴퓨터의 실제 LAN IP를 확인하고 `--host-ip 192.168.x.x`로 안내 주소를 지정할 수 있다.

Windows 방화벽 알림이 뜨면 시연용 개인 네트워크에서 수신기 TCP 8554 연결을 허용한다. 공유기 외부 포트 개방은 필요 없다. 학교/게스트 Wi-Fi의 기기 간 통신 차단이 있으면 같은 공유기의 기기 간 통신이 가능한 네트워크를 사용한다.

Larix는 RTSP 송출을 지원한다: [공식 안내](https://softvelum.com/larix/).
수신기는 [MediaMTX](https://mediamtx.org/docs/features/publish-and-read)를 사용한다.

## 3. 보정

```sh
.venv-gui/bin/python docs/도구/occupancy_v7/calibrate.py
```

Windows에서는 맨 앞 `.venv-gui/bin/python`을 활성화한 가상환경의 `python`으로 바꾼다. `calibrate_v2.py`도 같은 입력 방식을 지원한다.

시작 화면의 **번호 / Larix 주소**에서 다음처럼 선택한다.

| 카메라 이름 예 | PARKIT 입력 | 촬영 방식 |
|---|---|---|
| CAM-1 | rtsp://127.0.0.1:8554/phone1 | 상부 촬영 · 박스 정중앙 |
| CAM-2 | rtsp://127.0.0.1:8554/phone2 | 상부 촬영 · 박스 정중앙 |
| CAM-3 | 실제 웹캠 번호 | 사선 촬영 · 기존 바닥 접점 |
| CAM-4 | 실제 웹캠 번호 | 사선 촬영 · 기존 바닥 접점 |

이 주소는 **수신기와 PARKIT을 같은 컴퓨터에서 실행할 때**의 주소다. 다른 컴퓨터에 수신기가 있으면 그 컴퓨터 IP를 쓴다. 웹캠 번호는 장치마다 다르므로 미리보기로 확인한다.

카메라를 하나씩 골라 보정 화면을 열고 촬영 방식, 기준점, 감시 칸을 설정한 다음 저장한다. 각 카메라의 보정은 별도로 한다. 새 휴대폰에 기존 웹캠 보정을 복사해서 쓰지 않는다. 렌즈·줌·크롭·촬영 방향을 바꾸면 다시 보정한다. 추가 렌즈 왜곡 보정을 자동 적용하지는 않는다.

시작 창의 썸네일은 USB 웹캠용이다. Larix 영상은 보정 화면을 연 뒤 표시된다. 주소 저장은 입력 연결만 저장하며 보정을 대신하지 않는다.

## 4. 점유 판정과 웹 지도

```sh
.venv-gui/bin/python docs/도구/occupancy_v7/occupancy.py --web
```

시작 화면에서 보정한 CAM-1~4 중 사용할 카메라를 체크하고 입력 주소를 확인한 뒤 판정 시작을 누른다. 휴대폰은 웹캠 번호 목록에 나타날 필요가 없다. 웹 지도는 http://127.0.0.1:8765/ 에서 본다.

연결 오류가 나면 수신기 → Larix 송출 → 보정 순서로 확인한다. 수신기 창에 phone1/phone2 publishing 메시지가 나와야 한다. 송출 전에는 보정 화면이 영상을 열 수 없다. 점유 프로그램은 끊어진 네트워크 입력에 재연결을 시도한다.

네트워크 지연이 카메라마다 크거나 다르면 차량 위치가 어긋날 수 있다. 현재 추적 시각은 프레임 수신 시각이며 휴대폰 촬영 시각을 정확히 동기화하는 시스템은 아니다. 실제 이동 차량으로 다중 카메라 ID 연결을 추가 검증해야 한다.

## 변경과 검증

- `camera_input.py`: 웹캠 번호/영상 주소 공통 처리.
- `calibrate.py`: 주소 선택·저장·재열기, 연결 실패 시 다른 카메라 보정으로 바뀌는 오류 방지. v2는 이 기능을 상속한다.
- `occupancy.py`: 카메라별 URL 입력, USB 웹캠과 혼합 선택.
- `parkit_video.py`: RTSP 연결/읽기 시간 제한 및 종료 시 동시 read/release 방지.
- `larix_receiver.py`, `larix_receiver.yml`: 수신기 설치·실행, phone1~4 경로.
- `test_camera_input.py`: 주소와 설정 저장, 입력 변경 실패 시 복구 검증.

프로젝트 가상환경을 활성화하고 이 폴더에서:

```sh
python -m unittest test_camera_input test_multi_camera test_vehicle_tracker test_tracking_integration test_tracking_replay test_mycar_links -q
```

Mac에서 합성 RTSP 영상 두 개를 MediaMTX로 동시에 송출해 서로 다른 프레임 수신, 보정 입력, 한 송출 중단 시 다른 입력 유지, 스레드 정상 종료를 확인했다. 실제 Larix 휴대폰과 Windows 실기기 검증은 별도로 필요하다. 이번 작업은 영상 입력 경로를 변경하며 기존 점유 판정·차량 매칭 알고리즘을 변경하지 않는다.
