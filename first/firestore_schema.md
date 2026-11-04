# Firestore `parkingspace` 컬렉션 데이터 모델 명세서

본 문서는 일반 사용자용 앱(모바일/웹) 및 기타 외부 시스템과의 연동을 위해 Firestore에 저장되는 주차장 데이터(`parkingspace` 컬렉션)의 필드 스키마와 데이터 형식을 정의합니다.

---

## 1. 단일 문서 구조 (`parkingspace/{parkingId}`)

| 필드명 | 데이터 타입 | 설명 | 예시 값 |
| :--- | :--- | :--- | :--- |
| **`name`** | `string` | 주차장 이름 | `"서울시청 앞 공영주차장"` |
| **`category`** | `string` | 주차장 구분 (`public`: 공영 / `private`: 민영 / `business`: 상가) | `"public"` |
| **`fee_type`** | `string` | 요금 유형 (`paid`: 유료 / `free`: 무료 / `conditional free`: 조건부 무료) | `"paid"` |
| **`location`** | `GeoPoint` | 주차장의 지리적 위/경도 좌표 | `latitude: 37.5665`, `longitude: 126.9780` |
| **`owner_uid`** | `string` | 해당 주차장을 등록한 관리자의 Firebase Auth UID | `"admin_user_uid_1234"` |
| **`parking_lots`** | `map (object)` | **[실시간 집계]** 주차칸 타입별 총 구획 수 및 공석(비어있는) 주차칸 수 | 아래 상세 스키마 참조 |
| **`zones`** | `array (list)` | **[도면 상세]** 구역(층)별 배경 이미지, 주차그룹, 개별 주차칸 좌표 및 점유 상태 목록 | 아래 상세 스키마 참조 |

---

## 2. `parking_lots` 필드 구조 (실시간 공석 집계 데이터)
사용자 앱에서 **"전체 주차 공간 대비 현재 주차 가능한 빈자리 현황"**을 목록이나 요약 정보로 빠르게 보여줄 때 사용합니다.

* **키(Key) 매핑**:
  - `general`: 일반 차량 구획
  - `compact`: 경차 전용 구획
  - `electric`: 전기차 충전 구획
  - `disabled`: 장애인 전용 구획
  - `National`: 국가유공자 우선 구획
  - `Female`: 여성우선 구획
* **내부 속성**:
  - `total`: 해당 주차장의 해당 차종 타입 전체 주차 면수 (`number`)
  - `current`: **현재 이용 가능한(비어있는) 공석 수** (`number`)

### JSON 데이터 예시:
```json
"parking_lots": {
  "general": { "total": 50, "current": 23 },
  "compact": { "total": 10, "current": 4 },
  "electric": { "total": 5, "current": 2 },
  "disabled": { "total": 3, "current": 3 },
  "Female": { "total": 5, "current": 1 },
  "National": { "total": 2, "current": 0 }
}
```

---

## 3. `zones` 필드 구조 (구역/층 및 주차칸 개별 상세 데이터)
사용자 앱 내의 **"실시간 주차면 지도(평면도)"**를 그리고, 각 주차칸의 정확한 위치와 실시간 차량 주차 여부를 시각적으로 렌더링할 때 사용하는 원천 데이터입니다.

* **구역(Zone)별 정보 (`array`)**:
  - `id`: 구역 고유 ID (`string`)
  - `name`: 구역 이름 (`string` - 예: `"B1층"`, `"B2층"`)
  - `bg_img_with_spots`: 주차칸이 그려진 도면 배경 이미지 URL (`string`)
  - `bg_img_empty`: 주차칸이 없는 빈 도면 배경 이미지 URL (`string`)
  - `bg_scale_x` / `bg_scale_y`: 배경 이미지의 가로/세로 배율 (`number`)
  - `bg_pos_x` / `bg_pos_y`: 배경 이미지의 X, Y 축 오프셋 (`number`)
  - `spot_opacity`: 주차칸 UI의 불투명도 (`number`)
  - `created_at`: 구역 생성 타임스탬프 (`number`)
  - **`groups` (`array`)**: 주차구획 라인(그룹) 목록
    - `id`: 그룹 고유 ID (`string`)
    - `name`: 그룹 이름 (`string` - 예: `"A구역"`)
    - `x` / `y`: 그룹의 기준 글로벌 X, Y 좌표 (`number`)
    - `rotation`: 그룹의 회전 각도 (`number`, 0 ~ 360)
  - **`spots` (`array`)**: 개별 주차 공간 목록
    - `id`: 주차칸 고유 ID (`string`)
    - `groupId`: 소속된 그룹(Line) ID (`string`)
    - `localX` / `localY`: 소속 그룹 기준 상대 X, Y 좌표 (`number`)
    - `type`: 주차칸 차종 타입 (`string` - `"NORMAL"`, `"COMPACT"`, `"EV"`, `"HANDICAP"`, `"PATRIOT"`, `"WOMAN"`)
    - **`occupied`**: **실시간 주차 상태** (`boolean`)
      - `true` : 🚗 **주차(점유) 중** (차량이 대어져 있음)
      - `false`: 🟢 **비어 있음** (주차 가능)

### JSON 데이터 예시:
```json
"zones": [
  {
    "id": "zone_1717800000000",
    "name": "B1층",
    "bg_img_empty": "https://firebasestorage.googleapis.com/v0/b/.../o/B1_empty.png",
    "bg_img_with_spots": "https://firebasestorage.googleapis.com/v0/b/.../o/B1_spots.png",
    "bg_scale_x": 1.0,
    "bg_scale_y": 1.0,
    "bg_pos_x": 100,
    "bg_pos_y": 100,
    "spot_opacity": 0.6,
    "created_at": 1717800000000,
    "groups": [
      {
        "id": "group_1717800001000",
        "name": "A구역",
        "x": 250,
        "y": 120,
        "rotation": 90
      }
    ],
    "spots": [
      {
        "id": "spot_1717800001000_0",
        "groupId": "group_1717800001000",
        "localX": 0,
        "localY": 0,
        "type": "NORMAL",
        "occupied": true
      },
      {
        "id": "spot_1717800001000_1",
        "groupId": "group_1717800001000",
        "localX": 40,
        "localY": 0,
        "type": "EV",
        "occupied": false
      }
    ]
  }
]
```
