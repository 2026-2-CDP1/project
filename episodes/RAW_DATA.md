# 원본 영상 데이터 구조 (GPU 서버)

> 서버에 올라간 촬영 원본(에피소드) 폴더 구조와 파일 형식 정리
> 기준 시점: 2026-11-03

---

## 1. 위치

```
/abr/coss37/data/raw/          (= ~/data/raw, 서버 접속 후)
```

- 원본은 **읽기 전용으로 사용**합니다. 파일 이름 변경, 삭제, 덮어쓰기를 하지 마세요.
- 가공 결과는 `~/out/본인이름/`처럼 원본 폴더 밖에 저장합니다.
- 영상은 GitHub(`~/project`)에 올리지 않습니다.

## 2. 폴더 구조

```
~/data/raw/
├── 2026-10-19/                  # 촬영 날짜
│   ├── ego/                     # 머리 착용 카메라 (1인칭)
│   │   ├── T1_E001/             # 에피소드 1개 = 폴더 1개
│   │   │   ├── data.mov
│   │   │   ├── data.jsonl
│   │   │   ├── calibration.json
│   │   │   └── metadata.json
│   │   ├── ...
│   │   └── recording_2026-10-19_18-25-29/   # 이름이 붙지 않은 녹화 (아래 5절 참고)
│   ├── exo_front/               # 삼각대 카메라 - 정면
│   └── exo_left/                # 삼각대 카메라 - 왼쪽
└── 2026-11-03/
    ├── ego/
    │   ├── 1103_check/          # 그날의 체커보드 보정 영상 (시점 폴더마다 있음)
    │   ├── T1_E006/
    │   └── ...
    ├── exo_front/
    ├── exo_left/
    └── exo_right/               # 삼각대 카메라 - 오른쪽 (유일하게 LiDAR 깊이 있음)
        └── T1_E006/
            ├── data.mov
            ├── data.jsonl
            ├── calibration.json
            ├── metadata.json
            └── frames2/         # 깊이 프레임 (exo_right만)
                ├── 00000000.png
                └── ...
```

**경로 규칙:** `~/data/raw/<촬영날짜>/<시점>/<에피소드ID>/`

같은 에피소드 ID 폴더가 시점마다 하나씩 있어요. 예를 들어 `T2_E040`은 `ego/T2_E040`, `exo_front/T2_E040`, `exo_left/T2_E040`, `exo_right/T2_E040`에 있어요. 네 폴더는 같은 장면을 동시에 찍은 거예요.

## 3. 에피소드 ID

`T{태스크번호}_E{세 자리 번호}` (예: `T2_E017`)

| 태스크 | 내용 |
|---|---|
| T1 | 컵 잡기 |
| T2 | 블록 쌓기 |
| T3 | 블록 옮기기 |

번호는 태스크마다 001부터 매기고, 날짜가 바뀌어도 이어서 매겨요. 그래서 한 날짜 안에서는 번호가 띄엄띄엄 있을 수 있어요(아래 표 참고).

## 4. 날짜별 보유 현황

### 2026-10-19 (약 8.1 GB)

| 시점 | T1 | T2 | 이름 없는 `recording_*` | 보정 영상 |
|---|---|---|---|---|
| ego | 45개 (001–005, 016–035, 046–065) | 25개 (001–005, 016–035) | 8 | 없음 |
| exo_front | 45개 (같음) | 25개 (같음) | 6 | 없음 |
| exo_left | 45개 (같음) | 25개 (같음) | 6 | 없음 |

- 이날은 **exo_right가 없어요.** 그래서 깊이 데이터도 없어요.

### 2026-11-03 (시점당 약 3.6–3.8 GB, exo_right는 깊이 포함으로 더 큼)

| 시점 | T1 | T2 | T3 | `recording_*` | `1103_check` |
|---|---|---|---|---|---|
| ego | 30개 (006–015, 036–045, 066–075) | 50개 (006–015, 036–075) | 35개 (001–035) | 5 | 있음 |
| exo_front | **29개** (`T1_E074` 없음) | 50개 | 35개 | 4 | 있음 |
| exo_left | 30개 | 50개 | 35개 | 5 | 있음 |
| exo_right | 30개 | 50개 | 35개 | 5 | 있음 |

## 5. 주의사항

- **보정은 날짜별로 따로 해야 해요.** 촬영 날마다 카메라를 다시 설치하기 때문에, 각 날짜의 에피소드는 그날의 보정 영상(`<MMDD>_check`, 예: `1103_check`)으로 보정해야 해요. 날짜를 섞으면 안 돼요.
- **10/19에는 `_check`라는 이름의 폴더가 없어요.** 보정 영상이 이름 없는 `recording_*` 폴더 중 하나일 가능성이 있는데, 아직 확인되지 않았어요.
- **`recording_*` 폴더**는 에피소드 ID가 붙지 않은 녹화예요(테스트, 재촬영 전 녹화 등). 내용이 확인되기 전에는 에피소드로 쓰지 마세요.
- **`exo_front/T1_E074`가 없어요(11/03).** 나머지 세 시점에는 있어요. 다중 시점 처리 시 이 에피소드를 따로 처리해야 해요.
- 폴더 이름이 촬영 프로토콜 문서(`sync/protocol.md`)와 달라요. 실제로는 `T2_E017_ego/` 대신 `ego/T2_E017/`, `CALIB_...` 대신 `1103_check/`를 써요.

## 6. 파일 형식

에피소드 폴더 하나에는 촬영 앱(Spectacular Rec)이 만든 파일이 그대로 들어 있어요.

### `data.mov` - 영상

- 1920×1440, 약 30 fps (RGB)
- 소리는 없어요(앱이 녹음하지 않음).

### `data.jsonl` - 프레임 시각 + IMU 센서

한 줄에 JSON 하나예요. 줄 종류는 두 가지예요.

**① 프레임 줄** - 영상 프레임 하나당 한 줄

```json
{"number": 0, "time": 54619.106,
 "frames": [{"cameraInd": 0, "colorFormat": "rgb", "exposureTimeSeconds": 0.0167,
             "calibration": {"focalLengthX": 1460.08, "focalLengthY": 1460.08,
                             "principalPointX": 958.59, "principalPointY": 713.65}}]}
```

- `number`: 프레임 번호(0부터). `data.mov`의 프레임 순서와 같아요.
- `time`: 초 단위 시각. **폰 내부 시계 기준이라 폰마다 시작값이 달라요.** 시점 간 맞추기는 시작/끝 박수 장면으로 해요.
- `calibration`: 그 프레임의 카메라 내부 파라미터. 자동초점 때문에 프레임마다 조금씩 달라질 수 있어요.
- exo_right는 `frames`에 항목이 하나 더 있어요(깊이).
  `{"cameraInd": 1, "colorFormat": "gray", "depthScale": 0.001, "aligned": true, ...}`

**② 센서 줄** - IMU 값 (각각 약 100 Hz)

```json
{"time": 54619.031, "sensor": {"type": "gyroscope", "values": [0.056, -0.128, -0.020]}}
```

| type | 의미 | 단위 |
|---|---|---|
| `accelerometer` | 가속도 (중력 포함) | m/s² |
| `gyroscope` | 각속도 | rad/s |
| `magnetometer` | 자기장 | µT |

예시 크기: 약 22초 에피소드 → 프레임 줄 약 650줄, 센서 줄 약 6,500줄

### `calibration.json` - 카메라 보정값 (녹화 시작 시점)

```json
{"cameras": [{"model": "pinhole", "imageWidth": 1920, "imageHeight": 1440,
              "focalLengthX": 1460.08, "focalLengthY": 1460.08,
              "principalPointX": 958.59, "principalPointY": 713.65,
              "imuToCamera": [[0,-1,0,0],[-1,0,0,0],[0,0,-1,0],[0,0,0,1]]}]}
```

- 핀홀 모델 내부 파라미터(fx, fy, cx, cy)와 IMU→카메라 4×4 변환 행렬이에요.
- exo_right는 `cameras`가 2개예요(0 = RGB, 1 = 깊이).
- **카메라 사이의 위치 관계(외부 파라미터)는 들어 있지 않아요.** 이건 그날의 `_check` 영상으로 따로 구해야 해요.

### `metadata.json` - 기기 정보

```json
{"platform": "ios", "device_model": "iPhone17,3"}
```

| 시점 | device_model |
|---|---|
| ego | iPhone17,3 |
| exo_front | iPhone18,5 |
| exo_left | iPhone14,7 |
| exo_right | iPhone19,2 (LiDAR) |

### `frames2/` - 깊이 프레임 (exo_right만)

- `00000000.png`, `00000001.png`, ... 파일 이름(8자리) = `data.jsonl`의 프레임 `number`
- 256×192, **16비트 흑백 PNG**
- 픽셀값 × `depthScale`(0.001) = **거리(m)**. 즉 픽셀값이 mm 단위예요.
- RGB 영상에 맞춰 정렬돼 있어요(`aligned: true`). 해상도는 RGB(1920×1440)의 1/7.5이니 좌표를 쓸 때 비율을 맞춰야 해요.
- 보통 PNG 수 = `data.jsonl` 프레임 줄 수예요. 그런데 **11/03의 `T1_E010`, `T2_E036`, `T2_E063`, `T3_E017`은 PNG가 마지막에 1장 더 있어요.** 이 마지막 PNG에는 대응하는 프레임 줄(시각)이 없어요. 원본부터 그렇게 저장된 거라, `data.jsonl`에 있는 `number`의 PNG만 쓰면 돼요.

```python
import cv2, numpy as np
d = cv2.imread("frames2/00000000.png", cv2.IMREAD_UNCHANGED)  # uint16, (192, 256)
depth_m = d.astype(np.float32) * 0.001                        # 단위: m
```
