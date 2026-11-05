# 원본 영상 데이터 구조 (GPU 서버)

> 서버에 올라간 촬영 원본(에피소드) 폴더 구조와 파일 형식 정리
> 기준 시점: 2026-11-04 (태스크 단위로 다시 올림. 이전 날짜별 구조는 삭제됨)

---

## 1. 위치

```
/abr/coss37/data/raw/          (= ~/data/raw, 서버 접속 후)
```

- 원본은 **읽기 전용으로 사용**합니다. 파일 이름 변경, 삭제, 덮어쓰기를 하지 마세요.
- 가공 결과는 `~/out/본인이름/`처럼 원본 폴더 밖에 저장합니다.
- 영상은 GitHub(`~/project`)에 올리지 않습니다.
- 원본 zip(`T{n}_{시점}.zip`)은 공유 드라이브에 있어요. 서버에서는 압축을 풀고 zip 내용과 파일 크기까지 대조 검증한 뒤 zip을 지웠어요.

## 2. 폴더 구조

```
~/data/raw/
├── T1/                          # 태스크
│   ├── ego/                     # 머리 착용 카메라 (1인칭)
│   │   ├── T1_E001/             # 에피소드 1개 = 폴더 1개
│   │   │   ├── data.mov
│   │   │   ├── data.jsonl
│   │   │   ├── calibration.json
│   │   │   └── metadata.json
│   │   ├── ...
│   │   ├── T1_E075/
│   │   └── T1_check/            # 이 태스크의 체커보드 보정 영상
│   ├── exo_front/               # 삼각대 카메라 - 정면 (기준 카메라)
│   ├── exo_left/                # 삼각대 카메라 - 왼쪽
│   └── exo_right/               # 삼각대 카메라 - 오른쪽 (유일하게 LiDAR 깊이 있음)
│       └── T1_E001/
│           ├── data.mov
│           ├── data.jsonl
│           ├── calibration.json
│           ├── metadata.json
│           └── frames2/         # 깊이 프레임 (exo_right만)
│               ├── 00000000.png
│               └── ...
├── T2/                          # 구조 같음
└── T3/
```

**경로 규칙:** `~/data/raw/<태스크>/<시점>/<에피소드ID>/`

같은 에피소드 ID 폴더가 시점마다 하나씩 있어요. 예를 들어 `T2_E040`은 `T2/ego/T2_E040`, `T2/exo_front/T2_E040`, `T2/exo_left/T2_E040`, `T2/exo_right/T2_E040`에 있어요. 네 폴더는 같은 장면을 동시에 찍은 거예요.

## 3. 에피소드 ID

`T{태스크번호}_E{세 자리 번호}` (예: `T2_E017`)

| 태스크 | 내용 |
|---|---|
| T1 | 컵 잡기 |
| T2 | 블록 쌓기 |
| T3 | 블록 옮기기 |

## 4. 보유 현황 (2026-11-04 검증)

| 태스크 | 시점마다 | 크기 |
|---|---|---|
| T1 | `T1_E001`~`T1_E075` 75개 + `T1_check` | 9.4 GB |
| T2 | `T2_E001`~`T2_E075` 75개 + `T2_check` | 7.0 GB |
| T3 | `T3_E001`~`T3_E075` 75개 + `T3_check` | 7.6 GB |

- 네 시점(ego, exo_front, exo_left, exo_right) 모두 빠진 번호 없음. 검증 에피소드(076~085)는 촬영하지 않았어요.
- 검증 내용: zip 안 파일과 개수·크기 일치, 폴더마다 핵심 파일 4개 있음, 3초 미만 녹화 없음, exo_right 깊이 PNG 수 = 프레임 수(1장 더 많은 경우 있음, 아래 6절), 회전 꼬리표 0°, 편집기로 다시 저장한 흔적 없음.
- 같은 에피소드라도 카메라마다 녹화 길이가 1~3초씩 달라요(손으로 녹화를 켜고 꺼서). `T3_E060`은 exo_front·exo_left가 약 7초 먼저 켜졌어요. 박수 사이 구간만 쓰므로 문제없어요.

## 5. 주의사항

- **보정은 태스크 단위예요.** 태스크마다 카메라를 다시 설치해서, 각 태스크 에피소드는 그 태스크의 `T{n}_check`로 만든 `sync/exo_calib_T{n}.json`을 써요.
- **카메라 간 시간 맞추기**는 `sync/sync.csv`를 써요 (`sync/README.md`).
- **프레임 수가 시점마다 달라요.** exo_front는 T1·T3 약 60 fps, T2 30 fps이고 나머지는 약 30 fps예요. 시점 간 대응은 프레임 번호가 아니라 `data.jsonl`의 시각으로 맞춰요.

## 6. 파일 형식

에피소드 폴더 하나에는 촬영 앱(Spectacular Rec)이 만든 파일이 그대로 들어 있어요.

### `data.mov` - 영상

- 1920×1440, RGB. exo_front T1·T3 약 60 fps, T2 30 fps / ego·exo_left·exo_right 약 30 fps
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
- `time`: 초 단위 시각. **폰 내부 시계 기준이라 폰마다 시작값이 달라요.** 시점 간 맞추기는 `sync/sync.csv`로 해요.
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
- **카메라 사이의 위치 관계(외부 파라미터)는 들어 있지 않아요.** 이건 태스크별 `T{n}_check` 영상으로 구한 `sync/exo_calib_T{n}.json`에 있어요.

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
- 일부 에피소드는 PNG가 프레임 줄보다 1장 더 많아요(앱이 원래 그렇게 저장함). 파일 이름으로 프레임 `number`와 맞추면 마지막 1장은 남으니 무시하면 돼요.
- 픽셀값 × `depthScale`(0.001) = **거리(m)**. 즉 픽셀값이 mm 단위예요.
- RGB 영상에 맞춰 정렬돼 있어요(`aligned: true`). 해상도는 RGB(1920×1440)의 1/7.5이니 좌표를 쓸 때 비율을 맞춰야 해요.

```python
import cv2, numpy as np
d = cv2.imread("frames2/00000000.png", cv2.IMREAD_UNCHANGED)  # uint16, (192, 256)
depth_m = d.astype(np.float32) * 0.001                        # 단위: m
```
