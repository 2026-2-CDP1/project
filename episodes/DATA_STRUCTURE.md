# 에피소드 가공 데이터 구조

> 원본 영상에서 `episodes/<ID>/`까지 거치는 입력 표, 중간 결과, 최종 폴더 형식 정리
> 원본 영상 폴더와 파일 형식은 [RAW_DATA.md](RAW_DATA.md)를 보세요.
> 기준 시점: 2026-11-04

---

## 1. 전체 흐름

```
~/data/raw/T{n}/<시점>/<ID>/          원본 (RAW_DATA.md)
        │
        │  extract_all.py  (MediaPipe, 영상 전체, 라벨 불필요)
        │     └ 작업 손: episodes/records.csv
        ▼
~/out/seohyeon/kp_cache_v2/T{n}/      중간 결과: 카메라별 2D / ego 엄지-검지 거리
        │
        │  process_episode.py --cache ... --start --end
        │     ├ 카메라 간 시간: sync/sync.csv
        │     └ 카메라 위치:    sync/exo_calib_T{n}.json
        ▼
<out>/episodes/<ID>/                  최종: 3D 키포인트, 손목 자세, 집기 거리, meta.json
        │
        │  labels.py init → 사람이 구간 라벨 작성 → validate.py
        ▼
5-1·5-2·6번이 읽는 형식
```

| 단계 | 상태 (2026-11-04) |
|---|---|
| 원본 업로드 | 완료. T1~T3 각 75개 에피소드 × 4시점 |
| sync.csv, exo_calib_T{n}.json | 완료 (2번 sync 담당) |
| records.csv | 완료. 225행, 촬영 기록지 PDF에서 옮김 |
| 2D 추출 (kp_cache_v2) | 완료. 900개, 오류 0 |
| `--start/--end` 라벨 | **미정**. 이게 있어야 process_episode.py 를 돌릴 수 있음 |
| episodes/<ID>/ 생성 | 아직 |

## 2. 입력 표

### 2.1 `episodes/records.csv` - 촬영 기록지

촬영 기록지(PDF)를 에피소드당 한 줄로 옮긴 표예요. `extract_all.py`가 작업 손을 여기서 읽어요.

| 열 | 예 | 설명 |
|---|---|---|
| `episode_id` | `T1_E001` | |
| `task` | `컵 잡기` | `constants.TASKS` 와 글자까지 같음 |
| `layout` | `보통 거리, 트인 곳에 컵을 둠` | 물체 배치 |
| `speed` | `보통` | |
| `purpose` | `학습` / `예비` | 001~065 학습, 066~075 예비 |
| `case_tags` | | `constants.CASES` 중에서, `\|` 구분 |
| `collector` | `정구현` | 전 에피소드 같음 |
| `hand` | `right` / `left` | 작업한 손 |
| `notes` | | 기록지 특이사항 |

**작업 손 규칙** (T1~T3 공통)

| 에피소드 | 손 |
|---|---|
| E001~E002 | 오른손 |
| E003~E005 | 왼손 |
| E006~E065 | 5개씩 번갈아, 오른손부터 (006~010 오른손, 011~015 왼손, …, 061~065 왼손) |
| E066~E075 | 짝수 번호 오른손, 홀수 번호 왼손 |

결과적으로 태스크마다 오른손 37개, 왼손 38개예요.

### 2.2 `sync/sync.csv` - 카메라 간 시간 차 (2번 작성)

기준 카메라는 **exo_front**이고 exo_front 행은 없어요. 에피소드마다 `left`, `right`, `ego` 3행이 있어서 총 675행(225 × 3)이에요. 전부 `usable=1`이에요.

| 열 | 설명 |
|---|---|
| `episode_id`, `camera` | `camera` = `left` / `right` / `ego` |
| `offset_start_s`, `offset_end_s` | 그 카메라 시각에 더하면 기준 시각이 되는 값 (s). 시작과 끝 |
| `drift_s` | `offset_end_s - offset_start_s` |
| `s0`, `s1` | 그 카메라 시각 기준 시작·끝 지점 (s) |
| `usable` | 1이면 사용 |
| `note` | `박수 (...)` 또는 `시계 모델 (...)` |

카메라 시각 `t`를 기준 시각으로 바꾸는 식 (`multiview.cam_time_to_ref`):

```
t_ref = t + off0 + (off1 - off0) · (t - s0) / (s1 - s0)
```

- **박수 행**: `s0`, `s1`이 실제 시작·끝 박수 시각이에요. note의 `frames ref A/B`가 exo_front 기준 박수 프레임 번호예요.
- **시계 모델 행**: 박수를 못 찾은 경우 폰 시계 관계로 추정한 값이에요. 이때 `s0`, `s1`은 첫·마지막 프레임 시각이라 **박수 위치가 아니에요.**
- 행 수: T1은 박수 158행 + 시계 모델 67행, T2는 전부 시계 모델, T3는 박수 3행. 시계 모델의 LOO 오차는 36 ms 이하예요.

### 2.3 `sync/exo_calib_T{n}.json` - exo 카메라 3대 위치 (2번 작성)

태스크마다 카메라를 다시 설치해서 파일이 태스크별로 있어요. `T2_E017`이면 `exo_calib_T2.json`이에요.

```
{
  "ref_camera": "front",
  "valid_for": "T1 세션 (T1_check)",
  "board": {"inner_corners": [21, 15], "square_m": 0.025, ...},
  "table_frame": "원점 = ..., x = front 에서 볼 때 오른쪽, y = 작업자 쪽, z = 위. 단위 m. pc = R @ p + t",
  "cameras": {
    "front": {"K", "R", "t", "dist", "image_size", "camera_center_table_m",
              "reproj_err_px_mean", "reproj_err_px_max", ...},
    "left":  {...},
    "right": {...}
  }
}
```

- **테이블 좌표계**: 체커보드 기준이고 단위는 m예요. 최종 3D 키포인트도 모두 이 좌표계예요.
  - 원점: front에서 가장 가까운 줄의 가장 왼쪽 내부 코너
  - x: front에서 볼 때 오른쪽, y: 작업자 쪽, z: 위
- `R`, `t`는 테이블 → 카메라 변환이에요 (`pc = R @ p + t`).
- 픽셀은 영상을 디코딩한 그대로예요(회전 없음). 재투영 오차는 약 0.8~1.1 px예요.

## 3. 중간 결과: `kp_cache_v2`

`extract_all.py`가 영상 **전체**에서 뽑아 둔 결과예요. MediaPipe가 가장 오래 걸리는 단계라 라벨 없이 먼저 돌려 둔 거예요. `process_episode.py --cache`가 이걸 읽으면 영상을 다시 처리하지 않아요.

```
~/out/seohyeon/kp_cache_v2/
├── T1/
│   ├── exo_front/T1_E001_right.npy     # 파일 이름 = <ID>_<작업 손>.npy
│   ├── exo_left/ ...
│   ├── exo_right/ ...
│   ├── ego/T1_E001_right.npy
│   └── summary.csv
├── T2/
└── T3/
```

| 파일 | shape | 내용 |
|---|---|---|
| `exo_*/<ID>_<hand>.npy` | `(Tc, 7, 2)` float | 픽셀 좌표 (u, v), 못 찾으면 NaN. 키포인트 순서는 아래 `EXO_NAMES` |
| `ego/<ID>_<hand>.npy` | `(Tc,)` float | 엄지끝-검지끝 거리. MediaPipe world landmark 기준 (약 m 단위), 못 찾으면 NaN |

- `Tc`는 그 카메라 영상의 프레임 수예요. exo_front는 약 60 fps, 나머지는 약 30 fps예요.
- 영상이 `data.jsonl` 프레임 줄보다 1프레임 많을 수 있어요(T1 30개). `io_utils.match_frames`가 마지막 프레임을 버려 맞춰요.

**`EXO_NAMES`** (`extractors.py`, `keypoint_names.json`과 같음)

| # | 이름 | 출처 |
|---|---|---|
| 0 | `shoulder` | Pose |
| 1 | `elbow` | Pose |
| 2 | `wrist` | Hands (Pose 손목 근처를 잘라서 검출) |
| 3 | `index_mcp` | Hands |
| 4 | `pinky_mcp` | Hands |
| 5 | `thumb_tip` | Hands |
| 6 | `index_tip` | Hands |

**`summary.csv`**: 작업 하나당 한 줄이에요.

| 열 | 설명 |
|---|---|
| `camera`, `episode_id`, `hand` | |
| `frames`, `times` | 영상 프레임 수, `data.jsonl` 시각 수 |
| `wrist`, `shoulder` | exo: 손목·어깨를 찾은 프레임 비율 |
| `pinch` | ego: 거리를 구한 프레임 비율 |
| `sec`, `error` | 걸린 시간, 오류 메시지(비어 있으면 정상) |

**검출 비율 중앙값** (영상 전체 기준이라 박수 전후 손이 안 보이는 구간도 포함)

| 카메라 | T1 | T2 | T3 |
|---|---|---|---|
| 작업 손 쪽 exo (오른손→exo_right, 왼손→exo_left) | 0.54 | 0.62 | 0.59 |
| 반대쪽 exo | 0.23 | 0.32 | 0.33 |
| exo_front | 0.50 | 0.41 | 0.51 |
| ego | 0.31 / 0.38 | 0.28 / 0.41 | 0.35 / 0.47 |

ego는 오른손 / 왼손 순서예요.

## 4. 최종 결과: `episodes/<ID>/`

`process_episode.py`가 만들고 `validate.py`가 검사해요. 모든 시계열의 길이는 `T = exo_end_frame - exo_start_frame`이에요. 프레임 축은 **exo_front 시간축**이고, 다른 카메라는 sync.csv로 이 시각에 맞춰 보간한 값이에요.

```
<out>/episodes/T1_E001/
├── keypoints.npy            (T, 7, 3)   3D 키포인트, 테이블 좌표계 m, 보정 후
├── keypoints_raw.npy        (T, 7, 3)   삼각측량 그대로 (보정 전)
├── keypoint_names.json      EXO_NAMES
├── wrist_pose.npy           (T, 7)      손목 위치 xyz + 쿼터니언 (w, x, y, z)
├── pinch_dist.npy           (T,)        ego 엄지-검지 거리 (기준 시간축으로 보간)
├── keypoints_2d.npy         (C, T, 7, 2) 카메라별 2D, 카메라 순서 = meta.cameras
├── triangulation_err.npy    (T, 7)      재투영 오차 px
├── exo_ts.csv               기준 시각 (s), 헤더 t_s, T줄
├── ego_ts.csv               ego 전체 프레임 시각 (s)
├── exo.mov                  exo_front 원본 영상 (기준 카메라)
├── exo_left.mov, exo_right.mov
├── ego.mov
├── quality.json
├── meta.json
└── labels.json              labels.py init 으로 만들고 사람이 채움
```

- NaN은 그 프레임에서 값을 못 구했다는 뜻이에요.
- 영상은 원본 전체를 복사해요. 구간 위치는 `meta.json`의 `exo_start_frame`, `exo_end_frame`으로 찾아요.
- `wrist_pose`의 손목 방향은 `wrist`, `index_mcp`, `pinky_mcp` 세 점으로 정해요. 쿼터니언 부호는 인접 프레임과 이어지도록 맞춰요.

### 4.1 `meta.json`

| 키 | 예 | 설명 |
|---|---|---|
| `episode_id`, `task` | `T1_E001`, `컵 잡기` | |
| `case_tags` | `["큰 이동"]` | |
| `collector`, `hand` | `정구현`, `right` | |
| `split` | `train` / `val` | 예비 에피소드는 대체한 원래 에피소드의 용도를 따름 |
| `exo_start_frame`, `exo_end_frame` | `95`, `410` | exo_front 영상 전체 기준 프레임 번호, end 미포함 |
| `exo_fps` | `59.94` | 구간의 실제 fps |
| `sync_offset_s`, `drift_s` | | ego 행 값 (가이드 형식) |
| `keypoint_method` | `triangulation` | |
| `cameras`, `ref_camera` | `["front","left","right"]`, `front` | |
| `camera_sync` | `{left: {s0,s1,off0,off1}, right: ...}` | |
| `smoothing` | `fill<=..., spike>..., savgol(...)` | 보정 설정 문자열 |
| `notes` | | |

### 4.2 `quality.json`

| 키 | shape | 설명 |
|---|---|---|
| `frames` | | T |
| `missing_raw`, `missing_after` | (7,) | 키포인트별 NaN 비율, 보정 전·후 |
| `spike_ratio` | (7,) | 튐으로 지운 비율 |
| `longest_gap_after` | (7,) | 보정 후 가장 긴 연속 NaN (프레임) |
| `tri_ok_ratio` | (7,) | 카메라 2대 이상으로 삼각측량된 비율 |
| `tri_reproj_median_px`, `tri_views_mean` | (7,) | |
| `pinch_missing` | | ego 거리 NaN 비율 |
| `names`, `settings`, `triangulation` | | 키포인트 이름, 보정·삼각측량 설정 |

### 4.3 `labels.json`

프레임 번호는 모두 **exo_front 영상 전체 기준**이에요(구간 안 상대 번호가 아님). 라벨 규칙은 [labeling_rules.md](labeling_rules.md)를 보세요.

```json
{
  "episode_id": "T1_E001", "frame_ref": "exo",
  "exo_start_frame": 95, "exo_end_frame": 410,
  "segments": [{"name": "approach", "start": 95, "end": 160},
               {"name": "grasp", "start": 161, "end": 190}, ...],
  "grasp_frame": 185
}
```

- 구간 이름은 `approach → grasp → transport → release → retreat` 순서예요. 앞 구간의 `end + 1`이 다음 구간의 `start`예요.
- 구간 안의 상대 번호가 필요하면 `grasp_frame - exo_start_frame`처럼 빼서 써요.

## 5. 미정 사항

- **`--start/--end`**: 초기 자세 첫 프레임과 종료 자세 도달 프레임이에요. 라벨링 방법(수동 또는 자동 후보)이 아직 정해지지 않았어요.
- **예비(E066~E075) split**: 어떤 에피소드를 대체하는지 정해야 `train/val`이 정해져요.
- **T2_E008 시간 맞추기**: sync.csv로 옮기면 exo_front 녹화 시각이 left·right·ego(기준 시각 약 111302~111312 s)와 겹치지 않아요. sync.csv note에도 "녹화 시작 차이 이상 - 확인"이 있어요. 2번 확인 전까지 이 에피소드는 처리에서 빼요 (`coverage.py`에서 유일한 오류).
