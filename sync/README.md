# sync/ — 카메라 시간 맞추기 · exo 외부 보정

> 담당: 신서연 (2. 촬영·싱크) · 기준 시점 2026-11-04 · 대상: 영상·키포인트·좌표를 쓰는 모든 역할

이 폴더에서 다른 역할이 가져다 쓰는 결과는 **두 가지**예요.

| 필요한 것 | 파일 | 한 줄 설명 |
|---|---|---|
| 카메라끼리 시간 맞추기 | `sync.csv` | 에피소드 × 카메라마다 기준 카메라(`exo_front`)와의 시간 차 |
| 2D → 3D (테이블 좌표) | `exo_calib_T1.json` · `exo_calib_T2.json` · `exo_calib_T3.json` | **태스크마다 하나**씩 있는 exo 카메라 3대의 K·R·t |

---

## 1. 대상 데이터

```
~/data/raw/T{1,2,3}/{exo_front, exo_left, exo_right, ego}/T{n}_E001 … T{n}_E075/   (data.mov, data.jsonl, …)
~/data/raw/T{1,2,3}/{exo_front, exo_left, exo_right}/T{n}_check/                    (체커보드 영상, 보정용)
```

- 태스크 3개 × 에피소드 75개 × 카메라 4대. 원본 zip은 `~/data/staging/`.
- 영상은 **회전 없이 저장된 그대로** 처리했어요 (`camera_setup.md` 규칙 1).

---

## 2. `sync.csv` — 시간 차 표

### 결과 요약

| 태스크 | 행 | `usable=1` | 만든 방법 | 검증 오차 (최대) |
|---|---|---|---|---|
| T1 | 225 | **225** | 박수 158 + 시계 모델 67 | 29 ms |
| T2 | 225 | **225** | 시계 모델 225 (손 표시 기준점 8에피소드) | 36 ms |
| T3 | 225 | **225** | 박수 3 + 시계 모델 222 (손 표시 기준점 10에피소드) | 36 ms |

검증 오차 = 기준점 에피소드를 하나씩 빼고 나머지로 그 에피소드를 예측했을 때의 오차(LOO). 1프레임 = 33 ms (30 fps).

### 열

| 열 | 뜻 |
|---|---|
| `episode_id` | `T2_E017` 형식 |
| `camera` | `left` / `right` / `ego` (기준 카메라 `exo_front`는 행이 없음 = 시간 차 0) |
| `offset_start_s`, `offset_end_s` | 기준 카메라 시각 − 이 카메라 시각 (s), `s0`·`s1` 시점에서 |
| `drift_s` | `offset_end_s − offset_start_s` |
| `s0`, `s1` | 이 카메라 자기 시각 (s). 위 두 offset을 잰 시점 |
| `usable` | 1 = 사용 가능. 지금은 전부 1 |
| `note` | 앞머리로 행 종류 구분 (아래) |

- 시각은 모두 각 폰 `data.jsonl`의 프레임 시각(`time`)이에요. 폰 부팅 기준 시계라 카메라마다 수만 초씩 차이 나는 게 정상이에요.

### 시간 변환

이 카메라의 시각 `t` → 기준 카메라(`exo_front`) 시각:

```
t_ref = t + offset_start_s + drift_s × (t − s0) / (s1 − s0)
```

3번 코드에 이미 있어요: `episodes/tools/io_utils.read_sync_row(path, episode_id, camera=...)` → `multiview.cam_time_to_ref(...)`.

### `note` 앞머리

| 앞머리 | 뜻 |
|---|---|
| `박수 (모델과 ±N ms)` | 박수 자동 검출 값을 그대로 쓴 행. 시계 모델과의 차이를 함께 적음 |
| `시계 모델 (구간 k, 기준점 n에피소드, LOO 95% N ms)` | 시계 모델로 계산한 행. `s0`·`s1`은 이 카메라의 첫·마지막 프레임 시각 |
| `…; 박수: <사유>` | 박수 자동 검출이 왜 실패했는지 (기록용, 지우지 않음) |

---

## 3. `exo_calib_T{n}.json` — exo 외부 보정

- **에피소드 ID의 태스크 번호로 파일을 골라요.** `T2_E017` → `exo_calib_T2.json`. 태스크마다 카메라 자리가 조금씩 달라요 (태스크 사이 2~4 cm).
- 형식은 3번 `episodes/tools/multiview.load_calibs`와 같아요.

```json
{"ref_camera": "front",
 "cameras": {"front": {"K": 3x3, "R": 3x3, "t": [3], "dist": [5]}, "left": {…}, "right": {…}}}
```

- 규약: 테이블 → 카메라, `p_cam = R @ p_table + t` (m). K는 앱이 기록한 값(왜곡 없음). 픽셀은 저장된 화면(1920×1440) 그대로예요.
- **테이블 좌표계:** 원점 = 체커보드 내부 코너 중 front에서 가장 가까운 줄의 가장 왼쪽 / x = front에서 볼 때 오른쪽 / y = front에서 멀어지는 쪽(작업자 쪽) / z = 위 / 단위 m.
- `left`·`right`는 **작업자 기준**이에요 (exo_left = front에서 볼 때 오른쪽 카메라).
- ego는 머리와 함께 움직여서 보정하지 않아요.

| 태스크 | 재투영 오차 평균 (front / left / right) | 축 방향 그림 확인 |
|---|---|---|
| T1 | 0.88 / 0.84 / 1.06 px | 확인함 |
| T2 | 0.82 / 0.99 / 0.88 px | 확인함 |
| T3 | 0.90 / 0.94 / 0.91 px | 확인함 |

> 예전 `exo_calib.json`(11/03 `1103_check`)은 지금 데이터와 카메라 자리가 달라서(front 19 cm) 2026-11-04에 지웠어요.

---

## 4. 어떻게 만들었나

```
① run_task.sh      박수 검출(detect_claps.py) → 표(make_sync_csv.py) → 놓친 카메라 안내 검출 반복 → 보정(make_exo_calib.py)
② fit_clock.py     박수가 없거나 틀린 행을 "세션 시계 모델"로 채움
③ label_sheets.py  (박수 자동 검출이 거의 안 되는 태스크만) 손 표시용 그림 → manual_T{n}.csv → ②에 --manual
```

**세션 시계 모델이란:** 폰 시계 차는 폰이 잠들지 않는 한 한 촬영 세션 안에서 거의 일정해요 (T1 31분 동안 0.6~12.5 ppm). 믿을 수 있는 박수(기준점) 몇 개로 카메라마다 "시계 차 = a + b × 시각" 직선을 맞추고, 나머지 에피소드는 이 직선으로 계산해요.
- 기준점은 박수 행 중 **에피소드 안 드리프트가 1프레임 이하**인 것만 써요. 에피소드 수십 초 동안 실제 드리프트는 1 ms 미만이라, 그보다 크면 박수를 잘못 잡은 거예요.
- T2·T3는 박수가 거의 자동 검출되지 않았어요. 짝짓기 기준값(세션 대표 시간 차)이 0.3~0.8 s 틀려 있었기 때문이에요. 그래서 `manual_T2.csv`·`manual_T3.csv`에 사람이 박수 프레임을 표시했어요.
- 녹화 시작 시각 차이가 3 s 넘게 바뀌면(폰이 잠듦) 구간을 나눠 따로 맞춰요. 지금 세 태스크는 모두 한 구간이에요.
- 자세한 근거와 시험 결과는 `protocol.md` 7절에 있어요.

### 다시 만들기 (서버 계산 노드)

```bash
# ① 박수 + 보정 (태스크마다, 출력은 ~/out/신서연/)
sbatch -p p02 -c 32 --mem=48G run_task.sh T2 0 32
# ③ 필요할 때만: 손 표시 그림
python label_sheets.py --raw ~/data/raw/T2 --claps ~/out/신서연/claps_T2 --auto 8 \
    --offsets exo_left=… exo_right=… ego=… --out ~/out/신서연/label_T2
# ② 시계 모델
python fit_clock.py --sync ~/out/신서연/sync_T2.csv --raw ~/data/raw/T2 \
    --cameras exo_left exo_right ego --manual manual_T2.csv --out sync_T2.csv --report clock_T2.csv
```

- 실행 환경: `~/miniconda3/envs/episodes/bin/python -I` (mediapipe·opencv 있음). `--mem`을 꼭 적어요 (안 적으면 노드 메모리를 전부 잡아요).
- 태스크별 결과를 헤더 한 줄만 남기고 이어 붙인 게 `sync.csv`예요.

---

## 5. 파일 목록

| 파일 | 종류 | 내용 |
|---|---|---|
| `sync.csv` | **결과** | 시간 차 표 (T1~T3, 675행) |
| `exo_calib_T1.json` `exo_calib_T2.json` `exo_calib_T3.json` | **결과** | 태스크별 exo 외부 보정 |
| `manual_T2.csv` `manual_T3.csv` | 입력 | 손으로 표시한 시작 박수 프레임 (시계 모델 기준점) |
| `run_task.sh` | 코드 | 태스크 하나 전체 실행 (①) |
| `detect_claps.py` | 코드 | 영상에서 박수 시각 검출 |
| `make_sync_csv.py` | 코드 | 박수를 카메라끼리 짝지어 시간 차 표 |
| `fit_clock.py` | 코드 | 세션 시계 모델로 빈 행 채우기 (②) |
| `label_sheets.py` | 코드 | 손 표시용 프레임 그림 (③) |
| `make_exo_calib.py` | 코드 | 체커보드 영상 → 외부 보정 |
| `protocol.md` | 문서 | 촬영 절차 (3번용), 7절 시간 차 표 규칙 |
| `camera_setup.md` | 문서 | 영상 방향·카메라 배치·보정 규칙 |

---

## 6. 주의 · 남은 일

- **촬영 도중 카메라가 움직였는지는 아직 검사하지 않았어요.** 보정은 태스크 시작 전 체커보드 한 번이라, 그 뒤에 삼각대를 건드렸다면 보정 파일만 보고는 알 수 없어요. 에피소드 첫 프레임을 체커보드 영상과 비교하는 검사를 할 예정이에요.
- T3 기준점은 E018~E065에 있어요. E001~E017·E066~E075는 직선을 바깥으로 늘린 값이에요 (기울기가 수 ppm이라 영향은 수 ms 예상).
- `camera_setup.md` 4절의 배치 표(1103_front_B 등)와 `sync/exo_calib/`·`setups.csv` 계획은 예전 날짜별 데이터(10/19·11/03) 기준이에요. 지금 데이터는 태스크마다 체커보드가 있어서 `exo_calib_T{n}.json`을 써요.
- 다음 촬영부터: 박수는 **빠르게 쳐서 손바닥이 맞닿는 순간이 뚜렷하게** 보이도록 해 주세요. 천천히 손을 모으면 자동 검출이 안 돼요 (T2·T3).
