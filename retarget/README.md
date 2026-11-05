# retarget — 사람 손목 궤적 → 로봇 관절 궤적 (4번, 제효정)

3번이 만든 에피소드(`episodes/<ID>/`)의 사람 손목·팔꿈치 궤적을 KUKA iiwa14 관절 각도로 바꾼다.
같은 목표를 세 알고리즘으로 풀어 결과를 비교한다. 5-1(재생), 5-2(지표), 6(RLDS)이 결과 npz를 쓴다.

| 폴더 이름 | 알고리즘 | 스크립트 | 방식 |
|---|---|---|---|
| `A_DLS` | 감쇠 최소제곱 IK | `dls.py` | pinocchio, 야코비안 반복, 관절 한계는 계산 뒤 잘라냄 |
| `B_mink` | 제약 QP IK | `mink_ik.py` | mink + daqp, 관절 한계를 QP 제약으로 |
| `C_eaik` | 닫힌 해 IK | `eaik_ik.py` | EAIK, 3번 관절을 격자로 바꿔 가며 해를 전부 구하고 고름 |

> 상태: **v0.1 잠정 (2026-11-05).** 시험 에피소드 1개(T1_E004)로 확인. 로봇 자리·팔 길이 비율은 멘토 확인 전이다 (아래 "미정").

---

## 파일

| 파일 | 내용 |
|---|---|
| `common.py` | 세 알고리즘 공용: 에피소드 읽기, 목표 만들기(좌표 변환·손목 방향·팔꿈치 방향), `to_robot_point`, npz 저장 |
| `dls.py`, `mink_ik.py`, `eaik_ik.py` | A, B, C. 에피소드 하나를 풀어 `<out>/<알고리즘>/<ID>.npz` 저장 |
| `eaik_check.py` | C 사전 점검 (EAIK 구조 인식, TCP 보정값, 왕복 검사, 속도) |
| `config.json` | 설정 기록. 스크립트 기본값과 같음 |

## 실행 (서버)

```bash
cd ~/project && git pull && source ~/use_robot.sh
EP=~/out/seohyeon/episodes/T1_E004          # 3번 결과 폴더의 에피소드
OUT=~/out/본인이름/retarget                 # ~/project 밖
sbatch -p p02 -n 1 -c 1 --mem=4G --job-name retarget_본인이름 -o $OUT/log_A.txt \
  --wrap "python retarget/dls.py --episode $EP --out $OUT"          # B: mink_ik.py, C: eaik_ik.py
```

- 인자 없이 실행하면 `config.json` 설정이다. `--brief`는 한 줄 요약.
- `srun`으로 화면에서 바로 볼 때도 `-n 1`을 넣는다 (빼면 같은 작업이 두 번 돌아 시간이 부풀려짐).
- `--mem`을 꼭 넣는다 (빼면 노드 메모리를 전부 요청해 오래 대기).
- C는 `robot` 환경에 `eaik 1.2.2`가 필요하다 (2026-11-05 설치). 점검: `python retarget/eaik_check.py`.

---

## 목표 만들기 (`common.make_targets`, 세 알고리즘 공통)

| 단계 | 내용 | 설정 (기본) |
|---|---|---|
| 로봇 자리 | 사람 평균 어깨 바로 아래 테이블 점 = 로봇 베이스. 로봇 x = 손 뻗는 쪽(테이블 −y), z = 위 | `--anchor table` |
| 팔 길이 비율 | 기준점에서 사람 좌표를 k배. k = 0.82 / (사람 위팔 + 아래팔, 에피소드 중앙값) | `--scale arm` |
| 손목 → TCP 방향 | 3번 손목 축(x = 검지뿌리 쪽, z = x × 새끼뿌리 쪽). 왼손은 x축 둘레 180° 돌려 오른손과 같은 뜻으로. 플랜지 z = 손바닥 밖 | `--r-off palm` |
| 팔꿈치 방향 | 사람 어깨·팔꿈치·손목에서 팔꿈치가 어깨–손목 축의 어느 쪽에 있는지 (팔 길이와 무관) | 세 알고리즘 모두 입력 (팀 결정 T10) |

TCP는 플랜지(`robot.urdf` `iiwa_link_ee_kuka` = `robot.xml` `attachment_site`). 좌표는 로봇 베이스, m, rad, 쿼터니언 (w, x, y, z).

### 빈 값(NaN) 처리 — 보간하지 않는다

| 상황 | 처리 | npz 표시 |
|---|---|---|
| 손목 위치 없음 | 목표 없음 → 직전 자세 유지, `time_ms` NaN | `target_pos` NaN |
| 맨 앞 프레임에 손목 없음 | 첫 해로 채움 (home → 첫 해 튐 방지) | 〃 |
| 손목 위치만 있고 방향 없음 (사람이 직접 찍은 프레임) | 방향을 직전 값으로 유지해 목표로 (`--pos-only skip`이면 목표 없음) | `target_orient_held` |
| 팔꿈치 없음 (구간 중간) | 직전 팔꿈치 유지 — A: 보조 목표 없이 이어 감 / B: 직전 로봇 팔꿈치 방향을 유지하는 점 / C: 직전 자세와 가장 가까운 해 | `elbow_used` False |
| 맨 앞 프레임에 팔꿈치 없음 | 처음 보인 팔꿈치 방향을 씀 (팔꿈치가 처음 보일 때 관절이 튀는 것 방지) | `target_elbow_backfilled` |

3번이 `keypoint_source.npy` (T, 7, "auto"/"manual"/"none")를 주면 손목을 직접 찍은 프레임을 `target_wrist_manual`로 표시한다.

---

## 알고리즘별

| | A `dls.py` | B `mink_ik.py` | C `eaik_ik.py` |
|---|---|---|---|
| 한 번 계산 | dq = Jᵀ(JJᵀ + λ²I)⁻¹e, λ 0.05, 최대 100회 | QP min Σ cost‖J dq + e‖², 최대 100회 | 3번 관절 1° 격자(341개) × 해 최대 8개 → 한계 안 해만 → 고름 → 0.1°로 다듬기 |
| 관절 한계 | 잘라냄 | QP 제약 | 한계 밖 해를 버림 |
| 팔꿈치 | 손목 우선, 남는 1자유도(영공간)로 사람 팔꿈치 쪽 (`--w-elbow 0.5`) | 손목 1 + 팔꿈치 0.1 가중치 (`--w-elbow 0.1`), home 쪽 1e-6 | 팔꿈치가 사람과 가장 비슷한 해 (`--select human`) |
| 시작 | 직전 프레임 해 (첫 프레임 home) | 〃 | 직전 자세와의 거리를 동점 처리에 씀 |

- `--w-elbow 0` / `--select continuity`면 사람 팔꿈치를 쓰지 않는다 (비교 실험용).
- C의 TCP 보정: EAIK 끝 프레임은 7번 관절 위치라 플랜지보다 플랜지 축으로 0.045 m 안쪽, 회전 차이 0. 시작할 때 자동으로 재고 확인한다.

---

## 출력 npz (`<out>/<알고리즘>/<에피소드ID>.npz`)

배열 길이 T = 에피소드 구간 프레임 수 (exo_front 시간축, 3번 키포인트와 같은 축).

| 키 | 형태 | 내용 | 쓰는 곳 |
|---|---|---|---|
| `q` | (T, 7) rad | 관절 각도 joint1~7. NaN 없음 (목표 없는 프레임은 직전 자세) | 5-1, 5-2, 6 |
| `target_pos` | (T, 3) m | 목표 TCP 위치, 로봇 베이스 좌표. 목표 없으면 NaN | 5-2 성공률 |
| `target_quat` | (T, 4) | 목표 TCP 방향 (w, x, y, z). 목표 없으면 NaN | 5-2 성공률 |
| `time_ms` | (T,) ms | 프레임당 IK 시간 (모델 로드·입출력 제외, 단일 스레드). 목표 없으면 NaN | 5-2 속도 |
| `elbow_used` | (T,) bool | 사람 팔꿈치를 입력에 쓴 프레임 | 5-2 |
| `info` | JSON 문자열 | 좌표 변환 값과 설정 (아래) | 5-2 `to_robot_point` |
| `target_orient_held`, `target_wrist_manual`, `target_elbow_backfilled` | (T,) bool | 위 빈 값 처리 표시 | 참고 |
| `tcp_error`, `rot_error` | (T,) m, rad | 4번이 계산한 손목 위치·방향 오차 | 참고 |
| `swivel_error` | (T,) rad | 4번이 계산한 팔 회전각 차이 | 참고 |
| `status` | (T,) | 0 성공 / 1 미수렴(C는 근사해) / 2 목표 없음·해 없음. **5-2는 쓰지 않음** (성공률은 공통 기준으로 직접 판정) | 참고 |
| `saturated` | (T, 7) bool | A: 마지막 반복에서 잘린 관절, B·C: 한계에서 1e-3 rad 이내. **5-2는 쓰지 않음** | 참고 |
| `iters` (A·B), `n_candidates`, `q3_pick` (C) | (T,) | 반복 수 / 후보 해 수, 고른 3번 관절 값 | 참고 |

### `info`와 좌표 변환

```python
from retarget.common import load_info, to_robot_point, to_table_point
info = load_info("A_DLS/T1_E004.npz")
p_robot = to_robot_point(p_table, info)    # 테이블 좌표 → 로봇 베이스:  scale·(p − src_table) @ R_table_to_baseᵀ + dst_base
p_table = to_table_point(p_robot, info)    # 역변환
```

- 주요 키: `scale`, `R_table_to_base`, `src_table`, `dst_base`, `hand`, `anchor`, `mount`, `base_z`, `r_off`, `exo_fps`, `human_arm_m`, `table_z_in_base`, 빈 값 처리 개수(`n_orient_held`, `n_wrist_manual`, `n_elbow_backfilled`).
- 같은 에피소드는 세 알고리즘의 `info`가 같다 (같은 설정으로 돌렸을 때). 5-2가 이것으로 설정 불일치를 잡는다.

---

## 시험 결과 (T1_E004, 2026-11-05)

컵 잡기, 왼손, 408프레임 중 손목 있는 293프레임. 설정은 `config.json`. 서버 `-n 1 -c 1`.

| | 손목 위치 오차 95% | 팔 회전각 차이 중앙 / 95% | 2번 관절 한계 1e-3 rad 이내 | IK 시간 중앙 |
|---|---|---|---|---|
| A | 0.1 mm | 32° / 46° | 25% | 8.6 ms |
| B | 2.0 mm | 26° / 42° | 91% | 46 ms |
| C | 1e-12 m 수준 | 26° / 42° | 67% | 38 ms |

- 사람 팔꿈치를 입력에서 빼면 팔 회전각 차이가 A 151°, B 123°, C 117°로 커진다 (로봇이 사람과 거의 반대쪽으로 팔꿈치를 듦).
- 팔 길이 비율 없이(1:1) 넣으면 A에서 목표 있는 프레임의 36%를 못 따라간다 (4번 관절 한계).
- 남는 26~32°는 iiwa 2번 관절 한계(±120°) 때문에 위팔을 사람처럼 늘어뜨리지 못해서 생긴다.
- C 사전 점검(`eaik_check.py`): 3번 관절 고정 시 구조 인식 `6R-SPHERICAL_SECOND_TWO_PARALLEL`, 왕복 1000/1000, 오차 2.5e-13 m, IK 1회 0.016 ms.

## 미정

| 항목 | 지금 | 누구 |
|---|---|---|
| 팔 길이 비율, 로봇 자리 | `arm`, `table` (10/16 "전처리 없음"과 다름, 1:1이면 36% 못 따라감) | 멘토 |
| 사람 팔꿈치 입력 | A·B·C 모두 입력 (팀 결정 T10, 10/16 "알고리즘별 방식 그대로"와 다름) | 멘토에 알림 |
| C의 3번 관절 처리 | 상수 고정이 아니라 프레임마다 격자 탐색 (10/16 "고정하지 말 것") | 멘토 확인 |
| 왼손 손목 축 | x축 둘레 180° 뒤집기 | 3번과 영상으로 확인 |
| C 쌍둥이 해 | 3번 +180°, 4·6번 부호 반대인 같은 팔 모양 해 중 어느 쪽을 고를지 규칙 없음 | 4번 |

## 변경 기록

| 날짜 | 버전 | 내용 |
|---|---|---|
| 2026-11-05 | 0.1 | A·B·C 첫 구현, 공통 목표·npz 형식, 시험 에피소드 확인 |
