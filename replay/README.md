# replay — 궤적 재생 (5-1, 신서연)

관절 각도 궤적을 로봇 모델에 입혀, 프레임마다 TCP 위치·방향과 접촉 목록을 돌려준다.
5-2(지표 계산, 신서연), 6(RLDS 변환 검증, 차서현), 8(예측 궤적 확인, 정구현)이 이 함수를 쓴다.

> **함수 형태 고정 (v1, 2026-10-16).** 아래 입력·반환 형태는 바꾸지 않는다. 바꿔야 하면 쓰는 사람(5-2·6·8번)에게 먼저 알리고 이 문서의 변경 기록에 적는다.
> 내부 구현(모델 파일, 장애물 장면)은 형태를 유지한 채 바뀔 수 있다.

---

## 함수 형태

```python
from replay.replay import replay      # 레포 루트에서

out = replay(q_traj, gripper=None)
```

### 입력

| 인자 | 형태 | 단위 | 설명 |
|---|---|---|---|
| `q_traj` | `(T, 7)` float | rad | 팔 관절 각도. 열 순서 `joint1` ~ `joint7` (`robot/robot.xml`). NaN·inf 불가 |
| `gripper` | `(T,)` float 또는 `None` | 0~1 | **그리퍼를 쓰지 않아 항상 무시됨.** 함수 형태를 지키려고 인자만 남겨 둠 (형태만 검사) |

### 반환 (`dict`)

| 키 | 형태 | 단위 | 설명 |
|---|---|---|---|
| `tcp_pos` | `(T, 3)` | m | TCP(= 플랜지 `attachment_site`) 위치, **로봇 베이스 좌표** |
| `tcp_quat` | `(T, 4)` | — | TCP 방향, 쿼터니언 **(w, x, y, z)** |
| `contacts` | 길이 `T`의 리스트 | — | 프레임마다 `[(몸체1, 몸체2, 거리_m), ...]`. **거리가 음수면 두 몸체가 서로 파고든 것** |
| `meta` | dict | — | `model`(모델 경로), `tcp_site`(TCP로 쓴 site 이름), `robot_bodies`(로봇 몸체 이름 목록), `collision_exclude`(검사에서 뺀 몸체 쌍), `env_bodies`(환경 상자 이름, 비어 있으면 장면 없음), `env_exclude`(환경 검사에서 뺀 쌍) |

- `tcp_pos`, `tcp_quat`, `contacts`는 역할 가이드 5-1 1단계의 형태 그대로다. `meta`는 추가 정보라 안 써도 된다.
- 자기 충돌과 환경 충돌 구분: 두 몸체가 모두 `meta["robot_bodies"]`에 있으면 자기 접촉, 아니면 환경(테이블·장애물) 접촉. 도우미 함수 `is_self_contact(contact, robot_bodies)`

### 예외

- `q_traj`가 `(T, 7)`이 아니거나 NaN·inf가 있으면 `ValueError`
- `gripper`가 `(T,)`가 아니면 `ValueError`

---

## 재생 방식 (바꾸지 않는 것)

- **기구학 재생:** 프레임마다 `qpos`에 관절 각도를 그대로 넣고 `mujoco.mj_forward`로 위치·접촉만 계산. `mj_step`은 쓰지 않는다 (중력·제어기가 각도를 바꿈)
- 팔 이외 관절은 없다 (그리퍼 없음). 전체 `qpos`는 keyframe `home` 값에서 시작해 팔 7관절만 바꾼다
- 접촉은 MuJoCo가 계산한 것 그대로. 단 `robot.xml`의 `<exclude>` 목록 대신 **`COLLISION_EXCLUDE`(2쌍, `collision_exclude.md`)**로 바꿔 읽는다. 원본 `robot.xml`은 건드리지 않는다

---

## 현재 상태와 한계 (v1)

| 항목 | 지금 | 바뀔 때 |
|---|---|---|
| 모델 | `robot/robot.xml` (Menagerie iiwa14), 끝단 장치 없음 | — |
| TCP | **`attachment_site`(플랜지)** — 그리퍼를 쓰지 않아 플랜지를 TCP로 확정 (2026-11-03 팀 결정) | — |
| 그리퍼 | **쓰지 않음** → `gripper` 인자는 형태만 검사하고 무시 | — |
| 충돌 형상 | 링크마다 구(sphere) 여러 개 (메시 아님) | — |
| 충돌 제외 쌍 | **2쌍** (base–link1, link5–link7). 원본 7쌍 중 나머지 5쌍은 닿지 않아 검사에 넣음 — `collision_exclude.md` (초안, 멘토 확인 전) | 모델이 바뀌면 다시 검사 |
| 환경(테이블·장애물) | 기본 `replay()`는 없음. `Replayer(env_boxes=…, env_exclude=…)`로 에피소드마다 넣음 — `replay/scene.py`, 근거 `scene.md` | 테이블 크기 측정·4번 `to_robot_point`가 들어오면 실제 장면 |
| 좌표 | 로봇 `base`가 월드 원점 → MuJoCo 월드 좌표 = 로봇 베이스 좌표 | — |

---

## 점검

```bash
# 노트북 또는 서버 (source ~/use_robot.sh 후). 결과는 ~/project 밖에 저장
python replay/replay.py --check --out ~/out/본인이름/replay_check
```

모델 로드, 반환 형태, home 자세 접촉, 무작위 자세의 자기 침투(몸체 쌍별)를 `replay_check.txt`로 저장한다.

2026-10-16 노트북 실행 결과 (mujoco 최신, 무작위 2000 자세, seed 0):
- 반환 형태 정상, 쿼터니언 크기 오차 1e-15
- home 자세 TCP(플랜지) 위치 (0.669, 0.000, 0.285) m, 접촉 0
- 자기 침투가 있는 자세 10개 (0.5%): base–link6 8, base–link7 4, base–link5 3 → 손목이 베이스 쪽으로 접힌 실제 충돌 자세

---

## 환경 장면 (v1.3)

```python
from replay.replay import Replayer
from replay.scene import episode_env, load_obstacles, load_config
boxes, exclude = episode_env(load_obstacles("obstacles.json").get("T3_E031"), load_config(), to_robot_point)
r = Replayer(env_boxes=boxes, env_exclude=exclude)
out = r.replay(q_traj)      # 반환 형태는 위와 같음. 환경 접촉은 is_self_contact가 거짓인 것
```

```bash
python replay/scene.py --check --out ~/out/본인이름/scene_check     # 합성 검증
```

- 기본 `replay(q_traj)`는 그대로 로봇만 재생한다. 장면은 `Replayer`에 상자 목록을 넘길 때만 들어간다
- 만드는 법·결정 사항은 [`scene.md`](scene.md)

---

## 4번 결과와 정합 확인 (역할 가이드 5-1 4단계)

4번 npz의 손목 오차(`tcp_error`·`rot_error`, pinocchio + `robot.urdf`)와, 같은 관절 각도를 `replay()`(MuJoCo + `robot.xml`)로 재생해 다시 계산한 오차를 프레임마다 비교한다. 두 FK가 같으면 차이가 1e-6 이하여야 한다.

```bash
python replay/consistency.py --retarget <4번 결과 폴더> --out ~/out/신서연/consistency --limit 5   # 처음엔 알고리즘마다 5개
```

- 출력: `consistency.md`(알고리즘별 통과 수·차이 최대), `consistency.csv`(파일마다). 하나라도 실패하면 종료 코드 1
- 실패하면 관절 순서, 모델 파일, TCP 위치(플랜지)부터 의심한다
- 확인 (2026-11-06, 노트북 합성): 오차를 맞게 저장한 npz는 차이 0 → 통과, 6·7번 관절 순서를 바꾼 npz는 12.5 mm → 실패로 잡음

---

## 나란히 보기 영상 (v1.4)

사람 exo_front 영상 | 알고리즘별 로봇 재생을 가로로 붙인 mp4 (역할 가이드 5-1 5단계). worst 에피소드 확인(지표 정의서 5절)과 파지 프레임 확인(2.11)에 쓴다.

```bash
# 4번 결과가 있을 때 (서버 GPU 노드는 MUJOCO_GL=egl)
python replay/side_by_side.py --episode <3번 결과>/T1_E004 --retarget <4번 결과> \
    --out ~/out/신서연/videos/T1_E004.mp4 [--view both] [--png 100]
# 4번 결과 전 확인용: 실제 exo 영상 + 가짜 궤적
python replay/side_by_side.py --demo --video <exo_front data.mov> --task T3 --fps 59.97 --view both --out demo.mp4
```

- **로봇 카메라 = 사람 영상과 같은 시점:** exo_front 카메라(`sync/exo_calib_T{n}.json`)를 4번 `to_robot_point(p, info)`로 로봇 좌표에 옮긴다. 변환이 평행 이동·축척·회전이라 화면 구도가 같아진다
- `--view both`: 아래 줄에 옆 시점 (팔꿈치가 테이블 아래로 가는지 보기 좋음)
- 실패 프레임(정의서 2.1과 같은 판정: 위치 > 1 cm, 방향 > 5°, 한계 밖)은 빨간 테두리 + FAIL, 목표 없음은 NO TARGET, `grasp_frame`은 GRASP
- 30 Hz로 맞춤 (60 fps는 2프레임마다 하나). 장애물 상자와 테이블(윤곽 크기의 판, 그림용)을 같이 그림
- 속도: 13초 에피소드, 알고리즘 2개, 두 줄 화면이 노트북에서 약 10초
- 영상은 레포에 넣지 않는다 (`~/out/신서연/videos/`)

---

## 변경 기록

| 날짜 | 버전 | 변경 |
|---|---|---|
| 2026-10-16 | v1 | 함수 형태 고정, 기구학 재생 구현 (플랜지 TCP, 그리퍼 없음) |
| 2026-10-16 | v1.1 | 충돌 제외 쌍을 원본 7쌍 → 2쌍 (`collision_exclude.md`), `meta`에 `collision_exclude` 추가. 함수 형태는 그대로 |
| 2026-11-03 | v1.2 | 그리퍼를 쓰지 않기로 함 → TCP를 플랜지 `attachment_site`로 확정, 플랜지·그리퍼 경고 삭제. `gripper` 인자는 남기고 무시. 함수 형태는 그대로 |
| 2026-11-05 | v1.4 | 나란히 보기 영상 `replay/side_by_side.py`. 그림용 카메라를 넣으려고 `Replayer(extra_worldbody=…)` 인자 추가. `replay()` 함수 형태·기본 동작은 그대로 (무작위 2000 자세 점검 결과 같음) |
| 2026-11-05 | v1.3 | 환경 장면: `Replayer(env_boxes, env_exclude)` 추가, `meta`에 `env_bodies`·`env_exclude` 추가, `replay/scene.py`. `replay()` 함수 형태·기본 동작은 그대로 (무작위 2000 자세 점검 결과 같음) |
