# replay — 궤적 재생 (5-1, 신서연)

관절 각도 궤적을 로봇 모델에 입혀, 프레임마다 TCP 위치·방향과 접촉 목록을 돌려준다.
5-2(지표 계산, 신서연), 6(RLDS 변환 검증, 차서현), 8(예측 궤적 확인, 정구현)이 이 함수를 쓴다.

> **함수 형태 고정 (v1, 2026-10-16).** 아래 입력·반환 형태는 바꾸지 않는다. 바꿔야 하면 쓰는 사람(5-2·6·8번)에게 먼저 알리고 이 문서의 변경 기록에 적는다.
> 내부 구현(모델 파일, TCP, 그리퍼, 장애물 장면)은 형태를 유지한 채 바뀔 수 있다.

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
| `gripper` | `(T,)` float 또는 `None` | 0~1 | 그리퍼 벌림 (0 닫힘, 1 열림). 그리퍼가 붙기 전에는 무시됨 |

### 반환 (`dict`)

| 키 | 형태 | 단위 | 설명 |
|---|---|---|---|
| `tcp_pos` | `(T, 3)` | m | TCP 위치, **로봇 베이스 좌표** |
| `tcp_quat` | `(T, 4)` | — | TCP 방향, 쿼터니언 **(w, x, y, z)** |
| `contacts` | 길이 `T`의 리스트 | — | 프레임마다 `[(몸체1, 몸체2, 거리_m), ...]`. **거리가 음수면 두 몸체가 서로 파고든 것** |
| `meta` | dict | — | `model`(모델 경로), `tcp_site`(TCP로 쓴 site 이름), `robot_bodies`(로봇 몸체 이름 목록) |

- `tcp_pos`, `tcp_quat`, `contacts`는 역할 가이드 5-1 1단계의 형태 그대로다. `meta`는 추가 정보라 안 써도 된다.
- 자기 충돌과 환경 충돌 구분: 두 몸체가 모두 `meta["robot_bodies"]`에 있으면 자기 접촉, 아니면 환경(테이블·장애물) 접촉. 도우미 함수 `is_self_contact(contact, robot_bodies)`

### 예외

- `q_traj`가 `(T, 7)`이 아니거나 NaN·inf가 있으면 `ValueError`
- `gripper`가 `(T,)`가 아니면 `ValueError`

---

## 재생 방식 (바꾸지 않는 것)

- **기구학 재생:** 프레임마다 `qpos`에 관절 각도를 그대로 넣고 `mujoco.mj_forward`로 위치·접촉만 계산. `mj_step`은 쓰지 않는다 (중력·제어기가 각도를 바꿈)
- 팔 이외 관절(그리퍼 등)은 keyframe `home` 값으로 둔다
- 접촉은 MuJoCo가 `robot.xml`의 `<contact><exclude>` 쌍을 뺀 상태로 계산한 것 그대로

---

## 현재 상태와 한계 (v1)

| 항목 | 지금 | 바뀔 때 |
|---|---|---|
| 모델 | `robot/robot.xml` (Menagerie iiwa14) | 1번이 그리퍼·TCP를 붙이면 같은 파일 |
| TCP | `tcp` site가 없어서 **`attachment_site`(플랜지)** 사용, 경고 출력 → **비교용 값으로 쓰지 말 것** | 1번이 `tcp` site를 추가하면 자동으로 그것을 씀 |
| 그리퍼 | 없음 → `gripper` 무시, 경고 출력 | 1번 `robot/gripper.csv`가 생기면 0~1을 관절값으로 옮겨 넣음 |
| 충돌 형상 | 링크마다 구(sphere) 여러 개 (메시 아님) | — |
| 충돌 제외 쌍 | 원본 7쌍 (base–link1/2/3, link1–link3, link3–link5, link4–link7, link5–link7) | 검토 후 `replay/collision_exclude.md`로 확정 (결과 보기 전) |
| 환경(테이블·장애물) | 없음 → 접촉은 모두 자기 접촉 | 장애물 시나리오가 확정되면 장면 파일 추가 |
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

## 변경 기록

| 날짜 | 버전 | 변경 |
|---|---|---|
| 2026-10-16 | v1 | 함수 형태 고정, 기구학 재생 구현 (플랜지 TCP, 그리퍼 없음) |
