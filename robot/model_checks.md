# iiwa14 모델 사전 검사 (그리퍼·TCP 추가 전)

검사일 2026-10-16. 노트북 임시 환경에서 실행: Python 3.10, mujoco 3.16.0, pin 4.1.0, ik-geo 1.0.3, mink 1.3.0, qpsolvers 4.13.0, daqp 0.10.3.
그리퍼와 TCP를 붙이기 전이라 팔 끝(플랜지·`link7`) 기준이다. TCP를 붙인 뒤 TCP 기준으로 다시 검사한다.

## 1. URDF ↔ MJCF 일치 (`check_robot.py`): 통과

- 관절 한계: 7개 모두 차이 0 (`joint_limits.csv`). 원본 MJCF는 소수 5자리로 반올림돼 있어 URDF 값으로 맞췄다 (`SOURCES.md`)
- FK: 플랜지(`attachment_site` ↔ `iiwa_link_ee_kuka`), 무작위 자세 200개, 최대 위치 차이 1.2e-15 m, 최대 방향 차이 2.4e-15 rad (기준 1e-6 이하)
- 렌더링: 영 자세와 관절별 +0.8 rad 자세 PNG 8장. 모든 링크가 표시되고, 각 관절이 예상한 축으로 돈다

## 2. 알고리즘 C, IK-Geo 적용 가능성 (`check_ikgeo.py`): 적용 가능

- 방식: 관절 3 고정 → 6자유도 `spherical_two_intersecting` (관절 1·2 어깨 교점, 관절 5·6·7 손목 교점)
- 링크 치수 (영 자세, z축): 베이스→어깨 0.36 m, 어깨→팔꿈치 0.42 m, 팔꿈치→손목 0.40 m, 손목→플랜지 0.081 m
- 결과 (무작위 자세 1000개): 풀이 실패 0, 자세당 정확해 8개, 해를 FK에 다시 넣었을 때 최대 오차 5.0e-12, 원래 관절각이 항상 해에 포함됨
- 4번 담당 참고:
  - `ik-geo` 파이썬 패키지는 **회전행렬을 전치 형태로 주고받는다.** 그대로 넘기면 틀린 해가 나온다.
  - 6자유도 전용이라 관절 3을 고정해야 한다. SEW 각 방식(stereo-sew)은 시험하지 않았다.
  - 해 8개에는 관절 한계를 넘는 해가 섞일 수 있다. 해를 고르는 기준은 4번 담당이 정한다.

## 3. 알고리즘 B, mink 구동 (`check_mink.py`): 동작함

- 조건: 도달 가능한 무작위 목표 200개, 매번 `home` 자세에서 출발, FrameTask(link7), ConfigurationLimit, solver daqp, dt 0.01, 최대 2000회 반복
- 수렴 기준: 위치 오차 1e-4 m, 방향 오차 1e-3 rad 미만
- 결과: 수렴 180/200 (90%), 관절 한계 위반 0, 반복 횟수 중앙값 7
- 실패 20개는 모두 관절 1~4개가 한계에 붙은 채 멈춘 경우다. 모델 문제가 아니라 반복 방식이 관절 한계에 막혀 지역 최솟값에 빠진 것이다. 실제 리타게팅은 직전 프레임 해에서 출발하므로 이 시험보다 유리하다.

## 서버에서 다시 실행

```bash
cd ~/project && git pull
source ~/use_robot.sh
python robot/check_robot.py --out ~/out/본인이름/robot_check   # 렌더링은 GPU 노드에서 MUJOCO_GL=egl, --render 추가
python robot/check_ikgeo.py   # ik-geo 설치 필요
python robot/check_mink.py    # mink, qpsolvers[daqp] 설치 필요
```
