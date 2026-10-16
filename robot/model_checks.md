# iiwa14 모델 사전 검사 (그리퍼·TCP 추가 전)

검사일 2026-10-16. 노트북 임시 환경에서 실행: Python 3.10, mujoco 3.16.0, pin 4.1.0.
그리퍼와 TCP를 붙이기 전이라 팔 끝(플랜지·`link7`) 기준이다. TCP를 붙인 뒤 TCP 기준으로 다시 검사한다.

## 1. URDF ↔ MJCF 일치 (`check_robot.py`): 통과

- 관절 한계: 7개 모두 차이 0 (`joint_limits.csv`). 원본 MJCF는 소수 5자리로 반올림돼 있어 URDF 값으로 맞췄다 (`SOURCES.md`)
- FK: 플랜지(`attachment_site` ↔ `iiwa_link_ee_kuka`), 무작위 자세 200개, 최대 위치 차이 1.2e-15 m, 최대 방향 차이 2.4e-15 rad (기준 1e-6 이하)
- 렌더링: 영 자세와 관절별 +0.8 rad 자세 PNG 8장. 모든 링크가 표시되고, 각 관절이 예상한 축으로 돈다

## 2. IK-Geo 적용 가능성 (역할 가이드 1단계의 팔 선택 조건): 가능, 4번 담당과 최종 확인 필요

- iiwa14는 관절 1·2·3이 어깨에서, 관절 5·6·7이 손목에서 한 점에 만나는 구조다. 링크 치수(영 자세, z축): 베이스→어깨 0.36 m, 어깨→팔꿈치 0.42 m, 팔꿈치→손목 0.40 m, 손목→플랜지 0.081 m
- 2026-10-16 왕복 검사(무작위 관절각 → FK → IK → FK, 자세 1000개)로 두 방식 모두 풀이 실패 0, 오차 1e-11 수준을 확인했다
  - 관절 3 고정 (`ik-geo` 파이썬 패키지 1.0.3, `spherical_two_intersecting`). 이 패키지는 회전행렬을 전치 형태로 주고받는다
  - SEW 각 (stereo-sew의 iiwa 14 R820 경우를 numpy로 옮겨 시험)
- 검사에 쓴 코드는 알고리즘 구현 범위(4번 역할)라 저장소에 남기지 않았다. C의 패키지·SEW 각 규칙은 4번 가이드 5단계대로 4번 담당이 멘토와 정한다

## 서버에서 다시 실행

```bash
cd ~/project && git pull
source ~/use_robot.sh
python robot/check_robot.py --out ~/out/본인이름/robot_check   # 렌더링은 GPU 노드에서 MUJOCO_GL=egl, --render 추가
```
