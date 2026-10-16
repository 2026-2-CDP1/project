# robot/ 파일 출처

로봇: KUKA LBR iiwa 14 R820 (7자유도, 그리퍼 미부착 상태)

| 파일 | 출처 | 변경 사항 |
|---|---|---|
| `robot.xml` (MJCF) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `kuka_iiwa_14/iiwa14.xml` @ `62ab00f` | 없음 (원본 그대로) |
| `assets/*.obj` | 위와 같은 폴더의 `assets/` | 없음 |
| `robot.urdf` | [RobotLocomotion/models](https://github.com/RobotLocomotion/models) `iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf` @ `f85076b` | visual 메시 경로를 `package://.../visual/*.gltf` → `assets/*.obj`로 변경 (같은 메시의 OBJ 변환본) |

- Menagerie README에 따르면 `iiwa14.xml`은 위 Drake URDF를 MuJoCo로 변환해 만든 것이다. 두 파일은 출처가 같다.
- 라이선스: Menagerie는 BSD-3-Clause(`LICENSE.menagerie`), Drake 모델은 `LICENSE.drake`.

## 관절 이름 대응

| MJCF | URDF |
|---|---|
| `joint1` ~ `joint7` | `iiwa_joint_1` ~ `iiwa_joint_7` |
| body `link7` | link `iiwa_link_7` |
| site `attachment_site` (link7 + 0.045 m, 플랜지) | link `iiwa_link_ee_kuka` (위치·방향 같음. `iiwa_link_ee`는 위치만 같고 y축으로 -90° 돌아가 있음) |

## 1차 확인 (그리퍼·TCP 추가 전)

- 관절 한계: 7개 모두 두 파일 일치 (±2.96706 / ±2.0944 / ±3.05433 rad, 제조사 사양 ±170° / ±120° / ±175°)
- FK: 관절 한계 내 무작위 자세 200개(seed 0), link7 기준 최대 위치 차이 1.1e-15 m, 최대 방향 차이 2.3e-15 rad → 일치
- 영 자세에서 link7 원점 높이 1.261 m
- 확인 환경: mujoco 3.16.0, pinocchio 4.1.0
