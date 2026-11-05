"""알고리즘 C(EAIK) 사전 점검. 구현 전에 한 번 돌려 결과를 robot/model_checks.md 에 옮긴다.

1. 구조 인식: 3번 관절(인덱스 2)을 고정했을 때 EAIK가 iiwa14를 풀 수 있는 구조로 보는지
2. TCP 보정값: EAIK 끝 프레임(7번 관절 위치, 영 자세 방향)과 우리 TCP(iiwa_link_ee_kuka) 사이 고정 변환
3. 왕복 검사: 무작위 관절각 -> pinocchio FK -> EAIK IK(3번을 정답 값으로 고정) -> 정답이 해 중에 있는지, 해의 FK 오차
4. 속도: 로봇 만들기(3번 값마다 하나) / IK 한 번 시간, 3번 관절 1° 격자 전체 만드는 시간

실행 (서버, source ~/use_robot.sh 후):
  sbatch -p p02 --job-name retarget_hyojeong -c 1 --mem=4G -o ~/out/hyojeong/retarget/eaik_check.txt \
    --wrap "python ~/out/hyojeong/eaik_check.py"
"""
import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from eaik.IK_URDF import UrdfRobot  # noqa: E402
from eaik.IK_HP import HPRobot  # noqa: E402

TCP_FRAME = "iiwa_link_ee_kuka"
LOCK = 2   # 3번 관절 (0부터 셈)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def as_full_q(sol, lock_value):
    """EAIK 해 -> 관절 7개. 고정 관절이 빠진 6개로 오면 끼워 넣는다."""
    s = np.asarray(sol, dtype=float).ravel()
    if len(s) == 7:
        return s
    if len(s) == 6:
        return np.insert(s, LOCK, lock_value)
    raise ValueError(f"해 길이가 {len(s)}")


def pose_err(T_a, T_b):
    return (float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])),
            float(Rot.from_matrix(T_a[:3, :3].T @ T_b[:3, :3]).magnitude()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="~/project/robot/robot.urdf")
    ap.add_argument("--n", type=int, default=1000, help="왕복 검사 자세 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    urdf = str(Path(a.urdf).expanduser())
    rng = np.random.default_rng(a.seed)

    # pinocchio (기준 FK)
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    fid = model.getFrameId(TCP_FRAME)
    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()

    def fk_pin(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        return data.oMf[fid].homogeneous.copy()

    # ---- 1. 구조 인식
    print("== 1. 구조 인식 (3번 관절 = 0 고정)")
    t0 = time.perf_counter()
    bot = UrdfRobot(urdf, [(LOCK, 0.0)])
    print(f"  URDF로 로봇 만들기 {1000 * (time.perf_counter() - t0):.1f} ms")
    print(f"  hasKnownDecomposition: {bot.hasKnownDecomposition()}")
    print(f"  hasSphericalWrist:     {bot.hasSphericalWrist()}")
    print(f"  getKinematicFamily:    {bot.getKinematicFamily()}")
    H, P = np.asarray(bot.getOriginal_H()), np.asarray(bot.getOriginal_P())
    if H.shape[0] == 3 and H.shape[1] != 3:
        H, P = H.T, P.T
    print(f"  H {H.shape}, P {P.shape}")
    print("  관절 축 H:\n" + "\n".join(f"    {i + 1}: {np.round(h, 4).tolist()}" for i, h in enumerate(H)))
    print("  축 사이 거리 P:\n" + "\n".join(f"    {i}: {np.round(p, 4).tolist()}" for i, p in enumerate(P)))

    # ---- 2. TCP 보정값: T_tcp = T_eaik @ T_off 가 모든 자세에서 같은지
    print("\n== 2. TCP 보정값 (EAIK 끝 프레임 -> iiwa_link_ee_kuka)")
    offs = []
    for _ in range(200):
        q = rng.uniform(lo, hi)
        q[LOCK] = 0.0
        T_e = np.asarray(bot.fwdKin(q))
        offs.append(np.linalg.inv(T_e) @ fk_pin(q))
    offs = np.array(offs)
    T_off = offs[0]
    dev_p = max(np.linalg.norm(o[:3, 3] - T_off[:3, 3]) for o in offs)
    dev_r = max(Rot.from_matrix(T_off[:3, :3].T @ o[:3, :3]).magnitude() for o in offs)
    print(f"  T_off 위치 {np.round(T_off[:3, 3], 6).tolist()} m")
    print(f"  T_off 회전 (회전벡터 rad) {np.round(Rot.from_matrix(T_off[:3, :3]).as_rotvec(), 6).tolist()}")
    print(f"  200자세에서 흔들림: 위치 {dev_p:.2e} m, 회전 {dev_r:.2e} rad  (1e-9 근처면 고정 변환)")
    T_off_inv = np.linalg.inv(T_off)

    # ---- 3. 왕복 검사
    print(f"\n== 3. 왕복 검사 ({a.n}자세, 3번 관절을 정답 값으로 고정)")
    build_ms, ik_ms, n_sol = [], [], []
    found, ls_only, none = 0, 0, 0
    worst_p, worst_r = 0.0, 0.0
    for _ in range(a.n):
        q = rng.uniform(lo, hi)
        T_tcp = fk_pin(q)
        t0 = time.perf_counter()
        r = HPRobot(H, P, [(LOCK, float(q[LOCK]))])
        build_ms.append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        sol = r.IK(T_tcp @ T_off_inv)
        ik_ms.append(1000 * (time.perf_counter() - t0))
        Q = [as_full_q(s, q[LOCK]) for s in sol.Q]
        exact = [qq for qq, ls in zip(Q, sol.is_LS) if not ls]
        n_sol.append(len(exact))
        if not Q:
            none += 1
            continue
        if not exact:
            ls_only += 1
            continue
        if min(np.linalg.norm(wrap(qq - q)) for qq in exact) < 1e-6:
            found += 1
        for qq in exact:
            ep, er = pose_err(fk_pin(qq), T_tcp)
            worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
    print(f"  정답 관절각을 해 중에서 찾음: {found}/{a.n}")
    print(f"  근사해(LS)만 나옴: {ls_only}, 해 없음: {none}")
    print(f"  정확한 해 개수: 평균 {np.mean(n_sol):.1f}, 최소 {min(n_sol)}, 최대 {max(n_sol)}")
    print(f"  정확한 해의 FK 오차 최대: 위치 {worst_p:.2e} m, 방향 {worst_r:.2e} rad  (기준 1e-6 이하)")

    # ---- 4. 속도
    print("\n== 4. 속도 (CPU 1개)")
    print(f"  HPRobot 만들기: 중앙 {np.median(build_ms):.3f} ms")
    print(f"  IK 한 번:       중앙 {np.median(ik_ms):.3f} ms")
    grid = np.radians(np.arange(-170, 171, 1.0))
    t0 = time.perf_counter()
    robots = [HPRobot(H, P, [(LOCK, float(v))]) for v in grid]
    print(f"  3번 관절 1° 격자 {len(robots)}개 만들기: {1000 * (time.perf_counter() - t0):.0f} ms (에피소드마다 한 번이면 됨)")
    print(f"  -> 프레임당 격자 전체 IK 예상: {len(grid) * np.median(ik_ms):.1f} ms")


if __name__ == "__main__":
    main()
