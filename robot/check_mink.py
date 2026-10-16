"""mink(알고리즘 B)가 robot.xml에서 동작하는지 확인.

도달 가능한 무작위 목표(무작위 관절각의 FK)를 만들고, 매번 home 자세에서 출발해 link7을 목표에 맞춘다.
매 프레임 직전 해에서 출발하는 실제 리타게팅보다 불리한 조건이다.

실행: python robot/check_mink.py [목표 개수]
필요: mujoco, mink, qpsolvers[daqp]
"""
import sys
from pathlib import Path

import mink
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DT, MAX_ITER, TOL_P, TOL_R = 0.01, 2000, 1e-4, 1e-3

m = mujoco.MjModel.from_xml_path(str(HERE / "robot.xml"))
d = mujoco.MjData(m)
lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
home = m.key("home").qpos.copy()


def fk(q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return d.body("link7").xmat.reshape(3, 3).copy(), d.body("link7").xpos.copy()


def main():
    cfg = mink.Configuration(m)
    task = mink.FrameTask("link7", "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-6)
    limits = [mink.ConfigurationLimit(m)]

    rng = np.random.default_rng(0)
    iters, p_errs, r_errs, fails, viol = [], [], [], [], 0
    for _ in range(N):
        Rt, pt = fk(rng.uniform(lo, hi))
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(Rt), pt))
        cfg.update(home.copy())
        for it in range(MAX_ITER):
            v = mink.solve_ik(cfg, [task], DT, solver="daqp", damping=1e-3, limits=limits)
            cfg.integrate_inplace(v, DT)
            e = task.compute_error(cfg)
            if np.linalg.norm(e[:3]) < TOL_P and np.linalg.norm(e[3:]) < TOL_R:
                break

        e, q = task.compute_error(cfg), cfg.q
        p_err, r_err = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            viol += 1
        if p_err < TOL_P and r_err < TOL_R:
            iters.append(it + 1)
            p_errs.append(p_err)
            r_errs.append(r_err)
        else:
            at_lim = int(np.sum((q < lo + 1e-3) | (q > hi - 1e-3)))
            fails.append((p_err, r_err, at_lim))

    print(f"목표 {N}개 | 수렴 {len(iters)}/{N} | 관절 한계 위반 {viol}")
    if iters:
        print(f"수렴 시 반복 횟수 중앙값 {int(np.median(iters))}, 최대 {max(iters)} | "
              f"최대 위치 오차 {max(p_errs):.1e} m, 최대 방향 오차 {max(r_errs):.1e} rad")
    for p_err, r_err, at_lim in fails:
        print(f"  실패: 위치 오차 {p_err:.3f} m, 방향 오차 {r_err:.3f} rad, 한계에 붙은 관절 {at_lim}개")


if __name__ == "__main__":
    main()
