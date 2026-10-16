"""sew_ik.py 왕복 검사: 무작위 관절각 -> FK(link7 자세, 팔꿈치·손목 위치 -> SEW 각) -> SEW IK -> 해를 다시 FK.

실행: python retarget/C_ikgeo/check_sew_ik.py [자세 개수]
필요: mujoco, numpy
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sew_ik import IiwaSEW, ry, rz  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

m = mujoco.MjModel.from_xml_path(str(HERE.parents[1] / "robot" / "robot.xml"))
d = mujoco.MjData(m)
lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]


def fk(q):
    """link7 회전·위치, 팔꿈치(관절 4)·손목(관절 6) 위치."""
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    return (d.body("link7").xmat.reshape(3, 3).copy(), d.body("link7").xpos.copy(),
            d.xanchor[3].copy(), d.xanchor[5].copy())


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ik = IiwaSEW.from_mujoco(m)
    print(f"치수: 어깨 {ik.S.round(4)}, 어깨-팔꿈치 {ik.L1:.4f} m, 팔꿈치-손목 {ik.L2:.4f} m, 손목-플랜지 {ik.d7:.4f} m")

    q = np.random.default_rng(5).uniform(lo, hi)
    Rf = rz(q[0]) @ ry(q[1]) @ rz(q[2]) @ ry(-q[3]) @ rz(q[4]) @ ry(q[5]) @ rz(q[6])
    print(f"회전 공식 vs MuJoCo 최대 차이: {np.abs(fk(q)[0] - Rf).max():.1e}")

    rng = np.random.default_rng(0)
    worst = worst_psi = worst_orig = 0.0
    fails, n_sols, n_in_lim, min_ref = 0, [], [], 1.0
    for _ in range(N):
        q = rng.uniform(lo, hi)
        Rt, pt, E, W = fk(q)
        psi = ik.psi_from_points(W, E)
        min_ref = min(min_ref, ik._frame(W)[4])
        sols = ik.ik(Rt, pt, psi)
        if not sols:
            fails += 1
            continue
        best, k = np.inf, 0
        for s in sols:
            R, p, E2, W2 = fk(s)
            worst = max(worst, np.linalg.norm(p - pt) + np.linalg.norm(R - Rt))
            worst_psi = max(worst_psi, abs(wrap(ik.psi_from_points(W2, E2) - psi)))
            best = min(best, np.abs(wrap(s - q)).max())
            k += bool(np.all((s >= lo) & (s <= hi)))
        n_sols.append(len(sols))
        n_in_lim.append(k)
        worst_orig = max(worst_orig, best)

    print(f"자세 {N}개 | 풀이 실패 {fails} | 자세당 해 {min(n_sols)}~{max(n_sols)}개 | "
          f"그중 관절 한계 안 평균 {np.mean(n_in_lim):.2f}개 (최소 {min(n_in_lim)})")
    print(f"해를 FK에 다시 넣었을 때 최대 오차 {worst:.2e} | 해의 SEW 각 오차 최대 {worst_psi:.2e} rad | "
          f"원래 관절각과 가장 가까운 해 차이 최대 {worst_orig:.2e} rad")
    print(f"기준 벡터 특이점 근접도 최소 {min_ref:.3f} (1 = 멀다, 0 = 특이점)")
    ok = fails == 0 and worst < 1e-9 and worst_psi < 1e-9 and worst_orig < 1e-9
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
