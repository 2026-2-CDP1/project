"""IK-Geo(알고리즘 C)가 iiwa14에서 풀리는지 확인하는 왕복 검사.

무작위 관절각 -> FK로 손끝 자세 -> IK-Geo로 역산 -> 해를 다시 FK에 넣어 확인.
ik-geo 파이썬 패키지는 6자유도 전용이라 관절 3을 고정한다 (IK-Geo 분류표의 iiwa "q3 고정" 방식).
관절 1·2가 어깨에서, 관절 5·6·7이 손목에서 한 점에 만나므로 spherical_two_intersecting을 쓴다.

실행: python robot/check_ikgeo.py [자세 개수]
필요: mujoco, ik-geo
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from ik_geo import Robot

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SIX = [0, 1, 3, 4, 5, 6]  # 관절 3(인덱스 2)을 뺀 6개

m = mujoco.MjModel.from_xml_path(str(HERE / "robot.xml"))
d = mujoco.MjData(m)
lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]


def fk(q):
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    return d.body("link7").xmat.reshape(3, 3).copy(), d.body("link7").xpos.copy()


def ikgeo_robot(q3):
    """관절 3을 q3로 고정한 6자유도 팔을 IK-Geo 형식(h: 관절 축, p: 관절 간 벡터)으로 만든다."""
    q = np.zeros(7)
    q[2] = q3
    R0, p_tool = fk(q)
    anc, ax = d.xanchor[:7].copy(), d.xaxis[:7].copy()
    shoulder, elbow, wrist = anc[1], anc[3], anc[5]
    pts = [shoulder, shoulder, elbow, wrist, wrist, wrist]
    p = [pts[0]] + [pts[i + 1] - pts[i] for i in range(5)] + [p_tool - wrist]
    h = [ax[i] for i in SIX]
    return Robot.spherical_two_intersecting([list(x) for x in h], [list(x) for x in p]), R0


def main():
    rng = np.random.default_rng(0)
    worst_sol = worst_orig = 0.0
    fails, n_ls, n_sols = 0, 0, []
    for _ in range(N):
        q = rng.uniform(lo, hi)
        Rt, pt = fk(q)
        robot, R0 = ikgeo_robot(q[2])
        # 주의: ik-geo 파이썬 패키지(1.0.3)는 회전행렬을 전치 형태로 주고받는다
        sols = robot.get_ik((Rt @ R0.T).T.tolist(), list(pt))

        best_orig, n_exact = np.inf, 0
        for s, is_ls in sols:
            if is_ls:  # 근사(최소제곱) 해는 제외
                n_ls += 1
                continue
            qq = q.copy()
            qq[SIX] = s
            R, p = fk(qq)
            worst_sol = max(worst_sol, np.linalg.norm(p - pt) + np.linalg.norm(R - Rt))
            dq = (qq - q + np.pi) % (2 * np.pi) - np.pi
            best_orig = min(best_orig, np.abs(dq).max())
            n_exact += 1
        n_sols.append(n_exact)
        if n_exact == 0:
            fails += 1
        else:
            worst_orig = max(worst_orig, best_orig)

    print(f"자세 {N}개 | 풀이 실패 {fails} | 자세당 정확해 평균 {np.mean(n_sols):.2f} "
          f"(최소 {min(n_sols)}, 최대 {max(n_sols)}) | 근사해 {n_ls}개 제외")
    print(f"정확해를 FK에 다시 넣었을 때 최대 오차: {worst_sol:.2e}")
    print(f"원래 관절각과 가장 가까운 해의 최대 차이: {worst_orig:.2e} rad")


if __name__ == "__main__":
    main()
