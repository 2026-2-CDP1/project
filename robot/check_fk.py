"""robot.xml(MJCF)과 robot.urdf가 같은 로봇인지 확인: 관절 한계 비교 + FK 일치 검사.

실행: python robot/check_fk.py [자세 개수]
필요: mujoco, pin
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

MJ_JOINTS = [f"joint{i}" for i in range(1, 8)]
PIN_JOINTS = [f"iiwa_joint_{i}" for i in range(1, 8)]
MJ_FRAME, PIN_FRAME = "link7", "iiwa_link_7"  # TCP 추가 전이라 팔 끝(link7) 기준

mj = mujoco.MjModel.from_xml_path(str(HERE / "robot.xml"))
md = mujoco.MjData(mj)
pm = pin.buildModelFromUrdf(str(HERE / "robot.urdf"))
pdata = pm.createData()

print("관절 한계 (rad)")
for a, b in zip(MJ_JOINTS, PIN_JOINTS):
    j = mj.joint(a).id
    iq = pm.joints[pm.getJointId(b)].idx_q
    print(f"  {a:7s} MJCF {mj.jnt_range[j]} (limited={bool(mj.jnt_limited[j])}) | "
          f"URDF [{pm.lowerPositionLimit[iq]:.5f} {pm.upperPositionLimit[iq]:.5f}]")

fid = pm.getFrameId(PIN_FRAME)
mj_adr = [mj.jnt_qposadr[mj.joint(n).id] for n in MJ_JOINTS]
pin_adr = [pm.joints[pm.getJointId(n)].idx_q for n in PIN_JOINTS]
lo = np.array([pm.lowerPositionLimit[i] for i in pin_adr])
hi = np.array([pm.upperPositionLimit[i] for i in pin_adr])

rng = np.random.default_rng(0)
worst_p = worst_r = 0.0
for _ in range(N):
    q7 = rng.uniform(lo, hi)

    md.qpos[:] = 0
    md.qpos[mj_adr] = q7
    mujoco.mj_forward(mj, md)
    p1 = md.body(MJ_FRAME).xpos.copy()
    R1 = md.body(MJ_FRAME).xmat.reshape(3, 3).copy()

    q = pin.neutral(pm)
    q[pin_adr] = q7
    pin.forwardKinematics(pm, pdata, q)
    pin.updateFramePlacements(pm, pdata)
    p2 = pdata.oMf[fid].translation
    R2 = pdata.oMf[fid].rotation

    worst_p = max(worst_p, np.linalg.norm(p1 - p2))
    worst_r = max(worst_r, np.linalg.norm(pin.log3(R1.T @ R2)))

print(f"FK 일치 검사 (자세 {N}개, seed 0)")
print(f"  최대 위치 차이: {worst_p:.2e} m | 최대 방향 차이: {worst_r:.2e} rad")
print("  판정:", "통과 (1e-6 이하)" if max(worst_p, worst_r) <= 1e-6 else "실패")
