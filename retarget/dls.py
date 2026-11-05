"""A_DLS: 감쇠 최소제곱(DLS) IK, pinocchio. 관절 한계는 매 반복 뒤 잘라낸다(clip).

주 목표 (손목 = TCP 위치·방향, 6차원):
  dq1 = J^T (J J^T + lam^2 I)^-1 e      (pinocchio 공식 IK 예제와 같은 오차·야코비안 정의, 프레임 LOCAL)
보조 목표 (팔꿈치 = 팔 회전각, 남는 1자유도):
  N = I - J+ J  (주 목표를 건드리지 않는 방향만 남기는 투영)
  dq2 = N (Je N)+ (w_elbow * (E_des - E))
  E_des: 로봇 어깨-손목 축 둘레에서 사람 팔꿈치와 같은 방향을 향하는 팔꿈치 위치 (common.robot_elbow_target)
- 프레임마다 직전 프레임 해에서 시작 (첫 프레임은 home). 맨 앞 빈 프레임은 첫 해로 채움
- 목표가 없는 프레임(NaN)은 직전 자세 유지 + status 2. 팔꿈치만 없는 프레임은 보조 목표 없이 풂
- status: 0 = 마지막에 손목 오차가 기준 이내, 1 = 아님(미수렴)
- saturated: 마지막 반복에서 한계로 잘린 관절

시험 실행 (서버, source ~/use_robot.sh 후, 한 줄씩):
  python dls.py --episode ~/out/seohyeon/episodes_test/T1_E004 --out ~/out/hyojeong/retarget/try --scale arm --r-off finger --w-elbow 0.5
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 계산 시간 비교용: 단일 스레드 (numpy 등이 읽기 전에 설정해야 함)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TCP_FRAME = "iiwa_link_ee_kuka"   # = MJCF attachment_site (플랜지)
ELBOW_FRAME = "iiwa_link_4"       # 원점 = 4번 관절 = 팔꿈치 (어깨에서 0.42 m)
WRIST_FRAME = "iiwa_link_6"       # 원점 = 6번 관절 = 손목 중심 (팔꿈치에서 0.40 m)
DEFAULT = {"lam": 0.05, "step": 1.0, "max_iter": 100, "tol_pos": 1e-4, "tol_rot": 1e-3,
           "w_elbow": 0.5, "tol_elbow_step": 1e-4}   # w_elbow 0.5: 팀 결정, A·B·C 모두 사람 팔꿈치를 입력으로 (0이면 끔)


class DLS:
    def __init__(self, urdf, lam, step, max_iter, tol_pos, tol_rot, w_elbow, tol_elbow_step):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(TCP_FRAME)
        self.fid_e = self.model.getFrameId(ELBOW_FRAME)
        self.fid_w = self.model.getFrameId(WRIST_FRAME)
        self.lo, self.hi = self.model.lowerPositionLimit.copy(), self.model.upperPositionLimit.copy()
        self.lam, self.step, self.max_iter = lam, step, max_iter
        self.tol_pos, self.tol_rot = tol_pos, tol_rot
        self.w_elbow, self.tol_elbow_step = w_elbow, tol_elbow_step

    def fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.fid]

    def points(self):
        """(어깨, 팔꿈치, 손목 중심) 현재 위치. fk() 뒤에 부른다."""
        return common.IIWA_SHOULDER, self.data.oMf[self.fid_e].translation.copy(), self.data.oMf[self.fid_w].translation.copy()

    def solve(self, q0, p_des, R_des, n_h=None):
        """한 프레임. 반환 (q, saturated(7,), status, 위치 오차 m, 방향 오차 rad, 팔 회전각 오차 rad, 반복 수)."""
        oMdes = pin.SE3(R_des, p_des)
        q = q0.copy()
        sat = np.zeros(7, bool)
        use_elbow = self.w_elbow > 0 and n_h is not None
        lam2_6, lam2_3, I7 = self.lam ** 2 * np.eye(6), self.lam ** 2 * np.eye(3), np.eye(7)
        for it in range(self.max_iter + 1):
            iMd = self.fk(q).actInv(oMdes)
            err = pin.log6(iMd).vector                      # TCP 기준 (위치 3, 회전 3)
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid)   # LOCAL
            J = -pin.Jlog6(iMd.inverse()) @ J
            Jp = J.T @ np.linalg.inv(J @ J.T + lam2_6)       # 감쇠 의사역행렬
            v = -Jp @ err
            v2 = np.zeros(7)
            # 팔꿈치는 반복 앞 절반에서만. 뒤 절반은 주 목표(손목)만 풀어 손목 정확도를 지킨다
            if use_elbow and it < self.max_iter // 2:
                S, E, W = self.points()
                E_des = common.robot_elbow_target(S, W, n_h)
                if E_des is not None:
                    Je = pin.computeFrameJacobian(self.model, self.data, q, self.fid_e,
                                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
                    N = I7 - Jp @ J
                    JeN = Je @ N
                    v2 = N @ (JeN.T @ np.linalg.solve(JeN @ JeN.T + lam2_3, self.w_elbow * (E_des - E)))
            done = (np.linalg.norm(err[:3]) < self.tol_pos and np.linalg.norm(err[3:]) < self.tol_rot
                    and np.linalg.norm(v2) < self.tol_elbow_step)
            if done or it == self.max_iter:
                break
            q_new = pin.integrate(self.model, q, self.step * (v + v2))
            if v2.any() and ((q_new < self.lo) | (q_new > self.hi)).any():
                q_new = pin.integrate(self.model, q, self.step * v)   # 팔꿈치 목표가 관절을 한계 밖으로 밀면 이번엔 손목만
            q = q_new
            sat = (q < self.lo) | (q > self.hi)
            q = np.clip(q, self.lo, self.hi)
        M = self.fk(q)
        pos_err = float(np.linalg.norm(M.translation - p_des))
        rot_err = float(np.linalg.norm(pin.log3(M.rotation.T @ R_des)))
        status = common.STATUS_OK if (pos_err < self.tol_pos and rot_err < self.tol_rot) else common.STATUS_MAXITER
        S, E, W = self.points()
        swv_err = common.swivel_error(S, E, W, n_h) if n_h is not None else np.nan
        return q, sat, status, pos_err, rot_err, swv_err, it


def run_episode(solver, targets, q_start=common.Q_HOME):
    T = len(targets["valid"])
    q_out = np.zeros((T, 7))
    sat_out = np.zeros((T, 7), bool)
    status = np.full(T, common.STATUS_NO_TARGET, np.int8)
    pos_err, rot_err, swv_err = (np.full(T, np.nan) for _ in range(3))
    iters = np.zeros(T, np.int32)
    time_ms = np.full(T, np.nan)   # 프레임당 IK 시간 (solve 호출만, 목표 없는 프레임은 NaN)
    elbow_used = np.zeros(T, bool)  # 사람 팔꿈치를 목표로 쓴 프레임
    q = np.asarray(q_start, float).copy()
    for t in range(T):
        if targets["valid"][t]:
            n_h = targets["elbow_dir"][t]
            n_h = n_h if np.isfinite(n_h).all() else None   # 팔꿈치 안 보이면 보조 목표 없이 -> 직전 자세에서 이어 감
            elbow_used[t] = n_h is not None and solver.w_elbow > 0
            t0 = time.perf_counter()
            q, sat_out[t], status[t], pos_err[t], rot_err[t], swv_err[t], iters[t] = solver.solve(
                q, targets["pos"][t], targets["rot"][t], n_h)
            time_ms[t] = 1000 * (time.perf_counter() - t0)
        q_out[t] = q          # 목표 없음 -> 직전 자세 유지
    solved = np.flatnonzero(status != common.STATUS_NO_TARGET)
    if len(solved):           # 맨 앞 빈 프레임은 home 대신 첫 해로 채움 (home -> 첫 해 튐 방지, status는 2 그대로)
        q_out[:solved[0]] = q_out[solved[0]]
    return q_out, sat_out, status, pos_err, rot_err, swv_err, iters, time_ms, elbow_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="3번 episodes/<ID>/ 폴더")
    ap.add_argument("--out", required=True, help="결과 루트 (~/project 밖). <out>/A_DLS/<ID>.npz 로 저장")
    ap.add_argument("--urdf", default="~/project/robot/robot.urdf")
    ap.add_argument("--scale", default="arm", help='"arm"(iiwa 팔 길이 / 사람 팔 길이, 기본) 또는 1.0 같은 숫자')
    ap.add_argument("--r-off", default="palm", choices=sorted(common.R_OFF_OPTIONS))
    ap.add_argument("--anchor", default="table", choices=common.ANCHORS)
    ap.add_argument("--mount", default="up", choices=common.MOUNTS, help="up: 세워 둠 / side: 옆으로 눕힘")
    ap.add_argument("--base-z", type=float, default=0.0, help="베이스를 위(+)·아래(-)로 옮길 거리 m")
    ap.add_argument("--pos-only", default="hold", choices=common.POS_ONLY_MODES,
                    help="손목 위치만 있고 방향이 없는 프레임: hold=직전 방향 유지 / skip=목표 없음")
    ap.add_argument("--brief", action="store_true", help="결과를 한 줄로만 출력")
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    ep = common.load_episode(a.episode)
    scale = a.scale if a.scale == "arm" else float(a.scale)
    targets = common.make_targets(ep, scale=scale, r_off=a.r_off, anchor=a.anchor, mount=a.mount, base_z=a.base_z,
                                  pos_only=a.pos_only)
    solver = DLS(Path(a.urdf).expanduser(), a.lam, a.step, a.max_iter, a.tol_pos, a.tol_rot,
                 a.w_elbow, a.tol_elbow_step)

    t0 = time.time()
    q, sat, status, pos_err, rot_err, swv_err, iters, time_ms, elbow_used = run_episode(solver, targets)
    sec = time.time() - t0

    eid = ep["meta"]["episode_id"]
    path = common.save_result(Path(a.out) / "A_DLS" / f"{eid}.npz", q, sat, status, pos_err, targets,
                              rot_error=rot_err, swivel_error=swv_err, iters=iters, time_ms=time_ms,
                              elbow_used=elbow_used)
    info = targets["info"]
    s = np.degrees(swv_err[np.isfinite(swv_err)])
    if a.brief:
        label = f"A {a.mount:4s} {a.anchor:8s} z{a.base_z:+.1f} {a.r_off:6s} w={a.w_elbow:<4}"
        print(common.brief_line(label, q, sat, status, pos_err, rot_err, swv_err, info) + " | " + common.timing_text(time_ms))
        return
    print(f"{eid} hand={info['hand']} scale={info['scale']:.3f} (사람 팔 {info['human_arm_m']:.3f} m) "
          f"r_off={a.r_off} anchor={a.anchor} w_elbow={a.w_elbow}")
    print(common.summarize("A_DLS", q, sat, status, pos_err, rot_err, sec))
    if len(s):
        print(f"  팔 회전각 오차 deg  중앙 {np.median(s):.1f}  95% {np.percentile(s, 95):.1f}  최대 {s.max():.1f}  ({len(s)}프레임)")
    print(f"  반복 수 중앙 {np.median(iters[status != 2]):.0f}  최대 {iters.max()}")
    print(f"  IK 시간 {common.timing_text(time_ms)}")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
