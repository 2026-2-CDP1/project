"""B_mink: 제약 QP IK (mink 1.3.0, MuJoCo 모델 robot/robot.xml). 관절 한계는 QP 제약이라 넘지 않는다.

매 반복 QP:  min Σ cost_i ||J_i Δq + e_i||² + damping ||Δq||²   s.t. 관절 한계 (ConfigurationLimit)
목표(task):
  - tcp:     플랜지 attachment_site 위치·방향 (cost 1, 1)
  - elbow:   팔꿈치 link4 위치 (cost w_elbow, 방향 0). 목표 = common.elbow_goals (사람과 같은 팔 회전각이 되는 점)
  - posture: home 자세 쪽으로 약하게 (cost posture). 팔꿈치 목표가 없을 때 남는 자유도를 정하는 역할
A와 달리 손목과 팔꿈치를 우선순위 없이 가중치로 함께 푼다 (PROJECT_CONTEXT.md 1절).
- 프레임마다 직전 해에서 시작, 목표 없는 프레임은 직전 자세 유지 + status 2, 맨 앞 빈 프레임은 첫 해로 채움 (A와 같음)
- status: 0 = 마지막에 TCP 오차가 기준 이내, 1 = 아님
- saturated: 관절이 한계에서 sat_eps 이내 (B는 잘라내기가 없어서 이렇게 정의, metrics/definitions.md 10절 확인 필요)

시험 실행 (서버, source ~/use_robot.sh 후):
  python mink_ik.py --episode ~/out/seohyeon/episodes_test/T1_E004 --out ~/out/hyojeong/retarget/try --scale arm --anchor table
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 계산 시간 비교용: 단일 스레드 (numpy 등이 읽기 전에 설정해야 함)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import mink  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TCP_SITE = "attachment_site"
ELBOW_BODY, WRIST_BODY = "link4", "link6"   # 원점 = 팔꿈치 / 손목 중심
# posture·damping은 아주 작게: 가중치 합 방식이라 posture가 크면 TCP를 mm 단위로 끌어당겨 수렴 기준(0.1 mm)을 못 넘음.
# QP가 매 반복 최소점을 정확히 풀기 때문에, 다른 목표가 없는 남는 자유도에서는 작은 가중치로도 home 쪽으로 충분히 움직임
# w_elbow 0: 기존 알고리즘 그대로 손목만 (v1.2). v1.1은 0.1 (팔꿈치 입력, 0.5는 손목이 34 mm 밀려 0.1로)
DEFAULT = {"w_elbow": 0.0, "posture": 1e-6, "damping": 1e-8, "lm_damping": 1.0, "max_iter": 100,
           "tol_pos": 1e-4, "tol_rot": 1e-3, "tol_step": 1e-5, "sat_eps": 1e-3}


class MinkIK:
    def __init__(self, xml, w_elbow, posture, damping, lm_damping, max_iter, tol_pos, tol_rot, tol_step, sat_eps):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.cfg = mink.Configuration(self.model)
        self.tcp = mink.FrameTask(TCP_SITE, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=lm_damping)
        self.elbow = mink.FrameTask(ELBOW_BODY, "body", position_cost=w_elbow, orientation_cost=0.0)
        self.posture = mink.PostureTask(self.model, cost=posture)
        self.posture.set_target(common.Q_HOME)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.w_elbow, self.damping, self.max_iter = w_elbow, damping, max_iter
        self.tol_pos, self.tol_rot, self.tol_step, self.sat_eps = tol_pos, tol_rot, tol_step, sat_eps
        self.lo, self.hi = self.model.jnt_range[:7, 0].copy(), self.model.jnt_range[:7, 1].copy()
        self.sid = self.model.site(TCP_SITE).id
        self.eid, self.wid = self.model.body(ELBOW_BODY).id, self.model.body(WRIST_BODY).id

    def tcp_errors(self, p_des, R_des):
        d = self.cfg.data
        R = d.site_xmat[self.sid].reshape(3, 3)
        return float(np.linalg.norm(d.site_xpos[self.sid] - p_des)), float(Rot.from_matrix(R.T @ R_des).magnitude())

    def solve(self, q0, p_des, R_des, E_des=None, n_h=None):
        """한 프레임. 반환 (q, saturated(7,), status, 위치 오차, 방향 오차, 팔 회전각 오차, 반복 수)."""
        self.cfg.update(q0)
        self.tcp.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_des), p_des))
        tasks = [self.tcp, self.posture]
        if self.w_elbow > 0 and E_des is not None:
            self.elbow.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(), E_des))
            tasks.append(self.elbow)
        for it in range(self.max_iter + 1):
            pe, re = self.tcp_errors(p_des, R_des)
            if it == self.max_iter:
                break
            try:
                v = mink.solve_ik(self.cfg, tasks, 1.0, "daqp", damping=self.damping, limits=self.limits)
            except mink.NoSolutionFound:
                break             # QP 실패 -> 지금 자세에서 멈춤 (TCP 오차로 status 결정)
            self.cfg.integrate_inplace(v, 1.0)
            if pe < self.tol_pos and re < self.tol_rot and np.linalg.norm(v) < self.tol_step:
                break
        q = self.cfg.q[:7].copy()
        pe, re = self.tcp_errors(p_des, R_des)
        status = common.STATUS_OK if (pe < self.tol_pos and re < self.tol_rot) else common.STATUS_MAXITER
        sat = (q - self.lo < self.sat_eps) | (self.hi - q < self.sat_eps)
        d = self.cfg.data
        swv = (common.swivel_error(common.IIWA_SHOULDER, d.xpos[self.eid].copy(), d.xpos[self.wid].copy(), n_h)
               if n_h is not None else np.nan)
        return q, sat, status, pe, re, swv, it


def run_episode(solver, targets, q_start=common.Q_HOME, restart_tol=0.0):
    """restart_tol > 0 이면: 직전 자세에서 푼 손목 위치 오차가 restart_tol(m)보다 크면 home 에서 한 번 더 풀고
    더 나은 쪽(common.task_score)을 고른다 (A와 같은 규칙). 두 번 푼 시간 모두 time_ms 에 포함."""
    T = len(targets["valid"])
    E_goal = common.elbow_goals(targets)
    q_out = np.zeros((T, 7))
    sat_out = np.zeros((T, 7), bool)
    status = np.full(T, common.STATUS_NO_TARGET, np.int8)
    pos_err, rot_err, swv_err = (np.full(T, np.nan) for _ in range(3))
    iters = np.zeros(T, np.int32)
    time_ms = np.full(T, np.nan)   # 프레임당 IK 시간 (solve 호출만, 목표 없는 프레임은 NaN)
    elbow_used = np.zeros(T, bool)  # 사람 팔꿈치를 목표로 쓴 프레임
    n_prev = None                   # 직전 프레임 로봇 팔꿈치 방향 (어깨-손목 축 기준)
    restarted = np.zeros(T, bool)   # home 에서 다시 푼 결과를 쓴 프레임
    q = np.asarray(q_start, float).copy()
    for t in range(T):
        if targets["valid"][t]:
            n_h = targets["elbow_dir"][t]
            n_h = n_h if np.isfinite(n_h).all() else None
            E = E_goal[t] if np.isfinite(E_goal[t]).all() else None
            if E is not None:
                elbow_used[t] = solver.w_elbow > 0
            elif n_prev is not None and solver.w_elbow > 0:
                # 사람 팔꿈치가 안 보이면 직전 프레임 로봇 팔꿈치 방향을 유지하는 점을 목표로 (A·C의 '직전 자세 유지'와 맞춤).
                # 목표가 없으면 남는 자유도가 home 쪽으로 당겨져 팔꿈치가 튀었다 돌아옴
                W_des = targets["pos"][t] - common.WRIST_TO_FLANGE * targets["rot"][t][:, 2]
                E = common.robot_elbow_target(common.IIWA_SHOULDER, W_des, n_prev)
            t0 = time.perf_counter()
            res = solver.solve(q, targets["pos"][t], targets["rot"][t], E, n_h)
            if restart_tol > 0 and not res[3] <= restart_tol:
                res2 = solver.solve(np.asarray(q_start, float), targets["pos"][t], targets["rot"][t], E, n_h)
                if common.task_score(res2[3], res2[4]) < common.task_score(res[3], res[4]):
                    res, restarted[t] = res2, True
            time_ms[t] = 1000 * (time.perf_counter() - t0)
            q, sat_out[t], status[t], pos_err[t], rot_err[t], swv_err[t], iters[t] = res
            solver.cfg.update(q)            # 다시 푼 경우에도 고른 자세 기준으로 팔꿈치 방향을 잼
            d = solver.cfg.data
            n_prev = common.elbow_direction(common.IIWA_SHOULDER, d.xpos[solver.eid].copy(), d.xpos[solver.wid].copy())
        q_out[t] = q
    solved = np.flatnonzero(status != common.STATUS_NO_TARGET)
    if len(solved):
        q_out[:solved[0]] = q_out[solved[0]]
    return q_out, sat_out, status, pos_err, rot_err, swv_err, iters, time_ms, elbow_used, restarted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="3번 episodes/<ID>/ 폴더")
    ap.add_argument("--out", required=True, help="결과 루트 (~/project 밖). <out>/B_mink/<ID>.npz 로 저장")
    ap.add_argument("--xml", default="~/project/robot/robot.xml")
    ap.add_argument("--scale", default="arm", help='"arm"(iiwa 팔 길이 / 사람 팔 길이, 기본) 또는 1.0 같은 숫자')
    ap.add_argument("--r-off", default="palm", choices=sorted(common.R_OFF_OPTIONS))
    ap.add_argument("--anchor", default="table", choices=common.ANCHORS)
    ap.add_argument("--mount", default="up", choices=common.MOUNTS, help="up: 세워 둠 / side: 옆으로 눕힘")
    ap.add_argument("--base-z", type=float, default=0.0, help="베이스를 위(+)·아래(-)로 옮길 거리 m")
    ap.add_argument("--pos-only", default="hold", choices=common.POS_ONLY_MODES,
                    help="손목 위치만 있고 방향이 없는 프레임: hold=직전 방향 유지 / skip=목표 없음")
    ap.add_argument("--restart-tol", type=float, default=0.0,
                    help="손목 위치 오차가 이 값(m)보다 크면 home 에서 다시 풀어 나은 쪽 선택 (0 = 끔)")
    ap.add_argument("--brief", action="store_true", help="결과를 한 줄로만 출력")
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    ep = common.load_episode(a.episode)
    scale = a.scale if a.scale == "arm" else float(a.scale)
    targets = common.make_targets(ep, scale=scale, r_off=a.r_off, anchor=a.anchor, mount=a.mount, base_z=a.base_z,
                                  pos_only=a.pos_only)
    solver = MinkIK(Path(a.xml).expanduser(), a.w_elbow, a.posture, a.damping, a.lm_damping, a.max_iter,
                    a.tol_pos, a.tol_rot, a.tol_step, a.sat_eps)

    t0 = time.time()
    q, sat, status, pos_err, rot_err, swv_err, iters, time_ms, elbow_used, restarted = run_episode(
        solver, targets, restart_tol=a.restart_tol)
    sec = time.time() - t0

    eid = ep["meta"]["episode_id"]
    path = common.save_result(Path(a.out) / "B_mink" / f"{eid}.npz", q, sat, status, pos_err, targets,
                              rot_error=rot_err, swivel_error=swv_err, iters=iters, time_ms=time_ms,
                              elbow_used=elbow_used, restarted=restarted)
    info = targets["info"]
    if a.brief:
        label = f"{eid} B w={a.w_elbow:<4} it={a.max_iter:<4} rs={a.restart_tol:<5}"
        print(common.brief_line(label, q, sat, status, pos_err, rot_err, swv_err, info) + " | " + common.timing_text(time_ms)
              + f" | 위치최대 {1000 * np.nanmax(pos_err):.0f} mm, 1cm초과 {int(np.nansum(pos_err > 0.01))} | 재시작 {int(restarted.sum())}")
        return
    print(f"{eid} hand={info['hand']} scale={info['scale']:.3f} r_off={a.r_off} anchor={a.anchor} w_elbow={a.w_elbow}")
    print(common.summarize("B_mink", q, sat, status, pos_err, rot_err, sec))
    s = np.degrees(swv_err[np.isfinite(swv_err)])
    if len(s):
        print(f"  팔 회전각 오차 deg  중앙 {np.median(s):.1f}  95% {np.percentile(s, 95):.1f}  최대 {s.max():.1f}  ({len(s)}프레임)")
    print(f"  반복 수 중앙 {np.median(iters[status != 2]):.0f}  최대 {iters.max()}")
    print(f"  IK 시간 {common.timing_text(time_ms)}")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
