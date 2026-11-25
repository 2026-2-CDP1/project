"""C_eaik: 닫힌 해 IK (EAIK 1.2.2, robot/robot.urdf). 반복 없이 기하 공식으로 해를 전부 구한 뒤 하나를 고른다.

EAIK는 7관절 팔을 관절 하나(3번)를 정해진 값에 고정해 6관절처럼 푼다 (fixed_axes). 그래서
  1) 3번 관절 값을 격자(기본 1°)로 바꿔 가며 EAIK로 해를 전부 구하고 (값마다 해 최대 8개)
  2) 관절 한계 밖 해는 버리고
  3) 남은 해 중 하나를 고른다
       select=human:      팔꿈치 방향이 사람과 가장 비슷한 해 (사람 SEW 각이 C 의 원래 입력)
                          사람 팔꿈치가 안 보이는 프레임은 continuity 로
       select=continuity: 직전 프레임 자세와 가장 가까운 해 (각도 2π 감기 고려)
  4) 고른 3번 값 근처를 촘촘히(기본 0.1°) 다시 풀어 다듬는다
select=sew (멘토 11/6 제안: 사람 SEW 각으로 고정 → 격자 탐색 없이 해석해 → 직전 프레임과 가장 가까운 해):
  1) 사람 팔꿈치 방향으로 로봇 팔꿈치 점 E 를 정함 (어깨–손목 축 둘레 원 위, common.robot_elbow_target 과 같음)
  2) E 를 만드는 1·2번 관절 (q1, q2) 두 쌍 → 손목 중심이 팔 평면에 오도록 하는 3번 관절 값을 FK로 바로 계산 (쌍마다 q3, q3+π)
  3) 그 3번 값으로 EAIK 해 → 관절 한계 안 + 팔꿈치가 E 인 해만 → 직전 프레임과 가장 가까운 해
  4) 한계 안 해가 없으면 팔 회전각을 사람 값에서 ±1°씩 벌려 가며 다시 (한계 안에서 사람과 가장 가까운 팔 회전각)
  5) 그래도 없거나(손목이 안 닿음 등) 사람 팔꿈치가 안 보이는 프레임은 격자 방식(직전 자세와 가장 가까운 해)으로
select=smooth (11/25, sew·continuity 의 해 바뀜 튐 대책, C 저크 RMS 최대 7,401):
  sew·continuity 는 프레임마다 "직전 자세와 가장 가까운 해"를 고르는 탐욕 방식이라, 쓰던 해가 관절 한계·도달 범위 밖으로
  사라지는 순간 다른 갈래로 한 번에 넘어감 (관절 튐 → 저크 폭증). smooth 는 에피소드 전체를 보고 해 경로를 고른다.
  1) 프레임마다 후보: 3번 관절 성긴 격자(dp_grid_deg, 기본 5°) × EAIK 해 (한계 안, 정확한 해 우선)
  2) 후보 점수(unary) = w_swivel × (사람과 팔 회전각 차이)² + w_limit × (관절 한계 10° 안으로 들어간 정도)²
     → 팔 자세는 사람을 따르고, 한계에 붙어 있다가 떨어져 나가는 해는 미리 피함
  3) 이웃 프레임 비용(pairwise) = Σ (관절 속도 / 속도 한계)² + w_over × Σ (속도 한계 넘은 만큼)²
  4) Viterbi(DP)로 (unary + pairwise) 합이 가장 작은 해 경로. 갈래를 바꿔야만 하면 두 갈래가 가장 가까운 프레임에서,
     앞으로 올 프레임까지 보고 미리 바꿈 (탐욕 방식은 못 함)
  5) 고른 경로에서 3번 관절(여유 자유도) 값을 시간축으로 가우시안 스무딩(smooth_sigma 프레임) → 그 값으로 다시 풀어
     같은 갈래 해(고른 해와 가장 가까운 것)로 바꿈. 손목은 그대로 정확, 격자 계단(5°)이 관절 궤적에 안 남음
  목표 없음·해 없음 프레임은 지금처럼 직전 자세 유지. DP 는 그 공백을 건너 (공백 길이만큼 긴 Δt 로) 잇는다.
- TCP 보정: EAIK 끝 프레임 = 7번 관절 위치(우리 TCP보다 플랜지 축으로 0.045 m 안쪽), 회전 차이 0 (eaik_check.py 로 확인)
- status: 0 = 정확한 해, 1 = 근사해(LS)만 있음, 2 = 목표 없음 또는 한계 안 해 없음(직전 자세 유지)
- saturated: 관절이 한계에서 sat_eps 이내 (5-2와 정할 공통 규칙 제안안)
- 프레임마다 직전 해에서 시작(첫 프레임 기준은 home), 맨 앞 빈 프레임은 첫 해로 채움 (A·B와 같음)

시험 실행 (서버, source ~/use_robot.sh 후):
  python eaik_ik.py --episode ~/out/seohyeon/episodes_test/T1_E004 --out ~/out/hyojeong/retarget/try --scale arm --anchor table
  python eaik_ik.py --episode ... --out ... --select smooth --brief     # 새 해 선택 규칙
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from eaik.IK_HP import HPRobot  # noqa: E402
from eaik.IK_URDF import UrdfRobot  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TCP_FRAME, ELBOW_FRAME, WRIST_FRAME = "iiwa_link_ee_kuka", "iiwa_link_4", "iiwa_link_6"
LOCK = 2             # 3번 관절 (0부터 셈)
TIE = 1e-3           # human 모드에서 팔꿈치 차이가 거의 같을 때 직전 자세와 가까운 쪽을 고르는 가중치 (rad/rad)
SELECTS = ("human", "continuity", "sew", "smooth")
SEW_STEP = np.radians(1.0)   # sew 모드: 사람 팔 회전각으로 한계 안 해가 없을 때 벗어나 보는 간격
ELBOW_TOL = 1e-5             # sew 모드: 해의 팔꿈치가 정한 팔꿈치 점과 같다고 보는 거리 (m)
LIM_MARGIN = np.radians(10.0)  # smooth: 관절 한계에서 이 안쪽으로 들어가면 unary 벌점
REFINE_MAX = np.radians(20.0)  # smooth: 스무딩한 3번 값으로 다시 푼 해가 DP 해에서 이보다 멀면(다른 갈래) 버림
DEFAULT = {"grid_deg": 1.0, "refine_deg": 0.1, "sat_eps": 1e-3,
           # smooth 전용 (다른 select 에서는 안 씀)
           "dp_grid_deg": 5.0,     # 후보용 3번 관절 격자
           "w_swivel": 8.0,        # 팔 회전각 차이 (rad²) 가중치. 20° 차이 ≈ 한 관절이 속도 한계로 움직이는 한 프레임 비용
           "w_limit": 1.0,         # 관절 한계 근접 벌점 가중치
           "w_over": 100.0,        # 속도 한계 초과 벌점 가중치
           "smooth_sigma": 2.0}    # 3번 관절 스무딩 (프레임). 0 이면 안 함


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class EaikIK:
    def __init__(self, urdf, select, grid_deg, refine_deg, sat_eps, dp_grid_deg=DEFAULT["dp_grid_deg"],
                 w_swivel=DEFAULT["w_swivel"], w_limit=DEFAULT["w_limit"], w_over=DEFAULT["w_over"],
                 smooth_sigma=DEFAULT["smooth_sigma"]):
        urdf = str(urdf)
        base = UrdfRobot(urdf, [(LOCK, 0.0)])
        if not base.hasKnownDecomposition():
            raise RuntimeError("EAIK가 3번 관절 고정 iiwa14를 풀 수 있는 구조로 보지 않음")
        H, P = np.asarray(base.getOriginal_H()), np.asarray(base.getOriginal_P())
        if H.shape[0] == 3 and H.shape[1] != 3:
            H, P = H.T, P.T
        self.H, self.P = H, P

        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(TCP_FRAME)
        self.fid_e = self.model.getFrameId(ELBOW_FRAME)
        self.fid_w = self.model.getFrameId(WRIST_FRAME)
        self.lo, self.hi = self.model.lowerPositionLimit.copy(), self.model.upperPositionLimit.copy()
        self.select, self.refine, self.sat_eps = select, np.radians(refine_deg), sat_eps
        self.step = np.radians(grid_deg)
        self.vmax = self.model.velocityLimit.copy()          # robot.urdf <limit velocity> (rad/s)
        self.w_swivel, self.w_limit, self.w_over, self.sigma = w_swivel, w_limit, w_over, smooth_sigma

        # TCP 보정: T_tcp = T_eaik @ T_off  ->  EAIK 목표 = T_tcp @ inv(T_off). 여러 자세에서 같은지 확인
        rng = np.random.default_rng(0)
        offs = []
        for _ in range(5):
            q = rng.uniform(self.lo, self.hi)
            q[LOCK] = 0.0
            offs.append(np.linalg.inv(np.asarray(base.fwdKin(q))) @ self.fk(q)[0])
        if max(np.abs(o - offs[0]).max() for o in offs) > 1e-9:
            raise RuntimeError("EAIK 끝 프레임과 TCP 사이 변환이 자세마다 다름")
        self.T_off_inv = np.linalg.inv(offs[0])

        # 팔꿈치 위치는 1·2번 관절로만 정해짐 (3번은 위팔 축 둘레 회전이라 팔꿈치 점을 안 옮김). 공식이 pinocchio와 같은지 확인
        qs = rng.uniform(self.lo, self.hi, size=(20, 7))
        diff = max(np.linalg.norm(self.elbow_points(q[None])[0] - self.fk(q)[1]) for q in qs)
        if diff > 1e-9:
            raise RuntimeError(f"팔꿈치 위치 공식이 pinocchio와 다름 ({diff:.2e} m)")

        # 3번 관절 격자마다 EAIK 로봇을 미리 만들어 둔다 (하나에 약 0.02 ms)
        self.grid = np.arange(self.lo[LOCK], self.hi[LOCK] + 1e-12, self.step)
        self.robots = [HPRobot(H, P, [(LOCK, float(v))]) for v in self.grid]
        if select == "smooth":                               # DP 후보용 성긴 격자
            self.dp_grid = np.arange(self.lo[LOCK], self.hi[LOCK] + 1e-12, np.radians(dp_grid_deg))
            self.dp_robots = [HPRobot(H, P, [(LOCK, float(v))]) for v in self.dp_grid]

    def fk(self, q):
        """(TCP 4x4, 팔꿈치, 손목 중심)."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.fid].homogeneous.copy(),
                self.data.oMf[self.fid_e].translation.copy(), self.data.oMf[self.fid_w].translation.copy())

    @staticmethod
    def elbow_points(Q):
        """(N,7) -> (N,3) 팔꿈치 위치. 어깨 + 0.42 × (1·2번 관절로 돌린 z축)."""
        q1, q2 = Q[:, 0], Q[:, 1]
        u = np.stack([np.cos(q1) * np.sin(q2), np.sin(q1) * np.sin(q2), np.cos(q2)], axis=1)
        return common.IIWA_SHOULDER + common.L_UPPER * u

    def candidates(self, robot, q3, T_e):
        """한 3번 값에서 EAIK 해 -> 관절 한계 안 해들 [(q, is_LS)]."""
        sol = robot.IK(T_e)
        out = []
        for s, ls in zip(sol.Q, sol.is_LS):
            s = np.asarray(s, dtype=float).ravel()
            q = s if len(s) == 7 else np.insert(s, LOCK, q3)
            q = wrap(q)
            if np.all(q >= self.lo - 1e-9) and np.all(q <= self.hi + 1e-9):
                out.append((np.clip(q, self.lo, self.hi), bool(ls)))
        return out

    def scores(self, Q, q_prev, n_h, W):
        """작을수록 좋음. human: 팔꿈치 방향 차이(rad) + TIE × 직전 자세와 거리. continuity: 직전 자세와 거리."""
        dist = np.linalg.norm(wrap(Q - q_prev), axis=1)
        if self.select != "human" or n_h is None:
            return dist
        S = common.IIWA_SHOULDER
        u = (W - S) / np.linalg.norm(W - S)
        n_hp = n_h - np.dot(n_h, u) * u
        nn = np.linalg.norm(n_hp)
        if nn < 1e-6:
            return dist
        n_hp /= nn
        v = self.elbow_points(Q) - S
        p = v - (v @ u)[:, None] * u
        pn = np.linalg.norm(p, axis=1)
        ang = np.where(pn > 1e-6, np.arccos(np.clip((p @ n_hp) / np.maximum(pn, 1e-12), -1, 1)), np.pi)
        return ang + TIE * dist

    def pick(self, cand, q_prev, n_h, W):
        """후보 [(q, is_LS, q3)] 중 정확한 해를 먼저, 그중 점수가 가장 낮은 것."""
        exact = [c for c in cand if not c[1]]
        pool = exact if exact else cand
        Q = np.array([c[0] for c in pool])
        i = int(np.argmin(self.scores(Q, q_prev, n_h, W)))
        return pool[i], bool(exact)

    def q3_for_elbow(self, q1, q2, E, W):
        """1·2번 관절이 (q1, q2)일 때 아래팔이 손목 중심 W 쪽을 향하게 하는 3번 관절 값 (q4 < 0 기준 하나, 다른 하나는 +π).
        3번 관절은 위팔 축 둘레 회전이므로, 위팔 축에 수직인 평면에서 아래팔 방향을 W−E 방향으로 돌리는 각."""
        e = (E - common.IIWA_SHOULDER) / common.L_UPPER
        perp = lambda v: v - np.dot(v, e) * e
        t = perp(W - E)
        if np.linalg.norm(t) < 1e-9:
            return None                                       # 팔이 일직선: 3번 값이 정해지지 않음
        q = np.array([q1, q2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])
        _, E0, W0 = self.fk(q)
        q[2] = np.pi / 2
        _, _, W90 = self.fk(q)
        p0, p90 = perp(W0 - E0), perp(W90 - E0)
        return float(np.arctan2(np.dot(t, p90) / np.linalg.norm(p90), np.dot(t, p0) / np.linalg.norm(p0)))

    def solve_sew(self, q_prev, T_e, W, n_h):
        """사람 팔 회전각으로 팔꿈치를 고정해 푼다. 반환 (q, 후보 수, 3번 값, 벗어난 각 rad) 또는 None."""
        S = common.IIWA_SHOULDER
        u = W - S
        d = np.linalg.norm(u)
        if not (abs(common.L_UPPER - common.L_FORE) < d < common.L_UPPER + common.L_FORE):
            return None                                       # 손목이 안 닿거나 너무 가까움 → 격자 방식(근사해)
        u /= d
        n = n_h - np.dot(n_h, u) * u
        if np.linalg.norm(n) < 1e-6:
            return None
        n /= np.linalg.norm(n)
        b = np.cross(u, n)
        for k in range(int(np.pi / SEW_STEP) + 1):
            for delta in ((0.0,) if k == 0 else (k * SEW_STEP, -k * SEW_STEP)):
                E = common.robot_elbow_target(S, W, np.cos(delta) * n + np.sin(delta) * b)
                if E is None:
                    return None
                e = (E - S) / common.L_UPPER
                q2 = float(np.arccos(np.clip(e[2], -1.0, 1.0)))
                q1 = float(np.arctan2(e[1], e[0]))
                pool = []
                for a1, a2 in ((q1, q2), (wrap(q1 + np.pi), -q2)):     # 같은 팔꿈치 점을 만드는 두 쌍
                    q3 = self.q3_for_elbow(a1, a2, E, W)
                    if q3 is None:
                        continue
                    for v in (q3, wrap(q3 + np.pi)):
                        if not (self.lo[LOCK] <= v <= self.hi[LOCK]):
                            continue
                        for q, ls in self.candidates(HPRobot(self.H, self.P, [(LOCK, float(v))]), v, T_e):
                            if not ls and np.linalg.norm(self.elbow_points(q[None])[0] - E) < ELBOW_TOL:
                                pool.append((q, v))
                if pool:
                    Q = np.array([p[0] for p in pool])
                    i = int(np.argmin(np.linalg.norm(wrap(Q - q_prev), axis=1)))   # 직전 프레임과 가장 가까운 해
                    return pool[i][0], len(pool), pool[i][1], delta
        return None

    def solve(self, q_prev, p_des, R_des, n_h=None):
        """한 프레임. 반환 (q, saturated(7,), status, 위치 오차, 방향 오차, 팔 회전각 오차, 후보 수, 고른 3번 값)."""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_des, p_des
        T_e = T @ self.T_off_inv
        W = p_des - common.WRIST_TO_FLANGE * R_des[:, 2]     # 손목 중심 목표 (모든 해가 같음)

        self.last_sew = np.nan                                # sew 로 푼 프레임: 사람 팔 회전각에서 벗어난 각 (rad), 아니면 NaN
        if self.select == "sew" and n_h is not None:
            r = self.solve_sew(q_prev, T_e, W, n_h)
            if r is not None:
                q, n_cand, q3, self.last_sew = r
                T_q, E, Wq = self.fk(q)
                pe = float(np.linalg.norm(T_q[:3, 3] - p_des))
                re = float(Rot.from_matrix(T_q[:3, :3].T @ R_des).magnitude())
                swv = common.swivel_error(common.IIWA_SHOULDER, E, Wq, n_h)
                return q, self.saturated(q), common.STATUS_OK, pe, re, swv, n_cand, float(q3)

        cand =[(q, ls, q3) for q3, r in zip(self.grid, self.robots) for q, ls in self.candidates(r, q3, T_e)]
        if not cand:
            return q_prev.copy(), self.saturated(q_prev), common.STATUS_NO_TARGET, np.nan, np.nan, np.nan, 0, np.nan
        best, exact = self.pick(cand, q_prev, n_h, W)
        n_cand = len(cand)

        # 고른 3번 값 근처를 촘촘히 다시 풀기
        if self.refine > 0:
            fine = best[2] + np.arange(-self.step, self.step + 1e-12, self.refine)
            fine = fine[(fine >= self.lo[LOCK]) & (fine <= self.hi[LOCK])]
            more = [(q, ls, q3) for q3 in fine for q, ls in self.candidates(HPRobot(self.H, self.P, [(LOCK, float(q3))]), q3, T_e)]
            best, exact = self.pick([best] + more, q_prev, n_h, W)

        q = best[0]
        T_q, E, Wq = self.fk(q)
        pe = float(np.linalg.norm(T_q[:3, 3] - p_des))
        re = float(Rot.from_matrix(T_q[:3, :3].T @ R_des).magnitude())
        status = common.STATUS_OK if exact else common.STATUS_MAXITER   # 1 = 근사해만
        swv = common.swivel_error(common.IIWA_SHOULDER, E, Wq, n_h) if n_h is not None else np.nan
        return q, self.saturated(q), status, pe, re, swv, n_cand, float(best[2])

    def swivel_angles(self, Q, n_h, W):
        """(K,7) 해들의 팔 회전각이 사람과 다른 각 (rad). 사람 팔꿈치가 없거나 팔이 일직선이면 0."""
        if n_h is None:
            return np.zeros(len(Q))
        S = common.IIWA_SHOULDER
        u = (W - S) / np.linalg.norm(W - S)
        n_hp = n_h - np.dot(n_h, u) * u
        nn = np.linalg.norm(n_hp)
        if nn < 1e-6:
            return np.zeros(len(Q))
        n_hp /= nn
        v = self.elbow_points(Q) - S
        p = v - (v @ u)[:, None] * u
        pn = np.linalg.norm(p, axis=1)
        return np.where(pn > 1e-6, np.arccos(np.clip((p @ n_hp) / np.maximum(pn, 1e-12), -1, 1)), 0.0)

    def frame_candidates(self, p_des, R_des, n_h):
        """smooth: 한 프레임 후보. 반환 dict(Q (K,7), q3 (K,), unary (K,), exact bool, T_e) 또는 None (해 없음)."""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_des, p_des
        T_e = T @ self.T_off_inv
        W = p_des - common.WRIST_TO_FLANGE * R_des[:, 2]
        cand = [(q, ls, q3) for q3, r in zip(self.dp_grid, self.dp_robots) for q, ls in self.candidates(r, q3, T_e)]
        if not cand:
            return None
        exact = any(not c[1] for c in cand)
        pool = [c for c in cand if not c[1]] if exact else cand
        Q = np.array([c[0] for c in pool])
        q3 = np.array([c[2] for c in pool])
        margin = np.minimum(Q - self.lo, self.hi - Q)
        lim = (np.maximum(0.0, 1.0 - margin / LIM_MARGIN) ** 2).sum(axis=1)
        unary = self.w_swivel * self.swivel_angles(Q, n_h, W) ** 2 + self.w_limit * lim
        return {"Q": Q, "q3": q3, "unary": unary, "exact": exact, "T_e": T_e, "n_all": len(cand)}

    def pair_cost(self, Qa, Qb, dt):
        """(Ka,7), (Kb,7) -> (Ka,Kb). 관절 속도를 속도 한계로 나눈 값의 제곱합 + 한계 초과 벌점."""
        v = np.abs(wrap(Qb[None, :, :] - Qa[:, None, :])) / (dt * self.vmax)
        return (v ** 2).sum(axis=2) + self.w_over * (np.maximum(v - 1.0, 0.0) ** 2).sum(axis=2)

    def viterbi(self, frames, cands, fps):
        """해가 있는 프레임들(frames, 오름차순)의 후보 중 전체 비용이 가장 작은 경로 -> 프레임마다 고른 후보 번호."""
        cost = cands[0]["unary"].copy()
        back = []
        for k in range(1, len(frames)):
            dt = (frames[k] - frames[k - 1]) / fps          # 목표 없는 공백을 건너면 그만큼 긴 Δt
            tot = cost[:, None] + self.pair_cost(cands[k - 1]["Q"], cands[k]["Q"], dt)
            idx = np.argmin(tot, axis=0)
            cost = tot[idx, np.arange(tot.shape[1])] + cands[k]["unary"]
            back.append(idx)
        path = [int(np.argmin(cost))]
        for idx in reversed(back):
            path.append(int(idx[path[-1]]))
        return path[::-1]

    def smooth_q3(self, frames, Q, q3, fps):
        """고른 경로의 3번 관절 값을 이어지는 구간(속도 한계 안)마다 가우시안 스무딩."""
        if self.sigma <= 0 or len(frames) < 3:
            return q3.copy()
        dt = np.diff(frames) / fps
        jump = (np.abs(wrap(np.diff(Q, axis=0))) / (dt[:, None] * self.vmax)).max(axis=1) > 1.0
        out = q3.copy()
        cuts = np.concatenate([[0], np.flatnonzero(jump) + 1, [len(q3)]])
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 3:
                out[a:b] = gaussian_filter1d(q3[a:b], self.sigma, mode="nearest")
        return np.clip(out, self.lo[LOCK], self.hi[LOCK])

    def resolve_at(self, q_ref, q3, T_e):
        """3번 값 q3 에서 다시 풀어 q_ref 와 같은 갈래(가장 가까운 정확한 해) -> q 또는 None."""
        sol = [q for q, ls in self.candidates(HPRobot(self.H, self.P, [(LOCK, float(q3))]), q3, T_e) if not ls]
        if not sol:
            return None
        Q = np.array(sol)
        d = np.linalg.norm(wrap(Q - q_ref), axis=1)
        i = int(np.argmin(d))
        return Q[i] if d[i] < REFINE_MAX else None

    def saturated(self, q):
        return (q - self.lo < self.sat_eps) | (self.hi - q < self.sat_eps)


def run_episode_smooth(solver, targets):
    """select=smooth: 에피소드 전체 후보 -> Viterbi -> 3번 관절 스무딩. 반환 형식은 run_episode 와 같음."""
    T = len(targets["valid"])
    fps = float(targets["info"].get("exo_fps") or 60.0)
    q_out = np.zeros((T, 7))
    sat_out = np.zeros((T, 7), bool)
    status = np.full(T, common.STATUS_NO_TARGET, np.int8)
    pos_err, rot_err, swv_err, q3_pick, time_ms = (np.full(T, np.nan) for _ in range(5))
    n_cand = np.zeros(T, np.int32)
    elbow_used = np.zeros(T, bool)

    frames, cands, nhs = [], [], []
    for t in np.flatnonzero(targets["valid"]):
        n_h = targets["elbow_dir"][t]
        n_h = n_h if np.isfinite(n_h).all() else None
        t0 = time.perf_counter()
        c = solver.frame_candidates(targets["pos"][t], targets["rot"][t], n_h)
        time_ms[t] = 1000 * (time.perf_counter() - t0)
        if c is None:
            continue
        frames.append(int(t)); cands.append(c); nhs.append(n_h)
        n_cand[t] = c["n_all"]
        elbow_used[t] = n_h is not None

    branch_switch = np.zeros(T, bool)   # 속도 한계를 넘는 프레임 사이 변화 (DP 로도 못 피한 갈래 바뀜)
    if frames:
        t0 = time.perf_counter()
        path = solver.viterbi(frames, cands, fps)
        Q = np.array([c["Q"][i] for c, i in zip(cands, path)])
        q3 = np.array([c["q3"][i] for c, i in zip(cands, path)])
        q3_s = solver.smooth_q3(np.array(frames), Q, q3, fps)
        for k, t in enumerate(frames):
            if abs(q3_s[k] - q3[k]) > 1e-9:
                r = solver.resolve_at(Q[k], q3_s[k], cands[k]["T_e"])
                if r is not None:
                    Q[k], q3[k] = r, q3_s[k]
        per = 1000 * (time.perf_counter() - t0) / len(frames)   # DP·스무딩 시간은 프레임에 똑같이 나눠 더함
        for k, t in enumerate(frames):
            q = Q[k]
            T_q, E, Wq = solver.fk(q)
            W = targets["pos"][t] - common.WRIST_TO_FLANGE * targets["rot"][t][:, 2]
            q_out[t], sat_out[t], q3_pick[t] = q, solver.saturated(q), q3[k]
            status[t] = common.STATUS_OK if cands[k]["exact"] else common.STATUS_MAXITER
            pos_err[t] = float(np.linalg.norm(T_q[:3, 3] - targets["pos"][t]))
            rot_err[t] = float(Rot.from_matrix(T_q[:3, :3].T @ targets["rot"][t]).magnitude())
            swv_err[t] = common.swivel_error(common.IIWA_SHOULDER, E, Wq, nhs[k]) if nhs[k] is not None else np.nan
            time_ms[t] += per
            if k:
                dt = (t - frames[k - 1]) / fps
                branch_switch[t] = (np.abs(wrap(Q[k] - Q[k - 1])) / (dt * solver.vmax)).max() > 1.0
        # 목표 없음·해 없음 -> 직전 자세 유지 (A·B·sew 와 같은 규칙), 맨 앞은 첫 해로
        for t in range(T):
            if status[t] == common.STATUS_NO_TARGET:
                q_out[t] = q_out[t - 1] if t else Q[0]
                sat_out[t] = solver.saturated(q_out[t])
    else:
        q_out[:] = common.Q_HOME
    solver.episode_sew_delta = np.full(T, np.nan)
    solver.episode_branch_switch = branch_switch
    return q_out, sat_out, status, pos_err, rot_err, swv_err, n_cand, q3_pick, time_ms, elbow_used


def run_episode(solver, targets, q_start=common.Q_HOME):
    if solver.select == "smooth":
        return run_episode_smooth(solver, targets)
    T = len(targets["valid"])
    q_out = np.zeros((T, 7))
    sat_out = np.zeros((T, 7), bool)
    status = np.full(T, common.STATUS_NO_TARGET, np.int8)
    pos_err, rot_err, swv_err, q3_pick, time_ms = (np.full(T, np.nan) for _ in range(5))
    n_cand = np.zeros(T, np.int32)
    elbow_used = np.zeros(T, bool)  # 사람 팔꿈치로 해를 고른 프레임 (아니면 직전 자세와 가장 가까운 해)
    sew_delta = np.full(T, np.nan)  # sew 모드로 푼 프레임의 팔 회전각 벗어남 (rad). NaN = 격자로 풂
    q = np.asarray(q_start, float).copy()
    for t in range(T):
        if targets["valid"][t]:
            n_h = targets["elbow_dir"][t]
            n_h = n_h if np.isfinite(n_h).all() else None
            elbow_used[t] = n_h is not None and solver.select in ("human", "sew")
            t0 = time.perf_counter()
            q, sat_out[t], status[t], pos_err[t], rot_err[t], swv_err[t], n_cand[t], q3_pick[t] = solver.solve(
                q, targets["pos"][t], targets["rot"][t], n_h)
            time_ms[t] = 1000 * (time.perf_counter() - t0)
            sew_delta[t] = getattr(solver, "last_sew", np.nan)
        q_out[t] = q          # 목표 없음 / 해 없음 -> 직전 자세 유지
    solved = np.flatnonzero(status != common.STATUS_NO_TARGET)
    if len(solved):
        q_out[:solved[0]] = q_out[solved[0]]
    solver.episode_sew_delta = sew_delta   # 반환 형식(run_all.py)을 바꾸지 않으려고 풀이기에 남김
    return q_out, sat_out, status, pos_err, rot_err, swv_err, n_cand, q3_pick, time_ms, elbow_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="3번 episodes/<ID>/ 폴더")
    ap.add_argument("--out", required=True, help="결과 루트 (~/project 밖). <out>/C_eaik/<ID>.npz 로 저장")
    ap.add_argument("--urdf", default="~/project/robot/robot.urdf")
    ap.add_argument("--scale", default="arm", help='"arm"(iiwa 팔 길이 / 사람 팔 길이, 기본) 또는 1.0 같은 숫자')
    ap.add_argument("--r-off", default="palm", choices=sorted(common.R_OFF_OPTIONS))
    ap.add_argument("--anchor", default="table", choices=common.ANCHORS)
    ap.add_argument("--mount", default="up", choices=common.MOUNTS)
    ap.add_argument("--base-z", type=float, default=0.0)
    ap.add_argument("--pos-only", default="hold", choices=common.POS_ONLY_MODES,
                    help="손목 위치만 있고 방향이 없는 프레임: hold=직전 방향 유지 / skip=목표 없음")
    ap.add_argument("--select", default="sew", choices=SELECTS, help="해 고르는 기준 (sew = v1.3 본 결과, human = 격자, smooth = 에피소드 전체 DP + 3번 관절 스무딩)")
    ap.add_argument("--brief", action="store_true", help="결과를 한 줄로만 출력")
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    ep = common.load_episode(a.episode)
    scale = a.scale if a.scale == "arm" else float(a.scale)
    targets = common.make_targets(ep, scale=scale, r_off=a.r_off, anchor=a.anchor, mount=a.mount, base_z=a.base_z,
                                  pos_only=a.pos_only)
    solver = EaikIK(Path(a.urdf).expanduser(), a.select, a.grid_deg, a.refine_deg, a.sat_eps)

    t0 = time.time()
    q, sat, status, pos_err, rot_err, swv_err, n_cand, q3_pick, time_ms, elbow_used = run_episode(solver, targets)
    sec = time.time() - t0

    eid = ep["meta"]["episode_id"]
    path = common.save_result(Path(a.out) / "C_eaik" / f"{eid}.npz", q, sat, status, pos_err, targets,
                              rot_error=rot_err, swivel_error=swv_err, n_candidates=n_cand, q3_pick=q3_pick,
                              time_ms=time_ms, elbow_used=elbow_used)
    info = targets["info"]
    if a.brief:
        label = f"C {a.mount:4s} {a.anchor:8s} z{a.base_z:+.1f} {a.r_off:6s} {a.select:10s}"
        print(common.brief_line(label, q, sat, status, pos_err, rot_err, swv_err, info) + " | " + common.timing_text(time_ms))
        if a.select == "sew":
            sd = solver.episode_sew_delta
            used, ok = np.isfinite(sd), np.isfinite(time_ms)
            dd = np.degrees(np.abs(sd[used]))
            print(f"    sew 로 푼 프레임 {used.sum()} / 목표 {ok.sum()} | 팔 회전각 벗어남 0° {int((dd < 1e-9).sum())}, "
                  f"중앙 {np.median(dd) if len(dd) else np.nan:.0f}° 95% {np.percentile(dd, 95) if len(dd) else np.nan:.0f}° | "
                  f"시간 ms 중앙 sew {np.median(time_ms[used]) if used.any() else np.nan:.2f} / 격자로 넘어감 "
                  f"{np.median(time_ms[ok & ~used]) if (ok & ~used).any() else np.nan:.2f} ({(ok & ~used).sum()}프레임)")
        if a.select == "smooth":
            print(f"    속도 한계를 넘는 갈래 바뀜 {int(solver.episode_branch_switch.sum())}회")
        jump = np.degrees(np.abs(wrap(np.diff(q, axis=0))).max(axis=1))
        print(f"    프레임 사이 관절 최대 변화 99% {np.percentile(jump, 99):.1f}°, 30° 넘는 프레임 {int((jump > 30).sum())}")
        return
    print(f"{eid} hand={info['hand']} scale={info['scale']:.3f} r_off={a.r_off} anchor={a.anchor} select={a.select}")
    print(common.summarize("C_eaik", q, sat, status, pos_err, rot_err, sec))
    s = np.degrees(swv_err[np.isfinite(swv_err)])
    if len(s):
        print(f"  팔 회전각 오차 deg  중앙 {np.median(s):.1f}  95% {np.percentile(s, 95):.1f}  최대 {s.max():.1f}  ({len(s)}프레임)")
    ok = status != common.STATUS_NO_TARGET
    if ok.any():
        print(f"  후보 해 수 중앙 {np.median(n_cand[ok]):.0f}  /  3번 관절 고른 값 deg 범위 "
              f"{np.degrees(np.nanmin(q3_pick)):.1f} ~ {np.degrees(np.nanmax(q3_pick)):.1f}")
        jump = np.degrees(np.abs(wrap(np.diff(q[ok], axis=0))).max(axis=1))
        print(f"  연속 프레임 사이 관절 최대 변화 deg  중앙 {np.median(jump):.2f}  최대 {jump.max():.1f}")
    print(f"  IK 시간 {common.timing_text(time_ms)}")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
