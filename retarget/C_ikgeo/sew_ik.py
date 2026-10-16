"""KUKA iiwa14용 SEW 각 닫힌 해 IK (stereo-sew의 iiwa 경우를 파이썬으로 옮긴 프로토타입).

영 자세에서 관절 축(월드): z, y, z, -y, z, y, z. 어깨 S(관절 1·2·3 교점), 팔꿈치 E(관절 4), 손목 W(관절 5·6·7 교점),
플랜지(link7 원점)가 모두 z축 위에 있고 link7 방향은 단위행렬이다.
  R_07 = Rz(q1) Ry(q2) Rz(q3) Ry(-q4) Rz(q5) Ry(q6) Rz(q7)
SEW 각 psi: 기준 벡터 r과 S→W가 이루는 평면에서, S→W 축 둘레로 팔꿈치가 돌아간 각 (conventional SEW).
"""
import numpy as np


class IiwaSEW:
    def __init__(self, S, L_se, L_ew, d_wf, r=(0.0, 0.0, 1.0)):
        self.S = np.asarray(S, float)          # 베이스 기준 어깨 위치
        self.L1, self.L2, self.d7 = L_se, L_ew, d_wf
        self.r = np.asarray(r, float) / np.linalg.norm(r)

    @classmethod
    def from_mujoco(cls, m, r=(0.0, 0.0, 1.0)):
        import mujoco
        d = mujoco.MjData(m)
        d.qpos[:] = 0
        mujoco.mj_forward(m, d)
        S, E, W = d.xanchor[1], d.xanchor[3], d.xanchor[5]
        F = d.body("link7").xpos
        return cls(S.copy(), np.linalg.norm(E - S), np.linalg.norm(W - E), np.linalg.norm(F - W), r)

    # --- SEW 각 정의 ---
    def _frame(self, W):
        sw = W - self.S
        d = np.linalg.norm(sw)
        e = sw / d
        n0 = self.r - np.dot(self.r, e) * e     # r에서 S→W 성분을 뺀 방향 = psi 0 방향
        nn = np.linalg.norm(n0)
        return d, e, n0 / nn, np.cross(e, n0 / nn), nn

    def psi_from_points(self, W, E):
        _, e, n0, n1, _ = self._frame(W)
        v = E - self.S
        return np.arctan2(np.dot(n1, v), np.dot(n0, v))

    # --- IK ---
    def ik(self, R07, p07, psi):
        """link7 목표 자세(R07, p07)와 SEW 각 psi로 관절각 해 8개(어깨·팔꿈치·손목 각 2해)를 돌려준다.

        도달할 수 없으면 빈 리스트. 관절 한계는 거르지 않는다.
        주의: q2·q4·q6가 0에 가까운 자세(팔을 일직선으로 편 경우 등)는 0에 가까운 값으로 나누게 되어 아직 처리하지 않았다.
        """
        W = p07 - self.d7 * R07[:, 2]
        d, e, n0, n1, ref_norm = self._frame(W)
        L1, L2 = self.L1, self.L2
        if d > L1 + L2 or d < abs(L1 - L2):
            return []
        # 팔꿈치 위치: S→W 축 둘레 원 위, psi로 지정
        a = (L1**2 - L2**2 + d**2) / (2 * d)
        h = np.sqrt(max(L1**2 - a**2, 0.0))
        E = self.S + a * e + h * (np.cos(psi) * n0 + np.sin(psi) * n1)
        u = (E - self.S) / L1                  # 위팔 방향
        f = (W - E) / L2                       # 아래팔 방향
        c4 = np.clip(np.dot(u, f), -1, 1)       # 영 자세(일직선)에서 q4 = 0

        sols = []
        for s2 in (1, -1):                     # 어깨 2해
            q2 = s2 * np.arccos(np.clip(u[2], -1, 1))
            sq2 = np.sin(q2)
            q1 = np.arctan2(u[1] / sq2, u[0] / sq2)
            R02 = rz(q1) @ ry(q2)
            A = R02.T @ f                      # = Rz(q3) [-sin q4, 0, cos q4]
            for s4 in (1, -1):                 # 팔꿈치 2해
                q4 = s4 * np.arccos(c4)
                sq4 = np.sin(q4)
                q3 = np.arctan2(-A[1] / sq4, -A[0] / sq4)
                R04 = R02 @ rz(q3) @ ry(-q4)
                R47 = R04.T @ R07              # = Rz(q5) Ry(q6) Rz(q7)
                for s6 in (1, -1):             # 손목 2해
                    q6 = s6 * np.arccos(np.clip(R47[2, 2], -1, 1))
                    sq6 = np.sin(q6)
                    q5 = np.arctan2(R47[1, 2] / sq6, R47[0, 2] / sq6)
                    q7 = np.arctan2(R47[2, 1] / sq6, -R47[2, 0] / sq6)
                    sols.append(np.array([q1, q2, q3, q4, q5, q6, q7]))
        return sols


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
