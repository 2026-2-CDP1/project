"""5-1 궤적 재생: 관절 각도 궤적을 로봇 모델에 입혀 프레임별 TCP 자세와 접촉 목록을 돌려준다.

기구학 재생이다. 관절 각도를 qpos에 그대로 넣고 mj_forward로 위치·접촉만 다시 계산한다.
mj_step은 쓰지 않는다 (중력·제어기가 각도를 바꿔 버림). 함수 형태는 replay/README.md에 고정.

사용 (레포 루트에서):
    from replay.replay import replay
    out = replay(q_traj)                       # q_traj: (T, 7) rad
    out["tcp_pos"], out["tcp_quat"], out["contacts"]

점검 (모델 로드·형태·정상 자세 접촉):
    python replay/replay.py --check --out ~/out/본인이름/replay_check
"""
import argparse
import re
import warnings
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "robot" / "robot.xml"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
TCP_SITES = ("tcp", "attachment_site")  # 앞에 있는 것부터 찾는다. tcp는 그리퍼가 붙은 뒤 1번이 추가
# 자기 충돌 검사에서 빼는 몸체 쌍. 근거는 replay/collision_exclude.md. robot.xml의 <exclude>는 이 목록으로 바꿔 읽는다
COLLISION_EXCLUDE = {("base", "link1"), ("link5", "link7")}


def load_model(model_path, exclude=COLLISION_EXCLUDE):
    """robot.xml을 읽되 충돌 제외 쌍을 exclude로 바꿔 컴파일한다 (원본 파일은 건드리지 않음)."""
    path = Path(model_path).resolve()
    xml = path.read_text(encoding="utf-8")
    xml = re.sub(r'<exclude body1="[^"]+" body2="[^"]+"\s*/>', "", xml)
    rule = "".join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in sorted(exclude))
    if "<contact>" in xml:
        xml = xml.replace("<contact>", "<contact>" + rule, 1)
    else:
        xml = xml.replace("</mujoco>", f"<contact>{rule}</contact></mujoco>", 1)
    # 메시 경로를 절대 경로로 (문자열로 컴파일하므로)
    xml = re.sub(r'meshdir="([^"]+)"', lambda mo: f'meshdir="{(path.parent / mo.group(1)).as_posix()}"', xml)
    return mujoco.MjModel.from_xml_string(xml)


class Replayer:
    """모델을 한 번 읽어 두고 궤적을 여러 번 재생한다."""

    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = Path(model_path)
        self.m = load_model(self.model_path)
        self.collision_exclude = sorted(COLLISION_EXCLUDE)
        self.d = mujoco.MjData(self.m)
        self.arm_adr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ARM_JOINTS])

        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.m.nsite)]
        self.tcp_site = next((s for s in TCP_SITES if s in names), None)
        if self.tcp_site is None:
            raise ValueError(f"TCP site가 없음: {TCP_SITES} 중 하나가 {self.model_path}에 있어야 함")
        if self.tcp_site != "tcp":
            warnings.warn(f"'tcp' site가 아직 없어 '{self.tcp_site}'(플랜지)를 TCP로 씀. 비교용 값으로 쓰지 말 것")
        self.tcp_id = self.m.site(self.tcp_site).id

        # 팔 이외 관절(그리퍼 등)의 기본값: keyframe "home", 없으면 qpos0
        key = next((k for k in range(self.m.nkey) if self.m.key(k).name == "home"), None)
        self.q_full_home = (self.m.key_qpos[key] if key is not None else self.m.qpos0).copy()

        # 로봇 몸체 = base 아래 모든 body. 그 밖의 body(테이블·장애물)와의 접촉은 환경 접촉
        base = self.m.body("base").id
        self.robot_bodies = {
            self.m.body(i).name for i in range(self.m.nbody)
            if i == base or self._is_descendant(i, base)
        }
        self.has_gripper = False  # 그리퍼 관절·gripper.csv가 생기면 여기서 읽는다

    def _is_descendant(self, i, root):
        while i != 0:
            if i == root:
                return True
            i = self.m.body_parentid[i]
        return False

    def replay(self, q_traj, gripper=None):
        q_traj = np.asarray(q_traj, dtype=float)
        if q_traj.ndim != 2 or q_traj.shape[1] != 7:
            raise ValueError(f"q_traj는 (T, 7)이어야 함: {q_traj.shape}")
        if not np.all(np.isfinite(q_traj)):
            raise ValueError("q_traj에 NaN/inf가 있음 (실패 프레임도 4번 규칙대로 유한한 값이어야 함)")
        if gripper is not None:
            gripper = np.asarray(gripper, dtype=float)
            if gripper.shape != (len(q_traj),):
                raise ValueError(f"gripper는 (T,)이어야 함: {gripper.shape}")
            if not self.has_gripper:
                warnings.warn("모델에 그리퍼가 아직 없어 gripper 값은 무시함")

        T = len(q_traj)
        tcp_pos = np.zeros((T, 3))
        tcp_quat = np.zeros((T, 4))
        contacts = []
        m, d = self.m, self.d
        for t in range(T):
            d.qpos[:] = self.q_full_home
            d.qpos[self.arm_adr] = q_traj[t]
            mujoco.mj_forward(m, d)  # 위치·접촉만 계산, 시간은 흐르지 않음
            tcp_pos[t] = d.site_xpos[self.tcp_id]
            mujoco.mju_mat2Quat(tcp_quat[t], d.site_xmat[self.tcp_id])  # (w, x, y, z)
            frame = []
            for i in range(d.ncon):
                c = d.contact[i]
                b1 = m.body(m.geom_bodyid[c.geom1]).name
                b2 = m.body(m.geom_bodyid[c.geom2]).name
                frame.append((b1, b2, float(c.dist)))
            contacts.append(frame)
        return {
            "tcp_pos": tcp_pos,
            "tcp_quat": tcp_quat,
            "contacts": contacts,
            "meta": {
                "model": str(self.model_path),
                "tcp_site": self.tcp_site,
                "robot_bodies": sorted(self.robot_bodies),
                "collision_exclude": [list(p) for p in self.collision_exclude],
            },
        }


_default = None


def replay(q_traj, gripper=None):
    """기본 모델(robot/robot.xml)로 재생. 형태는 replay/README.md 참고."""
    global _default
    if _default is None:
        _default = Replayer()
    return _default.replay(q_traj, gripper)


def is_self_contact(contact, robot_bodies):
    """(body1, body2, dist)가 로봇 몸체끼리의 접촉인지."""
    return contact[0] in robot_bodies and contact[1] in robot_bodies


def check(out_dir, n, seed):
    """모델 로드, 반환 형태, home·무작위 자세의 접촉을 점검해 텍스트로 저장."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Replayer()
    lo, hi = r.m.jnt_range[:7, 0], r.m.jnt_range[:7, 1]
    rng = np.random.default_rng(seed)
    q = np.vstack([r.q_full_home[r.arm_adr], rng.uniform(lo, hi, size=(n, 7))])
    out = r.replay(q)

    lines = [
        f"model: {r.model_path}",
        f"tcp_site: {r.tcp_site}",
        f"robot_bodies: {sorted(r.robot_bodies)}",
        f"collision_exclude: {r.collision_exclude}",
        f"shapes: tcp_pos {out['tcp_pos'].shape}, tcp_quat {out['tcp_quat'].shape}, contacts {len(out['contacts'])}",
        f"quat norm max |1-|q||: {np.abs(1 - np.linalg.norm(out['tcp_quat'], axis=1)).max():.2e}",
        f"home 자세 TCP 위치 (m): {np.round(out['tcp_pos'][0], 4).tolist()}",
        f"home 자세 접촉 수: {len(out['contacts'][0])}",
    ]
    pairs = Counter()
    frames_with_pen = 0
    for frame in out["contacts"][1:]:
        pen = [c for c in frame if c[2] < 0 and is_self_contact(c, r.robot_bodies)]
        frames_with_pen += bool(pen)
        for pair in {tuple(sorted((b1, b2))) for b1, b2, _ in pen}:  # 자세마다 쌍 하나는 한 번만
            pairs[pair] += 1
    lines.append(f"무작위 자세 {n}개 (seed {seed}) 중 자기 침투가 있는 자세: {frames_with_pen}")
    lines.append("몸체 쌍별 침투 자세 수 (collision_exclude 검토용):")
    lines += [f"  {a}–{b}: {k}" for (a, b), k in pairs.most_common()] or ["  없음"]
    text = "\n".join(lines)
    (out_dir / "replay_check.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n저장: {out_dir / 'replay_check.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="모델 로드·형태·접촉 점검")
    ap.add_argument("--out", help="결과 저장 폴더 (~/project 밖, 예: ~/out/본인이름/replay_check)")
    ap.add_argument("--n", type=int, default=200, help="무작위 자세 개수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.check or not a.out:
        ap.error("--check --out <폴더> 로 실행")
    check(a.out, a.n, a.seed)
