"""robot/ 모델 점검: 로드, 관절 한계 비교, FK 일치 검사, 오프스크린 렌더링(PNG).

서버 사용 예 (렌더링은 GPU 노드에서):
    source ~/use_robot.sh
    cd ~/project
    python robot/check_robot.py --out ~/out/본인이름/robot_check            # 로드·한계·FK만
    MUJOCO_GL=egl python robot/check_robot.py --out ~/out/본인이름/robot_check --render

비교 프레임: 그리퍼·TCP가 붙기 전에는 플랜지(MJCF site `attachment_site` ↔ URDF `iiwa_link_ee_kuka`).
TCP가 추가되면 --mj-site tcp --pin-frame tcp 로 같은 검사를 다시 돌린다.
"""
import argparse
import csv
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

ROBOT_DIR = Path(__file__).resolve().parent
MJ_JOINTS = [f"joint{i}" for i in range(1, 8)]
PIN_JOINTS = [f"iiwa_joint_{i}" for i in range(1, 8)]


def load():
    mj = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "robot.xml"))
    pm = pin.buildModelFromUrdf(str(ROBOT_DIR / "robot.urdf"))
    return mj, pm


def joint_limits(mj, pm, out_dir):
    rows, worst = [], 0.0
    for mj_name, pin_name in zip(MJ_JOINTS, PIN_JOINTS):
        j = mj.joint(mj_name).id
        assert mj.jnt_limited[j], f"{mj_name}: MJCF 관절 한계가 꺼져 있음"
        mj_lo, mj_hi = mj.jnt_range[j]
        iq = pm.joints[pm.getJointId(pin_name)].idx_q
        pin_lo, pin_hi = pm.lowerPositionLimit[iq], pm.upperPositionLimit[iq]
        worst = max(worst, abs(mj_lo - pin_lo), abs(mj_hi - pin_hi))
        rows.append((mj_name, pin_name, mj_lo, mj_hi, pin_lo, pin_hi))
    with open(out_dir / "joint_limits_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mjcf_joint", "urdf_joint", "mjcf_lower", "mjcf_upper", "urdf_lower", "urdf_upper"])
        w.writerows(rows)
    print(f"[limits] 최대 차이 {worst:.3e} rad")
    return worst


def fk_check(mj, pm, mj_site, pin_frame, n, seed):
    md, pd = mujoco.MjData(mj), pm.createData()
    fid = pm.getFrameId(pin_frame)
    assert fid < len(pm.frames), f"URDF에 프레임 {pin_frame} 없음"
    mj_adr = [mj.jnt_qposadr[mj.joint(nm).id] for nm in MJ_JOINTS]
    pin_adr = [pm.joints[pm.getJointId(nm)].idx_q for nm in PIN_JOINTS]
    lo, hi = pm.lowerPositionLimit[pin_adr], pm.upperPositionLimit[pin_adr]
    rng = np.random.default_rng(seed)
    worst_p = worst_r = 0.0
    for _ in range(n):
        q7 = rng.uniform(lo, hi)
        md.qpos[:] = 0
        md.qpos[mj_adr] = q7
        mujoco.mj_forward(mj, md)
        p1, R1 = md.site(mj_site).xpos.copy(), md.site(mj_site).xmat.reshape(3, 3).copy()
        q = pin.neutral(pm)
        q[pin_adr] = q7
        pin.forwardKinematics(pm, pd, q)
        pin.updateFramePlacements(pm, pd)
        p2, R2 = pd.oMf[fid].translation, pd.oMf[fid].rotation
        worst_p = max(worst_p, np.linalg.norm(p1 - p2))
        worst_r = max(worst_r, np.linalg.norm(pin.log3(R1.T @ R2)))
    print(f"[fk] {mj_site} ↔ {pin_frame}, 자세 {n}개(seed {seed}): "
          f"최대 위치 차이 {worst_p:.3e} m, 최대 방향 차이 {worst_r:.3e} rad")
    return worst_p, worst_r


def write_png(path, rgb):
    """표준 라이브러리만으로 RGB 배열을 PNG로 저장 (robot 환경에 imageio·PIL 없음)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def render(mj, out_dir, width=640, height=480):
    """영 자세와 관절별 +0.8 rad 자세를 PNG로 저장 (뷰어 대신 눈으로 확인용)."""
    md = mujoco.MjData(mj)
    renderer = mujoco.Renderer(mj, height, width)  # 모델 기본 오프스크린 버퍼 640x480
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20
    poses = {"zero": np.zeros(7)}
    for i in range(7):
        q = np.zeros(7)
        q[i] = 0.8
        poses[f"joint{i + 1}_+0.8"] = q
    for name, q in poses.items():
        md.qpos[:7] = q
        mujoco.mj_forward(mj, md)
        renderer.update_scene(md, cam)
        write_png(out_dir / f"{name}.png", renderer.render())
    print(f"[render] PNG {len(poses)}장 저장: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 저장 폴더 (~/project 밖, 예: ~/out/본인이름/robot_check)")
    ap.add_argument("--mj-site", default="attachment_site")
    ap.add_argument("--pin-frame", default="iiwa_link_ee_kuka")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--render", action="store_true", help="오프스크린 렌더링 (GPU 노드에서 MUJOCO_GL=egl)")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    mj, pm = load()
    print(f"[load] OK  mujoco {mujoco.__version__}, pinocchio {pin.__version__}, MJCF nq={mj.nq}, URDF nq={pm.nq}")

    lim = joint_limits(mj, pm, out_dir)
    wp, wr = fk_check(mj, pm, args.mj_site, args.pin_frame, args.n, args.seed)
    if args.render:
        render(mj, out_dir)

    ok = lim <= args.tol and wp <= args.tol and wr <= args.tol
    print("PASS" if ok else f"FAIL (기준 {args.tol})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
