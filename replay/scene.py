"""5-1 환경 장면: 테이블·장애물 상자를 로봇 베이스 좌표로 옮겨 재생 장면에 넣는다.

지표(정의서 v0.4 2.6 환경 충돌률)는 장애물 상자만 쓰고, 테이블은 상자 대신 면으로 판정한다 (metrics/metrics.py).
테이블 상자는 영상 등에서 그려 볼 때만 쓴다.

입력
- 장애물: 3번 obstacles.json (episodes/tools/obstacles.py). 테이블 좌표계 상자 {center, size, yaw_deg}
- 테이블: replay/scene_config.json 의 테이블 윗면 사각형 (테이블 좌표계, 윗면 z = 0)
- 좌표 변환: 4번 to_robot_point (테이블 좌표 점 → 로봇 베이스 좌표 점)

옮기는 방법: 상자 꼭짓점 8개를 to_robot_point로 하나씩 옮기고, 옮긴 꼭짓점으로 상자를 다시 세운다.
4번 변환은 "평행 이동 + 축척 s + 회전 R_tb"라 상자는 상자로 옮겨지고 크기는 s배가 된다. 그래서 s나 R_tb를
따로 몰라도 된다. 변환이 상자 모양을 깨면(축마다 축척이 다르거나 기울어짐) 오류를 낸다.

사용 (레포 루트에서):
    from replay.scene import episode_env, load_obstacles, load_config
    boxes, exclude = episode_env(load_obstacles(path).get("T3_E031"), load_config(), to_robot_point)
    r = Replayer(env_boxes=boxes, env_exclude=exclude)        # replay.replay.Replayer

    python replay/scene.py --check --out ~/out/신서연/scene_check      # 합성 검증
    python replay/scene.py --show --obstacles <obstacles.json> --id T3_E031 --identity-transform
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "scene_config.json"
# replay/ 안에서 실행하면 "replay"가 폴더 대신 replay.py로 잡히므로, 스크립트 폴더를 빼고 레포 루트를 넣는다
def _is_here(p, here=Path(__file__).resolve().parent):
    """같은 폴더인지 (한글 경로가 NFC/NFD로 다르게 들어와도 같게 보도록 문자열 대신 samefile로)."""
    try:
        return os.path.samefile(p or ".", here)
    except OSError:
        return False


sys.path[:] = [p for p in sys.path if not _is_here(p)]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- 테이블 좌표계 상자

def box_corners(center, size, yaw_deg):
    """상자 → 꼭짓점 8개 (아래 4개, 위 4개). size = [yaw 방향 길이, 두께, 높이] (obstacles.py와 같은 규약)."""
    (cx, cy, cz), (L, T, H), a = center, size, math.radians(yaw_deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    base = [np.array([cx, cy]) + sx * L / 2 * u + sy * T / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.array([[*b, cz - H / 2] for b in base] + [[*b, cz + H / 2] for b in base])


def table_box(spec):
    """scene_config의 테이블 윗면 사각형 → 테이블 좌표계 상자 (윗면이 z = 0)."""
    (x0, x1), (y0, y1), th = spec["x"], spec["y"], spec["thickness"]
    return {"center": [(x0 + x1) / 2, (y0 + y1) / 2, -th / 2], "size": [x1 - x0, y1 - y0, th], "yaw_deg": 0.0}


# ---------------------------------------------------------------- 로봇 좌표로 옮기기

def mat_to_quat(R):
    """회전 행렬 → 쿼터니언 (w, x, y, z)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def map_box(name, corners, to_robot, tol=1e-6):
    """테이블 좌표 꼭짓점 8개 → 로봇 좌표 상자 {name, center, size, quat}. 상자 모양이 깨지면 ValueError."""
    P = np.array([np.asarray(to_robot(p), float) for p in corners])
    ex, ey, ez = P[1] - P[0], P[3] - P[0], P[4] - P[0]          # 길이·두께·높이 방향 모서리
    size = np.array([np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)])
    R = np.column_stack([ex / size[0], ey / size[1], ez / size[2]])
    scale = size / np.array([np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[0]),
                             np.linalg.norm(corners[4] - corners[0])])
    if (np.abs(R.T @ R - np.eye(3)).max() > tol or np.linalg.det(R) < 0
            or np.ptp(scale) > tol * max(1.0, scale.max())):
        raise ValueError(f"{name}: to_robot_point가 상자 모양을 유지하지 않음 (평행 이동·축척·회전만 허용)")
    center = P.mean(axis=0)
    return {"name": name, "center": center.tolist(), "size": size.tolist(), "quat": mat_to_quat(R).tolist()}


# ---------------------------------------------------------------- 에피소드 장면

def load_config(path=CONFIG):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_obstacles(path):
    """3번 obstacles.json → {에피소드ID: 항목}."""
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        return json.load(f)["episodes"]


def episode_env(entry, config, to_robot):
    """에피소드 하나의 환경 상자 목록과 검사 제외 쌍 (로봇 베이스 좌표).

    entry   obstacles.json의 그 에피소드 항목. None이면 장애물 없음 (테이블만)
    반환    (boxes, env_exclude). 테이블 크기가 설정에 없으면 테이블은 빼고 경고
    """
    boxes = []
    tab = config.get("table") or {}
    if all(tab.get(k) is not None for k in ("x", "y", "thickness")):
        t = table_box(tab)
        boxes.append(map_box("env_table", box_corners(t["center"], t["size"], t["yaw_deg"]), to_robot))
    else:
        print("[주의] scene_config.json에 테이블 크기가 없어 테이블 없이 장애물만 넣음", file=sys.stderr)
    for i, ob in enumerate((entry or {}).get("obstacles", []), 1):
        name = f"env_{ob.get('name') or f'obs{i}'}"
        boxes.append(map_box(name, box_corners(ob["center"], ob["size"], ob["yaw_deg"]), to_robot))
    names = {b["name"] for b in boxes}
    exclude = [(body, "env_table") for body in config.get("table_exclude_bodies", []) if "env_table" in names]
    return boxes, exclude


# ---------------------------------------------------------------- 검증

def check(out_dir):
    """합성 검증: 옮기기(항등·축척+회전), 상자 모양 깨짐, 환경 접촉 검출, 테이블 제외 쌍."""
    sys.path.insert(0, str(REPO))
    from replay.replay import Replayer, is_self_contact

    rows = []

    def rec(name, how, ok, detail=""):
        rows.append((name, how, ok, detail))
        print(f"[{'통과' if ok else '실패'}] {name} {detail}")

    ob = {"center": [0.3, 0.2, 0.075], "size": [0.2, 0.01, 0.15], "yaw_deg": 30.0}
    C = box_corners(ob["center"], ob["size"], ob["yaw_deg"])

    b = map_box("t", C, lambda p: p)
    rec("항등 변환", "obstacles.json 상자 (yaw 30°)", np.allclose(b["center"], ob["center"])
        and np.allclose(b["size"], ob["size"]) and np.allclose(b["quat"], [math.cos(math.radians(15)), 0, 0,
                                                                              math.sin(math.radians(15))]),
        f"center {np.round(b['center'], 4).tolist()} size {np.round(b['size'], 4).tolist()}")

    s, p0, c0 = 0.8, np.array([0.1, -0.4, 0.35]), np.array([0.0, 0.0, 0.36])
    a = math.radians(90)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    f = lambda p: c0 + s * Rz @ (np.asarray(p) - p0)              # 4번 to_robot_point 형태
    b = map_box("t", C, f)
    rec("축척 0.8 + 90° 회전", "p_robot = c0 + s·R·(p − p0)",
        np.allclose(b["center"], f(np.array(ob["center"]))) and np.allclose(b["size"], s * np.array(ob["size"]))
        and np.allclose(abs(np.array(b["quat"]) @ [math.cos(math.radians(60)), 0, 0, math.sin(math.radians(60))]), 1),
        f"size {np.round(b['size'], 4).tolist()} (= 0.8배)")

    try:
        map_box("t", C, lambda p: np.asarray(p) * [1.0, 2.0, 1.0])
        rec("축마다 다른 축척 거부", "x·z는 1배, y만 2배", False)
    except ValueError:
        rec("축마다 다른 축척 거부", "x·z는 1배, y만 2배", True, "ValueError")

    r0 = Replayer()
    home = r0.q_full_home[r0.arm_adr]
    tcp = r0.replay(home[None])["tcp_pos"][0]
    box = {"name": "env_test", "center": tcp.tolist(), "size": [0.1, 0.1, 0.1], "quat": [1, 0, 0, 0]}
    r = Replayer(env_boxes=[box])
    out = r.replay(np.vstack([home, home + [math.pi / 2, 0, 0, 0, 0, 0, 0]]))
    env = [[c for c in fr if not is_self_contact(c, r.robot_bodies) and c[2] < 0] for fr in out["contacts"]]
    rec("환경 접촉 검출", "home TCP 자리에 10 cm 상자, home → joint1 +90°",
        bool(env[0]) and not env[1] and set(r.robot_bodies) == set(r0.robot_bodies),
        f"home {len(env[0])}개, 돌린 뒤 {len(env[1])}개, 로봇 몸체 목록 그대로")

    slab = {"name": "env_table", "center": [0, 0, 0.1], "size": [1.0, 1.0, 0.04], "quat": [1, 0, 0, 0]}
    hit = lambda rr: sorted({c[0] if c[1] == "env_table" else c[1] for c in rr.replay(home[None])["contacts"][0]
                             if "env_table" in c[:2] and c[2] < 0})
    without = hit(Replayer(env_boxes=[slab]))
    with_ex = hit(Replayer(env_boxes=[slab], env_exclude=[("link1", "env_table")]))
    rec("base는 테이블과 원래 검사 안 함", "베이스를 관통하는 판 (base·테이블 모두 월드 고정 → MuJoCo가 검사 안 함)",
        "base" not in without, f"판에 닿은 몸체 {without}")
    rec("테이블 제외 쌍", "같은 판, (link1, env_table) 제외 전·후",
        "link1" in without and "link1" not in with_ex, f"제외 전 {without}, 제외 후 {with_ex}")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 환경 장면 합성 검증", "", f"`python replay/scene.py --check` — {sum(r[2] for r in rows)}/{len(rows)} 통과",
             "", "| 검사 | 만드는 법 | 판정 | 값 |", "|---|---|---|---|"]
    lines += [f"| {n} | {h} | {'통과' if ok else '**실패**'} | {d} |" for n, h, ok, d in rows]
    (out_dir / "scene_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{sum(r[2] for r in rows)}/{len(rows)} 통과 → {out_dir / 'scene_check.md'}")
    return all(r[2] for r in rows)


def show(a):
    """에피소드 하나의 장면 상자를 로봇 좌표로 출력하고 home 자세 환경 접촉을 본다."""
    sys.path.insert(0, str(REPO))
    from replay.replay import Replayer, is_self_contact
    if a.identity_transform:
        to_robot = lambda p: np.asarray(p, float)
    else:
        sys.path.insert(0, str(REPO / "retarget"))
        from common import to_robot_point as to_robot  # 4번 제효정 (없으면 --identity-transform)
    boxes, exclude = episode_env(load_obstacles(a.obstacles).get(a.id), load_config(), to_robot)
    for b in boxes:
        print(b["name"], "center", np.round(b["center"], 4).tolist(), "size", np.round(b["size"], 4).tolist())
    r = Replayer(env_boxes=boxes, env_exclude=exclude)
    con = [c for c in r.replay(r.q_full_home[r.arm_adr][None])["contacts"][0]
           if not is_self_contact(c, r.robot_bodies) and c[2] < 0]
    print(f"home 자세 환경 침투: {con or '없음'} (제외 쌍 {exclude})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-1 환경 장면 (테이블·장애물)")
    ap.add_argument("--check", action="store_true", help="합성 검증")
    ap.add_argument("--show", action="store_true", help="에피소드 하나의 장면 확인")
    ap.add_argument("--out", help="--check 결과 폴더 (~/project 밖)")
    ap.add_argument("--obstacles", help="3번 obstacles.json")
    ap.add_argument("--id", help="에피소드 ID")
    ap.add_argument("--identity-transform", action="store_true", help="4번 변환 대신 항등 (시험용)")
    a = ap.parse_args()
    if a.check and a.out:
        sys.exit(0 if check(a.out) else 1)
    elif a.show and a.obstacles and a.id:
        show(a)
    else:
        ap.error("--check --out <폴더> 또는 --show --obstacles <json> --id <ID>")
