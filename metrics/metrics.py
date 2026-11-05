"""5-2 지표 계산: 리타게팅 결과(관절 궤적)를 재생해 정의서의 지표를 계산한다.

정의는 metrics/definitions.md(v0.2)가 기준이다. 이 코드와 정의서가 다르면 정의서를 따르고 코드를 고친다.
모든 지표는 에피소드 하나 x 알고리즘 하나 단위이고, 낮을수록 좋다.

GPU는 필요 없다 (렌더링 없이 mj_forward만). 서버에서는 source ~/use_robot.sh 후 srun (--gres 없이) CPU 노드에서.

사용 (레포 루트에서, mujoco가 있는 환경):
    # 전 에피소드 x 알고리즘 → metrics_long.csv, episode_info.csv
    python metrics/metrics.py run --retarget <4번 결과 폴더> --episodes <3번 결과 폴더> \
        --sync sync/sync.csv --out ~/out/신서연/metrics

    # 합성 데이터 검증 (정답을 아는 궤적으로 코드 확인)
    python metrics/selftest.py --out ~/out/신서연/metrics_selftest

폴더 구조 (각 담당 문서 기준):
    <retarget>/<알고리즘>/<에피소드ID>.npz    q (T,7), status (T,), saturated (T,7), pos_err (T,)   ← 4번
    <episodes>/<에피소드ID>/                   keypoints.npy (T,7,3), keypoint_names.json, exo_ts.csv,
                                              meta.json, labels.json                                ← 3번
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIMITS_CSV = REPO / "robot" / "joint_limits.csv"
EPS_LIMIT = 1e-6          # 2.1 수치 오차로 보는 한계 초과량 (rad)
JERK_HZ = 30.0            # 2.6 저크는 이 주기로 맞춰 계산
JERK_DT_TOL = 1e-3        # 2.6 맞춘 Δt가 1/30 s와 이 비율 넘게 다르면 표시

# 순위 사용 여부는 definitions.md 7절. 표·집계에서 이 순서로 쓴다
METRICS = [
    "limit_violation_rate", "limit_saturation_rate",
    "self_collision_rate", "max_penetration_m",
    "ik_failure_rate", "ik_not_converged_rate", "ik_no_solution_rate",
    "jerk_rms", "grasp_dist_m",
    "track_pos_err_mean_m", "env_collision_rate", "env_max_penetration_m",
]
REFERENCE_ONLY = {"track_pos_err_mean_m", "env_collision_rate", "env_max_penetration_m"}


# ---------------------------------------------------------------- 지표 하나씩 (정의서 2절)

def load_limits(path=LIMITS_CSV):
    """1번 joint_limits.csv → lo, hi (7,) rad. 관절 순서 joint1~7을 확인한다."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["joint"] for r in rows]
    if names != [f"joint{i}" for i in range(1, 8)]:
        raise ValueError(f"joint_limits.csv 관절 순서가 joint1~7이 아님: {names}")
    lo = np.array([float(r["lower_rad"]) for r in rows])
    hi = np.array([float(r["upper_rad"]) for r in rows])
    return lo, hi


def limit_violation_rate(q, lo, hi, eps=EPS_LIMIT):
    """2.1 한 관절이라도 한계를 eps 넘게 벗어난 프레임 비율."""
    viol = ((q < lo - eps) | (q > hi + eps)).any(axis=1)
    return float(viol.mean())


def limit_saturation_rate(saturated):
    """2.2 한 관절이라도 saturated가 참인 프레임 비율."""
    return float(np.asarray(saturated, bool).any(axis=1).mean())


def contact_depths(contacts, robot_bodies):
    """프레임별 가장 깊은 자기 접촉·환경 접촉 거리 (접촉 없으면 0). 음수 = 파고듦.

    자기 접촉 = 두 몸체가 모두 로봇 몸체 (replay.is_self_contact와 같은 기준), 나머지는 환경 접촉.
    """
    robot = set(robot_bodies)
    self_d = np.zeros(len(contacts))
    env_d = np.zeros(len(contacts))
    for t, frame in enumerate(contacts):
        for b1, b2, dist in frame:
            if b1 in robot and b2 in robot:
                self_d[t] = min(self_d[t], dist)
            else:
                env_d[t] = min(env_d[t], dist)
    return self_d, env_d


def collision_rate(depth):
    """2.3 / 2.9 파고든 접촉이 있는 프레임 비율."""
    return float((depth < 0).mean())


def max_penetration(depth):
    """2.4 / 2.9 가장 깊게 파고든 깊이 (m). 없으면 0."""
    return float(max(0.0, -depth.min())) if len(depth) else 0.0


def ik_rates(status):
    """2.5 status 0 성공, 1 미수렴, 2 해 없음. 각 알고리즘 출력 그대로 센다 (재탐색 없음, Q11)."""
    status = np.asarray(status)
    bad = set(np.unique(status)) - {0, 1, 2}
    if bad:
        raise ValueError(f"status에 0·1·2 밖의 값이 있음: {sorted(bad)} (4번과 상태 코드 확인)")
    return {
        "ik_failure_rate": float((status != 0).mean()),
        "ik_not_converged_rate": float((status == 1).mean()),
        "ik_no_solution_rate": float((status == 2).mean()),
    }


def jerk_rms(q, ts, hz=JERK_HZ):
    """2.6 30 Hz로 맞춘 궤적(보간·스무딩 없이 k칸마다 고름)의 저크 RMS (rad/s³).

    반환: (jerk_rms, k, dt). 60 fps 에피소드면 k = 2, 30 fps면 k = 1.
    """
    q = np.asarray(q, float)
    ts = np.asarray(ts, float)
    step = np.median(np.diff(ts))
    k = max(1, int(round((1.0 / hz) / step)))
    q_s, ts_s = q[::k], ts[::k]
    if len(q_s) < 4:
        return float("nan"), k, float("nan")
    dt = float(np.median(np.diff(ts_s)))
    jerk = np.diff(q_s, n=3, axis=0) / dt ** 3
    return float(np.sqrt(np.mean(jerk ** 2))), k, dt


def human_grasp_point(kp, names, g):
    """2.7 파지 프레임 g의 엄지 끝·검지 끝 중점 (테이블 좌표, m). 하나라도 NaN이면 None."""
    i_th, i_ix = names.index("thumb_tip"), names.index("index_tip")
    p = (kp[g, i_th] + kp[g, i_ix]) / 2.0
    return None if not np.all(np.isfinite(p)) else p


# ---------------------------------------------------------------- 에피소드 하나

def episode_metrics(res, lo, hi, ts, replayer, g=None, p_human=None, has_env_scene=False):
    """에피소드 하나 x 알고리즘 하나의 지표 (정의서 2절).

    res          4번 npz 내용 (q, status, saturated, pos_err)
    ts           (T,) exo_front 프레임 시각 s (저크 Δt용)
    replayer     replay.replay.Replayer (재생 + 로봇 몸체 목록)
    g            파지 프레임 배열 인덱스 = grasp_frame − exo_start_frame. None이면 파지점 거리 NaN
    p_human      로봇 좌표로 옮긴 사람 파지점 (3,). None이면 파지점 거리 NaN
    has_env_scene  재생 장면에 테이블·장애물이 있는지. 없으면 환경 지표를 0이 아니라 NaN으로 둔다
    반환: (지표 dict, 정보 dict)
    """
    q = np.asarray(res["q"], float)
    T = len(q)
    for key, shape in [("status", (T,)), ("saturated", (T, 7)), ("pos_err", (T,))]:
        if np.shape(res[key]) != shape:
            raise ValueError(f"npz {key} 형태 {np.shape(res[key])} ≠ {shape}")
    if len(ts) != T:
        raise ValueError(f"exo_ts 길이 {len(ts)} ≠ 궤적 길이 {T}")

    out = replayer.replay(q)  # gripper는 쓰지 않음 (그리퍼 없음, 11/3)
    self_d, env_d = contact_depths(out["contacts"], replayer.robot_bodies)
    j, k, dt = jerk_rms(q, ts)

    m = {
        "limit_violation_rate": limit_violation_rate(q, lo, hi),
        "limit_saturation_rate": limit_saturation_rate(res["saturated"]),
        "self_collision_rate": collision_rate(self_d),
        "max_penetration_m": max_penetration(self_d),
        **ik_rates(res["status"]),
        "jerk_rms": j,
        "grasp_dist_m": float("nan"),
        "track_pos_err_mean_m": float(np.mean(res["pos_err"])),
        "env_collision_rate": collision_rate(env_d) if has_env_scene else float("nan"),
        "env_max_penetration_m": max_penetration(env_d) if has_env_scene else float("nan"),
    }
    info = {"frames": T, "jerk_step_k": k, "jerk_dt_s": dt,
            "jerk_dt_flag": bool(np.isfinite(dt) and abs(dt * JERK_HZ - 1) > JERK_DT_TOL),
            "grasp_note": "", "env_scene": has_env_scene}
    if g is None:
        info["grasp_note"] = "grasp_frame 없음"
    elif not 0 <= g < T:
        raise ValueError(f"파지 프레임 인덱스 g={g}가 구간 0~{T - 1} 밖 (grasp_frame·exo_start_frame 확인)")
    elif p_human is None:
        info["grasp_note"] = "파지 프레임 엄지 끝·검지 끝 3D NaN"
    else:
        m["grasp_dist_m"] = float(np.linalg.norm(out["tcp_pos"][g] - np.asarray(p_human)))
    return m, info


# ---------------------------------------------------------------- 전 에피소드 실행

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_ts(path):
    """3번 exo_ts.csv (헤더 t_s) → (T,) s."""
    with open(path, newline="", encoding="utf-8") as f:
        return np.array([float(r["t_s"]) for r in csv.DictReader(f)])


def unusable_episodes(sync_csv):
    """sync.csv에서 카메라 행 중 하나라도 usable=0인 에피소드 (정의서 8절)."""
    bad = set()
    with open(sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r["usable"]).strip() != "1":
                bad.add(r["episode_id"])
    return bad


def load_to_robot_point(identity):
    """4번 retarget/common.py의 to_robot_point (테이블 좌표 → 로봇 베이스 좌표)."""
    if identity:
        print("[주의] --identity-transform: 테이블 좌표를 로봇 좌표로 그대로 씀 (시험용, 실제 결과에 쓰지 말 것)")
        return lambda p: np.asarray(p, float)
    import sys
    sys.path.insert(0, str(REPO / "retarget"))
    try:
        from common import to_robot_point  # 4번 제효정
    except ImportError as e:
        raise SystemExit(f"retarget/common.py의 to_robot_point를 못 읽음 ({e}). "
                         "4번 코드가 들어오기 전 시험이면 --identity-transform") from None
    return to_robot_point


def run(a):
    import sys
    sys.path.insert(0, str(REPO))
    from replay.replay import Replayer

    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = load_limits()
    replayer = Replayer()
    to_robot = load_to_robot_point(a.identity_transform)
    skip = unusable_episodes(a.sync) if a.sync else set()
    ret_root, ep_root = Path(a.retarget).expanduser(), Path(a.episodes).expanduser()
    # 알고리즘 = 4번 결과 폴더 아래 npz가 있는 하위 폴더 (이름은 4번이 정함: A_DLS, B_mink, C_… 등)
    found = sorted(p.name for p in ret_root.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    algos = [x for x in a.algos if x in found] if a.algos else found
    if not algos:
        raise SystemExit(f"{ret_root} 아래에 npz가 든 알고리즘 폴더가 없음")

    long_rows, info_rows = [], []
    for algo in algos:
        for npz in sorted((ret_root / algo).glob("*.npz")):
            ep = npz.stem
            base = {"algo": algo, "episode_id": ep}
            if a.ids and ep not in a.ids:
                continue
            if ep in skip:
                info_rows.append({**base, "status": "제외: sync.csv usable=0"})
                continue
            d = ep_root / ep
            try:
                res = dict(np.load(npz))
                meta, labels = read_json(d / "meta.json"), read_json(d / "labels.json")
                ts = read_ts(d / "exo_ts.csv")
                kp = np.load(d / "keypoints.npy")
                names = read_json(d / "keypoint_names.json")
                if len(kp) != len(res["q"]):
                    raise ValueError(f"keypoints 길이 {len(kp)} ≠ 궤적 길이 {len(res['q'])}")
                g = p_human = None
                if labels.get("grasp_frame") is not None:
                    g = int(labels["grasp_frame"]) - int(labels["exo_start_frame"])
                    if 0 <= g < len(kp):
                        p = human_grasp_point(kp, names, g)
                        p_human = None if p is None else to_robot(p)
                m, info = episode_metrics(res, lo, hi, ts, replayer, g, p_human,
                                          has_env_scene=a.env_scene)
            except Exception as e:  # 실패를 지우지 않음: 어떤 에피소드가 왜 안 됐는지 남긴다
                info_rows.append({**base, "status": f"오류: {e}"})
                print(f"[오류] {algo} {ep}: {e}")
                continue
            tags = meta.get("case_tags") or []
            for name in METRICS:
                long_rows.append({**base, "task": meta.get("task", ""),
                                  "case_tags": "|".join(tags), "metric": name, "value": m[name]})
            info_rows.append({**base, "status": "ok", **info})

    write_csv(out_dir / "metrics_long.csv", long_rows,
              ["algo", "episode_id", "task", "case_tags", "metric", "value"])
    write_csv(out_dir / "episode_info.csv", info_rows,
              ["algo", "episode_id", "status", "frames", "jerk_step_k", "jerk_dt_s",
               "jerk_dt_flag", "grasp_note", "env_scene"])
    n_ok = sum(r["status"] == "ok" for r in info_rows)
    print(f"알고리즘 {algos}, 계산 {n_ok}건, 제외·오류 {len(info_rows) - n_ok}건 → {out_dir}")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-2 지표 계산 (metrics/definitions.md v0.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="전 에피소드 x 알고리즘 지표 → metrics_long.csv")
    r.add_argument("--retarget", required=True, help="4번 결과 폴더 (<알고리즘>/<에피소드ID>.npz)")
    r.add_argument("--episodes", required=True, help="3번 결과 폴더 (<에피소드ID>/...)")
    r.add_argument("--sync", default=str(REPO / "sync" / "sync.csv"), help="usable=0 에피소드 제외용")
    r.add_argument("--out", required=True, help="출력 폴더 (~/project 밖, 예: ~/out/신서연/metrics)")
    r.add_argument("--algos", nargs="*", help="계산할 알고리즘 폴더 이름 (기본: --retarget 아래 있는 것 모두)")
    r.add_argument("--ids", nargs="*", help="이 에피소드만 (시험용)")
    r.add_argument("--env-scene", action="store_true",
                   help="재생 장면에 테이블·장애물이 있을 때만. 없으면 환경 지표는 NaN")
    r.add_argument("--identity-transform", action="store_true",
                   help="to_robot_point 대신 항등 변환 (4번 코드 전 시험용)")
    run(ap.parse_args())
