"""5-2 지표 계산: 리타게팅 결과(관절 궤적)를 재생해 정의서의 지표를 계산한다.

정의는 metrics/definitions.md(v0.4)가 기준이다. 이 코드와 정의서가 다르면 정의서를 따르고 코드를 고친다.
모든 지표는 에피소드 하나 x 알고리즘 하나 단위이고, 낮을수록 좋다 (성공률은 실패율 = 1 − 성공률로 저장).

GPU는 필요 없다 (렌더링 없이 mj_forward만). 서버에서는 source ~/use_robot.sh 후 srun (--gres 없이) CPU 노드에서.

사용 (레포 루트에서, mujoco가 있는 환경):
    python metrics/metrics.py run --retarget <4번 결과 폴더> --episodes <3번 결과 폴더> \
        --obstacles episodes/obstacles.json --out ~/out/신서연/metrics      # 테이블 윤곽은 metrics/table_outline.json
    python metrics/selftest.py --out ~/out/신서연/metrics_selftest      # 합성 데이터 검증

폴더 구조 (각 담당 문서 기준):
    <retarget>/<알고리즘>/<에피소드ID>.npz    q (T,7), target_pos (T,3), target_quat (T,4 wxyz) (손목 미검출 = NaN),
                                              time_ms (T,), info (JSON: 좌표 변환 값) [, pos_err, rot_err]   ← 4번
    <episodes>/<에피소드ID>/                   keypoints.npy (T,7,3), keypoint_names.json, exo_ts.csv,
                                              meta.json, labels.json                                ← 3번
"""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LIMITS_CSV = REPO / "robot" / "joint_limits.csv"
TABLE_OUTLINE = REPO / "metrics" / "table_outline.json"
URDF = REPO / "robot" / "robot.urdf"
EPS_LIMIT = 1e-6                 # 한계 초과로 보지 않는 수치 오차 (rad)
SUCCESS_POS_M = 0.01             # 2.1 성공: 위치 오차 1 cm 이하
SUCCESS_ROT_RAD = np.deg2rad(5)  # 2.1 성공: 방향 오차 5° 이하
BUDGET_MS = 1000.0 / 30.0        # 2.3 실시간 기준: 30 Hz 한 프레임 = 33.3 ms
PROX_RAD = np.deg2rad(1.0)       # 2.11 한계 근접으로 보는 거리 δ = 1°
RESAMPLE_HZ = 30.0               # 2.4·2.5 저크·속도는 이 주기로 맞춰 계산
DT_TOL = 1e-3                    # 맞춘 Δt가 1/30 s와 이 비율 넘게 다르면 표시
SEW_VERTICAL_COS = 0.99          # 2.2 어깨–손목 축이 거의 수직이면 팔 회전각을 정의할 수 없어 그 프레임 제외
ROBOT_SEW_BODIES = ("link2", "link4", "link6")   # 로봇 어깨·팔꿈치·손목 = 관절 2·4·6 중심 (iiwa SEW 규약)

# 묶음은 definitions.md 7절. 표·집계에서 이 순서로 쓴다
METRICS = [
    # 정확도
    "retarget_fail_rate", "arm_angle_diff_rad",
    # 속도
    "compute_ms_median", "compute_ms_p95", "over_budget_rate",
    # 궤적 품질
    "jerk_rms", "velocity_limit_exceed_rate", "self_collision_rate", "env_collision_rate",
    # 안전 확인
    "limit_violation_rate",
    # 세부·참고
    "env_obstacle_rate", "env_elbow_below_table_rate",
    "track_pos_err_mean_m", "track_pos_err_max_m", "track_rot_err_mean_rad", "track_rot_err_max_rad",
    "limit_proximity_rate", "grasp_dist_m", "min_manipulability",
]
RANKED = {
    "정확도": ["retarget_fail_rate"],      # arm_angle_diff_rad는 참고 (세 알고리즘 모두 팔꿈치 입력, T10)
    "속도": ["compute_ms_median", "compute_ms_p95"],
    "궤적 품질": ["jerk_rms", "velocity_limit_exceed_rate", "self_collision_rate", "env_collision_rate"],
}


# ---------------------------------------------------------------- 로봇 한계 (1번 파일)

def load_limits(path=LIMITS_CSV, urdf=URDF):
    """관절 하한·상한 (joint_limits.csv)과 속도 한계 (robot.urdf <limit velocity>) → lo, hi, vmax (7,)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["joint"] for r in rows]
    if names != [f"joint{i}" for i in range(1, 8)]:
        raise ValueError(f"joint_limits.csv 관절 순서가 joint1~7이 아님: {names}")
    lo = np.array([float(r["lower_rad"]) for r in rows])
    hi = np.array([float(r["upper_rad"]) for r in rows])
    # <transmission> 안의 같은 이름 <joint>는 빼고 실제 관절(type 있음)만
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint") if j.get("type")}
    vmax = np.array([float(joints[r["urdf_joint"]].find("limit").get("velocity")) for r in rows])
    return lo, hi, vmax


# ---------------------------------------------------------------- 정확도

def quat_angle(q1, q2):
    """쿼터니언 (w,x,y,z) 두 개 사이 회전각 (rad), (T,). q와 −q는 같은 방향."""
    q1, q2 = np.atleast_2d(q1), np.atleast_2d(q2)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0.0, 1.0))


def in_limits(q, lo, hi, eps=EPS_LIMIT):
    """프레임마다 모든 관절이 한계 안인지 (T,) bool."""
    return ~((q < lo - eps) | (q > hi + eps)).any(axis=1)


def tracking_errors(tcp_pos, tcp_quat, target_pos, target_quat):
    """재생한 TCP와 목표 손목 자세의 위치 오차 (m), 방향 오차 (rad). 모든 알고리즘에 같은 계산."""
    return (np.linalg.norm(np.asarray(tcp_pos) - np.asarray(target_pos), axis=1),
            quat_angle(tcp_quat, target_quat))


def retarget_fail_rate(pos_err, rot_err, ok_limits):
    """2.1 성공 = 위치 ≤ 1 cm, 방향 ≤ 5°, 관절 한계 안. 실패율 = 1 − 성공 프레임 비율.

    목표가 NaN인 프레임(손목 미검출)은 판정할 수 없어 분모에서 뺀다. 반환 (실패율, 목표 없는 프레임 비율).
    """
    valid = np.isfinite(pos_err) & np.isfinite(rot_err)
    if not valid.any():
        return float("nan"), 1.0
    success = (pos_err[valid] <= SUCCESS_POS_M) & (rot_err[valid] <= SUCCESS_ROT_RAD) & ok_limits[valid]
    return float(1.0 - success.mean()), float(1.0 - valid.mean())


def sew_angle(S, E, W, down=(0.0, 0.0, -1.0)):
    """2.2 팔 회전각 ψ (rad): 어깨–손목 축 둘레로 팔꿈치가 '아래 방향'에서 돌아간 각. (T,) 정의 안 되면 NaN."""
    S, E, W = (np.atleast_2d(np.asarray(x, float)) for x in (S, E, W))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = W - S
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        d = np.broadcast_to(np.asarray(down, float), a.shape)
        perp = lambda v: v - np.sum(v * a, axis=1, keepdims=True) * a
        e, r = perp(E - S), perp(d)
        psi = np.arctan2(np.sum(a * np.cross(r, e), axis=1), np.sum(r * e, axis=1))
        bad = (np.abs(np.sum(a * d, axis=1)) > SEW_VERTICAL_COS) | (np.linalg.norm(e, axis=1) < 1e-6)
    psi[bad | ~np.all(np.isfinite(S + E + W), axis=1)] = np.nan
    return psi


def arm_angle_diff(psi_robot, psi_human):
    """2.2 프레임별 |ψ_로봇 − ψ_사람| (2π 주기)의 중앙값 (rad). 둘 다 정의된 프레임만. 반환 (값, 쓴 프레임 비율)."""
    ok = np.isfinite(psi_robot) & np.isfinite(psi_human)
    if not ok.any():
        return float("nan"), 0.0
    d = np.abs(np.angle(np.exp(1j * (psi_robot[ok] - psi_human[ok]))))
    return float(np.median(d)), float(ok.mean())


# ---------------------------------------------------------------- 속도

def compute_time(time_ms, budget=BUDGET_MS):
    """2.3 첫 프레임(준비 시간)을 뺀 계산 시간의 중앙값·95% 값 (ms)과 실시간 기준을 넘은 프레임 비율."""
    t = np.asarray(time_ms, float)[1:]
    if not len(t):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(t)), float(np.percentile(t, 95)), float((t > budget).mean())


# ---------------------------------------------------------------- 궤적 품질

def resample(q, ts, hz=RESAMPLE_HZ):
    """30 Hz로 맞춘 궤적 (보간·스무딩 없이 k칸마다 고름). 반환 (q_s, dt, k). 60 fps면 k=2, 30 fps면 k=1."""
    q, ts = np.asarray(q, float), np.asarray(ts, float)
    k = max(1, int(round((1.0 / hz) / np.median(np.diff(ts)))))
    q_s, ts_s = q[::k], ts[::k]
    dt = float(np.median(np.diff(ts_s))) if len(ts_s) > 1 else float("nan")
    return q_s, dt, k


def jerk_rms(q, ts):
    """2.4 30 Hz로 맞춘 궤적의 저크 RMS (rad/s³). 반환 (jerk_rms, k, dt)."""
    q_s, dt, k = resample(q, ts)
    if len(q_s) < 4:
        return float("nan"), k, dt
    jerk = np.diff(q_s, n=3, axis=0) / dt ** 3
    return float(np.sqrt(np.mean(jerk ** 2))), k, dt


def velocity_limit_exceed_rate(q, ts, vmax):
    """2.5 30 Hz로 맞춘 궤적에서 한 관절이라도 |Δq/Δt|가 속도 한계를 넘은 구간의 비율."""
    q_s, dt, _ = resample(q, ts)
    if len(q_s) < 2:
        return float("nan")
    return float((np.abs(np.diff(q_s, axis=0)) / dt > vmax).any(axis=1).mean())


def contact_depths(contacts, robot_bodies):
    """프레임별 가장 깊은 자기 접촉·환경 접촉 거리 (접촉 없으면 0). 음수 = 파고듦.

    자기 접촉 = 두 몸체가 모두 로봇 몸체 (replay.is_self_contact와 같은 기준), 나머지는 환경(장애물) 접촉.
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


def table_frame_inverse(to_robot):
    """to_robot_point(테이블 → 로봇)의 역변환 (로봇 → 테이블). 평행 이동·축척·회전이라 점 4개로 정해진다."""
    o = np.asarray(to_robot(np.zeros(3)), float)
    M = np.column_stack([np.asarray(to_robot(e), float) - o for e in np.eye(3)])
    Minv = np.linalg.inv(M)
    return lambda P: (np.atleast_2d(P) - o) @ Minv.T


def inside_table(P_table, outline):
    """테이블 좌표 점들이 테이블 윗면 윤곽 타원 안(테이블 위 영역)인지 (T,). outline = table_outline.json 항목."""
    a = np.radians(outline["angle_deg"])
    axes = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = (np.atleast_2d(P_table)[:, :2] - np.asarray(outline["center"])) @ axes
    r = np.asarray(outline["radii"])
    return (q[:, 0] / r[0]) ** 2 + (q[:, 1] / r[1]) ** 2 <= 1.0


def elbow_below_table(E_robot, to_table, outline):
    """2.6② 로봇 팔꿈치(관절 4 중심)가 테이블 위 영역(윤곽 타원 안)에서 테이블 면(z = 0)보다 아래인지 (T,)."""
    P = to_table(E_robot)
    return (P[:, 2] < 0.0) & inside_table(P, outline)


# ---------------------------------------------------------------- 참고

def limit_proximity_rate(q, lo, hi, delta=PROX_RAD):
    """2.11 한 관절이라도 한계에서 delta 안에 있는(넘은 것 포함) 프레임 비율. 출력 각도만 본다."""
    return float(((q < lo + delta) | (q > hi - delta)).any(axis=1).mean())


def human_grasp_point(kp, names, g):
    """2.12 파지 프레임 g의 엄지 끝·검지 끝 중점 (테이블 좌표, m). 하나라도 NaN이면 None."""
    i_th, i_ix = names.index("thumb_tip"), names.index("index_tip")
    p = (kp[g, i_th] + kp[g, i_ix]) / 2.0
    return None if not np.all(np.isfinite(p)) else p


def robot_kinematics(replayer, q):
    """프레임별 로봇 어깨·팔꿈치·손목 위치 (T,3)x3 와 TCP 조작성 (T,) — 2.2, 2.6②, 2.13."""
    import mujoco
    m, d = replayer.m, replayer.d
    ids = [m.body(n).id for n in ROBOT_SEW_BODIES]
    T = len(q)
    P = np.zeros((3, T, 3))
    w = np.zeros(T)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    cols = [m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
    for t in range(T):
        d.qpos[:] = replayer.q_full_home
        d.qpos[replayer.arm_adr] = q[t]
        mujoco.mj_forward(m, d)
        for k, i in enumerate(ids):
            P[k, t] = d.xpos[i]
        mujoco.mj_jacSite(m, d, jp, jr, replayer.tcp_id)
        J = np.vstack([jp, jr])[:, cols]
        w[t] = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))
    return P[0], P[1], P[2], w


# ---------------------------------------------------------------- 에피소드 하나

def episode_metrics(res, lo, hi, vmax, ts, replayer, g=None, p_human=None, human_sew=None,
                    has_obstacles=False, table=None):
    """에피소드 하나 x 알고리즘 하나의 지표 (정의서 2절).

    res            4번 npz 내용. q 필수 / target_pos·target_quat(성공률·추종 오차), time_ms(속도)는 있으면.
                   status(알고리즘이 보고한 성공·실패)는 쓰지 않음 — 성공 판정은 2.1의 공통 기준으로 직접 함
    ts             (T,) exo_front 프레임 시각 s (저크·속도 Δt용)
    replayer       replay.replay.Replayer. 장애물이 있으면 그 상자를 넣어 만든 것 (replay/scene.py)
    g, p_human     파지 프레임 인덱스, 로봇 좌표 사람 파지점. 없으면 파지점 거리 NaN
    human_sew      로봇 좌표로 옮긴 사람 (어깨, 팔꿈치, 손목) 각 (T,3). 없으면 팔 회전각 차이 NaN
    has_obstacles  장애물 정보를 확인했는지 (obstacles.json을 읽음). 거짓이면 장애물 부분 NaN
    table          (to_robot_point, 테이블 윤곽 타원 dict). 없으면 팔꿈치–테이블 부분 NaN
    반환: (지표 dict, 정보 dict)
    """
    q = np.asarray(res["q"], float)
    T = len(q)
    shapes = {"time_ms": (T,), "pos_err": (T,), "rot_err": (T,),
              "target_pos": (T, 3), "target_quat": (T, 4)}
    for key, shape in shapes.items():
        if key in res and np.shape(res[key]) != shape:
            raise ValueError(f"npz {key} 형태 {np.shape(res[key])} ≠ {shape}")
    if len(ts) != T:
        raise ValueError(f"exo_ts 길이 {len(ts)} ≠ 궤적 길이 {T}")
    nan = float("nan")

    out = replayer.replay(q)  # gripper는 쓰지 않음 (그리퍼 없음, 11/3)
    self_d, obs_d = contact_depths(out["contacts"], replayer.robot_bodies)
    S, E, W, manip = robot_kinematics(replayer, q)
    j, k, dt = jerk_rms(q, ts)
    ok_lim = in_limits(q, lo, hi)

    # 정확도: 목표가 npz에 있으면 재생 결과로 직접 계산 (모든 알고리즘에 같은 계산)
    if "target_pos" in res and "target_quat" in res:
        pe, re = tracking_errors(out["tcp_pos"], out["tcp_quat"], res["target_pos"], res["target_quat"])
        (fail, missing), err_src = retarget_fail_rate(pe, re, ok_lim), "재생"
    else:
        pe = np.asarray(res["pos_err"], float) if "pos_err" in res else None
        re = np.asarray(res["rot_err"], float) if "rot_err" in res else None
        fail, missing = nan, nan
        err_src = "4번 보고값 (목표 없음 → 성공률 계산 안 함)" if pe is not None else "없음"
    # 추종 오차는 목표가 있는 프레임만 (NaN 제외). 전부 NaN이면 NaN
    stat = lambda x, f: float(f(x[np.isfinite(x)])) if x is not None and np.isfinite(x).any() else nan

    ms_med, ms_p95, over = compute_time(res["time_ms"]) if "time_ms" in res else (nan, nan, nan)

    obstacle = obs_d < 0 if has_obstacles else None
    below = elbow_below_table(E, table_frame_inverse(table[0]), table[1]) if table else None
    env = (obstacle | below) if (obstacle is not None and below is not None) else None

    m = {
        "retarget_fail_rate": fail,
        "arm_angle_diff_rad": nan,
        "compute_ms_median": ms_med, "compute_ms_p95": ms_p95, "over_budget_rate": over,
        "jerk_rms": j,
        "velocity_limit_exceed_rate": velocity_limit_exceed_rate(q, ts, vmax),
        "self_collision_rate": float((self_d < 0).mean()),
        "env_collision_rate": float(env.mean()) if env is not None else nan,
        "limit_violation_rate": float((~ok_lim).mean()),
        "env_obstacle_rate": float(obstacle.mean()) if obstacle is not None else nan,
        "env_elbow_below_table_rate": float(below.mean()) if below is not None else nan,
        "track_pos_err_mean_m": stat(pe, np.mean), "track_pos_err_max_m": stat(pe, np.max),
        "track_rot_err_mean_rad": stat(re, np.mean), "track_rot_err_max_rad": stat(re, np.max),
        "limit_proximity_rate": limit_proximity_rate(q, lo, hi),
        "grasp_dist_m": nan,
        "min_manipulability": float(manip.min()),
    }
    info = {"frames": T, "resample_k": k, "dt_s": dt,
            "dt_flag": bool(np.isfinite(dt) and abs(dt * RESAMPLE_HZ - 1) > DT_TOL),
            "error_source": err_src, "target_missing_rate": missing, "grasp_note": "", "arm_angle_frames": nan,
            "obstacles": has_obstacles, "table": table is not None}
    if human_sew is not None:
        m["arm_angle_diff_rad"], info["arm_angle_frames"] = arm_angle_diff(sew_angle(S, E, W), sew_angle(*human_sew))
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


def load_transform(identity):
    """4번 retarget/common.py의 to_robot_point(p, info) (테이블 좌표 → 로봇 베이스 좌표).

    변환 값(어깨 위치, 팔 길이 비율, 작업 손, 로봇 자리)은 에피소드마다 달라서 4번 npz의 info(JSON)에 들어 있고,
    그 info를 함수에 같이 넘긴다 (4번 답 T9, 2026-11-05). 반환: f(p, info) → (3,)
    """
    if identity:
        print("[주의] --identity-transform: 테이블 좌표를 로봇 좌표로 그대로 씀 (시험용, 실제 결과에 쓰지 말 것)")
        return lambda p, info: np.asarray(p, float)
    import sys
    sys.path.insert(0, str(REPO / "retarget"))
    try:
        from common import to_robot_point  # 4번 제효정
    except ImportError as e:
        raise SystemExit(f"retarget/common.py의 to_robot_point를 못 읽음 ({e}). "
                         "4번 코드가 들어오기 전 시험이면 --identity-transform") from None
    return to_robot_point


def read_info(res, identity):
    """npz의 info(JSON 문자열) → (dict, 비교용 문자열). 항등 변환 시험이면 없어도 됨."""
    if "info" not in res:
        if identity:
            return {}, ""
        raise ValueError("npz에 info(좌표 변환 값)가 없음 — 4번 save_result 확인")
    raw = res["info"]
    raw = raw.item() if hasattr(raw, "item") else raw
    raw = raw.decode() if isinstance(raw, bytes) else str(raw)
    info = json.loads(raw)
    return info, json.dumps(info, sort_keys=True)


def map_points(to_robot, P):
    """(T,3) 테이블 좌표 점들 → 로봇 좌표. NaN 프레임은 NaN 그대로."""
    out = np.full_like(P, np.nan, dtype=float)
    for t, p in enumerate(P):
        if np.all(np.isfinite(p)):
            out[t] = to_robot(p)
    return out


def run(a):
    import sys
    sys.path.insert(0, str(REPO))
    from replay.replay import Replayer
    from replay.scene import box_corners, load_obstacles, map_box

    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi, vmax = load_limits()
    base_replayer = Replayer()
    transform = load_transform(a.identity_transform)
    obstacles = load_obstacles(a.obstacles) if a.obstacles else None
    outlines = json.load(open(a.table_outline, encoding="utf-8")) if a.table_outline else {}
    missing = [k for k in ("T1", "T2", "T3") if not outlines.get(k)]
    if missing:
        print(f"[주의] 테이블 윤곽이 없는 태스크 {missing}: 환경 충돌률의 팔꿈치–테이블 부분과 환경 충돌률은 NaN")
    if obstacles is None:
        print("[주의] --obstacles가 없어 환경 충돌률의 장애물 부분과 환경 충돌률은 NaN")
    skip = unusable_episodes(a.sync) if a.sync else set()
    ret_root, ep_root = Path(a.retarget).expanduser(), Path(a.episodes).expanduser()
    # 알고리즘 = 4번 결과 폴더 아래 npz가 있는 하위 폴더 (이름은 4번이 정함: A_DLS, B_mink, C_… 등)
    found = sorted(p.name for p in ret_root.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    algos = [x for x in a.algos if x in found] if a.algos else found
    if not algos:
        raise SystemExit(f"{ret_root} 아래에 npz가 든 알고리즘 폴더가 없음")

    scenes = {}      # 에피소드마다 장애물 장면 (알고리즘끼리 같은 장면을 씀)
    ep_info = {}     # 에피소드마다 좌표 변환 값. 같은 에피소드는 알고리즘끼리 같아야 함
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
                info, info_key = read_info(res, a.identity_transform)
                if ep_info.setdefault(ep, info_key) != info_key:
                    raise ValueError("같은 에피소드인데 알고리즘마다 npz info(좌표 변환 값)가 다름 — 4번 확인")
                to_robot = lambda p, _info=info: np.asarray(transform(p, _info), float)
                if obstacles is not None and ep not in scenes:
                    # 장애물 상자만 장면에 넣는다 (테이블은 상자가 아니라 면으로 따로 판정, 2.6②)
                    boxes = [map_box(f"env_{o.get('name') or f'obs{i}'}", box_corners(o["center"], o["size"], o["yaw_deg"]),
                                     to_robot) for i, o in enumerate((obstacles.get(ep) or {}).get("obstacles", []), 1)]
                    scenes[ep] = Replayer(env_boxes=boxes) if boxes else base_replayer
                replayer = scenes.get(ep, base_replayer)
                human_sew = tuple(map_points(to_robot, kp[:, names.index(n)])
                                  for n in ("shoulder", "elbow", "wrist"))
                g = p_human = None
                if labels.get("grasp_frame") is not None:
                    g = int(labels["grasp_frame"]) - int(labels["exo_start_frame"])
                    if 0 <= g < len(kp):
                        p = human_grasp_point(kp, names, g)
                        p_human = None if p is None else to_robot(p)
                outline = outlines.get(ep.split("_")[0])
                table = (to_robot, outline) if outline else None
                m, info = episode_metrics(res, lo, hi, vmax, ts, replayer, g, p_human, human_sew,
                                          has_obstacles=obstacles is not None, table=table)
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
              ["algo", "episode_id", "status", "frames", "resample_k", "dt_s", "dt_flag", "error_source",
               "grasp_note", "arm_angle_frames", "obstacles", "table"])
    n_ok = sum(r["status"] == "ok" for r in info_rows)
    print(f"알고리즘 {algos}, 계산 {n_ok}건, 제외·오류 {len(info_rows) - n_ok}건 → {out_dir}")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-2 지표 계산 (metrics/definitions.md v0.4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="전 에피소드 x 알고리즘 지표 → metrics_long.csv")
    r.add_argument("--retarget", required=True, help="4번 결과 폴더 (<알고리즘>/<에피소드ID>.npz)")
    r.add_argument("--episodes", required=True, help="3번 결과 폴더 (<에피소드ID>/...)")
    r.add_argument("--sync", default=str(REPO / "sync" / "sync.csv"), help="usable=0 에피소드 제외용")
    r.add_argument("--obstacles", help="3번 obstacles.json (환경 충돌률의 장애물 부분)")
    r.add_argument("--table-outline", default=str(TABLE_OUTLINE),
                   help="태스크별 테이블 윗면 윤곽 타원 (환경 충돌률의 팔꿈치–테이블 부분, metrics/table_outline.py로 구함)")
    r.add_argument("--out", required=True, help="출력 폴더 (~/project 밖, 예: ~/out/신서연/metrics)")
    r.add_argument("--algos", nargs="*", help="계산할 알고리즘 폴더 이름 (기본: --retarget 아래 있는 것 모두)")
    r.add_argument("--ids", nargs="*", help="이 에피소드만 (시험용)")
    r.add_argument("--identity-transform", action="store_true",
                   help="to_robot_point 대신 항등 변환 (4번 코드 전 시험용)")
    run(ap.parse_args())
