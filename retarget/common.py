"""4번 리타게팅 공용 함수: 에피소드 읽기, 테이블 좌표 -> 로봇 베이스 좌표, 결과 저장.

세 알고리즘(A_DLS, B_mink, C_eaik)이 모두 이 파일의 목표(target)를 쓴다. 알고리즘 사이 차이는 IK 풀이 방식뿐이어야 한다.

좌표 규칙 (시험용 초안, 멘토 확인 전이라 인자로 바꿀 수 있게 둠):
- 로봇 어깨(베이스 위 0.36 m)를 에피소드 평균 사람 어깨 위치에 둔다. 에피소드마다 고정 (프레임마다 움직이지 않음)
- 로봇 x = 사람이 손을 뻗는 쪽(테이블 -y), 로봇 z = 위(테이블 z), 로봇 y = 테이블 +x
- scale: 1.0 = 1:1. "arm" = iiwa 팔 길이(0.82 m) / 사람 팔 길이(위팔+아래팔, 에피소드 중앙값). 어깨 기준으로 늘림
- 손목 방향: 3번 wrist_pose (x = 손목->검지뿌리, z = x × (손목->새끼뿌리)).
  오른손은 z가 손바닥 밖, 왼손은 손등 밖을 향하므로 왼손은 x축 둘레 180° 돌려(diag(1,-1,-1)) 오른손과 같은 뜻으로 맞춘다
- TCP(플랜지) 목표 방향 = 손목 방향 @ R_OFF. 기본 R_OFF = 단위행렬 -> 플랜지 z = 손바닥 밖, 플랜지 x = 손가락 쪽

단위 m, rad. 쿼터니언 저장 순서 (w, x, y, z).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

IIWA_SHOULDER = np.array([0.0, 0.0, 0.36])       # 베이스 -> 어깨 (robot/model_checks.md)
L_UPPER, L_FORE = 0.42, 0.40                     # iiwa 어깨->팔꿈치, 팔꿈치->손목 중심 (m)
IIWA_ARM = L_UPPER + L_FORE
Q_HOME = np.array([0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0])   # robot.xml keyframe "home"

# 테이블 좌표 -> 로봇 베이스 좌표 회전. p_base = R @ p_table
R_TABLE_TO_BASE = np.array([[0.0, -1.0, 0.0],    # 로봇 x = 테이블 -y (손 뻗는 쪽)
                            [1.0,  0.0, 0.0],    # 로봇 y = 테이블 +x
                            [0.0,  0.0, 1.0]])   # 로봇 z = 테이블 z (위)
LEFT_FIX = np.diag([1.0, -1.0, -1.0])            # 왼손 손목 축 -> 오른손과 같은 뜻
# 손목 축 -> 플랜지 축 (열 = 플랜지 x, y, z를 손목 축으로 쓴 것). 시험용 두 가지
R_OFF_OPTIONS = {
    "palm": np.eye(3),                            # 플랜지 z = 손바닥 밖, x = 손가락 쪽
    "finger": np.array([[0.0, 0.0, 1.0],          # 플랜지 z = 손가락 쪽(아래팔 연장), x = 손바닥 밖
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0]]),
}
R_OFF = R_OFF_OPTIONS["palm"]

STATUS_OK, STATUS_MAXITER, STATUS_NO_TARGET = 0, 1, 2


def load_episode(ep_dir):
    """3번 episodes/<ID>/ 폴더를 읽는다 (episodes/DATA_STRUCTURE.md 4절).

    keypoint_source.npy (T, 7) 가 있으면 같이 읽는다: 키포인트별 "auto" / "manual"(사람이 직접 찍음) / "none".
    없으면 None (모두 자동 추출로 본다)."""
    d = Path(ep_dir).expanduser()
    names = json.loads((d / "keypoint_names.json").read_text())
    src_file = d / "keypoint_source.npy"
    return {
        "dir": d,
        "meta": json.loads((d / "meta.json").read_text()),
        "names": names,
        "kp": np.load(d / "keypoints.npy"),          # (T, 7, 3) 테이블 좌표
        "wrist_pose": np.load(d / "wrist_pose.npy"),  # (T, 7) xyz + (w, x, y, z)
        "kp_source": np.load(src_file).astype(str) if src_file.exists() else None,
    }


POS_ONLY_MODES = ("hold", "skip")


def fill_position_only(pos, quat, mode="hold"):
    """손목 위치는 있는데 방향이 없는 프레임(예: 사람이 손목 위치만 찍은 프레임) 처리.

    hold: 방향을 직전에 방향이 있던 프레임 값으로 유지한다 (보간하지 않음, 앞쪽에 방향이 없으면 그대로 NaN)
    skip: 아무것도 채우지 않는다 -> 목표 없음 (status 2, 직전 자세 유지)
    반환 (quat 채운 것, orient_held (T,) bool = 방향을 유지해서 채운 프레임)
    """
    quat = quat.copy()
    held = np.zeros(len(quat), bool)
    if mode == "skip":
        return quat, held
    if mode != "hold":
        raise ValueError(f"pos_only 는 {POS_ONLY_MODES} 중 하나")
    last = None
    for t in range(len(quat)):
        if np.isfinite(quat[t]).all():
            last = quat[t]
        elif last is not None and np.isfinite(pos[t]).all():
            quat[t] = last
            held[t] = True
    return quat, held


def human_arm_length(kp, names):
    """위팔 + 아래팔 길이 (에피소드 중앙값, m)."""
    S, E, W = (kp[:, names.index(n)] for n in ("shoulder", "elbow", "wrist"))
    return float(np.nanmedian(np.linalg.norm(E - S, axis=1)) + np.nanmedian(np.linalg.norm(W - E, axis=1)))


ANCHORS = ("shoulder", "table")
MOUNTS = ("up", "side")


def to_robot_point(p_table, info):
    """테이블 좌표 점 -> 로봇 베이스 좌표 (m). 에피소드마다 다른 변환 값은 info에서 읽는다.

    p_table: (3,) 또는 (..., 3). info: make_targets가 만든 dict, 또는 npz에서 읽은 load_info(npz).
    리타게팅 목표와 똑같은 변환이다 (예: 5-2 파지점 거리에서 사람 손끝 -> 로봇 좌표).
        p_base = scale * (p_table - src_table) @ R_table_to_base.T + dst_base
    """
    p = np.asarray(p_table, dtype=float)
    R = np.asarray(info["R_table_to_base"], dtype=float)
    src = np.asarray(info["src_table"], dtype=float)
    dst = np.asarray(info["dst_base"], dtype=float)
    return (info["scale"] * (p - src)) @ R.T + dst


def task_score(pos_err, rot_err):
    """손목 목표를 얼마나 못 맞췄는지 한 숫자로 (m + 0.1 m/rad × rad). 재시작 결과 비교용. NaN이면 무한대."""
    s = pos_err + 0.1 * rot_err
    return s if np.isfinite(s) else np.inf


def to_table_point(p_base, info):
    """to_robot_point 의 역변환: 로봇 베이스 좌표 점 -> 테이블 좌표 (m). (예: 5-2 팔꿈치–테이블 판정)"""
    p = np.asarray(p_base, dtype=float)
    R = np.asarray(info["R_table_to_base"], dtype=float)
    src = np.asarray(info["src_table"], dtype=float)
    dst = np.asarray(info["dst_base"], dtype=float)
    return ((p - dst) @ R) / info["scale"] + src


def to_robot_rot(R_table, info):
    """테이블 좌표 회전행렬 (3,3) 또는 (...,3,3) -> 로봇 베이스 좌표. (위치와 달리 늘리기·옮기기 없음)"""
    return np.asarray(info["R_table_to_base"], dtype=float) @ np.asarray(R_table, dtype=float)


def load_info(npz):
    """리타게팅 결과 npz (경로 또는 np.load 결과) -> info dict."""
    d = np.load(Path(npz).expanduser()) if isinstance(npz, (str, Path)) else npz
    return json.loads(str(d["info"]))


def table_to_base_rotation(mount, hand):
    """테이블 좌표 -> 로봇 베이스 좌표 회전 (행 = 로봇 x, y, z 축을 테이블 좌표로 쓴 것).
    up:   로봇을 세워 둠. z = 위 (기본)
    side: 로봇을 옆으로 눕힘. z(1번 관절 축) = 작업 손 쪽 몸 바깥, x = 손 뻗는 쪽.
          사람 어깨처럼 2번 관절이 '옆 축 기준 120°'가 되어 위팔을 아래로 늘어뜨릴 수 있음.
          작업자는 테이블 -y 쪽을 보고 앉음 -> 오른쪽 = 테이블 -x, 왼쪽 = 테이블 +x"""
    if mount == "up":
        return R_TABLE_TO_BASE
    if mount == "side":
        x = np.array([0.0, -1.0, 0.0])
        z = np.array([-1.0, 0.0, 0.0]) if hand == "right" else np.array([1.0, 0.0, 0.0])
        return np.stack([x, np.cross(z, x), z])
    raise ValueError(f"mount는 {MOUNTS} 중 하나")


def make_targets(ep, scale=1.0, r_off="palm", anchor="shoulder", mount="up", base_z=0.0, pos_only="hold"):
    """에피소드 -> 로봇 베이스 좌표의 TCP 목표.

    mount: "up"(세워 둠) / "side"(옆으로 눕힘, anchor는 shoulder만)
    base_z: 로봇 베이스를 기준 위치보다 위(+)·아래(-)로 옮긴 거리 (m, 늘린 뒤 좌표). up일 때만
    pos_only: 손목 위치만 있고 방향이 없는 프레임 처리 (fill_position_only). 세 알고리즘에 똑같이 적용

    반환 dict:
      pos (T,3), rot (T,3,3)  목표 위치·방향. 못 구한 프레임은 NaN
      valid (T,)              목표가 있는 프레임
      elbow (T,3)             팔꿈치 (같은 변환). 팔 회전각 목표용, 없으면 NaN
      orient_held (T,)        방향을 직전 값으로 유지해 만든 목표
      wrist_manual (T,)       손목 위치가 사람이 직접 찍은 값 (keypoint_source 기준, 파일 없으면 모두 False)
      info                    변환 설정 (npz에 같이 저장)
    """
    kp, names, wp, meta = ep["kp"], ep["names"], ep["wrist_pose"], ep["meta"]
    s_mean = np.nanmean(kp[:, names.index("shoulder")], axis=0)
    arm = human_arm_length(kp, names)
    k = IIWA_ARM / arm if scale == "arm" else float(scale)
    R_tb = table_to_base_rotation(mount, meta.get("hand"))

    if mount == "side" and (anchor != "shoulder" or base_z != 0.0):
        raise ValueError("mount=side 는 anchor=shoulder, base_z=0 으로만 (로봇 어깨 = 사람 어깨)")
    if anchor == "shoulder":   # 사람 평균 어깨 -> 로봇 어깨. 테이블 면은 베이스보다 아래로 갈 수 있음
        src, dst = s_mean, IIWA_SHOULDER.copy()
    elif anchor == "table":    # 사람 어깨 바로 아래 테이블 점 -> 로봇 베이스. 테이블 면 = 베이스 면 (로봇을 테이블 위에 놓은 것)
        src, dst = np.array([s_mean[0], s_mean[1], 0.0]), np.zeros(3)
    else:
        raise ValueError(f"anchor는 {ANCHORS} 중 하나")
    dst = dst - np.array([0.0, 0.0, base_z])   # 베이스를 올리면 같은 점이 베이스 기준으로는 그만큼 낮아짐

    # 변환에 필요한 값은 전부 info에 넣는다. 5-2 등은 npz의 info만으로 to_robot_point를 쓸 수 있다
    info = {"scale": k, "scale_mode": str(scale), "human_arm_m": arm,
            "human_shoulder_table": s_mean.tolist(), "hand": meta.get("hand"),
            "exo_fps": meta.get("exo_fps"),   # 실시간 기준 프레임 간격 = 1000 / exo_fps ms (5-2 실시간 초과율)
            "R_table_to_base": R_tb.tolist(), "src_table": src.tolist(), "dst_base": dst.tolist(),
            "r_off": r_off, "anchor": anchor, "mount": mount, "base_z": base_z,
            # 베이스 기준 테이블 면 높이 (up일 때만 의미 있음. side는 베이스 z가 옆 방향이라 NaN)
            "table_z_in_base": float(k * (0.0 - src[2]) + dst[2]) if mount == "up" else float("nan"),
            "R_off": R_OFF_OPTIONS[r_off].tolist()}

    def to_base(p):  # (T,3) 테이블 -> 베이스
        return to_robot_point(p, info)

    # 손목 위치: wrist_pose 가 비어 있으면 keypoints 의 손목 (직접 찍은 위치가 keypoints 에만 있을 수 있음)
    wi = names.index("wrist")
    wrist_tab = np.where(np.isfinite(wp[:, :3]).all(1, keepdims=True), wp[:, :3], kp[:, wi])
    pos = to_base(wrist_tab)
    shoulder = to_base(kp[:, names.index("shoulder")])
    elbow = to_base(kp[:, names.index("elbow")])
    # 사람 팔꿈치 방향: 어깨-손목 축에 수직인 성분 (팔 회전각을 정하는 방향, 팔 길이와 무관)
    elbow_dir = np.full((len(wp), 3), np.nan)
    for t in range(len(wp)):
        n = elbow_direction(shoulder[t], elbow[t], pos[t])
        if n is not None:
            elbow_dir[t] = n
    # 맨 앞에서 팔꿈치가 안 보이는 프레임은 처음 보인 팔꿈치 방향을 쓴다 (손목의 '맨 앞 빈 프레임은 첫 해로' 규칙과 같은 원리).
    # 안 채우면 처음엔 home 쪽 자세로 풀다가 팔꿈치가 처음 보이는 프레임에서 관절이 한 번에 튐 (T1_E004 16->17: A 130°, B 93°, C 88°)
    elbow_backfilled = np.zeros(len(wp), bool)
    seen = np.flatnonzero(np.isfinite(elbow_dir).all(1))
    if len(seen) and seen[0] > 0:
        elbow_dir[:seen[0]] = elbow_dir[seen[0]]
        elbow_backfilled[:seen[0]] = True
    q, orient_held = fill_position_only(pos, wp[:, 3:], pos_only)
    valid = np.isfinite(pos).all(1) & np.isfinite(q).all(1)
    src_arr = ep.get("kp_source")
    wrist_manual = (src_arr[:, wi] == "manual") if src_arr is not None else np.zeros(len(wp), bool)
    elbow_backfilled &= valid
    info.update({"n_elbow_backfilled": int(elbow_backfilled.sum()),
                 "pos_only": pos_only, "n_orient_held": int(orient_held.sum()),
                 "n_wrist_manual": int(wrist_manual.sum()), "has_keypoint_source": src_arr is not None})
    rot = np.full((len(wp), 3, 3), np.nan)
    if valid.any():
        R_hand = Rot.from_quat(np.roll(q[valid], -1, axis=1)).as_matrix()   # (w,x,y,z) -> scipy (x,y,z,w)
        if meta.get("hand") == "left":
            R_hand = R_hand @ LEFT_FIX
        rot[valid] = R_tb @ R_hand @ R_OFF_OPTIONS[r_off]
    return {"pos": pos, "rot": rot, "valid": valid, "shoulder": shoulder, "elbow": elbow,
            "elbow_dir": elbow_dir, "elbow_backfilled": elbow_backfilled,
            "orient_held": orient_held, "wrist_manual": wrist_manual, "info": info}


# ---------------------------------------------------------------- 팔꿈치 (팔 회전각)
def _perp_unit(v, u):
    """v에서 단위벡터 u 방향 성분을 뺀 단위벡터. 거의 0이거나 NaN이면 None."""
    p = v - np.dot(v, u) * u
    n = np.linalg.norm(p)
    return p / n if np.isfinite(n) and n > 1e-6 else None


def elbow_direction(S, E, W):
    """어깨 S, 팔꿈치 E, 손목 W -> 어깨-손목 축에 수직인 팔꿈치 방향 (단위벡터). 팔이 곧거나 NaN이면 None."""
    a = W - S
    d = np.linalg.norm(a)
    if not np.isfinite(d) or d < 1e-6 or not np.isfinite(E).all():
        return None
    return _perp_unit(E - S, a / d)


def robot_elbow_target(S_r, W_r, n_h):
    """로봇 어깨 S_r, 손목 중심 W_r가 정해졌을 때, 팔꿈치가 사람과 같은 방향 n_h를 향하는 위치.
    팔꿈치는 S_r-W_r 축 둘레 원 위에서만 움직이므로, 그 원에서 n_h 쪽 점을 고른다 (팔 회전각 맞추기)."""
    a = W_r - S_r
    d = np.linalg.norm(a)
    if d < 1e-6:
        return None
    u = a / d
    n = _perp_unit(n_h, u)
    if n is None:
        return None
    x = (L_UPPER ** 2 - L_FORE ** 2 + d ** 2) / (2 * d)   # 어깨에서 원 중심까지
    r = np.sqrt(max(L_UPPER ** 2 - x ** 2, 0.0))           # 원 반지름
    return S_r + x * u + r * n


WRIST_TO_FLANGE = 0.081 + 0.045   # 손목 중심 -> 플랜지, 플랜지 z축 방향 (robot.urdf joint7, ee)


def elbow_goals(targets):
    """프레임마다 로봇 팔꿈치 목표 위치 (T,3). 손목 중심 목표 = TCP 목표 - 0.126 m × 플랜지 z.
    사람 팔꿈치 방향이나 TCP 목표가 없으면 NaN. (B는 이것을 가중치 목표로 쓴다)"""
    T = len(targets["valid"])
    out = np.full((T, 3), np.nan)
    for t in range(T):
        n_h = targets["elbow_dir"][t]
        if not targets["valid"][t] or not np.isfinite(n_h).all():
            continue
        W = targets["pos"][t] - WRIST_TO_FLANGE * targets["rot"][t][:, 2]
        e = robot_elbow_target(IIWA_SHOULDER, W, n_h)
        if e is not None:
            out[t] = e
    return out


def swivel_error(S_r, E_r, W_r, n_h):
    """로봇 팔꿈치 방향과 사람 팔꿈치 방향의 각도 차 (rad, 어깨-손목 축 둘레). 못 구하면 NaN."""
    n_r = elbow_direction(S_r, E_r, W_r)
    if n_r is None or n_h is None or not np.isfinite(n_h).all():
        return np.nan
    u = (W_r - S_r) / np.linalg.norm(W_r - S_r)
    n_hp = _perp_unit(n_h, u)
    if n_hp is None:
        return np.nan
    return float(np.arccos(np.clip(np.dot(n_r, n_hp), -1.0, 1.0)))


def rot_to_quat_wxyz(R):
    """(T,3,3) -> (T,4) (w,x,y,z). NaN 행은 NaN."""
    out = np.full((len(R), 4), np.nan)
    ok = np.isfinite(R).all(axis=(1, 2))
    if ok.any():
        out[ok] = np.roll(Rot.from_matrix(R[ok]).as_quat(), 1, axis=1)
    return out


def save_result(path, q, saturated, status, tcp_error, targets, **extra):
    """metrics/definitions.md 1절 형식으로 npz 저장. extra는 참고용 추가 배열."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q=q, saturated=saturated, status=status, tcp_error=tcp_error,
             target_pos=targets["pos"], target_quat=rot_to_quat_wxyz(targets["rot"]),
             target_orient_held=targets["orient_held"],     # 방향을 직전 값으로 유지해 만든 목표 (T,)
             target_wrist_manual=targets["wrist_manual"],   # 손목 위치가 사람이 직접 찍은 값 (T,)
             target_elbow_backfilled=targets["elbow_backfilled"],  # 맨 앞에서 처음 보인 팔꿈치 방향으로 채운 프레임 (T,)
             info=json.dumps(targets["info"], ensure_ascii=False), **extra)
    return path


def summarize(name, q, saturated, status, tcp_error, rot_error, sec):
    """한 에피소드 결과를 사람이 읽을 요약 문자열로."""
    T = len(status)
    solved = status != STATUS_NO_TARGET
    lines = [f"[{name}] 프레임 {T}, 계산 {sec:.1f} s ({1000 * sec / T:.1f} ms/프레임)",
             f"  status 0 성공 {np.sum(status == 0)} / 1 미수렴 {np.sum(status == 1)} / 2 목표 없음 {np.sum(status == 2)}"]
    if solved.any():
        e, r = tcp_error[solved], rot_error[solved]
        lines += [f"  위치 오차 m   중앙 {np.median(e):.4f}  95% {np.percentile(e, 95):.4f}  최대 {e.max():.4f}",
                  f"  방향 오차 rad 중앙 {np.median(r):.4f}  95% {np.percentile(r, 95):.4f}  최대 {r.max():.4f}",
                  f"  포화 프레임 비율 {saturated[solved].any(1).mean():.3f}  관절별 {np.round(saturated[solved].mean(0), 3).tolist()}",
                  f"  관절 중앙값 rad (푼 프레임) {np.round(np.median(q[solved], 0), 2).tolist()}"]
    lines.append(f"  관절 범위 rad min {np.round(q.min(0), 2).tolist()}  max {np.round(q.max(0), 2).tolist()}")
    return "\n".join(lines)


def timing_text(frame_ms):
    """프레임당 IK 시간 요약 (목표 있는 프레임만). 모델 로드·파일 입출력은 포함하지 않음."""
    f = frame_ms[np.isfinite(frame_ms)]
    if not len(f):
        return "IK 시간 -"
    return f"{np.median(f):.2f} ms/프레임 (중앙, 95% {np.percentile(f, 95):.2f}, {len(f)}프레임)"


def brief_line(label, q, saturated, status, pos_err, rot_err, swv_err, info):
    """시험 비교용 한 줄 요약."""
    ok = status != STATUS_NO_TARGET
    s = np.degrees(swv_err[np.isfinite(swv_err)])
    sw = f"중앙 {np.median(s):.0f} 95% {np.percentile(s, 95):.0f}" if len(s) else "-"
    extra = ""
    if info.get("n_orient_held") or info.get("n_wrist_manual"):
        extra = f" | 방향유지 {info.get('n_orient_held', 0)} 직접찍음 {info.get('n_wrist_manual', 0)}"
    return (f"{label}{extra} | 성공 {np.sum(status == 0):3d} 미수렴 {np.sum(status == 1):3d} | "
            f"위치95 {np.percentile(pos_err[ok], 95):.4f} 방향95 {np.percentile(rot_err[ok], 95):.3f} | "
            f"포화 {np.round(saturated[ok].mean(0), 2).tolist()} | 회전각오차 {sw} | "
            f"q2 {np.median(q[ok, 1]):.2f} q4 {np.median(q[ok, 3]):.2f} q6 {np.median(q[ok, 5]):.2f} | "
            f"테이블z {info['table_z_in_base']:.2f}")
