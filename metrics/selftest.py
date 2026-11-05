"""5-2 지표 코드 합성 데이터 검증 (정의서 6절). 정답을 아는 궤적으로 metrics.py가 정답을 내는지 본다.

모두 통과해야 실데이터에 쓴다. 결과는 --out 폴더에 selftest.md로 저장하고, 실패가 있으면 종료 코드 1.

    python metrics/selftest.py --out ~/out/신서연/metrics_selftest
"""
import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "metrics"))

from replay.replay import DEFAULT_MODEL, Replayer  # noqa: E402
import metrics as M  # noqa: E402

N = 100  # 합성 궤적 프레임 수
RESULTS = []
NAMES = ["shoulder", "elbow", "wrist", "index_mcp", "pinky_mcp", "thumb_tip", "index_tip"]


def check(name, how, got, want, tol=1e-9, rel=False):
    """결과 한 줄 기록. rel=True면 상대 오차, 아니면 절대 오차로 비교. want가 NaN이면 got도 NaN이어야 통과."""
    if isinstance(want, float) and np.isnan(want):
        ok = isinstance(got, float) and np.isnan(got)
    else:
        err = abs(got - want) / abs(want) if rel else abs(got - want)
        ok = bool(err <= tol)
    RESULTS.append((name, how, got, want, ok))
    print(f"[{'통과' if ok else '실패'}] {name}: {got} (정답 {want})")
    return ok


def base_res(q):
    T = len(q)
    return {"q": q, "status": np.zeros(T, int), "pos_err": np.zeros(T)}


def ts_of(T, fps):
    return np.arange(T) / fps


def find_self_collision_pose(r, lo, hi):
    """관절 한계 안 무작위 자세(seed 0)에서 자기 침투가 있는 첫 자세 (collision_exclude.md와 같은 방식)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(lo, hi, size=(200, 7))
        self_d, _ = M.contact_depths(r.replay(q)["contacts"], r.robot_bodies)
        idx = np.flatnonzero(self_d < 0)
        if len(idx):
            return q[idx[0]], float(self_d[idx[0]])
    raise RuntimeError("자기 충돌 자세를 못 찾음")


def replayer_with_box(center, half):
    """robot.xml에 상자 하나를 더한 임시 모델 (환경 충돌 검증용, 원본 파일은 그대로)."""
    return Replayer(env_boxes=[{"name": "env_test", "center": list(center), "size": [2 * half] * 3,
                                "quat": [1, 0, 0, 0]}])


def rotate_about(p, origin, axis, ang):
    """점 p를 origin을 지나는 axis 둘레로 ang만큼 돌림 (로드리게스)."""
    k = axis / np.linalg.norm(axis)
    v = p - origin
    return origin + v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * (k @ v) * (1 - np.cos(ang))


def main(out_dir):
    lo, hi, vmax = M.load_limits()
    r = Replayer()
    home = r.q_full_home[r.arm_adr].copy()
    q_home = np.tile(home, (N, 1))
    run = lambda res, **kw: M.episode_metrics(res, lo, hi, vmax, kw.pop("ts", ts_of(len(res["q"]), 30)), kw.pop("r", r), **kw)

    check("속도 한계 읽기 (rad/s)", "robot.urdf joint1 velocity = 85°/s", vmax[0], np.deg2rad(85), tol=1e-9)

    # 1. 관절 한계 위반율 (안전 확인)
    q = q_home.copy()
    q[:10, 0] = hi[0] + 0.1
    check("관절 한계 위반율", "100프레임 중 10프레임에서 joint1 = 상한 + 0.1 rad", M.limit_violation_rate(q, lo, hi), 0.1)
    q = q_home.copy()
    q[:10, 0] = hi[0] + 0.5e-6
    check("관절 한계 위반율 (수치 오차)", "10프레임에서 상한 + 0.5e-6 rad (ε 1e-6 미만은 안 셈)",
          M.limit_violation_rate(q, lo, hi), 0.0)

    # 2. 관절 한계 근접률: 한계 0.5° 안 10프레임 + 한계 정확히 위 5프레임(A가 잘라낸 경우) + 2° 떨어진 20프레임(안 셈)
    q = q_home.copy()
    q[:10, 3] = hi[3] - np.deg2rad(0.5)
    q[10:15, 1] = lo[1]
    q[15:35, 5] = hi[5] - np.deg2rad(2.0)
    check("관절 한계 근접률", "한계 0.5° 안 10 + 한계 위 5 + 2° 떨어짐 20 (안 셈)", M.limit_proximity_rate(q, lo, hi), 0.15)
    check("home 자세 근접률", "home만 100프레임", M.limit_proximity_rate(q_home, lo, hi), 0.0)

    # 3. 자기 충돌률·최대 침투 깊이
    q_col, depth = find_self_collision_pose(r, lo, hi)
    q = q_home.copy()
    q[40:60] = q_col
    m, _ = run(base_res(q))
    check("자기 충돌률", "자기 충돌 자세(무작위 seed 0)를 100프레임 중 20프레임", m["self_collision_rate"], 0.2)
    check("최대 침투 깊이 (m)", "같은 궤적. 정답 = 그 자세 하나를 재생한 침투 깊이", m["max_penetration_m"], -depth)

    # 4. IK 실패율
    status = np.zeros(N, int)
    status[:7] = 1
    status[50:55] = 2
    ik = M.ik_rates(status)
    check("IK 실패율", "status 1이 7개, 2가 5개", ik["ik_failure_rate"], 0.12)
    check("IK 미수렴 비율", "같은 배열", ik["ik_not_converged_rate"], 0.07)
    check("IK 해 없음 비율", "같은 배열", ik["ik_no_solution_rate"], 0.05)

    # 5. 저크: 7관절 모두 q = home + A·sin(ωt), 4주기. 정답 A·ω³/√2 (차분 근사라 1% 허용)
    A, f = 0.2, 0.5
    w = 2 * np.pi * f
    for fps, label in [(30.0, "30 fps"), (60.0, "60 fps → 30 Hz"), (59.97, "59.97 fps (본 촬영 T1·T3)")]:
        T = int(round(4 / f * fps))
        ts = ts_of(T, fps)
        q = home + A * np.sin(w * ts)[:, None] * np.ones(7)
        j, k, dt = M.jerk_rms(q, ts)
        check(f"저크 RMS ({label})", f"7관절 q = home + {A}·sin(2π·{f}·t), 4주기, k={k}", j, A * w ** 3 / np.sqrt(2),
              tol=0.01, rel=True)

    # 6. 속도 한계 초과율: joint1을 일정 속도로. 30 Hz 구간 수 기준
    for fps in (30.0, 60.0):
        T = int(fps * 3) + 1                                       # 3초
        ts = ts_of(T, fps)
        q = np.tile(home, (T, 1))
        speed = np.where(ts < 1.0, 0.5 * vmax[0], 1.5 * vmax[0])  # 처음 1초는 한계의 절반, 나머지 2초는 1.5배
        q[:, 0] = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(ts))])
        want = 2.0 / 3.0
        check(f"속도 한계 초과율 ({fps:.0f} fps)", "joint1 처음 1초 한계 0.5배, 2초 1.5배", M.velocity_limit_exceed_rate(q, ts, vmax),
              want, tol=0.02)
    q = np.tile(home, (N, 1))
    q[50:, 2] += 0.2                                               # 한 프레임에 0.2 rad 점프 (6 rad/s)
    check("속도 한계 초과율 (점프 1번)", "joint3이 한 프레임에 0.2 rad 점프 → 99구간 중 1", M.velocity_limit_exceed_rate(q, ts_of(N, 30), vmax),
          1 / 99, tol=1e-9)

    # 7. 팔 회전각: 사람 = 로봇 → 0, 사람 팔꿈치를 어깨–손목 축 둘레로 30° 돌림 → 30°
    q = q_home.copy()
    q[:, 2] = np.linspace(-0.5, 0.5, N)                            # joint3로 팔꿈치를 움직임
    S, E, W, _ = M.robot_kinematics(r, q)
    m, info = run(base_res(q), human_sew=(S, E, W))
    check("팔 회전각 차이 (같은 팔)", "사람 어깨·팔꿈치·손목 = 로봇 관절 2·4·6 위치", m["arm_angle_diff_rad"], 0.0, tol=1e-9)
    E30 = np.array([rotate_about(E[t], S[t], W[t] - S[t], np.deg2rad(30)) for t in range(N)])
    m, info = run(base_res(q), human_sew=(S, E30, W))
    check("팔 회전각 차이 (30° 돌림)", "사람 팔꿈치만 축 둘레로 30° 돌림", np.rad2deg(m["arm_angle_diff_rad"]), 30.0, tol=1e-6)
    E30[:20] = np.nan
    m, info = run(base_res(q), human_sew=(S, E30, W))
    check("팔 회전각 (팔꿈치 NaN 20%)", "NaN 프레임은 빼고 계산, 쓴 프레임 비율 0.8", info["arm_angle_frames"], 0.8)
    psi = M.sew_angle([0, 0, 0], [0.1, 0, -0.2], [0, 0, -0.5])
    check("팔 회전각 (어깨–손목 수직)", "축이 아래 방향과 같으면 정의 안 됨 → NaN", float(psi[0]), float("nan"))

    # 8. 최소 조작성: home은 양수, 관절 모두 0(팔을 위로 곧게 편 특이 자세)은 0
    _, _, _, w_home = M.robot_kinematics(r, home[None])
    _, _, _, w_zero = M.robot_kinematics(r, np.zeros((1, 7)))
    check("최소 조작성 (home > 0)", "home 자세", float(w_home[0] > 1e-3), 1.0)
    check("최소 조작성 (곧게 편 자세)", "관절 모두 0 = 특이 자세", float(w_zero[0]), 0.0, tol=1e-6)

    # 9. 파지점 거리 (참고): 사람 파지점 = 로봇 TCP → 0
    g = 30
    tcp = r.replay(q_home)["tcp_pos"][g]
    kp = np.zeros((N, 7, 3))
    kp[g, 5] = tcp + [0.01, 0, 0]
    kp[g, 6] = tcp - [0.01, 0, 0]
    m, _ = run(base_res(q_home), g=g, p_human=M.human_grasp_point(kp, NAMES, g))
    check("파지점 거리 (m)", "엄지 끝·검지 끝 중점 = TCP (항등 변환)", m["grasp_dist_m"], 0.0)
    kp[g, 5] = np.nan
    m, _ = run(base_res(q_home), g=g, p_human=M.human_grasp_point(kp, NAMES, g))
    check("파지점 거리 (키포인트 NaN)", "파지 프레임 엄지 끝이 NaN → NaN, 이 지표에서만 뺌", m["grasp_dist_m"], float("nan"))

    # 10. 환경 충돌률 (참고)
    rb = replayer_with_box(tcp, 0.05)
    q = q_home.copy()
    q[25:, 0] = home[0] + np.pi / 2
    m, _ = run(base_res(q), r=rb, has_env_scene=True)
    check("환경 충돌률", "home TCP에 상자, 100프레임 중 25프레임이 home", m["env_collision_rate"], 0.25)
    check("환경 충돌이 자기 충돌률에 안 섞임", "같은 궤적", m["self_collision_rate"], 0.0)
    m, _ = run(base_res(q_home))
    check("장면 없을 때 환경 충돌률", "테이블·장애물 장면이 없으면 0이 아니라 NaN", m["env_collision_rate"], float("nan"))

    # 11. 참고: 방향 오차·계산 시간
    res = base_res(q_home)
    res["rot_err"] = np.full(N, 0.02)
    res["time_ms"] = np.r_[np.full(90, 1.0), np.full(10, 50.0)]
    m, info = run(res)
    check("방향 추종 오차 평균 (rad)", "rot_err 모두 0.02", m["track_rot_err_mean_rad"], 0.02)
    check("계산 시간 중앙값 (ms)", "90프레임 1 ms, 10프레임 50 ms", m["compute_ms_median"], 1.0)
    check("계산 시간 95% (ms)", "같은 배열", info["compute_ms_p95"], 50.0)
    check("npz에 rot_err·time_ms 없음", "없으면 NaN", run(base_res(q_home))[0]["compute_ms_median"], float("nan"))

    # 12. 실행 경로
    check("전체 실행 (run)", "가짜 4번·3번 폴더 → metrics_long.csv, 60 fps 에피소드 하나", run_end_to_end(r, home), 1)

    write_report(out_dir)
    return all(ok for *_, ok in RESULTS)


def run_end_to_end(r, home):
    """폴더 구조·파일 형식을 그대로 흉내 내서 run()을 돌린다. (오류 0, 행 수·값이 맞으면 1)"""
    root = Path(tempfile.mkdtemp())
    T, fps, start = 120, 59.97, 95
    (root / "ret" / "A_test").mkdir(parents=True)
    ep = root / "eps" / "T1_E999"
    ep.mkdir(parents=True)
    q = np.tile(home, (T, 1))
    np.savez(root / "ret" / "A_test" / "T1_E999.npz", q=q, status=np.zeros(T, int), pos_err=np.full(T, 0.01),
             rot_err=np.full(T, 0.02), time_ms=np.full(T, 3.0))
    S, E, W, _ = M.robot_kinematics(r, q)
    kp = np.zeros((T, 7, 3))
    kp[:, 0], kp[:, 1], kp[:, 2] = S, E, W                        # 사람 팔 = 로봇 팔 → 팔 회전각 차이 0
    np.save(ep / "keypoints.npy", kp)
    (ep / "keypoint_names.json").write_text(json.dumps(NAMES))
    (ep / "meta.json").write_text(json.dumps({"task": "컵 잡기", "case_tags": ["큰 이동", "가장자리"],
                                             "exo_start_frame": start}, ensure_ascii=False))
    (ep / "labels.json").write_text(json.dumps({"exo_start_frame": start, "exo_end_frame": start + T,
                                               "grasp_frame": start + 50}))
    with open(ep / "exo_ts.csv", "w", newline="") as f:
        f.write("t_s\n" + "\n".join(f"{t:.6f}" for t in 1000 + np.arange(T) / fps) + "\n")
    out = root / "out"
    a = argparse.Namespace(retarget=root / "ret", episodes=root / "eps", sync=None, out=out, algos=None,
                           ids=None, env_scene=False, identity_transform=True)
    M.run(a)
    with open(out / "metrics_long.csv", encoding="utf-8") as f:
        rows = {row["metric"]: row for row in csv.DictReader(f)}
    with open(out / "episode_info.csv", encoding="utf-8") as f:
        info = list(csv.DictReader(f))
    ok = (len(rows) == len(M.METRICS) and info[0]["status"] == "ok" and info[0]["resample_k"] == "2"
          and rows["ik_failure_rate"]["case_tags"] == "큰 이동|가장자리"
          and abs(float(rows["arm_angle_diff_rad"]["value"])) < 1e-9
          and float(rows["compute_ms_median"]["value"]) == 3.0)
    return 1 if ok else 0


def write_report(out_dir):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = sum(ok for *_, ok in RESULTS)
    lines = ["# 지표 코드 합성 데이터 검증 결과 (정의서 v0.3)", "",
             f"`python metrics/selftest.py` — {n_ok}/{len(RESULTS)} 통과", "",
             "| 검사 | 만드는 법 | 결과 | 정답 | 판정 |", "|---|---|---|---|---|"]
    fmt = (lambda v: f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
    for name, how, got, want, ok in RESULTS:
        lines.append(f"| {name} | {how} | {fmt(got)} | {fmt(want)} | {'통과' if ok else '**실패**'} |")
    lines += ["", "- 저크는 3차 차분이 미분의 근사이고 양 끝 3프레임이 빠져서 해석해보다 1% 미만 작게 나온다 (허용 1%).",
              "- 60 fps·59.97 fps 궤적이 30 fps와 같은 값을 내면 30 Hz로 맞추는 처리(정의서 2.6·2.7)가 맞게 동작한 것이다.",
              "- 속도 초과율은 30 Hz 구간 경계가 1초에 딱 맞지 않아 2% 안에서 비교했다.",
              "- 환경 충돌은 원본 모델에 상자 하나를 더한 장면(replay/scene.py와 같은 방식)으로 확인했다."]
    (out_dir / "selftest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(RESULTS)} 통과 → {out_dir / 'selftest.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 폴더 (~/project 밖, 예: ~/out/신서연/metrics_selftest)")
    sys.exit(0 if main(ap.parse_args().out) else 1)
