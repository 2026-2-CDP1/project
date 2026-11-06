"""5-2 지표 코드 합성 데이터 검증 (정의서 6절). 정답을 아는 궤적으로 metrics.py가 정답을 내는지 본다.

모두 통과해야 실데이터에 쓴다. 결과는 --out 폴더에 selftest.md로 저장하고, 실패가 있으면 종료 코드 1.

    python metrics/selftest.py --out ~/out/신서연/metrics_selftest
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "metrics"))

from replay.replay import Replayer  # noqa: E402
import metrics as M  # noqa: E402
import summary as S  # noqa: E402

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


def ts_of(T, fps):
    return np.arange(T) / fps


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([np.cos(a), 0, 0, np.sin(a)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def find_self_collision_pose(r, lo, hi):
    """관절 한계 안 무작위 자세(seed 0)에서 자기 침투가 있는 첫 자세 (collision_exclude.md와 같은 방식)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(lo, hi, size=(200, 7))
        self_d, _ = M.contact_depths(r.replay(q)["contacts"], r.robot_bodies)
        idx = np.flatnonzero(self_d < 0)
        if len(idx):
            return q[idx[0]]
    raise RuntimeError("자기 충돌 자세를 못 찾음")


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

    def res_of(q, targets=True):
        """q를 그대로 재생한 TCP를 목표로 둔 npz 내용 (= 완벽하게 따라간 경우)."""
        res = {"q": q}
        if targets:
            out = r.replay(q)
            res["target_pos"], res["target_quat"] = out["tcp_pos"].copy(), out["tcp_quat"].copy()
        return res

    def run(res, rr=None, ts=None, **kw):
        return M.episode_metrics(res, lo, hi, vmax, ts if ts is not None else ts_of(len(res["q"]), 30), rr or r, **kw)

    check("속도 한계 읽기 (rad/s)", "robot.urdf joint1 velocity = 85°/s", vmax[0], np.deg2rad(85))

    # ---- 정확도: 리타게팅 성공률 (실패율로 저장)
    check("쿼터니언 사이 각", "z축 10° 회전", float(M.quat_angle(quat_z(0), quat_z(10))[0]), np.deg2rad(10), tol=1e-9)
    check("쿼터니언 부호", "q와 −q는 같은 방향", float(M.quat_angle(quat_z(30), -quat_z(30))[0]), 0.0, tol=1e-6)
    m, info = run(res_of(q_home))
    check("성공률: 완벽히 따라감", "목표 = 재생한 TCP → 실패율 0", m["retarget_fail_rate"], 0.0)
    res = res_of(q_home)
    res["target_pos"][:20] += [0.02, 0, 0]                                   # 20프레임 2 cm 벗어남
    res["target_pos"][20:30] += [0.005, 0, 0]                                # 10프레임 5 mm (성공)
    for t in range(30, 40):                                                  # 10프레임 방향 10° 벗어남
        res["target_quat"][t] = quat_mul(res["target_quat"][t], quat_z(10))
    for t in range(40, 50):                                                  # 10프레임 방향 3° (성공)
        res["target_quat"][t] = quat_mul(res["target_quat"][t], quat_z(3))
    m, _ = run(res)
    check("성공률: 위치·방향 기준", "2 cm 20프레임 + 10° 10프레임 실패, 5 mm·3°는 성공 → 실패율 0.3", m["retarget_fail_rate"], 0.3)
    check("위치 추종 오차 최대 (m)", "같은 궤적", m["track_pos_err_max_m"], 0.02, tol=1e-9)
    check("방향 추종 오차 최대 (rad)", "같은 궤적", m["track_rot_err_max_rad"], np.deg2rad(10), tol=1e-6)
    q = q_home.copy()
    q[:5, 0] = hi[0] + 0.05                                                  # 한계 밖이지만 목표는 그 자세 그대로
    m, _ = run(res_of(q))
    check("성공률: 한계 밖은 실패", "목표에는 맞았지만 관절 한계 밖 5프레임 → 실패율 0.05", m["retarget_fail_rate"], 0.05)
    check("관절 한계 위반율 (안전 확인)", "같은 궤적", m["limit_violation_rate"], 0.05)
    res = {"q": q_home, "pos_err": np.full(N, 0.003)}
    m, info = run(res)
    check("목표 없음", "npz에 목표가 없으면 성공률 NaN, 추종 오차는 4번 보고값", m["retarget_fail_rate"], float("nan"))
    check("목표 없음 → 4번 pos_err", "pos_err 모두 3 mm", m["track_pos_err_mean_m"], 0.003)
    m, info = run({"q": q_home, "tcp_error": np.full(N, 0.004), "rot_error": np.full(N, 0.01)})
    check("목표 없음 → 4번 tcp_error", "4번 save_result 이름 tcp_error 모두 4 mm", m["track_pos_err_mean_m"], 0.004)
    check("목표 없음 → 4번 rot_error", "4번 save_result 이름 rot_error 모두 0.01 rad", m["track_rot_err_mean_rad"], 0.01)
    res = res_of(q_home)
    res["target_pos"][:10] = np.nan                                         # 손목 미검출 10프레임 (4번: 목표 NaN)
    res["target_quat"][:10] = np.nan
    res["target_pos"][10:30] += [0.02, 0, 0]                                 # 20프레임 2 cm 벗어남
    m, info = run(res)
    check("성공률: 목표 NaN 프레임", "목표 NaN 10 + 2 cm 20 → NaN은 분모에서 빼서 실패율 20/90", m["retarget_fail_rate"], 20 / 90)
    check("목표 없는 프레임 비율", "같은 궤적 → episode_info에 0.1", info["target_missing_rate"], 0.1)
    check("추종 오차 (NaN 제외)", "평균 = 2 cm × 20 / 90", m["track_pos_err_mean_m"], 0.02 * 20 / 90, tol=1e-9)
    res["target_pos"][:] = np.nan
    res["target_quat"][:] = np.nan
    check("목표 전부 NaN", "판정할 프레임이 없음 → NaN", run(res)[0]["retarget_fail_rate"], float("nan"))

    # ---- 정확도: 팔 회전각 차이
    q = q_home.copy()
    q[:, 2] = np.linspace(-0.5, 0.5, N)
    S, E, W, _ = M.robot_kinematics(r, q)
    m, info = run(res_of(q), human_sew=(S, E, W))
    check("팔 회전각 차이 (같은 팔)", "사람 어깨·팔꿈치·손목 = 로봇 관절 2·4·6 위치", m["arm_angle_diff_rad"], 0.0)
    E30 = np.array([rotate_about(E[t], S[t], W[t] - S[t], np.deg2rad(30)) for t in range(N)])
    m, _ = run(res_of(q), human_sew=(S, E30, W))
    check("팔 회전각 차이 (30° 돌림)", "사람 팔꿈치만 축 둘레로 30° 돌림", np.rad2deg(m["arm_angle_diff_rad"]), 30.0, tol=1e-6)
    E30[:20] = np.nan
    _, info = run(res_of(q), human_sew=(S, E30, W))
    check("팔 회전각 (팔꿈치 NaN 20%)", "NaN 프레임은 빼고 계산, 쓴 프레임 비율 0.8", info["arm_angle_frames"], 0.8)
    psi = M.sew_angle([0, 0, 0], [0.1, 0, -0.2], [0, 0, -0.5])
    check("팔 회전각 (어깨–손목 수직)", "축이 아래 방향과 같으면 정의 안 됨 → NaN", float(psi[0]), float("nan"))

    # ---- 속도: 계산 시간
    res = res_of(q_home)
    res["time_ms"] = np.r_[500.0, np.full(89, 1.0), np.full(10, 50.0)]       # 첫 프레임 500 ms(준비)는 뺌
    m, _ = run(res)
    check("계산 시간 중앙값 (ms)", "첫 프레임 500 ms 제외, 89프레임 1 ms + 10프레임 50 ms", m["compute_ms_median"], 1.0)
    check("계산 시간 95% (ms)", "같은 배열 (99프레임 중 상위 10개가 50 ms)", m["compute_ms_p95"], 50.0)
    check("실시간 초과율", "33.3 ms 넘는 프레임 10 / 99", m["over_budget_rate"], 10 / 99)
    check("time_ms 없음", "없으면 NaN", run(res_of(q_home))[0]["compute_ms_median"], float("nan"))
    res = res_of(q_home)
    res["time_ms"] = np.r_[np.full(10, np.nan), 500.0, np.full(79, 2.0), np.full(10, 40.0)]   # 목표 없는 10프레임 = NaN
    res["elbow_used"] = np.r_[np.zeros(30, bool), np.ones(70, bool)]
    m, info = run(res)
    check("계산 시간 (NaN 프레임)", "NaN 10 빼고, 남은 첫 프레임 500 ms 빼고 → 중앙값 2 ms", m["compute_ms_median"], 2.0)
    check("실시간 초과율 (NaN 프레임)", "40 ms 10 / 남은 89프레임", m["over_budget_rate"], 10 / 89)
    check("계산 시간 NaN 비율", "episode_info에 0.1", info["time_missing_rate"], 0.1)
    check("팔꿈치 사용 비율", "elbow_used 70/100 → episode_info에 0.7", info["elbow_used_rate"], 0.7)

    # ---- 궤적 품질: 저크
    A, f = 0.2, 0.5
    w = 2 * np.pi * f
    for fps, label in [(30.0, "30 fps"), (60.0, "60 fps → 30 Hz"), (59.97, "59.97 fps (본 촬영 T1·T3)")]:
        T = int(round(4 / f * fps))
        ts = ts_of(T, fps)
        q = home + A * np.sin(w * ts)[:, None] * np.ones(7)
        j, k, _ = M.jerk_rms(q, ts)
        check(f"저크 RMS ({label})", f"7관절 q = home + {A}·sin(2π·{f}·t), 4주기, k={k}", j, A * w ** 3 / np.sqrt(2),
              tol=0.01, rel=True)

    # ---- 궤적 품질: 속도 한계 초과율
    for fps in (30.0, 60.0):
        T = int(fps * 3) + 1
        ts = ts_of(T, fps)
        q = np.tile(home, (T, 1))
        speed = np.where(ts < 1.0, 0.5 * vmax[0], 1.5 * vmax[0])
        q[:, 0] = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(ts))])
        check(f"속도 한계 초과율 ({fps:.0f} fps)", "joint1 처음 1초 한계 0.5배, 2초 1.5배",
              M.velocity_limit_exceed_rate(q, ts, vmax), 2 / 3, tol=0.02)
    q = q_home.copy()
    q[50:, 2] += 0.2
    check("속도 한계 초과율 (점프 1번)", "joint3이 한 프레임에 0.2 rad 점프 → 99구간 중 1",
          M.velocity_limit_exceed_rate(q, ts_of(N, 30), vmax), 1 / 99)

    # ---- 궤적 품질: 자기 충돌률
    q = q_home.copy()
    q[40:60] = find_self_collision_pose(r, lo, hi)
    check("자기 충돌률", "자기 충돌 자세(무작위 seed 0)를 100프레임 중 20프레임", run(res_of(q))[0]["self_collision_rate"], 0.2)

    # ---- 궤적 품질: 환경 충돌률 = ① 장애물 파고듦 or ② 테이블 위 영역에서 팔꿈치가 테이블 면 아래
    f_sim = lambda p: np.array([0.1, -0.2, 0.3]) + 0.8 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.asarray(p)
    poly = {"polygon": [[0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.2, 0.4], [0.0, 0.2]]}  # 오각형 (위가 뾰족)
    P_table = np.array([[0.2, 0.3, -0.05], [0.35, 0.35, -0.05], [0.2, 0.1, 0.05]])  # 안·아래 / 밖(경계 상자 안) / 위
    got = M.elbow_below_table(np.array([f_sim(p) for p in P_table]), M.table_frame_inverse(f_sim), poly)
    check("팔꿈치–테이블 판정", "축척 0.8·회전 변환 + 오각형 윤곽: 윤곽 안 아래 / 윤곽 밖(뾰족한 꼭지 옆) / 위 → 참·거짓·거짓",
          float((got == [True, False, False]).all()), 1.0)
    tcp_home = r.replay(home[None])["tcp_pos"][0]
    rb = Replayer(env_boxes=[{"name": "env_obs1", "center": tcp_home.tolist(), "size": [0.1] * 3, "quat": [1, 0, 0, 0]}])
    q = q_home.copy()
    q[25:, 0] = home[0] - np.pi / 2                                          # 25프레임 home(상자에 닿음), 75프레임 돌림
    _, E_home, _, _ = M.robot_kinematics(r, home[None])
    _, E_rot, _, _ = M.robot_kinematics(r, q[-1:])
    lift = lambda p: np.asarray(p, float) + [0, 0, E_home[0, 2] + 0.05]      # 테이블 면(z=0)이 로봇 좌표에서 팔꿈치보다 5 cm 위
    c = E_rot[0, :2]
    sq = {"polygon": [[c[0] - 0.1, c[1] - 0.1], [c[0] + 0.1, c[1] - 0.1], [c[0] + 0.1, c[1] + 0.1], [c[0] - 0.1, c[1] + 0.1]]}
    m, _ = run(res_of(q), rr=rb, has_obstacles=True, table=(lift, sq))           # 돌린 팔꿈치만 윤곽 안
    check("환경: 장애물 부분", "home TCP에 상자, 25프레임", m["env_obstacle_rate"], 0.25)
    check("환경: 팔꿈치–테이블 부분", "돌린 75프레임만 테이블 위 영역에서 면 아래", m["env_elbow_below_table_rate"], 0.75)
    check("환경 충돌률 (둘 중 하나)", "25 + 75 프레임 → 1.0", m["env_collision_rate"], 1.0)
    check("환경 접촉이 자기 충돌률에 안 섞임", "같은 궤적", m["self_collision_rate"], 0.0)
    m, _ = run(res_of(q_home))
    check("장애물·테이블 정보 없음", "없으면 0이 아니라 NaN", m["env_collision_rate"], float("nan"))

    # ---- 참고: 한계 근접률, 파지점 거리, 최소 조작성
    q = q_home.copy()
    q[:10, 3] = hi[3] - np.deg2rad(0.5)
    q[10:15, 1] = lo[1]
    q[15:35, 5] = hi[5] - np.deg2rad(2.0)
    check("관절 한계 근접률 (참고)", "한계 0.5° 안 10 + 한계값 5 + 2° 떨어짐 20 (안 셈)", M.limit_proximity_rate(q, lo, hi), 0.15)
    g = 30
    kp = np.zeros((N, 7, 3))
    kp[g, 5], kp[g, 6] = tcp_home + [0.01, 0, 0], tcp_home - [0.01, 0, 0]
    m, _ = run(res_of(q_home), g=g, p_human=M.human_grasp_point(kp, NAMES, g))
    check("파지점 거리 (참고)", "엄지 끝·검지 끝 중점 = TCP", m["grasp_dist_m"], 0.0)
    kp[g, 5] = np.nan
    m, _ = run(res_of(q_home), g=g, p_human=M.human_grasp_point(kp, NAMES, g))
    check("파지점 거리 (키포인트 NaN)", "NaN → 이 지표에서만 뺌", m["grasp_dist_m"], float("nan"))
    _, _, _, w_zero = M.robot_kinematics(r, np.zeros((1, 7)))
    check("최소 조작성 (곧게 편 자세)", "관절 모두 0 = 특이 자세", float(w_zero[0]), 0.0, tol=1e-6)

    ok_run, ok_info = run_end_to_end(r, home)
    check("전체 실행 (run)", "가짜 4번·3번 폴더 + obstacles.json + 테이블 가장자리 → metrics_long.csv", ok_run, 1)
    check("npz info로 좌표 변환 + labels.json 없음", "info 같으면 ok·다르면 오류 / labels.json 없는 에피소드도 ok, 파지점 거리만 NaN", ok_info, 1)
    summary_checks()
    write_report(out_dir)
    return all(ok for *_, ok in RESULTS)


def summary_checks():
    """집계(summary.py, 정의서 4·5절): 값을 아는 가짜 metrics_long.csv로 표·목록을 확인."""
    root = Path(tempfile.mkdtemp())
    eps = [f"T1_E{i:03d}" for i in range(1, 7)]
    tags = {e: ("큰 이동" if i < 4 else "정밀 구간") for i, e in enumerate(eps)}
    val = {  # (metric, algo) → 에피소드 6개 값
        ("jerk_rms", "A_DLS"): [1, 2, 3, 4, 5, 60], ("jerk_rms", "B_mink"): [0.5, 1, 1, 1, 2, 2],
        ("retarget_fail_rate", "A_DLS"): [0.1] * 6, ("retarget_fail_rate", "B_mink"): [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        ("min_manipulability", "A_DLS"): [0.05, 0.01, 0.03, 0.04, 0.02, 0.0],
        ("limit_violation_rate", "B_mink"): [0, 0, 0, 0, 0.02, 0],
    }
    with open(root / "metrics_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "episode_id", "task", "case_tags", "metric", "value"])
        for algo in ("A_DLS", "B_mink"):
            for i, e in enumerate(eps):
                for m in M.METRICS:
                    v = val.get((m, algo), [0.0] * 6)[i]
                    w.writerow([algo, e, "컵 잡기", tags[e], m, v])
    with open(root / "episode_info.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "episode_id", "status", "dt_flag", "grasp_note"])
        for algo in ("A_DLS", "B_mink"):
            for e in eps:
                st = "오류: 시험" if (algo == "B_mink" and e == "T1_E006") else "ok"     # E006은 B만 오류
                w.writerow([algo, e, st, "False", ""])
    algos, common, dropped, agg = S.run(root, root / "out")
    check("집계: 공통 에피소드", "B에서만 오류 난 E006은 A에서도 뺌 → 5개", len(common), 5)
    check("집계: 뺀 에피소드 목록", "E006이 이유와 함께 목록에", float("T1_E006" in dropped), 1.0)
    n, _, med, worst, ws = agg[("jerk_rms", S.ALL, "A_DLS")]
    check("집계: 중앙값", "A 저크 1~5 (E006의 60은 빠짐) → 중앙값 3", med, 3.0)
    check("집계: 최악값", "같은 칸 → 최댓값 5", worst, 5.0)
    check("집계: 케이스별", "큰 이동 E001~E004 → A 저크 1~4 중앙값 2.5", agg[("jerk_rms", "큰 이동", "A_DLS")][2], 2.5)
    check("집계: 최소 조작성 최악", "클수록 좋은 지표 → 최솟값 0.01이 최악", agg[("min_manipulability", S.ALL, "A_DLS")][3], 0.01)
    check("집계: 성공률 표시", "실패율 0.1 → 성공률 90.0%", float(S.fmt("retarget_fail_rate", 0.1) == "90.0%"), 1.0)
    md = (root / "out" / "summary.md").read_text(encoding="utf-8")
    worst_md = (root / "out" / "worst_episodes.md").read_text(encoding="utf-8")
    check("집계: 표본 적음 표시", "큰 이동 n=4·정밀 구간 n=1 칸에 ⚠", float("n=4 ⚠" in md and "n=1 ⚠" in md), 1.0)
    check("집계: 가장 좋은 알고리즘 굵게", "전체 저크 B(중앙값 1)가 굵게", float("| 전체 | 3 rad/s³ (5 rad/s³) n=5 | **1 rad/s³" in md), 1.0)
    check("집계: 안전 확인", "B 위반율 0이 아님 → 경고", float("0이 아닌 칸이 있음" in md), 1.0)
    check("집계: worst 목록", "A 저크 최악 E005 5가 맨 앞", float("T1_E005 5 rad/s³<br>T1_E004 4 rad/s³" in worst_md), 1.0)


def run_end_to_end(r, home):
    """폴더 구조·파일 형식을 그대로 흉내 내서 run()을 돌린다. 반환 (실행 경로 ok, info 처리 ok) — 각각 1이면 통과.

    좌표 변환은 4번 to_robot_point(p, info) 대신 "info의 offset만큼 옮기기"로 바꿔 끼운다.
    알고리즘 2개 x 에피소드 2개. T1_E998만 알고리즘끼리 info를 다르게 줘서 오류로 기록되는지 본다.
    """
    root = Path(tempfile.mkdtemp())
    T, fps, start = 120, 59.97, 95
    off = np.array([0.0, 0.0, 0.1])
    q = np.tile(home, (T, 1))
    out = r.replay(q)
    S, E, W, _ = M.robot_kinematics(r, q)
    kp = np.zeros((T, 7, 3))
    kp[:, 0], kp[:, 1], kp[:, 2] = S - off, E - off, W - off             # 테이블 좌표 = 로봇 좌표 − offset
    for eid in ("T1_E998", "T1_E999"):
        ep = root / "eps" / eid
        ep.mkdir(parents=True)
        np.save(ep / "keypoints.npy", kp)
        (ep / "keypoint_names.json").write_text(json.dumps(NAMES))
        (ep / "meta.json").write_text(json.dumps({"task": "컵 잡기", "case_tags": ["큰 이동", "가장자리"],
                                                 "exo_start_frame": start}, ensure_ascii=False))
        if eid == "T1_E999":                                                # E998은 labels.json 없음 (3번 처리 직후 상태)
            (ep / "labels.json").write_text(json.dumps({"exo_start_frame": start, "exo_end_frame": start + T,
                                                       "grasp_frame": start + 50}))
        with open(ep / "exo_ts.csv", "w", newline="") as f:
            f.write("t_s\n" + "\n".join(f"{t:.6f}" for t in 1000 + np.arange(T) / fps) + "\n")
        for algo in ("A_test", "B_test"):
            (root / "ret" / algo).mkdir(parents=True, exist_ok=True)
            o = off if not (eid == "T1_E998" and algo == "B_test") else off * 2   # E998은 B만 info가 다름
            np.savez(root / "ret" / algo / f"{eid}.npz", q=q, target_pos=out["tcp_pos"], target_quat=out["tcp_quat"],
                     time_ms=np.full(T, 3.0), info=json.dumps({"offset": o.tolist()}))
    outl = root / "table_outline.json"
    outl.write_text(json.dumps({"T1": {"polygon": [[-5, -5], [5, -5], [5, 5], [-5, 5]]}, "T2": None, "T3": None}))
    obs = root / "obstacles.json"
    obs.write_text(json.dumps({"frame": "test", "episodes": {"T3_E001": {"obstacles": []}}}))   # 이 에피소드들은 장애물 없음
    a = argparse.Namespace(retarget=root / "ret", episodes=root / "eps", sync=None, out=root / "out", algos=None,
                           ids=None, obstacles=obs, table_outline=outl, identity_transform=False)
    saved = M.load_transform
    M.load_transform = lambda identity: (lambda p, info: np.asarray(p, float) + np.asarray(info["offset"]))
    try:
        M.run(a)
    finally:
        M.load_transform = saved
    with open(root / "out" / "metrics_long.csv", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f)]
    with open(root / "out" / "episode_info.csv", encoding="utf-8") as f:
        info = {(row["algo"], row["episode_id"]): row for row in csv.DictReader(f)}
    v = {row["metric"]: float(row["value"]) for row in rows if row["algo"] == "A_test" and row["episode_id"] == "T1_E999"}
    ok_run = (len(v) == len(M.METRICS) and info[("A_test", "T1_E999")]["resample_k"] == "2"
              and rows[0]["case_tags"] == "큰 이동|가장자리" and v["retarget_fail_rate"] == 0.0
              and abs(v["arm_angle_diff_rad"]) < 1e-9 and v["compute_ms_median"] == 3.0 and v["env_collision_rate"] == 0.0)
    status = {k: row["status"] for k, row in info.items()}
    no_lab = info[("A_test", "T1_E998")]
    g998 = [row for row in rows if row["algo"] == "A_test" and row["episode_id"] == "T1_E998" and row["metric"] == "grasp_dist_m"]
    ok_info = ok_info_labels = (no_lab["status"] == "ok" and "labels.json 없음" in no_lab["grasp_note"]
                                and g998 and g998[0]["value"] in ("nan", "NaN", ""))
    ok_info = ok_info_labels and (status[("A_test", "T1_E998")] == "ok" and status[("A_test", "T1_E999")] == "ok"
               and status[("B_test", "T1_E999")] == "ok" and "info" in status[("B_test", "T1_E998")]
               and status[("B_test", "T1_E998")].startswith("오류"))
    return (1 if ok_run else 0), (1 if ok_info else 0)


def write_report(out_dir):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = sum(ok for *_, ok in RESULTS)
    lines = ["# 지표 코드 합성 데이터 검증 결과 (정의서 v1.0)", "",
             f"`python metrics/selftest.py` — {n_ok}/{len(RESULTS)} 통과", "",
             "| 검사 | 만드는 법 | 결과 | 정답 | 판정 |", "|---|---|---|---|---|"]
    fmt = (lambda v: f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
    for name, how, got, want, ok in RESULTS:
        lines.append(f"| {name} | {how} | {fmt(got)} | {fmt(want)} | {'통과' if ok else '**실패**'} |")
    lines += ["", "- 저크는 3차 차분이 미분의 근사이고 양 끝 3프레임이 빠져서 해석해보다 1% 미만 작게 나온다 (허용 1%).",
              "- 60 fps·59.97 fps 궤적이 30 fps와 같은 값을 내면 30 Hz로 맞추는 처리(정의서 2.4·2.5)가 맞게 동작한 것이다.",
              "- 속도 초과율은 30 Hz 구간 경계가 1초에 딱 맞지 않아 2% 안에서 비교했다.",
              "- 장애물은 원본 모델에 상자 하나를 더한 장면(replay/scene.py와 같은 방식), 테이블 면은 평행 이동·축척·회전 변환으로 확인했다."]
    (out_dir / "selftest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(RESULTS)} 통과 → {out_dir / 'selftest.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 폴더 (~/project 밖, 예: ~/out/신서연/metrics_selftest)")
    sys.exit(0 if main(ap.parse_args().out) else 1)
