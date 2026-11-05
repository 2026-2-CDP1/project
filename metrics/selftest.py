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
    return {"q": q, "status": np.zeros(T, int), "saturated": np.zeros((T, 7), bool), "pos_err": np.zeros(T)}


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
    xml = Path(DEFAULT_MODEL).read_text(encoding="utf-8")
    xml = re.sub(r'meshdir="([^"]+)"',
                 lambda mo: f'meshdir="{(Path(DEFAULT_MODEL).parent / mo.group(1)).resolve().as_posix()}"', xml)
    box = (f'<body name="test_box" pos="{center[0]} {center[1]} {center[2]}">'
           f'<geom type="box" size="{half} {half} {half}"/></body>')
    xml = xml.replace("<worldbody>", "<worldbody>" + box, 1)
    tmp = Path(tempfile.mkdtemp()) / "robot_box.xml"
    tmp.write_text(xml, encoding="utf-8")
    return Replayer(tmp)


def main(out_dir):
    lo, hi = M.load_limits()
    r = Replayer()
    home = r.q_full_home[r.arm_adr].copy()
    q_home = np.tile(home, (N, 1))

    # 1. 관절 한계 위반율: 100프레임 중 10프레임에서 관절 1을 한계 + 0.1 rad
    q = q_home.copy()
    q[:10, 0] = hi[0] + 0.1
    check("관절 한계 위반율", "100프레임 중 10프레임에서 joint1 = 상한 + 0.1 rad",
          M.limit_violation_rate(q, lo, hi), 0.1)
    q = q_home.copy()
    q[:10, 0] = hi[0] + 0.5e-6
    check("관절 한계 위반율 (수치 오차)", "10프레임에서 상한 + 0.5e-6 rad (ε 1e-6 미만은 안 셈)",
          M.limit_violation_rate(q, lo, hi), 0.0)

    # 2. 포화율: saturated 15프레임
    sat = np.zeros((N, 7), bool)
    sat[20:35, 3] = True
    check("관절 한계 포화율", "saturated가 참인 프레임 15개", M.limit_saturation_rate(sat), 0.15)

    # 3. 자기 충돌률·최대 침투 깊이: 충돌 자세를 100프레임 중 20프레임
    q_col, depth = find_self_collision_pose(r, lo, hi)
    q = q_home.copy()
    q[40:60] = q_col
    res = base_res(q)
    m, _ = M.episode_metrics(res, lo, hi, ts_of(N, 30), r)
    check("자기 충돌률", "자기 충돌 자세(무작위 seed 0)를 100프레임 중 20프레임", m["self_collision_rate"], 0.2)
    check("최대 침투 깊이 (m)", "같은 궤적. 정답 = 그 자세 하나를 재생한 침투 깊이",
          m["max_penetration_m"], -depth)
    check("home 자세 자기 충돌률", "home만 100프레임", M.episode_metrics(base_res(q_home), lo, hi,
          ts_of(N, 30), r)[0]["self_collision_rate"], 0.0)

    # 4. IK 실패율: 상태 코드 직접 만들기 (1 미수렴 7개, 2 해 없음 5개)
    status = np.zeros(N, int)
    status[:7] = 1
    status[50:55] = 2
    ik = M.ik_rates(status)
    check("IK 실패율", "status 1이 7개, 2가 5개", ik["ik_failure_rate"], 0.12)
    check("IK 미수렴 비율", "같은 배열", ik["ik_not_converged_rate"], 0.07)
    check("IK 해 없음 비율", "같은 배열", ik["ik_no_solution_rate"], 0.05)

    # 5. 저크: 7관절 모두 q = home + A·sin(ωt), 주기 정수 번. 정답 A·ω³/√2 (차분 근사라 1% 허용)
    A, f = 0.2, 0.5
    w = 2 * np.pi * f
    want = A * w ** 3 / np.sqrt(2)
    for fps, label in [(30.0, "30 fps"), (60.0, "60 fps → 30 Hz로 맞춤"), (59.97, "59.97 fps (본 촬영 T1·T3)")]:
        T = int(round(4 / f * fps))  # 4주기
        ts = ts_of(T, fps)
        q = home + A * np.sin(w * ts)[:, None] * np.ones(7)
        j, k, dt = M.jerk_rms(q, ts)
        check(f"저크 RMS ({label})", f"7관절 q = home + {A}·sin(2π·{f}·t), 4주기, k={k}, Δt={dt:.5f}",
              j, want, tol=0.01, rel=True)
    _, k60, _ = M.jerk_rms(np.zeros((240, 7)), ts_of(240, 60))
    check("60 fps 고르는 간격 k", "60 fps면 2프레임마다 하나", k60, 2)

    # 6. 파지점 거리: 사람 파지점 = 로봇 TCP → 0
    g = 30
    tcp = r.replay(q_home)["tcp_pos"][g]
    names = ["wrist", "index_mcp", "pinky_mcp", "thumb_mcp", "x", "thumb_tip", "index_tip"]
    kp = np.zeros((N, 7, 3))
    kp[g, 5] = tcp + [0.01, 0, 0]
    kp[g, 6] = tcp - [0.01, 0, 0]  # 중점 = TCP
    p = M.human_grasp_point(kp, names, g)
    m, info = M.episode_metrics(base_res(q_home), lo, hi, ts_of(N, 30), r, g, p)
    check("파지점 거리 (m)", "엄지 끝·검지 끝 중점 = TCP (항등 변환)", m["grasp_dist_m"], 0.0, tol=1e-9)
    kp[g, 5] = np.nan
    p = M.human_grasp_point(kp, names, g)
    m, info = M.episode_metrics(base_res(q_home), lo, hi, ts_of(N, 30), r, g, p)
    check("파지점 거리 (키포인트 NaN)", "파지 프레임 엄지 끝이 NaN → NaN, 이 지표에서만 뺌",
          m["grasp_dist_m"], float("nan"))

    # 7. 환경 충돌률: home TCP 위치에 한 변 10 cm 상자. 25프레임 home(닿음), 75프레임 joint1 +90°(떨어짐)
    rb = replayer_with_box(tcp, 0.05)
    q = q_home.copy()
    q[25:, 0] = home[0] + np.pi / 2
    m, _ = M.episode_metrics(base_res(q), lo, hi, ts_of(N, 30), rb, has_env_scene=True)
    check("환경 충돌률", "home TCP에 상자, 100프레임 중 25프레임이 home", m["env_collision_rate"], 0.25)
    check("환경 충돌이 자기 충돌률에 안 섞임", "같은 궤적", m["self_collision_rate"], 0.0)
    m, _ = M.episode_metrics(base_res(q_home), lo, hi, ts_of(N, 30), r, has_env_scene=False)
    check("장면 없을 때 환경 충돌률", "테이블·장애물 장면이 없으면 0이 아니라 NaN", m["env_collision_rate"],
          float("nan"))

    # 8. 실행 경로: 가짜 폴더(4번 npz + 3번 에피소드) → metrics_long.csv
    check("전체 실행 (run)", "가짜 4번·3번 폴더 → metrics_long.csv, 60 fps 에피소드 하나", *run_end_to_end(home), 1)

    write_report(out_dir)
    return all(ok for *_, ok in RESULTS)


def run_end_to_end(home):
    """폴더 구조·파일 형식을 그대로 흉내 내서 run()을 돌린다. (오류 0, 행 수 맞으면 1)"""
    root = Path(tempfile.mkdtemp())
    T, fps, start = 120, 59.97, 95
    (root / "ret" / "A_test").mkdir(parents=True)
    ep = root / "eps" / "T1_E999"
    ep.mkdir(parents=True)
    q = np.tile(home, (T, 1))
    np.savez(root / "ret" / "A_test" / "T1_E999.npz", q=q, status=np.zeros(T, int),
             saturated=np.zeros((T, 7), bool), pos_err=np.full(T, 0.01))
    np.save(ep / "keypoints.npy", np.zeros((T, 7, 3)))
    names = ["wrist", "index_mcp", "pinky_mcp", "thumb_mcp", "x", "thumb_tip", "index_tip"]
    (ep / "keypoint_names.json").write_text(json.dumps(names))
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
        rows = list(csv.DictReader(f))
    with open(out / "episode_info.csv", encoding="utf-8") as f:
        info = list(csv.DictReader(f))
    ok = (len(rows) == len(M.METRICS) and info[0]["status"] == "ok" and info[0]["jerk_step_k"] == "2"
          and rows[0]["case_tags"] == "큰 이동|가장자리")
    return (1 if ok else 0),


def write_report(out_dir):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = sum(ok for *_, ok in RESULTS)
    lines = ["# 지표 코드 합성 데이터 검증 결과", "",
             f"`python metrics/selftest.py` — {n_ok}/{len(RESULTS)} 통과", "",
             "| 검사 | 만드는 법 | 결과 | 정답 | 판정 |", "|---|---|---|---|---|"]
    for name, how, got, want, ok in RESULTS:
        fmt = (lambda v: f"{v:.6g}" if isinstance(v, float) else str(v))
        lines.append(f"| {name} | {how} | {fmt(got)} | {fmt(want)} | {'통과' if ok else '**실패**'} |")
    lines += ["", "- 저크는 3차 차분이 미분의 근사이고 양 끝 3프레임이 빠져서 해석해보다 1% 미만 작게 나온다 (허용 1%).",
              "- 60 fps·59.97 fps 궤적이 30 fps와 같은 값을 내면 30 Hz로 맞추는 처리(정의서 2.6)가 맞게 동작한 것이다.",
              "- 환경 충돌은 원본 모델에 상자 하나를 더한 임시 모델로 확인했다 (실제 장면은 `replay/scene` 작업에서)."]
    (out_dir / "selftest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(RESULTS)} 통과 → {out_dir / 'selftest.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 폴더 (~/project 밖, 예: ~/out/신서연/metrics_selftest)")
    sys.exit(0 if main(ap.parse_args().out) else 1)
