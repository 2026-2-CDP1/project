"""4번 리타게팅 일괄 실행: 3번 에피소드 전체 × 알고리즘(A_DLS, B_mink, C_eaik)을 config.json 설정 하나로 푼다.

- 설정은 retarget/config.json 에서만 읽는다 (모든 에피소드·알고리즘에 같은 설정)
- 목표(targets)는 에피소드마다 한 번 만들어 세 알고리즘이 같이 쓴다 -> info 가 알고리즘끼리 같음 (5-2 확인 조건)
- 이미 npz가 있으면 건너뜀 (--force 로 다시). 중간에 끊겨도 다시 실행하면 이어서 진행
- 실패는 멈추지 않고 summary.csv 에 이유와 함께 기록 (실패도 지우지 않는다)
- <out>/run_info.json 에 노드 이름·설정·패키지 버전·git 커밋을 남긴다 (5-2 정의서 2.3 측정 조건)

출력 (5-2 metrics.py --retarget <out> 로 그대로 읽음):
  <out>/<알고리즘>/<에피소드ID>.npz
  <out>/summary.csv      에피소드 × 알고리즘 한 줄 요약 (점검용)
  <out>/run_info.json    실행 조건

실행 (서버, ~/project 에서 source ~/use_robot.sh 후). 계산 시간 비교를 위해 같은 노드(-w), 작업 1개(-n 1), CPU 1개(-c 1):
  sbatch -p p02 -w DIS01 -n 1 -c 1 --mem=4G --job-name retarget_본인이름 -o ~/out/본인이름/retarget/run_all_%j.log \\
    --wrap "python retarget/run_all.py --episodes <3번 episodes 폴더> --out ~/out/본인이름/retarget"
  알고리즘별로 나눠 동시에: 같은 -w 노드로 sbatch 를 세 번, 각각 --algos A_DLS / B_mink / C_eaik
  시험: --ids "T1_E00*" 또는 --limit 3
"""
import argparse
import csv
import fnmatch
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# 레포 밖(예: ~/out/본인이름/)에 복사해 시험할 때는 ~/project/robot 의 모델을 쓴다
ROBOT_DIR = REPO / "robot" if (REPO / "robot" / "robot.urdf").exists() else Path("~/project/robot").expanduser()
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ALGOS = ("A_DLS", "B_mink", "C_eaik")
DEFAULT_EXCLUDE = ["T2_E008"]   # sync 시각이 겹치지 않음 (episodes/DATA_STRUCTURE.md 5절)
SUMMARY_COLS = ["episode_id", "algo", "state", "note", "frames", "target_frames", "n_ok", "n_maxiter", "n_status2",
                "pos_err_p95_mm", "pos_err_max_mm", "n_pos_over_1cm", "rot_err_p95_deg", "swivel_med_deg", "swivel_p95_deg",
                "time_med_ms", "time_p95_ms", "elbow_used_rate", "n_orient_held", "n_wrist_manual",
                "n_elbow_backfilled", "n_restarted", "sec", "node", "finished_at"]


# ---------------------------------------------------------------- 설정 · 풀이기
def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("common", *ALGOS):
        if k not in cfg:
            raise ValueError(f"config.json 에 '{k}' 가 없음")
    return cfg


def make_solver(algo, cfg):
    """알고리즘 모듈의 DEFAULT 를 config.json 값으로 덮어 풀이기를 만든다 (모델 로드는 한 번만)."""
    if algo == "A_DLS":
        import dls as mod
        kw = {k: cfg["A_DLS"].get(k, v) for k, v in mod.DEFAULT.items()}
        return mod, mod.DLS(ROBOT_DIR / "robot.urdf", **kw)
    if algo == "B_mink":
        import mink_ik as mod
        kw = {k: cfg["B_mink"].get(k, v) for k, v in mod.DEFAULT.items()}
        return mod, mod.MinkIK(ROBOT_DIR / "robot.xml", **kw)
    if algo == "C_eaik":
        import eaik_ik as mod
        kw = {k: cfg["C_eaik"].get(k, v) for k, v in mod.DEFAULT.items()}
        return mod, mod.EaikIK(ROBOT_DIR / "robot.urdf", cfg["C_eaik"].get("select", "human"), **kw)
    raise ValueError(algo)


def solve(algo, mod, solver, targets, cfg):
    """run_episode 반환을 (q, sat, status, pos_err, rot_err, extra dict) 로 맞춘다."""
    if algo == "C_eaik":
        q, sat, status, pe, re, swv, n_cand, q3, time_ms, used = mod.run_episode(solver, targets)
        extra = dict(rot_error=re, swivel_error=swv, n_candidates=n_cand, q3_pick=q3, time_ms=time_ms, elbow_used=used)
        if cfg["C_eaik"].get("select") == "sew" and hasattr(solver, "episode_sew_delta"):   # SEW 고정: 사람 각에서 벗어난 각
            extra.update(sew_delta=solver.episode_sew_delta)
    else:
        q, sat, status, pe, re, swv, iters, time_ms, used, restarted = mod.run_episode(
            solver, targets, restart_tol=cfg[algo].get("restart_tol", 0.0))
        extra = dict(rot_error=re, swivel_error=swv, iters=iters, time_ms=time_ms, elbow_used=used, restarted=restarted)
        if hasattr(solver, "episode_limit"):   # A: 프레임별 관절 한계에 걸린 반복 수·관절
            extra.update(limit_iters=solver.episode_limit[0], limit_joints=solver.episode_limit[1])
    return q, sat, status, pe, re, extra


# ---------------------------------------------------------------- 에피소드 목록 · 요약
def find_episodes(root, ids, exclude, limit):
    root = Path(root).expanduser()
    eps = sorted(d for d in root.iterdir() if d.is_dir() and (d / "keypoints.npy").exists() and (d / "meta.json").exists())
    if ids:
        eps = [d for d in eps if any(fnmatch.fnmatch(d.name, p) for p in ids)]
    keep = [d for d in eps if d.name not in exclude]
    skipped = [d for d in eps if d.name in exclude]
    return (keep[:limit] if limit else keep), skipped


def global_scale(root, exclude, common_cfg):
    """scale = "arm_global": 모든 에피소드(제외 목록 빼고, --ids·--limit 와 무관)의 사람 팔 길이(위팔+아래팔,
    에피소드별 중앙값)의 중앙값 하나로 축척 k = 0.82 / 그 값. 같은 사람이 찍은 데이터라 팔 길이는 하나여야 하고,
    에피소드별로 하면 키포인트 잡음(예: T1_E036 팔꿈치)이 그 에피소드 궤적 전체 크기를 바꿔 버림 (v1.1).
    config 에 human_arm_m 숫자가 있으면 그 값을 쓴다. 반환 (k, 팔 길이, {에피소드: 팔 길이}) / arm_global 이 아니면 None."""
    if common_cfg["scale"] != "arm_global":
        return None
    per = {}
    fixed = common_cfg.get("human_arm_m")
    if fixed is None:
        for d in sorted(Path(root).expanduser().iterdir()):
            if d.is_dir() and d.name not in exclude and (d / "keypoints.npy").exists():
                names = json.loads((d / "keypoint_names.json").read_text())
                per[d.name] = common.human_arm_length(np.load(d / "keypoints.npy"), names)
        vals = np.array([v for v in per.values() if np.isfinite(v)])
        if not len(vals):
            raise ValueError("사람 팔 길이를 구할 에피소드가 없음")
        arm = float(np.median(vals))
    else:
        arm = float(fixed)
    return common.IIWA_ARM / arm, arm, per


def p(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summary_row(eid, algo, state, note, node, sec=float("nan"), q=None, status=None, pe=None, re=None,
                extra=None, info=None):
    row = {c: "" for c in SUMMARY_COLS}
    row.update(episode_id=eid, algo=algo, state=state, note=note, node=node,
               finished_at=datetime.now().isoformat(timespec="seconds"))
    if q is None:
        return row
    swv = np.degrees(extra["swivel_error"])
    # target_frames: 목표가 있어 IK를 푼 프레임 (= time_ms 가 있는 프레임. C의 '한계 안 해 없음'도 푼 것이라 포함)
    row.update(frames=len(q), target_frames=int(np.sum(np.isfinite(extra["time_ms"]))),
               n_ok=int(np.sum(status == 0)), n_maxiter=int(np.sum(status == 1)), n_status2=int(np.sum(status == 2)),
               pos_err_p95_mm=round(1000 * p(pe, 95), 4), pos_err_max_mm=round(1000 * p(pe, 100), 2),
               n_pos_over_1cm=int(np.nansum(pe > 0.01)), rot_err_p95_deg=round(float(np.degrees(p(re, 95))), 4),
               swivel_med_deg=round(p(swv, 50), 2), swivel_p95_deg=round(p(swv, 95), 2),
               time_med_ms=round(p(extra["time_ms"], 50), 3), time_p95_ms=round(p(extra["time_ms"], 95), 3),
               elbow_used_rate=round(float(np.mean(extra["elbow_used"])), 4),
               n_orient_held=info.get("n_orient_held", ""), n_wrist_manual=info.get("n_wrist_manual", ""),
               n_elbow_backfilled=info.get("n_elbow_backfilled", ""), sec=round(sec, 2),
               n_restarted=int(np.sum(extra["restarted"])) if "restarted" in extra else "")
    return row


def append_summary(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def package_versions():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "pinocchio", "mujoco", "mink", "qpsolvers", "eaik"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[name] = f"없음 ({type(e).__name__})"
    return out


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description="4번 리타게팅 일괄 실행")
    ap.add_argument("--episodes", required=True, help="3번 결과 폴더 (그 아래 <에피소드ID>/keypoints.npy ...)")
    ap.add_argument("--out", required=True, help="결과 폴더 (~/project 밖). <out>/<알고리즘>/<ID>.npz")
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=ALGOS)
    ap.add_argument("--ids", nargs="*", help='에피소드 ID 패턴 (예: "T1_*" "T2_E01?")')
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="제외할 에피소드 ID")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (시험용)")
    ap.add_argument("--force", action="store_true", help="이미 있는 npz도 다시 계산")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    c = cfg["common"]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # 알고리즘별로 나눠 동시에 돌릴 때 같은 파일에 동시에 쓰지 않도록, 일부만 돌리면 summary_<알고리즘>.csv
    algos = list(dict.fromkeys(a.algos))
    summary = out / ("summary.csv" if set(algos) == set(ALGOS) else f"summary_{'_'.join(algos)}.csv")
    node = socket.gethostname()
    eps, excluded = find_episodes(a.episodes, a.ids, set(a.exclude or []), a.limit)
    print(f"노드 {node} | 에피소드 {len(eps)}개 (제외 {len(excluded)}) | 알고리즘 {' '.join(algos)} | 결과 {out}", flush=True)
    gs = global_scale(a.episodes, set(a.exclude or []), c)
    scale_arg = gs[0] if gs else c["scale"]
    if gs:
        arms = np.array(list(gs[2].values())) if gs[2] else np.array([gs[1]])
        print(f"  축척 (전체 중앙값): 사람 팔 {gs[1]:.4f} m → k = {gs[0]:.4f}  "
              f"(에피소드 {len(gs[2])}개, 팔 길이 범위 {np.nanmin(arms):.3f}~{np.nanmax(arms):.3f} m)", flush=True)

    run_info = {"started_at": datetime.now().isoformat(timespec="seconds"), "node": node,
                "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
                "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "git_commit": git_commit(), "args": vars(a), "config": cfg, "packages": package_versions(),
                "scale": ({"mode": "arm_global", "k": gs[0], "human_arm_m": gs[1], "per_episode_arm_m": gs[2]}
                          if gs else {"mode": str(c["scale"])}),
                "episodes": [d.name for d in eps], "excluded": [d.name for d in excluded]}
    (out / f"run_info_{run_info['slurm_job'] or 'local'}.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    for d in excluded:
        for algo in algos:
            append_summary(summary, summary_row(d.name, algo, "excluded", "제외 목록", node))

    solvers = {}
    for algo in algos:
        t0 = time.time()
        solvers[algo] = make_solver(algo, cfg)
        print(f"  {algo} 준비 {time.time() - t0:.1f} s", flush=True)

    counts = {"ok": 0, "skipped": 0, "error": 0}
    for i, d in enumerate(eps, 1):
        eid = d.name
        todo = [al for al in algos if a.force or not (out / al / f"{eid}.npz").exists()]
        if not todo:
            counts["skipped"] += len(algos)
            continue
        try:
            ep = common.load_episode(d)
            targets = common.make_targets(ep, scale=scale_arg, r_off=c["r_off"], anchor=c["anchor"],
                                          mount=c["mount"], base_z=c["base_z"], pos_only=c["pos_only"])
            if gs:   # info 에 전체 중앙값 축척임을 남김 (human_arm_m 은 그 에피소드 자체 값 그대로)
                targets["info"].update(scale_mode="arm_global", human_arm_global_m=gs[1])
        except Exception as e:  # noqa: BLE001
            for algo in todo:
                append_summary(summary, summary_row(eid, algo, "error", f"목표 만들기: {type(e).__name__}: {e}", node))
            counts["error"] += len(todo)
            print(f"[{i}/{len(eps)}] {eid} 목표 만들기 실패: {e}", flush=True)
            continue
        msg = []
        for algo in todo:
            mod, solver = solvers[algo]
            try:
                t0 = time.time()
                q, sat, status, pe, re, extra = solve(algo, mod, solver, targets, cfg)
                sec = time.time() - t0
                common.save_result(out / algo / f"{eid}.npz", q, sat, status, pe, targets, **extra)
                append_summary(summary, summary_row(eid, algo, "ok", "", node, sec, q, status, pe, re, extra, targets["info"]))
                counts["ok"] += 1
                msg.append(f"{algo} {sec:.1f}s")
            except Exception as e:  # noqa: BLE001
                append_summary(summary, summary_row(eid, algo, "error", f"{type(e).__name__}: {e}", node))
                counts["error"] += 1
                msg.append(f"{algo} 오류")
                traceback.print_exc()
        print(f"[{i}/{len(eps)}] {eid} | " + " | ".join(msg), flush=True)

    print(f"끝: 계산 {counts['ok']} / 건너뜀(이미 있음) {counts['skipped']} / 오류 {counts['error']} | 요약 {summary}", flush=True)


if __name__ == "__main__":
    main()
