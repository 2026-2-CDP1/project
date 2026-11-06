"""성공률을 '로봇이 할 수 있는 목표'와 '할 수 없는 목표'로 나눠 본다.

할 수 없는 목표 = C(EAIK)가 3번 관절 격자 전체에서 관절 한계 안 '정확한' 해를 하나도 못 찾은 프레임
(C status 2 = 한계 안 해 없음, status 1 = 근사해(LS)만 있음. 목표는 있음).
C는 해를 전부 계산하므로, 이 프레임은 iiwa가 한계 안에서 어떤 자세로도 만들 수 없는 손목 위치·방향이다 (격자 1° 근사).
나머지(할 수 있는 목표)에서 A·B·C 성공률 = 알고리즘 차이.

할 수 없는 프레임의 원인 추정:
  거리   로봇 어깨 → 손목 중심 목표가 0.82 m 초과 (팔이 안 닿음) 또는 0.41 m 미만 (팔꿈치를 다 접어도 모자람)
  방향   거리는 되는데 해가 없음 (손목·관절 한계 때문에 그 방향을 못 만듦)

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_feasible.py --out ~/out/hyojeong/retarget/v1
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import common  # noqa: E402  (같은 retarget 폴더, 성공 판정 기준)

ALGOS = ("A_DLS", "B_mink", "C_eaik")
POS_TOL, ROT_TOL = common.SUCCESS_POS_M, np.radians(common.SUCCESS_ROT_DEG)   # main 에서 --pos-tol-cm 등으로 덮어씀


def wrist_center_dist(p, qt):
    w, x, y, z = qt.T
    zf = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    return np.linalg.norm(p - 0.126 * zf - np.array([0, 0, 0.36]), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default="~/project/episodes/records.csv")
    common.add_success_args(ap)
    a = ap.parse_args()
    global POS_TOL, ROT_TOL
    POS_TOL, ROT_TOL = common.success_tols(a)
    print(f"판정 기준: {common.success_text(a)}")
    out = Path(a.out).expanduser()
    try:
        rec = {r["episode_id"]: r for r in csv.DictReader(open(Path(a.records).expanduser(), encoding="utf-8"))}
    except FileNotFoundError:
        rec = {}

    tot = defaultdict(lambda: defaultdict(int))       # algo -> 칸
    cause = defaultdict(int)
    by_layout = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (task, layout) -> algo -> [성공, 할 수 있는 프레임]
    infeas_layout = defaultdict(lambda: [0, 0])                    # (task, layout) -> [못 하는 프레임, 목표 프레임]
    rows = []
    for pc in sorted((out / "C_eaik").glob("*.npz")):
        eid = pc.stem
        c = np.load(pc)
        has = np.isfinite(c["target_pos"]).all(1)
        # C가 정확한 해를 못 찾은 프레임 = status 2 (한계 안 해 없음) + status 1 (근사해만, 예: 팔이 안 닿는 거리)
        infeas = has & (c["status"] != 0)
        cause["(C 근사해만 = status 1)"] += int((has & (c["status"] == 1)).sum())
        feas = has & ~infeas
        r = wrist_center_dist(c["target_pos"], c["target_quat"])
        far, near = infeas & (r > 0.82), infeas & (r < 0.41)
        cause["거리: 0.82 m 초과 (안 닿음)"] += int(far.sum())
        cause["거리: 0.41 m 미만 (너무 가까움)"] += int(near.sum())
        cause["방향·관절 한계 (거리는 됨)"] += int((infeas & ~far & ~near).sum())
        key = (eid[:2], rec.get(eid, {}).get("layout", "(기록 없음)"))
        infeas_layout[key][0] += int(infeas.sum())
        infeas_layout[key][1] += int(has.sum())
        row = {"episode_id": eid, "target_frames": int(has.sum()), "infeasible_frames": int(infeas.sum())}
        for al in ALGOS:
            p = out / al / f"{eid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            succ = has & (d["tcp_error"] <= POS_TOL) & (d["rot_error"] <= ROT_TOL)
            t = tot[al]
            t["target"] += int(has.sum())
            t["feasible"] += int(feas.sum())
            t["succ_all"] += int(succ.sum())
            t["succ_feasible"] += int((succ & feas).sum())
            t["succ_infeasible"] += int((succ & infeas).sum())
            by_layout[key][al][0] += int((succ & feas).sum())
            by_layout[key][al][1] += int(feas.sum())
            row[f"{al}_feasible_success"] = round((succ & feas).sum() / feas.sum(), 4) if feas.sum() else ""
        rows.append(row)

    T = tot["C_eaik"]["target"]
    F = tot["C_eaik"]["feasible"]
    print(f"== 1. 목표 프레임 {T}개 중 로봇이 할 수 없는 목표(C 해 없음) {T - F}개 ({(T - F) / max(T, 1):.1%})")
    for k, v in cause.items():
        print(f"     {k}: {v} ({v / max(T - F, 1):.0%})")

    print("\n== 2. 성공률: 전체 / 할 수 있는 목표만 / (할 수 없는 목표에서 성공한 프레임 수)")
    for al in ALGOS:
        t = tot[al]
        if not t["target"]:
            continue
        print(f"  {al:7s} | 전체 {t['succ_all'] / t['target']:.1%} | 할 수 있는 목표만 {t['succ_feasible'] / max(t['feasible'], 1):.1%} "
              f"| ({t['succ_infeasible']})")

    print("\n== 3. 배치별: 할 수 없는 목표 비율 | 할 수 있는 목표에서 A·B·C 성공률 — 할 수 없는 비율 높은 순")
    for key in sorted(infeas_layout, key=lambda k: -infeas_layout[k][0] / max(infeas_layout[k][1], 1)):
        n_inf, n = infeas_layout[key]
        g = by_layout[key]
        print(f"  {key[0]} {key[1][:36]:36s} | 못 함 {n_inf / max(n, 1):5.1%} | " + " | ".join(
            f"{al[0]} {g[al][0] / g[al][1]:6.1%}" if g[al][1] else f"{al[0]}    -  " for al in ALGOS))

    with open(out / "feasible_by_episode.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["episode_id", "target_frames", "infeasible_frames"] + [f"{al}_feasible_success" for al in ALGOS]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out / 'feasible_by_episode.csv'}")


if __name__ == "__main__":
    main()
