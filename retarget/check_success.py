"""리타게팅 성공률 요약 — 5-2 지표 정의서 2.1 기준 (재생 대신 4번 npz의 오차로 빠르게).

success[t] = 손목 위치 오차 ≤ 1 cm  and  방향 오차 ≤ 5°  and  모든 관절이 한계 안 (ε = 1e-6 rad)
목표가 NaN인 프레임(손목 미검출)은 분모에서 뺌. 에피소드 성공률 = 성공 프레임 / 목표 있는 프레임.
(4번 tcp_error·rot_error 는 같은 robot.urdf FK로 계산해 5-2 재생 결과와 같다. 공식 값은 5-2 metrics.py)

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_success.py --out ~/out/hyojeong/retarget/v1
출력: 화면 요약 + <out>/success_by_episode.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGOS = ("A_DLS", "B_mink", "C_eaik")
POS_TOL, ROT_TOL, LIM_EPS = 0.01, np.radians(5.0), 1e-6


def load_limits(path):
    lo, hi = [], []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        lo.append(float(r["lower_rad"]))
        hi.append(float(r["upper_rad"]))
    return np.array(lo), np.array(hi)


def load_records(path):
    try:
        return {r["episode_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    except FileNotFoundError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run_all.py --out 폴더")
    ap.add_argument("--records", default="~/project/episodes/records.csv")
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    lo, hi = load_limits(Path(a.limits).expanduser())
    rec = load_records(Path(a.records).expanduser())

    rows = []
    for al in ALGOS:
        for p in sorted((out / al).glob("*.npz")):
            d = np.load(p)
            has = np.isfinite(d["target_pos"]).all(1)
            q = d["q"]
            inlim = np.all((q >= lo - LIM_EPS) & (q <= hi + LIM_EPS), axis=1)
            pe, re = d["tcp_error"], d["rot_error"]
            ok_pos, ok_rot = has & (pe <= POS_TOL), has & (re <= ROT_TOL)
            succ = ok_pos & ok_rot & inlim
            n = int(has.sum())
            r = rec.get(p.stem, {})
            rows.append({"episode_id": p.stem, "task": p.stem[:2], "algo": al, "layout": r.get("layout", ""),
                         "hand": r.get("hand", ""), "frames": len(q), "target_frames": n, "success_frames": int(succ.sum()),
                         "success_rate": succ.sum() / n if n else np.nan,
                         "fail_pos": int((has & ~ok_pos).sum()), "fail_rot_only": int((ok_pos & ~ok_rot).sum()),
                         "out_of_limit": int((has & ~inlim).sum())})
    if not rows:
        print("결과 npz 없음")
        return

    with open(out / "success_by_episode.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("== 1. 알고리즘별 성공률 (손목 1 cm · 5° · 한계 안, 목표 없는 프레임 제외)")
    print("  algo    | 에피소드 | 전체 프레임 기준 | 에피소드 중앙 | 100% 성공 | 90% 이상 | 50~90% | 50% 미만 | 0%   | 실패 원인 (위치 / 방향만 / 한계 밖 프레임)")
    for al in ALGOS:
        R = [r for r in rows if r["algo"] == al]
        if not R:
            print(f"  {al:7s} | 결과 없음")
            continue
        s = np.array([r["success_rate"] for r in R])
        tot = sum(r["success_frames"] for r in R) / max(sum(r["target_frames"] for r in R), 1)
        print(f"  {al:7s} | {len(R):6d} | {tot:13.1%} | {np.nanmedian(s):10.1%} | {np.sum(s >= 0.9999):7d} | "
              f"{np.sum(s >= 0.9):6d} | {np.sum((s >= 0.5) & (s < 0.9)):5d} | {np.sum(s < 0.5):6d} | {np.sum(s <= 0.0001):4d} | "
              f"{sum(r['fail_pos'] for r in R)} / {sum(r['fail_rot_only'] for r in R)} / {sum(r['out_of_limit'] for r in R)}")

    print("\n== 2. 태스크별 (에피소드 성공률 평균)")
    for t in ("T1", "T2", "T3"):
        print(f"  {t}: " + " | ".join(
            f"{al} {np.nanmean([r['success_rate'] for r in rows if r['algo'] == al and r['task'] == t]):.1%}"
            for al in ALGOS if any(r["algo"] == al and r["task"] == t for r in rows)))

    print("\n== 3. 배치(records.csv layout)별 (에피소드 성공률 평균, 에피소드 수) — A 성공률 낮은 순")
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["task"], r["layout"] or "(기록 없음)")][r["algo"]].append(r["success_rate"])
    order = sorted(groups, key=lambda k: np.nanmean(groups[k]["A_DLS"]) if groups[k]["A_DLS"] else 1)
    for k in order:
        g = groups[k]
        n = max(len(v) for v in g.values())
        print(f"  {k[0]} {k[1][:38]:38s} ({n:2d}) | " + " | ".join(
            f"{al[0]} {np.nanmean(g[al]):6.1%}" if g[al] else f"{al[0]}    -  " for al in ALGOS))

    print("\n== 4. 성공률이 가장 낮은 에피소드 (알고리즘별 5개)")
    for al in ALGOS:
        R = sorted([r for r in rows if r["algo"] == al], key=lambda r: r["success_rate"])[:5]
        print(f"  {al}: " + ", ".join(f"{r['episode_id']} {r['success_rate']:.0%}" for r in R))
    print(f"\n저장: {out / 'success_by_episode.csv'}")


if __name__ == "__main__":
    main()
