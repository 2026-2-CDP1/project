"""실패가 구간 어디서 나는지: 처음 0.5초 / 가운데 / 마지막 0.5초로 나눈 실패율 (태스크별) — 3번 요청 (11/6).

프레임 판정은 check_success.py 와 같음 (손목 1 cm · 5° · 한계 안, 목표 NaN 프레임은 뺌).
할 수 없는 목표 = C가 정확한 해를 못 찾은 프레임 (check_feasible.py 와 같음).
구간이 1초보다 짧은 에피소드(T2 빠른 조건 등)는 가운데 없이 앞 절반 = 처음, 뒤 절반 = 끝으로 셈 (개수를 따로 표시).
T2 는 빠른 조건(E026~E035, E068, E069)을 따로 한 줄 더 보여 줌.

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_where.py --out ~/out/hyojeong/retarget/v4
출력: 화면 표 + <out>/where_by_episode.csv (에피소드 × 위치별 목표 수·할 수 없는 수·A·B·C 실패 수)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGOS = ("A_DLS", "B_mink", "C_eaik")
POS_TOL, ROT_TOL, LIM_EPS = 0.01, np.radians(5.0), 1e-6
PARTS = ("처음", "가운데", "끝")
T2_FAST = {f"T2_E{n:03d}" for n in list(range(26, 36)) + [68, 69]}


def wrist_center_dist(p, qt):
    """로봇 어깨(베이스 위 0.36 m) → 손목 중심(플랜지에서 축 따라 0.126 m 안쪽) 거리. check_feasible.py 와 같음."""
    w, x, y, z = qt.T
    zf = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    return np.linalg.norm(p - 0.126 * zf - np.array([0, 0, 0.36]), axis=1)


def parts(T, fps, edge_s):
    """프레임마다 0 처음 / 1 가운데 / 2 끝."""
    n = int(round(edge_s * fps))
    idx = np.arange(T)
    if T < 2 * n:                       # 짧은 구간: 반씩
        return np.where(idx < T / 2, 0, 2), True
    return np.where(idx < n, 0, np.where(idx >= T - n, 2, 1)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge", type=float, default=0.5, help="처음·끝 길이 (초)")
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    lim = list(csv.DictReader(open(Path(a.limits).expanduser(), encoding="utf-8")))
    lo, hi = np.array([float(r["lower_rad"]) for r in lim]), np.array([float(r["upper_rad"]) for r in lim])

    # 그룹 -> 위치 -> [목표, 할 수 없음, A 실패, B 실패, C 실패, 안 닿음, 너무 가까움]
    agg = defaultdict(lambda: np.zeros((3, 4 + len(ALGOS)), int))
    short = defaultdict(int)
    rows = []
    for pc in sorted((out / "C_eaik").glob("*.npz")):
        eid = pc.stem
        c = np.load(pc)
        info = json.loads(str(c["info"]))
        has = np.isfinite(c["target_pos"]).all(1)
        infeas = has & (c["status"] != 0)
        r = wrist_center_dist(c["target_pos"], c["target_quat"])
        far, near = infeas & (r > 0.82), infeas & (r < 0.41)
        part, is_short = parts(len(has), float(info.get("exo_fps", 60.0)), a.edge)
        fails = []
        for al in ALGOS:
            p = out / al / f"{eid}.npz"
            if not p.exists():
                fails.append(None)
                continue
            d = np.load(p)
            inlim = np.all((d["q"] >= lo - LIM_EPS) & (d["q"] <= hi + LIM_EPS), axis=1)
            ok = (d["tcp_error"] <= POS_TOL) & (d["rot_error"] <= ROT_TOL) & inlim
            fails.append(has & ~ok)
        groups = [eid[:2]] + (["T2 빠름"] if eid in T2_FAST else ["T2 나머지"] if eid.startswith("T2") else [])
        for g in groups + ["전체"]:
            short[g] += is_short
        row = {"episode_id": eid, "short": int(is_short)}
        for k in range(3):
            m = part == k
            v = ([int((has & m).sum()), int((infeas & m).sum())] + [int((f & m).sum()) if f is not None else 0 for f in fails]
                 + [int((far & m).sum()), int((near & m).sum())])
            for g in groups + ["전체"]:
                agg[g][k] += v
            row.update({f"{PARTS[k]}_목표": v[0], f"{PARTS[k]}_할수없음": v[1],
                        **{f"{PARTS[k]}_{al[0]}실패": v[2 + i] for i, al in enumerate(ALGOS)},
                        f"{PARTS[k]}_안닿음": v[-2], f"{PARTS[k]}_너무가까움": v[-1]})
        rows.append(row)

    print(f"== 위치별 실패율 (처음·끝 = {a.edge}초, 목표 있는 프레임 기준). 괄호 = 목표 프레임 수")
    print("  그룹        위치   (목표)  | 할 수 없음 (안 닿음 / 너무 가까움) |  A 실패  B 실패  C 실패")
    order = ["T1", "T2", "T2 나머지", "T2 빠름", "T3", "전체"]
    for g in order:
        if g not in agg:
            continue
        for k in range(3):
            v = agg[g][k]
            if not v[0]:
                continue
            r = lambda x: f"{100 * x / v[0]:5.1f}%"
            print(f"  {g:9s} {PARTS[k]:4s} ({v[0]:6d}) |   {r(v[1])} ({r(v[5])} / {r(v[6])}) | {r(v[2])}  {r(v[3])}  {r(v[4])}")
        print(f"  {'':9s} (1초보다 짧아 반씩 나눈 에피소드 {short[g]}개)")
    with open(out / "where_by_episode.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out / 'where_by_episode.csv'}")


if __name__ == "__main__":
    main()
