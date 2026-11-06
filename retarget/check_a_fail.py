"""A(DLS)가 '할 수 있는 목표'에서 실패한 프레임의 원인 진단 (A 수정 전에 먼저 확인).

할 수 있는 목표 = C 가 정확한 해를 찾은 프레임 (check_feasible.py 와 같은 기준).
그중 A 실패 프레임에 대해:
  1) 관절 한계에 걸렸나: A saturated(마지막 반복에서 한계로 잘린 관절)가 하나라도 있는 비율, 관절별 횟수
     → 높으면 원인 = 한계를 '잘라내기(clip)'로 처리해서 그 관절이 멈추고 나머지 관절이 보정을 못 함
  2) 같은 프레임에서 B 는 성공했나, 성공했다면 A·B 자세가 얼마나 다른가 (관절별 차이 중앙값, q2·q4·q6 부호가 다른 비율)
     → 부호가 다르면 '다른 갈래(branch)'에 갇힌 것
  3) 실패가 길게 이어지나: 실패 구간 길이 분포 (한 번 갇히면 오래 못 빠져나오는지)

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_a_fail.py --out ~/out/hyojeong/retarget/v1
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

POS_TOL, ROT_TOL, LIM_EPS = 0.01, np.radians(5.0), 1e-6


def runs(mask):
    """True 가 이어지는 구간 길이들."""
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    lim = list(csv.DictReader(open(Path(a.limits).expanduser(), encoding="utf-8")))
    lo, hi = np.array([float(r["lower_rad"]) for r in lim]), np.array([float(r["upper_rad"]) for r in lim])

    n_feas = n_fail = n_sat = n_b_ok = n_sign = n_atlim = 0
    sat_joint, atlim_joint = np.zeros(7, int), np.zeros(7, int)
    dq_all, run_len, worst = [], [], []
    for pc in sorted((out / "C_eaik").glob("*.npz")):
        eid = pc.stem
        pa, pb = out / "A_DLS" / f"{eid}.npz", out / "B_mink" / f"{eid}.npz"
        if not (pa.exists() and pb.exists()):
            continue
        c, A, B = np.load(pc), np.load(pa), np.load(pb)
        has = np.isfinite(c["target_pos"]).all(1)
        feas = has & (c["status"] == 0)

        def ok(d):
            inlim = np.all((d["q"] >= lo - LIM_EPS) & (d["q"] <= hi + LIM_EPS), axis=1)
            return (d["tcp_error"] <= POS_TOL) & (d["rot_error"] <= ROT_TOL) & inlim

        fa = feas & ~ok(A)
        bo = fa & ok(B)
        n_feas += int(feas.sum())
        n_fail += int(fa.sum())
        sat = A["saturated"][fa]
        n_sat += int(sat.any(1).sum())
        sat_joint += sat.sum(0)
        # 한계에 '붙어 있는' 관절 (잘림 표시와 별개로 실제 값이 한계 0.5° 안)
        near = (A["q"][fa] <= lo + np.radians(0.5)) | (A["q"][fa] >= hi - np.radians(0.5))
        n_atlim += int(near.any(1).sum())
        atlim_joint += near.sum(0)
        n_b_ok += int(bo.sum())
        if bo.any():
            qa, qb = A["q"][bo], B["q"][bo]
            dq_all.append(np.abs(qa - qb))
            n_sign += int((np.sign(qa[:, [1, 3, 5]]) != np.sign(qb[:, [1, 3, 5]])).any(1).sum())
        r = runs(fa)
        run_len += r
        if fa.sum():
            worst.append((int(fa.sum()), int(feas.sum()), eid, max(r)))

    if not n_fail:
        print("할 수 있는 목표에서 A 실패 없음")
        return
    print(f"== 할 수 있는 목표 {n_feas}프레임 중 A 실패 {n_fail} ({n_fail / n_feas:.1%})")
    print(f"\n== 1. 관절 한계")
    print(f"  A 실패 중 한계로 잘린 관절이 있는 프레임: {n_sat} ({n_sat / n_fail:.1%})")
    print(f"  A 실패 중 관절이 한계 0.5° 안에 붙어 있는 프레임: {n_atlim} ({n_atlim / n_fail:.1%})")
    print("  관절별 (잘림 / 붙어 있음): " + "  ".join(f"q{j + 1} {sat_joint[j]}/{atlim_joint[j]}" for j in range(7)))
    print(f"\n== 2. 같은 프레임 B")
    print(f"  A 실패 중 B 성공: {n_b_ok} ({n_b_ok / n_fail:.1%})")
    if dq_all:
        dq = np.degrees(np.concatenate(dq_all))
        print("  그 프레임에서 |A − B| 관절별 중앙값(°): " + "  ".join(f"q{j + 1} {np.median(dq[:, j]):.0f}" for j in range(7)))
        print(f"  q2·q4·q6 중 부호가 다른 관절이 있는 비율 (다른 갈래): {n_sign / n_b_ok:.1%}")
    print(f"\n== 3. 실패 구간 길이 (프레임)")
    rl = np.array(run_len)
    print(f"  구간 {len(rl)}개 | 중앙 {np.median(rl):.0f} | 90% {np.percentile(rl, 90):.0f} | 최대 {rl.max()}")
    print(f"  30프레임(1초) 이상 구간이 차지하는 실패 프레임: {rl[rl >= 30].sum() / rl.sum():.1%}")
    c = Counter("1-2" if x <= 2 else "3-9" if x < 10 else "10-29" if x < 30 else "30+" for x in rl)
    print("  길이별 구간 수: " + "  ".join(f"{k} {c[k]}" for k in ("1-2", "3-9", "10-29", "30+")))
    print(f"\n== A 실패가 많은 에피소드 10개 (실패 / 할 수 있는 프레임, 최장 구간)")
    for nf, nt, eid, mx in sorted(worst, reverse=True)[:10]:
        print(f"  {eid}  {nf:5d} / {nt:5d}  최장 {mx}")


if __name__ == "__main__":
    main()
