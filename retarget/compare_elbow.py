"""사람 팔꿈치 입력 유무 비교 (results_v1_2.md 8-1절): A·B·C 를 팔꿈치 없이 / 팔꿈치 써서 한 표로, 전체와 태스크별.

분모는 할 수 있는 목표 (--ref 폴더 C_eaik 의 status 0, 모든 실행에 같은 프레임).
판정은 check_success.py 와 같음 (common.SUCCESS_*: 손목 0.5 cm · 1° · 한계 안).

  팔 회전각 30° 안   사람과 팔 회전각(swivel) 차이가 30° 이하인 프레임 (팔꿈치가 사람과 비슷한 자세)
  둘 다             손목 성공이면서 30° 안
  한계에 붙음        관절 하나라도 한계에서 --sat-deg 안
  30° 튐            프레임 사이 관절 최대 변화가 30° 넘는 횟수 (모든 프레임)

사용 (서버, source ~/use_robot.sh 후, ~/project 에서):
  python retarget/compare_elbow.py --root ~/out/hyojeong/retarget
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import common  # noqa: E402  (같은 retarget 폴더, 성공 판정 기준)

DEFAULT_RUNS = ["A 없음=v1_2/A_DLS", "A 있음 0.5=v4/A_DLS", "B 없음=v1_2/B_mink", "B 있음 0.1=v4/B_mink",
                "C 없음(연속)=v1_3_Ccont/C_eaik", "C 있음(SEW)=v1_2_Csew/C_eaik"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="실행 폴더들이 있는 곳 (예: ~/out/hyojeong/retarget)")
    ap.add_argument("--ref", default="v1_2", help="할 수 있는 목표를 정하는 실행 폴더 (C_eaik 사용)")
    ap.add_argument("--run", action="append", help='"이름=폴더/알고리즘" (여러 번). 없으면 팔꿈치 유무 기본 6개')
    ap.add_argument("--swivel-deg", type=float, default=30.0, help="사람과 비슷한 자세로 보는 팔 회전각 차이 (도)")
    ap.add_argument("--sat-deg", type=float, default=np.degrees(1e-3), help="관절 한계에 붙음으로 보는 거리 (도)")
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    common.add_success_args(ap)
    a = ap.parse_args()
    pos_tol, rot_tol = common.success_tols(a)
    root = Path(a.root).expanduser()
    lim = list(csv.DictReader(open(Path(a.limits).expanduser(), encoding="utf-8")))
    lo, hi = np.array([float(r["lower_rad"]) for r in lim]), np.array([float(r["upper_rad"]) for r in lim])
    sat = np.radians(a.sat_deg)
    ids = sorted(p.stem for p in (root / a.ref / "C_eaik").glob("*.npz"))
    runs = [s.split("=", 1) for s in (a.run or DEFAULT_RUNS)]
    print(f"판정 기준: {common.success_text(a)} | 분모 = 할 수 있는 목표 ({a.ref}/C_eaik status 0)")

    for task in ["전체"] + sorted({e[:2] for e in ids}):
        eids = ids if task == "전체" else [e for e in ids if e.startswith(task)]
        print(f"\n== {task} (에피소드 {len(eids)}개)")
        print(f"  {'':14s} | 손목 성공 | 팔 회전각 차 중앙 | {a.swivel_deg:.0f}° 안 | 둘 다 | 한계에 붙음 | 30° 튐 | 시간 중앙 / 95% ms")
        for name, path in runs:
            n = ok = near_s = both = sat_n = jump = 0
            sw, tm = [], []
            for e in eids:
                c = np.load(root / a.ref / "C_eaik" / f"{e}.npz")
                has = np.isfinite(c["target_pos"]).all(1)
                feas = has & (c["status"] == 0)
                d = np.load(root / path / f"{e}.npz")
                q = d["q"]
                inlim = np.all((q >= lo - 1e-6) & (q <= hi + 1e-6), axis=1)
                good = (d["tcp_error"] <= pos_tol) & (d["rot_error"] <= rot_tol) & inlim
                s = np.degrees(d["swivel_error"])
                close = np.isfinite(s) & (s <= a.swivel_deg)
                near = ((q - lo < sat) | (hi - q < sat)).any(1)
                n += feas.sum(); ok += (good & feas).sum(); near_s += (close & feas).sum()
                both += (close & good & feas).sum(); sat_n += (near & feas).sum()
                sw += list(s[feas & np.isfinite(s)])
                t = d["time_ms"][has][1:]
                tm += list(t[np.isfinite(t)])
                jump += int((np.degrees(np.abs(np.diff(q, axis=0))).max(1) > 30).sum())
            print(f"  {name:14s} | {100 * ok / n:6.1f}% | {np.median(sw):8.1f}° | {100 * near_s / n:5.1f}% | "
                  f"{100 * both / n:5.1f}% | {100 * sat_n / n:6.1f}% | {jump:4d} | {np.median(tm):5.1f} / {np.percentile(tm, 95):6.1f}")
    print(f"\n할 수 있는 목표 기준 (에피소드 {len(ids)}개). v4 = 설정 v1.1 (A·B 팔꿈치 입력, 그 외 v1.2와 같음)")


if __name__ == "__main__":
    main()
