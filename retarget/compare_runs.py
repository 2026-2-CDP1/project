"""같은 데이터로 돌린 여러 실행(설정 한 항목만 바꾼 비교)을 한 표로: 성공률, 팔 회전각 차이, 관절 한계, 튐, 시간.

실행 하나 = "이름=폴더/알고리즘" (폴더는 --root 아래, 알고리즘 폴더에 <ID>.npz).
목표 있는 프레임·할 수 있는 목표(C status 0)는 --ref 폴더의 C_eaik 로 정한다 (모든 실행에 같은 분모).
판정은 check_success.py 와 같음 (common.SUCCESS_*: 손목 0.5 cm · 1° · 한계 안).

  관절 한계에 붙음  관절이 한계에서 --sat-deg 안 (기본 1e-3 rad ≈ 0.057°)
  30° 넘는 튐      프레임 사이 관절 최대 변화가 30° 넘는 횟수
  A 기록           npz 에 limit_iters 가 있으면 (dls.py) 할 수 있는 목표에서 한계에 걸린 프레임과 그 성공
  C sew 기록       npz 에 sew_delta 가 있으면 SEW 로 직접 푼 프레임 수와 사람 각에서 벗어난 각

사용 (서버, source ~/use_robot.sh 후):
  python retarget/compare_runs.py --root ~/out/hyojeong/retarget --ref v1_2 \\
      --run "A clip=v1_2/A_DLS" --run "A clamp=v1_2_Aclamp/A_DLS" --run "B=v1_2/B_mink" \\
      --run "C 격자=v1_2/C_eaik" --run "C SEW=v1_2_Csew/C_eaik"
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import common  # noqa: E402  (같은 retarget 폴더, 성공 판정 기준)

DEFAULT_RUNS = ["A clip=v1_2/A_DLS", "A clamp=v1_2_Aclamp/A_DLS", "B=v1_2/B_mink",
                "C 격자=v1_2/C_eaik", "C SEW=v1_2_Csew/C_eaik"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="실행 폴더들이 있는 곳 (예: ~/out/hyojeong/retarget)")
    ap.add_argument("--ref", default="v1_2", help="분모(목표·할 수 있는 목표)를 정하는 실행 폴더 (C_eaik 사용)")
    ap.add_argument("--run", action="append", help='"이름=폴더/알고리즘" (여러 번). 없으면 v1.2 기본 비교 5개')
    ap.add_argument("--sat-deg", type=float, default=np.degrees(1e-3), help="관절 한계에 붙음으로 보는 거리 (도)")
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    common.add_success_args(ap)
    a = ap.parse_args()
    pos_tol, rot_tol = common.success_tols(a)
    print(f"판정 기준: {common.success_text(a)}")
    root = Path(a.root).expanduser()
    lim = list(csv.DictReader(open(Path(a.limits).expanduser(), encoding="utf-8")))
    lo, hi = np.array([float(r["lower_rad"]) for r in lim]), np.array([float(r["upper_rad"]) for r in lim])
    sat = np.radians(a.sat_deg)
    ids = sorted(p.stem for p in (root / a.ref / "C_eaik").glob("*.npz"))

    print("이름        | 전체 성공 | 할 수 있는 목표 | 팔 회전각 차 중앙 / 95% | 한계에 붙은 프레임 | 30° 튐 | 시간 중앙 / 95% ms")
    for spec in a.run or DEFAULT_RUNS:
        name, path = spec.split("=", 1)
        n = feas_n = ok_n = okf_n = jump30 = sewn = hit = hit_ok = 0
        swv, t_ms, delta, sat_j, sat_any = [], [], [], np.zeros(7), 0
        for eid in ids:
            c = np.load(root / a.ref / "C_eaik" / f"{eid}.npz")
            has = np.isfinite(c["target_pos"]).all(1)
            feas = has & (c["status"] == 0)
            d = np.load(root / path / f"{eid}.npz")
            inlim = np.all((d["q"] >= lo - 1e-6) & (d["q"] <= hi + 1e-6), axis=1)
            ok = (d["tcp_error"] <= pos_tol) & (d["rot_error"] <= rot_tol) & inlim
            n += has.sum(); feas_n += feas.sum(); ok_n += (ok & has).sum(); okf_n += (ok & feas).sum()
            s = d["swivel_error"][has]
            swv += list(np.degrees(s[np.isfinite(s)]))
            q = d["q"][has]
            near = (q - lo < sat) | (hi - q < sat)
            sat_j += near.sum(0); sat_any += near.any(1).sum()
            jump30 += int((np.degrees(np.abs(np.diff(d["q"], axis=0))).max(1) > 30).sum())
            t = d["time_ms"][has][1:]
            t_ms += list(t[np.isfinite(t)])
            if "sew_delta" in d.files:
                sd = d["sew_delta"][has]
                sewn += np.isfinite(sd).sum(); delta += list(np.degrees(np.abs(sd[np.isfinite(sd)])))
            if "limit_iters" in d.files:
                h = (d["limit_iters"] > 0) & feas
                hit += h.sum(); hit_ok += (h & ok).sum()
        sw = np.array(swv)
        print(f"{name:11s} | {100 * ok_n / n:6.1f}% | {100 * okf_n / max(feas_n, 1):6.1f}% | "
              f"{np.median(sw):5.1f}° / {np.percentile(sw, 95):5.1f}° | {100 * sat_any / n:5.1f}% | {jump30:5d} | "
              f"{np.median(t_ms):6.2f} / {np.percentile(t_ms, 95):6.2f}")
        print("    관절별 한계에 붙음 %: " + " ".join(f"q{j + 1} {100 * sat_j[j] / n:.1f}" for j in range(7)))
        if hit:
            print(f"    할 수 있는 목표에서 한계에 걸린 프레임 {hit} ({100 * hit / feas_n:.1f}%) 중 성공 {hit_ok} ({100 * hit_ok / hit:.1f}%)")
        if sewn:
            dl = np.array(delta)
            print(f"    SEW 로 직접 푼 프레임 {sewn} / {n} ({100 * sewn / n:.1f}%) | 사람 각 그대로 {(dl < 1e-6).sum()} | "
                  f"벗어난 각 중앙 {np.median(dl):.0f}° 95% {np.percentile(dl, 95):.0f}°")
    print(f"\n분모: {a.ref}/C_eaik 기준 목표 {n} 프레임, 할 수 있는 목표 {feas_n} 프레임 (에피소드 {len(ids)}개)")


if __name__ == "__main__":
    main()
