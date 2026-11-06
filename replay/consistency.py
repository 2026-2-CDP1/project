"""5-1 정합 확인 (역할 가이드 5-1 4단계): 4번이 pinocchio로 계산한 손목 오차와 우리 MuJoCo 재생으로 계산한 오차가 같은지.

4번 npz의 tcp_error(위치, m)·rot_error(방향, rad)는 "최종 관절 각도의 FK 손끝 ↔ 목표" 오차다.
같은 관절 각도를 replay()로 재생해 같은 목표와의 오차를 다시 구하고 프레임마다 비교한다.
두 FK(4번 pinocchio + robot.urdf, 5-1 MuJoCo + robot.xml)가 같으면 차이가 수치 오차 수준(1e-6)이어야 한다.
다르면 관절 순서, 모델 파일, TCP 위치(플랜지)부터 의심한다.

    python replay/consistency.py --retarget <4번 결과 폴더> --out ~/out/신서연/consistency [--limit 5]

출력: <out>/consistency.md (알고리즘별 요약), <out>/consistency.csv (파일마다).
한계: 오차의 '크기'를 비교하므로 위치 자체가 같은 거리만큼 다른 방향으로 어긋나는 경우는 못 잡는다. 프레임이 많아서
실제로 그런 일이 계속 맞아떨어질 가능성은 매우 낮다. 4번이 FK 손끝 위치를 npz에 같이 저장하면 직접 비교로 바꾼다.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _is_here(p, here=Path(__file__).resolve().parent):
    """같은 폴더인지 (한글 경로가 NFC/NFD로 다르게 들어와도 같게 보도록 문자열 대신 samefile로)."""
    try:
        return os.path.samefile(p or ".", here)
    except OSError:
        return False


sys.path[:] = [p for p in sys.path if not _is_here(p)]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "metrics"))

TOL_POS_M = 1e-6
TOL_ROT_RAD = 1e-6
POS_KEYS, ROT_KEYS = ("tcp_error", "pos_err"), ("rot_error", "rot_err")


def check_file(npz, replayer):
    """npz 하나 → dict(프레임 수, 비교한 프레임 수, 위치·방향 차이 최대, 통과 여부, 메모)."""
    import metrics as M
    res = dict(np.load(npz))
    q = np.asarray(res["q"], float)
    pk = next((k for k in POS_KEYS if k in res), None)
    rk = next((k for k in ROT_KEYS if k in res), None)
    row = {"file": f"{npz.parent.name}/{npz.name}", "frames": len(q)}
    if "target_pos" not in res or "target_quat" not in res or pk is None:
        return {**row, "compared": 0, "pass": False, "note": "target_pos·target_quat·tcp_error 중 없는 키"}
    out = replayer.replay(q)
    pe, re = M.tracking_errors(out["tcp_pos"], out["tcp_quat"], res["target_pos"], res["target_quat"])
    theirs_p = np.asarray(res[pk], float)
    theirs_r = np.asarray(res[rk], float) if rk else np.full(len(q), np.nan)
    ok = np.isfinite(pe) & np.isfinite(theirs_p)
    if not ok.any():
        return {**row, "compared": 0, "pass": False, "note": "비교할 프레임 없음 (목표가 모두 NaN)"}
    dp = np.abs(pe[ok] - theirs_p[ok])
    okr = ok & np.isfinite(re) & np.isfinite(theirs_r)
    dr = np.abs(re[okr] - theirs_r[okr]) if okr.any() else np.array([np.nan])
    worst = int(np.flatnonzero(ok)[np.argmax(dp)])
    passed = bool(dp.max() <= TOL_POS_M and (np.isnan(dr).all() or np.nanmax(dr) <= TOL_ROT_RAD))
    return {**row, "compared": int(ok.sum()), "max_dpos_m": float(dp.max()), "max_drot_rad": float(np.nanmax(dr)),
            "worst_frame": worst, "pass": passed, "note": ""}


def main():
    ap = argparse.ArgumentParser(description="5-1 정합 확인: 4번 FK 오차 ↔ 5-1 재생 오차")
    ap.add_argument("--retarget", required=True, help="4번 결과 폴더 (<알고리즘>/<에피소드ID>.npz)")
    ap.add_argument("--out", required=True, help="출력 폴더 (~/project 밖)")
    ap.add_argument("--algos", nargs="*", help="알고리즘 폴더 (기본: 모두)")
    ap.add_argument("--limit", type=int, help="알고리즘마다 앞에서부터 이 개수만 (빠른 확인)")
    a = ap.parse_args()
    from replay.replay import Replayer

    root, out = Path(a.retarget).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    algos = a.algos or sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    r = Replayer()
    rows = []
    for algo in algos:
        files = sorted((root / algo).glob("*.npz"))[: a.limit]
        for f in files:
            try:
                rows.append({"algo": algo, **check_file(f, r)})
            except Exception as e:  # 실패를 지우지 않음
                rows.append({"algo": algo, "file": f"{algo}/{f.name}", "pass": False, "note": f"오류: {e}"})
    fields = ["algo", "file", "frames", "compared", "max_dpos_m", "max_drot_rad", "worst_frame", "pass", "note"]
    with open(out / "consistency.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    L = ["# 5-1 정합 확인 (consistency)", "",
         f"> 4번 pinocchio FK 손목 오차(`tcp_error`, `rot_error`) ↔ 5-1 MuJoCo 재생으로 다시 계산한 오차. "
         f"기준: 프레임마다 차이 ≤ {TOL_POS_M:g} m, ≤ {TOL_ROT_RAD:g} rad", "",
         "| 알고리즘 | 파일 | 통과 | 위치 차이 최대 | 방향 차이 최대 | 가장 큰 파일 |", "|---|---|---|---|---|---|"]
    for algo in algos:
        rs = [x for x in rows if x["algo"] == algo]
        good = [x for x in rs if "max_dpos_m" in x]
        mp = max((x["max_dpos_m"] for x in good), default=float("nan"))
        mr = max((x["max_drot_rad"] for x in good if np.isfinite(x["max_drot_rad"])), default=float("nan"))
        wf = max(good, key=lambda x: x["max_dpos_m"])["file"] if good else "—"
        L.append(f"| {algo} | {len(rs)} | {sum(x['pass'] for x in rs)} | {mp:.2e} m | {mr:.2e} rad | {wf} |")
    bad = [x for x in rows if not x["pass"]]
    L += ["", "## 통과 못 한 파일", ""] + ([f"- {x['file']}: {x.get('note') or ''} 위치 차이 {x.get('max_dpos_m', float('nan')):.2e} m "
                                          f"(프레임 {x.get('worst_frame', '—')})" for x in bad] or ["없음"])
    (out / "consistency.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(rows)}개 중 통과 {len(rows) - len(bad)} → {out / 'consistency.md'}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
