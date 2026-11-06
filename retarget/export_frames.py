"""에피소드별 프레임 판정을 작은 JSON 하나로 (결과 문서의 '실패 구간' 화면용).

프레임마다 한 글자:
  알고리즘 줄 (A·B·C): "." 성공 (0.5 cm · 1° · 한계 안, common.SUCCESS_*)  "x" 실패  "-" 목표 없음(손목 미검출)
  할 수 없는 목표 줄:   " " 할 수 있음  "f" 안 닿음(> 0.82 m)  "n" 너무 가까움(< 0.41 m)  "o" 방향·관절 한계  "-" 목표 없음
  (할 수 없는 목표 = C가 정확한 해를 못 찾은 프레임, check_feasible.py 와 같은 기준)
영상 위치: exo_start_frame(3번 meta.json) + 프레임 번호, exo_fps 로 초 환산.

사용 (서버, source ~/use_robot.sh 후):
  python retarget/export_frames.py --out ~/out/hyojeong/retarget/v1 --episodes ~/out/seohyeon/episodes_v1
출력: <out>/frames.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common  # noqa: E402  (같은 retarget 폴더, 성공 판정 기준)

ALGOS = ("A_DLS", "B_mink", "C_eaik")
POS_TOL, ROT_TOL, LIM_EPS = common.SUCCESS_POS_M, np.radians(common.SUCCESS_ROT_DEG), 1e-6   # main 에서 --pos-tol-cm 등으로 덮어씀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--records", default="~/project/episodes/records.csv")
    ap.add_argument("--limits", default="~/project/robot/joint_limits.csv")
    common.add_success_args(ap)
    a = ap.parse_args()
    global POS_TOL, ROT_TOL
    POS_TOL, ROT_TOL = common.success_tols(a)
    print(f"판정 기준: {common.success_text(a)}")
    out, epr = Path(a.out).expanduser(), Path(a.episodes).expanduser()
    lim = list(csv.DictReader(open(Path(a.limits).expanduser(), encoding="utf-8")))
    lo, hi = np.array([float(r["lower_rad"]) for r in lim]), np.array([float(r["upper_rad"]) for r in lim])
    try:
        rec = {r["episode_id"]: r for r in csv.DictReader(open(Path(a.records).expanduser(), encoding="utf-8"))}
    except FileNotFoundError:
        rec = {}

    eps = []
    for pc in sorted((out / "C_eaik").glob("*.npz")):
        eid = pc.stem
        c = np.load(pc)
        has = np.isfinite(c["target_pos"]).all(1)
        w, x, y, z = c["target_quat"].T
        zf = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
        r = np.linalg.norm(c["target_pos"] - 0.126 * zf - np.array([0, 0, 0.36]), axis=1)
        infeas = has & (c["status"] != 0)
        inf_s = "".join("-" if not h else (" " if not i else ("f" if rr > 0.82 else "n" if rr < 0.41 else "o"))
                        for h, i, rr in zip(has, infeas, r))
        rows = {}
        for al in ALGOS:
            p = out / al / f"{eid}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            inlim = np.all((d["q"] >= lo - LIM_EPS) & (d["q"] <= hi + LIM_EPS), axis=1)
            succ = has & (d["tcp_error"] <= POS_TOL) & (d["rot_error"] <= ROT_TOL) & inlim
            rows[al[0]] = "".join("-" if not h else ("." if s else "x") for h, s in zip(has, succ))
        meta_p = epr / eid / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        eps.append({"id": eid, "layout": rec.get(eid, {}).get("layout", ""), "hand": meta.get("hand", ""),
                    "start": meta.get("exo_start_frame"), "fps": meta.get("exo_fps"), "inf": inf_s, **rows})
    (out / "frames.json").write_text(json.dumps(eps, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"에피소드 {len(eps)}개 → {out / 'frames.json'} ({(out / 'frames.json').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
