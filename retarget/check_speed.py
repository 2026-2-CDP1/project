"""로봇 관절 속도 한계 확인: 리타게팅한 관절 궤적을 영상 속도 그대로 재생하면 iiwa 속도 한계를 넘는가 — 3번 요청 (T2 빠른 조건, 11/6).

관절 속도 = |q[t+1] − q[t]| × exo_fps (rad/s). 한계 = robot.urdf <limit velocity> (joint1·2 85°/s ... joint7 135°/s).
참고용 빠른 확인. 공식 값은 5-2 metrics.py (속도 관련 지표).
사람 손목 속도(목표 TCP 속도, m/s)도 같이 보여 줌 (3번이 본 1.2~1.7 m/s 와 비교).

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_speed.py --out ~/out/hyojeong/retarget/v4
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGOS = ("A_DLS", "B_mink", "C_eaik")
T2_FAST = {f"T2_E{n:03d}" for n in list(range(26, 36)) + [68, 69]}


def urdf_velocity(path):
    txt = Path(path).expanduser().read_text(encoding="utf-8")
    v = []
    for j in range(1, 8):
        m = re.search(rf'<joint name="iiwa_joint_{j}".*?<limit[^>]*velocity="([0-9.eE+-]+)"', txt, re.S)
        v.append(float(m.group(1)))
    return np.array(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--urdf", default="~/project/robot/robot.urdf")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    vmax = urdf_velocity(a.urdf)
    print("속도 한계 (°/s): " + " ".join(f"q{j + 1} {np.degrees(v):.0f}" for j, v in enumerate(vmax)))

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # 그룹 -> 알고리즘 -> [넘는 프레임, 전체 프레임, 넘는 에피소드]
    worst = defaultdict(list)
    hum = {}
    for pc in sorted((out / "C_eaik").glob("*.npz")):
        eid = pc.stem
        c = np.load(pc)
        fps = float(json.loads(str(c["info"])).get("exo_fps", 60.0))
        p = c["target_pos"]
        hv = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
        hv = hv[np.isfinite(hv)]
        hum[eid] = np.percentile(hv, 95) if len(hv) else np.nan
        groups = [eid[:2], "전체"] + (["T2 빠름"] if eid in T2_FAST else [])
        for al in ALGOS:
            f = out / al / f"{eid}.npz"
            if not f.exists():
                continue
            q = np.load(f)["q"]
            ratio = np.abs(np.diff(q, axis=0)) * fps / vmax          # 1 넘으면 한계 초과
            over = (ratio > 1).any(1)
            for g in groups:
                s = agg[g][al]
                s[0] += int(over.sum())
                s[1] += len(over)
                s[2] += int(over.any())
            worst[al].append((float(ratio.max()), int(over.sum()), len(over), eid, int(ratio.max(0).argmax()) + 1))

    print("\n== 속도 한계를 넘는 프레임 비율 (어느 한 관절이라도) | 넘는 프레임이 있는 에피소드 수")
    for g in ("T1", "T2", "T2 빠름", "T3", "전체"):
        if g in agg:
            print(f"  {g:7s} " + " | ".join(f"{al[0]} {s[0] / max(s[1], 1):6.1%} ({s[2]}개)" for al, s in agg[g].items()))
    print(f"\n== 한계 대비 최대 속도가 큰 에피소드 (알고리즘별 {a.top}개): 최대 배수, 넘는 프레임/전체, 관절, 사람 손목 속도 p95")
    for al in ALGOS:
        print(f"  {al}")
        for r, n, T, eid, j in sorted(worst[al], reverse=True)[:a.top]:
            print(f"    {eid}  ×{r:5.1f}  {n:3d}/{T:3d}  q{j}  손목 {hum[eid]:.2f} m/s" + ("  (T2 빠름)" if eid in T2_FAST else ""))
    fast = [hum[e] for e in sorted(hum) if e in T2_FAST]
    if fast:
        print(f"\n참고: T2 빠름 {len(fast)}개 목표 TCP 속도 p95 중앙 {np.nanmedian(fast):.2f} m/s (로봇 좌표, 축척 반영)")


if __name__ == "__main__":
    main()
