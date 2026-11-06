"""run_all.py 결과 점검: 이상한 에피소드를 골라낸다 (5-2·3번에 넘기기 전).

사용 (서버, source ~/use_robot.sh 후):
  python retarget/check_results.py --out ~/out/hyojeong/retarget/v1 --episodes ~/out/seohyeon/episodes_v1

보는 것
  1. 알고리즘별 상태(ok/error/excluded) 개수, 오류 목록
  2. 알고리즘별 요약: 목표 있는 프레임 비율, 손목 오차, 팔 회전각, 시간 (에피소드 중앙 / 최악)
  3. 골라낼 에피소드
     - 목표 있는 프레임 비율 < 0.95 (손목 빈칸)
     - C도 1 cm 넘게 빗나간 프레임이 있음 (공식 해로도 못 푼 목표)
     - A·B가 1 cm 넘게 빗나간 프레임 비율 > 30% (해 갈래 갇힘)
     - IK 시간이 알고리즘 중앙값의 3배 넘음
     - 사람 팔 길이가 전체 중앙값에서 5 cm 넘게 벗어남 (키포인트 확인)
     - 직접 찍은 손목 / 방향 유지 프레임이 있는 에피소드
  4. --zneg 에피소드(시작 손목 z 음수, 3번 보고)의 첫 프레임: 목표 높이(베이스 기준)와 A·B·C 오차
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGOS = ("A_DLS", "B_mink", "C_eaik")
ZNEG = ["T2_E003", "T2_E036", "T2_E050", "T2_E058", "T2_E059", "T2_E060", "T2_E063",
        "T2_E064", "T2_E066", "T2_E068", "T2_E070", "T2_E071"]


def f(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError):
        return np.nan
    return v


def load_rows(out):
    """summary.csv + summary_<알고리즘>.csv 를 합치고, 같은 에피소드×알고리즘은 마지막 줄만."""
    last = {}
    for p in sorted(out.glob("summary*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            last[(r["episode_id"], r["algo"])] = r
    return list(last.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run_all.py --out 폴더")
    ap.add_argument("--episodes", help="3번 episodes 폴더 (--zneg 확인용)")
    ap.add_argument("--zneg", nargs="*", default=ZNEG)
    ap.add_argument("--zneg-frames", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    rows = load_rows(out)
    ok = [r for r in rows if r["state"] == "ok"]

    print("== 1. 상태")
    st = defaultdict(lambda: defaultdict(int))
    for r in rows:
        st[r["algo"]][r["state"]] += 1
    for al in ALGOS:
        print(f"  {al}: " + (", ".join(f"{k} {v}" for k, v in st[al].items()) or "결과 없음"))
    for r in rows:
        if r["state"] == "error":
            print(f"  오류 {r['episode_id']} {r['algo']}: {r['note'][:150]}")

    print("\n== 2. 알고리즘별 (에피소드 값의 중앙 / 최악)")
    print("  algo    | 목표 프레임 비율 | 손목 95% mm | 1cm 초과 비율 | 팔 회전각 중앙 deg | IK 시간 중앙 ms | 에피소드 수")
    med_time = {}
    for al in ALGOS:
        R = [r for r in ok if r["algo"] == al]
        if not R:
            continue
        tgt = np.array([f(r, "target_frames") / f(r, "frames") for r in R])
        pe = np.array([f(r, "pos_err_p95_mm") for r in R])
        over = np.array([f(r, "n_pos_over_1cm") / max(f(r, "target_frames"), 1) for r in R])
        sw = np.array([f(r, "swivel_med_deg") for r in R])
        tm = np.array([f(r, "time_med_ms") for r in R])
        med_time[al] = np.nanmedian(tm)
        print(f"  {al:7s} | {np.nanmedian(tgt):.2f} / {np.nanmin(tgt):.2f} | {np.nanmedian(pe):.2f} / {np.nanmax(pe):.0f} | "
              f"{np.nanmedian(over):.2f} / {np.nanmax(over):.2f} | {np.nanmedian(sw):.0f} / {np.nanmax(sw):.0f} | "
              f"{np.nanmedian(tm):.1f} / {np.nanmax(tm):.1f} | {len(R)}")

    print("\n== 3. 골라낼 에피소드")
    flags = defaultdict(list)
    for r in ok:
        eid, al = r["episode_id"], r["algo"]
        tgt = f(r, "target_frames") / max(f(r, "frames"), 1)
        over = f(r, "n_pos_over_1cm") / max(f(r, "target_frames"), 1)
        if al == "A_DLS" and tgt < 0.95:
            flags["목표 있는 프레임 < 95% (손목 빈칸)"].append(f"{eid} {tgt:.0%}")
        if al == "C_eaik" and f(r, "n_pos_over_1cm") > 0:
            flags["C도 1 cm 넘게 빗나감 (공식 해로도 못 푼 목표)"].append(f"{eid} {int(f(r, 'n_pos_over_1cm'))}프레임 최대 {f(r, 'pos_err_max_mm'):.0f} mm")
        if al in ("A_DLS", "B_mink") and over > 0.3:
            flags[f"{al} 1 cm 초과 > 30% (해 갈래 갇힘)"].append(f"{eid} {over:.0%}")
        if al in med_time and f(r, "time_med_ms") > 3 * med_time[al]:
            flags[f"{al} 시간 > 중앙 3배"].append(f"{eid} {f(r, 'time_med_ms'):.1f} ms")
        if al == "A_DLS" and (f(r, "n_wrist_manual") > 0 or f(r, "n_orient_held") > 0):
            flags["직접 찍은 손목 / 방향 유지"].append(f"{eid} 찍음 {int(f(r, 'n_wrist_manual'))} 방향유지 {int(f(r, 'n_orient_held'))}")
    # 사람 팔 길이 (npz info)
    arms = {}
    for p in sorted((out / "A_DLS").glob("*.npz")):
        info = json.loads(str(np.load(p)["info"]))
        arms[p.stem] = info.get("human_arm_m", np.nan)
    if arms:
        m = np.nanmedian(list(arms.values()))
        for eid, v in arms.items():
            if abs(v - m) > 0.05:
                flags[f"사람 팔 길이가 중앙({m:.3f} m)에서 5 cm 넘게 벗어남"].append(f"{eid} {v:.3f} m")
    for k, v in flags.items():
        print(f"  [{k}] {len(v)}개")
        for x in v[:15]:
            print(f"     {x}")
        if len(v) > 15:
            print(f"     … 외 {len(v) - 15}개")

    if a.zneg:
        print(f"\n== 4. 시작 손목 z 음수 에피소드 (첫 {a.zneg_frames}프레임): 목표 높이 = 로봇 베이스 기준 m, 오차 mm")
        for eid in a.zneg:
            line = []
            for al in ALGOS:
                p = out / al / f"{eid}.npz"
                if not p.exists():
                    line.append(f"{al} 없음")
                    continue
                d = np.load(p)
                n = a.zneg_frames
                if al == "A_DLS":
                    z = d["target_pos"][:n, 2]
                    line.insert(0, "목표 z " + " ".join(f"{v:+.2f}" for v in z))
                e = 1000 * d["tcp_error"][:n]
                line.append(f"{al[0]} 오차 최대 {np.nanmax(e):.0f}")
            print(f"  {eid}: " + " | ".join(line))


if __name__ == "__main__":
    main()
