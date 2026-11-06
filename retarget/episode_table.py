"""에피소드별 A·B·C 성공률 비교표 (check_success.py 가 만든 success_by_episode.csv 를 옆으로 펼침).

판정 (에피소드 성공률 기준, 5-2 프레임 판정: 1 cm · 5° · 한계 안):
  O  성공률 ≥ 90%      △  50% 이상 90% 미만      X  50% 미만

사용 (서버, source ~/use_robot.sh 후, check_success.py 를 먼저 실행):
  python retarget/episode_table.py --out ~/out/hyojeong/retarget/v1
출력: 화면(패턴별 개수 + 표) + <out>/success_table.csv (엑셀용)
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

ALGOS = ("A_DLS", "B_mink", "C_eaik")


def mark(s):
    if s == "" or s is None:
        return "-"
    s = float(s)
    return "O" if s >= 0.9 else ("△" if s >= 0.5 else "X")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true", help="표는 파일로만, 화면에는 요약만")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    src = out / "success_by_episode.csv"
    eps = defaultdict(dict)
    for r in csv.DictReader(open(src, encoding="utf-8")):
        e = eps[r["episode_id"]]
        e["layout"], e["hand"], e["target_frames"] = r["layout"], r["hand"], r["target_frames"]
        e[r["algo"]] = r["success_rate"]

    table = []
    for eid in sorted(eps):
        e = eps[eid]
        marks = "".join(mark(e.get(al)) for al in ALGOS)
        table.append({"episode_id": eid, "task": eid[:2], "layout": e["layout"], "hand": e["hand"],
                      "target_frames": e["target_frames"],
                      **{f"{al}_success": (f"{float(e[al]):.3f}" if e.get(al) not in (None, "") else "") for al in ALGOS},
                      **{f"{al}_mark": mark(e.get(al)) for al in ALGOS}, "pattern_ABC": marks})
    with open(out / "success_table.csv", "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀 한글
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    print("== 판정 패턴 (A·B·C 순서, O ≥90% / △ 50~90% / X <50%) — 에피소드 수")
    for pat, n in Counter(r["pattern_ABC"] for r in table).most_common():
        print(f"  {pat}  {n:3d}개")
    print("\n== 알고리즘별 판정 개수")
    for al in ALGOS:
        c = Counter(r[f"{al}_mark"] for r in table)
        print(f"  {al}: O {c['O']} / △ {c['△']} / X {c['X']}" + (f" / 결과 없음 {c['-']}" if c['-'] else ""))
    if not a.quiet:
        print("\n== 에피소드별 (성공률 %)")
        print("  에피소드  손    목표  |   A     B     C  | 판정 | 배치")
        for r in table:
            pct = lambda al: f"{100 * float(r[f'{al}_success']):5.1f}" if r[f"{al}_success"] else "   - "
            print(f"  {r['episode_id']} {r['hand'][:1] or '-'} {r['target_frames']:>5} | {pct('A_DLS')} {pct('B_mink')} {pct('C_eaik')} | "
                  f"{r['pattern_ABC']}  | {r['layout'][:40]}")
    print(f"\n저장: {out / 'success_table.csv'}")


if __name__ == "__main__":
    main()
