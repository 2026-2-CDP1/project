"""촬영 계획표 (1단계): CSV 양식 생성 + 케이스 편중 검사.

  python plan.py template plan.csv     # 양식
  python plan.py check plan.csv        # 태스크별 케이스 분포·용도별 수량
계획표 열: id_start,id_end,task,layout,speed,case_tags,purpose  (예: T1_E001,T1_E015,컵 잡기,보통 거리,보통,정밀 구간,학습)
"""
import csv, sys
from collections import Counter
from constants import TASKS, CASES, PURPOSES

HEAD = ["id_start", "id_end", "task", "layout", "speed", "case_tags", "purpose"]


def n_eps(a, b):
    return int(b.split("_E")[1]) - int(a.split("_E")[1]) + 1


def check(path):
    cnt, purp, unknown = Counter(), Counter(), []
    for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8")), start=2):
        n = n_eps(r["id_start"], r["id_end"])
        if r["task"] not in TASKS:
            unknown.append(f"{i}행 태스크 '{r['task']}' (허용: {TASKS})")
        if r["purpose"] not in PURPOSES:
            unknown.append(f"{i}행 용도 '{r['purpose']}' (허용: {PURPOSES})")
        purp[(r["task"], r["purpose"])] += n
        for c in [s.strip() for s in r["case_tags"].replace("|", ",").split(",") if s.strip()]:
            if c not in CASES:
                unknown.append(f"{i}행 케이스 태그 '{c}' (허용: {CASES})")
                continue
            cnt[(r["task"], c)] += n
    for u in unknown:
        print("[알 수 없는 값]", u)
    for task in sorted({t for t, _ in cnt} | {t for t, _ in purp}):
        print(f"[{task}] " + ", ".join(f"{p} {purp[(task, p)]}" for p in PURPOSES))
        for c in CASES:
            n = cnt[(task, c)]
            print(f"   {c:<10}{n:>4}" + ("   <-- 비어 있음" if n == 0 else ""))


if __name__ == "__main__":
    if sys.argv[1] == "template":
        with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(HEAD)
            w.writerow(["T1_E001", "T1_E015", "컵 잡기", "보통 거리", "보통", "정밀 구간", "학습"])
            w.writerow(["T1_E016", "T1_E030", "컵 잡기", "먼 곳", "보통", "큰 이동,가장자리", "학습"])
    else:
        check(sys.argv[2])
