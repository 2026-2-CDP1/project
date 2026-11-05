"""5-2 집계: metrics_long.csv → 케이스 × 알고리즘 표 (summary.md), 이상값 목록 (worst_episodes.md).

정의는 metrics/definitions.md(v1.0 고정) 4·5절. numpy만 쓴다 (mujoco·pandas 불필요).

    python metrics/summary.py --in ~/out/신서연/metrics --out ~/out/신서연/summary

입력 (metrics.py run의 출력):
    metrics_long.csv   algo, episode_id, task, case_tags, metric, value
    episode_info.csv   algo, episode_id, status, … (없어도 됨)
출력:
    summary.md         축 순서(정확도 → 속도 → 궤적 품질 → 안전 확인 → 참고)의 지표별 표 + 논문 결론용 요약
    summary.csv        metric, case, algo, n, n_nan, median, worst (기계용)
    worst_episodes.md  순위 지표 칸마다 가장 나쁜 에피소드 3개 (나란히 보기 영상으로 확인할 목록)

공정성: 모든 알고리즘에서 계산된(ok) **공통 에피소드만** 집계한다 (정의서 원칙 6). 빠진 에피소드는 목록에 적는다.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import METRICS  # noqa: E402

ALL = "전체"
MIN_N = 5  # 이보다 적은 칸은 표본 적음 표시 (정의서 4절, 멘토 Q5 답 없음 → 초안)

# 지표 표시 정보: 축, 묶음, 이름, 표시 변환. 정의서 7절과 같아야 한다 (어긋나면 import 때 오류)
pct = ("%", lambda v: 100 * v, "{:.1f}")
META = {
    "retarget_fail_rate": ("정확도", "순위", "리타게팅 성공률", ("%", lambda v: 100 * (1 - v), "{:.1f}"), "inverse"),
    "arm_angle_diff_rad": ("정확도", "참고", "팔 회전각 차이 (사람 대비)", ("°", np.degrees, "{:.1f}"), None),
    "compute_ms_median": ("속도", "순위", "계산 시간 중앙값", (" ms", lambda v: v, "{:.2f}"), None),
    "compute_ms_p95": ("속도", "순위", "계산 시간 95%", (" ms", lambda v: v, "{:.2f}"), None),
    "over_budget_rate": ("속도", "보조", "실시간 초과율 (33.3 ms)", pct, None),
    "jerk_rms": ("궤적 품질", "순위", "저크 RMS", (" rad/s³", lambda v: v, "{:.0f}"), None),
    "velocity_limit_exceed_rate": ("궤적 품질", "순위", "관절 속도 한계 초과율", pct, None),
    "self_collision_rate": ("궤적 품질", "순위", "자기 충돌률", pct, None),
    "env_collision_rate": ("궤적 품질", "순위", "환경 충돌률 (장애물 + 팔꿈치–테이블)", pct, None),
    "limit_violation_rate": ("안전 확인", "안전", "관절 한계 위반율 (0이어야 함)", pct, None),
    "env_obstacle_rate": ("궤적 품질", "세부", "환경 충돌 ① 장애물", pct, None),
    "env_elbow_below_table_rate": ("궤적 품질", "세부", "환경 충돌 ② 팔꿈치–테이블", pct, None),
    "track_pos_err_mean_m": ("정확도", "참고", "위치 추종 오차 평균", (" mm", lambda v: 1000 * v, "{:.1f}"), None),
    "track_pos_err_max_m": ("정확도", "참고", "위치 추종 오차 최대", (" mm", lambda v: 1000 * v, "{:.1f}"), None),
    "track_rot_err_mean_rad": ("정확도", "참고", "방향 추종 오차 평균", ("°", np.degrees, "{:.2f}"), None),
    "track_rot_err_max_rad": ("정확도", "참고", "방향 추종 오차 최대", ("°", np.degrees, "{:.2f}"), None),
    "limit_proximity_rate": ("궤적 품질", "참고", "관절 한계 근접률 (1°)", pct, None),
    "grasp_dist_m": ("정확도", "참고", "파지점 거리", (" cm", lambda v: 100 * v, "{:.1f}"), None),
    "min_manipulability": ("궤적 품질", "참고", "최소 조작성 (클수록 특이 자세에서 멂)", ("", lambda v: v, "{:.3f}"), "higher"),
}
missing = set(METRICS) ^ set(META)
if missing:
    raise ImportError(f"summary.py META와 metrics.py METRICS가 다름: {sorted(missing)}")
AXES = ["정확도", "속도", "궤적 품질", "안전 확인"]
GROUP_ORDER = {"순위": 0, "보조": 1, "세부": 2, "안전": 3, "참고": 4}
SPEED_ALL_ONLY = True  # 속도 축은 케이스별 차이가 작아 "전체" 한 줄만 (정의서 4절)


# ---------------------------------------------------------------- 읽기

def read_long(path):
    """→ values[(metric, algo, ep)] = float, tags[ep] = [태그], task[ep]."""
    values, tags, task = {}, {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r["value"]
            values[(r["metric"], r["algo"], r["episode_id"])] = float(v) if v not in ("", "nan", "NaN") else float("nan")
            tags[r["episode_id"]] = [t for t in r["case_tags"].split("|") if t]
            task[r["episode_id"]] = r["task"]
    return values, tags, task


def read_info(path):
    """→ {(algo, ep): row}. 파일이 없으면 빈 dict."""
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["algo"], r["episode_id"]): r for r in csv.DictReader(f)}


def common_episodes(values, info):
    """모든 알고리즘에서 계산된 에피소드 (공통 집합)와 빠진 에피소드 {ep: 이유}."""
    algos = sorted({a for _, a, _ in values})
    eps = sorted({e for _, _, e in values} | {e for _, e in info})
    ok = {a: {e for (m, aa, e) in values if aa == a} for a in algos}
    dropped = {}
    for e in eps:
        why = []
        for a in algos:
            row = info.get((a, e))
            if row is not None and row.get("status") != "ok":
                why.append(f"{a}: {row.get('status')}")
            elif e not in ok[a]:
                why.append(f"{a}: 결과 없음")
        if why:
            dropped[e] = "; ".join(why)
    return algos, [e for e in eps if e not in dropped], dropped


# ---------------------------------------------------------------- 집계

def cases_of(eps, tags):
    """전체 + 태그 (에피소드 수 많은 순)."""
    cnt = Counter(t for e in eps for t in tags.get(e, []))
    return [ALL] + [t for t, _ in cnt.most_common()]


def cell(values, metric, algo, eps):
    """칸 하나: (n, n_nan, 중앙값, 최악값, 최악 에피소드 [(값, ep)] 나쁜 순)."""
    higher_better = META[metric][4] == "higher"
    pairs = [(values.get((metric, algo, e), float("nan")), e) for e in eps]
    fin = [(v, e) for v, e in pairs if np.isfinite(v)]
    n, n_nan = len(fin), len(pairs) - len(fin)
    if not fin:
        return n, n_nan, float("nan"), float("nan"), []
    vals = np.array([v for v, _ in fin])
    worst_sorted = sorted(fin, key=lambda p: p[0], reverse=not higher_better)   # 나쁜 순
    return n, n_nan, float(np.median(vals)), float(worst_sorted[0][0]), worst_sorted


def aggregate(values, tags, eps, algos):
    """→ {(metric, case, algo): (n, n_nan, median, worst, worst_sorted)}."""
    out = {}
    for metric in METRICS:
        for case in cases_of(eps, tags):
            sel = eps if case == ALL else [e for e in eps if case in tags.get(e, [])]
            for algo in algos:
                out[(metric, case, algo)] = cell(values, metric, algo, sel)
    return out


# ---------------------------------------------------------------- 표시

def fmt(metric, v):
    if not np.isfinite(v):
        return "—"
    unit, conv, f = META[metric][3]
    return f.format(conv(v)) + unit


def best_algo(metric, case, algos, agg):
    """중앙값이 가장 좋은 알고리즘 (같으면 여럿). 낮을수록 좋음, 최소 조작성만 높을수록."""
    med = {a: agg[(metric, case, a)][2] for a in algos if np.isfinite(agg[(metric, case, a)][2])}
    if len(med) < 2:
        return set()
    target = max(med.values()) if META[metric][4] == "higher" else min(med.values())
    best = {a for a, v in med.items() if np.isclose(v, target)}
    return set() if len(best) == len(med) else best


def metric_table(metric, cases, algos, agg, min_n):
    lines = ["| 케이스 | " + " | ".join(algos) + " |", "|---|" + "---|" * len(algos)]
    for case in cases:
        best = best_algo(metric, case, algos, agg)
        row = []
        for a in algos:
            n, n_nan, med, worst, _ = agg[(metric, case, a)]
            txt = f"{fmt(metric, med)} ({fmt(metric, worst)}) n={n}"
            if n_nan:
                txt += f", NaN {n_nan}"
            if n < min_n:
                txt += " ⚠"
            row.append(f"**{txt}**" if a in best else txt)
        lines.append(f"| {case} | " + " | ".join(row) + " |")
    return lines


def paper_summary(algos, agg):
    """알고리즘마다 한 줄: 축별 대표 지표의 전체 중앙값 (논문 결론용, 정의서 4절)."""
    cols = ["retarget_fail_rate", "compute_ms_median", "compute_ms_p95", "jerk_rms",
            "velocity_limit_exceed_rate", "self_collision_rate", "env_collision_rate", "arm_angle_diff_rad"]
    head = ["알고리즘"] + [META[c][2] + ("" if META[c][1] != "참고" else " (참고)") for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for a in algos:
        lines.append(f"| {a} | " + " | ".join(fmt(c, agg[(c, ALL, a)][2]) for c in cols) + " |")
    return lines


def write_summary(out_dir, algos, eps, dropped, tags, task, agg, info, min_n):
    cases = cases_of(eps, tags)
    L = ["# 지표 집계 (summary)", "",
         f"> 정의: `metrics/definitions.md` v1.0 (고정). 알고리즘 {len(algos)}개, 공통 에피소드 {len(eps)}개 "
         f"(태스크 {dict(sorted(Counter(task[e] for e in eps).items()))})",
         f"> 칸 = 중앙값 (최악값) n=에피소드 수. **굵게** = 그 행에서 중앙값이 가장 좋은 알고리즘. ⚠ = n < {min_n} (표본 적음)",
         "> 모든 지표는 낮을수록 좋음. 단 리타게팅 성공률은 높을수록, 최소 조작성은 클수록 좋음 (표시만 바꿈)", "",
         "## 논문 결론용 요약 (전체 에피소드 중앙값)", ""]
    L += paper_summary(algos, agg)
    for axis in AXES:
        ms = sorted((m for m in METRICS if META[m][0] == axis), key=lambda m: (GROUP_ORDER[META[m][1]], METRICS.index(m)))
        L += ["", f"## {axis}"]
        for m in ms:
            group = META[m][1]
            if group == "안전":
                bad = [(c, a, agg[(m, c, a)][3]) for c in cases for a in algos
                       if np.isfinite(agg[(m, c, a)][3]) and agg[(m, c, a)][3] > 0]
                L += ["", f"### {META[m][2]} — 안전 확인", ""]
                L += ["모든 칸 0 (정상)."] if not bad else \
                    ["**0이 아닌 칸이 있음 → 계산·구현 오류부터 확인**", ""] + [f"- {c} · {a}: 최악 {fmt(m, v)}" for c, a, v in bad]
                continue
            L += ["", f"### {META[m][2]} (`{m}`) — {group}", ""]
            L += metric_table(m, [ALL] if (axis == "속도" and SPEED_ALL_ONLY) else cases, algos, agg, min_n)
    L += ["", "## 집계에서 뺀 에피소드", ""]
    L += [f"- {e}: {why}" for e, why in sorted(dropped.items())] or ["없음"]
    flags = defaultdict(list)
    for (a, e), r in info.items():
        if e not in eps:
            continue
        if r.get("dt_flag") == "True":
            flags["30 Hz로 맞춘 Δt가 1/30 s와 0.1% 넘게 다름"].append(f"{a} {e}")
        if r.get("grasp_note"):
            flags[f"파지점 거리 없음: {r['grasp_note']}"].append(f"{a} {e}")
    if flags:
        L += ["", "## 표시할 에피소드", ""]
        for k, v in flags.items():
            L.append(f"- {k}: {len(v)}건 ({', '.join(sorted(v)[:10])}{' …' if len(v) > 10 else ''})")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_csv(out_dir, agg):
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "case", "algo", "n", "n_nan", "median", "worst"])
        for (m, c, a), (n, n_nan, med, worst, _) in agg.items():
            w.writerow([m, c, a, n, n_nan, med, worst])


def write_worst(out_dir, algos, eps, tags, agg, k=3):
    L = ["# 이상값 확인 목록 (worst_episodes)", "",
         "> 정의서 5절: 순위 지표 칸마다 가장 나쁜 에피소드 3개. 5-1 나란히 보기 영상으로 보고 "
         "알고리즘의 실제 거동인지, 계산·검출 오류인지 판단한다. 오류면 정의서 9절에 적고 모든 알고리즘에 다시 적용."]
    for m in [m for m in METRICS if META[m][1] == "순위"]:
        L += ["", f"## {META[m][2]} (`{m}`)", "", "| 케이스 | " + " | ".join(algos) + " |", "|---|" + "---|" * len(algos)]
        for case in ([ALL] if META[m][0] == "속도" else cases_of(eps, tags)):
            row = []
            for a in algos:
                ws = agg[(m, case, a)][4][:k]
                row.append("<br>".join(f"{e} {fmt(m, v)}" for v, e in ws) or "—")
            L.append(f"| {case} | " + " | ".join(row) + " |")
    (out_dir / "worst_episodes.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run(in_dir, out_dir, min_n=MIN_N):
    in_dir, out_dir = Path(in_dir).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    values, tags, task = read_long(in_dir / "metrics_long.csv")
    info = read_info(in_dir / "episode_info.csv")
    algos, eps, dropped = common_episodes(values, info)
    if not eps:
        raise SystemExit("모든 알고리즘에서 계산된 공통 에피소드가 없음")
    agg = aggregate(values, tags, eps, algos)
    write_summary(out_dir, algos, eps, dropped, tags, task, agg, info, min_n)
    write_csv(out_dir, agg)
    write_worst(out_dir, algos, eps, tags, agg)
    print(f"알고리즘 {algos}, 공통 에피소드 {len(eps)}개, 뺀 에피소드 {len(dropped)}개 → {out_dir}")
    return algos, eps, dropped, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-2 집계 (metrics/definitions.md v1.0 4·5절)")
    ap.add_argument("--in", dest="inp", required=True, help="metrics.py run 출력 폴더 (metrics_long.csv, episode_info.csv)")
    ap.add_argument("--out", required=True, help="출력 폴더 (~/project 밖, 예: ~/out/신서연/summary)")
    ap.add_argument("--min-n", type=int, default=MIN_N, help=f"표본 적음 표시 기준 (기본 {MIN_N})")
    a = ap.parse_args()
    run(a.inp, a.out, a.min_n)
