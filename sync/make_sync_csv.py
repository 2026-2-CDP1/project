"""detect_claps.py 결과 -> sync.csv (3번 episodes/tools/io_utils.read_sync_row 형식).

열: episode_id, camera, offset_start_s, offset_end_s, drift_s, s0, s1, usable, note
  - 기준 카메라(--ref, 정면 exo) 대비 나머지 카메라마다 한 행. 기준 카메라 자신은 행 없음.
  - offset = 기준 카메라 박수 시각 - 이 카메라 박수 시각 (s). 시작·끝 박수 각각.
  - drift_s = offset_end_s - offset_start_s
  - s0, s1 = 이 카메라 자기 시각으로 잰 시작·끝 박수 시각 (드리프트 선형 보간용)
  - 시각은 모두 각 폰의 data.jsonl 프레임 시각. 박수 시각은 각 카메라 영상에서 그 에피소드마다 잰 값.

박수 고르기
  1) 카메라마다 박수 후보가 여러 개일 수 있다 (손을 빠르게 모은 다른 동작, 놓친 박수).
  2) 두 폰의 시계 차는 한 촬영 세션 안에서 거의 일정하다. 모든 에피소드의 후보 시각 차이를 모아
     카메라 쌍마다 가장 많이 모이는 값(세션 대표값)을 구하고, 이것으로 모든 후보를 기준 카메라 시간축에 모은다.
     대표값은 "어느 후보끼리 같은 박수인가" 를 고르는 데만 쓴다. 표의 시간 차로 쓰지 않는다.
  3) 에피소드마다 --tol 안에 모이는 후보 묶음 = 박수 하나. 카메라 --min-support 대 이상이 본 묶음 중
     가장 이른 것 = 시작 박수, 가장 늦은 것 = 끝 박수 (둘 사이 --min-gap 초 이상).
  4) 박수 묶음에 빠진 카메라는 --guide-out 파일로 내보낸다 -> detect_claps.py --guide 로 그 카메라 영상에서
     짐작 시각 ± 0.3 s 안을 다시 찾는다 (안내 검출) -> 이 스크립트를 다시 돌리면 반영된다.

usable = 0: 시작·끝 박수 중 하나라도 기준 카메라나 이 카메라에서 못 찾음, 폴더 없음, 검출 오류.
  드리프트 허용 기준은 멘토 확인 전이라 usable 에 반영하지 않고, |drift| > --drift-warn 이면 note 에 표시.
  안내 검출로 찾은 값은 note 에 "안내" 로 표시.

사용 예
  python make_sync_csv.py --claps ~/out/신서연/claps/2026-11-03 --ref exo_front \\
      --cameras exo_left exo_right ego --out sync.csv --guide-out guide.json
"""
import argparse, csv, json, sys
from collections import deque
from pathlib import Path
import numpy as np


def load(claps_dir, ep, cam):
    p = Path(claps_dir) / f"{ep}_{cam}.json"
    if not p.exists():
        return None, "검출 결과 없음"
    r = json.load(open(p))
    if r.get("missing"):
        return None, f"{cam} 폴더 없음"
    if "error" in r:
        return None, f"{cam} 검출 오류"
    return r, ""


def session_mode(diffs, tol):
    """차이 값들 중 ±tol 안에 가장 많이 모이는 곳의 중앙값과 개수."""
    diffs = np.sort(np.asarray(diffs, float))
    if len(diffs) == 0:
        return None, 0
    counts = [int(np.sum(np.abs(diffs - x) <= tol)) for x in diffs]
    best = diffs[int(np.argmax(counts))]
    near = diffs[np.abs(diffs - best) <= tol]
    return float(np.median(near)), len(near)


def clap_list(r):
    """1단계 후보. 후보가 하나도 없는 카메라(예: ego)는 안내 검출 결과를 대신 쓴다."""
    if r is None:
        return []
    return r["claps"] if r["claps"] else r.get("guided", [])


def clock_offsets(eps, cams, ref, data, tol, min_pairs):
    """카메라마다 c[cam] (기준 시각 ~= 이 카메라 시각 + c[cam]). 짝이 많은 쌍부터 이어 붙인다."""
    has_cand = {c: any(data[(ep, c)][0] and data[(ep, c)][0]["claps"] for ep in eps) for c in cams}
    times = {(ep, c): [p["time"] for p in (data[(ep, c)][0]["claps"] if has_cand[c] else clap_list(data[(ep, c)][0]))]
             if data[(ep, c)][0] else [] for ep in eps for c in cams}
    edge = {}
    for i in cams:
        for j in cams:
            if i < j:
                diffs = [ti - tj for ep in eps for ti in times[(ep, i)] for tj in times[(ep, j)]]
                m, n = session_mode(diffs, tol)
                if m is not None and n >= min_pairs:
                    edge[(i, j)] = (m, n); edge[(j, i)] = (-m, n)
    c, how, q = {ref: 0.0}, {ref: "기준"}, deque([ref])
    while q:       # 넓이 우선, 짝 많은 간선 우선
        i = q.popleft()
        for (a, b), (m, n) in sorted(edge.items(), key=lambda kv: -kv[1][1]):
            if a == i and b not in c:
                c[b] = c[i] + m          # t_ref - t_b = (t_ref - t_a) + (t_a - t_b)
                how[b] = f"{how[i]} -> {b} (짝 {n})" if i != ref else f"{ref}-{b} 짝 {n}"
                q.append(b)
    return c, how, edge


def coarse_offsets(eps, cams, data, c):
    """대표값이 없는 카메라: 녹화 시작 시각으로 대략 c[cam] (버튼 누르는 차이만큼 1~2 s 오차).
    = 중앙값_에피소드( 다른 카메라 첫 프레임을 기준 시간축에 옮긴 시각 - 이 카메라 첫 프레임 시각 )."""
    out = {}
    for cam in cams:
        if cam in c:
            continue
        v = [data[(ep, x)][0]["t0"] + c[x] - data[(ep, cam)][0]["t0"]
             for ep in eps for x in c if data[(ep, x)][0] and data[(ep, cam)][0]]
        if v:
            out[cam] = float(np.median(v))
    return out


def clusters(items, tol):
    """items: [(기준 시간축 시각, cam, clap)] -> 시각순으로 tol 안에 이어지는 묶음."""
    items = sorted(items, key=lambda x: x[0])
    out, cur = [], []
    for it in items:
        if cur and it[0] - cur[0][0] > tol:
            out.append(cur); cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claps", required=True, help="detect_claps.py --out 폴더")
    ap.add_argument("--ref", required=True, help="기준 카메라 폴더 이름 (예: exo_front)")
    ap.add_argument("--cameras", nargs="+", required=True, help="기준 외 카메라 폴더 이름들")
    ap.add_argument("--names", nargs="*", default=[],
                    help="폴더 이름=sync.csv camera 값 (기본: exo_ 를 뗀 이름. 예: exo_left -> left)")
    ap.add_argument("--episodes", nargs="*", help="생략하면 --claps 폴더에 결과가 있는 에피소드 전부")
    ap.add_argument("--tol", type=float, default=0.15, help="같은 박수로 볼 허용 차이 (s)")
    ap.add_argument("--min-pairs", type=int, default=3, help="세션 대표값을 믿을 최소 짝 수")
    ap.add_argument("--min-support", type=int, default=2, help="박수로 인정할 최소 카메라 수 (1단계 후보 기준)")
    ap.add_argument("--single-drop", type=float, default=2.0,
                    help="카메라 1대만 본 후보라도 drop 이 이 이상이면 박수로 인정 (다른 카메라는 안내 검출로 확인)")
    ap.add_argument("--coarse-half", type=float, default=2.5, help="대표값 없는 카메라의 넓은 안내 검출 반폭 (s)")
    ap.add_argument("--min-gap", type=float, default=3.0, help="시작·끝 박수 최소 간격 (s)")
    ap.add_argument("--drift-warn", type=float, default=0.05, help="note 에 표시할 드리프트 (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--guide-out", help="빠진 카메라 안내 검출 목록 (detect_claps.py --guide 입력)")
    ap.add_argument("--report", help="에피소드·카메라별 고른 박수 표 (csv, 확인용)")
    a = ap.parse_args(argv)

    cams = [a.ref] + a.cameras
    names = {c: (c[4:] if c.startswith("exo_") else c) for c in a.cameras}
    names.update(dict(s.split("=") for s in a.names))
    eps = a.episodes or sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                                for p in Path(a.claps).glob("T*_E*_*.json")})
    data = {(ep, c): load(a.claps, ep, c) for ep in eps for c in cams}

    c, how, edge = clock_offsets(eps, cams, a.ref, data, a.tol, a.min_pairs)
    coarse = coarse_offsets(eps, cams, data, c)
    for cam in cams[1:]:
        if cam in c:
            print(f"{cam}: 기준 - 이 카메라 대표 시간 차 {c[cam]:.3f} s ({how[cam]})")
        elif cam in coarse:
            print(f"{cam}: 대표값 없음 -> 녹화 시작 시각으로 대략 {coarse[cam]:.3f} s, ±{a.coarse_half} s 안내 검출 요청")
        else:
            print(f"{cam}: 대표값 없음 (짝 부족) -> 이 카메라 행은 usable=0")

    rows, guide, report = [], {}, []
    for ep in eps:
        items = []
        for cam in cams:
            r = data[(ep, cam)][0]
            if r is None or cam not in c:
                continue
            items += [(p["time"] + c[cam], cam, p) for p in r["claps"]]
            items += [(p["time"] + c[cam], cam, p) for p in r.get("guided", [])]
        groups = []
        for g in clusters(items, a.tol):
            cand = [p for _, _, p in g if not p.get("guided")]
            support = {cam for _, cam, p in g if not p.get("guided")}
            if len(support) >= a.min_support or (cand and max(p["drop"] for p in cand) >= a.single_drop):
                groups.append(g)
        picks = {}
        if groups and groups[-1][0][0] - groups[0][0][0] >= a.min_gap:
            for name, g in (("start", groups[0]), ("end", groups[-1])):
                center = float(np.median([x[0] for x in g if not x[2].get("guided")]))
                for cam in cams:
                    if data[(ep, cam)][0] is None:
                        continue
                    if cam not in c:
                        if cam in coarse:   # 넓은 안내 검출
                            guide.setdefault("_half", {})[cam] = a.coarse_half
                            guide.setdefault(ep, {}).setdefault(cam, []).append(round(center - coarse[cam], 4))
                        continue
                    mine = [x for x in g if x[1] == cam]
                    if mine:   # 1단계 후보 우선, 그다음 묶음 중심에 가까운 것
                        best = min(mine, key=lambda x: (bool(x[2].get("guided")), abs(x[0] - center)))
                        picks[(name, cam)] = best[2]
                    else:
                        guide.setdefault(ep, {}).setdefault(cam, []).append(round(center - c[cam], 4))
        for (name, cam), p in sorted(picks.items()):
            report.append({"episode_id": ep, "clap": name, "camera": cam, "frame": p["frame"],
                           "time": f"{p['time']:.4f}", "on_ref_axis": f"{p['time'] + c[cam]:.4f}",
                           "guided": int(bool(p.get("guided"))), "drop": p["drop"]})

        for cam in a.cameras:
            row = {"episode_id": ep, "camera": names[cam], "offset_start_s": "", "offset_end_s": "",
                   "drift_s": "", "s0": "", "s1": "", "usable": 0, "note": ""}
            err = data[(ep, a.ref)][1] or data[(ep, cam)][1]
            need = [("start", a.ref), ("start", cam), ("end", a.ref), ("end", cam)]
            if err:
                row["note"] = err
            elif cam not in c:
                row["note"] = "세션 대표값 없음 (짝 부족)"
            elif not groups or not picks:
                row["note"] = "여러 카메라가 함께 본 시작·끝 박수 없음"
            elif any(k not in picks for k in need):
                miss = sorted({f"{'시작' if n == 'start' else '끝'}({names.get(cc, cc)})" for n, cc in need
                               if (n, cc) not in picks})
                row["note"] = "박수 못 찾음: " + ", ".join(miss)
            else:
                rs, cs, re_, ce = (picks[k] for k in need)
                o0, o1 = rs["time"] - cs["time"], re_["time"] - ce["time"]
                notes = [f"frames ref {rs['frame']}/{re_['frame']}, cam {cs['frame']}/{ce['frame']}"]
                g = [nm for nm, p in zip(["ref시작", "시작", "ref끝", "끝"], (rs, cs, re_, ce)) if p.get("guided")]
                if g:
                    notes.insert(0, "안내 검출: " + ",".join(g))
                if abs(o1 - o0) > a.drift_warn:
                    notes.insert(0, "드리프트 큼 - 확인")
                row.update(offset_start_s=f"{o0:.4f}", offset_end_s=f"{o1:.4f}", drift_s=f"{o1 - o0:.4f}",
                           s0=f"{cs['time']:.4f}", s1=f"{ce['time']:.4f}", usable=1, note="; ".join(notes))
            rows.append(row)

    cols = ["episode_id", "camera", "offset_start_s", "offset_end_s", "drift_s", "s0", "s1", "usable", "note"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    if a.report:
        with open(a.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0]) if report else ["episode_id"])
            w.writeheader(); w.writerows(report)
    if a.guide_out:
        json.dump(guide, open(a.guide_out, "w"), indent=1)
        print(f"{a.guide_out}: 안내 검출 {sum(len(v) for v in guide.values())} (에피소드·카메라)")
    ok = sum(int(r["usable"]) == 1 for r in rows)
    print(f"{a.out}: {len(rows)}행, usable=1 {ok}행")


if __name__ == "__main__":
    sys.exit(main())
