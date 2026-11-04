"""세션 시계 모델로 sync.csv 의 빈 행 채우기 (2번 싱크).

박수 검출(make_sync_csv.py)이 실패한 에피소드도 싱크를 맞추기 위한 단계.
폰 시계(data.jsonl time) 사이 차이는 폰이 잠들지 않는 한 한 촬영 세션 안에서 거의 일정하다
(T1 31분 동안 기울기 0.6~12.5 ppm, 30분에 최대 약 20 ms). 그래서 믿을 수 있는 박수(기준점)
몇 개로 카메라마다 "시계 차 = a + b * 기준 시각" 직선을 맞추고, 박수가 없거나 틀린 행을 이 직선으로 채운다.

기준점
  1) sync.csv 의 usable=1 행 중 |drift_s| <= --drift-max 인 것: 시작·끝 박수 두 점.
     에피소드 길이(수십 초) 안의 실제 드리프트는 1 ms 미만이라, 드리프트가 1프레임을 넘으면
     시작·끝 중 하나를 잘못 잡은 것으로 보고 기준점에서 뺀다.
  2) --manual 손 표시 CSV (episode_id, camera, clap, frame[, note]): 같은 에피소드·같은 박수(clap)를
     기준 카메라와 다른 카메라에서 사람이 고른 프레임 번호 (0부터, data.jsonl 프레임 순서).
     label_sheets.py 가 만든 그림의 번호를 그대로 적는다.
  기준점은 모델과 2프레임(--reject) 넘게 어긋나면 이상값으로 빼고 report 에 남긴다 (지우지 않음).

세션 구간
  폰이 잠들면 시계 차가 수 초 이상 뛴다. 에피소드마다 "기준 카메라 녹화 시작 시각 - 이 카메라 녹화 시작 시각"
  (버튼 누르는 차이 때문에 ±1~2 s 흔들림)의 이동 중앙값이 --jump 초 넘게 바뀌면 새 구간으로 나눈다.
  구간마다 따로 맞추고, 기준점이 없는 구간은 usable=0.

출력: sync.csv 와 같은 열. 행 종류는 note 앞머리로 구분
  "박수"       : 박수 값 그대로 (기준점으로 쓴 행). 모델과의 차이를 함께 적음
  "시계 모델"  : s0/s1 = 이 카메라 첫·마지막 프레임 시각, offset = 그 시각의 모델 값
                 (io_utils.read_sync_row / cam_time_to_ref 식 그대로 쓸 수 있음)
  usable=0     : 폴더 없음, 또는 그 구간에 기준점 부족

사용 예
  python fit_clock.py --sync sync_T1_claps.csv --raw ~/data/raw/T1 --ref exo_front \\
      --cameras exo_left exo_right ego --out sync_T1.csv --report clock_T1.csv
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

COLS = ["episode_id", "camera", "offset_start_s", "offset_end_s", "drift_s", "s0", "s1", "usable", "note"]


def frame_times(jsonl):
    """data.jsonl 의 영상 프레임 시각 (T,) s. detect_claps.frame_times 와 같은 규칙."""
    ts = []
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "frames" in o:
            ts.append(float(o["time"] if "time" in o else o["frames"][0]["time"]))
    return np.array(ts)


def f4(x):
    return f"{x:.4f}"


def fit(points, slope_min_eps, slope_min_span):
    """points: [(t_ref, off, ep)] -> (a, b, t_c). off(t) = a + b * (t - t_c). 기준점 에피소드가 적거나
    시간 폭이 짧으면 기울기 0 (상수)."""
    t = np.array([p[0] for p in points]); y = np.array([p[1] for p in points])
    t_c = float(np.median(t))
    n_eps = len({p[2] for p in points})
    if n_eps >= slope_min_eps and t.max() - t.min() >= slope_min_span:
        A = np.c_[np.ones_like(t), t - t_c]
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(a), float(b), t_c
    return float(np.median(y)), 0.0, t_c


def robust_fit(points, reject, slope_min_eps, slope_min_span):
    """중앙값에서 크게 벗어난 점을 빼고 맞춘 뒤, 모델에서 reject 초 넘게 어긋난 점을 빼고 다시 맞춘다."""
    pts = list(points)
    if not pts:
        return None, [], []
    med = np.median([p[1] for p in pts])
    keep = [p for p in pts if abs(p[1] - med) <= max(3 * reject, 0.1)]
    for _ in range(3):
        if not keep:
            break
        m = fit(keep, slope_min_eps, slope_min_span)
        new = [p for p in pts if abs(p[1] - (m[0] + m[1] * (p[0] - m[2]))) <= reject]
        if {id(p) for p in new} == {id(p) for p in keep}:
            break
        keep = new
    if not keep:
        return None, [], pts
    m = fit(keep, slope_min_eps, slope_min_span)
    rej = [p for p in pts if p not in keep]
    return m, keep, rej


def evaluate(m, t):
    return m[0] + m[1] * (t - m[2])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", required=True, help="make_sync_csv.py 결과 (박수 기반 표)")
    ap.add_argument("--raw", required=True, help="카메라 폴더들이 들어 있는 폴더 (예: ~/data/raw/T1)")
    ap.add_argument("--ref", default="exo_front", help="기준 카메라 폴더 이름")
    ap.add_argument("--cameras", nargs="+", required=True, help="기준 외 카메라 폴더 이름들")
    ap.add_argument("--names", nargs="*", default=[], help="폴더 이름=sync.csv camera 값 (기본: exo_ 를 뗀 이름)")
    ap.add_argument("--manual", nargs="*", default=[], help="손 표시 CSV (episode_id,camera,clap,frame)")
    ap.add_argument("--drift-max", type=float, default=0.034, help="박수 행을 기준점으로 쓸 최대 |drift_s| (s, 1프레임)")
    ap.add_argument("--reject", type=float, default=0.067, help="모델에서 이만큼(s) 넘게 어긋난 기준점은 이상값 (2프레임)")
    ap.add_argument("--jump", type=float, default=3.0, help="녹화 시작 시각 차이가 이만큼(s) 바뀌면 새 구간")
    ap.add_argument("--slope-min-eps", type=int, default=6, help="기울기를 맞출 최소 기준점 에피소드 수 (미만이면 상수)")
    ap.add_argument("--slope-min-span", type=float, default=300.0, help="기울기를 맞출 최소 시간 폭 (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="기준점·이상값·LOO 오차 표 (csv)")
    a = ap.parse_args(argv)

    raw = Path(a.raw).expanduser()
    names = {c: (c[4:] if c.startswith("exo_") else c) for c in a.cameras}
    names.update(dict(s.split("=") for s in a.names))
    folder_of = {v: k for k, v in names.items()}

    rows = {(r["episode_id"], r["camera"]): r for r in csv.DictReader(open(a.sync, newline=""))}
    eps = sorted(p.name for p in (raw / a.ref).glob("T*_E*") if p.is_dir())

    ft = {}
    def times(ep, cam):
        if (ep, cam) not in ft:
            j = raw / cam / ep / "data.jsonl"
            ft[(ep, cam)] = frame_times(j) if j.exists() and (raw / cam / ep / "data.mov").exists() else None
        return ft[(ep, cam)]

    manual = {}
    for path in a.manual:
        for r in csv.DictReader(open(path, newline="")):
            if r.get("frame", "").strip() == "":
                continue
            manual.setdefault((r["episode_id"], r["clap"].strip()), {})[r["camera"].strip()] = int(r["frame"])

    out_rows, report = [], []
    for cam in a.cameras:
        name = names[cam]
        # 1) 구간 나누기: 녹화 시작 시각 차이의 이동 중앙값
        coarse = {}
        for ep in eps:
            tr, tc = times(ep, a.ref), times(ep, cam)
            if tr is not None and tc is not None and len(tr) and len(tc):
                coarse[ep] = tr[0] - tc[0]
        ce = [ep for ep in eps if ep in coarse]
        vals = np.array([coarse[ep] for ep in ce])
        smooth = np.array([np.median(vals[max(0, i - 2): i + 3]) for i in range(len(vals))])
        seg = {}
        k = 0
        for i, ep in enumerate(ce):
            if i and abs(smooth[i] - smooth[i - 1]) > a.jump:
                k += 1
            seg[ep] = k
        odd = {ep for i, ep in enumerate(ce) if abs(vals[i] - smooth[i]) > a.jump}

        # 2) 기준점 모으기
        pts = {}
        for ep in ce:
            r = rows.get((ep, name))
            if r and r["usable"] == "1" and r["drift_s"] != "" and abs(float(r["drift_s"])) <= a.drift_max:
                o0, o1, s0, s1 = (float(r[c]) for c in ("offset_start_s", "offset_end_s", "s0", "s1"))
                pts.setdefault(seg[ep], []).extend([(s0 + o0, o0, ep, "박수 시작"), (s1 + o1, o1, ep, "박수 끝")])
            for clap in ("start", "end"):
                mk = manual.get((ep, clap), {})
                if a.ref in mk and cam in mk:
                    tr, tc = times(ep, a.ref), times(ep, cam)
                    t_ref = tr[mk[a.ref]]
                    pts.setdefault(seg[ep], []).append((t_ref, t_ref - tc[mk[cam]], ep, f"손 표시 {clap}"))

        # 3) 구간마다 맞추기 + LOO
        models = {}
        for s in sorted(set(seg.values())):
            p = [(x[0], x[1], x[2]) for x in pts.get(s, [])]
            src = {(x[0], x[1], x[2]): x[3] for x in pts.get(s, [])}
            m, keep, rej = robust_fit(p, a.reject, a.slope_min_eps, a.slope_min_span)
            loo = []
            if m is not None:
                for ep in sorted({x[2] for x in keep}):
                    rest = [x for x in keep if x[2] != ep]
                    if not rest:
                        continue
                    mm = fit(rest, a.slope_min_eps, a.slope_min_span)
                    loo += [abs(x[1] - evaluate(mm, x[0])) for x in keep if x[2] == ep]
            loo = np.array(loo)
            models[s] = (m, len({x[2] for x in keep}), loo)
            seg_eps = [ep for ep in ce if seg[ep] == s]
            msg = (f"{name} 구간 {s} ({seg_eps[0]}~{seg_eps[-1]}, {len(seg_eps)}개): ")
            if m is None:
                print(msg + "기준점 없음 -> usable=0")
            else:
                print(msg + f"기준점 {len(keep)}점/{len({x[2] for x in keep})}에피소드, 이상값 {len(rej)}, "
                      f"기울기 {m[1]*1e6:.1f} ppm, "
                      + (f"LOO |err| 중앙 {np.median(loo)*1000:.1f} ms, 95% {np.percentile(loo,95)*1000:.1f} ms, "
                         f"최대 {loo.max()*1000:.1f} ms" if len(loo) else "LOO 불가 (기준점 에피소드 1개)"))
            for x in p:
                res = x[1] - evaluate(m, x[0]) if m is not None else float("nan")
                report.append({"camera": name, "segment": s, "episode_id": x[2], "source": src[x],
                               "t_ref": f4(x[0]), "offset": f4(x[1]), "resid_ms": f"{res*1000:.1f}",
                               "used": int(x in keep)})

        # 4) 행 만들기
        for ep in eps:
            r = rows.get((ep, name), {})
            old = r.get("note", "")
            tc = times(ep, cam)
            if tc is None or times(ep, a.ref) is None:
                out_rows.append({**{c: "" for c in COLS}, "episode_id": ep, "camera": name, "usable": 0,
                                 "note": "폴더 없음" if tc is None else "기준 카메라 폴더 없음"})
                continue
            m, n_eps, loo = models.get(seg.get(ep), (None, 0, np.array([])))
            flag = "; 녹화 시작 차이 이상 - 확인" if ep in odd else ""
            is_anchor = any(x["episode_id"] == ep and x["camera"] == name and x["used"] and x["source"].startswith("박수")
                            for x in report)
            if is_anchor:
                o0 = float(r["offset_start_s"])
                d = (o0 - evaluate(m, float(r["s0"]) + o0)) * 1000
                out_rows.append({**{c: r[c] for c in COLS}, "note": f"박수 (모델과 {d:+.0f} ms); {old}{flag}"})
                continue
            if m is None:
                out_rows.append({**{c: "" for c in COLS}, "episode_id": ep, "camera": name, "usable": 0,
                                 "note": f"시계 모델 기준점 없음 (구간 {seg.get(ep)}); 박수: {old}{flag}"})
                continue
            s0, s1 = float(tc[0]), float(tc[-1])
            o0 = evaluate(m, s0 + evaluate(m, s0 + m[0]))
            o1 = evaluate(m, s1 + evaluate(m, s1 + m[0]))
            q = f"LOO 95% {np.percentile(loo,95)*1000:.0f} ms" if len(loo) else "LOO 불가"
            out_rows.append({"episode_id": ep, "camera": name, "offset_start_s": f4(o0), "offset_end_s": f4(o1),
                             "drift_s": f4(o1 - o0), "s0": f4(s0), "s1": f4(s1), "usable": 1,
                             "note": f"시계 모델 (구간 {seg[ep]}, 기준점 {n_eps}에피소드, {q}); 박수: {old}{flag}"})

    order = {n: i for i, n in enumerate(names[c] for c in a.cameras)}
    out_rows.sort(key=lambda r: (r["episode_id"], order[r["camera"]]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(out_rows)
    n1 = sum(int(r["usable"]) == 1 for r in out_rows)
    nb = sum(str(r["note"]).startswith("박수") for r in out_rows)
    print(f"{a.out}: {len(out_rows)}행, usable=1 {n1}행 (박수 {nb}, 시계 모델 {n1 - nb})")
    if a.report:
        with open(a.report, "w", newline="") as f:
            w = csv.DictWriter(f, ["camera", "segment", "episode_id", "source", "t_ref", "offset", "resid_ms", "used"])
            w.writeheader(); w.writerows(report)


if __name__ == "__main__":
    sys.exit(main())
