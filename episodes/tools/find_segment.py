"""박수 없이 동작 구간을 찾는다. 촬영 순서: 시작 박수 -> 손을 무릎에 -> 동작 -> 손을 무릎에 -> 끝 박수.

무릎 위에서는 작업 손 쪽 카메라도 손을 놓친다. 그 카메라에서 짧은 끊김을 메운 뒤 손이 보이는 가장 긴 구간을 동작으로 본다.
3D 기준이면 쥐는 동안 끊기지만 (다른 카메라가 가려짐), 작업 손 쪽 카메라는 쥐는 동안에도 손을 본다.
extract_all.py 캐시와 sync.csv 만 읽는다.

  python find_segment.py --raw ~/data/raw --cache ~/out/seohyeon/kp_cache_v2 --sync ../../sync/sync.csv --out ~/out/seohyeon/segments.csv

출력 열: episode_id, hand, start, end(미포함) front 프레임, sec, seen(구간 안 손 보이는 비율), runs(긴 구간 수),
  second(두 번째로 긴 구간 길이 s), clap_start, clap_end(sync.csv note 에 있으면), in_claps, error
박수 프레임을 아는 에피소드로 '구간이 박수 사이에 들어오는가' 를 집계해 출력한다.
"""
import argparse, csv, re
from pathlib import Path
import numpy as np

from extractors import EXO_NAMES
from io_utils import find_episode_folder, kp_cache_path, match_frames, read_frame_times, read_records, read_sync_row
from multiview import align_to_ref

REF = "front"
FIELDS = ["episode_id", "hand", "start", "end", "sec", "seen", "runs", "second", "clap_start", "clap_end", "in_claps", "error"]


def clap_notes(sync_csv):
    out = {}
    with open(sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.search(r"frames ref (\d+)/(\d+)", r["note"])
            if m:
                out[r["episode_id"]] = (int(m[1]), int(m[2]))
    return out


def seen_on_front(raw, cache, sync_csv, eid, hand):
    """작업 손 쪽 카메라가 손목을 잡았는가, 기준 카메라 프레임마다 (T,). 시각 t_ref 도 함께."""
    t_ref = read_frame_times(find_episode_folder(raw, eid, f"exo_{REF}"))
    t_c = read_frame_times(find_episode_folder(raw, eid, f"exo_{hand}"))
    uv = match_frames(np.load(kp_cache_path(cache, f"exo_{hand}", eid, hand)), t_c, hand)
    w = uv[:, EXO_NAMES.index("wrist")]
    a = align_to_ref(t_c, w, t_ref, read_sync_row(sync_csv, eid, camera=hand), max_gap_factor=3.0)
    return np.isfinite(a).all(axis=1), t_ref


def runs_of(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def find_run(seen, t, max_gap_s, min_run_s):
    """(시작, 끝 미포함, 긴 구간 수, 두 번째 길이 s). max_gap_s 보다 짧은 끊김은 메운다."""
    m = seen.copy()
    for a, b in runs_of(~seen):
        if a > 0 and b < len(seen) and t[b] - t[a - 1] < max_gap_s:
            m[a:b] = True
    R = sorted(runs_of(m), key=lambda r: t[r[1] - 1] - t[r[0]], reverse=True)
    if not R:
        return None
    dur = [t[b - 1] - t[a] for a, b in R]
    return R[0][0], R[0][1], sum(d >= min_run_s for d in dur), (dur[1] if len(dur) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="태스크 폴더들의 상위, 예: ~/data/raw")
    ap.add_argument("--cache", required=True, help="태스크별 캐시의 상위")
    ap.add_argument("--sync", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-gap", type=float, default=0.25, help="이보다 짧은 끊김은 메움 (s)")
    ap.add_argument("--min-run", type=float, default=0.5, help="긴 구간으로 세는 길이 (s)")
    a = ap.parse_args()

    rec, claps = read_records(a.records), clap_notes(a.sync)
    raw, cache = Path(a.raw).expanduser(), Path(a.cache).expanduser()
    rows = []
    for eid in a.ids or sorted(rec):
        h, T = rec[eid]["hand"], eid.split("_")[0]
        row = {"episode_id": eid, "hand": h}
        try:
            seen, t = seen_on_front(raw / T, cache / T, a.sync, eid, h)
            r = find_run(seen, t, a.max_gap, a.min_run)
            if r is None:
                raise ValueError("손이 보이는 구간 없음")
            s, e, n, sec2 = r
            row.update(start=s, end=e, sec=round(t[e - 1] - t[s], 2), seen=round(float(seen[s:e].mean()), 3), runs=n, second=round(sec2, 2))
            if eid in claps:
                c0, c1 = claps[eid]
                row.update(clap_start=c0, clap_end=c1, in_claps=int(c0 <= s and e - 1 <= c1))
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"
        rows.append(row)
    p = Path(a.out).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    good = [r for r in rows if not r.get("error")]
    print(f"에피소드 {len(rows)}, 오류 {len(rows) - len(good)}: {[(r['episode_id'], r['error']) for r in rows if r.get('error')][:5]}")
    print("구간 길이 s: 최소 %.1f, 중앙값 %.1f, 최대 %.1f" % (min(r["sec"] for r in good), np.median([r["sec"] for r in good]), max(r["sec"] for r in good)))
    print("구간 안 손 보이는 비율 중앙값 %.2f, 최저 %.2f" % (np.median([r["seen"] for r in good]), min(r["seen"] for r in good)))
    chk = [r for r in good if "in_claps" in r]
    if chk:
        bad = [r for r in chk if not r["in_claps"]]
        m0 = [r["start"] - r["clap_start"] for r in chk]; m1 = [r["clap_end"] - r["end"] for r in chk]
        print(f"\n박수를 아는 {len(chk)}개: 구간이 박수 사이 {len(chk) - len(bad)}개, 벗어남 {len(bad)}개")
        print(f"  시작 박수 뒤 프레임: 중앙값 {np.median(m0):.0f}, 최소 {min(m0)} / 끝 박수 앞 프레임: 중앙값 {np.median(m1):.0f}, 최소 {min(m1)}")
        print("  벗어남:", [(r["episode_id"], r["start"], r["end"], r["clap_start"], r["clap_end"]) for r in bad][:15])
    for T in ("T1", "T2", "T3"):
        g = [r for r in good if r["episode_id"].startswith(T)]
        if g:
            print(f"{T}: {len(g)}개, 길이 중앙값 {np.median([r['sec'] for r in g]):.1f} s, 최소 {min(r['sec'] for r in g):.1f} s")
    short = sorted(good, key=lambda r: r["sec"])[:8]
    print("짧은 구간:", [(r["episode_id"], r["sec"]) for r in short])
    amb = sorted(good, key=lambda r: -(r["second"] / max(r["sec"], 0.01)))[:8]
    print("두 번째 구간이 긴 순 (헷갈릴 위험):", [(r["episode_id"], r["sec"], r["second"]) for r in amb])
    print("저장 ->", p)


if __name__ == "__main__":
    main()
