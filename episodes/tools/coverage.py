"""삼각측량이 될지 미리 보기: 같은 순간에 exo 카메라 몇 대가 손을 잡았는지 (기준 카메라 시간축).

extract_all.py 의 캐시만 읽는다 (영상·MediaPipe 없음). 시간 맞추기는 process_episode.py 와 같은 함수
(read_sync_row, align_to_ref) 를 쓰므로, 삼각측량 성공률(min_views=2)의 상한에 가깝다.

  python coverage.py --raw ~/data/raw/T1 --cache ~/out/seohyeon/kp_cache_v2/T1 \\
      --sync ../../sync/sync.csv --out ~/out/seohyeon/coverage_T1.csv

구간: 네 카메라(exo 3대 + ego)가 모두 녹화 중인 기준 시각 범위. --start/--end 가 아직 없어서
박수 전후 손을 내린 구간도 들어 있다 (실제 에피소드 구간에서는 이보다 높게 나온다).

출력 열 (에피소드당 한 줄):
  frames            구간 프레임 수 (기준 카메라)
  wrist_0..3        손목을 잡은 exo 카메라가 0/1/2/3대인 프레임 비율
  wrist_2plus       2대 이상 = 손목 삼각측량 가능
  tips_2plus        엄지끝·검지끝 둘 다 2대 이상
  hand_2plus        손 키포인트 5개(wrist, index_mcp, pinky_mcp, thumb_tip, index_tip) 모두 2대 이상
  pair_<a>_<b>      손목을 그 두 카메라가 같이 잡은 비율
  ego_pinch         ego 엄지-검지 거리가 있는 비율 (같은 구간)
"""
import argparse, csv
from itertools import combinations
from pathlib import Path
import numpy as np

from extractors import EXO_NAMES
from io_utils import (find_episode_folder, read_frame_times, read_sync_row, read_records,
                      kp_cache_path, match_frames, SyncError)
from multiview import align_to_ref, interp_track, cam_time_to_ref, DEFAULT_TRI

REF, OTHERS = "front", ["left", "right"]
HAND_KP = ["wrist", "index_mcp", "pinky_mcp", "thumb_tip", "index_tip"]
PAIRS = list(combinations([REF] + OTHERS, 2))
FIELDS = (["episode_id", "hand", "frames"] + [f"wrist_{n}" for n in range(4)]
          + ["wrist_2plus", "tips_2plus", "hand_2plus"] + [f"pair_{a}_{b}" for a, b in PAIRS]
          + ["ego_pinch", "error"])


def episode(raw, cache, sync_csv, eid, hand):
    """에피소드 하나 -> 출력 행 dict."""
    seen, span = {}, []
    t_ref = read_frame_times(find_episode_folder(raw, eid, f"exo_{REF}"))
    for c in [REF] + OTHERS:
        t_c = read_frame_times(find_episode_folder(raw, eid, f"exo_{c}"))
        uv = match_frames(np.load(kp_cache_path(cache, f"exo_{c}", eid, hand)), t_c, c)
        if c == REF:
            seen[c], span = uv, [(t_c[0], t_c[-1])]
        else:
            s = read_sync_row(sync_csv, eid, camera=c)
            seen[c] = align_to_ref(t_c, uv, t_ref, s)
            r = cam_time_to_ref(t_c[[0, -1]], **s); span.append((r[0], r[1]))
    t_e = read_frame_times(find_episode_folder(raw, eid, "ego"))
    d = match_frames(np.load(kp_cache_path(cache, "ego", eid, hand)), t_e, "ego")
    s = read_sync_row(sync_csv, eid, camera="ego")
    t_e_ref = cam_time_to_ref(t_e, **s); span.append((t_e_ref[0], t_e_ref[-1]))
    pinch = interp_track(t_e_ref, d, t_ref, DEFAULT_TRI["max_interp_gap"] * np.median(np.diff(t_e)))

    lo, hi = max(a for a, _ in span), min(b for _, b in span)
    win = (t_ref >= lo) & (t_ref <= hi)
    if not win.any():
        raise ValueError("카메라들이 동시에 녹화한 구간이 없음 (sync.csv 확인)")
    ok = {c: np.isfinite(seen[c][win]).all(axis=2) for c in seen}       # (T,K) 카메라별
    nv = sum(ok.values())                                               # (T,K) 잡은 카메라 수
    k = {n: EXO_NAMES.index(n) for n in HAND_KP}
    row = {"episode_id": eid, "hand": hand, "frames": int(win.sum())}
    for n in range(4):
        row[f"wrist_{n}"] = (nv[:, k["wrist"]] == n).mean()
    row["wrist_2plus"] = (nv[:, k["wrist"]] >= 2).mean()
    row["tips_2plus"] = ((nv[:, k["thumb_tip"]] >= 2) & (nv[:, k["index_tip"]] >= 2)).mean()
    row["hand_2plus"] = (nv[:, [k[n] for n in HAND_KP]] >= 2).all(axis=1).mean()
    for a, b in PAIRS:
        row[f"pair_{a}_{b}"] = (ok[a][:, k["wrist"]] & ok[b][:, k["wrist"]]).mean()
    row["ego_pinch"] = np.isfinite(pinch[win]).mean()
    return {f: (f"{v:.3f}" if isinstance(v, float) else v) for f, v in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="태스크 폴더, 예: ~/data/raw/T1")
    ap.add_argument("--cache", required=True, help="extract_all.py 의 --out, 예: ~/out/seohyeon/kp_cache_v2/T1")
    ap.add_argument("--sync", required=True)
    ap.add_argument("--out", required=True, help="결과 CSV (~/project 밖)")
    ap.add_argument("--ids", nargs="*", help="생략하면 exo_front 아래 *_E* 전부")
    ap.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="--records 에 없는 에피소드에 쓸 손")
    a = ap.parse_args()

    raw, cache = Path(a.raw).expanduser(), Path(a.cache).expanduser()
    ids = a.ids or sorted(p.name for p in (raw / "exo_front").glob("*_E*") if p.is_dir())
    hand = {k: r["hand"] for k, r in read_records(a.records).items() if r["hand"] in ("right", "left")}
    rows = []
    for i, eid in enumerate(ids, 1):
        h = hand.get(eid, a.hand)
        try:
            r = episode(raw, cache, a.sync, eid, h)
        except (OSError, ValueError, SyncError) as e:
            r = {"episode_id": eid, "hand": h, "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        print(f"[{i}/{len(ids)}] {eid} {h:<5} " + (f"오류 {r['error']}" if r.get("error") else
              f"손목 2대+ {r['wrist_2plus']}  손 전체 2대+ {r['hand_2plus']}  ego {r['ego_pinch']}"), flush=True)

    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    good = [r for r in rows if not r.get("error")]
    print(f"\n오류 {len(rows) - len(good)}개. 저장 -> {out}")
    for h in ("right", "left"):
        g = [r for r in good if r["hand"] == h]
        if not g:
            continue
        print(f"\n[{h}] {len(g)}개   중앙값 (최저)")
        for f in ["wrist_2plus", "wrist_3", "tips_2plus", "hand_2plus"] + [f"pair_{a}_{b}" for a, b in PAIRS] + ["ego_pinch"]:
            v = [float(r[f]) for r in g]
            print(f"  {f:<18} {np.median(v):.2f} ({min(v):.2f})")
        low = sorted(g, key=lambda r: float(r["wrist_2plus"]))[:5]
        print("  손목 2대+ 낮은 순:", ", ".join(f"{r['episode_id']} {r['wrist_2plus']}" for r in low))


if __name__ == "__main__":
    main()
