"""sync.csv note 에 front 박수 프레임이 있는 에피소드를 박수 사이 구간으로 한꺼번에 처리한다 (시험용).

정식 --start/--end 는 labeling_rules.md 기준인데 아직 없어서, 시작 박수 ~ 끝 박수를 임시 구간으로 쓴다.
손, 작업자, 태그, 용도는 records.csv, 보정 파일은 io_utils.find_calib, 2D 는 extract_all.py 캐시에서 읽는다.
예비(E066~) 의 split 은 아직 정해지지 않아 train 으로 두고 notes 에 적는다.

  sbatch -p p02 -c 16 --mem=32G -o ~/out/seohyeon/process_all.log --wrap \\
      "cd ~/project/episodes/tools && python process_all.py --raw ~/data/raw --cache ~/out/seohyeon/kp_cache_v2 \\
       --sync ../../sync/sync.csv --calib-dir ../../sync --out ~/out/seohyeon/episodes_clap --jobs 16"

출력: <out>/<ID>/ (process_episode.py 형식, 영상은 --copy-video 가 없으면 원본 링크), <out>/summary.csv
  frames, tri_wrist(손목 3D 비율), reproj_wrist, missing_wrist, pinch_missing,
  run_start, run_end, run_len: 손목 3D 가 이어진 가장 긴 구간 (front 영상 전체 프레임 번호)
이미 처리한 에피소드는 건너뛴다 (--redo 로 다시).
"""
import argparse, csv, json, os, re, subprocess, sys
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np

from extractors import EXO_NAMES
from io_utils import find_calib, read_records

FIELDS = ["episode_id", "hand", "calib", "start", "end", "frames", "tri_wrist", "reproj_wrist", "missing_wrist",
          "pinch_missing", "run_start", "run_end", "run_len", "error"]


def clap_frames(sync_csv):
    """{episode_id: (시작 박수, 끝 박수)} front 프레임 번호. note 의 'frames ref A/B'."""
    out = {}
    with open(sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.search(r"frames ref (\d+)/(\d+)", r["note"])
            if m:
                out[r["episode_id"]] = (int(m[1]), int(m[2]))
    return out


def longest_run(ok):
    """True 가 이어진 가장 긴 구간 (시작, 끝 미포함)."""
    d = np.diff(np.r_[0, ok.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if not len(s):
        return 0, 0
    i = int(np.argmax(e - s))
    return int(s[i]), int(e[i])


def summarize(row, ep):
    q = json.load(open(ep / "quality.json"))
    w = EXO_NAMES.index("wrist")
    s, e = longest_run(np.isfinite(np.load(ep / "keypoints.npy")[:, w]).all(axis=1))
    row.update(frames=q["frames"], tri_wrist=q["tri_ok_ratio"][w], reproj_wrist=q["tri_reproj_median_px"][w],
               missing_wrist=q["missing_after"][w], pinch_missing=round(q["pinch_missing"], 4),
               run_start=row["start"] + s, run_end=row["start"] + e, run_len=e - s)


def run_one(job):
    a, eid, rec, start, end = job
    task_dir = eid.split("_")[0]
    calib = find_calib(a.calib_dir, eid)
    row = {"episode_id": eid, "hand": rec["hand"], "calib": calib.name, "start": start, "end": end}
    ep = Path(a.out).expanduser() / eid
    if a.redo or not (ep / "meta.json").exists():
        reserve = rec["purpose"] == "예비"
        cmd = [sys.executable, str(Path(__file__).with_name("process_episode.py")),
               "--raw", str(Path(a.raw).expanduser() / task_dir), "--id", eid, "--task", rec["task"],
               "--collector", rec["collector"], "--hand", rec["hand"], "--tags", rec["case_tags"],
               "--split", "train", "--start", str(start), "--end", str(end + 1),
               "--calib", str(calib), "--sync", a.sync,
               "--cache", str(Path(a.cache).expanduser() / task_dir), "--out", str(Path(a.out).expanduser()),
               "--notes", "박수 사이 임시 구간" + (", 예비 (split 미정)" if reserve else "")]
        if not a.copy_video:
            cmd.append("--link-video")
        if Path(a.wrist_clicks).exists():
            cmd += ["--wrist-clicks", a.wrist_clicks]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            row["error"] = (r.stderr or r.stdout).strip().splitlines()[-1]
            return row
    summarize(row, ep)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="태스크 폴더들의 상위, 예: ~/data/raw")
    ap.add_argument("--cache", required=True, help="태스크별 캐시의 상위, 예: ~/out/seohyeon/kp_cache_v2")
    ap.add_argument("--sync", required=True)
    ap.add_argument("--calib-dir", required=True, help="exo_calib 파일이 있는 폴더 (sync/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", help="생략하면 박수 프레임이 있는 에피소드 전부")
    ap.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--copy-video", action="store_true", help="영상을 복사 (기본은 원본 링크)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--wrist-clicks", default=str(Path(__file__).resolve().parent.parent / "wrist_clicks.json"))
    a = ap.parse_args()

    claps, rec = clap_frames(a.sync), read_records(a.records)
    ids = sorted(a.ids) if a.ids else sorted(claps)
    missing = [i for i in ids if i not in claps]
    if missing:
        raise SystemExit(f"[중단] 박수 프레임 없음: {missing}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(a, i, rec[i], *claps[i]) for i in ids]
    print(f"에피소드 {len(jobs)}개, 병렬 {a.jobs}", flush=True)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPool(a.jobs) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            msg = f"오류 {row['error']}" if row.get("error") else f"T={row['frames']} 손목 {row['tri_wrist']} 이어진 구간 {row['run_len']}"
            print(f"[{k}/{len(jobs)}] {row['episode_id']} {row['calib'][10:-5]:<8} {msg}", flush=True)
    rows.sort(key=lambda r: r["episode_id"])
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    good = [r for r in rows if not r.get("error")]
    print(f"\n끝. 오류 {len(rows) - len(good)}개. 요약: {out / 'summary.csv'}")
    for T in sorted({r["episode_id"][:2] for r in good}):
        g = [r for r in good if r["episode_id"].startswith(T)]
        med = lambda k: float(np.median([float(r[k]) for r in g]))
        print(f"  {T}: {len(g)}개, 손목 3D {med('tri_wrist'):.2f}, 재투영 {med('reproj_wrist'):.1f} px, 보정 후 손목 빈 비율 "
              f"{med('missing_wrist'):.2f}, ego 빈 비율 {med('pinch_missing'):.2f}, 이어진 구간 {med('run_len'):.0f} 프레임")


if __name__ == "__main__":
    main()
