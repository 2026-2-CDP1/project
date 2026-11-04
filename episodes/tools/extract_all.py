"""태스크 하나의 모든 에피소드·카메라에서 2D 키포인트(exo) / 엄지-검지 거리(ego)를 영상 전체로 미리 뽑아 둔다.

MediaPipe 가 제일 오래 걸리는 단계라 라벨(--start/--end) 없이 먼저 돌려 두고,
process_episode.py --cache 로 읽으면 구간 자르기·삼각측량·보정만 해서 금방 끝난다.

  # 계산 노드에서 (--mem 꼭 적기)
  sbatch -p p02 -c 32 --mem=48G -o ~/out/seohyeon/extract_T1.log \\
      --wrap "cd ~/project/episodes/tools && python extract_all.py --raw ~/data/raw/T1 --out ~/out/seohyeon/kp_cache/T1 --jobs 32"

출력(--out):
  <camera>/<ID>_<hand>.npy   exo: (T,K,2) 픽셀 NaN=못 찾음 / ego: (T,) 거리
  summary.csv                작업마다 프레임 수·검출 비율·오류 (다시 돌리면 이어 붙음)
이미 있는 파일은 건너뛴다 (중간에 끊겨도 같은 명령으로 이어서). 다시 뽑으려면 --redo.
"""
import argparse, csv, os, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from io_utils import find_episode_folder, find_video, read_frame_times, kp_cache_path

CAMS = ["exo_front", "exo_left", "exo_right", "ego"]
FIELDS = ["camera", "episode_id", "frames", "times", "wrist", "shoulder", "pinch", "sec", "error"]


def run_one(job):
    raw, out, cam, eid, hand = job
    import extractors as ex
    row = {"camera": cam, "episode_id": eid}
    t0 = time.time()
    try:
        folder = find_episode_folder(raw, eid, cam)
        n_t = len(read_frame_times(folder))
        if cam == "ego":
            x = ex.extract_ego_pinch(find_video(folder), hand)
            row["pinch"] = f"{np.isfinite(x).mean():.3f}"
        else:
            x = ex.extract_exo_2d(find_video(folder), hand)
            ok = np.isfinite(x).all(axis=2).mean(axis=0)
            row["wrist"] = f"{ok[ex.EXO_NAMES.index('wrist')]:.3f}"
            row["shoulder"] = f"{ok[ex.EXO_NAMES.index('shoulder')]:.3f}"
        row.update(frames=len(x), times=n_t)
        if len(x) not in (n_t, n_t + 1):           # +1 은 process_episode 가 마지막 프레임을 버려 맞춘다
            row["error"] = "영상 프레임 수 != data.jsonl 시각 수"
        p = kp_cache_path(out, cam, eid, hand)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npy")
        np.save(tmp, x)
        tmp.replace(p)                                  # 끊겨도 반쪽 파일이 남지 않게
    except Exception as e:                              # 한 개 실패로 전체가 멈추지 않게 기록만
        row["error"] = f"{type(e).__name__}: {e}"
    row["sec"] = f"{time.time() - t0:.0f}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="태스크 폴더, 예: ~/data/raw/T1")
    ap.add_argument("--out", required=True, help="~/project 밖, 예: ~/out/seohyeon/kp_cache/T1")
    ap.add_argument("--ids", nargs="*", help="생략하면 exo_front 아래 *_E* 전부")
    ap.add_argument("--cams", nargs="+", default=CAMS)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    raw, out = Path(a.raw).expanduser(), Path(a.out).expanduser()
    ids = a.ids or sorted(p.name for p in (raw / "exo_front").glob("*_E*") if p.is_dir())
    jobs = [(raw, out, c, i, a.hand) for i in ids for c in a.cams
            if a.redo or not kp_cache_path(out, c, i, a.hand).exists()]
    print(f"에피소드 {len(ids)}개 × 카메라 {len(a.cams)}대 중 할 일 {len(jobs)}개, 병렬 {a.jobs}", flush=True)
    if not jobs:
        return

    out.mkdir(parents=True, exist_ok=True)
    summ = out / "summary.csv"
    new = not summ.exists()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    n_err = 0
    with open(summ, "a", newline="") as f, get_context("spawn").Pool(a.jobs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            w.writerow(row); f.flush()
            n_err += bool(row.get("error"))
            rate = (f"pinch {row['pinch']}" if "pinch" in row else
                    f"wrist {row['wrist']} shoulder {row['shoulder']}" if "wrist" in row else "")
            print(f"[{k}/{len(jobs)}] {row['episode_id']} {row['camera']:<9} {rate}  {row['sec']}s"
                  f"{'  <- ' + row['error'] if row.get('error') else ''}", flush=True)
    print(f"끝. 오류 {n_err}개. 요약: {summ}")


if __name__ == "__main__":
    main()
