"""sync.csv 의 시간 차를 적용해 같은 기준 시각의 프레임을 카메라별로 나란히 놓고 눈으로 확인한다.

시작·끝 박수 순간에 모든 카메라에서 손이 맞닿는 장면이어야 싱크가 맞는 것.
  python sync_check.py --raw ~/data/raw/2026-11-03 --id T1_E006 --sync ../../sync/sync.csv \
      --out ~/out/seohyeon/sync_check

출력: <id>_start.png, <id>_end.png  (열 = 카메라, 행 = 기준 시각 -0.10 / 0 / +0.10 s. 가운데 줄이 박수)
표준출력: 박수의 기준 시각이 카메라마다 얼마나 어긋나는지 (s0 + offset_start 이 모든 카메라에서 같아야 함)

기준 카메라(exo_front)는 sync.csv 에 행이 없다 (시간 차 0). 카메라 c 의 시각 t 를 기준 시각으로:
  t + off0 + drift * (t - s0) / (s1 - s0)        (io_utils.read_sync_row / hand_math.sync_ego_to_exo 와 같은 식)
"""
import argparse, csv
from pathlib import Path
import numpy as np

from io_utils import find_episode_folder, find_video, read_frame_times

REF = "exo_front"


def to_ref_time(t, row):
    """카메라 시각 t -> 기준 시각. row=None 이면 기준 카메라 (그대로)."""
    t = np.asarray(t, dtype=float)
    if row is None:
        return t
    s0, s1, off0, off1 = row["s0"], row["s1"], row["off0"], row["off1"]
    return t + off0 + (off1 - off0) * (t - s0) / (s1 - s0)


def load_rows(sync_csv, episode_id):
    """{'left': {...}, 'right': {...}, 'ego': {...}} (usable=1 인 행만)."""
    out = {}
    with open(sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["episode_id"] != episode_id or r.get("usable", "1").strip() != "1":
                continue
            out[r["camera"].strip()] = {k: float(r[c]) for k, c in
                                        (("off0", "offset_start_s"), ("off1", "offset_end_s"), ("s0", "s0"), ("s1", "s1"))}
    return out


def nearest_frame(t_cam_all, row, t_ref):
    """기준 시각 t_ref 에 가장 가까운 이 카메라의 프레임 번호와 시간 오차(s)."""
    r = to_ref_time(t_cam_all, row)
    i = int(np.argmin(np.abs(r - t_ref)))
    return i, float(r[i] - t_ref)


def read_frame(video, idx):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--sync", required=True)
    ap.add_argument("--out", required=True, help="~/project 밖 경로")
    ap.add_argument("--cams", nargs="+", default=[REF, "exo_left", "exo_right", "ego"])
    ap.add_argument("--steps", type=float, nargs="+", default=[-0.10, 0.0, 0.10], help="기준 시각 앞뒤 (s)")
    ap.add_argument("--width", type=int, default=480, help="칸 하나의 너비(px)")
    a = ap.parse_args()

    import cv2
    rows = load_rows(a.sync, a.id)
    if not rows:
        raise SystemExit(f"[중단] {a.sync} 에 {a.id} 의 usable=1 행이 없음")
    names = [c for c in a.cams if c == REF or c.replace("exo_", "") in rows or c in rows]
    key = lambda c: None if c == REF else rows[c.replace("exo_", "")] if c.replace("exo_", "") in rows else rows[c]
    skipped = [c for c in a.cams if c not in names]
    if skipped:
        print("건너뜀 (sync.csv 에 행 없음):", skipped)

    # 박수의 기준 시각: 카메라마다 s0 + off0 (시작), s1 + off1 (끝). 모두 같아야 한다.
    ref_claps = {"start": {}, "end": {}}
    for c in names:
        r = key(c)
        if r is not None:
            ref_claps["start"][c] = r["s0"] + r["off0"]
            ref_claps["end"][c] = r["s1"] + r["off1"]
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    times, videos = {}, {}
    for c in names:
        folder = find_episode_folder(a.raw, a.id, c)
        times[c], videos[c] = read_frame_times(folder), find_video(folder)

    for which in ("start", "end"):
        vals = ref_claps[which]
        if not vals:
            continue
        t_ref = float(np.median(list(vals.values())))
        print(f"\n[{which} 박수] 기준 시각 {t_ref:.4f} s")
        for c, v in vals.items():
            print(f"   {c:<10} 박수 기준 시각 {v:.4f} s  (중앙값과 차이 {v - t_ref:+.4f} s)")
        grid = []
        for step in a.steps:
            line = []
            for c in names:
                i, err = nearest_frame(times[c], key(c), t_ref + step)
                img = read_frame(videos[c], i)
                if img is None:
                    img = np.zeros((360, 480, 3), np.uint8)
                h = int(img.shape[0] * a.width / img.shape[1])
                img = cv2.resize(img, (a.width, h))
                cv2.putText(img, f"{c} f{i} ({err*1000:+.0f}ms)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if step == 0.0:
                    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 4)
                line.append(img)
            hmax = max(i.shape[0] for i in line)
            line = [cv2.copyMakeBorder(i, 0, hmax - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in line]
            grid.append(np.hstack(line))
        path = out / f"{a.id}_{which}.png"
        cv2.imwrite(str(path), np.vstack(grid))
        print("saved ->", path)


if __name__ == "__main__":
    main()
