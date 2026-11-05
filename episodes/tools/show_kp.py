"""처리된 에피소드의 키포인트를 카메라 3대 사진에 그려 한 장으로 모은다 (발표·확인용).

  python show_kp.py ~/out/seohyeon/episodes_clap/T1_E040 --raw ~/data/raw/T1 --out ~/out/seohyeon/kp_images
  python show_kp.py <에피소드> --raw <태스크 폴더> --out <폴더> --frames 300 400 500 --mode 3d --names

--mode 2d  각 카메라에서 추출한 2D (기본). 그 카메라가 놓친 키포인트는 그리지 않는다.
--mode 3d  3D 키포인트를 각 카메라에 다시 그린 것 (keypoints.npy, 보정 후).
--frames 를 빼면 손목 3D 가 이어진 가장 긴 구간에서 고르게 --n 장 고른다. 번호는 front 영상 전체 기준.
출력: <out>/<ID>_<mode>.jpg (행 = 프레임, 열 = 카메라)
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

from hand_math import table_to_pixel
from io_utils import find_calib, find_episode_folder, read_frame_times
from sync_check import nearest_frame, read_frame

BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "index_mcp"), ("wrist", "pinky_mcp"),
         ("index_mcp", "pinky_mcp"), ("wrist", "thumb_tip"), ("index_mcp", "index_tip")]
COLOR = {"shoulder": (255, 160, 0), "elbow": (255, 160, 0), "wrist": (0, 255, 0), "index_mcp": (0, 200, 255),
         "pinky_mcp": (0, 200, 255), "thumb_tip": (60, 60, 255), "index_tip": (60, 60, 255)}


def longest_run(ok):
    d = np.diff(np.r_[0, ok.astype(int), 0])
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if not len(a):
        return 0, len(ok)
    i = int(np.argmax(b - a))
    return int(a[i]), int(b[i])


def draw(img, uv, names, labels):
    pts = {n: uv[k] for k, n in enumerate(names) if np.isfinite(uv[k]).all()}
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 255, 255), 4, cv2.LINE_AA)
    for n, p in pts.items():
        q = tuple(p.astype(int))
        cv2.circle(img, q, 9, COLOR[n], -1, cv2.LINE_AA)
        cv2.circle(img, q, 9, (0, 0, 0), 2, cv2.LINE_AA)
        if labels:
            cv2.putText(img, n, (q[0] + 12, q[1] - 10), 0, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, n, (q[0] + 12, q[1] - 10), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="process_episode.py 결과 폴더")
    ap.add_argument("--raw", required=True, help="태스크 폴더, 예: ~/data/raw/T1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="*", help="front 영상 전체 기준 프레임 번호")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    ap.add_argument("--names", action="store_true", help="키포인트 이름도 적기")
    ap.add_argument("--calib-dir", default=str(Path(__file__).resolve().parents[2] / "sync"), help="--mode 3d 에서 씀")
    ap.add_argument("--width", type=int, default=640, help="칸 너비 (px)")
    a = ap.parse_args()

    ep, raw = Path(a.episode).expanduser(), Path(a.raw).expanduser()
    meta = json.load(open(ep / "meta.json"))
    names, eid, s = json.load(open(ep / "keypoint_names.json")), meta["episode_id"], meta["exo_start_frame"]
    cams = meta["cameras"]
    uv2 = np.load(ep / "keypoints_2d.npy")
    kp = np.load(ep / "keypoints.npy")
    T = len(kp)
    if a.frames:
        frames = a.frames
        bad = [f for f in frames if not s <= f < s + T]
        if bad:
            raise SystemExit(f"[중단] 프레임 {bad} 이 구간 {s}~{s + T - 1} 밖")
    else:
        lo, hi = longest_run(np.isfinite(kp[:, names.index("wrist")]).all(axis=1))
        frames = [s + int(i) for i in np.linspace(lo, hi - 1, a.n)]
    calib = None
    if a.mode == "3d":
        from multiview import load_calibs
        calib, _ = load_calibs(json.load(open(find_calib(a.calib_dir, eid))))

    folder = {c: find_episode_folder(raw, eid, f"exo_{c}") for c in cams}
    times = {c: read_frame_times(folder[c]) for c in cams}
    video = {c: next(ep / f"{'exo' if c == cams[0] else 'exo_' + c}{x}" for x in (".mov", ".mp4")
                     if (ep / f"{'exo' if c == cams[0] else 'exo_' + c}{x}").exists()) for c in cams}
    rows = []
    for f in frames:
        tiles = []
        for i, c in enumerate(cams):
            j = f if i == 0 else nearest_frame(times[c], meta["camera_sync"][c], times[cams[0]][f])[0]
            img = read_frame(video[c], j)
            if img is None:
                raise SystemExit(f"[중단] {video[c]} 프레임 {j} 을 읽지 못함")
            if a.mode == "2d":
                uv = uv2[i, f - s]
            else:
                K, R, t, dist = calib[c]
                ok = np.isfinite(kp[f - s]).all(axis=1)
                uv = np.full((len(names), 2), np.nan)
                if ok.any():
                    uv[ok] = table_to_pixel(kp[f - s][ok], K, R, t, dist)
            draw(img, uv, names, a.names)
            cv2.putText(img, f"{eid} {c} f{f}", (20, 60), 0, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(img, f"{eid} {c} f{f}", (20, 60), 0, 1.6, (0, 255, 255), 3, cv2.LINE_AA)
            h = int(img.shape[0] * a.width / img.shape[1])
            tiles.append(cv2.resize(img, (a.width, h), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(tiles))
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{eid}_{a.mode}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("저장 ->", path, "프레임", frames)


if __name__ == "__main__":
    main()
