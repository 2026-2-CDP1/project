"""exo 영상 한 개에서 2D 키포인트가 제대로 잡히는지 눈으로 확인 (보정 파일 없이 가능).

  python check_2d.py ~/data/raw/2026-11-03/exo_front/T1_E006 --out ~/out/seohyeon/check_2d
  python check_2d.py <영상 또는 녹화 폴더> [--hand right] [--mirrored] [--max-frames 300] [--n-png 8]

출력(--out):
  <이름>_2d.npy      (T,K,2) 픽셀, 못 찾으면 NaN
  <이름>_NNNNN.png   키포인트를 그린 프레임 n-png 장 (손 좌우 라벨이 맞는지 확인)
  표준출력           키포인트별 검출 비율
"""
import argparse
from pathlib import Path
import numpy as np

import extractors as ex
from io_utils import find_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="영상 파일 또는 녹화 폴더(data.mov 가 있는)")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--mirrored", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300, help="앞쪽 이만큼만 처리 (기본 300)")
    ap.add_argument("--n-png", type=int, default=8)
    ap.add_argument("--out", required=True, help="~/project 밖 경로 (예: ~/out/본인이름/check_2d)")
    a = ap.parse_args()

    import cv2
    src = Path(a.video).expanduser()
    video = find_video(src) if src.is_dir() else src
    name = src.name if src.is_dir() else src.stem
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    uv = ex.extract_exo_2d(video, a.hand, a.mirrored, a.max_frames)
    T = len(uv)
    if T == 0:
        raise SystemExit(f"[중단] {video}: 읽은 프레임이 없음")
    np.save(out / f"{name}_2d.npy", uv)

    print(f"{video}: {T} 프레임, hand={a.hand}, mirrored={a.mirrored}")
    print(f"{'키포인트':<12}{'검출 비율':>10}")
    for k, n in enumerate(ex.EXO_NAMES):
        print(f"{n:<12}{np.isfinite(uv[:, k]).all(axis=1).mean():>10.1%}")

    pick = set(np.linspace(0, T - 1, min(a.n_png, T)).astype(int).tolist())
    cap = cv2.VideoCapture(str(video))
    for f in range(T):
        ok, img = cap.read()
        if not ok:
            break
        if f not in pick:
            continue
        for k, n in enumerate(ex.EXO_NAMES):
            if np.isfinite(uv[f, k]).all():
                p = (int(uv[f, k, 0]), int(uv[f, k, 1]))
                cv2.circle(img, p, 6, (0, 255, 0), -1)
                cv2.putText(img, n, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{name}_{f:05d}.png"), img)
    cap.release()
    print("saved ->", out)


if __name__ == "__main__":
    main()
