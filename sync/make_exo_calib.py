"""exo 카메라 외부 보정 -> exo_calib.json (3번 episodes/tools/multiview.load_calibs 형식).

입력: 촬영 날 첫 에피소드 전에 찍은 체커보드 영상 (카메라마다 같은 이름 폴더, 예: 1103_check).
출력:
  {"ref_camera": "front",
   "board": {...}, "table_frame": "...",
   "cameras": {"front": {"K": 3x3, "R": 3x3, "t": [3], "dist": [5], ...}, "left": {...}, "right": {...}}}
  규약: 테이블 -> 카메라, pc = R @ p + t (m). K 는 앱이 기록한 값 (calibration.json, 왜곡 없음 pinhole).
  K·R·t 는 영상을 디코딩한 그대로의 픽셀 좌표 기준 (회전 보정 없음, 1920x1440).

테이블 좌표계 (2번 정의)
  - 원점: 체커보드 내부 코너 중, 기준 카메라(front)에서 볼 때 가장 가까운 줄의 가장 왼쪽 코너
  - x: front 에서 볼 때 오른쪽, y: front 에서 멀어지는 쪽 (작업자 쪽), z: 위 (테이블 면에서 위로). 단위 m
  - 보드가 180° 대칭이라 다른 카메라는 "카메라가 작업자 반대편(front 쪽, 보드 중심보다 y 작음)" 인 방향을 고른다.
    -> 결과 그림 (<camera>_axes.jpg) 에서 세 카메라의 x(빨강)·y(초록)가 같은 물리 방향인지 눈으로 확인할 것.

방법: 카메라마다 보드 전체(--pattern)가 잡힌 프레임의 코너를 모아 중앙값 -> solvePnP (반복법) -> 재투영 오차.

사용 예
  python make_exo_calib.py --raw ~/data/raw/2026-11-03 --calib-folder 1103_check \\
      --cameras exo_front=front exo_left=left exo_right=right --ref front --out exo_calib.json --figs ~/out/신서연/calib
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2


def app_K(folder):
    c = json.load(open(Path(folder) / "calibration.json"))["cameras"][0]
    return np.array([[c["focalLengthX"], 0, c["principalPointX"]],
                     [0, c["focalLengthY"], c["principalPointY"]],
                     [0, 0, 1]], float), (c["imageWidth"], c["imageHeight"])


def detect(video, cols, rows, step):
    """보드 전체가 잡힌 프레임의 코너 (N, rows, cols, 2). 프레임마다 순서를 첫 검출에 맞춘다 (180° 뒤집힘 정리)."""
    cap = cv2.VideoCapture(str(video))
    out, frame, i = [], None, 0
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            ok2, c, meta = cv2.findChessboardCornersSBWithMeta(g, (5, 4), flags)
            if ok2 and sorted(meta.shape) == sorted((rows, cols)):
                c = c.reshape(-1, 2)
                # 코너 배열을 (rows, cols) 격자로: 두 가지 나누는 법 중 이웃 간격이 고른 것
                grid = next((g for g in (c.reshape(rows, cols, 2), c.reshape(cols, rows, 2).transpose(1, 0, 2))
                             if _grid_ok(g)), None)
                if grid is not None:
                    if out and np.linalg.norm(grid[0, 0] - out[0][0, 0]) > np.linalg.norm(grid[-1, -1] - out[0][0, 0]):
                        grid = grid[::-1, ::-1]
                    out.append(grid)
                    if frame is None:
                        frame = f.copy()
        i += 1
    return np.array(out), frame


def _grid_ok(grid):
    """이웃 코너 간격이 고른지 (행·열을 잘못 나눴으면 간격이 크게 튐)."""
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    return dx.max() < 3 * np.median(dx) and dy.max() < 3 * np.median(dy)


def board_points(rows, cols, sq):
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([j * sq, i * sq, np.zeros_like(j, float)], -1).reshape(-1, 3).astype(np.float64)


def orient(grid, K, rows, cols, sq, mode, ref_pose=None):
    """grid (rows, cols, 2) 의 네 가지 순서(그대로·좌우·상하·180°) 중 규칙에 맞는 것으로 solvePnP.
    mode="ref": 원점이 화면 아래쪽(가까운 줄)·왼쪽, x 가 화면 오른쪽, z 가 위(카메라 쪽).
    mode="other": z 가 위이고, 카메라 중심 y 가 보드 중심 y 보다 작은 것 (front 와 같은 쪽)."""
    obj = board_points(rows, cols, sq)
    cands = []
    for name, g in (("as_is", grid), ("flip_lr", grid[:, ::-1]), ("flip_ud", grid[::-1, :]), ("rot180", grid[::-1, ::-1])):
        img = g.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).ravel()                      # 카메라 중심 (테이블 좌표)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        cands.append(dict(name=name, R=R, t=tvec.ravel(), C=C, err=err, g=g))
    center_y = (rows - 1) * sq / 2
    up = [c for c in cands if c["C"][2] > 0]           # 오른손 좌표계에서 카메라가 보드 위쪽
    if mode == "ref":
        # 원점 코너가 화면 아래(v 큼)·왼쪽(u 작음), x 축이 화면 오른쪽
        def score(c):
            g = c["g"]
            return (g[0, 0, 1] - g[-1, 0, 1]) + (g[0, -1, 0] - g[0, 0, 0])
        best = max(up, key=score)
    else:
        same_side = [c for c in up if c["C"][1] < center_y]
        best = min(same_side or up, key=lambda c: c["err"].mean())
    return best


def draw_axes(frame, K, R, t, sq, rows, cols, path, name):
    f = frame.copy()
    obj = board_points(rows, cols, sq)
    pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, None)
    for p in pts.reshape(-1, 2):
        cv2.circle(f, tuple(int(v) for v in p), 3, (0, 255, 255), -1)
    L = 0.15
    ax = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], float)
    p, _ = cv2.projectPoints(ax, cv2.Rodrigues(R)[0], t, K, None)
    p = [tuple(int(v) for v in q) for q in p.reshape(-1, 2)]
    for k, col in ((1, (0, 0, 255)), (2, (0, 200, 0)), (3, (255, 0, 0))):
        cv2.arrowedLine(f, p[0], p[k], col, 6, tipLength=0.2)
    cv2.putText(f, f"{name}: x red, y green, z blue", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.imwrite(str(path), cv2.resize(f, (960, 720)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="촬영 날짜 폴더")
    ap.add_argument("--calib-folder", required=True, help="카메라 폴더 안의 체커보드 영상 폴더 이름")
    ap.add_argument("--cameras", nargs="+", required=True, help="폴더=이름 (예: exo_front=front)")
    ap.add_argument("--ref", default="front")
    ap.add_argument("--pattern", default="21x15", help="내부 코너 수 열x행 (기본 21x15)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸 크기 (m)")
    ap.add_argument("--step", type=int, default=5, help="몇 프레임마다 검출할지")
    ap.add_argument("--max-err", type=float, default=2.0, help="평균 재투영 오차 경고 (px)")
    ap.add_argument("--valid-for", default="", help="이 보정값을 쓸 촬영 범위 설명 (카메라를 다시 놓으면 새로 보정)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", help="축 그림 저장 폴더")
    a = ap.parse_args(argv)
    cols, rows = map(int, a.pattern.lower().split("x"))
    raw = Path(a.raw).expanduser()
    cams = dict(s.split("=") for s in a.cameras)
    order = sorted(cams, key=lambda k: cams[k] != a.ref)            # 기준 카메라 먼저
    figs = Path(a.figs).expanduser() if a.figs else None
    if figs:
        figs.mkdir(parents=True, exist_ok=True)

    result = {"ref_camera": a.ref, "valid_for": a.valid_for,
              "board": {"inner_corners": [cols, rows], "square_m": a.square,
                        "source": f"{raw.name}/<camera>/{a.calib_folder}"},
              "table_frame": "원점 = 체커보드 내부 코너 중 front 에서 가장 가까운 줄의 가장 왼쪽. "
                             "x = front 에서 볼 때 오른쪽, y = front 에서 멀어지는 쪽(작업자 쪽), z = 위. 단위 m. "
                             "pc = R @ p + t (테이블 -> 카메라)",
              "cameras": {}}
    for folder in order:
        name = cams[folder]
        vid_dir = raw / folder / a.calib_folder
        K, wh = app_K(vid_dir)
        grids, frame = detect(vid_dir / "data.mov", cols, rows, a.step)
        if len(grids) == 0:
            print(f"{name}: 보드 전체가 잡힌 프레임 없음 -> 건너뜀"); continue
        grid = np.median(grids, axis=0)
        spread = float(np.median(np.linalg.norm(grids - grid, axis=3)))
        best = orient(grid, K, rows, cols, a.square, "ref" if name == a.ref else "other")
        R, t, C, err = best["R"], best["t"], best["C"], best["err"]
        result["cameras"][name] = {
            "K": K.round(4).tolist(), "R": R.round(8).tolist(), "t": t.round(6).tolist(), "dist": [0.0] * 5,
            "image_size": list(wh), "camera_center_table_m": C.round(4).tolist(),
            "reproj_err_px_mean": round(float(err.mean()), 3), "reproj_err_px_max": round(float(err.max()), 3),
            "n_frames": int(len(grids)), "corner_spread_px": round(spread, 3), "corner_order": best["name"]}
        flag = "  <- 경고: 재투영 오차 큼" if err.mean() > a.max_err else ""
        print(f"{name}: 프레임 {len(grids)}장, 재투영 오차 평균 {err.mean():.2f} px / 최대 {err.max():.2f} px, "
              f"카메라 위치 (테이블) {np.round(C, 3)} m{flag}")
        if figs:
            draw_axes(frame, K, R, t.reshape(3, 1), a.square, rows, cols, figs / f"{name}_axes.jpg", name)
    json.dump(result, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{a.out} 저장")


if __name__ == "__main__":
    sys.exit(main())
