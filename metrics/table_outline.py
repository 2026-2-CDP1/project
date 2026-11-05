"""테이블 윗면 윤곽을 exo_front 프레임에서 찾아 테이블 좌표(z = 0 면) 위의 다각형으로 만든다 (정의서 2.6②).

테이블이 앞뒤가 평평하고 양끝이 둥근 좌식 테이블이라, "테이블 위 영역"을 윗면 윤곽 다각형 안쪽으로 정의한다.
태스크마다 체커보드 위치가 달라 테이블 좌표에서 테이블 위치가 다르므로 태스크마다 한 번 구해
metrics/table_outline.json에 적는다.

방법
1. exo_front 프레임 하나 (회전 없이 저장된 그대로: ffmpeg -noautorotate)
2. 가장자리 점 찾기 — 흰 테이블이 끝나는 곳 (테이블 흰색보다 25 넘게 어두워지는 첫 픽셀, 3픽셀 연속)
   - 세로: 열마다 기준 행에서 위로(먼 쪽)·아래로(가까운 쪽)
   - 가로: 행마다 양끝 기준 열에서 왼쪽·오른쪽으로 (거의 세로인 양끝 가장자리)
   - 체커보드가 테이블을 덮은 프레임은 --far-segments로 보드 위 흰 띠 행을 열 범위마다 준다
3. 그 픽셀을 카메라 광선과 테이블 면(z = 0)의 교점으로 바꿈 (sync/exo_calib_T{n}.json front, 왜곡 없음)
4. 이상한 점(물체·손·아크릴 판·배경) 거르기: 점들을 중심에서 본 각도 순으로 놓고, 각도로 이웃한 점들의
   반지름 중앙값에서 --tol(기본 3 cm) 넘게 벗어난 점을 뺌 (모양 가정 없이 "가장자리는 부드럽게 이어진다"만 씀)
5. 남은 점의 볼록 껍질 = 윤곽 다각형. 확인용 그림에 다시 그림 (--out)

    ffmpeg -noautorotate -ss 1 -i <exo_front data.mov> -frames:v 1 frame.png
    python metrics/table_outline.py --calib sync/exo_calib_T3.json --frame frame.png --out outline.png \\
        --far-row 815 --near-row 950 --cols 400 1360 --rows 760 1100 --side-starts 430 1330

한계: 가까운 쪽 가장자리는 상판 옆면 아래 끝이 잡혀 실제보다 1~2 cm 바깥으로 나올 수 있다. 볼록 껍질이라
점 사이는 직선으로 이어 실제 둥근 가장자리보다 조금 안쪽이 된다 (점 간격 20픽셀 ≈ 1~2 cm).
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw


def load_front(calib):
    c = json.load(open(calib, encoding="utf-8"))["cameras"]["front"]
    K, R, t = np.array(c["K"]), np.array(c["R"]), np.array(c["t"]).reshape(3)
    if np.any(np.array(c.get("dist", [0]), float) != 0):
        raise SystemExit("왜곡 계수가 0이 아님 — 이 도구는 왜곡 없는 보정만 다룸")
    return K, R, t


def pixel_to_table(u, v, K, R, t):
    C = -R.T @ t
    d = R.T @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    return C + (-C[2] / d[2]) * d


def table_to_pixel(P, K, R, t):
    Pc = P @ R.T + t
    return np.c_[K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]]


def first_drop(line, start, step, drop=25, table_white=None, white_tol=40):
    """line[start]에서 step 방향으로 가며 시작점 밝기보다 drop 넘게 어두운 픽셀이 3개 연속되는 첫 위치.

    table_white가 있으면 시작점이 테이블 흰색 − white_tol보다 밝을 때만 찾는다 (테이블 밖에서 시작한 탐색은 버림. 테이블 210~245, 바닥·배경 100~165).
    """
    white = np.median(line[max(0, start - 5):start + 5])
    if table_white is not None and white < table_white - white_tol:
        return None
    i = start
    while 5 < i < len(line) - 5:
        nxt = line[i - 3:i] if step < 0 else line[i + 1:i + 4]
        if (nxt < white - drop).all():
            return i
        i += step
    return None


def edge_points(gray, far_row, near_row, cols, rows=(), side_starts=None, far_segments=()):
    """세로(열마다 위·아래)·가로(행마다 왼쪽·오른쪽) 탐색으로 가장자리 픽셀 (u, v) 목록.

    테이블 흰색 = 가까운 쪽 기준 행(near_row)의 가운데 열들 밝기 중앙값. 시작점이 이 밝기가 아니면 그 탐색은 버린다.
    """
    cols = list(cols)
    mid = cols[len(cols) // 4: 3 * len(cols) // 4]
    table_white = float(np.median(gray[near_row, mid]))
    pix = []
    for x in cols:
        fr = next((r for x0, x1, r in far_segments if x0 <= x < x1), far_row)
        for start, step in ((fr, -1), (near_row, +1)):
            y = first_drop(gray[:, x], start, step, table_white=table_white)
            if y is not None:
                pix.append((x, y))
    if side_starts:
        for y in rows:
            for start, step in ((side_starts[0], -1), (side_starts[1], +1)):
                x = first_drop(gray[y, :], start, step, table_white=table_white)
                if x is not None:
                    pix.append((x, y))
    return np.array(pix)


def fit_ellipse(P):
    """Fitzgibbon 직접 최소제곱 (항상 타원). 반환 (중심 (2,), 반지름 (2,), 축 행렬 (2,2)). 이상한 점 거르기용."""
    m = P.mean(axis=0)
    x, y = (P - m).T
    D1, D2 = np.c_[x * x, x * y, y * y], np.c_[x, y, np.ones_like(x)]
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    _, V = np.linalg.eig(M)
    a1 = np.real(V[:, np.argmax(4 * V[0] * V[2] - V[1] ** 2 > 0)])
    a, b, c, d, e, f = (*a1, *(T @ a1))
    Q = np.array([[a, b / 2], [b / 2, c]])
    ctr = np.linalg.solve(2 * Q, -np.array([d, e]))
    k = -(f + 0.5 * np.array([d, e]) @ ctr)
    ev, axes = np.linalg.eigh(Q / k)
    return ctr + m, 1 / np.sqrt(ev), axes


def local_outliers(P, keep, tol, k=2):
    """중심에서 본 각도 순으로 앞뒤 k개 이웃의 반지름 중앙값과 tol 넘게 다른 점 → 거짓 (keep 갱신)."""
    idx = np.flatnonzero(keep)
    c = np.median(P[idx], axis=0)
    ang = np.arctan2(P[idx, 1] - c[1], P[idx, 0] - c[0])
    order = idx[np.argsort(ang)]
    r = np.hypot(*(P[order] - c).T)
    n = len(order)
    new = keep.copy()
    for j in range(n):
        nb = [r[(j + o) % n] for o in range(-k, k + 1) if o != 0]
        if abs(r[j] - np.median(nb)) > tol:
            new[order[j]] = False
    return new


def radial_resid(P, ctr, radii, axes):
    q = (P - ctr) @ axes
    return np.abs(np.hypot(q[:, 0] / radii[0], q[:, 1] / radii[1]) - 1)


def convex_hull(P):
    """2D 볼록 껍질 (monotone chain), 반시계 방향 꼭짓점."""
    P = sorted(map(tuple, P))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(V):
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    ap = argparse.ArgumentParser(description="테이블 윗면 윤곽 → 테이블 좌표 다각형")
    ap.add_argument("--calib", required=True, help="sync/exo_calib_T{n}.json")
    ap.add_argument("--frame", required=True, help="exo_front 프레임 (회전 없이 저장된 그대로)")
    ap.add_argument("--out", required=True, help="윤곽을 그린 확인용 그림")
    ap.add_argument("--far-row", type=int, required=True, help="먼 쪽 가장자리 바로 안쪽 행 (테이블 위)")
    ap.add_argument("--near-row", type=int, required=True, help="가까운 쪽 가장자리 바로 안쪽 행")
    ap.add_argument("--cols", type=int, nargs=2, required=True, help="세로 탐색할 열 범위")
    ap.add_argument("--rows", type=int, nargs=2, help="가로 탐색할 행 범위 (테이블 양끝)")
    ap.add_argument("--side-starts", type=int, nargs=2, help="가로 탐색 시작 열 (왼쪽 끝 안쪽, 오른쪽 끝 안쪽)")
    ap.add_argument("--far-segments", nargs="*", default=[],
                    help="열 범위별 먼 쪽 기준 행 'x0:x1:row' (체커보드가 테이블 가운데를 덮은 프레임: 보드 위 흰 띠 행)")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.03, help="이웃 점 반지름 중앙값에서 이보다 벗어나면 이상한 점 (m)")
    a = ap.parse_args()

    K, R, t = load_front(a.calib)
    im = Image.open(a.frame).convert("RGB")
    gray = np.asarray(im.convert("L"), float)
    rows = range(a.rows[0], a.rows[1] + 1, a.step) if a.rows else ()
    pix = edge_points(gray, a.far_row, a.near_row, range(a.cols[0], a.cols[1] + 1, a.step), rows, a.side_starts,
                      [tuple(int(v) for v in seg.split(":")) for seg in a.far_segments])
    P = np.array([pixel_to_table(u, v, K, R, t)[:2] for u, v in pix])
    keep = (np.abs(P[:, 0] - np.median(P[:, 0])) < 0.8) & (np.abs(P[:, 1] - np.median(P[:, 1])) < 0.6)
    for _ in range(3):
        keep = local_outliers(P, keep, a.tol)
    hull = convex_hull(P[keep])

    d = ImageDraw.Draw(im)
    for (u, v), k in zip(pix, keep):
        d.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(0, 200, 0) if k else (255, 0, 0), width=3)
    uv = table_to_pixel(np.c_[hull, np.zeros(len(hull))], K, R, t)
    d.line([tuple(p) for p in uv] + [tuple(uv[0])], fill=(255, 0, 255), width=3)
    im.save(a.out)

    res = {"polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in hull],
           "points_used": f"{int(keep.sum())}/{len(P)}",
           "size_m": [round(float(np.ptp(hull[:, 0])), 3), round(float(np.ptp(hull[:, 1])), 3)],
           "area_m2": round(polygon_area(hull), 4),
           "x_range_m": [round(float(hull[:, 0].min()), 3), round(float(hull[:, 0].max()), 3)],
           "y_range_m": [round(float(hull[:, 1].min()), 3), round(float(hull[:, 1].max()), 3)]}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
