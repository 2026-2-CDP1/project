"""테이블 윗면 윤곽을 exo_front 프레임에서 찾아 테이블 좌표(z = 0 면) 위의 타원으로 맞춘다 (정의서 2.6②).

테이블이 타원형 좌식 테이블이라, "테이블 위 영역"을 윤곽 타원 안쪽으로 정의한다. 태스크마다 체커보드 위치가
달라 테이블 좌표에서 테이블 위치가 다르므로 태스크마다 한 번 구해 metrics/table_outline.json에 적는다.

방법
1. exo_front 프레임 하나 (회전 없이 저장된 그대로: ffmpeg -noautorotate)
2. 열마다 테이블 안쪽 기준 행에서 위로(먼 쪽)·아래로(가까운 쪽) 가며 흰 테이블이 끝나는 행을 찾음
3. 그 픽셀을 카메라 광선과 테이블 면(z = 0)의 교점으로 바꿈 (sync/exo_calib_T{n}.json front, 왜곡 없음)
4. Fitzgibbon 직접 최소제곱으로 타원 맞춤, 잔차 큰 점(물체·손·아크릴 판)은 빼고 다시 맞춤
5. 윤곽을 프레임에 다시 그려 눈으로 확인 (--out)

    ffmpeg -noautorotate -ss 1 -i <exo_front data.mov> -frames:v 1 frame.png
    python metrics/table_outline.py --calib sync/exo_calib_T3.json --frame frame.png --out outline.png \
        --far-row 815 --near-row 950 --cols 400 1360

한계: 가까운 쪽 가장자리는 상판 옆면 아래 끝이 잡혀 실제보다 1~2 cm 바깥으로 나올 수 있다. 먼 쪽은 사람 손·팔이
가장자리를 가리는 열이 있어 잔차로 걸러 낸다.
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


def edge_points(gray, far_row, near_row, cols, drop=25):
    """열마다 기준 행에서 위·아래로 가며 테이블 흰색보다 drop 넘게 어두워지는 첫 행 (3행 연속)."""
    H = gray.shape[0]
    pix = []
    for x in cols:
        col = gray[:, x]
        for start, step in ((far_row, -1), (near_row, +1)):
            white = np.median(col[start - 5:start + 5])
            y = start
            while 5 < y < H - 5:
                nxt = col[y - 3:y] if step < 0 else col[y + 1:y + 4]
                if (nxt < white - drop).all():
                    pix.append((x, y))
                    break
                y += step
    return np.array(pix)


def fit_ellipse(P):
    """Fitzgibbon 직접 최소제곱 (항상 타원). 반환 (중심 (2,), 반지름 (2,), 축 행렬 (2,2))."""
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


def radial_resid(P, ctr, radii, axes):
    q = (P - ctr) @ axes
    return np.abs(np.hypot(q[:, 0] / radii[0], q[:, 1] / radii[1]) - 1)


def main():
    ap = argparse.ArgumentParser(description="테이블 윗면 윤곽 → 테이블 좌표 타원")
    ap.add_argument("--calib", required=True, help="sync/exo_calib_T{n}.json")
    ap.add_argument("--frame", required=True, help="exo_front 프레임 (회전 없이 저장된 그대로)")
    ap.add_argument("--out", required=True, help="윤곽을 그린 확인용 그림")
    ap.add_argument("--far-row", type=int, required=True, help="먼 쪽 가장자리 바로 안쪽 행 (테이블 위)")
    ap.add_argument("--near-row", type=int, required=True, help="가까운 쪽 가장자리 바로 안쪽 행")
    ap.add_argument("--cols", type=int, nargs=2, required=True, help="찾을 열 범위 (테이블 왼쪽~오른쪽 끝)")
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()

    K, R, t = load_front(a.calib)
    im = Image.open(a.frame).convert("RGB")
    pix = edge_points(np.asarray(im.convert("L"), float), a.far_row, a.near_row,
                      range(a.cols[0], a.cols[1] + 1, a.step))
    P = np.array([pixel_to_table(u, v, K, R, t)[:2] for u, v in pix])
    keep = (np.abs(P[:, 0] - np.median(P[:, 0])) < 0.8) & (np.abs(P[:, 1] - np.median(P[:, 1])) < 0.6)
    for _ in range(5):
        ctr, radii, axes = fit_ellipse(P[keep])
        r = radial_resid(P, ctr, radii, axes)
        keep = r < max(0.02, 3 * np.median(r[keep]))
    ctr, radii, axes = fit_ellipse(P[keep])
    r = radial_resid(P, ctr, radii, axes)
    ang = float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0])))

    d = ImageDraw.Draw(im)
    for (u, v), k in zip(pix, keep):
        d.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(0, 200, 0) if k else (255, 0, 0), width=3)
    th = np.linspace(0, 2 * np.pi, 200)
    E = ctr + np.c_[radii[0] * np.cos(th), radii[1] * np.sin(th)] @ axes.T
    uv = table_to_pixel(np.c_[E, np.zeros(len(E))], K, R, t)
    d.line([tuple(p) for p in uv] + [tuple(uv[0])], fill=(255, 0, 255), width=3)
    im.save(a.out)

    res = {"center": [round(float(v), 4) for v in ctr], "radii": [round(float(v), 4) for v in radii],
           "angle_deg": round(ang, 2), "points_used": f"{int(keep.sum())}/{len(P)}",
           "resid_median": round(float(np.median(r[keep])), 4),
           "size_m": [round(2 * float(radii.max()), 2), round(2 * float(radii.min()), 2)],
           "y_range_m": [round(float(E[:, 1].min()), 3), round(float(E[:, 1].max()), 3)]}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
