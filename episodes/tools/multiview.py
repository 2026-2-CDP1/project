"""exo 카메라 여러 대(정면·왼쪽·오른쪽)로 3D 키포인트 만들기 (역할 가이드 4단계 '삼각측량').

깊이 카메라 대신 쓴다. 흐름:
  1) 카메라마다 2D 키포인트 (자기 프레임 시각 기준)
  2) align_to_ref: 박수로 구한 시간 차로 각 카메라 시각을 기준 카메라(정면) 시간축으로 옮기고,
     2D 키포인트를 기준 카메라 프레임 시각에 맞춰 시간 보간 (프레임 번호끼리 짝짓지 않는다)
  3) triangulate_series: 프레임·키포인트마다 보이는 카메라(2대 이상)의 광선이 가장 가깝게 만나는 점 (DLT)
     + 재투영 오차 검사. 2대 미만이거나 오차가 크면 NaN.

보정값 규약 (2번 exo_calib.json 과 같음): 테이블 -> 카메라, pc = R @ p + t. K, dist 는 카메라별.
"""
import numpy as np

DEFAULT_TRI = {
    "min_views": 2,          # 이보다 적은 카메라에 보이면 NaN
    "max_reproj_px": 15.0,   # 재투영 오차(카메라별 최대) 허용치. 파일럿을 보고 멘토와 확정 (임시값)
    "max_interp_gap": 1.5,   # 시간 보간: 앞뒤 프레임 간격이 이 배수 x 정상 간격을 넘으면 보간하지 않음
}


def load_calibs(calib: dict):
    """exo_calib.json -> {카메라 이름: (K, R, t, dist)}, 기준 카메라 이름.

    여러 대: {"ref_camera": "front", "cameras": {"front": {K,R,t,dist}, "left": {...}, "right": {...}}}
    한 대(예전 형식): {K,R,t,dist} -> {"front": ...}
    """
    from hand_math import load_calib
    if "cameras" in calib:
        cams = {n: load_calib(c) for n, c in calib["cameras"].items()}
        ref = calib.get("ref_camera", "front")
    else:
        cams, ref = {"front": load_calib(calib)}, "front"
    if ref not in cams:
        raise ValueError(f"ref_camera '{ref}' 가 cameras {list(cams)} 에 없음")
    return cams, ref


def cam_time_to_ref(t_cam, s0, s1, off0, off1):
    """카메라 자기 시각 -> 기준 카메라 시각. off = 기준 박수 시각 - 이 카메라 박수 시각 (드리프트 선형).
    s0, s1: 이 카메라 기준 시작·끝 박수 시각. 3번 가이드 5단계 ego 식과 같은 형태."""
    t_cam = np.asarray(t_cam, dtype=float)
    if s1 == s0:
        return t_cam + off0
    return t_cam + off0 + (off1 - off0) * (t_cam - s0) / (s1 - s0)


def interp_track(t_src, x_src, t_dst, max_gap_s):
    """(N,...) 시계열을 t_dst 시각으로 선형 보간. 앞뒤 샘플 중 하나라도 NaN 이거나
    둘 사이 간격이 max_gap_s 보다 크거나 범위 밖이면 NaN."""
    t_src = np.asarray(t_src, float); x_src = np.asarray(x_src, float); t_dst = np.asarray(t_dst, float)
    out = np.full((len(t_dst),) + x_src.shape[1:], np.nan)
    j = np.searchsorted(t_src, t_dst, side="right")  # t_src[j-1] <= t < t_src[j]
    for i in range(len(t_dst)):
        a = j[i] - 1
        if a < 0:
            continue                       # 첫 샘플보다 앞
        if t_src[a] == t_dst[i]:
            out[i] = x_src[a]
            continue
        b = a + 1
        if b >= len(t_src):
            continue                       # 마지막 샘플보다 뒤
        dt = t_src[b] - t_src[a]
        if dt <= 0 or dt > max_gap_s:
            continue
        w = (t_dst[i] - t_src[a]) / dt
        out[i] = (1 - w) * x_src[a] + w * x_src[b]   # 한쪽이 NaN 이면 결과도 NaN
    return out


def align_to_ref(t_cam, uv_cam, t_ref, sync, max_gap_factor=DEFAULT_TRI["max_interp_gap"]):
    """카메라 2D 키포인트 (Tc,K,2) -> 기준 카메라 시각 t_ref 에 맞춘 (T,K,2)."""
    t_on_ref = cam_time_to_ref(t_cam, **sync)
    step = np.median(np.diff(t_cam))
    return interp_track(t_on_ref, uv_cam, t_ref, max_gap_factor * step)


def _normalized(uv, K, dist):
    """왜곡 보정한 정규 좌표 (N,2)."""
    import cv2
    pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist).reshape(-1, 2)


def _dlt(xn, Rs, ts):
    """정규 좌표 xn[i]=(x,y), 카메라 i 의 [R|t] -> 3D 점 (최소제곱)."""
    A = []
    for (x, y), R, t in zip(xn, Rs, ts):
        P = np.hstack([R, t.reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _reproj(X, uv, cams):
    """카메라별 재투영 오차 (px)."""
    from hand_math import table_to_pixel
    return np.array([np.linalg.norm(table_to_pixel(X[None], K, R, t, d)[0] - p) for p, (K, R, t, d) in zip(uv, cams)])


def _in_front(X, cams):
    return all((R @ X + t)[2] > 0 for (K, R, t, d) in cams)


def triangulate_point(uv_list, cam_list, max_reproj_px):
    """보이는 카메라들의 2D 점 -> (X, 최대 재투영 오차, 사용한 카메라 수).
    3대 이상에서 오차가 크면 한 대씩 빼 보고 가장 좋은 2대 이상 조합을 쓴다 (한 카메라의 오검출 대비)."""
    n = len(uv_list)
    xn = [_normalized(p[None], K, d)[0] for p, (K, R, t, d) in zip(uv_list, cam_list)]

    def solve(idx):
        X = _dlt([xn[i] for i in idx], [cam_list[i][1] for i in idx], [cam_list[i][2] for i in idx])
        cams = [cam_list[i] for i in idx]
        if not _in_front(X, cams):
            return None, np.inf
        return X, _reproj(X, [uv_list[i] for i in idx], cams).max()

    best_X, best_e, best_n = None, np.inf, 0
    X, e = solve(list(range(n)))
    if X is not None:
        best_X, best_e, best_n = X, e, n
    if n >= 3 and best_e > max_reproj_px:
        for drop in range(n):
            idx = [i for i in range(n) if i != drop]
            X, e = solve(idx)
            if X is not None and e < best_e:
                best_X, best_e, best_n = X, e, n - 1
    if best_X is None or best_e > max_reproj_px:
        return np.full(3, np.nan), best_e, 0
    return best_X, best_e, best_n


def triangulate_series(uv_views, cam_list, settings=None):
    """uv_views: (C,T,K,2) 기준 시간축에 맞춘 카메라별 2D, NaN = 안 보임.
    반환: X (T,K,3), 재투영 오차 (T,K) px, 사용 카메라 수 (T,K)."""
    st = {**DEFAULT_TRI, **(settings or {})}
    C, T, K, _ = uv_views.shape
    X = np.full((T, K, 3), np.nan); err = np.full((T, K), np.nan); nv = np.zeros((T, K), int)
    for f in range(T):
        for k in range(K):
            vis = [c for c in range(C) if np.isfinite(uv_views[c, f, k]).all()]
            if len(vis) < st["min_views"]:
                continue
            x, e, n = triangulate_point([uv_views[c, f, k] for c in vis], [cam_list[c] for c in vis], st["max_reproj_px"])
            X[f, k], nv[f, k] = x, n
            if np.isfinite(e):
                err[f, k] = e
    return X, err, nv


def tri_report(err, nv, min_views=2):
    """품질 표용 요약 (키포인트별)."""
    ok = nv >= min_views
    med = [float(np.nanmedian(err[ok[:, k], k])) if ok[:, k].any() else None for k in range(err.shape[1])]
    return {
        "tri_ok_ratio": ok.mean(axis=0).round(4).tolist(),
        "tri_reproj_median_px": [None if m is None else round(m, 2) for m in med],
        "tri_views_mean": nv.mean(axis=0).round(2).tolist(),
    }
