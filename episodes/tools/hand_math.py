"""좌표 변환·시간 맞추기·손목 방향 계산 (역할 가이드 4~6단계)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def load_calib(calib: dict):
    K = np.array(calib["K"], dtype=np.float64)
    R = np.array(calib["R"], dtype=np.float64)
    t = np.array(calib["t"], dtype=np.float64).ravel()
    dist = np.array(calib.get("dist", np.zeros(5)), dtype=np.float64)
    return K, R, t, dist


def pixel_to_table(u, v, depth_img, K, R, t, depth_scale=0.001, win=3, dist=None, color_wh=None):
    """2D 픽셀 + 정렬된 깊이 -> 테이블 좌표(m). 깊이가 없으면 NaN.

    depth_scale: 깊이 단위를 m로 바꾸는 값 (mm 단위면 0.001). 장비에 맞게 지정.
    R, t: 테이블 -> 카메라 변환 (pc = R @ p + t), 가이드 코드와 동일한 규약.
    dist: 렌즈 왜곡 계수. 깊이는 원본(왜곡된) 픽셀에서 읽고, 역투영 전에 픽셀만 왜곡 보정한다.
    color_wh: 컬러 영상 (너비, 높이). 깊이 해상도가 다르면(LiDAR 깊이는 보통 더 작음) 깊이를 읽을 때만
        픽셀 좌표를 깊이 해상도로 환산한다. K, u, v 는 컬러 기준 그대로 쓴다.
    """
    if not (np.isfinite(u) and np.isfinite(v)):
        return np.full(3, np.nan)
    h, w = depth_img.shape[:2]
    ud, vd = u, v
    if color_wh is not None and (color_wh[0] != w or color_wh[1] != h):
        ud, vd = u * w / color_wh[0], v * h / color_wh[1]
    ui, vi = int(round(ud)), int(round(vd))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return np.full(3, np.nan)
    patch = depth_img[max(vi - win, 0):vi + win + 1, max(ui - win, 0):ui + win + 1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return np.full(3, np.nan)
    z = np.median(valid) * depth_scale
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if dist is not None and np.any(dist):
        import cv2
        u, v = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, dist, P=K)[0, 0]
    pc = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
    return R.T @ (pc - t)


def table_to_pixel(pts_table, K, R, t, dist):
    """테이블 좌표(N,3) -> exo 픽셀(N,2). 겹쳐 그리기 확인용 (8단계)."""
    import cv2
    pts = np.asarray(pts_table, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(R)
    px, _ = cv2.projectPoints(pts, rvec, t, K, dist)
    return px.reshape(-1, 2)


def sync_ego_to_exo(t_ego, d_ego, t_exo, s0, s1, off0, off1):
    """ego 거리 시계열을 exo 시간축으로 옮긴다 (5단계). 드리프트는 선형 보간."""
    t_ego = np.asarray(t_ego, dtype=float)
    off = off0 + (off1 - off0) * (t_ego - s0) / (s1 - s0)
    return np.interp(t_exo, t_ego + off, d_ego, left=np.nan, right=np.nan)


def wrist_quat(wrist, index_mcp, pinky_mcp):
    """손목 좌표축 -> 쿼터니언 (w,x,y,z). x: 손목->검지뿌리, z: 손바닥 법선 (6단계)."""
    x = index_mcp - wrist
    v = pinky_mcp - wrist
    nx = np.linalg.norm(x)
    if not np.isfinite(nx) or nx < 1e-9:
        return np.full(4, np.nan)
    x = x / nx
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if not np.isfinite(nz) or nz < 1e-9:
        return np.full(4, np.nan)
    z = z / nz
    y = np.cross(z, x)
    Rm = np.stack([x, y, z], axis=1)
    return np.roll(Rot.from_matrix(Rm).as_quat(), 1)  # scipy(x,y,z,w) -> (w,x,y,z)


def wrist_pose_series(kp, names):
    """keypoints (T,K,3) -> wrist_pose (T,7) = 위치3 + (w,x,y,z). 누락 프레임은 NaN."""
    idx = {n: i for i, n in enumerate(names)}
    w, i, p = kp[:, idx["wrist"]], kp[:, idx["index_mcp"]], kp[:, idx["pinky_mcp"]]
    out = np.full((len(kp), 7), np.nan)
    out[:, :3] = w
    for f in range(len(kp)):
        out[f, 3:] = wrist_quat(w[f], i[f], p[f])
    # q 와 -q 는 같은 회전: 인접 프레임 사이 부호를 연속으로 맞춘다
    prev = None
    for f in range(len(out)):
        q = out[f, 3:]
        if np.isnan(q).any():
            continue
        if prev is not None and np.dot(q, prev) < 0:
            out[f, 3:] = -q
        prev = out[f, 3:]
    return out
