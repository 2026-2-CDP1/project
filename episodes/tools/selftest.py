"""합성 데이터로 핵심 함수 점검: python selftest.py"""
import numpy as np
from hand_math import wrist_quat, wrist_pose_series, pixel_to_table, sync_ego_to_exo, table_to_pixel
from quality import correct, fill_short_gaps

# 1) 쿼터니언: x=(1,0,0), 손바닥 법선 z=(0,0,1) -> 항등
q = wrist_quat(np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]))
assert np.allclose(q, [1, 0, 0, 0]), q

# 2) 투영 <-> 역투영 왕복
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.]])
th = 0.3; R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]); t = np.array([0.1, -0.2, 1.5])
p = np.array([0.2, 0.1, 0.05])
u, v = table_to_pixel(p[None], K, R, t, np.zeros(5))[0]
zc = (R @ p + t)[2]
depth = np.full((480, 640), int(zc * 1000), dtype=np.uint16)
assert np.allclose(pixel_to_table(u, v, depth, K, R, t), p, atol=2e-3)

# 3) 싱크: ego 시각 t 의 값이 t+0.5 에 대응 -> exo 시각 x 에서 값은 x-0.5 (드리프트 없음)
t_exo = np.arange(0, 10, 1 / 30); t_ego = np.arange(0, 10, 1 / 60)
out = sync_ego_to_exo(t_ego, t_ego, t_exo, 0, 10, 0.5, 0.5)
m = (t_exo > 0.6) & (t_exo < 9.4)
assert np.allclose(out[m], t_exo[m] - 0.5, atol=1e-6)

# 4) 보정: 짧은 누락은 채우고, 긴 누락은 NaN 유지, 튐은 제거
T = 120; kp = np.zeros((T, 2, 3)); kp[:, :, 0] = np.linspace(0, 0.5, T)[:, None]
kp[10:13] = np.nan; kp[50:70] = np.nan; kp[90] += 3.0
c, rep, _ = correct(kp, 30)
assert not np.isnan(c[11]).any()
assert np.isnan(c[60]).all()
assert abs(c[90, 0, 0] - 0.5 * 90 / 119) < 0.02  # 튄 값(+3)이 사라지고 보간됨

# 5) wrist_pose 부호 연속성
names = ["wrist", "index_mcp", "pinky_mcp"]
k3 = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]]), (5, 1, 1))
wp = wrist_pose_series(k3, names); assert wp.shape == (5, 7)
# 6) 렌즈 왜곡: 가장자리 점도 복원 (k1=-0.1)
dist = np.array([-0.1, 0, 0, 0, 0.])
for p in (np.array([0.45, 0.3, 0.0]), np.array([0.0, 0.0, 0.0])):
    u, v = table_to_pixel(p[None], K, R, t, dist)[0]
    depth = np.full((480, 640), int((R @ p + t)[2] * 1000), dtype=np.uint16)
    assert np.allclose(pixel_to_table(u, v, depth, K, R, t, dist=dist), p, atol=3e-3)
    assert np.linalg.norm(pixel_to_table(u, v, depth, K, R, t) - p) > 3e-3 or abs(p).sum() == 0  # 왜곡 무시하면 틀림

# 7) sync.csv: 실제 열 이름, 없는 에피소드/usable=0 은 에러
import tempfile, os
from io_utils import read_sync_row, SyncError
f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
f.write("episode_id,offset_start_s,offset_end_s,drift_s,usable\nT1_E001,0.5,0.5,0,1\nT1_E002,0.5,0.52,0.02,1\nT1_E003,0.5,0.5,0,0\n"); f.close()
assert read_sync_row(f.name, "T1_E001")["off0"] == 0.5
for args in (("T1_E999",), ("T1_E003",), ("T1_E002",)):  # 없음 / usable=0 / 드리프트인데 s0,s1 없음
    try: read_sync_row(f.name, *args); raise AssertionError(args)
    except SyncError: pass
assert read_sync_row(f.name, "T1_E002", 1.0, 60.0)["s1"] == 60.0
os.unlink(f.name)

# 8) 라벨: 전체 프레임 기준, 순서, 연속성
from labels import check
meta = {"episode_id": "T1_E001", "exo_start_frame": 100, "exo_end_frame": 300}
lab = {"episode_id": "T1_E001", "frame_ref": "exo", "exo_start_frame": 100, "exo_end_frame": 300,
       "segments": [{"name": "approach", "start": 112, "end": 157}, {"name": "grasp", "start": 158, "end": 170}], "grasp_frame": 164}
assert check(lab, meta) == []
lab["segments"].reverse(); assert check(lab, meta)
lab["segments"].reverse(); lab["segments"][1]["start"] = 160; assert check(lab, meta)
print("selftest OK")

# 9) 깊이 해상도가 컬러와 다를 때 (컬러 640x480, 깊이 320x240)
p = np.array([0.2, 0.1, 0.05]); u, v = table_to_pixel(p[None], K, R, t, np.zeros(5))[0]
d_small = np.full((240, 320), int((R @ p + t)[2] * 1000), dtype=np.uint16)
assert np.allclose(pixel_to_table(u, v, d_small, K, R, t, color_wh=(640, 480)), p, atol=2e-3)
# 10) 라벨: grasp_frame 이 grasp 구간 밖이면 에러
lab["segments"][1]["start"] = 158; lab["segments"].sort(key=lambda s: s["start"]); lab["grasp_frame"] = 190
assert check(lab, meta)
print("selftest 9~10 OK")

# ---------- 11~14) exo 3대 삼각측량 ----------
import json, tempfile, os
from multiview import triangulate_series, align_to_ref, cam_time_to_ref, load_calibs
from io_utils import read_jsonl_frame_times, read_sync_row, SyncError
from scipy.spatial.transform import Rotation as Rot


def look_at(cam_pos, target=np.array([0.3, 0.0, 0.05])):
    """테이블 좌표의 카메라 위치 -> (R, t), pc = R p + t. 카메라 z 축이 target 을 향함."""
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam_pos

Kc = np.array([[1400, 0, 960], [0, 1400, 720], [0, 0, 1.]])
dists = [np.array([0.08, -0.15, 0, 0, 0]), np.array([-0.05, 0.02, 0, 0, 0]), np.zeros(5)]
cams = []
for pos, dd in zip([np.array([0.3, -0.9, 0.6]), np.array([-0.5, -0.5, 0.6]), np.array([1.1, -0.5, 0.6])], dists):
    R_, t_ = look_at(pos); cams.append((Kc, R_, t_, dd))

# 11) 정지 점 여러 개: 3대 모두 보이면 복원, 1대만 보이면 NaN, 1대 오검출(+80px)은 빼고 복원
P = np.array([[0.3, 0.0, 0.05], [0.55, 0.2, 0.15], [0.05, -0.15, 0.3]])
uv = np.stack([table_to_pixel(P, *c[:3], c[3]) for c in cams])[:, None]       # (C,1,K,2)
X, err, nv = triangulate_series(uv, cams)
assert np.abs(X[0] - P).max() < 1e-4 and (nv == 3).all(), (X, err)
uv1 = uv.copy(); uv1[1:, 0, 0] = np.nan                                       # 0번 점: 정면만 보임
uv1[2, 0, 1] += 80                                                            # 1번 점: 오른쪽 카메라 오검출
X1, err1, nv1 = triangulate_series(uv1, cams)
assert np.isnan(X1[0, 0]).all() and nv1[0, 0] == 0
assert np.abs(X1[0, 1] - P[1]).max() < 1e-4 and nv1[0, 1] == 2, (X1[0, 1], err1[0, 1])

# 12) 움직이는 손목 (최대 약 1.3 m/s), 카메라마다 시계가 다르고 드리프트 있음
#     박수 시간 차로 기준 시간축에 보간하면 mm 수준, 프레임 번호로 짝지으면 cm 수준
def traj(tt):  # 테이블 좌표 (m)
    return np.stack([0.3 + 0.25 * np.sin(2 * np.pi * 0.8 * tt), 0.1 * np.cos(2 * np.pi * 0.8 * tt), 0.1 + 0.05 * tt / 10], -1)

fps = 30.0
t_ref = np.arange(0, 10, 1 / fps)                                     # 정면(기준) 시계 = 실제 시각
clocks = {"left": (0.437, 1.0 + 2e-4), "right": (-1.212, 1.0 - 1.5e-4)}   # 카메라 시계 = a0 + rate x 실제 시각
uv_views = [table_to_pixel(traj(t_ref), *cams[0][:3], cams[0][3])[:, None]]
naive = [uv_views[0]]
for ci, name in ((1, "left"), (2, "right")):
    a0, rate = clocks[name]
    t_true = np.arange(0.019 * ci, 10, 1 / 29.97)                     # 이 카메라가 실제로 찍은 순간
    t_own = a0 + rate * t_true                                         # 이 카메라 시계로 적힌 시각
    uvc = table_to_pixel(traj(t_true), *cams[ci][:3], cams[ci][3])[:, None]
    s0, s1 = a0 + rate * 1.0, a0 + rate * 9.0                          # 박수(실제 1 s, 9 s)의 이 카메라 시각
    sync = {"s0": s0, "s1": s1, "off0": 1.0 - s0, "off1": 9.0 - s1}   # 기준 박수 시각 - 이 카메라 박수 시각
    uv_views.append(align_to_ref(t_own, uvc, t_ref, sync))
    naive.append(uvc[:len(t_ref)])
for v in (uv_views, naive):
    n = min(len(x) for x in v)
    for i in range(len(v)): v[i] = v[i][:n]
Xa, _, _ = triangulate_series(np.stack(uv_views), cams, {"max_reproj_px": 1e9})
Xn, _, _ = triangulate_series(np.stack(naive), cams, {"max_reproj_px": 1e9})
truth = traj(t_ref[:len(Xa)])
ea = np.nanmax(np.linalg.norm(Xa[:, 0] - truth, axis=1)[30:-30])
en = np.nanmax(np.linalg.norm(Xn[:, 0] - truth[:len(Xn)], axis=1)[30:-30])
assert ea < 0.003, ea           # 시간 보간: 3 mm 미만
assert en > 0.02, en            # 프레임 번호 짝짓기: 2 cm 넘게 틀림
print(f"   시간 보간 최대 오차 {ea*1000:.2f} mm / 프레임 번호 짝짓기 {en*1000:.1f} mm")

# 13) data.jsonl (프레임 줄만), sync.csv 카메라별 행
f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
f.write('{"sensor":{"type":"gyroscope","values":[0,0,0]},"time":0.001}\n'
        '{"frames":[{"cameraInd":0,"time":0.0333}],"number":0,"time":0.0333}\n'
        '{"frames":[{"cameraInd":0,"time":0.0667}],"number":1,"time":0.0667}\n'); f.close()
assert np.allclose(read_jsonl_frame_times(f.name), [0.0333, 0.0667]); os.unlink(f.name)
f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
f.write("episode_id,camera,offset_start_s,offset_end_s,drift_s,s0,s1,usable,note\n"
        "T1_E001,left,0.43,0.44,0.01,2.0,40.0,1,\nT1_E001,ego,1.2,1.2,0,,,1,\nT1_E001,right,0.1,0.1,0,,,0,가려짐\n"); f.close()
assert read_sync_row(f.name, "T1_E001", camera="left")["s1"] == 40.0
assert read_sync_row(f.name, "T1_E001", camera="ego")["off0"] == 1.2
for cam in ("right", "front"):
    try: read_sync_row(f.name, "T1_E001", camera=cam); raise AssertionError(cam)
    except SyncError: pass
os.unlink(f.name)

# 14) 보정 파일: 여러 대 / 예전 한 대 형식
cj = {"ref_camera": "front", "cameras": {n: {"K": Kc.tolist(), "R": c[1].tolist(), "t": c[2].tolist(), "dist": c[3].tolist()}
                                          for n, c in zip(["front", "left", "right"], cams)}}
cs, ref = load_calibs(cj); assert ref == "front" and list(cs) == ["front", "left", "right"]
cs, ref = load_calibs(cj["cameras"]["front"]); assert ref == "front" and len(cs) == 1
print("selftest 11~14 (삼각측량) OK")
