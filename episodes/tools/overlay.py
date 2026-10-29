"""키포인트를 기준 카메라(exo) 영상에 다시 그려 눈으로 확인 (8단계).

  python overlay.py episodes/T1_E001 ../../sync/exo_calib.json out.mp4
점이 손·팔 위에 찍히는지, 특히 오른손에 찍히는지(왼손 아님) 파일럿에서 확인한다.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from hand_math import table_to_pixel
from multiview import load_calibs

d, calib, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
cams, ref = load_calibs(calib)
K, R, t, dist = cams[ref]
kp = np.load(d / "keypoints.npy")
start = json.load(open(d / "meta.json"))["exo_start_frame"]  # exo 영상은 전체가 복사돼 있음
video = next((p for p in (d / "exo.mov", d / "exo.mp4") if p.exists()), None)
if video is None:
    raise SystemExit("exo.mov / exo.mp4 없음")
cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
vw = None
for f in range(len(kp)):
    ok, img = cap.read()
    if not ok:
        break
    valid = ~np.isnan(kp[f]).any(axis=1)
    if valid.any():
        for (u, v) in table_to_pixel(kp[f][valid], K, R, t, dist):
            cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), -1)
    if vw is None:
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
    vw.write(img)
if vw is None:
    raise SystemExit("읽은 프레임이 없음 (exo 영상 / start 프레임 확인)")
vw.release()
print("saved", out)
