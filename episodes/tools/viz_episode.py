"""에피소드 하나를 눈으로 보여 주는 영상·이미지 (리타게팅 담당 설명용).

위: 카메라 3대 사진에 3D 키포인트를 다시 그린 것 / 아래 왼쪽: 테이블 좌표계 3D (팔, 손목 궤적, 장애물) /
아래 오른쪽: 손목 x·y·z 와 집기 거리 그래프 (회색 = 값 없음, 빨간 선 = 지금).

  python viz_episode.py ~/out/seohyeon/episodes_clap/T1_E040 --raw ~/data/raw/T1 --out ~/out/seohyeon/viz

출력: <out>/<ID>_viz.png (손목 3D 가 이어진 구간 가운데 프레임), <out>/<ID>_viz.mp4 (--no-video 면 생략)
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

from hand_math import table_to_pixel
from io_utils import find_calib, find_episode_folder, read_frame_times
from multiview import load_calibs
from obstacles import BOX_EDGES, box_corners
from show_kp import draw, longest_run
from sync_check import nearest_frame, read_frame

ARM = ["shoulder", "elbow", "wrist", "index_mcp", "index_tip"]


class Viz:
    def __init__(self, a):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.a, self.plt = a, plt
        self.ep, raw = Path(a.episode).expanduser(), Path(a.raw).expanduser()
        self.meta = m = json.load(open(self.ep / "meta.json"))
        self.names = json.load(open(self.ep / "keypoint_names.json"))
        self.kp, self.pinch = np.load(self.ep / "keypoints.npy"), np.load(self.ep / "pinch_dist.npy")
        self.t = np.loadtxt(self.ep / "exo_ts.csv", skiprows=1)
        self.s, self.cams, self.eid = m["exo_start_frame"], m["cameras"], m["episode_id"]
        self.calib, _ = load_calibs(json.load(open(find_calib(a.calib_dir, self.eid))))
        self.times = {c: read_frame_times(find_episode_folder(raw, self.eid, f"exo_{c}")) for c in self.cams}
        stem = lambda c: "exo" if c == self.cams[0] else f"exo_{c}"
        self.video = {c: next(self.ep / (stem(c) + x) for x in (".mov", ".mp4") if (self.ep / (stem(c) + x)).exists())
                      for c in self.cams}
        self.obs = []
        p = Path(a.obstacles).expanduser()
        if p.exists():
            self.obs = json.load(open(p))["episodes"].get(self.eid, {}).get("obstacles", [])
        self.i = {n: k for k, n in enumerate(self.names)}
        pts = self.kp[np.isfinite(self.kp).all(axis=2)]
        self.lo, self.hi = pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05
        for o in self.obs:
            c = box_corners(o)
            self.lo, self.hi = np.minimum(self.lo, c.min(axis=0)), np.maximum(self.hi, c.max(axis=0))
        self.lo[2] = min(self.lo[2], 0.0)
        self.fig = plt.figure(figsize=(a.width * 3 / 100, 5.2), dpi=100)

    def cameras(self, f):
        i = f - self.s
        tiles = []
        for k, c in enumerate(self.cams):
            j = f if k == 0 else nearest_frame(self.times[c], self.meta["camera_sync"][c], self.times[self.cams[0]][f])[0]
            img = read_frame(self.video[c], j)
            K, R, t, dist = self.calib[c]
            for ob in self.obs:
                px = table_to_pixel(box_corners(ob), K, R, t, dist).astype(int)
                for e0, e1 in BOX_EDGES:
                    cv2.line(img, tuple(px[e0]), tuple(px[e1]), (255, 200, 0), 3, cv2.LINE_AA)
            ok = np.isfinite(self.kp[i]).all(axis=1)
            uv = np.full((len(self.names), 2), np.nan)
            if ok.any():
                uv[ok] = table_to_pixel(self.kp[i][ok], K, R, t, dist)
            draw(img, uv, self.names, False)
            cv2.putText(img, f"{c} f{j}", (20, 70), 0, 2.2, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(img, f"{c} f{j}", (20, 70), 0, 2.2, (0, 255, 255), 4, cv2.LINE_AA)
            h = int(img.shape[0] * self.a.width / img.shape[1])
            tiles.append(cv2.resize(img, (self.a.width, h), interpolation=cv2.INTER_AREA))
        return np.hstack(tiles)

    def plots(self, f):
        i = f - self.s
        fig = self.fig
        fig.clf()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        w = self.kp[:, self.i["wrist"]]
        ax.plot(w[:i + 1, 0], w[:i + 1, 1], w[:i + 1, 2], color="tab:green", lw=1.5, label="wrist path")
        arm = np.array([self.kp[i, self.i[n]] for n in ARM])
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color="tab:blue", lw=3, ms=5, label="arm now")
        for ob in self.obs:
            c = box_corners(ob)
            for e0, e1 in BOX_EDGES:
                ax.plot(*zip(c[e0], c[e1]), color="tab:orange", lw=1.5)
        gx, gy = np.meshgrid(np.linspace(self.lo[0], self.hi[0], 2), np.linspace(self.lo[1], self.hi[1], 2))
        ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.8", alpha=0.3)
        ax.set_xlim(self.lo[0], self.hi[0]); ax.set_ylim(self.lo[1], self.hi[1]); ax.set_zlim(self.lo[2], self.hi[2])
        ax.set_box_aspect(self.hi - self.lo)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=25, azim=-70)
        ax.legend(loc="upper left", fontsize=8)
        t = self.t - self.t[0]
        a1 = fig.add_subplot(2, 2, 2)
        d = np.diff(np.r_[0, np.isnan(w).any(axis=1).astype(int), 0])
        for x0, x1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            a1.axvspan(t[x0], t[min(x1, len(t) - 1)], color="0.87", lw=0)
        for k, n in enumerate("xyz"):
            a1.plot(t, w[:, k], label=n)
        a1.axvline(t[i], color="r"); a1.set_ylabel("wrist (m)"); a1.legend(loc="upper right", ncol=3, fontsize=8); a1.grid(alpha=0.3)
        a2 = fig.add_subplot(2, 2, 4, sharex=a1)
        a2.plot(t, self.pinch, color="tab:purple"); a2.axvline(t[i], color="r")
        a2.set_ylabel("pinch (m)"); a2.set_xlabel("time (s)"); a2.grid(alpha=0.3)
        m = self.meta
        fig.suptitle(f"{self.eid}  hand={m['hand']}  table frame: x right, y toward person, z up", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def frame(self, f):
        top, bot = self.cameras(f), self.plots(f)
        if bot.shape[1] != top.shape[1]:
            bot = cv2.resize(bot, (top.shape[1], bot.shape[0]))
        return np.vstack([top, bot])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="process_episode.py 결과 폴더")
    ap.add_argument("--raw", required=True, help="태스크 폴더, 예: ~/data/raw/T1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir", default=str(Path(__file__).resolve().parents[2] / "sync"))
    ap.add_argument("--obstacles", default=str(Path(__file__).resolve().parent.parent / "obstacles.json"))
    ap.add_argument("--width", type=int, default=480, help="카메라 칸 너비 (px)")
    ap.add_argument("--stride", type=int, default=2, help="영상에 쓸 프레임 간격")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    v = Viz(a)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lo, hi = longest_run(np.isfinite(v.kp[:, v.i["wrist"]]).all(axis=1))
    mid = v.s + (lo + hi) // 2
    png = out / f"{v.eid}_viz.png"
    cv2.imwrite(str(png), v.frame(mid))
    print("저장 ->", png, "프레임", mid, flush=True)
    if a.no_video:
        return
    first = v.frame(v.s)
    h, w = first.shape[:2]
    path = out / f"{v.eid}_viz.mp4"
    fps = v.meta["exo_fps"] / a.stride
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for f in range(v.s, v.s + len(v.kp), a.stride):
        vw.write(v.frame(f)); n += 1
        if n % 50 == 0:
            print(f"  {n} 프레임", flush=True)
    vw.release()
    print("저장 ->", path, f"{n} 프레임, {fps:.0f} fps")


if __name__ == "__main__":
    main()
