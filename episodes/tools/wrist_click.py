"""손목 3D 가 없는 구간을 사람이 직접 찍어 채운다 (손이 컵에 가려 카메라 2대가 못 잡는 구간).

찍을 프레임: 3D 는 없는데 작업 손 쪽 카메라의 Hands 는 손을 잡은 프레임 (손을 무릎에 둔 구간은 저절로 빠짐).
그 카메라의 값은 이미 있으므로, 다른 카메라 한 대에서 손목을 찍으면 3D 가 나온다. 찍은 프레임 사이는 보간한다.
정확도를 재려고 손이 잘 보이는 프레임도 몇 장 섞는다. 이런 프레임에서는 답이 되는 Hands 점을 작업 손 쪽에만 보여 준다.
손목 기준: Hands 의 wrist 점과 같은 곳 (손바닥이 시작되는 손목 주름 가운데).

1) 서버: 프레임 뽑기
  python wrist_click.py frames ~/out/seohyeon/episodes_clap --raw ~/data/raw --ids T1_E006 T1_E067 --out ~/out/seohyeon/wrist_frames
  로컬로 받기: scp -P 10000 -r coss37@155.230.81.91:~/out/seohyeon/wrist_frames ~/Downloads/

2) 로컬: 찍기 (창 하나에 front·left·right·도움말)
  python wrist_click.py click --frames ~/Downloads/wrist_frames

  왼쪽 클릭  손목 위치 (잘 보이는 카메라 칸에서. 빨간 점은 Hands 가 찾은 손목, 노란 점은 앞 프레임에 찍은 곳)
  s          저장하고 다음      x  안 보임으로 저장하고 다음
  u          이 프레임 클릭 지우기
  d / a      저장 없이 다음 / 이전      q  끝내기
  저장할 때마다 <frames>/wrist_clicks.json 에 바로 쓴다.

3) 서버: 3D 로 만들고 평가 (wrist_clicks.json 을 서버로 올린 뒤)
  python wrist_click.py apply ~/out/seohyeon/episodes_clap --clicks wrist_clicks.json --calib-dir ../../sync
  출력: <에피소드>/wrist_manual.npy (T,3) 테이블 좌표 m, 찍은 값이 없으면 NaN. keypoints.npy 는 바꾸지 않는다.
        <에피소드>/wrist_manual_mask.npy (T,) 1 = 찍은 프레임, 2 = 그 사이를 보간한 프레임
"""
import argparse, json
from pathlib import Path
import numpy as np

VIEWS = ["front", "left", "right"]
WRIST = 2       # extractors.EXO_NAMES 의 wrist


def runs(mask, min_len=1):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def pick_frames(kp_raw, uv2, wc, step, min_gap, n_check):
    """(빈 구간 프레임 목록, 정확도 확인 프레임 목록). 번호는 구간 안 상대 번호."""
    no3d = np.isnan(kp_raw[:, WRIST]).any(axis=1)
    seen = np.isfinite(uv2[:, :, WRIST]).all(axis=2)
    gap = []
    for a, b in runs(no3d & seen[wc], min_gap):
        gap += sorted(set(range(a, b, step)) | {b - 1})
    others = np.delete(seen, wc, axis=0).any(axis=0)
    good = np.flatnonzero(~no3d & seen[wc] & others)
    check = good[np.linspace(0, len(good) - 1, n_check).astype(int)].tolist() if len(good) >= n_check else []
    return gap, check


def cmd_frames(a):
    import cv2
    from io_utils import find_episode_folder, read_frame_times
    from sync_check import nearest_frame, read_frame
    root, raw, out = (Path(p).expanduser() for p in (a.root, a.raw, a.out))
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for eid in a.ids:
        ep = root / eid
        meta = json.load(open(ep / "meta.json"))
        s, cams = meta["exo_start_frame"], meta["cameras"]
        uv2, kp_raw = np.load(ep / "keypoints_2d.npy"), np.load(ep / "keypoints_raw.npy")
        wc = cams.index(meta["hand"])
        gap, check = pick_frames(kp_raw, uv2, wc, a.step, a.min_gap, a.n_check)
        folder = {c: find_episode_folder(raw / eid.split("_")[0], eid, f"exo_{c}") for c in cams}
        times = {c: read_frame_times(folder[c]) for c in cams}
        stem = lambda c: "exo" if c == cams[0] else f"exo_{c}"
        video = {c: next(ep / (stem(c) + x) for x in (".mov", ".mp4") if (ep / (stem(c) + x)).exists()) for c in cams}
        (out / eid).mkdir(exist_ok=True)
        items = []
        for i in sorted(set(gap) | set(check)):
            f = s + i
            frames, hands = {}, {}
            for k, c in enumerate(cams):
                j = f if k == 0 else nearest_frame(times[c], meta["camera_sync"][c], times[cams[0]][f])[0]
                img = read_frame(video[c], j)
                img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / eid / f"f{f}_{c}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                frames[c] = int(j)
                if np.isfinite(uv2[k, i, WRIST]).all() and (k == wc or i not in check):
                    hands[c] = uv2[k, i, WRIST].tolist()
            items.append({"frame": f, "cam_frames": frames, "hands": hands})
        index[eid] = {"hand": meta["hand"], "scale": a.scale, "items": items, "check": [s + i for i in check]}
        json.dump(index, open(out / "index.json", "w"), ensure_ascii=False, indent=1)
        print(f"{eid} {meta['hand']}: 찍을 프레임 {len(gap)}, 확인용 {len(check)}", flush=True)
    print("저장 ->", out)


class Clicker:
    def __init__(self, a):
        self.a, self.dir = a, Path(a.frames).expanduser()
        self.index = json.load(open(self.dir / "index.json"))
        self.items = [(e, it) for e in sorted(self.index) for it in self.index[e]["items"]]
        self.out = self.dir / "wrist_clicks.json"
        self.data = json.load(open(self.out)) if self.out.exists() else {}
        self.k = next((i for i, (e, it) in enumerate(self.items) if str(it["frame"]) not in self.data.get(e, {})), 0)
        self.load()

    def load(self):
        import cv2
        self.eid, self.it = self.items[self.k]
        self.scale = self.index[self.eid]["scale"]
        self.imgs = {c: cv2.imread(str(self.dir / self.eid / f"f{self.it['frame']}_{c}.jpg")) for c in VIEWS}
        saved = self.data.get(self.eid, {}).get(str(self.it["frame"]))
        self.clicks = dict(saved["clicks"]) if saved else {}
        self.msg = "loaded saved" if saved else ""

    def rect(self, i):
        w = self.a.tile; h = w * 3 // 4
        return (i % 2) * w, (i // 2) * h, w, h

    def on_mouse(self, ev, x, y, *_):
        import cv2
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        for i, c in enumerate(VIEWS):
            x0, y0, w, h = self.rect(i)
            if self.imgs[c] is not None and x0 <= x < x0 + w and y0 <= y < y0 + h:
                f = w / self.imgs[c].shape[1] * self.scale      # 원본 픽셀 -> 칸 픽셀
                self.clicks[c] = [(x - x0) / f, (y - y0) / f]
                self.msg = f"clicked {c}"
                return

    def prev_click(self):
        d = self.data.get(self.eid, {})
        keys = [int(f) for f in d if int(f) < self.it["frame"] and d[f]["clicks"]]
        return d[str(max(keys))]["clicks"] if keys else {}

    def draw(self):
        import cv2
        w = self.a.tile; h = w * 3 // 4
        canvas = np.zeros((2 * h, 2 * w, 3), np.uint8)
        prev = self.prev_click()
        for i, c in enumerate(VIEWS):
            x0, y0, _, _ = self.rect(i)
            img = self.imgs[c]
            if img is None:
                continue
            f = w / img.shape[1] * self.scale
            tile = cv2.resize(img, (w, h))
            pt = lambda p: (int(p[0] * f), int(p[1] * f))
            if c in prev:
                cv2.circle(tile, pt(prev[c]), 5, (0, 255, 255), 2, cv2.LINE_AA)
            if c in self.it["hands"]:
                cv2.circle(tile, pt(self.it["hands"][c]), 5, (0, 0, 255), -1, cv2.LINE_AA)
            if c in self.clicks:
                cv2.drawMarker(tile, pt(self.clicks[c]), (255, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.putText(tile, f"{c} f{self.it['cam_frames'][c]}", (8, 28), 0, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            canvas[y0:y0 + h, x0:x0 + w] = tile
        done = sum(len(v) for v in self.data.values())
        L = [f"{self.eid} hand={self.index[self.eid]['hand']}  frame {self.it['frame']}", f"{self.k + 1}/{len(self.items)}  saved {done}",
             "clicks: " + (", ".join(self.clicks) or "none"), "",
             "click: wrist (best view)   red = Hands, yellow = previous",
             "s: save+next   x: not visible+next   u: clear", "d/a: next/prev   q: quit", "", self.msg]
        for i, s in enumerate(L):
            cv2.putText(canvas, s, (w + 12, h + 30 + 26 * i), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def save(self, none=False):
        self.data.setdefault(self.eid, {})[str(self.it["frame"])] = {"clicks": {} if none else self.clicks, "none": none}
        json.dump(self.data, open(self.out, "w"), ensure_ascii=False, indent=1)

    def key(self, ch):
        if ch == "q":
            return False
        if ch == "u":
            self.clicks = {}
        elif ch == "s":
            if not self.clicks:
                self.msg = "no click (x if not visible)"; return True
            self.save(); return self.move(1)
        elif ch == "x":
            self.save(True); return self.move(1)
        elif ch in "da":
            return self.move(1 if ch == "d" else -1)
        return True

    def move(self, step):
        if not 0 <= self.k + step < len(self.items):
            self.msg = "end"; return True
        self.k += step; self.load(); return True


def cmd_click(a):
    import cv2
    c = Clicker(a)
    cv2.namedWindow("wrist", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("wrist", c.on_mouse)
    while True:
        cv2.imshow("wrist", c.draw())
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not c.key(chr(k)):
            break
    cv2.destroyAllWindows()
    print(f"저장 -> {c.out}")


def triangulate(frame_clicks, hands_uv, cams, wc_name, max_px):
    """찍은 점 + 작업 손 쪽 Hands -> 3D. 작업 손 쪽 칸을 직접 찍었으면 그 값을 쓴다."""
    from multiview import triangulate_point
    uv = dict(frame_clicks)
    if wc_name not in uv and np.isfinite(hands_uv).all():
        uv[wc_name] = hands_uv.tolist()
    if len(uv) < 2:
        return None, None
    names = list(uv)
    X, e, n = triangulate_point([np.array(uv[c], float) for c in names], [cams[c] for c in names], max_px)
    return (X, e) if n >= 2 else (None, e)


def cmd_apply(a):
    from io_utils import find_calib
    from multiview import DEFAULT_TRI, load_calibs
    root = Path(a.root).expanduser()
    clicks = json.load(open(Path(a.clicks).expanduser()))
    allerr = []
    for eid in sorted(clicks):
        ep = root / eid
        meta = json.load(open(ep / "meta.json"))
        s, cams_order, side = meta["exo_start_frame"], meta["cameras"], meta["hand"]
        cams, _ = load_calibs(json.load(open(find_calib(a.calib_dir, eid))))
        uv2, raw = np.load(ep / "keypoints_2d.npy"), np.load(ep / "keypoints_raw.npy")
        t = np.loadtxt(ep / "exo_ts.csv", skiprows=1)
        wc = cams_order.index(side)
        no3d = np.isnan(raw[:, WRIST]).any(axis=1)
        T = len(t)
        out, mask, errs, rej = np.full((T, 3), np.nan), np.zeros(T, np.uint8), [], 0
        pts = {}
        for fs, v in clicks[eid].items():
            i = int(fs) - s
            if v.get("none") or not v["clicks"] or not 0 <= i < T:      # 구간이 바뀌어도 그 안의 값만 쓴다
                continue
            X, _ = triangulate(v["clicks"], uv2[wc, i, WRIST], cams, side, a.max_px or DEFAULT_TRI["max_reproj_px"])
            if X is None:
                rej += 1
                continue
            if no3d[i]:
                pts[i] = X
            else:
                errs.append(np.linalg.norm(X - raw[i, WRIST]) * 100)
        keys = sorted(pts)
        for i in keys:
            out[i], mask[i] = pts[i], 1
        for p, q in zip(keys, keys[1:]):
            if q - p <= a.max_interp:
                w = (t[p + 1:q] - t[p]) / (t[q] - t[p])
                out[p + 1:q] = pts[p] + w[:, None] * (pts[q] - pts[p])
                mask[p + 1:q] = 2
        np.save(ep / "wrist_manual.npy", out)
        np.save(ep / "wrist_manual_mask.npy", mask)
        allerr += errs
        elbow = raw[:, 1]
        arm = np.linalg.norm(out - elbow, axis=1) * 100
        arm0 = np.linalg.norm(raw[:, WRIST] - elbow, axis=1) * 100
        step = np.linalg.norm(np.diff(out, axis=0), axis=1) * 100
        gap_fill = float((np.isfinite(out).all(axis=1) & no3d).sum())
        print(f"{eid}: 찍은 3D {len(keys)}, 재투영 탈락 {rej}, 채운 프레임 {gap_fill:.0f}")
        if keys:
            print(f"  아래팔 cm: Hands 3D {np.nanmedian(arm0):.1f}, 찍은 값 {np.nanmedian(arm):.1f} (10~90% {np.nanpercentile(arm, 10):.1f}~{np.nanpercentile(arm, 90):.1f})")
            print(f"  프레임 간 이동 cm: 중앙값 {np.nanmedian(step):.2f}, 99% {np.nanpercentile(step, 99):.2f}, z {np.nanmin(out[:, 2]):.3f}~{np.nanmax(out[:, 2]):.3f} m")
        if errs:
            print(f"  확인용 프레임과 Hands 3D 차이 cm: {', '.join(f'{e:.1f}' for e in errs)}")
    if allerr:
        print(f"\n전체 확인용 {len(allerr)}개: 중앙값 {np.median(allerr):.1f} cm, 90% {np.percentile(allerr, 90):.1f} cm, 최대 {max(allerr):.1f} cm")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("root", help="process_all.py 의 --out")
    f.add_argument("--raw", required=True, help="태스크 폴더들의 상위, 예: ~/data/raw")
    f.add_argument("--ids", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--step", type=int, default=12, help="찍을 프레임 간격 (기준 카메라 프레임)")
    f.add_argument("--min-gap", type=int, default=15, help="이보다 짧은 빈 구간은 건너뜀 (프레임)")
    f.add_argument("--n-check", type=int, default=5, help="정확도 확인용 프레임 수 (에피소드당)")
    f.add_argument("--scale", type=float, default=0.5)
    c = sub.add_parser("click")
    c.add_argument("--frames", required=True)
    c.add_argument("--tile", type=int, default=640)
    p = sub.add_parser("apply")
    p.add_argument("root")
    p.add_argument("--clicks", required=True)
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--max-px", type=float, default=None)
    p.add_argument("--max-interp", type=int, default=30, help="찍은 프레임 간격이 이보다 크면 보간하지 않음")
    a = ap.parse_args()
    {"frames": cmd_frames, "click": cmd_click, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
