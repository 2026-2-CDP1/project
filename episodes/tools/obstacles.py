"""장애물(방해물·아크릴 판) 위치를 테이블 좌표계 박스로 만든다 (환경 충돌률용, metrics/definitions.md 2.9).

장애물은 모두 테이블 위에 서 있으므로, 바닥이 테이블에 닿는 끝점을 영상에서 클릭하면 카메라 한 대로도
테이블 좌표(z=0 평면)가 나온다. 높이는 잰 값(--height)을 쓰고, 다른 카메라에 박스를 그려 맞는지 본다.
투명 아크릴은 깊이(LiDAR)에 안 잡히지만 RGB 에서 테두리·받침이 보이므로 사람이 찍는다.

1) 서버: 에피소드마다 front/left/right 같은 순간 프레임을 뽑는다 (sync.csv 로 시각 맞춤)
  python obstacles.py frames --raw ~/data/raw --sync ../../sync/sync.csv --out ~/out/seohyeon/obstacle_frames
  (--ids 를 빼면 records.csv layout 에 방해물·장애물·아크릴·파티션이 들어간 에피소드 + EXTRA, 84개)
  로컬로 받기: scp -P 10000 -r coss37@155.230.81.91:~/out/seohyeon/obstacle_frames .

2) 로컬: 클릭 (창 하나에 front·left·right·도움말)
  python obstacles.py click --frames obstacle_frames --calib-dir ../../sync

  왼쪽 클릭  장애물 바닥 끝점 (어느 카메라 칸이든). 판이면 양 끝 2점, 상자면 바닥 모서리 3~4점
  n          지금 점들로 장애물 하나 완성 (아크릴 3개면 3번)
  u          마지막 점 지우기 (점이 없으면 마지막 장애물 지우기)
  [ ]        마지막 장애물 높이 -1 / +1 cm
  f          다른 시점 프레임 (손이 가리면)
  p          이전에 저장한 에피소드의 장애물 복사 (같은 배치가 이어질 때, 그린 박스로 확인)
  s          저장하고 다음 에피소드      x  장애물 없음으로 저장
  d / a      저장 없이 다음 / 이전        q  끝내기
  저장할 때마다 --out 파일(기본 <frames>/obstacles.json)에 바로 쓴다. 다시 열면 저장한 값부터 보인다.

출력 obstacles.json:
  {"frame": 설명, "episodes": {"T3_E031": {"calib": "exo_calib_T3.json", "front_frame": 812,
     "obstacles": [{"name": "obs1", "center": [x, y, z], "size": [길이, 두께, 높이], "yaw_deg": 각도,
                    "base_points": [[x, y], ...]}]}}}
  테이블 좌표계(exo_calib_T{n}.json 의 table_frame), m. center 는 박스 중심(z = 높이/2).
  size[0] 은 yaw 방향(+x 에서 z 축으로 돌린 각도) 길이.
"""
import argparse, json, math
from pathlib import Path
import numpy as np

from io_utils import find_calib

KEYWORDS = ["방해물", "장애물", "아크릴", "파티션"]
EXTRA = ["T3_E074", "T3_E075"]     # 기록지 layout 이 "몸 바로 앞에서" 뿐이지만 아크릴 판 배치 (T3 몸 바로 앞)
VIEWS = ["front", "left", "right"]
FRAME_NOTE = ("테이블 좌표계 (exo_calib_T{n}.json table_frame), m. center = 박스 중심, "
              "size = [yaw 방향 길이, 두께, 높이], yaw_deg = +x 에서 z 축으로 돌린 각도")


# ---------------------------------------------------------------- 계산
def pixel_to_plane(uv, K, R, t, dist, z0=0.0):
    """원본 픽셀 (u, v) -> 테이블 평면 z=z0 위의 점 (x, y, z). 평면 뒤쪽이면 None."""
    import cv2
    n = cv2.undistortPoints(np.array([[uv]], np.float64), K, dist).reshape(2)
    C = -R.T @ t                                   # 카메라 중심 (테이블 좌표)
    d = R.T @ np.array([n[0], n[1], 1.0])
    if abs(d[2]) < 1e-9:
        return None
    s = (z0 - C[2]) / d[2]
    return None if s <= 0 else C + s * d


def make_box(pts_xy, height, thick):
    """바닥 끝점들 (N>=2, xy) -> 박스. 2점이면 그 선분을 길이로, 3점 이상이면 최소 외접 사각형."""
    p = np.asarray(pts_xy, float)
    if len(p) == 2:
        c, v = p.mean(axis=0), p[1] - p[0]
        L, T, yaw = float(np.hypot(*v)), thick, math.degrees(math.atan2(v[1], v[0]))
    else:
        import cv2
        (cx, cy), (w, h), ang = cv2.minAreaRect(p.astype(np.float32))
        c = np.array([cx, cy])
        L, T, yaw = (w, h, ang) if w >= h else (h, w, ang + 90)
        T = max(T, thick)
    yaw = (yaw + 90) % 180 - 90                    # 박스는 180° 대칭
    return {"center": [round(float(c[0]), 4), round(float(c[1]), 4), round(height / 2, 4)],
            "size": [round(float(L), 4), round(float(T), 4), round(height, 4)],
            "yaw_deg": round(float(yaw), 2), "base_points": [[round(float(x), 4), round(float(y), 4)] for x, y in p]}


def box_corners(ob):
    """박스 -> 꼭짓점 8개 (테이블 좌표). 아래 4개, 위 4개 순서."""
    (cx, cy, cz), (L, T, H), a = ob["center"], ob["size"], math.radians(ob["yaw_deg"])
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    base = [np.array([cx, cy]) + sx * L / 2 * u + sy * T / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.array([[*b, cz - H / 2] for b in base] + [[*b, cz + H / 2] for b in base])


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------------------------------------------------------------- 1) 서버: 프레임 뽑기
def cmd_frames(a):
    import cv2
    from io_utils import find_episode_folder, find_video, read_frame_times, read_records
    from sync_check import load_rows, nearest_frame, read_frame

    raw, out = Path(a.raw).expanduser(), Path(a.out).expanduser()
    rec = read_records(a.records)
    ids = a.ids or [k for k, r in rec.items() if any(w in r["layout"] for w in a.keywords) or k in a.extra]
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.load(open(index_path)) if index_path.exists() else {}
    for n, eid in enumerate(sorted(ids), 1):
        task_raw = raw / eid.split("_")[0]
        rows = load_rows(a.sync, eid)
        t_f = read_frame_times(find_episode_folder(task_raw, eid, "exo_front"))
        times = {c: read_frame_times(find_episode_folder(task_raw, eid, f"exo_{c}")) for c in VIEWS}
        videos = {c: find_video(find_episode_folder(task_raw, eid, f"exo_{c}")) for c in VIEWS}
        shots = []
        for k, frac in enumerate(a.fracs):
            i_f = int(round(frac * (len(t_f) - 1)))
            shot = {"front_frame": i_f, "frames": {}, "err_ms": {}}
            for c in VIEWS:
                if c == "front":
                    i, err = i_f, 0.0
                elif c in rows:
                    i, err = nearest_frame(times[c], rows[c], t_f[i_f])
                else:
                    continue                               # sync.csv 에 행 없음 (그 카메라 칸은 비움)
                img = read_frame(videos[c], i)
                if img is None:
                    continue
                img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                (out / eid).mkdir(exist_ok=True)
                cv2.imwrite(str(out / eid / f"s{k}_{c}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                shot["frames"][c], shot["err_ms"][c] = int(i), round(err * 1000, 1)
            shots.append(shot)
        r = rec.get(eid, {})
        index[eid] = {"layout": r.get("layout", ""), "scale": a.scale, "shots": shots}
        json.dump(index, open(index_path, "w"), ensure_ascii=False, indent=1)
        far = {c: e for s in shots for c, e in s["err_ms"].items() if abs(e) > 100}
        print(f"[{n}/{len(ids)}] {eid} {r.get('layout', '')[:30]}" + (f"  시각 차 큼(ms): {far}" if far else ""), flush=True)
    print("저장 ->", out)


# ---------------------------------------------------------------- 2) 로컬: 클릭
class Clicker:
    def __init__(self, a):
        self.a, self.dir = a, Path(a.frames).expanduser()
        self.index = json.load(open(self.dir / "index.json"))
        self.ids = a.ids or sorted(self.index)
        self.out = Path(a.out).expanduser() if a.out else self.dir / "obstacles.json"
        self.data = json.load(open(self.out)) if self.out.exists() else {"frame": FRAME_NOTE, "episodes": {}}
        self.calibs = {}
        self.k = 0 if not a.start_from else self.ids.index(a.start_from)
        self.load()

    # 에피소드 상태
    def load(self):
        self.eid = self.ids[self.k]
        info, saved = self.index[self.eid], self.data["episodes"].get(self.eid)
        self.shot = 0
        if saved:                                       # 저장했던 시점 프레임으로 연다
            self.shot = next((i for i, s in enumerate(info["shots"]) if s["front_frame"] == saved["front_frame"]), 0)
        self.obs = [dict(o) for o in saved["obstacles"]] if saved else []
        self.pts, self.msg = [], "loaded saved" if saved else ""
        self.load_images()

    def load_images(self):
        import cv2
        from multiview import load_calibs
        info = self.index[self.eid]
        self.scale = info["scale"]
        self.imgs = {c: cv2.imread(str(self.dir / self.eid / f"s{self.shot}_{c}.jpg")) for c in VIEWS}
        path = find_calib(self.a.calib_dir, self.eid)
        self.calib_name = path.name
        if path.name not in self.calibs:
            self.calibs[path.name] = load_calibs(json.load(open(path)))[0]
        self.cams = self.calibs[path.name]

    def save(self, obs):
        info = self.index[self.eid]
        self.data["episodes"][self.eid] = {"calib": self.calib_name, "layout": info["layout"],
                                           "front_frame": info["shots"][self.shot]["front_frame"], "obstacles": obs}
        json.dump(self.data, open(self.out, "w"), ensure_ascii=False, indent=1)

    # 화면 배치: 2x2 (front, left / right, 도움말)
    def tile_rect(self, i):
        w = self.a.tile
        h = int(w * 3 / 4)
        return (i % 2) * w, (i // 2) * h, w, h

    def on_mouse(self, ev, x, y, *_):
        import cv2
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        for i, c in enumerate(VIEWS):
            x0, y0, w, h = self.tile_rect(i)
            img = self.imgs[c]
            if img is None or not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            f = w / img.shape[1]                        # 칸 배율
            uv = ((x - x0) / f / self.scale, (y - y0) / f / self.scale)   # 원본 픽셀
            K, R, t, dist = self.cams[c]
            p = pixel_to_plane(uv, K, R, t, dist)
            if p is None:
                self.msg = "not on table plane"
            else:
                self.pts.append(p[:2].tolist())
                self.msg = f"point {len(self.pts)}: ({p[0]:.3f}, {p[1]:.3f}) m [{c}]"
            return

    def draw(self):
        import cv2
        from hand_math import table_to_pixel
        w = self.a.tile
        h = int(w * 3 / 4)
        canvas = np.zeros((2 * h, 2 * w, 3), np.uint8)
        for i, c in enumerate(VIEWS):
            x0, y0, _, _ = self.tile_rect(i)
            img = self.imgs[c]
            if img is None:
                cv2.putText(canvas, f"{c}: no image", (x0 + 10, y0 + 40), 0, 1, (0, 0, 255), 2)
                continue
            f = w / img.shape[1]
            tile = cv2.resize(img, (w, int(img.shape[0] * f)))
            K, R, t, dist = self.cams[c]
            to_px = lambda P: table_to_pixel(P, K, R, t, dist) * self.scale * f
            for j, ob in enumerate(self.obs):
                px = to_px(box_corners(ob)).astype(int)
                col = (0, 255, 0) if j < len(self.obs) - 1 else (0, 200, 255)
                for e0, e1 in BOX_EDGES:
                    cv2.line(tile, tuple(px[e0]), tuple(px[e1]), col, 2)
            if self.pts:
                px = to_px(np.array([[x, y, 0.0] for x, y in self.pts])).astype(int)
                for q in px:
                    cv2.circle(tile, tuple(q), 5, (255, 0, 255), -1)
                if len(px) > 1:
                    cv2.polylines(tile, [px.reshape(-1, 1, 2)], False, (255, 0, 255), 1)
            fr = self.index[self.eid]["shots"][self.shot]["frames"].get(c)
            cv2.putText(tile, f"{c} f{fr}", (8, 28), 0, 0.8, (0, 255, 255), 2)
            canvas[y0:y0 + tile.shape[0], x0:x0 + w] = tile[:h]
        self.draw_help(canvas, w, h)
        return canvas

    def draw_help(self, canvas, x0, y0):
        """창 글자는 OpenCV 기본 글꼴이라 영어만."""
        import cv2
        saved = self.eid in self.data["episodes"]
        lines = [f"{self.eid} ({self.k + 1}/{len(self.ids)}) {'[saved]' if saved else ''}",
                 f"shot {self.shot + 1}/{len(self.index[self.eid]['shots'])}  height {self.a.height * 100:.0f}cm  thick {self.a.thick * 100:.0f}cm",
                 f"points {len(self.pts)}"]
        for j, ob in enumerate(self.obs):
            L, T, H = ob["size"]
            lines.append(f"obs{j + 1}: L {L * 100:.1f} T {T * 100:.1f} H {H * 100:.0f} cm, yaw {ob['yaw_deg']:.0f}")
        lines += ["", "click: base point   n: finish obstacle   u: undo",
                  "[ ]: height -/+1cm  f: other shot   p: copy previous",
                  "s: save+next   x: none+next   d/a: next/prev   q: quit", "", self.msg]
        for i, s in enumerate(lines):
            cv2.putText(canvas, s, (x0 + 12, y0 + 30 + 26 * i), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def key(self, ch):
        """키 처리. False 면 끝."""
        if ch == "q":
            return False
        if ch == "n":
            if len(self.pts) < 2:
                self.msg = "need 2+ points"
            else:
                ob = make_box(self.pts, self.a.height, self.a.thick)
                self.obs.append({"name": f"obs{len(self.obs) + 1}", **ob}); self.pts = []
                self.msg = f"obs{len(self.obs)} length {ob['size'][0] * 100:.1f} cm"
        elif ch == "u":
            if self.pts:
                self.pts.pop()
            elif self.obs:
                self.obs.pop()
        elif ch in "[]" and self.obs:
            ob = self.obs[-1]
            H = max(0.01, ob["size"][2] + (0.01 if ch == "]" else -0.01))
            ob["size"][2], ob["center"][2] = round(H, 4), round(H / 2, 4)
        elif ch == "f":
            self.shot = (self.shot + 1) % len(self.index[self.eid]["shots"]); self.load_images()
        elif ch == "p":
            prev = [e for e in self.ids[:self.k] if e in self.data["episodes"]]
            if prev:
                self.obs = [json.loads(json.dumps(o)) for o in self.data["episodes"][prev[-1]]["obstacles"]]
                self.msg = f"copied from {prev[-1]} - check boxes"
            else:
                self.msg = "no saved episode before"
        elif ch in "sx":
            if ch == "s" and self.pts:
                self.msg = "unfinished points (n or u)"; return True
            self.save([] if ch == "x" else self.obs)
            return self.move(1)
        elif ch in "da":
            return self.move(1 if ch == "d" else -1)
        return True

    def move(self, step):
        if not 0 <= self.k + step < len(self.ids):
            self.msg = "last episode" if step > 0 else "first episode"; return True
        self.k += step; self.load(); return True


def cmd_click(a):
    import cv2
    c = Clicker(a)
    cv2.namedWindow("obstacles", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("obstacles", c.on_mouse)
    while True:
        cv2.imshow("obstacles", c.draw())
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not c.key(chr(k)):
            break
    cv2.destroyAllWindows()
    print(f"저장된 에피소드 {len(c.data['episodes'])}개 -> {c.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="서버: 카메라 3대 같은 순간 프레임 뽑기")
    f.add_argument("--raw", required=True, help="태스크 폴더들의 상위, 예: ~/data/raw")
    f.add_argument("--sync", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ids", nargs="*")
    f.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    f.add_argument("--keywords", nargs="+", default=KEYWORDS, help="--ids 가 없을 때 layout 에서 찾을 말")
    f.add_argument("--extra", nargs="*", default=EXTRA, help="키워드에 안 걸리지만 장애물이 있는 에피소드")
    f.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="front 영상 위치 (0~1), 시점 후보")
    f.add_argument("--scale", type=float, default=0.5, help="저장 배율 (0.5 = 960x720)")
    c = sub.add_parser("click", help="로컬: 장애물 바닥 끝점 클릭")
    c.add_argument("--frames", required=True, help="frames 의 --out 을 받은 폴더")
    c.add_argument("--calib-dir", default=str(Path(__file__).resolve().parents[2] / "sync"))
    c.add_argument("--out", help="기본 <frames>/obstacles.json")
    c.add_argument("--ids", nargs="*")
    c.add_argument("--start-from")
    c.add_argument("--height", type=float, default=0.15, help="장애물 높이 (m)")
    c.add_argument("--thick", type=float, default=0.01, help="판 두께 (m), 2점으로 찍을 때")
    c.add_argument("--tile", type=int, default=640, help="카메라 칸 너비 (px)")
    a = ap.parse_args()
    (cmd_frames if a.cmd == "frames" else cmd_click)(a)


if __name__ == "__main__":
    main()
