"""파지 프레임(grasp_frame)을 사람이 표시하고 labels.json 을 만든다. 행동 구간(approach 등)은 팀 결정으로 만들지 않는다.

기준: 작업 손이 물체를 완전히 쥔 첫 사진. 여러 번 잡으면 첫 번째만 (../GRASP_MARKING.md).

1) 서버: 사진 뽑기 (동작 구간 안, 0.1초 간격, front + 작업 손 쪽 카메라). --part 로 나눠 뽑으면 시간이 줄고 각자 자기 폴더만 받으면 된다.
  python grasp_click.py frames --segments ~/out/seohyeon/segments_final_v4.csv --raw ~/data/raw --sync ../../sync/sync.csv \\
      --out ~/out/seohyeon/grasp_frames_1 --part 1/3

2) 로컬: 표시
  python grasp_click.py click --frames ~/Downloads/grasp_frames_1 --name 홍길동
  d / a  다음 / 이전 사진    e / w  5장 앞 / 뒤    1  지금 사진을 파지 프레임으로    s  저장하고 다음 에피소드
  b  이전 에피소드           x  쥐는 순간을 알 수 없음      q  끝내기
  저장할 때마다 <frames>/grasp_marks_<이름>.json 에 바로 쓴다. 다시 켜면 안 한 에피소드부터.

3) 서버: labels.json 만들기 (process_all.py 결과 폴더의 에피소드마다)
  python grasp_click.py apply ~/out/seohyeon/episodes_v4 --marks ../grasp_marks_*.json
  프레임 번호는 front 영상 전체 기준이라 구간이 바뀌어도 값을 다시 쓸 수 있다. 구간 밖이면 grasp_frame 을 null 로 두고 알린다.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def split_part(rows, part):
    if not part:
        return rows
    k, n = (int(x) for x in part.split("/"))
    if not 1 <= k <= n:
        raise SystemExit(f"[중단] --part {part}: 1/{n} ~ {n}/{n} 중 하나")
    return [rows[i] for i in np.array_split(np.arange(len(rows)), n)[k - 1]]


def cmd_frames(a):
    import cv2
    from io_utils import find_episode_folder, find_video, read_frame_times, read_records
    from sync_check import load_rows, nearest_frame
    rec = read_records(a.records)
    with open(Path(a.segments).expanduser(), newline="", encoding="utf-8") as f:
        rows = split_part(list(csv.DictReader(f)), a.part)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    index = {}
    size = (a.width, a.width * 3 // 4)
    for n, r in enumerate(rows, 1):
        eid, hand = r["episode_id"], rec[r["episode_id"]]["hand"]
        raw = Path(a.raw).expanduser() / eid.split("_")[0]
        d0, d1 = find_episode_folder(raw, eid, "exo_front"), find_episode_folder(raw, eid, f"exo_{hand}")
        t0, t1, sync = read_frame_times(d0), read_frame_times(d1), load_rows(a.sync, eid)
        s, e = int(r["start"]), min(int(r["end"]), len(t0))
        picks = sorted(set(np.searchsorted(t0, np.arange(t0[s], t0[e - 1] + 1e-6, a.step)).tolist()))
        (out / eid).mkdir(exist_ok=True)
        c0, c1, items = cv2.VideoCapture(str(find_video(d0))), cv2.VideoCapture(str(find_video(d1))), []
        for f in picks:
            c0.set(cv2.CAP_PROP_POS_FRAMES, f); ok0, im0 = c0.read()
            c1.set(cv2.CAP_PROP_POS_FRAMES, nearest_frame(t1, sync[hand], t0[f])[0]); ok1, im1 = c1.read()
            if ok0 and ok1:
                tile = np.hstack([cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (im0, im1)])
                cv2.imwrite(str(out / eid / f"f{f}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 85])
                items.append(f)
        c0.release(); c1.release()
        index[eid] = {"hand": hand, "task": rec[eid]["task"], "frames": items, "times": [float(t0[f] - t0[s]) for f in items]}
        json.dump(index, open(out / "index.json", "w"), ensure_ascii=False)
        print(f"[{n}/{len(rows)}] {eid} {hand} 사진 {len(items)}", flush=True)
    print("저장 ->", out)


class Marker:
    def __init__(self, a):
        self.dir = Path(a.frames).expanduser()
        self.index = json.load(open(self.dir / "index.json"))
        self.ids = sorted(self.index)
        self.out = self.dir / f"grasp_marks_{a.name}.json"
        self.data = json.load(open(self.out)) if self.out.exists() else {}
        self.k = next((i for i, e in enumerate(self.ids) if e not in self.data), 0)
        self.load()

    def load(self):
        self.eid = self.ids[self.k]
        info, saved = self.index[self.eid], self.data.get(self.eid)
        self.frames, self.times = info["frames"], info["times"]
        self.grasp = saved.get("grasp_frame") if saved else None
        self.i = self.frames.index(self.grasp) if self.grasp in self.frames else len(self.frames) // 3
        self.msg = "loaded saved" if saved else ""

    def draw(self):
        import cv2
        f = self.frames[self.i]
        img = cv2.imread(str(self.dir / self.eid / f"f{f}.jpg"))
        W = img.shape[1]
        panel = np.zeros((165, W, 3), np.uint8)
        x = lambda v: int(self.frames.index(v) / max(len(self.frames) - 1, 1) * (W - 40)) + 20
        cv2.line(panel, (20, 28), (W - 20, 28), (120, 120, 120), 2)
        cv2.line(panel, (x(f), 13), (x(f), 43), (255, 255, 255), 2)
        if self.grasp in self.frames:
            cv2.drawMarker(panel, (x(self.grasp), 28), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 18, 3)
        info = self.index[self.eid]
        L = [f"{self.eid} hand={info['hand']} ({self.k + 1}/{len(self.ids)}, done {len(self.data)})   frame {f}   {self.times[self.i]:.1f}s   grasp: {self.grasp}",
             "d/a: next/prev photo   e/w: +5/-5   1: set grasp (first fully gripped)",
             "s: save+next episode   b: prev episode   x: unknown+next   q: quit", self.msg]
        for n, s in enumerate(L):
            cv2.putText(panel, s, (20, 66 + 26 * n), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([img, panel])

    def save(self, unknown=False):
        self.data[self.eid] = {"unknown": True} if unknown else {"grasp_frame": self.grasp}
        json.dump(self.data, open(self.out, "w"), ensure_ascii=False, indent=1)

    def key(self, ch):
        n = len(self.frames)
        if ch == "q":
            return False
        if ch in "da":
            self.i = min(max(self.i + (1 if ch == "d" else -1), 0), n - 1)
        elif ch in "ew":
            self.i = min(max(self.i + (5 if ch == "e" else -5), 0), n - 1)
        elif ch == "1":
            self.grasp = self.frames[self.i]; self.msg = "grasp set"
        elif ch == "s":
            if self.grasp is None:
                self.msg = "press 1 first (x if unknown)"; return True
            self.save(); return self.move(1)
        elif ch == "x":
            self.save(True); return self.move(1)
        elif ch == "b":
            return self.move(-1)
        return True

    def move(self, step):
        if not 0 <= self.k + step < len(self.ids):
            self.msg = "end of list"; return True
        self.k += step; self.load(); return True


def cmd_click(a):
    import cv2
    m = Marker(a)
    cv2.namedWindow("grasp", cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow("grasp", m.draw())
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not m.key(chr(k)):
            break
    cv2.destroyAllWindows()
    print(f"표시한 에피소드 {len(m.data)}개 -> {m.out}")


def cmd_apply(a):
    from labels import check
    marks = {}
    for path in a.marks:
        for e, v in json.load(open(Path(path).expanduser())).items():
            if e in marks and marks[e] != v:
                print(f"  두 사람 표시 다름 {e}: {marks[e]} / {v} ({Path(path).name}) -> 앞 파일 값 사용")
            marks.setdefault(e, v)
    root = Path(a.root).expanduser()
    made, none, outside, bad = 0, [], [], []
    for ep in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta = json.load(open(ep / "meta.json"))
        g = marks.get(ep.name, {}).get("grasp_frame")
        if g is None:
            none.append(ep.name)
        elif not meta["exo_start_frame"] <= g < meta["exo_end_frame"]:
            outside.append((ep.name, g, meta["exo_start_frame"], meta["exo_end_frame"])); g = None
        lab = {"episode_id": meta["episode_id"], "frame_ref": "exo", "exo_start_frame": meta["exo_start_frame"],
               "exo_end_frame": meta["exo_end_frame"], "segments": [], "grasp_frame": g}
        json.dump(lab, open(ep / "labels.json", "w"), ensure_ascii=False, indent=2)
        made += 1
        errs = [x for x in check(lab, meta) if "grasp_frame 없음" not in x]
        if errs:
            bad.append((ep.name, errs))
    print(f"labels.json {made}개. grasp_frame 없음 {len(none)}: {none[:20]}\n구간 밖 {len(outside)}: {outside[:20]}\n검사 오류 {bad[:10]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--segments", required=True); f.add_argument("--raw", required=True)
    f.add_argument("--sync", required=True); f.add_argument("--out", required=True)
    f.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    f.add_argument("--part", help="n 등분한 k 번째, 예: 2/3")
    f.add_argument("--step", type=float, default=0.1, help="사진 간격 (s)")
    f.add_argument("--width", type=int, default=560)
    c = sub.add_parser("click"); c.add_argument("--frames", required=True); c.add_argument("--name", required=True)
    p = sub.add_parser("apply"); p.add_argument("root"); p.add_argument("--marks", nargs="+", required=True)
    a = ap.parse_args()
    {"frames": cmd_frames, "click": cmd_click, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
