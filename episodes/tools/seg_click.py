"""자동으로 동작 구간을 못 나눈 에피소드(find_segment.py 의 check=1)의 시작·끝을 사람이 표시한다.

손을 테이블 가장자리에 둔 채 박수·대기를 하면 카메라가 손을 계속 봐서 구간이 나뉘지 않는다.
시작 = 손이 테이블 위로 올라오는 첫 프레임 (박수·대기 뒤), 끝 = 손이 테이블에서 내려가기 전 마지막 프레임.

1) 서버: 사진 뽑기 (0.2초 간격, front + 작업 손 쪽 카메라)
  python seg_click.py frames --segments ~/out/seohyeon/segments.csv --raw ~/data/raw --sync ../../sync/sync.csv --out ~/out/seohyeon/seg_frames
  로컬로 받기: scp -P 10000 -r coss37@155.230.81.91:~/out/seohyeon/seg_frames ~/Downloads/

2) 로컬: 표시
  python seg_click.py click --frames ~/Downloads/seg_frames

  d / a      다음 / 이전 사진 (0.2초)      e / w  다음 / 이전 5장
  1 / 2      지금 사진을 시작 / 끝으로
  s          저장하고 다음 에피소드        b  이전 에피소드
  x          동작을 알 수 없음으로 저장    q  끝내기
  저장할 때마다 <frames>/seg_marks.json 에 바로 쓴다. 다시 켜면 안 한 에피소드부터.

3) 합치기: 자동 구간 + 표시한 구간 -> 일괄 처리가 읽는 표
  python seg_click.py merge --segments ~/out/seohyeon/segments.csv --marks ../seg_marks.json --out ../segments_final.csv
  열: episode_id, start, end(미포함), method(auto/manual). 없는 에피소드는 표에 넣지 않는다.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def read_segments(path):
    with open(Path(path).expanduser(), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_frames(a):
    import cv2
    from io_utils import find_episode_folder, find_video, read_frame_times
    from sync_check import load_rows, nearest_frame, read_frame
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_segments(a.segments) if r.get("check") == "1"]
    index = {}
    for n, r in enumerate(rows, 1):
        eid, hand = r["episode_id"], r["hand"]
        raw = Path(a.raw).expanduser() / eid.split("_")[0]
        d0, d1 = find_episode_folder(raw, eid, "exo_front"), find_episode_folder(raw, eid, f"exo_{hand}")
        t0, t1, sync = read_frame_times(d0), read_frame_times(d1), load_rows(a.sync, eid)
        cap = cv2.VideoCapture(str(find_video(d0)))
        (out / eid).mkdir(exist_ok=True)
        picks = np.searchsorted(t0, np.arange(t0[0], t0[-1], a.step)).tolist()
        items = []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im0 = cap.read()
            if not ok:
                continue
            j = nearest_frame(t1, sync[hand], t0[f])[0]
            im1 = read_frame(find_video(d1), j)
            if im1 is None:
                continue
            tiles = [cv2.resize(x, (a.width, a.width * 3 // 4), interpolation=cv2.INTER_AREA) for x in (im0, im1)]
            cv2.imwrite(str(out / eid / f"f{f}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
            items.append(f)
        cap.release()
        index[eid] = {"hand": hand, "frames": items, "start": int(r["start"]), "end": int(r["end"]), "runs": int(r["runs"]), "t0": float(t0[0]),
                      "times": [float(t0[f] - t0[0]) for f in items]}
        json.dump(index, open(out / "index.json", "w"), ensure_ascii=False)
        print(f"[{n}/{len(rows)}] {eid} {hand} 사진 {len(items)}", flush=True)
    print("저장 ->", out)


class Marker:
    def __init__(self, a):
        self.a, self.dir = a, Path(a.frames).expanduser()
        self.index = json.load(open(self.dir / "index.json"))
        self.ids = sorted(self.index)
        self.out = self.dir / "seg_marks.json"
        self.data = json.load(open(self.out)) if self.out.exists() else {}
        self.k = next((i for i, e in enumerate(self.ids) if e not in self.data), 0)
        self.load()

    def load(self):
        self.eid = self.ids[self.k]
        info, saved = self.index[self.eid], self.data.get(self.eid)
        self.frames, self.times = info["frames"], info["times"]
        self.start = saved.get("start") if saved else None
        self.end = saved.get("end") if saved else None
        anchor = self.start if self.start is not None else info["start"]
        self.i = int(np.argmin(np.abs(np.array(self.frames) - anchor)))
        self.msg = "loaded saved" if saved else ""

    def draw(self):
        import cv2
        f = self.frames[self.i]
        img = cv2.imread(str(self.dir / self.eid / f"f{f}.jpg"))
        H, W = img.shape[:2]
        panel = np.zeros((190, W, 3), np.uint8)
        x = lambda v: int(np.searchsorted(self.frames, v) / max(len(self.frames) - 1, 1) * (W - 40)) + 20
        cv2.line(panel, (20, 30), (W - 20, 30), (120, 120, 120), 2)
        cv2.line(panel, (x(f), 15), (x(f), 45), (255, 255, 255), 2)
        if self.start is not None:
            cv2.drawMarker(panel, (x(self.start), 30), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 18, 3)
        if self.end is not None:
            cv2.drawMarker(panel, (x(self.end), 30), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 18, 3)
        info = self.index[self.eid]
        L = [f"{self.eid} hand={info['hand']} ({self.k + 1}/{len(self.ids)}, done {len(self.data)})   frame {f}   t={self.times[self.i]:.1f}s",
             f"start: {self.start}   end: {self.end}   (left: front, right: {info['hand']})",
             "d/a: next/prev photo   e/w: +5/-5   1: set start   2: set end",
             "s: save+next episode   b: prev episode   x: unknown+next   q: quit", self.msg]
        for n, s in enumerate(L):
            cv2.putText(panel, s, (20, 70 + 26 * n), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([img, panel])

    def save(self, unknown=False):
        self.data[self.eid] = {"unknown": True} if unknown else {"start": self.start, "end": self.end}
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
            self.start = self.frames[self.i]; self.msg = "start set"
        elif ch == "2":
            self.end = self.frames[self.i]; self.msg = "end set"
        elif ch == "s":
            if self.start is None or self.end is None or self.start >= self.end:
                self.msg = "need start < end"; return True
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
    cv2.namedWindow("segment", cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow("segment", m.draw())
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not m.key(chr(k)):
            break
    cv2.destroyAllWindows()
    print(f"표시한 에피소드 {len(m.data)}개 -> {m.out}")


def cmd_merge(a):
    rows, marks = read_segments(a.segments), json.load(open(Path(a.marks).expanduser()))
    out, missing, unknown = [], [], []
    for r in rows:
        e = r["episode_id"]
        if r.get("error"):
            missing.append(e)
        elif r.get("check") != "1":
            out.append({"episode_id": e, "start": r["start"], "end": r["end"], "method": "auto"})
        elif e in marks and not marks[e].get("unknown"):
            out.append({"episode_id": e, "start": marks[e]["start"], "end": int(marks[e]["end"]) + 1, "method": "manual"})
        elif e in marks:
            unknown.append(e)
        else:
            missing.append(e)
    p = Path(a.out).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "start", "end", "method"]); w.writeheader(); w.writerows(out)
    n = sum(r["method"] == "manual" for r in out)
    print(f"저장 {p}: 자동 {len(out) - n}, 사람 표시 {n}, 알 수 없음 {unknown}, 빠짐 {missing}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--segments", required=True); f.add_argument("--raw", required=True)
    f.add_argument("--sync", required=True); f.add_argument("--out", required=True)
    f.add_argument("--step", type=float, default=0.2, help="사진 간격 (s)")
    f.add_argument("--width", type=int, default=560)
    c = sub.add_parser("click"); c.add_argument("--frames", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--segments", required=True); m.add_argument("--marks", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"frames": cmd_frames, "click": cmd_click, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
