"""손 표시용 박수 그림 (2번 싱크). 박수 자동 검출이 안 되는 태스크에서 fit_clock.py 기준점을 만들 때 쓴다.

에피소드 몇 개를 골라 카메라마다 시작 박수 근처 프레임을 번호와 함께 한 장에 늘어놓는다.
사람이 그림을 보고 "두 손바닥이 처음 맞닿은 프레임" 번호를 manual CSV 에 적는다 -> fit_clock.py --manual.

창 위치
  기준 카메라: 1단계 박수 후보(drop >= --min-drop) 중 가장 이른 것 ± --half-ref 초. 후보가 없으면 녹화 시작 후 1~5 s.
  다른 카메라: 기준 창 중심을 대략 시계 차(--offsets, make_sync_csv 로그의 대표값)로 옮긴 곳 ± --half 초.
  프레임 번호 = 영상 디코딩 순서 = data.jsonl 프레임 순서 (0부터).

출력 (--out 폴더)
  <episode>_<camera>.jpg  프레임 그림 (번호는 왼쪽 위)
  manual_<task>.csv       적을 표 (frame 칸만 비어 있음)

사용 예
  python label_sheets.py --raw ~/data/raw/T2 --claps ~/out/신서연/claps_T2 --auto 8 \\
      --offsets exo_left=-78403.105 exo_right=-75556.881 ego=26884.009 --out ~/out/신서연/label_T2
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import cv2

CAMS = ["exo_front", "exo_left", "exo_right", "ego"]
TW, TH, COLS = 320, 240, 8


def frame_times(jsonl):
    ts = []
    for line in open(jsonl):
        line = line.strip()
        if line:
            o = json.loads(line)
            if "frames" in o:
                ts.append(float(o["time"] if "time" in o else o["frames"][0]["time"]))
    return np.array(ts)


def sheet(video, idx, path, title):
    want = set(idx)
    cap = cv2.VideoCapture(str(video))
    tiles, i = {}, 0
    while i <= max(idx):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            t = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (92, 30), (0, 0, 0), -1)
            cv2.putText(t, str(i), (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles[i] = t
        i += 1
    cap.release()
    idx = [k for k in idx if k in tiles]
    rows = (len(idx) + COLS - 1) // COLS
    img = np.full((40 + rows * TH, COLS * TW, 3), 255, np.uint8)
    cv2.putText(img, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for n, k in enumerate(idx):
        r, c = divmod(n, COLS)
        img[40 + r * TH: 40 + (r + 1) * TH, c * TW: (c + 1) * TW] = tiles[k]
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def first_cand(claps_dir, ep, min_drop):
    p = Path(claps_dir) / f"{ep}_exo_front.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    # 시작 박수만: 영상 앞쪽 절반에 있는 후보
    mid = (r["t0"] + r["t_last"]) / 2
    c = [q["time"] for q in r.get("claps", []) + r.get("guided", []) if q["drop"] >= min_drop and q["time"] < mid]
    return min(c) if c else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--claps", required=True, help="detect_claps.py 1단계 결과 폴더 (기준 카메라 후보 위치용)")
    ap.add_argument("--offsets", nargs="+", required=True, help="카메라=대략 시계 차 (기준 - 이 카메라, s)")
    ap.add_argument("--episodes", nargs="*", help="에피소드 ID (없으면 --auto)")
    ap.add_argument("--auto", type=int, default=8, help="기준 카메라 후보가 있는 에피소드 중 고르게 이만큼")
    ap.add_argument("--min-drop", type=float, default=1.0)
    ap.add_argument("--half-ref", type=float, default=0.8, help="기준 카메라 창 반폭 (s)")
    ap.add_argument("--half", type=float, default=0.9, help="다른 카메라 창 반폭 (s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw, out = Path(a.raw).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    offs = {k: float(v) for k, v in (s.split("=") for s in a.offsets)}
    task = raw.name

    eps = a.episodes
    if not eps:
        all_eps = sorted(p.name for p in (raw / "exo_front").glob("T*_E*") if p.is_dir())
        ok = [ep for ep in all_eps if first_cand(a.claps, ep, a.min_drop) is not None]
        pick = np.linspace(0, len(ok) - 1, min(a.auto, len(ok))).round().astype(int)
        eps = [ok[i] for i in sorted(set(pick))]
    print("에피소드:", " ".join(eps))

    rows = []
    for ep in eps:
        tr = frame_times(raw / "exo_front" / ep / "data.jsonl")
        c = first_cand(a.claps, ep, a.min_drop)
        center, half = (c, a.half_ref) if c is not None else (tr[0] + 3.0, 2.0)
        for cam in CAMS:
            folder = raw / cam / ep
            if not (folder / "data.mov").exists():
                print(f"{ep} {cam}: 폴더 없음"); continue
            t = frame_times(folder / "data.jsonl")
            cc = center - offs.get(cam, 0.0)        # 이 카메라 시각
            h = half if cam == "exo_front" else a.half
            idx = [int(i) for i in np.where((t >= cc - h) & (t <= cc + h))[0]]
            if not idx:
                print(f"{ep} {cam}: 창이 영상 밖 ({cc - t[0]:.1f} s)"); continue
            sheet(folder / "data.mov", idx, out / f"{ep}_{cam}.jpg",
                  f"{ep} {cam}  frames {idx[0]}-{idx[-1]}  (start clap: first frame palms touch)")
            rows.append({"episode_id": ep, "camera": cam, "clap": "start", "frame": "", "note": ""})
            print(f"{ep} {cam}: 프레임 {idx[0]}~{idx[-1]} ({len(idx)}장)")
    with open(out / f"manual_{task}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["episode_id", "camera", "clap", "frame", "note"]); w.writeheader(); w.writerows(rows)
    print("표:", out / f"manual_{task}.csv")


if __name__ == "__main__":
    sys.exit(main())
