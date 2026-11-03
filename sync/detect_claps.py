"""박수 시각 검출 (2번 싱크). 녹화 폴더 하나 -> 시작·끝 박수 시각 (그 폰의 data.jsonl 시각, s).

박수 = 두 손바닥이 맞닿는 순간. 우리 촬영의 박수는 "빠르게 모았다가 잠깐 붙인 채 멈춤" 형태라
맞닿는 순간에 손 움직임이 갑자기 멈춘다. 이를 두 단계로 찾는다.

  1) 대략 위치 (후보): MediaPipe HandLandmarker 로 프레임마다 두 손(왼손·오른손)의 손바닥 중심 거리
     (손 크기로 나눈 값)를 잰다. FAST_S 초 안에 FAR_D 이상 떨어져 있던 두 손이 CLOSE_D 이하로 모이면 후보.
     손이 겹쳐 두 손 인식이 끊기면(직전 거리 <= APPROACH_D, GAP_S 이상) 붙은 것(거리 0)으로 본다.
  2) 정밀 위치: 후보 앞 손 위치로 잡은 영역(ROI)에서 프레임 간 밝기 차이(움직임 양) E[k] 를 구하고,
     "움직이다가 멈춤" 이 가장 뚜렷한 프레임 = E[k] - E[k+1] 이 가장 큰 k 를 맞닿는 프레임으로 본다.
     정밀도는 그 폰의 약 1프레임 (30 fps 33 ms, 60 fps 17 ms).
  후보 중 어느 것이 시작·끝 박수인지는 make_sync_csv.py 가 카메라끼리 맞춰 보고 고른다.
  손이 작게 찍혀 후보를 놓친 카메라는 make_sync_csv.py --guide-out 이 짐작 시각을 주고,
  이 스크립트 --guide 로 그 시각 ± --guide-half 초 안에서 2) 만 다시 한다 (안내 검출).

  시각은 프레임 번호가 아니라 data.jsonl 프레임 시각 (폰마다 fps가 다름).
  폰이 돌아간 채 저장된 카메라는 --rot 로 회전을 준다 (예: exo_front=180).

출력: --out 폴더에 카메라·에피소드마다
  <episode>_<camera>.json  검출값 (실패도 그대로 기록)
  <episode>_<camera>.jpg   눈 확인용: 윗줄 시작 박수, 아랫줄 끝 박수, 맞닿는 프레임 앞뒤 -3..+3 (가운데 빨간 테두리)

사용 예 (episodes 환경: mediapipe·opencv 있음)
  python detect_claps.py --model hand_landmarker.task --out ~/out/신서연/claps/2026-11-03 \\
      --raw ~/data/raw/2026-11-03 --cameras exo_front exo_left exo_right ego --rot exo_front=180
  (--episodes 를 빼면 첫 카메라 폴더의 T*_E* 전부)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

PALM = [0, 5, 9, 13, 17]          # 손목 + 손가락 뿌리 4개
ROTS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
WIDTH = 640                       # 손 인식용 축소 크기 (긴 변, px)
APPROACH_D = 2.5                  # 이 거리(손 크기 배수) 이하에서 인식이 끊기면 손이 겹친 것으로 봄
GAP_S = 0.15                      # 그 끊김이 이 시간 이상일 때만
CLOSE_D, FAR_D, FAST_S = 1.6, 3.0, 0.6   # FAST_S 초 안에 FAR_D 이상 -> CLOSE_D 이하로 다가오면 박수 후보
LOOK_BACK_S = 0.5                 # ROI 를 잡는 후보 앞 구간
SEARCH_S = (0.25, 0.35)           # 정밀 위치를 찾는 구간: 후보 - 앞, 후보 + 뒤


def frame_times(jsonl):
    """data.jsonl 의 영상 프레임 시각 (T,) s. 3번 io_utils.read_jsonl_frame_times 와 같은 규칙."""
    ts = []
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "frames" in o:
            ts.append(float(o["time"] if "time" in o else o["frames"][0]["time"]))
    return np.array(ts)


def read_frames(video, rot):
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if ROTS[rot] is not None:
            f = cv2.rotate(f, ROTS[rot])
        h, w = f.shape[:2]
        s = WIDTH / max(h, w)
        yield cv2.resize(f, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    cap.release()


def landmarker(model):
    from mediapipe.tasks.python import vision, BaseOptions
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opts)


def track(video, t, model, rot):
    """프레임마다 손 인식. 반환: frames(BGR 축소), hands[k] = [(21,2) px], d (T,) 손바닥 거리."""
    import mediapipe as mp
    frames, hands, d = [], [], []
    with landmarker(model) as lm:
        for i, f in enumerate(read_frames(video, rot)):
            if i >= len(t):
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, int(round((t[i] - t[0]) * 1000)))
            H, W = f.shape[:2]
            hs = [np.array([[p.x * W, p.y * H] for p in hl]) for hl in r.hand_landmarks]
            sides = [h[0].category_name for h in r.handedness]
            di = np.nan
            if len(hs) == 2 and sides[0] != sides[1]:       # 같은 손을 두 번 잡은 경우 제외
                scale = np.mean([np.linalg.norm(h[9] - h[0]) for h in hs])
                if scale > 1e-6:
                    di = float(np.linalg.norm(hs[0][PALM].mean(0) - hs[1][PALM].mean(0)) / scale)
            frames.append(f); hands.append(hs); d.append(di)
    return frames, hands, np.array(d)


def merged_distance(t, d):
    """두 손이 가까운 상태(<= APPROACH_D)에서 바로 이어지는 인식 끊김(GAP_S 이상)은 손이 겹친 것으로 보고 0."""
    de, T = d.copy(), len(d)
    k = 1
    while k < T:
        if not np.isfinite(d[k]) and np.isfinite(d[k - 1]) and d[k - 1] <= APPROACH_D:
            j = k
            while j < T and not np.isfinite(d[j]):
                j += 1
            if t[min(j, T - 1)] - t[k] >= GAP_S:
                de[k:j] = 0.0
            k = j
        k += 1
    return de


def candidates(t, d):
    """박수 후보: 거리가 CLOSE_D 아래로 내려가는 프레임 중, 직전 FAST_S 초 안에 FAR_D 이상 떨어져 있던 것."""
    de = merged_distance(t, d)
    out = []
    for k in range(1, len(de)):
        if not (np.isfinite(de[k]) and de[k] <= CLOSE_D):
            continue
        prev = de[k - 1]
        if np.isfinite(prev) and prev <= CLOSE_D:
            continue
        lo = np.searchsorted(t, t[k] - FAST_S)
        before = de[lo:k]
        if np.any(np.isfinite(before)) and np.nanmax(before) >= FAR_D:
            out.append(k)
    return out, de


def hand_boxes(hands):
    """프레임마다 보이는 손 전체를 감싸는 상자 [x0,y0,x1,y1] (손 없으면 None). json 에 저장해 2단계에서 재사용."""
    out = []
    for hs in hands:
        if not hs:
            out.append(None); continue
        p = np.vstack(hs)
        out.append([round(float(v), 1) for v in (*p.min(0), *p.max(0))])
    return out


def roi_from(boxes, lo, hi, W, H):
    """boxes[lo:hi] 를 모두 감싸는 정사각형 ROI (50% 여유). 손이 없으면 None."""
    bs = [b for b in boxes[max(0, lo):max(0, hi)] if b is not None]
    if not bs:
        return None
    b = np.array(bs)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 0.75 * max(x1 - x0, y1 - y0, 20)
    return [int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))]


def stop_frame(frames, t, a, b, roi):
    """프레임 a..b 중 '움직이다 멈춤' 이 가장 뚜렷한 k: E[k] - E[k+1] 최대 (E = ROI 프레임 간 밝기 차이)."""
    T = len(frames)
    a, b = max(1, a), min(T - 2, b)
    if b <= a or roi is None:
        return None
    xa, ya, xb, yb = roi
    gray = {k: cv2.cvtColor(frames[k][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
            for k in range(a - 1, b + 2)}
    E = {k: float(np.mean(np.abs(gray[k] - gray[k - 1]))) for k in range(a, b + 2)}
    score = {k: E[k] - E[k + 1] for k in range(a, b + 1)}
    k = max(score, key=score.get)
    return {"frame": int(k), "time": float(t[k]), "drop": round(score[k], 3),
            "E_k": round(E[k], 3), "E_next": round(E[k + 1], 3), "roi": roi,
            "E": {str(kk): round(v, 3) for kk, v in E.items()}}


def refine(frames, boxes, t, k0):
    """1단계: 손 인식 후보 k0 근처 정밀 위치. ROI = 후보 직전 LOOK_BACK_S 초의 손 영역."""
    fps = (len(t) - 1) / (t[-1] - t[0])
    H, W = frames[0].shape[:2]
    roi = roi_from(boxes, k0 - int(round(LOOK_BACK_S * fps)), k0 + 1, W, H)
    p = stop_frame(frames, t, k0 - int(round(SEARCH_S[0] * fps)), k0 + int(round(SEARCH_S[1] * fps)), roi)
    if p:
        p["candidate_frame"] = int(k0)
    return p


def refine_guided(frames, boxes, t, t_exp, half_s):
    """2단계: 다른 카메라 박수로 짐작한 시각 t_exp ± half_s 안에서 정밀 위치.
    ROI = t_exp ± 1 s 의 손 영역 (없으면 에피소드 전체의 손 영역)."""
    H, W = frames[0].shape[:2]
    k_exp = int(np.argmin(np.abs(t - t_exp)))
    fps = (len(t) - 1) / (t[-1] - t[0])
    roi = roi_from(boxes, k_exp - int(fps), k_exp + int(fps) + 1, W, H) or roi_from(boxes, 0, len(boxes), W, H)
    a = int(np.searchsorted(t, t_exp - half_s))
    b = int(np.searchsorted(t, t_exp + half_s)) - 1
    p = stop_frame(frames, t, a, b, roi)
    if p:
        p.update(guided=True, expected_time=float(t_exp))
    return p


def contact_sheet(frames, picks, path):
    rows = []
    for p in picks:
        tiles = []
        for k in range(-3, 4):
            if p is None or not (0 <= p["frame"] + k < len(frames)):
                tiles.append(np.zeros((160, 160, 3), np.uint8)); continue
            xa, ya, xb, yb = p["roi"]
            f = cv2.resize(frames[p["frame"] + k][ya:yb, xa:xb], (160, 160))
            cv2.putText(f, str(p["frame"] + k), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if k == 0:
                cv2.rectangle(f, (0, 0), (159, 159), (0, 0, 255), 3)
            tiles.append(f)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))


def detect(folder, model, rot):
    folder = Path(folder)
    t = frame_times(folder / "data.jsonl")
    frames, hands, d = track(folder / "data.mov", t, model, rot)
    T = len(frames)
    t = t[:T]
    c, de = candidates(t, d)
    boxes = hand_boxes(hands)
    claps = [p for p in (refine(frames, boxes, t, k) for k in c) if p is not None]
    res = {"folder": str(folder), "n_frames": T, "n_ts": int(len(frame_times(folder / "data.jsonl"))),
           "t0": float(t[0]), "t_last": float(t[-1]),
           "fps": float(round((T - 1) / (t[-1] - t[0]), 3)), "rotation": rot, "candidates": c,
           "claps": claps,     # 후보 전부 (시작·끝 선택은 make_sync_csv.py 가 카메라끼리 맞춰 본다)
           "signal_d": [None if np.isnan(x) else round(float(x), 3) for x in d],
           "signal_d_merged": [None if np.isnan(x) else round(float(x), 3) for x in de],
           "signal_n_hands": [len(h) for h in hands], "hand_boxes": boxes}
    return res, frames


def run_guided(raw, out, guide, rots, half_s):
    """guide.json {episode: {camera: [짐작 시각, ...]}, "_half": {camera: 반폭}} -> 각 json 에 "guided" 목록을 채운다
    (손 인식 다시 안 함). "_half" 에 있는 카메라는 그 반폭으로 넓게 찾는다."""
    halves = guide.pop("_half", {})
    for ep, cams in guide.items():
        for cam, times in cams.items():
            h = halves.get(cam, half_s)
            js = out / f"{ep}_{cam}.json"
            res = json.load(open(js))
            if "hand_boxes" not in res:
                continue
            folder = raw / cam / ep
            t = frame_times(folder / "data.jsonl")
            frames = list(read_frames(folder / "data.mov", rots.get(cam, 0)))[:len(t)]
            t = t[:len(frames)]
            new = [p for p in (refine_guided(frames, res["hand_boxes"], t, te, h) for te in times) if p]
            old = [p for p in res.get("guided", [])
                   if all(abs(p["expected_time"] - q["expected_time"]) > 1e-3 for q in new)]
            res["guided"] = sorted(old + new, key=lambda p: p["time"])
            json.dump(res, open(js, "w"))
            if res["guided"]:
                contact_sheet(frames, res["guided"], out / f"{ep}_{cam}_guided.jpg")
            got = ", ".join(f"f{p['frame']}(drop {p['drop']})" for p in new) or "없음"
            print(f"{ep} {cam}: 안내 검출 {got}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="촬영 날짜 폴더 (카메라 폴더들이 들어 있음)")
    ap.add_argument("--cameras", nargs="+", required=True, help="카메라 폴더 이름들")
    ap.add_argument("--episodes", nargs="*", help="에피소드 ID (생략하면 첫 카메라 폴더의 T*_E* 전부)")
    ap.add_argument("--rot", nargs="*", default=[], help="카메라=각도 (0/90/180/270, 시계 방향)")
    ap.add_argument("--model", required=True, help="hand_landmarker.task 경로")
    ap.add_argument("--out", required=True)
    ap.add_argument("--redo", action="store_true", help="이미 있는 결과도 다시 계산")
    ap.add_argument("--guide", help="2단계: make_sync_csv.py --guide-out 파일. 주면 1단계 대신 안내 검출만 한다")
    ap.add_argument("--guide-half", type=float, default=0.3, help="안내 검출 탐색 반폭 (s)")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rots = {k: int(v) for k, v in (s.split("=") for s in a.rot)}
    if a.guide:
        return run_guided(raw, out, json.load(open(a.guide)), rots, a.guide_half)
    eps = a.episodes or sorted(p.name for p in (raw / a.cameras[0]).glob("T*_E*") if p.is_dir())
    for ep in eps:
        for cam in a.cameras:
            js = out / f"{ep}_{cam}.json"
            folder = raw / cam / ep
            if js.exists() and not a.redo:
                continue
            if not (folder / "data.mov").exists():
                json.dump({"folder": str(folder), "missing": True}, open(js, "w"))
                print(f"{ep} {cam}: 폴더/영상 없음", flush=True)
                continue
            try:
                res, frames = detect(folder, a.model, rots.get(cam, 0))
            except Exception as e:  # 실패도 기록 (지우지 않는다)
                json.dump({"folder": str(folder), "error": repr(e)}, open(js, "w"))
                print(f"{ep} {cam}: 오류 {e!r}", flush=True)
                continue
            json.dump(res, open(js, "w"))
            if res["claps"]:
                contact_sheet(frames, res["claps"], out / f"{ep}_{cam}.jpg")
            got = ", ".join(f"f{p['frame']}(drop {p['drop']})" for p in res["claps"]) or "없음"
            print(f"{ep} {cam}: fps={res['fps']} 박수 후보 {got}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
