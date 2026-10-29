import csv, json
from pathlib import Path
import numpy as np


def read_ts(path):
    """프레임 시각 CSV -> (T,) 초. 헤더가 있으면 'timestamp'/'t'/'time' 열, 없으면 마지막 열."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    head = rows[0]
    try:
        float(head[-1]); data = rows; col = -1
    except ValueError:
        low = [h.strip().lower() for h in head]
        col = next((low.index(k) for k in ("timestamp", "t", "time", "t_s") if k in low), -1)
        data = rows[1:]
    return np.array([float(r[col]) for r in data])


def read_jsonl_frame_times(path):
    """Spectacular Rec 의 data.jsonl -> 영상 프레임별 시각 (T,) 초.

    프레임 줄은 "frames" 키를 가진 줄로 본다 (센서 줄은 "sensor" 키). 시각은 그 줄의 "time"
    (폰 부팅 후 경과 초, 폰마다 기준이 다르므로 sync.csv 의 시간 차로 맞춘다).
    2026-10-19 녹화로 확인: 프레임 줄 수 == ffprobe 프레임 수, "number" 0부터 연속, 간격 일정.
    """
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "frames" not in o:
                continue
            fr = o["frames"][0] if o["frames"] else {}
            ts.append(float(o["time"] if "time" in o else fr["time"]))
    if not ts:
        raise ValueError(f"{path}: 'frames' 줄이 없음. data.jsonl 형식 확인 필요")
    return np.array(ts)


def read_frame_times(folder):
    """녹화 폴더 -> 프레임 시각. data.jsonl 우선, 없으면 *_ts.csv / ts.csv."""
    folder = Path(folder)
    if (folder / "data.jsonl").exists():
        return read_jsonl_frame_times(folder / "data.jsonl")
    cands = sorted(folder.glob("*ts.csv"))
    if cands:
        return read_ts(cands[0])
    raise FileNotFoundError(f"{folder}: data.jsonl 도 *_ts.csv 도 없음")


def find_video(folder):
    folder = Path(folder)
    for n in ("data.mov", "data.mp4"):
        if (folder / n).exists():
            return folder / n
    vids = sorted(list(folder.glob("*.mov")) + list(folder.glob("*.mp4")))
    if not vids:
        raise FileNotFoundError(f"{folder}: 영상 없음")
    return vids[0]


class SyncError(Exception):
    pass


def read_sync_row(path, episode_id, s0=None, s1=None, camera="ego"):
    """2번 sync.csv 에서 (에피소드, 카메라) 행을 읽는다.

    열: episode_id, camera, offset_start_s, offset_end_s, drift_s, s0, s1, usable, note
      camera : 기준 카메라(정면 exo)에 맞출 카메라 이름 (left, right, ego). 기준 카메라 자신은 행이 없어도 됨.
      offset : 기준 카메라 박수 시각 - 이 카메라 박수 시각 (s)
      s0, s1 : 이 카메라 기준 시작·끝 박수 시각 (s). 드리프트가 있을 때 필요.
    camera 열이 없는 예전 형식이면 모든 행을 ego 로 본다.

    s0, s1 우선순위: 1) 함수 인자  2) sync.csv 의 s0/s1 열  3) 드리프트 0 이면 필요 없음.
    셋 다 아니면 SyncError. 행이 없거나 usable=0 이어도 SyncError (다른 행으로 대체하지 않는다).
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    has_cam = bool(rows) and "camera" in rows[0]
    row = next((r for r in rows if r.get("episode_id") == episode_id
                and (not has_cam or r.get("camera", "").strip() == camera)), None)
    if row is None or (not has_cam and camera != "ego"):
        raise SyncError(f"{path} 에 {episode_id} / {camera} 행이 없음")
    if str(row.get("usable", "1")).strip() in ("0", "false", "False", ""):
        raise SyncError(f"{episode_id} / {camera}: usable=0 -> 건너뜀")
    off0, off1 = float(row["offset_start_s"]), float(row["offset_end_s"])
    if s0 is None and row.get("s0"):
        s0 = float(row["s0"])
    if s1 is None and row.get("s1"):
        s1 = float(row["s1"])
    if s0 is None or s1 is None:
        if abs(off1 - off0) > 1e-9:
            raise SyncError(f"{episode_id} / {camera}: 드리프트가 있는데 박수 시각 s0/s1 이 없음. "
                            "sync.csv 에 s0, s1 열을 채워 달라고 2번에게 요청")
        s0, s1 = 0.0, 1.0  # 오프셋이 일정하면 값이 쓰이지 않는다
    return {"s0": s0, "s1": s1, "off0": off0, "off1": off1}
