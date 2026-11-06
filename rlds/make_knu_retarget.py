"""리타게팅 결과(관절 궤적 npz) + exo_front 영상 -> OpenVLA 학습용 에피소드 묶음 (RLDS 만들기 전 단계).

결정 (PROJECT_CONTEXT.md 5절 「OpenVLA 학습」):
  - 이미지: exo_front 1장, 256x256 RGB (OpenVLA 로더가 224로 다시 줄임)
  - 간격: 10 Hz. exo 영상 fps 에서 가장 가까운 정수 간격으로 고름 (60 fps -> 6장마다, 30 fps -> 3장마다)
  - 상태 state[k]  = q[idx[k]]       (현재 7관절 각도, rad)
  - 행동 action[k] = q[idx[k + 1]]   (다음 스텝 7관절 각도, rad). 마지막 스텝은 다음이 없어 뺀다
  - 지시문: 태스크마다 한 문장 (INSTRUCTIONS)

입력
  --retarget  <알고리즘 폴더>/<에피소드ID>.npz   (retarget/README.md 출력 npz, q 는 exo_front 시간축)
  --episodes  <에피소드 폴더>/<에피소드ID>/meta.json, exo.mov   (3번 process_episode 형식)
              q[i] 는 exo_front 영상 프레임 meta["exo_start_frame"] + i 에 해당
출력
  --out/<에피소드ID>.npz : image (N,256,256,3) uint8, state (N,7), action (N,7), instruction, frames (N,) 원본 프레임 번호
  --out/summary.csv      : 에피소드별 처리 결과 (빼거나 실패한 것도 이유와 함께 남김)

영상은 저장된 그대로 읽는다 (CAP_PROP_ORIENTATION_AUTO = 0, sync/protocol.md 8절).

  python make_knu_retarget.py --retarget ~/out/hyojeong/retarget/<실행>/C_eaik \\
      --episodes ~/out/seohyeon/episodes_clap --out ~/vla/data/knu_retarget_src
"""
import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np

INSTRUCTIONS = {"T1": "pick up the cup", "T2": "stack the block", "T3": "move the block"}
FIELDS = ["episode_id", "status", "reason", "hand", "exo_fps", "stride", "q_frames", "steps", "start_frame", "end_frame"]


def read_frames(video, frame_ids, size):
    """frame_ids(오름차순) 위치의 프레임을 RGB size x size 로 읽는다."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video}")
    want = set(int(f) for f in frame_ids)
    last = max(want)
    out, f = {}, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(want))
    f = min(want)
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        if f in want:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out[f] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        f += 1
    cap.release()
    missing = [i for i in frame_ids if int(i) not in out]
    if missing:
        raise RuntimeError(f"프레임 {len(missing)}개를 못 읽음 (첫 번째 {missing[0]})")
    return np.stack([out[int(i)] for i in frame_ids])


def convert(npz_path, ep_dir, out_dir, hz, size):
    eid = npz_path.stem
    row = {"episode_id": eid}
    meta_f, video = ep_dir / "meta.json", ep_dir / "exo.mov"
    if not meta_f.exists() or not video.exists():
        return {**row, "status": "skip", "reason": "에피소드 폴더에 meta.json 또는 exo.mov 없음"}
    meta = json.loads(meta_f.read_text())
    z = np.load(npz_path, allow_pickle=True)
    q = z["q"].astype(np.float32)
    fps = float(meta["exo_fps"])
    stride = max(1, int(round(fps / hz)))
    start = int(meta["exo_start_frame"])
    row.update(hand=meta.get("hand", ""), exo_fps=fps, stride=stride, q_frames=len(q), start_frame=start,
               end_frame=start + len(q))
    if not np.isfinite(q).all():
        return {**row, "status": "skip", "reason": "q 에 NaN/inf"}
    idx = np.arange(0, len(q), stride)
    if len(idx) < 2:
        return {**row, "status": "skip", "reason": f"스텝이 {len(idx)}개뿐 (행동을 만들 수 없음)"}
    task = eid.split("_")[0]
    images = read_frames(video, start + idx, size)
    np.savez_compressed(out_dir / f"{eid}.npz",
                        image=images[:-1], state=q[idx[:-1]], action=q[idx[1:]],
                        frames=(start + idx[:-1]).astype(np.int32), instruction=INSTRUCTIONS[task],
                        episode_id=eid, hand=meta.get("hand", ""))
    return {**row, "status": "ok", "reason": "", "steps": len(idx) - 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retarget", required=True, help="알고리즘 결과 폴더 (예: .../C_eaik)")
    ap.add_argument("--episodes", required=True, help="3번 에피소드 폴더들의 상위 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--ids", nargs="*", help="이 에피소드만")
    a = ap.parse_args()
    src, eps, out = Path(a.retarget).expanduser(), Path(a.episodes).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("T*_E*.npz"))
    if a.ids:
        files = [f for f in files if f.stem in set(a.ids)]
    rows = []
    for i, f in enumerate(files, 1):
        try:
            r = convert(f, eps / f.stem, out, a.hz, a.size)
        except Exception as e:  # 실패도 지우지 않고 기록
            r = {"episode_id": f.stem, "status": "error", "reason": str(e)}
        rows.append(r)
        print(f"[{i}/{len(files)}] {f.stem} {r['status']} {r.get('steps', '')} {r.get('reason', '')}", flush=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"끝. ok {len(ok)} / {len(rows)}, 스텝 합 {sum(r['steps'] for r in ok)} -> {out}")


if __name__ == "__main__":
    main()
