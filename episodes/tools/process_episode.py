"""원본 촬영물 -> episodes/<ID>/ 형식 (역할 가이드 4~6, 8, 9단계). exo 3대 삼각측량 버전.

입력: --raw 아래에 프로토콜 이름 규칙대로 바꾼 녹화 폴더 (폴더 안 파일은 앱이 만든 그대로)
  <raw>/T1_E001_exo_front/  data.mov, data.jsonl, ...   <- 기준 카메라 (라벨·프레임 번호의 'exo')
  <raw>/T1_E001_exo_left/
  <raw>/T1_E001_exo_right/
  <raw>/T1_E001_ego/
보정값: exo_calib.json 의 cameras.{front,left,right} (테이블 좌표계 공통), ref_camera
시간 차: sync.csv 의 (episode_id, camera) 행. camera = left, right, ego (기준 카메라 대비)

예:
  python process_episode.py --raw ~/raw --id T1_E001 --task "컵 잡기" --collector 홍길동 \\
      --hand right --tags "큰 이동|정밀 구간" --split train --start 95 --end 410 \\
      --calib ../../sync/exo_calib.json --sync ../../sync/sync.csv --out ~/out/본인이름/episodes
--start/--end: 기준 카메라 프레임 번호. 시작 = 초기 자세 첫 프레임, 끝 = 종료 자세 도달 프레임 (labeling_rules.md).
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np

import extractors as ex
from hand_math import sync_ego_to_exo, wrist_pose_series
from multiview import load_calibs, align_to_ref, triangulate_series, tri_report, DEFAULT_TRI
from constants import TASKS, CASES, SPLITS
from io_utils import read_frame_times, find_video, read_sync_row, SyncError
from quality import correct, settings_string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="녹화 폴더들(<ID>_exo_front 등)이 있는 상위 폴더")
    ap.add_argument("--id", required=True)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--collector", required=True)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--tags", default="", help="쉼표 또는 | 구분: " + ",".join(CASES))
    ap.add_argument("--split", required=True, choices=list(SPLITS) + list(SPLITS.values()),
                    help="train/val (학습/검증도 받음). 예비 번호는 대체한 원래 에피소드의 용도")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sync", required=True)
    ap.add_argument("--out", required=True, help="episodes 루트 (~/project 밖 경로 권장)")
    ap.add_argument("--start", type=int, required=True, help="에피소드 구간 시작 (기준 카메라 프레임)")
    ap.add_argument("--end", type=int, required=True, help="끝 (기준 카메라 프레임, 미포함)")
    ap.add_argument("--max-reproj-px", type=float, default=DEFAULT_TRI["max_reproj_px"])
    ap.add_argument("--mirrored", action="store_true", help="입력 영상이 좌우 반전된 경우")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    tags = [s.strip() for s in a.tags.replace("|", ",").split(",") if s.strip()]
    bad = [c for c in tags if c not in CASES]
    if bad:
        ap.error(f"알 수 없는 케이스 태그 {bad}. 허용: {CASES}")
    split = SPLITS.get(a.split, a.split)

    cams, ref = load_calibs(json.load(open(a.calib)))
    names = [ref] + [c for c in cams if c != ref]          # 기준 카메라를 맨 앞에
    raw, out = Path(a.raw), Path(a.out) / a.id
    folder = {c: raw / f"{a.id}_exo_{c}" for c in names}
    folder["ego"] = raw / f"{a.id}_ego"
    missing = [str(p) for p in folder.values() if not p.is_dir()]
    if missing:
        raise SystemExit(f"[중단] 녹화 폴더 없음: {missing}")

    # 시간 차 (기준 카메라 제외 전부)
    try:
        sync = {c: read_sync_row(a.sync, a.id, camera=c) for c in names[1:] + ["ego"]}
    except SyncError as e:
        raise SystemExit(f"[건너뜀/중단] {e}")

    # 기준 시간축
    t_ref_all = read_frame_times(folder[ref])
    if not (0 <= a.start < a.end <= len(t_ref_all)):
        raise SystemExit(f"[중단] --start {a.start} / --end {a.end} 가 기준 카메라 프레임 수 {len(t_ref_all)} 범위 밖")
    t_ref = t_ref_all[a.start:a.end]
    T = len(t_ref)
    fps = float(round((T - 1) / (t_ref[-1] - t_ref[0]), 3))

    # 4단계: 카메라별 2D -> 기준 시간축 -> 삼각측량
    uv_views = []
    for c in names:
        t_c = read_frame_times(folder[c])
        uv_c = ex.extract_exo_2d(find_video(folder[c]), a.hand, a.mirrored)
        if len(uv_c) != len(t_c):
            raise SystemExit(f"[중단] {c}: 영상 프레임 {len(uv_c)} != 시각 {len(t_c)} (data.jsonl 형식 확인)")
        if c == ref:
            uv_views.append(uv_c[a.start:a.end])
        else:
            uv_views.append(align_to_ref(t_c, uv_c, t_ref, sync[c]))
    uv_views = np.stack(uv_views)                                   # (C,T,K,2)
    tri_st = {**DEFAULT_TRI, "max_reproj_px": a.max_reproj_px}
    kp_raw, tri_err, n_views = triangulate_series(uv_views, [cams[c] for c in names], tri_st)

    # 8단계: 보정 (한 번만)
    kp, report, st = correct(kp_raw, fps)

    # 6단계: 손목 자세 (보정된 키포인트 기준)
    wrist_pose = wrist_pose_series(kp, ex.EXO_NAMES)

    # 4·5단계: ego 거리 -> 기준 시간축
    t_ego = read_frame_times(folder["ego"])
    d_ego = ex.extract_ego_pinch(find_video(folder["ego"]), a.hand, a.mirrored)
    if len(d_ego) != len(t_ego):
        raise SystemExit(f"[중단] ego: 영상 프레임 {len(d_ego)} != 시각 {len(t_ego)}")
    pinch = sync_ego_to_exo(t_ego, d_ego, t_ref, **sync["ego"])

    # 9단계: 저장
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "keypoints_raw.npy", kp_raw)
    np.save(out / "keypoints.npy", kp)
    np.save(out / "wrist_pose.npy", wrist_pose)
    np.save(out / "pinch_dist.npy", pinch)
    np.save(out / "keypoints_2d.npy", uv_views)          # (C,T,K,2) 기준 시간축, 카메라 순서 = meta.cameras
    np.save(out / "triangulation_err.npy", tri_err)      # (T,K) px
    json.dump(ex.EXO_NAMES, open(out / "keypoint_names.json", "w"), ensure_ascii=False)
    np.savetxt(out / "exo_ts.csv", t_ref, header="t_s", comments="", fmt="%.6f")
    np.savetxt(out / "ego_ts.csv", t_ego, header="t_s", comments="", fmt="%.6f")
    v = find_video(folder[ref]); shutil.copy2(v, out / f"exo{v.suffix}")      # 기준 카메라 = exo
    for c in names[1:]:
        v = find_video(folder[c]); shutil.copy2(v, out / f"exo_{c}{v.suffix}")
    v = find_video(folder["ego"]); shutil.copy2(v, out / f"ego{v.suffix}")
    json.dump({**report, **tri_report(tri_err, n_views, tri_st["min_views"]),
               "names": ex.EXO_NAMES, "settings": st, "triangulation": tri_st,
               "pinch_missing": float(np.isnan(pinch).mean())},
              open(out / "quality.json", "w"), ensure_ascii=False, indent=1)
    meta = {
        "episode_id": a.id, "task": a.task,
        "case_tags": tags,
        "collector": a.collector, "hand": a.hand, "split": split,
        "exo_start_frame": a.start, "exo_end_frame": a.end,   # 기준 카메라(exo) 영상 전체 기준, end 미포함
        "exo_fps": fps,
        "sync_offset_s": sync["ego"]["off0"], "drift_s": sync["ego"]["off1"] - sync["ego"]["off0"],  # ego 기준 (가이드 형식)
        "keypoint_method": "triangulation",
        "cameras": names, "ref_camera": ref,
        "camera_sync": {c: sync[c] for c in names[1:]},
        "smoothing": settings_string(st) + f", tri(min_views={tri_st['min_views']},reproj<={tri_st['max_reproj_px']}px)",
        "notes": a.notes,
    }
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=2)
    print(f"{a.id}: T={T}, 삼각측량 성공률={tri_report(tri_err, n_views)['tri_ok_ratio']}")
    print("labels.json: python labels.py init", out, "로 틀을 만든 뒤 채운다 (7단계).")


if __name__ == "__main__":
    main()
