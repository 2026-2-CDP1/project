"""에피소드 폴더 형식 검사 + 품질 표 + 케이스 분포 (완료 체크리스트용).

  python validate.py <episodes 루트>
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from constants import TASKS, CASES, SPLITS

REQUIRED = ["exo_ts.csv", "ego_ts.csv", "keypoints_raw.npy", "keypoints.npy",
            "keypoint_names.json", "wrist_pose.npy", "pinch_dist.npy", "labels.json", "meta.json"]
META_KEYS = ["episode_id", "task", "case_tags", "collector", "hand", "split", "exo_fps",
             "sync_offset_s", "drift_s", "smoothing", "notes", "exo_start_frame", "exo_end_frame"]


def check_episode(d: Path):
    errs = [f"{n} 없음" for n in REQUIRED if not (d / n).exists()]
    for v in ("exo", "ego"):
        if not ((d / f"{v}.mp4").exists() or (d / f"{v}.mov").exists()):
            errs.append(f"{v}.mp4/.mov 없음")
    if errs:
        return errs, None
    kp, raw = np.load(d / "keypoints.npy"), np.load(d / "keypoints_raw.npy")
    wp, pd_ = np.load(d / "wrist_pose.npy"), np.load(d / "pinch_dist.npy")
    names = json.load(open(d / "keypoint_names.json"))
    meta = json.load(open(d / "meta.json"))
    T = len(kp)
    if kp.shape != raw.shape: errs.append("keypoints 와 raw shape 불일치")
    if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[1] != len(names): errs.append(f"keypoints shape {kp.shape} / names {len(names)}")
    if wp.shape != (T, 7): errs.append(f"wrist_pose shape {wp.shape}")
    if pd_.shape != (T,): errs.append(f"pinch_dist shape {pd_.shape}")
    for n in ("thumb_tip", "index_tip"):
        if n not in names: errs.append(f"keypoint_names 에 {n} 없음")
    q = wp[:, 3:]; ok = ~np.isnan(q).any(axis=1)
    if ok.any() and not np.allclose(np.linalg.norm(q[ok], axis=1), 1, atol=1e-4): errs.append("쿼터니언 정규화 안 됨")
    errs += [f"meta.json 에 {k} 없음" for k in META_KEYS if k not in meta]
    if meta.get("episode_id") != d.name: errs.append("meta.episode_id 와 폴더명 불일치")
    if not str(d.name).startswith("T") or "_E" not in d.name: errs.append("에피소드 ID 형식 T{n}_E{nnn}")
    if meta.get("task") not in TASKS: errs.append(f"task '{meta.get('task')}' 는 {TASKS} 중 하나여야 함")
    if meta.get("split") not in SPLITS.values(): errs.append(f"split '{meta.get('split')}' 는 {list(SPLITS.values())} 중 하나여야 함")
    if meta.get("keypoint_method") == "triangulation":
        for c in meta.get("cameras", [])[1:]:
            v = [x for x in (d / f"exo_{c}.mov", d / f"exo_{c}.mp4") if x.exists()]
            if not v: errs.append(f"exo_{c} 영상 없음")
    if "exo_start_frame" in meta and "exo_end_frame" in meta and meta["exo_end_frame"] - meta["exo_start_frame"] != T:
        errs.append("exo_end_frame - exo_start_frame 이 키포인트 프레임 수 T 와 다름")
    bad = [c for c in meta.get("case_tags", []) if c not in CASES]
    if bad: errs.append(f"알 수 없는 케이스 태그 {bad}")
    from labels import check
    errs += ["labels: " + e for e in check(json.load(open(d / "labels.json")), meta)]
    return errs, (meta, kp, raw, pd_)


def main(root):
    eps = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p.name[:1] == "T"))
    tags, rows, fail = Counter(), [], 0
    for d in eps:
        errs, info = check_episode(d)
        if errs:
            fail += 1; print(f"[FAIL] {d.name}: " + "; ".join(errs))
        if info:
            meta, kp, raw, pd_ = info
            tags.update((meta["task"], c) for c in meta["case_tags"])
            q = json.load(open(d / "quality.json")) if (d / "quality.json").exists() else {}
            rows.append((d.name, np.isnan(raw).any(axis=2).mean(), np.mean(q.get("spike_ratio", [np.nan])), np.isnan(pd_).mean()))
    print(f"\n에피소드 {len(eps)}개, 형식 오류 {fail}개")
    if rows:
        print(f"\n{'ID':<12}{'누락률':>8}{'튐률':>8}{'pinch누락':>10}")
        for r in rows: print(f"{r[0]:<12}{r[1]:>8.3f}{r[2]:>8.3f}{r[3]:>10.3f}")
        print("\n태스크 × 케이스 태그 수 (빈 칸 주의)")
        for task in sorted({t for t, _ in tags}):
            print(task, {c: tags[(task, c)] for c in CASES})


if __name__ == "__main__":
    main(sys.argv[1])
