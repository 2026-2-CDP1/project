"""촬영 중 움직인 exo 카메라의 위치를 손 키포인트로 다시 구한다 (체커보드 영상이 없을 때).

front 는 움직이지 않았다고 보고 기준으로 둔다. 다른 두 카메라로 손 3D 를 만들고, 그 3D 와 움직인 카메라의
2D 짝으로 시작값을 구하고(PnP), 세 카메라를 동시에 맞춘다. extract_all.py 캐시와 sync.csv 만 읽는다.

--free      위치와 방향을 모두 푸는 카메라
--rot-only  위치는 그대로 두고 방향만 푸는 카메라 (조금 움직인 카메라)
둘 다 움직였으면 하나는 --rot-only 로 둔다. front 만 고정하면 전체 크기(스케일)가 정해지지 않는다.

  # T1: E017 부터 right 가 크게, left 가 조금 움직임
  python recalib.py solve --raw ~/data/raw/T1 --cache ~/out/seohyeon/kp_cache_v2/T1 --sync ../../sync/sync.csv \\
      --calib ../../sync/exo_calib_T1.json --first T1_E017 --last T1_E075 --free right --rot-only left \\
      --out ~/out/seohyeon/recalib/exo_calib_T1_E017.json

  # 장애물 박스를 새 보정값으로 그려 확인 (박스는 front 로만 찍어서 이 계산과 무관한 검증)
  python recalib.py check --frames ~/out/seohyeon/obstacle_frames --obstacles ../obstacles.json \\
      --calib ~/out/seohyeon/recalib/exo_calib_T1_E017.json --ids T1_E036 T1_E066 --out ~/out/seohyeon/recalib/check

solve 는 에피소드 4개 중 1개를 계산에서 빼 두고, 그 에피소드로 이전·새 보정값을 비교해 출력한다.
"""
import argparse, copy, json
from pathlib import Path
import numpy as np

from extractors import EXO_NAMES
from hand_math import table_to_pixel
from io_utils import find_episode_folder, kp_cache_path, match_frames, read_frame_times, read_records, read_sync_row
from multiview import DEFAULT_TRI, align_to_ref, load_calibs, triangulate_series

REF = "front"
HAND_KP = ["wrist", "index_mcp", "pinky_mcp", "thumb_tip", "index_tip"]


def load_episode(raw, cache, sync_csv, eid, hand, stride, max_speed):
    """에피소드 -> {카메라: (N,K,2)} 기준 시간축. 손이 빠른 프레임은 뺀다 (시간 차 오차가 픽셀 오차가 되므로)."""
    t_ref = read_frame_times(find_episode_folder(raw, eid, f"exo_{REF}"))
    out = {}
    for c in [REF, "left", "right"]:
        t_c = read_frame_times(find_episode_folder(raw, eid, f"exo_{c}"))
        uv = match_frames(np.load(kp_cache_path(cache, f"exo_{c}", eid, hand)), t_c, c)
        out[c] = uv if c == REF else align_to_ref(t_c, uv, t_ref, read_sync_row(sync_csv, eid, camera=c))
    w = out[REF][:, EXO_NAMES.index("wrist")]
    speed = np.linalg.norm(np.gradient(w, t_ref, axis=0), axis=1)
    keep = np.flatnonzero(speed < max_speed)[::stride]
    idx = [EXO_NAMES.index(n) for n in HAND_KP]
    return {c: v[keep][:, idx] for c, v in out.items()}


def undistort(uv, cam):
    import cv2
    K, _, _, dist = cam
    return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)


def tri_two(uv_a, cam_a, uv_b, cam_b):
    """두 카메라 2D (N,2) -> 3D (N,3), 카메라별 재투영 오차 중 큰 값 (N,)."""
    import cv2
    Pa = np.hstack([cam_a[1], cam_a[2][:, None]])
    Pb = np.hstack([cam_b[1], cam_b[2][:, None]])
    h = cv2.triangulatePoints(Pa, Pb, undistort(uv_a, cam_a).T, undistort(uv_b, cam_b).T)
    X = (h[:3] / h[3]).T
    err = np.maximum(reproj(X, uv_a, cam_a), reproj(X, uv_b, cam_b))
    return X, err


def reproj(X, uv, cam):
    K, R, t, dist = cam
    return np.linalg.norm(table_to_pixel(X, K, R, t, dist) - uv, axis=1)


def pairs(data, a, b, target):
    """세 카메라가 모두 잡은 점만. (uv_a, uv_b, uv_target)"""
    A, B, C = (data[c].reshape(-1, 2) for c in (a, b, target))
    ok = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1) & np.isfinite(C).all(axis=1)
    return A[ok], B[ok], C[ok]


def solve_free(X, uv, cam, px):
    """위치·방향 모두. 이전 값에 기대지 않고 푼 뒤 inlier 로 다듬는다."""
    import cv2
    K, _, _, dist = cam
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, dist, reprojectionError=px, iterationsCount=2000,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None:
        raise SystemExit("[중단] PnP 실패: 점이 부족하거나 대응이 맞지 않음")
    i = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[i], uv[i], K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return (K, R, tvec.ravel(), dist), len(i)


def solve_rot(X, uv, cam, px):
    """위치 고정, 방향만. 카메라 중심 C 를 유지해 t = -R C."""
    import cv2
    from scipy.optimize import least_squares
    K, R0, t0, dist = cam
    C = -R0.T @ t0
    r0, _ = cv2.Rodrigues(R0)

    def resid(r):
        R, _ = cv2.Rodrigues(r)
        p, _ = cv2.projectPoints(X, r, -R @ C, K, dist)
        return (p.reshape(-1, 2) - uv).ravel()

    s = least_squares(resid, r0.ravel(), loss="soft_l1", f_scale=px)
    R, _ = cv2.Rodrigues(s.x)
    return (K, R, -R @ C, dist), int((np.linalg.norm(resid(s.x).reshape(-1, 2), axis=1) < px).sum())


def bundle(data, cams, free, rot_only, names, px, max_points):
    """세 카메라를 동시에 맞춘다. 한 대씩 번갈아 풀면 틀린 카메라에 다른 카메라가 맞춰진 채 멈춘다
    (가짜 데이터 시험: 오차는 3 px 인데 방향이 0.8° 틀림). 3D 는 매번 세 카메라 광선의 중점으로 다시 만든다."""
    import cv2
    from scipy.optimize import least_squares
    uv = [data[c].reshape(-1, 2) for c in names]
    ok = np.logical_and.reduce([np.isfinite(u).all(axis=1) for u in uv])
    uv = [u[ok] for u in uv]
    if len(uv[0]) > max_points:
        pick = np.linspace(0, len(uv[0]) - 1, max_points).astype(int)
        uv = [u[pick] for u in uv]
    rays = [np.c_[undistort(u, cams[c]), np.ones(len(u))] for u, c in zip(uv, names)]
    centers = {c: -cams[c][1].T @ cams[c][2] for c in names}

    def unpack(x):
        out, i = dict(cams), 0
        for c in free:
            R, _ = cv2.Rodrigues(x[i:i + 3])
            out[c] = (cams[c][0], R, x[i + 3:i + 6], cams[c][3]); i += 6
        for c in rot_only:
            R, _ = cv2.Rodrigues(x[i:i + 3])
            out[c] = (cams[c][0], R, -R @ centers[c], cams[c][3]); i += 3
        return out

    def resid(x):
        cur = unpack(x)
        A, b = np.zeros((len(uv[0]), 3, 3)), np.zeros((len(uv[0]), 3))
        np.seterr(divide="ignore", over="ignore", invalid="ignore")    # 시험 스텝이 크게 튈 때 경고만 끈다
        for c, n in zip(names, rays):
            K, R, t, _ = cur[c]
            d = n @ R                                  # R^T n
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            M = np.eye(3) - d[:, :, None] * d[:, None, :]
            A += M
            b += M @ (-R.T @ t)
        X = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        return np.concatenate([(table_to_pixel(X, *[cur[c][i] for i in range(4)]) - u).ravel() for c, u in zip(names, uv)])

    x0 = []
    for c in free:
        x0 += [cv2.Rodrigues(cams[c][1])[0].ravel(), cams[c][2]]
    for c in rot_only:
        x0 += [cv2.Rodrigues(cams[c][1])[0].ravel()]
    s = least_squares(resid, np.concatenate(x0), loss="soft_l1", f_scale=px, x_scale="jac", max_nfev=200)
    e = np.linalg.norm(resid(s.x).reshape(-1, 2), axis=1)
    return unpack(s.x), len(uv[0]), float(np.median(e))


def report(data, cams, names, limit=4000):
    """multiview.triangulate_series 그대로 돌린 품질 (카메라 3대가 잡은 프레임에서 3대를 다 쓰는 비율이 핵심)."""
    uv = np.stack([data[c] for c in names]).astype(float)                 # (C,T,K,2)
    all3 = np.isfinite(uv).all(axis=3).all(axis=0)                        # (T,K)
    frames = np.flatnonzero(all3.any(axis=1))
    frames = frames[np.linspace(0, len(frames) - 1, min(limit, len(frames))).astype(int)]
    _, err, nv = triangulate_series(uv[:, frames], [cams[c] for c in names])
    a3 = all3[frames]
    return {"3대 본 점": int(a3.sum()), "3대 모두 사용": round(float((nv[a3] == 3).mean()), 3),
            "2대로 줄임": round(float((nv[a3] == 2).mean()), 3), "버림": round(float((nv[a3] == 0).mean()), 3),
            "오차 중앙값 px": round(float(np.nanmedian(err[a3 & (nv > 0)])), 2) if (a3 & (nv > 0)).any() else None}


def cmd_solve(a):
    raw, cache = Path(a.raw).expanduser(), Path(a.cache).expanduser()
    calib = json.load(open(Path(a.calib).expanduser()))
    cams, ref = load_calibs(calib)
    assert ref == REF, "기준 카메라가 front 가 아님"
    hand = {k: r["hand"] for k, r in read_records(a.records).items()}
    ids = [p.name for p in sorted((raw / f"exo_{REF}").glob("*_E*")) if a.first <= p.name <= a.last]
    if not ids:
        raise SystemExit("[중단] 대상 에피소드 없음")
    held = ids[::4]
    fit, test, skipped = [], [], []
    for eid in ids:
        try:
            d = load_episode(raw, cache, a.sync, eid, hand[eid], a.stride, a.max_speed)
        except Exception as e:                       # T2_E008 처럼 시간이 안 겹치는 에피소드는 빼고 간다
            skipped.append(f"{eid}: {e}")
            continue
        (test if eid in held else fit).append(d)
    print(f"에피소드 {len(ids)}개: 계산 {len(fit)}, 검증 {len(test)}, 건너뜀 {len(skipped)} {skipped[:3]}")
    merge = lambda L: {c: np.concatenate([d[c] for d in L]) for c in [REF, "left", "right"]}
    train, hold = merge(fit), merge(test)

    names = [REF, "left", "right"]
    new = dict(cams)
    before = report(hold, cams, names)
    for rnd in range(a.rounds):
        log = []
        for c, fn in [(c, solve_free) for c in a.free] + [(c, solve_rot) for c in a.rot_only]:
            other = next(o for o in names if o not in (REF, c))
            A, B, C = pairs(train, REF, other, c)
            X, err = tri_two(A, new[REF], B, new[other])
            good = err < a.tri_px
            if good.sum() < 100:
                raise SystemExit(f"[중단] {c}: 쓸 수 있는 점이 {good.sum()}개 뿐")
            X, C = X[good], C[good]
            if len(X) > a.max_points:
                pick = np.linspace(0, len(X) - 1, a.max_points).astype(int)
                X, C = X[pick], C[pick]
            new[c], n = fn(X, C, new[c], a.pnp_px)
            log.append(f"{c}: 점 {len(X)}, inlier {n / len(X):.2f}, 오차 중앙값 {np.median(reproj(X, C, new[c])):.1f} px")
        print(f"[{rnd + 1}/{a.rounds}] " + " | ".join(log), flush=True)
    new, n, med = bundle(train, new, a.free, a.rot_only, names, a.pnp_px, a.max_points)
    print(f"[동시 맞춤] 점 {n}, 오차 중앙값 {med:.1f} px", flush=True)
    after = report(hold, new, names)

    print("\n검증 에피소드 (계산에 안 씀):")
    print("  이전 보정값:", before)
    print("  새 보정값  :", after)
    for c in a.free + a.rot_only:
        Co, Cn = -cams[c][1].T @ cams[c][2], -new[c][1].T @ new[c][2]
        cos = (np.trace(new[c][1] @ cams[c][1].T) - 1) / 2
        print(f"  {c}: 방향 변화 {np.degrees(np.arccos(np.clip(cos, -1, 1))):.2f}°, 위치 변화 {np.linalg.norm(Cn - Co) * 100:.1f} cm")

    out = copy.deepcopy(calib)
    out["valid_for"] = f"{a.first}~{a.last} ({Path(a.calib).name} 에서 손 키포인트로 다시 구함)"
    out["recalib"] = {"free": a.free, "rot_only": a.rot_only, "based_on": Path(a.calib).name,
                      "holdout_before": before, "holdout_after": after}
    for c in a.free + a.rot_only:
        e = out["cameras"][c]
        e["R"], e["t"] = new[c][1].tolist(), new[c][2].tolist()
        e["camera_center_table_m"] = (-new[c][1].T @ new[c][2]).tolist()
        for k in ("reproj_err_px_mean", "reproj_err_px_max", "n_frames", "corner_spread_px", "corner_order"):
            e.pop(k, None)
    p = Path(a.out).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("저장 ->", p)


def cmd_check(a):
    """obstacles.py 가 뽑은 사진에 장애물 박스를 그린다. 카메라별로 판과 겹치면 그 보정값이 맞다."""
    import cv2
    from obstacles import BOX_EDGES, VIEWS, box_corners
    frames, out = Path(a.frames).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    obs = json.load(open(Path(a.obstacles).expanduser()))["episodes"]
    index = json.load(open(frames / "index.json"))
    cams, _ = load_calibs(json.load(open(Path(a.calib).expanduser())))
    for eid in a.ids:
        scale = index[eid]["scale"]
        shot = next((i for i, s in enumerate(index[eid]["shots"]) if s["front_frame"] == obs[eid]["front_frame"]), 0)
        tiles = []
        for c in VIEWS:
            img = cv2.imread(str(frames / eid / f"s{shot}_{c}.jpg"))
            K, R, t, dist = cams[c]
            for ob in obs[eid]["obstacles"]:
                px = (table_to_pixel(box_corners(ob), K, R, t, dist) * scale).astype(int)
                for e0, e1 in BOX_EDGES:
                    cv2.line(img, tuple(px[e0]), tuple(px[e1]), (0, 255, 0), 2)
            cv2.putText(img, f"{eid} {c}", (10, 35), 0, 1, (0, 255, 255), 2)
            tiles.append(img)
        cv2.imwrite(str(out / f"{eid}.jpg"), np.hstack(tiles))
        print("저장 ->", out / f"{eid}.jpg")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--raw", required=True, help="태스크 폴더, 예: ~/data/raw/T1")
    s.add_argument("--cache", required=True)
    s.add_argument("--sync", required=True)
    s.add_argument("--calib", required=True, help="이동 전 보정값 exo_calib_T{n}.json")
    s.add_argument("--first", required=True)
    s.add_argument("--last", required=True)
    s.add_argument("--free", nargs="*", default=[], help="위치·방향을 모두 풀 카메라")
    s.add_argument("--rot-only", nargs="*", default=[], help="방향만 풀 카메라")
    s.add_argument("--out", required=True)
    s.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "records.csv"))
    s.add_argument("--rounds", type=int, default=2, help="동시 맞춤 전에 한 대씩 푸는 횟수 (시작값용)")
    s.add_argument("--stride", type=int, default=3, help="기준 카메라 프레임 간격")
    s.add_argument("--max-speed", type=float, default=300, help="기준 카메라 손목 속도 상한 (px/s)")
    s.add_argument("--tri-px", type=float, default=DEFAULT_TRI["max_reproj_px"], help="3D 를 만들 때 허용 오차")
    s.add_argument("--pnp-px", type=float, default=8.0, help="PnP inlier 기준")
    s.add_argument("--max-points", type=int, default=30000)
    k = sub.add_parser("check")
    k.add_argument("--frames", required=True)
    k.add_argument("--obstacles", required=True)
    k.add_argument("--calib", required=True)
    k.add_argument("--ids", nargs="+", required=True)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_solve if a.cmd == "solve" else cmd_check)(a)


if __name__ == "__main__":
    main()
