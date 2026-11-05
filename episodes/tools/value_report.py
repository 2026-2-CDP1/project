"""리타게팅 담당에게 보여 줄 값 요약: 처리된 에피소드들의 좌표 범위·속도·빈 비율, 예시 에피소드 궤적 그림.

  python value_report.py ~/out/seohyeon/episodes_clap --example T1_E040 --out ~/out/seohyeon/value_report

출력(--out):
  values.md        전체 요약 표 + 예시 에피소드의 실제 값 몇 줄
  <ID>_traj.png    예시 에피소드: 손목 x·y·z, 팔꿈치 x·y·z, 집기 거리, 손목 속도 (회색 = 값 없음)
값은 처리 결과 그대로 (keypoints.npy = 테이블 좌표계 m, 보정 후). 구간은 그 결과를 만들 때 쓴 구간이다.
"""
import argparse, json
from pathlib import Path
import numpy as np


def load(ep):
    names = json.load(open(ep / "keypoint_names.json"))
    return {"names": names, "meta": json.load(open(ep / "meta.json")), "kp": np.load(ep / "keypoints.npy"),
            "wp": np.load(ep / "wrist_pose.npy"), "pinch": np.load(ep / "pinch_dist.npy"),
            "t": np.loadtxt(ep / "exo_ts.csv", skiprows=1)}


def rng(x):
    """1~99% 범위와 중앙값 문자열. 값이 없으면 '-'."""
    x = x[np.isfinite(x)]
    return "-" if not len(x) else f"{np.percentile(x, 1):.3f} ~ {np.percentile(x, 99):.3f} (중앙값 {np.median(x):.3f})"


def speed(kp, t):
    return np.linalg.norm(np.gradient(kp, t, axis=0), axis=1)


def shade(ax, t, bad):
    d = np.diff(np.r_[0, bad.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="0.85", lw=0)


def plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = d["t"] - d["t"][0]
    w, e = d["kp"][:, d["names"].index("wrist")], d["kp"][:, d["names"].index("elbow")]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for a, x, title in ((ax[0], w, "wrist position (m)"), (ax[1], e, "elbow position (m)")):
        shade(a, t, np.isnan(x).any(axis=1))
        for i, n in enumerate("xyz"):
            a.plot(t, x[:, i], label=n)
        a.set_ylabel(title); a.legend(loc="upper right", ncol=3); a.grid(alpha=0.3)
    shade(ax[2], t, np.isnan(d["pinch"]))
    ax[2].plot(t, d["pinch"]); ax[2].set_ylabel("pinch distance (m)"); ax[2].grid(alpha=0.3)
    shade(ax[3], t, np.isnan(w).any(axis=1))
    ax[3].plot(t, speed(w, d["t"])); ax[3].set_ylabel("wrist speed (m/s)"); ax[3].set_xlabel("time (s)"); ax[3].grid(alpha=0.3)
    m = d["meta"]
    ax[0].set_title(f"{m['episode_id']}  hand={m['hand']}  frames {m['exo_start_frame']}~{m['exo_end_frame']}  (gray = no value)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="process_all.py 의 --out")
    ap.add_argument("--example", required=True, help="궤적 그림과 값 예시를 보여 줄 에피소드")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    eps = sorted(p for p in root.iterdir() if (p / "keypoints.npy").exists())
    data = [load(p) for p in eps]
    names = data[0]["names"]
    idx = {n: i for i, n in enumerate(names)}
    cat = lambda f: np.concatenate([f(d) for d in data])
    L = [f"# 추출 값 요약 ({len(data)}개 에피소드, {sum(len(d['kp']) for d in data)} 프레임)", "",
         "테이블 좌표계, 단위 m (x = front 에서 볼 때 오른쪽, y = 작업자 쪽, z = 위). 범위는 1~99%.", "",
         "| 값 | x | y | z |", "|---|---|---|---|"]
    for n in ("shoulder", "elbow", "wrist", "index_tip"):
        k = cat(lambda d: d["kp"][:, idx[n]])
        L.append(f"| {n} | {rng(k[:, 0])} | {rng(k[:, 1])} | {rng(k[:, 2])} |")
    arm = cat(lambda d: np.linalg.norm(d["kp"][:, idx["wrist"]] - d["kp"][:, idx["elbow"]], axis=1))
    sp = cat(lambda d: speed(d["kp"][:, idx["wrist"]], d["t"]))
    fps = [d["meta"]["exo_fps"] for d in data]
    L += ["", "| 값 | 범위 |", "|---|---|",
          f"| 아래팔 길이 (m) | {rng(arm)} |", f"| 손목 속도 (m/s) | {rng(sp)} |",
          f"| 집기 거리 pinch_dist (m) | {rng(cat(lambda d: d['pinch']))} |",
          f"| 프레임 수 T | {min(len(d['kp']) for d in data)} ~ {max(len(d['kp']) for d in data)} |",
          f"| 기준 카메라 fps | {min(fps):.1f} ~ {max(fps):.1f} |", "",
          "| 빈 비율 (NaN) | 중앙값 | 최악 |", "|---|---|---|"]
    for label, f in (("손목", lambda d: np.isnan(d["kp"][:, idx["wrist"]]).any(axis=1)),
                     ("팔꿈치", lambda d: np.isnan(d["kp"][:, idx["elbow"]]).any(axis=1)),
                     ("손목 방향", lambda d: np.isnan(d["wp"][:, 3:]).any(axis=1)),
                     ("집기 거리", lambda d: np.isnan(d["pinch"]))):
        v = [f(d).mean() for d in data]
        L.append(f"| {label} | {np.median(v):.2f} | {max(v):.2f} |")
    ex = next(d for d, p in zip(data, eps) if p.name == a.example)
    m = ex["meta"]
    L += ["", f"## 예시 {a.example} (hand={m['hand']}, 프레임 {m['exo_start_frame']}~{m['exo_end_frame']}, T={len(ex['kp'])})", "",
          "| 프레임 | 손목 x, y, z | 쿼터니언 w, x, y, z | 팔꿈치 x, y, z | 집기 거리 |", "|---|---|---|---|---|"]
    ok = np.flatnonzero(np.isfinite(ex["wp"]).all(axis=1) & np.isfinite(ex["kp"][:, idx["elbow"]]).all(axis=1))
    fmt = lambda v: ", ".join(f"{x:.3f}" for x in v)
    for f in ok[np.linspace(0, len(ok) - 1, 6).astype(int)] if len(ok) else []:
        L.append(f"| {m['exo_start_frame'] + f} | {fmt(ex['wp'][f, :3])} | {fmt(ex['wp'][f, 3:])} | {fmt(ex['kp'][f, idx['elbow']])} | {ex['pinch'][f]:.3f} |")
    (out / "values.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot(ex, out / f"{a.example}_traj.png")
    print("\n".join(L)); print("\n저장 ->", out)


if __name__ == "__main__":
    main()
