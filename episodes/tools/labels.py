"""행동 구간 라벨 (역할 가이드 7단계): 틀 만들기, 검증, 이중 라벨링 일관성 비교.

프레임 번호는 exo.mp4 '전체' 기준 (잘린 구간의 상대 번호가 아님). 구간 범위는 meta.json 의
exo_start_frame ~ exo_end_frame(미포함) 이고 labels.json 에도 같은 값을 적는다.
5-2번: g = grasp_frame - exo_start_frame, 5-1번: exo_start_frame 만큼 건너뛰기, 8번: start_frame.

  python labels.py init   ../T1_E001            # meta.json 에서 start/end 를 가져와 labels.json 틀 생성
  python labels.py check  ../T1_E001            # labels.json 을 meta.json 과 대조 검사
  python labels.py compare A_episodes B_episodes [--tol 3]   # 두 사람의 라벨 폴더(<ID>/labels.json) 비교
"""
import argparse, json
from pathlib import Path

SEGMENTS = ["approach", "grasp", "transport", "release", "retreat"]


def init(ep_dir):
    ep = Path(ep_dir)
    meta = json.load(open(ep / "meta.json"))
    lab = {"episode_id": meta["episode_id"], "frame_ref": "exo",
           "exo_start_frame": meta["exo_start_frame"], "exo_end_frame": meta["exo_end_frame"],
           "segments": [], "grasp_frame": None}
    path = ep / "labels.json"
    if path.exists():
        raise SystemExit(f"{path} 이미 있음 (덮어쓰지 않음)")
    json.dump(lab, open(path, "w"), ensure_ascii=False, indent=2)
    print("created", path)


def check(lab, meta=None):
    errs = []
    if lab.get("frame_ref") != "exo":
        errs.append("frame_ref 는 'exo'")
    start, end = lab.get("exo_start_frame"), lab.get("exo_end_frame")
    if start is None or end is None:
        errs.append("labels.json 에 exo_start_frame / exo_end_frame 없음")
    if meta is not None:
        for k in ("exo_start_frame", "exo_end_frame"):
            if lab.get(k) != meta.get(k):
                errs.append(f"{k}: labels({lab.get(k)}) != meta({meta.get(k)})")
        if lab.get("episode_id") != meta.get("episode_id"):
            errs.append("episode_id 가 meta 와 다름")
    segs = lab.get("segments", [])
    order = [SEGMENTS.index(s["name"]) for s in segs if s["name"] in SEGMENTS]
    for s in segs:
        if s["name"] not in SEGMENTS:
            errs.append(f"알 수 없는 구간 이름: {s['name']}")
    if order != sorted(order) or len(set(order)) != len(order):
        errs.append(f"구간 순서가 {SEGMENTS} 순서가 아니거나 중복")
    for s in segs:
        if s["start"] > s["end"]:
            errs.append(f"{s['name']}: start > end")
        if start is not None and not (start <= s["start"] and s["end"] < end):
            errs.append(f"{s['name']}: [{s['start']},{s['end']}] 가 에피소드 구간 [{start},{end}) 밖")
    for a, b in zip(segs, segs[1:]):
        if b["start"] != a["end"] + 1:
            errs.append(f"{a['name']}.end({a['end']}) + 1 != {b['name']}.start({b['start']})")
    g = lab.get("grasp_frame")
    if g is None:
        errs.append("grasp_frame 없음")
    elif start is not None and not (start <= g < end):
        errs.append(f"grasp_frame {g} 가 에피소드 구간 밖")
    else:
        gs = [s for s in segs if s["name"] == "grasp"]
        if gs and g < gs[0]["start"]:
            errs.append("grasp_frame 이 grasp 구간 시작보다 앞 (완전히 쥔 프레임은 처음 닿은 뒤)")
        if gs and g > gs[0]["end"]:
            errs.append("grasp_frame 이 grasp 구간 끝보다 뒤 (완전히 쥔 프레임은 물체를 들기 전)")
    return errs


def compare(dir_a, dir_b, tol):
    """구간별 시작 프레임 차이. tol 초과 구간은 규칙이 모호한 후보."""
    stats = {}
    for fa in sorted(Path(dir_a).glob("*/labels.json")):
        fb = Path(dir_b) / fa.parent.name / "labels.json"
        if not fb.exists():
            continue
        la, lb = json.load(open(fa)), json.load(open(fb))
        sa = {s["name"]: s["start"] for s in la["segments"]}
        sb = {s["name"]: s["start"] for s in lb["segments"]}
        for n in SEGMENTS:
            if n in sa and n in sb:
                stats.setdefault(n, []).append(abs(sa[n] - sb[n]))
        if la.get("grasp_frame") is not None and lb.get("grasp_frame") is not None:
            stats.setdefault("grasp_frame", []).append(abs(la["grasp_frame"] - lb["grasp_frame"]))
    print(f"{'구간':<12}{'n':>4}{'평균':>8}{'최대':>6}{'>tol':>6}")
    for n, ds in stats.items():
        print(f"{n:<12}{len(ds):>4}{sum(ds)/len(ds):>8.1f}{max(ds):>6}{sum(d > tol for d in ds):>6}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("init", "check"):
        sp.add_parser(n).add_argument("ep_dir")
    m = sp.add_parser("compare"); m.add_argument("a"); m.add_argument("b"); m.add_argument("--tol", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "init":
        init(a.ep_dir)
    elif a.cmd == "check":
        ep = Path(a.ep_dir)
        errs = check(json.load(open(ep / "labels.json")), json.load(open(ep / "meta.json")))
        print("OK" if not errs else "\n".join(errs))
    else:
        compare(a.a, a.b, a.tol)
