"""5-1 나란히 보기 영상: 사람 exo_front 영상 | 알고리즘별 로봇 재생 을 가로로 붙인 mp4 (역할 가이드 5-1 5단계).

쓰는 곳: worst_episodes.md의 이상값 확인 (지표 정의서 5절), 파지 프레임 확인 (2.11), 발표 그림.

    # 실제 (4번 결과가 있을 때)
    python replay/side_by_side.py --episode <3번 결과>/T1_E004 --retarget <4번 결과> \\
        --obstacles episodes/obstacles.json --out ~/out/신서연/videos/T1_E004.mp4 [--view both]
    # 데모 (4번 결과 전: 실제 exo 영상 + 가짜 궤적)
    python replay/side_by_side.py --demo --video <exo_front data.mov> --task T3 --out demo.mp4

화면
- 맨 왼쪽: exo_front 사람 영상 (에피소드 구간 exo_start_frame ~ exo_end_frame)
- 그 오른쪽: 알고리즘마다 로봇 재생. 카메라는 exo_front 카메라(sync/exo_calib_T{n}.json)를 4번 좌표 변환
  to_robot_point(p, info)로 로봇 좌표에 옮긴 것 → 사람 영상과 같은 구도 (변환이 평행 이동·축척·회전이라 그림이 같아짐)
- 실패 프레임(정의서 2.1: 위치 > 1 cm 또는 방향 > 5° 또는 관절 한계 밖)은 빨간 테두리 + FAIL, 목표 없음은 회색 NO TARGET
- grasp_frame은 GRASP 표시
- --view both면 아래 줄에 옆에서 본 로봇 (팔꿈치 높이·테이블 아래로 가는지 보기 좋음)
- 30 Hz로 맞춤 (60 fps 에피소드는 2프레임마다 하나, 지표 2.4와 같음)

서버: GPU 노드에서 MUJOCO_GL=egl (import 전에 설정). 노트북(Mac)은 그대로 됨. 영상은 레포에 넣지 않는다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
# replay/ 안에서 실행하면 "replay"가 폴더 대신 replay.py로 잡히므로, 스크립트 폴더를 빼고 레포 루트를 넣는다
def _is_here(p, here=Path(__file__).resolve().parent):
    """같은 폴더인지 (한글 경로가 NFC/NFD로 다르게 들어와도 같게 보도록 문자열 대신 samefile로)."""
    try:
        return os.path.samefile(p or ".", here)
    except OSError:
        return False


sys.path[:] = [p for p in sys.path if not _is_here(p)]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "metrics"))

PANEL_W, PANEL_H = 480, 360          # 1920x1440 (4:3)의 1/4
OUT_HZ = 30.0
SIDE_LOOKAT, SIDE_FROM = np.array([0.45, 0.0, 0.35]), np.array([0.45, -2.1, 0.85])


# ---------------------------------------------------------------- 카메라

def exo_camera_xml(calib_json, to_robot, name="exo"):
    """exo_front 카메라 → 로봇 좌표 MuJoCo 카메라 (pos, xyaxes, fovy). 변환은 평행 이동·축척·회전이어야 함."""
    c = json.load(open(calib_json, encoding="utf-8"))["cameras"]["front"]
    K, R, t = np.array(c["K"]), np.array(c["R"]), np.array(c["t"]).reshape(3)
    H = c.get("image_size", [1920, 1440])[1]
    C = -R.T @ t                                         # 카메라 중심 (테이블 좌표)
    o = np.asarray(to_robot(np.zeros(3)), float)
    M = np.column_stack([np.asarray(to_robot(e), float) - o for e in np.eye(3)])
    s = np.linalg.norm(M, axis=0).mean()
    Rot = M / s
    pos = np.asarray(to_robot(C), float)
    x_cam, y_cam = Rot @ R[0], Rot @ R[1]                # 카메라 오른쪽, 아래 (OpenCV 규약)
    fovy = np.degrees(2 * np.arctan(H / 2 / K[1, 1]))
    f = lambda v: " ".join(f"{x:.6f}" for x in v)
    # MuJoCo 카메라: x = 오른쪽, y = 위, -z 방향을 봄
    return f'<camera name="{name}" pos="{f(pos)}" xyaxes="{f(x_cam)} {f(-y_cam)}" fovy="{fovy:.3f}"/>'


def side_camera_xml(name="side"):
    fwd = SIDE_LOOKAT - SIDE_FROM
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    f = lambda v: " ".join(f"{v_:.6f}" for v_ in v)
    return f'<camera name="{name}" pos="{f(SIDE_FROM)}" xyaxes="{f(x)} {f(y)}" fovy="45"/>'


# ---------------------------------------------------------------- 장면

def scene_boxes(task, to_robot, obstacles_entry, outline_json):
    """그림용 테이블 판(윤곽 크기) + 장애물 상자 → 로봇 좌표 상자 목록."""
    from replay.scene import box_corners, map_box
    boxes = []
    if outline_json and Path(outline_json).exists():
        o = json.load(open(outline_json, encoding="utf-8")).get(task)
        if o:
            P = np.asarray(o["polygon"])
            c = (P.min(0) + P.max(0)) / 2
            size = [*(P.max(0) - P.min(0)), 0.02]
            boxes.append(map_box("vis_table", box_corners([c[0], c[1], -0.01], size, 0.0), to_robot))
    for i, ob in enumerate((obstacles_entry or {}).get("obstacles", []), 1):
        boxes.append(map_box(f"env_{ob.get('name') or f'obs{i}'}",
                             box_corners(ob["center"], ob["size"], ob["yaw_deg"]), to_robot))
    return boxes


# ---------------------------------------------------------------- 영상 읽기·쓰기

def read_frames(video, keep):
    """영상 프레임 번호 집합 keep만 (480x360 RGB). 회전 없이 저장된 그대로 (보정 기준)."""
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-noautorotate", "-i", str(video),
                          "-vf", f"scale={PANEL_W}:{PANEL_H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    nbytes, out, i, last = PANEL_W * PANEL_H * 3, {}, 0, max(keep)
    while i <= last:
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        if i in keep:
            out[i] = np.frombuffer(buf, np.uint8).reshape(PANEL_H, PANEL_W, 3)
        i += 1
    p.stdout.close()
    p.kill()
    return out


def open_writer(path, w, h, hz=OUT_HZ):
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(hz), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", str(Path(path).expanduser())], stdin=subprocess.PIPE)


def font(size=18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                     # 오래된 Pillow
        return ImageFont.load_default()


def label(img, lines, color=(255, 255, 255), border=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    if border:
        d.rectangle([2, 2, im.width - 3, im.height - 3], outline=border, width=6)
    y = 6
    for text, col in lines:
        d.rectangle([4, y - 2, 12 + 10 * len(text), y + 20], fill=(0, 0, 0))
        d.text((8, y), text, fill=col or color, font=font())
        y += 24
    return np.asarray(im)


# ---------------------------------------------------------------- 판정 (지표 정의서 2.1과 같은 계산)

def frame_status(res, out, lo, hi):
    """프레임마다 'ok' / 'fail' / 'no_target' / '' (목표가 npz에 없으면 '')."""
    import metrics as M
    T = len(res["q"])
    if "target_pos" not in res or "target_quat" not in res:
        return [""] * T
    pe, re = M.tracking_errors(out["tcp_pos"], out["tcp_quat"], res["target_pos"], res["target_quat"])
    ok_lim = M.in_limits(np.asarray(res["q"], float), lo, hi)
    st = []
    for t in range(T):
        if not (np.isfinite(pe[t]) and np.isfinite(re[t])):
            st.append("no_target")
        elif pe[t] <= M.SUCCESS_POS_M and re[t] <= M.SUCCESS_ROT_RAD and ok_lim[t]:
            st.append("ok")
        else:
            st.append("fail")
    return st


# ---------------------------------------------------------------- 그리기

def render_episode(out_path, video, start, end, ts, algos, task, calib_json, to_robot, obstacles_entry,
                   outline_json, grasp=None, view="exo", png=None):
    """algos: [(이름, npz dict)]. 프레임 구간 [start, end), ts: 구간 프레임 시각 (T,)."""
    import mujoco
    import metrics as M
    from replay.replay import Replayer

    lo, hi, _ = M.load_limits()
    cams = exo_camera_xml(calib_json, to_robot) + side_camera_xml()
    r = Replayer(env_boxes=scene_boxes(task, to_robot, obstacles_entry, outline_json), extra_worldbody=cams)
    m, d = r.m, r.d
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, PANEL_W), max(m.vis.global_.offheight, PANEL_H)
    ren = mujoco.Renderer(m, PANEL_H, PANEL_W)

    T = end - start
    k = max(1, int(round((1 / OUT_HZ) / np.median(np.diff(ts))))) if len(ts) > 1 else 1   # 30 Hz로 맞춤
    idx = list(range(0, T, k))
    frames = read_frames(video, {start + i for i in idx})
    status = {}
    for name, res in algos:
        if len(res["q"]) != T:
            raise ValueError(f"{name}: 궤적 길이 {len(res['q'])} ≠ 구간 {T}")
        status[name] = frame_status(res, r.replay(np.asarray(res["q"], float)), lo, hi)

    rows = 2 if view == "both" else 1
    W, Hh = PANEL_W * (1 + len(algos)), PANEL_H * rows
    writer = None if png is not None else open_writer(out_path, W, Hh)
    for n, i in enumerate(idx):
        if png is not None and n != png:
            continue
        cam_view = "side" if view == "side" else "exo"
        exo = frames.get(start + i, np.zeros((PANEL_H, PANEL_W, 3), np.uint8))
        g = grasp is not None and abs(i - grasp) < k
        top = [label(exo, [(f"exo  frame {start + i}", None), (f"t = {ts[i] - ts[0]:.2f} s", None)]
                     + ([("GRASP", (255, 220, 0))] if g else []))]
        bottom = [label(np.full((PANEL_H, PANEL_W, 3), 30, np.uint8), [("side view", None)])] if rows == 2 else []
        for name, res in algos:
            d.qpos[:] = r.q_full_home
            d.qpos[r.arm_adr] = res["q"][i]
            mujoco.mj_forward(m, d)
            st = status[name][i]
            tag = {"fail": ("FAIL", (255, 60, 60)), "no_target": ("NO TARGET", (170, 170, 170))}.get(st)
            border = (255, 40, 40) if st == "fail" else None
            for cam, dst in ((cam_view, top), ("side", bottom if rows == 2 else None)):
                if dst is None:
                    continue
                ren.update_scene(d, camera=cam)
                dst.append(label(ren.render(), [(name, None)] + ([tag] if tag else [])
                                 + ([("GRASP", (255, 220, 0))] if g else []), border=border))
        frame = np.vstack([np.hstack(top)] + ([np.hstack(bottom)] if rows == 2 else []))
        if png is not None:
            Image.fromarray(frame).save(png_path(out_path, png))
            break
        writer.stdin.write(frame.tobytes())
    if writer:
        writer.stdin.close()
        writer.wait()
    ren.close()
    fails = {nm: sum(s == "fail" for s in st) for nm, st in status.items()}
    print(f"프레임 {len(idx)}개 (30 Hz, k={k}), 실패 프레임 {fails} → {png_path(out_path, png) if png is not None else out_path}")


def png_path(out_path, n):
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.with_name(f"{p.stem}_f{n}.png")


# ---------------------------------------------------------------- 입력 모으기

def read_ts(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return np.array([float(r["t_s"]) for r in csv.DictReader(f)])


def run_real(a):
    import metrics as M
    from replay.scene import load_obstacles
    ep_dir = Path(a.episode).expanduser()
    ep = ep_dir.name
    task = ep.split("_")[0]
    meta = json.load(open(ep_dir / "meta.json", encoding="utf-8"))
    labels = json.load(open(ep_dir / "labels.json", encoding="utf-8")) if (ep_dir / "labels.json").exists() else {}
    start, end = int(meta["exo_start_frame"]), int(meta["exo_end_frame"])
    ret = Path(a.retarget).expanduser()
    names = a.algos or sorted(p.name for p in ret.iterdir() if (p / f"{ep}.npz").exists())
    algos = [(n, dict(np.load(ret / n / f"{ep}.npz"))) for n in names]
    if not algos:
        raise SystemExit(f"{ret} 아래에 {ep}.npz가 없음")
    info, key = M.read_info(algos[0][1], a.identity_transform)
    for n, res in algos[1:]:
        if M.read_info(res, a.identity_transform)[1] != key:
            raise SystemExit(f"{n}: 같은 에피소드인데 npz info(좌표 변환 값)가 다름")
    transform = M.load_transform(a.identity_transform)
    to_robot = lambda p: np.asarray(transform(p, info), float)
    grasp = int(labels["grasp_frame"]) - start if labels.get("grasp_frame") is not None else None
    obstacles = load_obstacles(a.obstacles).get(ep) if a.obstacles else None
    render_episode(a.out, ep_dir / "exo.mov", start, end, read_ts(ep_dir / "exo_ts.csv"), algos, task,
                   REPO / "sync" / f"exo_calib_{task}.json", to_robot, obstacles, a.table_outline, grasp, a.view, a.png)


def demo_transform():
    """데모용 4번 변환 흉내: 테이블 좌표 → 로봇 좌표 (평행 이동·축척·회전). 로봇은 사람 어깨 자리, 테이블 쪽을 봄."""
    shoulder_table = np.array([0.12, 0.72, 0.30])       # 사람 오른쪽 어깨 근처 (T3 테이블 뒤쪽, 대략)
    shoulder_robot = np.array([0.0, 0.0, 0.36])         # iiwa14 관절 2 높이
    s = 1.3                                              # 로봇 팔 길이 / 사람 팔 길이 (대략)
    R_tb = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])   # 테이블 −y(작업자 → 카메라 쪽) = 로봇 +x
    return lambda p: shoulder_robot + s * R_tb @ (np.asarray(p, float) - shoulder_table)


def run_demo(a):
    """4번 결과 전 확인용: 실제 exo 영상 + 가짜 궤적 2개 (FAKE_smooth, FAKE_jumpy). 지표·결론에 쓰지 말 것."""
    from replay.replay import Replayer
    n_frames = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", a.video],
                                  capture_output=True, text=True, check=True).stdout.strip())
    fps = 30.0 if a.fps is None else a.fps
    start, end = 0, n_frames
    T = end - start
    ts = np.arange(T) / fps
    r = Replayer()
    home = r.q_full_home[r.arm_adr]
    w = 2 * np.pi * 0.25
    smooth = home + np.outer(np.sin(w * ts), [0.4, 0.2, 0.3, 0.25, 0.0, 0.3, 0.0])
    jumpy = smooth.copy()
    jumpy[T // 2:, 2] += 0.6                              # 가운데서 팔꿈치가 튀는 궤적 (해 전환 흉내)
    algos = []
    for name, q in (("FAKE_smooth", smooth), ("FAKE_jumpy", jumpy)):
        out = r.replay(q)
        tp, tq = out["tcp_pos"].copy(), out["tcp_quat"].copy()
        if name == "FAKE_jumpy":
            tp[T // 2:T // 2 + int(fps)] += [0.03, 0, 0]  # 1초 동안 목표에서 3 cm → FAIL 표시 확인
        algos.append((name, {"q": q, "target_pos": tp, "target_quat": tq}))
    render_episode(a.out, a.video, start, end, ts, algos, a.task, REPO / "sync" / f"exo_calib_{a.task}.json",
                   demo_transform(), None, a.table_outline, grasp=T // 3, view=a.view, png=a.png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="5-1 나란히 보기 영상 (사람 exo | 알고리즘별 로봇)")
    ap.add_argument("--episode", help="3번 결과 에피소드 폴더 (exo.mov, exo_ts.csv, meta.json, labels.json)")
    ap.add_argument("--retarget", help="4번 결과 폴더 (<알고리즘>/<에피소드ID>.npz)")
    ap.add_argument("--algos", nargs="*", help="알고리즘 폴더 이름 (기본: 그 에피소드 npz가 있는 것 모두)")
    ap.add_argument("--obstacles", default=str(REPO / "episodes" / "obstacles.json"))
    ap.add_argument("--table-outline", default=str(REPO / "metrics" / "table_outline.json"))
    ap.add_argument("--out", required=True, help="출력 mp4 (~/project 밖, 예: ~/out/신서연/videos/T1_E004.mp4)")
    ap.add_argument("--view", choices=["exo", "side", "both"], default="exo",
                    help="exo: 사람 영상과 같은 시점 / side: 옆 시점 / both: 두 줄")
    ap.add_argument("--png", type=int, help="이 번째 출력 프레임 하나만 PNG로 (확인용)")
    ap.add_argument("--identity-transform", action="store_true", help="4번 변환 대신 항등 (시험용)")
    ap.add_argument("--demo", action="store_true", help="4번 결과 전 확인용: --video + 가짜 궤적")
    ap.add_argument("--video", help="--demo: exo_front 영상")
    ap.add_argument("--task", default="T3", help="--demo: 보정·테이블 윤곽 태스크")
    ap.add_argument("--fps", type=float, help="--demo: 영상 fps (기본 30)")
    a = ap.parse_args()
    if a.demo:
        if not a.video:
            ap.error("--demo에는 --video")
        run_demo(a)
    else:
        if not (a.episode and a.retarget):
            ap.error("--episode와 --retarget (또는 --demo --video)")
        run_real(a)
