"""만든 knu_retarget RLDS 를 열어 형식과 값 범위를 확인한다 (6번 → 7번 넘기기 전 점검).

  source ~/use_openvla.sh
  python rlds/check_rlds.py --data-dir ~/vla/data

확인: 에피소드 수, 스텝 수, 이미지 형태, 상태·행동 7개 값 범위, 지시문 종류,
      행동이 다음 스텝 상태와 같은지 (action[k] == state[k+1]),
      관절 한계(robot/joint_limits.csv) 근처(1e-3 rad 이내) 비율
"""
import argparse, csv, os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def joint_limits():
    f = Path(__file__).resolve().parent.parent / "robot" / "joint_limits.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    return np.array([float(r["lower_rad"]) for r in rows]), np.array([float(r["upper_rad"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    b = tfds.builder("knu_retarget", data_dir=os.path.expanduser(a.data_dir))
    ds = b.as_dataset(split="train")
    n_ep, steps, instr, shapes, bad_link = 0, [], Counter(), Counter(), 0
    states, actions = [], []
    for ep in ds:  # steps 가 중첩 Dataset 이라 에피소드마다 따로 numpy 로 꺼낸다
        n_ep += 1
        s = list(ep["steps"].as_numpy_iterator())
        st = np.stack([x["observation"]["state"] for x in s])
        ac = np.stack([x["action"] for x in s])
        states.append(st); actions.append(ac); steps.append(len(ac))
        shapes[s[0]["observation"]["image"].shape] += 1
        instr.update(x["language_instruction"].decode() for x in s)
        if len(ac) > 1 and not np.allclose(ac[:-1], st[1:], atol=1e-6):
            bad_link += 1
    S, A = np.concatenate(states), np.concatenate(actions)
    print(f"에피소드 {n_ep}, 스텝 {len(A)} (에피소드당 최소 {min(steps)} / 중앙값 {int(np.median(steps))} / 최대 {max(steps)})")
    print("이미지 형태", dict(shapes), " 지시문", dict(instr))
    print("행동 = 다음 상태가 아닌 에피소드", bad_link)
    np.set_printoptions(precision=3, suppress=True)
    print("행동 최소", A.min(0), "\n행동 최대", A.max(0))
    lim = joint_limits()
    if lim is not None:
        near = (np.abs(A - lim[0]) < 1e-3) | (np.abs(A - lim[1]) < 1e-3)
        print("관절 한계 1e-3 rad 이내 비율 (관절별)", near.mean(0))
    assert bad_link == 0, "행동이 다음 스텝 상태와 다른 에피소드가 있음"
    print("CHECK_OK")


if __name__ == "__main__":
    main()
