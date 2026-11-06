"""학습한 OpenVLA 체크포인트로 학습 데이터를 다시 예측해, 학습 로그의 행동 정확도·L1 이 재현되는지 본다.
역할 가이드 7번 6단계. 재현되면 이 추론 코드(영상 전처리·가운데 자르기·정규화 되돌리기)를 8번 평가에 넘긴다.
재현되지 않으면 모델이 아니라 추론 코드 쪽 문제다.

  source ~/use_openvla.sh
  python rlds/train/predict_check.py --ckpt ~/out/guhyeon/runs/knu_retarget_c/<실행>--20000_chkpt \\
      --data-dir ~/vla/data --episodes 10 --out ~/out/guhyeon/predict_check

학습 로그와 같은 방식으로 잰다 (vla-scripts/finetune.py):
  - 정규화: BOUNDS_Q99. norm = clip(2 (x - q01) / (q99 - q01) - 1, -1, 1)  (mask 가 True 인 차원만)
  - 토큰: [-1, 1] 을 256 칸으로 나눈 칸 번호. action_accuracy = 예측 칸 == 정답 칸 비율
  - l1_loss = 정규화 공간에서 |예측 칸 중심 - 정답 칸 중심| 평균
image_aug True 로 학습했으므로 추론 이미지는 가운데 90% 면적을 잘라 224 로 키운다 (experiments/robot/openvla_utils.py 와 같음).

출력: <out>/steps.csv (스텝별), <out>/summary.json, <out>/pred/<에피소드ID>.npz (예측 관절각 q_pred, 정답 q_gt — 8번이 replay 로 재생)
"""
import argparse, csv, json, os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

N_BINS = 256
KEY = "knu_retarget"


def center_crop(img_uint8, crop_scale=0.9):
    """openvla_utils.crop_and_resize 와 같은 계산: 가운데 crop_scale 면적을 잘라 224x224 로."""
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(img_uint8), tf.float32)[None]
    s = float(np.sqrt(crop_scale))
    off = (1 - s) / 2
    x = tf.image.crop_and_resize(x, [[off, off, off + s, off + s]], [0], (224, 224))[0]
    x = tf.image.convert_image_dtype(tf.clip_by_value(x, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(x.numpy()).convert("RGB")


def to_bins(a, q01, q99, mask):
    """행동(rad) -> 정규화 [-1,1] -> 256칸 번호와 칸 중심 (학습 때 ActionTokenizer 와 같은 칸)."""
    n = np.where(mask, np.clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1), np.clip(a, -1, 1))
    edges = np.linspace(-1, 1, N_BINS)
    idx = np.clip(np.digitize(n, edges), 1, N_BINS - 1)          # ActionTokenizer: digitize 후 1..255
    centers = (edges[:-1] + edges[1:]) / 2
    return idx, centers[idx - 1]


def load(ckpt, stats_file):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
    stats_file = stats_file or Path(ckpt) / "dataset_statistics.json"
    vla.norm_stats = json.load(open(stats_file))
    return processor, vla


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="체크포인트 폴더 (<실행>--<스텝>_chkpt)")
    ap.add_argument("--stats", help="dataset_statistics.json (기본: 체크포인트 폴더 안)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--episodes", type=int, default=10, help="앞에서부터 몇 개 에피소드를 볼지")
    ap.add_argument("--no-center-crop", action="store_true", help="image_aug False 로 학습했을 때만")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    (out / "pred").mkdir(parents=True, exist_ok=True)

    processor, vla = load(os.path.expanduser(a.ckpt), a.stats and os.path.expanduser(a.stats))
    st = vla.norm_stats[KEY]["action"]
    q01, q99 = np.array(st["q01"]), np.array(st["q99"])
    mask = np.array(st.get("mask", [True] * len(q01)))

    ds = tfds.builder(KEY, data_dir=os.path.expanduser(a.data_dir)).as_dataset(split="train")
    rows, acc_all, l1_all = [], [], []
    for ep in ds.take(a.episodes):
        eid = ep["episode_metadata"]["episode_id"].numpy().decode()
        steps = list(ep["steps"].as_numpy_iterator())
        preds, gts = [], []
        for k, s in enumerate(steps):
            img = s["observation"]["image"]
            image = Image.fromarray(img).convert("RGB") if a.no_center_crop else center_crop(img)
            instr = s["language_instruction"].decode()
            prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            pred = vla.predict_action(**inputs, unnorm_key=KEY, do_sample=False)
            gt = s["action"]
            pi, pc = to_bins(pred, q01, q99, mask)
            gi, gc = to_bins(gt, q01, q99, mask)
            acc, l1 = float((pi == gi).mean()), float(np.abs(pc - gc).mean())
            acc_all.append(acc); l1_all.append(l1)
            preds.append(pred); gts.append(gt)
            rows.append({"episode_id": eid, "step": k, "action_accuracy": round(acc, 4), "l1_norm": round(l1, 4),
                         "abs_err_rad_mean": round(float(np.abs(pred - gt).mean()), 4)})
        np.savez(out / "pred" / f"{eid}.npz", q_pred=np.array(preds, np.float32), q_gt=np.array(gts, np.float32))
        print(f"{eid}: steps {len(steps)} acc {np.mean(acc_all[-len(steps):]):.3f} l1 {np.mean(l1_all[-len(steps):]):.3f}", flush=True)

    with open(out / "steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {"ckpt": a.ckpt, "episodes": a.episodes, "steps": len(rows), "center_crop": not a.no_center_crop,
               "action_accuracy": float(np.mean(acc_all)), "l1_norm": float(np.mean(l1_all)),
               "abs_err_rad_mean": float(np.mean([r["abs_err_rad_mean"] for r in rows]))}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
