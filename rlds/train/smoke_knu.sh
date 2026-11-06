#!/bin/bash
# knu_retarget 로 50 스텝 짧은 학습: 우리 RLDS 가 OpenVLA 로더(패치 포함)와 학습을 끝까지 통과하는지 점검.
# 확인할 것: 오류 없이 시작·종료 / 손실 감소 / dataset_statistics.json 의 action mask 가 7개 모두 True
#   sbatch -p p02 --gres=gpu:1 -c 8 --mem=64G --time=02:00:00 --job-name knu_smoke_본인이름 \
#       -o ~/out/본인이름/knu_smoke.log rlds/train/smoke_knu.sh
# 끝나면 체크포인트(약 15 GB)는 지우고 dataset_statistics.json 만 남긴다.
source ~/use_openvla.sh
VLA=${VLA:-$HOME/vla}
OUT=${OUT:-$HOME/out/guhyeon/runs/smoke_knu}
export TMPDIR=$VLA/tmp
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$VLA/openvla"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$VLA/models/openvla-7b" \
  --data_root_dir "$VLA/data" \
  --dataset_name knu_retarget \
  --run_root_dir "$OUT" \
  --adapter_tmp_dir "$OUT/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --shuffle_buffer_size 10000 \
  --max_steps 50 \
  --save_steps 50 \
  --save_latest_checkpoint_only False \
  --wandb_project knu_retarget_smoke \
  --wandb_entity offline
echo "EXIT $?"
STATS=$(find "$OUT" -name dataset_statistics.json | head -1)
python -c "import json,sys; a=json.load(open(sys.argv[1]))['knu_retarget']['action']; print('ACTION_MASK', a['mask'])" "$STATS"
