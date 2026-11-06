#!/bin/bash
# 역할 가이드 7번 1단계: 우리 데이터 전에 LIBERO 로 200 스텝 학습해 설치를 점검한다.
#   확인할 것: 오류 없이 시작·종료하는가 / 손실이 내려가는가 / 최대 GPU 메모리
# 2단계 시작 설정(배치 8, 누적 2 = 유효 16)으로 돌리고, 200 스텝에 체크포인트 1개를 저장해
# --save_latest_checkpoint_only False 저장 경로까지 점검한다. 끝나면 체크포인트(약 15 GB)는 지운다.
#   sbatch -p p02 --gres=gpu:1 -c 8 --mem=64G --time=03:00:00 --job-name openvla_smoke_본인이름 \
#       -o ~/out/본인이름/openvla_smoke.log rlds/env/smoke_libero.sh
source ~/use_openvla.sh
VLA=${VLA:-$HOME/vla}
OUT=${OUT:-$HOME/out/guhyeon/runs/smoke_libero}
export TMPDIR=$VLA/tmp
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$OUT/gpu_mem.csv" &
MON=$!
cd "$VLA/openvla"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$VLA/models/openvla-7b" \
  --data_root_dir "$VLA/data/modified_libero_rlds" \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "$OUT" \
  --adapter_tmp_dir "$OUT/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 200 \
  --save_steps 200 \
  --save_latest_checkpoint_only False \
  --wandb_project knu_retarget_smoke \
  --wandb_entity offline
RC=$?
kill $MON
echo "EXIT $RC"
echo "PEAK_MIB $(sort -t, -k2 -n "$OUT/gpu_mem.csv" | tail -1)"
ls -la "$OUT"
