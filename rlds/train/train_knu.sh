#!/bin/bash
# 본 학습: OpenVLA-7B LoRA, knu_retarget (알고리즘 C 최종 결과), 20,000 스텝 — 역할 가이드 7번 4단계.
# 설정 근거: PROJECT_CONTEXT.md 5절 「OpenVLA 학습」 (배치 8 × 누적 2 는 시험 학습 최대 메모리 39.8 GB 로 확인)
#   sbatch -p p02 --gres=gpu:1 -c 8 --mem=64G --time=2-00:00:00 --job-name openvla_train_본인이름 \
#       -o ~/out/본인이름/openvla_train.log rlds/train/train_knu.sh
# - 스텝당 약 2.9초 -> 20,000 스텝 약 16시간 + 체크포인트 10번 저장(각 약 1분)
# - 체크포인트 <실행 폴더>--<스텝>_chkpt 각 약 15 GB × 10개 ≈ 150 GB. 저장 공간을 먼저 확인
# - 학습 전에 GPU 사용 시간대를 팀에 공유 (역할 가이드 7번 2단계)
source ~/use_openvla.sh
VLA=${VLA:-$HOME/vla}
OUT=${OUT:-$HOME/out/guhyeon/runs/knu_retarget_c}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-2000}
export TMPDIR=$VLA/tmp
mkdir -p "$OUT"
df -h "$OUT" | tail -1
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 30 > "$OUT/gpu_mem.csv" &
MON=$!
cd "$VLA/openvla"
echo "openvla commit $(git rev-parse --short HEAD), patch: $(git diff --stat | tail -1)"
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
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_latest_checkpoint_only False \
  --wandb_project knu_retarget \
  --wandb_entity offline
RC=$?
kill $MON
echo "EXIT $RC"
echo "PEAK_MIB $(sort -t, -k2 -n "$OUT/gpu_mem.csv" | tail -1)"
ls -d "$OUT"/*_chkpt
