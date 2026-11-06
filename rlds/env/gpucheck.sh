#!/bin/bash
# GPU 노드에서 openvla 환경이 CUDA·flash-attn 을 실제로 쓰는지 확인 (10분 이내).
#   sbatch -p p02 --gres=gpu:1 -c 2 --mem=16G --time=00:10:00 --job-name gpucheck_본인이름 \
#       -o ~/out/본인이름/openvla_gpucheck.log rlds/env/gpucheck.sh
# sbatch --wrap "source ..." 은 sh 로 실행돼 환경이 안 켜지므로 반드시 이런 bash 스크립트로 제출한다.
source ~/use_openvla.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      "flash_attn", flash_attn.__version__, "out", tuple(flash_attn_func(q, q, q).shape))
PY
