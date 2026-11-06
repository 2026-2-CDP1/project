#!/bin/bash
# OpenVLA 학습 환경(openvla) 설치 — 역할 가이드 7번 「준비」. 공유 robot 환경과 별개.
#   bash rlds/env/setup_openvla.sh            (로그인 서버에서, 약 20분)
# 2026-11-06 서버에서 실제로 설치하며 막힌 곳을 고친 순서 그대로다.
set -eo pipefail

VLA=${VLA:-$HOME/vla}                 # OpenVLA 코드·모델·데이터를 두는 곳 (~/project 밖)
LOG_DIR=${LOG_DIR:-$HOME/out/guhyeon}
mkdir -p "$VLA/tmp" "$VLA/wheels" "$LOG_DIR"

# 비대화형 셸에서는 conda init 이 안 돼 있으므로 hook 으로 켠다 (use_robot.sh 방식은 여기서 실패)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
export PYTHONNOUSERSITE=1
export TMPDIR=$VLA/tmp                # pip 임시 폴더를 NFS 같은 곳에 두어야 휠 이동(cross-device link) 오류가 안 남
conda env list | awk '{print $1}' | grep -qx openvla || conda create -n openvla python=3.10 -y
conda activate openvla
export PATH=$CONDA_PREFIX/bin:$PATH
echo "== python: $(which python)"

# 1) PyTorch: 서버 드라이버(525~530)에 맞춰 CUDA 11.8 휠 (가이드 검증 버전 2.2.0)
pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# 2) OpenVLA 코드 (저장소가 transformers 4.40.1 · tokenizers 0.19.1 · timm 0.9.10 · peft 0.11.1 등을 고정)
cd "$VLA"
[ -d openvla ] || git clone https://github.com/openvla/openvla.git
cd openvla
echo "== openvla commit: $(git rev-parse HEAD)"
pip install --no-cache-dir -e .
pip install --no-cache-dir packaging ninja
ninja --version

# 3) flash-attn 2.5.5: pip 빌드 경로는 캐시 이동에서 실패해서, 미리 빌드된 휠을 직접 받는다
W=flash_attn-2.5.5+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
[ -f "$VLA/wheels/$W" ] || curl -sSLf -o "$VLA/wheels/$W" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/$W"
pip install --no-cache-dir "$VLA/wheels/$W"

# 4) 데이터 도구. tensorflow-metadata 최신(1.21)은 TF 2.15 의 protobuf 와 충돌 -> 1.14, 그에 맞춰 wandb 0.16.6
pip install --no-cache-dir tensorflow-datasets==4.9.3 "tensorflow-metadata==1.14.0" "wandb==0.16.6" "huggingface_hub[cli]"
# 5) 변환 코드(rlds/make_knu_retarget.py)가 영상을 읽는 데 씀
pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

python - <<'PY'
import torch, torchvision, transformers, tokenizers, timm, peft, flash_attn, tensorflow as tf, tensorflow_datasets as tfds, cv2
print("VERSIONS", "torch", torch.__version__, "cuda", torch.version.cuda, "torchvision", torchvision.__version__,
      "transformers", transformers.__version__, "tokenizers", tokenizers.__version__, "timm", timm.__version__,
      "peft", peft.__version__, "flash_attn", flash_attn.__version__, "tf", tf.__version__, "tfds", tfds.__version__,
      "cv2", cv2.__version__)
PY
pip freeze > "$LOG_DIR/openvla_env_freeze.txt"
echo "SETUP_DONE (설치 목록: $LOG_DIR/openvla_env_freeze.txt)"
