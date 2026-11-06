# OpenVLA 학습 환경 켜기:  source ~/use_openvla.sh
# 서버 홈에 이 파일을 복사해 둔다:  cp rlds/env/use_openvla.sh ~/use_openvla.sh
# use_robot.sh 와 같은 이유(홈의 옛 venv, ~/.local 패키지)로 PATH 와 user site 를 막는다.
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate openvla
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONNOUSERSITE=1
# ~/.bashrc 의 LD_LIBRARY_PATH(CUDA 12 cuDNN 등)가 PyTorch(cu118)의 cuDNN 을 가로채
# "GET was unable to find an engine to execute this computation" 오류가 난다. 필요한 CUDA 라이브러리는 pip 휠에 들어 있다.
unset LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
export HF_HOME=$HOME/vla/hf_cache
export WANDB_MODE=offline
echo "openvla 환경 활성화됨: $(which python)"
