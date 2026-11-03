#!/usr/bin/env bash
# 서버에서 영상 가공용 conda 환경(episodes)을 따로 만든다. 공유 robot 환경은 건드리지 않는다.
#   bash episodes/setup_env.sh [환경이름]      # 기본 환경 이름: episodes
set -euo pipefail

ENV_NAME="${1:-episodes}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 서버 홈이 이전 사용자의 venv라 PATH 가 꼬이므로 use_robot.sh 와 같은 방식으로 맞춘다
source ~/use_robot.sh >/dev/null 2>&1 || true
export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "환경 '$ENV_NAME' 이 이미 있음. 다른 이름을 주거나 'conda env remove -n $ENV_NAME' 후 다시 실행"
    exit 1
fi

conda create -n "$ENV_NAME" python=3.10 -y
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"

pip install -r "$HERE/requirements.txt"

# mediapipe 가 끌고 온 OpenCV 를 headless 하나로 정리 (두 종류가 섞이면 cv2 가 깨짐)
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
pip install --no-deps opencv-contrib-python-headless==4.11.0.86
python -c "import mediapipe as mp; assert hasattr(mp, 'solutions'), 'mediapipe 에 solutions 없음 (0.10.21 필요)'"
python -c "import cv2, mediapipe, scipy, numpy; print('ok', numpy.__version__)"

cat <<MSG

완료. 다음부터는 아래 줄로 켠다 (본인 홈의 ~/use_${ENV_NAME}.sh 로 저장해 두면 편함):
  conda activate $ENV_NAME
  export PATH=\$CONDA_PREFIX/bin:\$PATH
  export PYTHONNOUSERSITE=1
MSG
