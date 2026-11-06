#!/bin/bash
# 추론 점검을 GPU 노드에서 돌린다 (가이드 7번 6단계).
#   CKPT=~/out/guhyeon/runs/knu_retarget_c/<실행>--20000_chkpt EPISODES=10 \
#   sbatch -p p02 --gres=gpu:1 -c 4 --mem=48G --time=01:00:00 --job-name predict_check_본인이름 \
#       -o ~/out/본인이름/predict_check.log rlds/train/predict_check.sh
# 판정: summary.json 의 action_accuracy·l1_norm 이 학습 로그 마지막 값과 비슷하면 통과.
#       크게 다르면 모델이 아니라 추론 코드(가운데 자르기, 정규화 되돌리기) 문제.
set -e
source ~/use_openvla.sh
: "${CKPT:?CKPT (체크포인트 폴더) 를 지정}"
EPISODES=${EPISODES:-10}
OUT=${OUT:-$HOME/out/guhyeon/predict_check/$(basename "$CKPT")}
export TMPDIR=$HOME/vla/tmp
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/predict_check.py" --ckpt "$CKPT" --data-dir "${DATA_DIR:-$HOME/vla/data}" \
    --episodes "$EPISODES" --out "$OUT"
echo "결과: $OUT/summary.json, 스텝별 $OUT/steps.csv, 예측 궤적 $OUT/pred/"
