#!/bin/bash
# 태스크 하나 싱크·보정 전체 (계산 노드). 사용: sbatch ... run_task.sh <T> <front_rot> [병렬 수]
# 입력 ~/data/raw/<T>/<camera>/<T>_E###, 체커보드 <T>_check
# 출력 ~/out/신서연/{claps_<T>/, sync_<T>.csv, exo_calib_<T>.json, calib_<T>/}
set -u
T=$1; FROT=$2; NP=${3:-32}
S=~/project/sync
M=~/out/신서연/sync_tools/hand_landmarker.task
# TAG: 출력 이름 꼬리표 (설정을 바꿔 다시 돌릴 때), SYNC_ARGS: make_sync_csv.py 추가 인자
TAG=${TAG:-}; SYNC_ARGS=${SYNC_ARGS:-}
O=~/out/신서연; OUT=$O/claps_$T$TAG; RAW=~/data/raw/$T
PY=~/miniconda3/envs/episodes/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
CAMS="exo_front exo_left exo_right ego"
ROTS="exo_front=$FROT exo_left=0 exo_right=0 ego=0"
mkdir -p $OUT
echo "== $T$TAG ($SYNC_ARGS) 시작 $(date) $(hostname) front_rot=$FROT git=$(git -C ~/project rev-parse --short HEAD)"
JOBS=$OUT/jobs.txt; : > $JOBS
for c in $CAMS; do r=0; [ $c = exo_front ] && r=$FROT
  for d in $RAW/$c/${T}_E*/; do echo "$c $(basename $d) $r" >> $JOBS; done; done
echo "1단계 작업 $(wc -l < $JOBS)개"
xargs -P $NP -L 1 bash -c "$PY -I $S/detect_claps.py --raw $RAW --cameras \"\$0\" --episodes \"\$1\" --rot \"\$0=\$2\" --model $M --out $OUT 2>&1 | grep -E '^T[0-9]|Error|Traceback'" < $JOBS
echo "1단계 끝 $(date)"
for r in 1 2 3 4; do
  echo "== 라운드 $r 표 계산"
  $PY -I $S/make_sync_csv.py --claps $OUT --ref exo_front --cameras exo_left exo_right ego \
      --out $O/sync_$T$TAG.csv $SYNC_ARGS --guide-out $OUT/guide_r$r.json --report $OUT/picks.csv
  n=$($PY -I -c "import json;g=json.load(open('$OUT/guide_r$r.json'));g.pop('_half',None);print(sum(len(v) for v in g.values()))")
  [ "$n" = "0" ] && break
  echo "== 라운드 $r 안내 검출 $n"
  # 안내 검출은 (에피소드, 카메라)마다 결과 json 이 따로라 나눠서 병렬로 돌린다
  GD=$OUT/guide_r${r}_split; rm -rf $GD; mkdir -p $GD
  $PY -I -c "
import json, sys
g = json.load(open('$OUT/guide_r$r.json')); h = g.pop('_half', {})
for ep, cams in g.items():
    for cam, ts in cams.items():
        json.dump({ep: {cam: ts}, '_half': {k: v for k, v in h.items() if k == cam}}, open(f'$GD/{ep}_{cam}.json', 'w'))
"
  ls $GD/*.json | xargs -P $NP -I{} bash -c "$PY -I $S/detect_claps.py --raw $RAW --cameras $CAMS --rot $ROTS --model $M --out $OUT --guide {} 2>&1 | grep -E '^T[0-9]|Error|Traceback'"
done
echo "== 보정"
$PY -I $S/make_exo_calib.py --raw $RAW --calib-folder ${T}_check \
    --cameras exo_front=front exo_left=left exo_right=right --ref front \
    --valid-for "$T 세션 (${T}_check)" --out $O/exo_calib_$T$TAG.json --figs $O/calib_$T$TAG
echo "== $T 끝 $(date)"
