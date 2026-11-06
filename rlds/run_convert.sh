#!/bin/bash
# 리타게팅 결과 -> 에피소드 묶음(make_knu_retarget.py) 을 Slurm CPU 노드에서 돌린다.
#   RETARGET=~/out/hyojeong/retarget/<실행>/C_eaik EPISODES=~/out/seohyeon/<에피소드 폴더> \
#   sbatch -p p02 -c 8 --mem=32G --time=02:00:00 --job-name knu_convert_본인이름 \
#       -o ~/out/본인이름/knu_convert.log rlds/run_convert.sh
# 결과: $SRC/<에피소드ID>.npz, $SRC/summary.csv (기본 SRC=~/vla/data/knu_retarget_src, 다시 만들 때 지우고 새로 씀)
set -e
source ~/use_openvla.sh
: "${RETARGET:?RETARGET (알고리즘 결과 폴더, 예: .../C_eaik) 를 지정}"
: "${EPISODES:?EPISODES (3번 에피소드 폴더들의 상위 폴더) 를 지정}"
SRC=${SRC:-$HOME/vla/data/knu_retarget_src}
export TMPDIR=${TMPDIR_VLA:-$HOME/vla/tmp}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "$SRC"
python "$HERE/make_knu_retarget.py" --retarget "$RETARGET" --episodes "$EPISODES" --out "$SRC"
du -sh "$SRC"
echo CONVERT_DONE
