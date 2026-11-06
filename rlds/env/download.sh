#!/bin/bash
# OpenVLA 모델(약 15 GB)과 환경 점검용 LIBERO 데이터 한 묶음(약 1.8 GB)을 받는다 — 역할 가이드 7번 1단계.
#   nohup bash rlds/env/download.sh > ~/out/guhyeon/openvla_download.log 2>&1 &
# 가이드의 `git clone git@hf.co:...` 는 Hugging Face SSH 키가 필요해서 huggingface-cli 로 받는다.
set -e
source ~/use_openvla.sh >/dev/null
VLA=${VLA:-$HOME/vla}
export TMPDIR=$VLA/tmp
huggingface-cli download openvla/openvla-7b --local-dir "$VLA/models/openvla-7b"
echo MODEL_DONE
# LIBERO 수정판 중 시험에 쓰는 libero_spatial_no_noops 만 (전체는 약 10 GB). OpenVLA configs.py 에 이미 등록된 이름
huggingface-cli download openvla/modified_libero_rlds --repo-type dataset \
    --include "libero_spatial_no_noops/*" --local-dir "$VLA/data/modified_libero_rlds"
echo DATA_DONE
du -sh "$VLA/models/openvla-7b" "$VLA/data/modified_libero_rlds"
