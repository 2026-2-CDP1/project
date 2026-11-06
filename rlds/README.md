# rlds — VLA 데이터셋 변환 · OpenVLA 학습 준비 (6·7번, 정구현)

알고리즘 C(EAIK) 리타게팅 결과를 OpenVLA 학습 형식(RLDS, 데이터셋 이름 `knu_retarget`)으로 바꾸고, OpenVLA LoRA 학습을 돌린다.
결정 사항과 이유는 `PROJECT_CONTEXT.md` 5절 「OpenVLA 학습」.

---

## 1. OpenVLA 학습 환경 (`env/`)

공유 `robot` 환경과 **별도**의 `openvla` conda 환경이다. 서버 코드·모델·데이터는 `~/vla/`(레포 밖)에 둔다.

| 파일 | 하는 일 |
|---|---|
| `env/setup_openvla.sh` | `openvla` 환경 설치 (로그인 서버, 약 20분) |
| `env/use_openvla.sh` | 환경 켜기. 서버 홈에 복사해 `source ~/use_openvla.sh` |
| `env/download.sh` | 모델 `openvla-7b`(15 GB), LIBERO 시험 데이터(1.8 GB) 받기 |
| `env/gpucheck.sh` | GPU 노드에서 CUDA·flash-attn 동작 확인 |
| `env/smoke_libero.sh` | LIBERO 200스텝 시험 학습 (가이드 1단계) |

### 처음 한 번
```bash
cd ~/project
bash rlds/env/setup_openvla.sh                       # 설치
cp rlds/env/use_openvla.sh ~/use_openvla.sh          # 활성화 스크립트
nohup bash rlds/env/download.sh > ~/out/본인이름/openvla_download.log 2>&1 &
sbatch -p p02 --gres=gpu:1 -c 2 --mem=16G --time=00:10:00 --job-name gpucheck_본인이름 \
    -o ~/out/본인이름/openvla_gpucheck.log rlds/env/gpucheck.sh
```

### 설치된 버전 (2026-11-06)
역할 가이드 7번의 검증된 버전과 같다: torch 2.2.0+cu118, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, peft 0.11.1, flash-attn 2.5.5, tensorflow 2.15.0, tensorflow-datasets 4.9.3. OpenVLA 저장소 commit `c8f03f4`.

### 설치하며 막힌 곳 (스크립트에 반영됨)
| 증상 | 원인 | 해결 |
|---|---|---|
| `CondaError: Run 'conda init'` | 비대화형 셸 | `conda shell.bash hook`으로 켬 |
| flash-attn `Invalid cross-device link` | pip 임시 폴더와 캐시가 다른 파일시스템 | 미리 빌드된 휠을 직접 받아 설치, `TMPDIR`를 `~/vla/tmp`로 |
| `cannot import name 'runtime_version' from 'google.protobuf'` | `tensorflow-metadata` 최신(1.21)이 TF 2.15와 충돌 | `tensorflow-metadata==1.14.0`, `wandb==0.16.6` |
| 학습 첫 스텝 `GET was unable to find an engine` | `~/.bashrc`의 `LD_LIBRARY_PATH`가 CUDA 12 cuDNN을 먼저 잡음 | `use_openvla.sh`에서 `unset LD_LIBRARY_PATH` |
| sbatch에서 `No module named 'torch'` | `--wrap`은 sh로 실행돼 `source`가 안 됨 | bash 스크립트로 제출 |

### LIBERO 시험 학습 결과 (2026-11-06, 가이드 1단계)
| 항목 | 결과 |
|---|---|
| 시작·종료 | 200스텝 9분 40초, 체크포인트 저장까지 정상 |
| 손실 | 12.0 → 약 3.0 (행동 정확도 13% → 약 40%, L1 0.35 → 0.19) |
| 최대 GPU 메모리 | **39.8 GB** (배치 8, 누적 2) → 48 GB 안에 들어가 이 설정을 그대로 씀 |
| 체크포인트 | `...--200_chkpt` 폴더로 따로 저장, 개당 15 GB, 저장 약 1분 |
| 속도 | 스텝당 약 2.9초 → 20,000스텝 약 16시간 + 저장 시간 |

학습 기록은 wandb offline이라 서버 `~/vla/openvla/wandb/`에만 남는다.
