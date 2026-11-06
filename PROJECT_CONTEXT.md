# PROJECT_CONTEXT — 리타게팅 알고리즘 비교 연구

> 이 문서는 팀원과 코드 어시스턴트(VS Code 등)가 프로젝트 맥락을 한 번에 파악하기 위한 요약입니다.
> 연구 목적·평가 방법의 기준은 멘토 제공 「학생 안내서」와 「역할 가이드」입니다. 이 문서와 어긋나면 그쪽을 따릅니다.
> 최종 갱신: 2026-11-04

---

## 0. 팀원 전달 사항 (먼저 읽어주세요)

### 서버 사용법
```bash
# 1) 접속 (비밀번호는 별도 공유)
ssh coss37@155.230.81.91 -p 10000

# 2) 환경 활성화 — 매번
source ~/use_robot.sh

# 3) 오래 걸리는 작업이면 tmux 먼저
tmux new -s 본인이름_작업        # 재접속: tmux attach -t 본인이름_작업

# 4) GPU 노드 할당 (GPU 필요 없으면 --gres 생략)
srun --gres=gpu:1 -p p02 --job-name "작업_본인이름" --pty bash

# 5) 최신 코드 받고 실행 — 산출물은 ~/project 밖(~/out/본인이름)에 저장
cd ~/project && git pull
mkdir -p ~/out/본인이름
python ... --out ~/out/본인이름

# 6) 끝나면 반납
exit
```
- 산출물을 git에 올리는 방법은 6절 참고 (서버에서 결과 브랜치로 바로 커밋, 또는 `scp`로 노트북에 가져와 커밋)

### 작업 파이프라인
```
[각자 노트북]     브랜치 생성 → 코드 작성 → commit → push → PR → main에 merge
[서버 ~/project]  main 고정 → git pull → 실행만
[산출물]          서버 별도 폴더(worktree)에서 결과 브랜치 → `git c 본인이름` 커밋 → push → PR  (6절)
```
- 브랜치 이름: `역할폴더/작업내용` (예: `robot/iiwa14-model`, `retarget/dls`, `metrics/definitions`)
- `main`에 직접 push하지 않고 PR로 합칩니다
- 서버에서 돌려보고 싶은 코드는 먼저 PR로 main에 합친 뒤 pull 합니다

### 주의사항
- **서버 `~/project`에서 `git checkout`·파일 수정·`git commit` 금지** — 4명이 같은 계정·같은 폴더를 써서, 바꾸면 다른 사람의 실행·pull이 깨집니다
- 서버에서 커밋은 `~/project`가 아니라 `~/work/본인이름`에서만, `git commit` 대신 `git c 본인이름` (예: `git c 신서연 -m "메시지"`) — 전체 순서는 6절 「서버에서 커밋하는 법」
- **산출물은 `~/project` 안에 저장하지 않기** → `~/out/본인이름/`에 저장 (코드는 출력 경로를 인자로 받게 작성, 예: `--out`)
  - `~/project` 안에 생긴 파일이 나중에 main에 같은 경로로 합쳐지면 다음 사람의 `git pull`이 막히고, 두 사람이 같은 경로에 덮어쓰게 됨
- `git pull`이 에러로 멈추면 임의로 지우거나 되돌리지 말고 팀 채팅에 알려주세요 (누가 서버 `~/project`에서 커밋했거나 파일을 고쳐 둔 경우 멈출 수 있음. 커밋으로 갈라진 경우는 `pull.ff only` 설정으로 merge 대신 에러가 나게 해둠)
- 그냥 `conda activate robot` 금지 → 반드시 `source ~/use_robot.sh` (이유는 5절)
- `robot` 환경 패키지는 필요할 때 담당자가 직접 설치 — 반드시 5절 절차(백업 → 미리보기 → `python -m pip` 설치 → 확인 → 팀 알림·5절 표 갱신)대로
- GPU는 `--job-name "작업_본인이름"`으로 잡고, 끝나면 바로 `exit`
- 로그인 서버(ABRM02)에서 무거운 계산 직접 돌리지 않기 — 모든 팀 공용 입구
- p02는 7일 넘으면 강제 종료 → 긴 학습은 체크포인트 저장
- 남이 띄운 프로세스·tmux 세션 종료하지 않기, 홈의 기존 conda 환경(bert, lstm, tf 등) 삭제하지 않기
- 영상·`npy`/`npz`·체크포인트 같은 대용량은 커밋하지 않기 (`.gitignore`로 제외됨)
- 비밀번호·API 키는 레포·문서에 적지 않기

---

## 1. 프로젝트 개요

- 경북대 종합설계프로젝트 × ㈜크라우드웍스 산학협력 (멘토: 오진성 책임연구원)
- 팀원: 정구현(팀장), 차서현, 신서연, 제효정 — 컴퓨터학부 4학년
- 주제: 사람의 ego/exo 영상 → 손목 궤적 추출 → **리타게팅 알고리즘 3종을 같은 데이터로 비교** → 대표 알고리즘 데이터로 OpenVLA LoRA 1회 학습(파이프라인 실증)

### 전체 흐름
```
수집(ego·exo 촬영) → 가공(키포인트·행동 라벨) → 싱크 확인(지난 학기 검출 알고리즘 재사용)
→ 리타게팅(A·B·C) ───┬→ [연구 본체] MuJoCo 재생·지표 측정 → 알고리즘 × 케이스 비교표
                      └→ [보조] 대표 알고리즘 1개 → OpenVLA LoRA 학습 → MSE·CI-MSE 점검
```

### 비교 알고리즘
| 축 | 알고리즘 | 방식 | 팔꿈치 반영 | 구현 |
|---|---|---|---|---|
| A | DLS IK | 자코비안 반복, 관절 한계는 계산 후 자르기 (clip) | 손목만 (v1.2) | pinocchio 자체 구현 |
| B | 제약 QP IK | 반복, 관절 한계·속도를 QP 제약으로 | 손목만 (v1.2) | mink (MuJoCo 모델 사용) |
| C | 닫힌 해 | 반복 없이 기하 공식으로 해 계산 | 사람 SEW 각으로 팔꿈치를 고정해 해석해를 구하고 직전 프레임과 가장 가까운 해 선택 (한계 밖이면 가장 가까운 각, v1.3 11/7 확정. 이전: 3번 관절 격자 탐색) | EAIK (C++ 기반 파이썬 패키지, URDF 자동 분해) |

- 세 알고리즘 모두 **학습하지 않음**. 매 프레임 독립적으로 IK를 푸는 계산 절차.
- **C는 IK-Geo → EAIK로 변경 (2026-11-04).** 오프라인 회의에서 멘토 의견: EAIK는 C++ 기반에 `pip`로 바로 쓰고, URDF(`robot/robot.urdf`)에서 구조를 자동 분해함. 내부는 IK-Geo와 같은 기하 부분문제라 "닫힌 해" 축의 의미는 같음
  - EAIK는 7자유도를 관절 하나를 고정해 6자유도로 풀고, SEW 각을 직접 받지 않음 → 3번 관절 격자 탐색 래퍼(팔 회전각 맞추기)를 4번이 작성 (`retarget/eaik_ik.py`). 논문에는 "EAIK + 팔 회전각 1차원 탐색"으로 적음
- **D(PyRoki, 병렬 시드 최적화)는 제외 (2026-11-04).**
- 팔꿈치는 위치가 아니라 **팔 회전각**(어깨-손목 축 둘레 팔꿈치 방향)으로 맞춤. 사람·로봇 팔 길이가 달라 위치는 맞출 수 없고, 회전각은 길이와 무관함. 입력: 어깨·팔꿈치·손목 3D + 손 방향 (3번 키포인트에 모두 있음)
- **세 알고리즘 모두 기존 알고리즘 그대로 (2026-11-06 팀 결정, T10 대체).** A·B는 손목만 받고, C는 사람 SEW 각이 원래 입력. 멘토 11/6: 세 알고리즘 결과는 모두 관절각 7개, 팔꿈치는 FK로 구함 → 팔 회전각 차이는 알고리즘 특성으로 보고. (이전 11/05 결정: 세 알고리즘 모두 사람 팔꿈치 입력)
  - 사람 팔꿈치가 안 보이는 프레임은 셋 다 직전 팔꿈치를 유지, 보간하지 않음. 맨 앞 프레임은 처음 보인 팔꿈치 방향을 씀 (자세한 규칙: `retarget/README.md`)
  - 그래서 5-2의 팔 회전각 차이는 순위가 아닌 참고 지표 (`metrics/definitions.md` 2.2)
- 학습되는 것은 OpenVLA 하나뿐이며, 알고리즘 순위 비교에 쓰지 않음.

### 평가 지표 (5-2, 학습 없이 재생 결과로 계산)
관절 한계 위반율 / 관절 한계 포화율(보조) / 자기 충돌률 / 최대 침투 깊이 / IK 실패율 / 저크 RMS / 파지점 거리 / 추종 오차(참고용, 순위 미사용)

### 주장 가능 범위
"케이스 X에서 알고리즘 A가 **지표상** B보다 낫다"까지. 실제 로봇 작업 성공은 주장 불가.

---

## 2. 확정된 결정사항

| 항목 | 결정 |
|---|---|
| 타깃 로봇 | **KUKA LBR iiwa14 (R820)**, 7자유도 single-arm, **그리퍼 없음** → TCP = 플랜지 (MJCF `attachment_site`) (2026-11-03 결정) |
| 시뮬레이터 | **MuJoCo** (Isaac Sim 불가 판정, 4절 참고) |
| 촬영 장비 | iPhone (멘토 추천 앱: NeRFCapture / Spectacular Rec / AR Recorder — 연속 녹화가 기본인 Spectacular Rec 우선 테스트) |
| 비교 알고리즘 | A DLS / B mink / C **EAIK** (2026-11-04: C를 IK-Geo → EAIK, D PyRoki 제외) |
| 팔꿈치 입력 | **A·B 손목만, C는 사람 SEW 각이 원래 입력** (2026-11-06, "기존 알고리즘 그대로"). 안 보인 프레임은 직전 팔꿈치 유지 → `retarget/README.md` |
| 패키지 설치 | 서버 `robot` 환경 패키지는 **필요할 때 담당자가 직접 설치** (2026-11-05). 절차는 5절 |
| 학습 모델 | OpenVLA-7B + LoRA, **알고리즘 C(EAIK) 데이터로 1회** (2026-11-06). 데이터 형식 **RLDS**, max_steps **20,000**, 평가는 기존 계획 유지 → 5절 「OpenVLA 학습」 |
| 코드 관리 | GitHub `2026-2-CDP1/project`, 브랜치 → PR → main |

---

## 3. GPU 서버

### 구조
```
[노트북] ──ssh──▶ [로그인 서버 ABRM02] ──srun──▶ [GPU 노드 DIS01~06]
                         │                              │
                         └────── 공유 저장소 /abr (NFS) ──┘
```

- 로그인 서버: `ssh coss37@155.230.81.91 -p 10000`
  - ⚠️ 사업단 PDF 본문의 `155.230.135.209`는 **외부망에서 접속 안 됨**
  - 비밀번호는 별도 공유 (이 파일에 적지 말 것)
- 계정 `coss37`은 **팀원 4명이 공유**하는 리눅스 계정 1개
- 사용 가능 파티션: **p02만** (DIS01~06). DIS04 down, DIS05 drain 상태였음
- p02 타임리밋: **7일** (초과 시 작업 강제 종료 → 체크포인트 필수)
- 계정당 자원 상한(QOS) 있음 — 공유 계정이라 4명 합산

### 사양 (2026-10-05 확인)
| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 48GB |
| 드라이버 / CUDA | 525.105.17 / 12.0 |
| OS | Ubuntu 20.04 (로그인 서버·GPU 노드 동일), GLIBC 2.31 |
| 저장소 | `10.0.0.100:/data` → `/abr` (73TB 중 약 91% 사용, 타 팀과 공유) |
| 홈 | `/abr/coss37` (NFS — 어느 노드에서든 동일하게 보임) |
| 컨테이너 | Singularity 있음 (`/usr/local/bin/singularity`) |

### Slurm 기본
```bash
sinfo                      # 파티션·노드 상태
squeue -u coss37           # 우리 작업 목록
srun --gres=gpu:1 -p p02 --job-name "작업_본인이름" --pty bash   # GPU 노드 할당
exit                       # GPU 세션 종료 = 자원 반납
scancel <JOBID>            # 강제 반납
```
- GPU 필요 없는 무거운 계산: `srun -p p02 --job-name "..." --pty bash` (gres 생략)
- 오래 걸리는 작업은 **로그인 서버에서 tmux 먼저** 켜고 그 안에서 srun
  ```bash
  tmux new -s 이름      # 생성
  tmux attach -t 이름   # 재접속
  ```

---

## 4. 시뮬레이터 판정: Isaac Sim → MuJoCo

판정 규칙(역할 가이드 8단계): "Isaac Sim 먼저 검토, (a) 설치·구동 (b) URDF 구동 (c) 자기 충돌 보고가 되면 Isaac Sim, 안 되면 MuJoCo"

| 경로 | 결과 |
|---|---|
| Isaac Sim 5.x (pip) | GLIBC 2.34+ 필요 → 서버 2.31로 불가 |
| Isaac Sim 4.2 / 4.5 | 리눅스 최소 드라이버 535.129.03 → 서버 525로 불가 |
| Singularity 컨테이너 | OS(GLIBC)는 우회되지만 드라이버는 호스트 것 사용 → 4.x 불가 |
| Isaac Sim 2023.1.1 (20.04·드라이버 525 지원) | NGC 컨테이너 태그 삭제(manifest unknown), 공식 바이너리 다운로드 페이지 삭제 |
| 서버 내 기존 설치본 | 없음 |

- OS·드라이버 변경은 관리자 권한 필요 → **(a)에서 불가 → MuJoCo 채택**
- 연구 영향 없음: 기구학 재생 방식이라 물리엔진 차이 무관, 지표 전부 MuJoCo로 계산 가능, mink가 원래 MuJoCo 기반이라 스택 통일
- 재생 함수 `replay(q_traj, gripper)` 형태를 고정해 두면, 추후 드라이버가 535 이상이 될 경우 Isaac Sim 4.5 컨테이너로 내부만 교체 가능
- 상세 기록: `robot/sim_decision.md` (커밋 예정)

---

## 5. 서버 Python 환경

### 사용법 (매번)
```bash
source ~/use_robot.sh
```
`~/use_robot.sh` 내용:
```bash
conda activate robot
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONNOUSERSITE=1
echo "robot 환경 활성화됨: $(which python)"
```

### 왜 `conda activate robot`만 하면 안 되나 (공유 계정 함정)
1. **홈 디렉토리 자체가 이전 사용자의 venv**로 만들어져 있음 (`~/pyvenv.cfg`, `~/bin`, `~/lib`)
   → PATH에서 `~/bin/python`(3.10)이 conda 환경보다 먼저 잡힘 → `export PATH=...`로 우회
2. **`~/.local`에 이전 사용자 패키지**가 있어 conda 환경에 섞여 들어옴 → `PYTHONNOUSERSITE=1`로 차단
3. 홈에 있는 다른 conda 환경들(bert, lstm, tf 등)·ollama 등은 이전 사용자 것. **삭제하지 않음**

### `robot` 환경 (Python 3.10)
| 패키지 | 버전 |
|---|---|
| mujoco | 3.14.0 |
| pin (pinocchio) | 4.1.0 |
| numpy | 2.2.6 |
| absl-py | 2.5.0 |
| mink | 1.3.0 |
| qpsolvers | 4.13.0 |
| daqp | 0.10.3 |
| scipy | 1.15.3 |
| eaik | 1.2.2 |
| matplotlib | 3.10.9 |

- mink·qpsolvers·daqp·scipy는 4번 역할 가이드 준비 단계 패키지(알고리즘 B), 2026-10-16 추가
- eaik(알고리즘 C)·matplotlib, 2026-11-05 추가 (제효정). 함께 들어온 패키지: urchin, trimesh, lxml, pycollada, pyribbit, pyglet, networkx, pillow 등. 기존 패키지 버전은 바뀌지 않음. 설치 전 목록 백업: 서버 `~/robot_env_freeze_before_eaik.txt`

- 확인: `python -c "import mujoco, pinocchio; print(mujoco.__version__, pinocchio.__version__)"`
- 이후 추가 예정: OpenCV, pandas 등

### 패키지 설치 (필요할 때 담당자가 직접, 2026-11-05)
공유 환경이라 한 사람의 설치가 모두의 실행에 영향을 줍니다. 아래 순서를 지키고, **동시에 두 사람이 설치하지 않도록 설치 전 팀 채팅에 알립니다.**
```bash
source ~/use_robot.sh                                  # 프롬프트가 (robot) 인지 확인
which python; python -m pip --version                  # 둘 다 .../envs/robot/... 이어야 함
python -m pip freeze > ~/robot_env_freeze_before_<패키지>.txt          # 1) 백업
python -m pip install --dry-run <패키지>==<버전>                         # 2) 미리보기: numpy·mujoco·pinocchio·scipy 버전이 바뀌면 멈추고 팀과 상의
python -m pip install <패키지>==<버전>                                  # 3) 설치
diff ~/robot_env_freeze_before_<패키지>.txt <(python -m pip freeze) | grep "^<"   # 4) 기존 패키지가 바뀌었으면 출력됨 (비어 있어야 정상)
cd ~/project && python replay/replay.py --check --out ~/out/본인이름/replay_check  # 5) 재생 점검이 그대로 되는지
```
- **`pip` 대신 `python -m pip`**: `(base)` 상태나 PATH 순서 때문에 그냥 `pip`은 홈 폴더 옛 가상환경(`~/bin/pip`)에 설치될 수 있음 (2026-11-05 실제로 발생, 되돌림)
- 설치 후 위 패키지 표에 버전을 적고 팀 채팅에 알립니다. 버전은 `==`로 고정합니다

### OpenVLA 학습 (7번, `openvla` 환경, 2026-11-06)
`robot` 환경과 **별도**의 conda 환경. 공유 `robot` 환경은 건드리지 않음.
```bash
source ~/use_openvla.sh      # 매번. ~/.bashrc 의 LD_LIBRARY_PATH(CUDA 12 cuDNN)를 비워 PyTorch(cu118)와 충돌을 막음
```
- 설치 스크립트 `~/vla/setup_openvla.sh`, OpenVLA 코드 `~/vla/openvla` (commit `c8f03f4`), 모델 `~/vla/models/openvla-7b`
- 버전 (역할 가이드 7번의 검증된 버전과 동일): torch 2.2.0+cu118, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, peft 0.11.1, flash-attn 2.5.5, tensorflow 2.15.0, tensorflow-datasets 4.9.3. 전체 목록 `~/out/guhyeon/openvla_env_freeze.txt`
  - 설치 중 맞춘 것: `tensorflow-metadata` 1.14.0 (최신 1.21은 TF 2.15와 충돌), 그에 맞춰 protobuf 3.20.3, `wandb` 0.16.6
- Slurm 제출은 **bash 스크립트로** (`sbatch ... script.sh`). `--wrap "source ..."`은 sh로 실행돼 환경이 안 켜짐
- 학습 기록은 wandb offline (`WANDB_MODE=offline`, `~/vla/openvla/wandb/`)

**결정 (2026-11-06)**
| 항목 | 결정 | 이유 |
|---|---|---|
| 학습 데이터 | 알고리즘 **C(EAIK)** 결과 1개로 1회 | 리타게팅 시험에서 C가 가장 좋아 보임. 근거는 지표 결과가 나오면 확인해 기록 |
| 데이터 형식 | **RLDS** (`knu_retarget`) | OpenVLA 공식 학습 코드(`vla-scripts/finetune.py`)가 `RLDSDataset`만 읽음. 10/16 멘토가 말한 LeRobot은 쓰지 않고, 대신 observation·action·state 키가 빠지지 않는지 변환 때 점검 |
| 행동 (action) | **다음 스텝의 7관절 각도** `q[t+1]` (rad, 7개) | C 결과(`q`)를 변환 없이 정답으로 씀. 7관절이라 그리퍼 자리 없이 7개가 맞음. 예측값을 `replay()`로 바로 재생·평가 |
| OpenVLA 코드 수정 | 행동 형식 `JOINT_POS_ABS`(관절각 7개, 그리퍼 없음, 7개 모두 정규화) 추가 + 데이터셋 `knu_retarget` 등록. `prismatic/vla/datasets/rlds/oxe/`의 `configs.py`·`transforms.py`·`materialize.py` | OpenVLA 로더는 손끝 형식(`EEF_POS`·`EEF_R6`)만 받음. 손끝 형식으로 등록하면 마지막 값을 그리퍼로 보고 정규화를 건너뛰어 7번 관절(±3.05 rad)이 −1~1 밖에서 잘림. 데이터셋 등록(`configs.py`·`transforms.py`)은 원래 필요한 작업(역할 가이드 7번) |
| 상태 (state) | 현재 7관절 각도 `q[t]` | RLDS 형식상 기록 (OpenVLA 입력에는 안 씀) |
| 이미지 | **exo_front** 1장 | C 결과의 시간축이 exo_front라 프레임을 그대로 짝지음. 장면 전체가 보이고 손 놓친 비율이 낮음 (ego는 49~65%) |
| 프레임 간격 | **10 Hz** (T1·T3 60 fps는 6장마다, T2 30 fps는 3장마다 1장) | OpenVLA 사전학습 데이터 대부분이 3~20 Hz. 앞뒤 장면이 거의 같은 30~60 Hz는 다음 행동을 배울 거리가 적음 |
| 지시문 | T1 `pick up the cup` / T2 `stack the block` / T3 `move the block` (변환 때 확정) | 태스크마다 한 문장 |
| 학습 설정 | LoRA rank 32, 배치 8 × 누적 2 (유효 16), lr 5e-4, `image_aug True`, max_steps 20,000, save_steps 2,000, `--save_latest_checkpoint_only False` | 가이드 4단계 값. 배치는 시험 학습 최대 메모리 39.8 GB로 확인 |
| 저장 | 체크포인트 `~/out/guhyeon/runs/` (개당 약 15 GB × 10개 ≈ 150 GB), 레포 `runs/`에는 설정·로그만 | 산출물은 `~/project` 밖 |

- **시험 학습 (가이드 1단계, LIBERO `libero_spatial_no_noops` 200스텝):** 손실 12.0 → 약 3.0, 행동 정확도 13% → 약 40%, 최대 메모리 39.8 GB, 스텝당 약 2.9초 (20,000스텝 ≈ 16시간 + 저장 시간). 스크립트 `~/vla/smoke_libero.sh`
- `image_aug True`로 학습하므로 **추론 때 영상 가운데를 잘라 넣어야 함** (8번 평가에 전달)

---

## 6. Git 작업 방식

### 원칙
```
각자 노트북: 브랜치 → commit → push → PR → main   (본인 이름으로 기록)
서버 ~/project: main 고정, git pull → 실행만        (checkout·수정·커밋 금지)
서버 산출물:    worktree로 결과 브랜치 → git c 본인이름 커밋 → push → PR
```
- 서버 `~/project`에서 파일을 고쳐두거나 커밋하면 다음 사람의 `git pull`이 막힘
- 실행 산출물은 `~/project` 밖 `~/out/본인이름/`에 저장 (`~/project` 안에 생기면 그 파일이 main에 합쳐진 뒤 pull 충돌)
- 서버에서 생긴 산출물은 **서버에서 바로 커밋 가능**. 단, `~/project`가 아니라 본인 작업 폴더(worktree)에서

### 서버에서 커밋하는 법
예시: 신서연이 서버에서 실행해 `~/out/신서연/summary.md`로 저장한 결과를 `metrics/summary.md`로 올리는 경우
```bash
# 1) 최신 main 기준으로 본인 작업 폴더 + 결과 브랜치 만들기
cd ~/project && git fetch
git worktree add ~/work/신서연 -b results/metrics-summary origin/main

# 2) 올릴 산출물을 본인 폴더로 복사
cp ~/out/신서연/summary.md ~/work/신서연/metrics/

# 3) 본인 폴더에서 커밋 — 그냥 git commit 말고 git c 본인이름
cd ~/work/신서연
git add metrics/summary.md
git c 신서연 -m "지표 결과 요약 추가"

# 4) GitHub에 올리기 → GitHub 웹에서 PR 만들어 main에 merge
git push origin HEAD

# 5) 본인 폴더 + 서버에 남은 결과 브랜치 정리 (GitHub에는 push한 브랜치가 그대로 남음)
cd ~ && git -C ~/project worktree remove ~/work/신서연
git -C ~/project branch -D results/metrics-summary
```
- 사람마다 바꿀 곳: 산출물 `~/out/본인이름`, 작업 폴더 `~/work/본인이름`, 브랜치 `results/작업내용`(1·5단계 둘 다), 커밋 `git c 본인이름`
- 5단계 브랜치 삭제를 빼먹으면 다음에 같은 브랜치 이름으로 1단계를 할 때 `already exists` 에러
- **`git c 본인이름`** = 본인 이름·이메일로 커밋하는 명령 (서버 `~/.gitconfig`에 등록됨)
  - 사용 가능한 이름: `정구현` / `차서현` / `신서연` / `제효정`
  - 그냥 `git commit`을 쓰면 공유 계정이라 `coss37-server`로 기록되고 누구 잔디에도 안 찍힘
  - 이름을 틀리면 커밋하지 않고 사용법만 출력됨
  - 등록된 이메일이 본인 GitHub 계정(Settings → Emails)에 있어야 잔디에 찍힘
- 영상·`npz`·체크포인트 같은 대용량은 커밋하지 않음 (`.gitignore`로 제외됨)
- 노트북에서 커밋하고 싶으면 `scp`로 가져와도 됨
  ```bash
  scp -P 10000 coss37@155.230.81.91:~/out/신서연/summary.md ./metrics/
  ```

### 서버 ↔ GitHub 연결 (완료, 2026-10-05 `ssh -T github-cdp1` 인증 확인)
- 레포 전용 Deploy key: `~/.ssh/cdp1_deploy` (GitHub 레포 Settings → Deploy keys에 `coss37-server`, Read/write)
- 조직(2026-2-CDP1) Settings에서 Deploy keys 허용(Enabled) 해둠
- `~/.ssh/config`:
  ```
  Host github-cdp1
      HostName github.com
      Port 22
      User git
      IdentityFile ~/.ssh/cdp1_deploy
      IdentitiesOnly yes
      GSSAPIAuthentication no
  ```
- ⚠️ 서버 공통 설정 `/etc/ssh/ssh_config`에 `Host * / Port 10000`, `GSSAPIAuthentication yes`가 있음
  → `Port 22`를 명시하지 않으면 GitHub 접속이 10000번으로 가서 **응답 없이 멈춤**
- 서버에서 레포 주소는 `github-cdp1:2026-2-CDP1/project.git` (별칭 사용)
- 서버 `~/project`에 `git config pull.ff only` 설정 (누가 `~/project`에서 커밋해 GitHub과 갈라졌을 때 몰래 merge하지 않고 에러로 멈춤)

### .gitignore (대용량은 깃에 올리지 않음)
- 제외: 영상(`*.mp4`, `*.mov` 등), 키포인트·궤적(`*.npy`, `*.npz`), `*.tfrecord*`, 컨테이너(`*.sif`), `runs/` 안의 체크포인트·가중치(`*.pt`, `*.bin`, `*.safetensors`), `rlds/data/`, `sync/calib/`, `replay/videos/`
- `runs/`의 학습 설정·로그 같은 작은 파일은 커밋 대상
- 대용량 원본 데이터는 서버 `~/data/raw/`에 둠 (구조: `episodes/RAW_DATA.md`)

---

## 7. 공통 폴더 구조 (역할 가이드)

```
project/
├── robot/       # 1번: 팔 모델(robot.urdf, robot.xml), joint_limits.csv, fk_check.md, sim_decision.md
├── sync/        # 2번: README.md(먼저 읽기), sync.csv, exo_calib_T{n}.json, protocol.md, 싱크·보정 코드
├── episodes/    # 3번: <에피소드ID>/ 영상·키포인트·라벨·meta.json
├── retarget/    # 4번: common.py, config.json, <알고리즘>/<에피소드ID>.npz
├── replay/      # 5-1번: replay.py, 나란히 보기 영상
├── metrics/     # 5-2번: definitions.md, metrics.py, summary.md
├── rlds/        # 6번: RLDS 변환 코드
├── runs/        # 7번: 체크포인트, 로그
└── eval/        # 8번: 평가 결과
```
- 에피소드 ID: `T{태스크번호}_E{세 자리}` (예: `T2_E017`)
- 알고리즘 폴더명: `A_DLS`, `B_mink`, `C_eaik`

## 8. 공통 규칙 (역할 가이드)
- 단위: 길이 m, 각도 rad, 시간 s
- 쿼터니언 저장 순서: **(w, x, y, z)** — scipy `as_quat()` 기본값은 (x, y, z, w)이므로 변환 필요
- 좌표계: 영상 좌표는 테이블 좌표계(2번 정의), 로봇 궤적은 로봇 베이스 좌표계
- **영상 방향·카메라 보정: `sync/protocol.md` 8절 규칙을 따른다** (영상·키포인트·좌표를 다루는 역할 모두)
  - 영상은 **저장된 그대로** 읽는다 (`cv2.CAP_PROP_ORIENTATION_AUTO = 0`). 폰의 회전 꼬리표(0/90/180/270)는 믿을 수 없어서 무시
  - 거꾸로 찍힌 영상은 `sync/upright_<날짜>.csv`의 `upright_rot`만큼 **메모리에서만** 세워서 인식·학습 이미지에 쓰고, 좌표는 저장 화면 기준으로 되돌려 저장 (원본 파일은 고치지 않음)
  - 보정은 **태스크 단위**: 태스크마다 체커보드 `T{n}_check`로 만든 `sync/exo_calib_T{n}.json`을 쓴다. 에피소드 ID의 태스크 번호로 고른다 (`T2_E017` → `exo_calib_T2.json`)
- **카메라 간 시간 맞추기: `sync/sync.csv`** — 기준 카메라 `exo_front` 대비 `left`·`right`·`ego` 시간 차. `episodes/tools/io_utils.read_sync_row`로 읽는다. 쓰는 법·열 뜻은 `sync/README.md`
- 설정·지표 정의는 **결과 보기 전에 고정**. 바꾸면 이유 기록 후 모든 알고리즘에 재적용
- 풀이 실패·키포인트 누락은 **지우지 않고 기록** (그 자체가 비교 지표)
- 결과 넘기기 전에 받는 사람에게 파일 한두 개로 먼저 확인

---

## 9. 현재 진행 상황 & 다음 할 일

### 완료
- [x] GPU 서버 접속·구조 파악, Slurm 사용법 확인
- [x] Isaac Sim 검토 → MuJoCo 확정
- [x] 서버 `robot` 환경(MuJoCo 3.14.0·pinocchio 4.1.0) + `use_robot.sh`
- [x] 서버 ↔ GitHub Deploy key 연결, `~/project` 클론
- [x] 로컬 ↔ GitHub 연결 확인 (정구현)
- [x] 작업 흐름 확정: 노트북 브랜치 → PR → main / 서버는 main pull·실행 전용
- [x] 서버 `~/project` 설정: `pull.ff only`, 팀원별 커밋 명령 `git c 본인이름` 등록, main 동기화 확인
- [x] 원본 데이터 서버 정리 (2026-11-04): `~/data/raw/T{1,2,3}/{exo_front,exo_left,exo_right,ego}/T{n}_E001~E075` + 체커보드 `T{n}_check` (원본 zip 내용과 대조 검증 후 zip 삭제, 원본 zip은 공유 드라이브에 있음) → `episodes/RAW_DATA.md`
- [x] 2번 싱크·보정 (신서연, 2026-11-04): `sync/sync.csv` T1~T3 675행 모두 `usable=1` (검증 오차 최대 36 ms), `sync/exo_calib_T1~T3.json` (재투영 오차 평균 0.82~1.06 px) → `sync/README.md`

- [x] 1번 로봇 셋업: iiwa14 URDF·MJCF 확보, 관절 한계 일치, FK 일치(무작위 200자세, 오차 1e-15), TCP = 플랜지 → `robot/model_checks.md`
- [x] 5-1 재생 함수 `replay(q_traj)` v1 형태 고정 → `replay/README.md`
- [x] 비교 알고리즘 확정 (2026-11-04): A DLS / B mink / C EAIK, D 제외 (1절)
- [x] 7번 OpenVLA 환경 (정구현, 2026-11-06): `openvla` 환경 설치, 모델 다운로드, LIBERO 200스텝 시험 학습 통과, 학습 데이터 형식·설정 결정 (5절 「OpenVLA 학습」)
- [x] 6번 RLDS 변환 파이프라인 (정구현, 2026-11-06): OpenVLA 패치(`rlds/openvla/`), 변환(`rlds/make_knu_retarget.py`), RLDS 빌더·점검(`rlds/knu_retarget/`, `rlds/check_rlds.py`). C 리허설 결과 84개 → 5,595스텝 → OpenVLA 50스텝 학습 통과, 7관절 모두 정규화 확인 → `rlds/README.md`, `rlds/train/README.md`

### 다음
1. **3번 에피소드:** 에피소드마다 시작·끝 프레임 라벨(`episodes/labeling_rules.md`) → `episodes/tools/process_episode.py`로 키포인트·3D 궤적 생성 (보정은 태스크별 `--calib sync/exo_calib_T{n}.json`). 아직 실제 데이터에 돌리지 않음
2. **4번 리타게팅:** A·B·C 첫 구현 완료(2026-11-05, 시험 에피소드 T1_E004) → `retarget/README.md`. 남은 것: 멘토 확인(팔 길이 비율·로봇 자리, 팔꿈치 입력, C의 3번 관절 탐색), 3번 전체 에피소드가 나오면 일괄 실행 → `retarget/<알고리즘>/<에피소드ID>.npz`
   - C: EAIK 왕복 검사 1000/1000 통과 (`retarget/eaik_check.py`). `robot` 환경에 eaik 1.2.2 설치 (5절)
3. **5-2 지표:** `metrics/definitions.md` 멘토 확인 후 고정 → `metrics.py`, `summary.md`. 저크 Δt가 1/30으로 적혀 있으나 exo_front는 약 60 fps → 30 Hz로 맞출지 Δt를 고칠지 결정
4. **6 RLDS 변환 · 7 OpenVLA LoRA 학습 (정구현):** 파이프라인 점검 완료(리허설 84개). 남은 것: 최종 C 결과(225개)가 나오면 `rlds/run_convert.sh` → `rlds/build.py` → `rlds/check_rlds.py` → `rlds/train/train_knu.sh`(20,000스텝, 약 하루, 체크포인트 약 150 GB). 학습 후 추론 점검(가이드 7번 6단계)
5. **8 평가:** 시작 전

### 보류·확인 필요
- 멘토 쪽 소통 담당자 지정 후 미팅 일정
- C(EAIK)의 3번 관절 처리: 상수 고정이 아니라 프레임마다 격자 탐색 — 10/16 "3번 고정하지 말 것"에 맞는지 (4번, 멘토 확인)
- 팔꿈치 입력 팀 결정(T10)을 멘토에 알리기, 팔 길이 비율·로봇 자리 (4번, 멘토 확인)
- **C 결과의 관절 한계 근처 비율** (2026-11-06, RLDS 점검에서 발견): C 리허설 결과 5,595스텝 중 joint2 38.1%, joint6 6.1%, joint4 0.5%가 관절 한계(±2.094 rad) 1e-3 rad 이내. 팔 길이 비율·로봇 자리 설정 영향인지 확인하고, 최종 결과에서 다시 잼 (4번, `rlds/check_rlds.py`)
- 팀원별 `git c` 등록 이메일이 각자 GitHub 계정(Settings → Emails)에 등록돼 있는지 확인
- (선택) 서버 관리자에게 GPU 노드 드라이버 535+ 업그레이드 가능 여부 문의 → 가능하면 Isaac Sim 재검토
- (선택) 팀원별 SSH 공개키를 서버 `~/.ssh/authorized_keys`에 등록 → 비밀번호 없이 접속·VS Code Remote-SSH 편의
