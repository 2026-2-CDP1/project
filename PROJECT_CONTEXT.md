# PROJECT_CONTEXT — 리타게팅 알고리즘 비교 연구

> 이 문서는 팀원과 코드 어시스턴트(VS Code 등)가 프로젝트 맥락을 한 번에 파악하기 위한 요약입니다.
> 연구 목적·평가 방법의 기준은 멘토 제공 「학생 안내서」와 「역할 가이드」입니다. 이 문서와 어긋나면 그쪽을 따릅니다.
> 최종 갱신: 2026-10-05

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
- 브랜치 이름: `역할폴더/작업내용` (예: `robot/iiwa7-model`, `retarget/dls`, `metrics/definitions`)
- `main`에 직접 push하지 않고 PR로 합칩니다
- 서버에서 돌려보고 싶은 코드는 먼저 PR로 main에 합친 뒤 pull 합니다

### 주의사항
- **서버 `~/project`에서 `git checkout`·파일 수정·`git commit` 금지** — 4명이 같은 계정·같은 폴더를 써서, 바꾸면 다른 사람의 실행·pull이 깨집니다
- 서버에서 커밋은 `~/project`가 아니라 `~/work/본인이름`에서만, `git commit` 대신 `git c 본인이름` (예: `git c 신서연 -m "메시지"`) — 전체 순서는 6절 「서버에서 커밋하는 법」
- **산출물은 `~/project` 안에 저장하지 않기** → `~/out/본인이름/`에 저장 (코드는 출력 경로를 인자로 받게 작성, 예: `--out`)
  - `~/project` 안에 생긴 파일이 나중에 main에 같은 경로로 합쳐지면 다음 사람의 `git pull`이 막히고, 두 사람이 같은 경로에 덮어쓰게 됨
- `git pull`이 에러로 멈추면 임의로 지우거나 되돌리지 말고 팀 채팅에 알려주세요 (누가 서버 `~/project`에서 커밋했거나 파일을 고쳐 둔 경우 멈출 수 있음. 커밋으로 갈라진 경우는 `pull.ff only` 설정으로 merge 대신 에러가 나게 해둠)
- 그냥 `conda activate robot` 금지 → 반드시 `source ~/use_robot.sh` (이유는 5절)
- `robot` 환경에 `pip install` 금지 → 필요한 패키지는 정구현에게 요청
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
- 주제: 사람의 ego/exo 영상 → 손목 궤적 추출 → **리타게팅 알고리즘 4종을 같은 데이터로 비교** → 대표 알고리즘 데이터로 OpenVLA LoRA 1회 학습(파이프라인 실증)

### 전체 흐름
```
수집(ego·exo 촬영) → 가공(키포인트·행동 라벨) → 싱크 확인(지난 학기 검출 알고리즘 재사용)
→ 리타게팅(A·B·C·D) ─┬→ [연구 본체] MuJoCo 재생·지표 측정 → 알고리즘 × 케이스 비교표
                      └→ [보조] 대표 알고리즘 1개 → OpenVLA LoRA 학습 → MSE·CI-MSE 점검
```

### 비교 알고리즘
| 축 | 알고리즘 | 방식 | 구현 |
|---|---|---|---|
| A | DLS IK | 반복, 관절 한계는 계산 후 자르기 | pinocchio 자체 구현 |
| B | 제약 QP IK | 반복, 관절 한계를 QP 제약으로 | mink (MuJoCo 모델 사용) |
| C | 닫힌 해 | 반복 없이 기하 공식으로 해 전부 계산 | IK-Geo (7자유도는 SEW 각, stereo-sew) |
| D (조건부) | 병렬 시드 최적화 | 여러 시작점 동시 최적화 | PyRoki |

- 네 알고리즘 모두 **학습하지 않음**. 매 프레임 독립적으로 IK를 푸는 계산 절차.
- 학습되는 것은 OpenVLA 하나뿐이며, 알고리즘 순위 비교에 쓰지 않음.

### 평가 지표 (5-2, 학습 없이 재생 결과로 계산)
관절 한계 위반율 / 관절 한계 포화율(보조) / 자기 충돌률 / 최대 침투 깊이 / IK 실패율 / 저크 RMS / 파지점 거리 / 추종 오차(참고용, 순위 미사용)

### 주장 가능 범위
"케이스 X에서 알고리즘 A가 **지표상** B보다 낫다"까지. 실제 로봇 작업 성공은 주장 불가.

---

## 2. 확정된 결정사항

| 항목 | 결정 |
|---|---|
| 타깃 로봇 | **KUKA LBR iiwa7 (R800)**, 7자유도 single-arm + 그리퍼 |
| 시뮬레이터 | **MuJoCo** (Isaac Sim 불가 판정, 4절 참고) |
| 촬영 장비 | iPhone (멘토 추천 앱: NeRFCapture / Spectacular Rec / AR Recorder — 연속 녹화가 기본인 Spectacular Rec 우선 테스트) |
| 학습 모델 | OpenVLA-7B + LoRA, 대표 알고리즘 1개 데이터로 1회 |
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

- 확인: `python -c "import mujoco, pinocchio; print(mujoco.__version__, pinocchio.__version__)"`
- **이 환경에 팀원이 임의로 pip install 하지 않기** (공유 환경, 동시 설치 시 꼬임) → 필요 패키지는 정구현에게 요청
- 이후 추가 예정: mink, IK-Geo 파이썬 래퍼(멘토 확인 필요), OpenCV, pandas, matplotlib 등

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
- 대용량 데이터의 공용 저장 위치는 촬영 단계 전에 결정 예정 (서버 `/abr/coss37/` 아래 레포와 같은 폴더 구조 후보)

---

## 7. 공통 폴더 구조 (역할 가이드)

```
project/
├── robot/       # 1번: 팔 모델(robot.urdf, robot.xml), joint_limits.csv, gripper.csv, fk_check.md, sim_decision.md
├── sync/        # 2번: protocol.md, exo_calib.json, sync.csv
├── episodes/    # 3번: <에피소드ID>/ 영상·키포인트·라벨·meta.json
├── retarget/    # 4번: common.py, config.json, <알고리즘>/<에피소드ID>.npz
├── replay/      # 5-1번: replay.py, 나란히 보기 영상
├── metrics/     # 5-2번: definitions.md, metrics.py, summary.md
├── rlds/        # 6번: RLDS 변환 코드
├── runs/        # 7번: 체크포인트, 로그
└── eval/        # 8번: 평가 결과
```
- 에피소드 ID: `T{태스크번호}_E{세 자리}` (예: `T2_E017`)
- 알고리즘 폴더명: `A_DLS`, `B_mink`, `C_ikgeo`, `D_pyroki`

## 8. 공통 규칙 (역할 가이드)
- 단위: 길이 m, 각도 rad, 시간 s
- 쿼터니언 저장 순서: **(w, x, y, z)** — scipy `as_quat()` 기본값은 (x, y, z, w)이므로 변환 필요
- 좌표계: 영상 좌표는 테이블 좌표계(2번 정의), 로봇 궤적은 로봇 베이스 좌표계
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

### 다음 (1번 역할: 로봇·시뮬레이터 셋업)
1. KUKA LBR iiwa7 R800의 URDF·MJCF 확보
   - ⚠️ mink 예제·MuJoCo Menagerie에 흔한 건 **iiwa14**. iiwa7과 링크 길이가 다르니 정확한 모델인지 확인
   - IK-Geo 분류표에 iiwa 7 R800(q3 고정)이 닫힌 해 계열로 등록됨 → 알고리즘 C 적용 가능성 확인 (4번 담당과 함께)
2. 모델 확인 — 서버엔 모니터가 없으므로 `mujoco.viewer` 대신 **GPU 노드에서 오프스크린 렌더링(`MUJOCO_GL=egl`)으로 PNG 저장** 후 VS Code Remote-SSH 또는 scp로 확인
3. TCP 정의 (그리퍼 두 손가락 끝 중간점, MJCF는 site / URDF는 고정 관절 링크)
4. `joint_limits.csv` — 두 파일 관절 한계 비교·일치
5. FK 일치 검사 — 무작위 자세 200개, TCP 위치·방향 차이 ≤ 1e-6 → `fk_check.md`
6. `gripper.csv` — 그리퍼 열림/닫힘 관절값과 손가락 폭(m)
7. 4번·5-1번 담당이 `robot/` 파일을 로드해 보는지 확인

### 보류·확인 필요
- 멘토 쪽 소통 담당자 지정 후 미팅 일정
- IK-Geo 파이썬 패키지 종류(ik-geo / EAIK) 멘토 확인
- 대용량 데이터 공용 저장 위치
- 팀원별 `git c` 등록 이메일이 각자 GitHub 계정(Settings → Emails)에 등록돼 있는지 확인
- (선택) 서버 관리자에게 GPU 노드 드라이버 535+ 업그레이드 가능 여부 문의 → 가능하면 Isaac Sim 재검토
- (선택) 팀원별 SSH 공개키를 서버 `~/.ssh/authorized_keys`에 등록 → 비밀번호 없이 접속·VS Code Remote-SSH 편의
