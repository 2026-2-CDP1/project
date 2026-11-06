# OpenVLA 패치 — 관절각 행동 형식과 `knu_retarget` 등록

우리 행동(다음 스텝의 7관절 각도, 그리퍼 없음)을 OpenVLA가 학습할 수 있게 OpenVLA 코드 세 파일을 고친다.
결정 이유는 `PROJECT_CONTEXT.md` 5절 「OpenVLA 학습」.

## 왜 고치나
- OpenVLA 데이터 로더(`prismatic/vla/datasets/rlds/oxe/materialize.py`)는 행동 형식으로 손끝 기준 두 가지(`EEF_POS`, `EEF_R6`)만 받는다. 관절 형식(`JOINT_POS`)은 이름만 있고 넣으면 오류를 낸다.
- 손끝 형식(`EEF_POS`)으로 등록해 관절각을 넣으면, 마지막 값을 그리퍼(0~1)로 보고 정규화를 건너뛴다. 7번 관절(±3.05 rad)이 −1~1 밖에서 잘려 정보가 망가진다.
- 그래서 "관절각 7개, 그리퍼 없음, 7개 모두 정규화" 형식을 하나 추가한다. 데이터셋 등록(`configs.py`, `transforms.py`)은 원래도 해야 하는 일이다(역할 가이드 7번).

## 바뀌는 것 (`knu_retarget.patch`, OpenVLA commit `c8f03f4` 기준)
| 파일 | 변경 |
|---|---|
| `configs.py` | `ActionEncoding.JOINT_POS_ABS = 5` 추가, `OXE_DATASET_CONFIGS["knu_retarget"]` (이미지 `image`, 상태 `joint_state`) |
| `transforms.py` | `knu_retarget_dataset_transform`: 상태 7개를 `joint_state`로. 행동은 그대로 |
| `materialize.py` | `JOINT_POS_ABS` 허용. `absolute_action_mask = [True]*7`, `action_normalization_mask = [True]*7` |

정규화 마스크는 학습 때 `dataset_statistics.json`에 함께 저장된다. 추론 코드(`predict_action`)가 이 마스크를 보고 7개를 모두 원래 각도로 되돌리므로 추론 쪽은 고칠 필요가 없다.

## 적용
```bash
bash rlds/openvla/apply_patch.sh          # 기본 위치 ~/vla/openvla, 이미 적용돼 있으면 건너뜀
```
마지막 줄에 `OK knu_retarget: ... 'action_normalization_mask': [True, True, True, True, True, True, True]`가 나오면 된다.

되돌리기: `cd ~/vla/openvla && git apply --reverse ~/project/rlds/openvla/knu_retarget.patch`
