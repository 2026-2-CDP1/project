# knu_retarget — RLDS 빌더

`make_knu_retarget.py`(변환)가 만든 에피소드 npz를 OpenVLA가 읽는 RLDS로 묶는다. 데이터셋 이름 `knu_retarget`, 버전 1.0.0.

## 만들기
```bash
source ~/use_openvla.sh
KNU_RETARGET_SRC=~/vla/data/knu_retarget_src python rlds/build.py --data-dir ~/vla/data
python rlds/check_rlds.py --data-dir ~/vla/data       # 마지막 줄 CHECK_OK
```
결과: `~/vla/data/knu_retarget/1.0.0/` (OpenVLA 학습의 `--data_root_dir ~/vla/data --dataset_name knu_retarget`)

## 형식
| 키 | 형태 | 내용 |
|---|---|---|
| `steps/observation/image` | (256, 256, 3) uint8, jpeg | exo_front |
| `steps/observation/state` | (7,) float32 | 현재 7관절 각도 rad |
| `steps/action` | (7,) float32 | 다음 스텝 7관절 각도 rad |
| `steps/language_instruction` | 문자열 | 태스크별 지시문 |
| `steps/is_first`, `is_last`, `is_terminal` | bool | 첫·마지막 스텝 |
| `steps/reward`, `discount` | float32 | 마지막 스텝 reward 1, discount 1 |
| `episode_metadata/episode_id`, `hand`, `file_path` | 문자열 | 원본 추적용 |

키 이름은 OpenVLA 패치(`rlds/openvla/knu_retarget.patch`)의 `image_obs_keys = {"primary": "image"}`, `state_obs_keys = ["joint_state"]`(표준화 함수가 `state`에서 만듦)와 맞춘다.

## 2026-11-06 확인 (C 리허설 결과 84개)
- 에피소드 84, 스텝 5,595 (에피소드당 37~85, 중앙값 67), RLDS 121 MB
- 이미지 84개 모두 256×256, 행동 = 다음 스텝 상태 불일치 0
- 지시문: `pick up the cup` 4,145 / `move the block` 1,269 / `stack the block` 181 스텝 (리허설은 T2가 3개뿐)
- **관절 한계 1e-3 rad 이내:** joint2 38.1%, joint6 6.1%, joint4 0.5% → 리타게팅(4번)에 공유, 최종 결과에서 다시 확인
