"""knu_retarget RLDS 데이터셋 (tensorflow_datasets 빌더).

make_knu_retarget.py 가 만든 에피소드 npz 폴더(환경 변수 KNU_RETARGET_SRC)를 RLDS 로 묶는다.
키 이름은 OpenVLA 등록(openvla_knu_retarget.patch)과 맞춘다:
  steps/observation/image (256,256,3) uint8, steps/observation/state (7,) float32,
  steps/action (7,) float32, steps/language_instruction, is_first/is_last/is_terminal, reward, discount

  KNU_RETARGET_SRC=~/vla/data/knu_retarget_src python build.py --data-dir ~/vla/data
"""
import os
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


class KnuRetarget(tfds.core.GeneratorBasedBuilder):
    """사람 시연 영상 -> 알고리즘 C 리타게팅 관절 궤적 (KUKA iiwa14, 그리퍼 없음)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "exo_front 256x256, 10 Hz, action = 다음 스텝 7관절 각도 (rad)"}

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                                                     doc="exo_front RGB"),
                        "state": tfds.features.Tensor(shape=(7,), dtype=np.float32, doc="현재 7관절 각도 rad"),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32, doc="다음 스텝 7관절 각도 rad"),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_id": tfds.features.Text(),
                    "hand": tfds.features.Text(),
                    "file_path": tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager):
        src = Path(os.environ["KNU_RETARGET_SRC"]).expanduser()
        return {"train": self._generate_examples(sorted(src.glob("T*_E*.npz")))}

    def _generate_examples(self, files):
        for f in files:
            z = np.load(f, allow_pickle=True)
            n = len(z["action"])
            instr = str(z["instruction"])
            steps = [{
                "observation": {"image": z["image"][k], "state": z["state"][k].astype(np.float32)},
                "action": z["action"][k].astype(np.float32),
                "discount": 1.0,
                "reward": float(k == n - 1),
                "is_first": k == 0,
                "is_last": k == n - 1,
                "is_terminal": k == n - 1,
                "language_instruction": instr,
            } for k in range(n)]
            yield f.stem, {"steps": steps, "episode_metadata": {
                "episode_id": str(z["episode_id"]), "hand": str(z["hand"]), "file_path": str(f)}}
