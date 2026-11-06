#!/bin/bash
# OpenVLA 저장소에 knu_retarget 패치를 적용하고, 등록이 제대로 됐는지 확인한다.
#   bash rlds/openvla/apply_patch.sh            (기본 OpenVLA 위치: ~/vla/openvla)
#   OPENVLA=/다른/경로 bash rlds/openvla/apply_patch.sh
# 이미 적용돼 있으면 건너뛴다. 패치는 OpenVLA commit c8f03f4 기준으로 만들었다.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/knu_retarget.patch"
OPENVLA=${OPENVLA:-$HOME/vla/openvla}
BASE=c8f03f48af692657d3060c19588038c7220e9af9

cd "$OPENVLA"
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$BASE" ] || echo "주의: OpenVLA commit 이 $HEAD 로 패치 기준($BASE)과 다름. 적용이 실패하면 기준 commit 으로 맞춘다"

if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "이미 적용돼 있음"
else
    git apply --check "$PATCH"
    git apply "$PATCH"
    echo "패치 적용함"
fi
git diff --stat

# 확인: knu_retarget 이 관절각 7개를 모두 정규화하도록 등록됐는가
source ~/use_openvla.sh >/dev/null
python - <<'PY' 2>&1 | grep -v -E "tensorflow/|TF-TRT|cuda_|AVX|oneDNN|^$"
from pathlib import Path
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
k = make_oxe_dataset_kwargs("knu_retarget", Path("/tmp"))
assert k["absolute_action_mask"] == [True] * 7, k["absolute_action_mask"]
assert k["action_normalization_mask"] == [True] * 7, k["action_normalization_mask"]
print("OK knu_retarget:", {x: k[x] for x in ("image_obs_keys", "state_obs_keys", "action_normalization_mask")})
PY
