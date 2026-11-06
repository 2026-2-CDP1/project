"""knu_retarget RLDS 를 만든다 (<data-dir>/knu_retarget/1.0.0/).

  source ~/use_openvla.sh
  KNU_RETARGET_SRC=~/vla/data/knu_retarget_src python rlds/build.py --data-dir ~/vla/data
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knu_retarget import KnuRetarget  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    b = KnuRetarget(data_dir=str(Path(a.data_dir).expanduser()))
    b.download_and_prepare()
    info = b.info
    print(info.splits["train"].num_examples, "episodes ->", b.data_dir)


if __name__ == "__main__":
    main()
