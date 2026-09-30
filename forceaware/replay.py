"""Replay saved controls in the MuJoCo mesh scene."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from forceaware.replay import run_default_mujoco_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--contactaware", type=Path, help="CA result directory supplying camera_pose_7.npy")
    args = parser.parse_args()
    camera = None if args.contactaware is None else args.contactaware / "camera_pose_7.npy"
    run_default_mujoco_replay(args.rollout, args.xml, args.out_dir, camera_pose_path=camera)


if __name__ == "__main__":
    main()
