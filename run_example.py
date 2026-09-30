#!/usr/bin/env python3
"""Run ContactAware and ForceAware with one configured simulator geometry."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent
SEQUENCE = ROOT / "data/dexycb/dexycb_004_sugar_box_20200709_142802"


def prepare_sequence(source, destination):
    """Copy run inputs because wrist mounting edits scene XML files in place."""
    destination.mkdir()
    for name in ("mano_raw", "gs", "model", "scene"):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name)
    for xml in (destination / "scene").rglob("*.xml"):
        original = source / xml.relative_to(destination)

        def rebase(match):
            value = match.group(2)
            path = (original.parent / value).resolve()
            if path.is_relative_to(source):
                path = destination / path.relative_to(source)
            return 'file="' + os.path.relpath(path, xml.parent) + '"'

        xml.write_text(re.sub(r"file=([\"'])(.*?)\1", rebase, xml.read_text()))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=SEQUENCE)
    parser.add_argument("--hand", default="leaphand")
    parser.add_argument("--scene", choices=("gs", "mesh"), default="gs")
    parser.add_argument("--mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--contact-topk", type=int, choices=range(2, 9), default=8,
                        help="Soft query seed count; hard always uses 1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mano-models", type=Path, default=ROOT / "assets/mano")
    args = parser.parse_args()
    sequence = args.sequence_dir.expanduser().resolve()
    if not (sequence / "mano_raw/meta.json").is_file():
        parser.error(f"Missing sequence inputs: {sequence}")
    mano = args.mano_models.expanduser().resolve()
    if not (mano / "MANO_RIGHT.pkl").is_file():
        parser.error(f"Missing MANO_RIGHT.pkl: {mano}")
    output = (args.output_dir or ROOT / "results" / f"{args.scene}_{args.mode}" / sequence.name).resolve()
    if output.exists():
        parser.error(f"Results already exist: {output}; choose a new --output-dir")
    output.mkdir(parents=True)
    sequence = prepare_sequence(sequence, output / "sequence")
    topk = 1 if args.mode == "hard" else args.contact_topk
    # sdf is the existing XML filename for the simulator's triangle-mesh target.
    scene = "sdf" if args.scene == "mesh" else "gs"
    config = yaml.safe_load((ROOT / "forceaware/configs/forceaware.yaml").read_text())
    config["sequence"].update(hand=args.hand, scene=scene)
    config["contact_physics"]["contact_topk"] = topk
    config_path = output / "forceaware_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    environment = dict(os.environ)
    environment.update(MANO_MODEL_ROOT=str(mano), MANOPTH_ROOT=str(ROOT / "contactaware/third_party/manopth"),
                       HAND_ASSETS_ROOT=str(ROOT / "assets"),
                       COMFREE_WARP_ROOT=str(ROOT / "forceaware/third_party/comfree_warp"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1")
    stages = [
        ("contactaware", [str(ROOT / "contactaware/retarget.py"), "--sequence-dir", str(sequence),
                          "--hand", args.hand, "--output-dir", str(output / "contactaware"),
                          "--set", f"scene={scene}", "--set", f"contact_topk={topk}"]),
        ("forceaware", [str(ROOT / "forceaware/retarget.py"), "--sequence", str(sequence),
                        "--hand", args.hand, "--contactaware", str(output / "contactaware"),
                        "--config", str(config_path), "--output", str(output / "forceaware")]),
    ]
    timings = {}
    for stage, command in stages:
        print(f"Running {stage}: {args.scene}/{args.mode}", flush=True)
        started = time.perf_counter()
        subprocess.run([sys.executable, "-B", *command], cwd=ROOT, env=environment, check=True)
        timings[f"{stage}_wall_seconds"] = time.perf_counter() - started
        (output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(output / "contactaware/retarget.mp4")
    print(output / "forceaware/rollout.mp4")


if __name__ == "__main__":
    main()
