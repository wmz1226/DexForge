"""Run ForceAware MPC on a prepared reference sequence."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from forceaware.config import CONFIG_PATH, load_retarget_config
from forceaware.runtime import setup_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--config", default=CONFIG_PATH, type=Path)
    parser.add_argument("--contactaware", type=Path, help="ContactAware result directory")
    parser.add_argument("--hand", help="Robot name; otherwise use the configuration.")
    parser.add_argument("--contact-topk", type=int, choices=range(1, 9), default=None,
                        help="Override contact fusion for both planning and execution")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory; existing results are never overwritten.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    cfg = load_retarget_config(args.sequence, args.config, hand=args.hand,
                               contactaware_dir=args.contactaware)
    if args.contact_topk is not None:
        cfg = replace(cfg, simulator=replace(cfg.simulator, contact_topk=args.contact_topk))
    cfg = replace(cfg, sequence=replace(cfg.sequence, output_dir_name=str(output)))
    setup_runtime(cfg)
    from forceaware.workflow import rolling_mpc_retarget

    rolling_mpc_retarget(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
