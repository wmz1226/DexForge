"""Process-wide initialization required by the Warp backend."""

from __future__ import annotations

import os
import sys

from forceaware.config import RetargetConfig


def setup_runtime(cfg: RetargetConfig) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    warp_root = str(cfg.simulator.comfree_warp_root)
    if warp_root not in sys.path:
        sys.path.insert(0, warp_root)

    import warp as wp

    wp.init()
    wp.set_device(os.environ.get("FORCEAWARE_DEVICE", "cuda:0"))
