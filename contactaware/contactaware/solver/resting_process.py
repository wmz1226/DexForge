"""Settle the object once on the ground in the simulator, before world-pose post-processing."""

from __future__ import annotations

import numpy as np


def apply_initial_resting_correction(qpos, obj_pose, *, args, xml_path):
    from contactaware.solver.resting import apply_initial_resting_correction as settle

    hand, obj, summary = settle(np.asarray(qpos), np.asarray(obj_pose), args=args, xml_path=xml_path)
    if hand.shape != np.shape(qpos) or obj.shape != np.shape(obj_pose):
        raise ValueError("Settling returned an unexpected trajectory shape")
    if not np.isfinite(hand).all() or not np.isfinite(obj).all():
        raise FloatingPointError("Settling returned a non-finite trajectory")
    return hand, obj, summary
