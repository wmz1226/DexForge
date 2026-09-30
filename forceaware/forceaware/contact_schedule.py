"""Shared contact scheduling for ForceAware targets and metrics."""

from __future__ import annotations

import numpy as np


def contact_age_ramp(contact_mask: np.ndarray, ramp_frames: int) -> np.ndarray:
    """Return the objective's per-frame contact-onset multiplier."""
    mask = np.asarray(contact_mask)
    if mask.ndim != 2:
        raise ValueError(f"contact mask must be rank-2, got {mask.shape}")
    if ramp_frames < 0:
        raise ValueError("contact age ramp frames must be non-negative")
    if ramp_frames == 0:
        return np.ones_like(mask, dtype=np.float32)

    active = mask > 0.5
    ages = np.zeros_like(mask, dtype=np.float32)
    running = np.zeros(mask.shape[1], dtype=np.float32)
    for frame, row in enumerate(active):
        running = np.where(row, running + 1.0, 0.0)
        ages[frame] = running
    return np.clip((ages - 1.0) / float(ramp_frames), 0.0, 1.0)
