"""Contact states defined by contiguous runs of unchanged stable-anchor IDs."""

from __future__ import annotations

import numpy as np


def contact_mask_runs(mask: np.ndarray) -> tuple[tuple[int, int, np.ndarray], ...]:
    """Keep every nonempty constant-mask run in time order, including repeats."""
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or min(active.shape) == 0:
        raise ValueError("Contact mask must have nonempty shape (frames, contacts)")
    runs = []
    start = 0
    for frame in range(1, len(active) + 1):
        if frame < len(active) and np.array_equal(active[frame], active[start]):
            continue
        columns = np.flatnonzero(active[start]).astype(np.int32)
        if columns.size:
            runs.append((start, frame - 1, columns))
        start = frame
    return tuple(runs)
