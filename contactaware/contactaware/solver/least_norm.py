"""Minimum-norm correction with structurally absent variables removed exactly."""

import numpy as np


def structural_least_norm(matrix, rhs, free):
    """Keep NumPy's default rank cutoff and return exact zeros for absent columns."""
    selected = free & np.any(matrix != 0.0, axis=0)
    cutoff = np.finfo(matrix.dtype).eps * max(matrix.shape[0], int(np.count_nonzero(free)))
    solution = np.zeros(matrix.shape[1], dtype=matrix.dtype)
    solution[selected] = np.linalg.lstsq(matrix[:, selected], rhs, rcond=cutoff)[0]
    return solution
