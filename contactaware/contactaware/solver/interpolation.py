"""Shape-preserving interpolation of key positions and rotations (no overshoot between unevenly spaced keys)."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


def position_interpolator(times, positions):
    return PchipInterpolator(times, np.asarray(positions, dtype=np.float64), axis=0)


def continuous_rotation_vectors(rotations):
    """Rotation vectors relative to the first rotation, each on the branch closest to its predecessor."""
    vectors = (rotations[0].inv() * rotations).as_rotvec()
    for index in range(1, len(vectors)):
        angle = np.linalg.norm(vectors[index])
        if angle > 1e-9:
            axis = vectors[index] / angle
            candidates = [vectors[index] + 2 * np.pi * k * axis for k in (-1, 0, 1)]
            vectors[index] = min(candidates, key=lambda v: np.linalg.norm(v - vectors[index - 1]))
    return vectors


def rotation_interpolator(times, rotations):
    """Monotone cubic interpolation of rotation vectors relative to the first key, on a continuous branch."""
    rotations = Rotation.from_matrix(np.asarray(rotations, dtype=np.float64))
    reference = rotations[0]
    spline = PchipInterpolator(times, continuous_rotation_vectors(rotations), axis=0)
    return lambda query: (reference * Rotation.from_rotvec(np.atleast_2d(spline(query)))).as_matrix().reshape(
        np.shape(query) + (3, 3))
