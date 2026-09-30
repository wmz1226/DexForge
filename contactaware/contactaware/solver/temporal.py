"""Time-integrated velocity and acceleration in physical palm/joint coordinates."""


import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

TRAPEZOID_WEIGHT = 0.5
JOINT_ACCELERATION_MULTIPLIER = 16.0
PALM_ACCELERATION_MULTIPLIER = 32.0


def time_sample_weights(times, *, single_frame_seconds):
    """Trapezoidal tracking quadrature on the same physical time axis as motion."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 1:
        return np.asarray([single_frame_seconds])
    gaps = np.diff(times)
    return TRAPEZOID_WEIGHT * np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]]


def time_difference_rows(times, smoothing_seconds):
    """Quadrature rows for tau² integral |v|² + tau⁴ integral |a|²."""
    times = np.asarray(times, dtype=np.float64)
    gaps = np.diff(times)
    if smoothing_seconds < 0. or np.any(gaps <= 0.):
        raise ValueError("Smoothing time must be nonnegative and key times strictly increasing")
    velocity = np.diff(np.eye(len(times)), axis=0) / gaps[:, None]
    rows = [smoothing_seconds * np.sqrt(gaps)[:, None] * velocity]
    if len(times) > 2:
        widths = (gaps[:-1] + gaps[1:]) / 2.
        acceleration = np.diff(velocity, axis=0) / widths[:, None]
        rows.append(smoothing_seconds**2 * np.sqrt(widths)[:, None] * acceleration)
    return np.concatenate(rows, axis=0)


def physical_motion_operator(times, seconds, weights, *, acceleration_multiplier=1.0):
    """One velocity/acceleration model for sparse keys and the full frame grid."""
    differences = time_difference_rows(times, seconds)
    operator = sparse.kron(sparse.csc_matrix(differences),
                           sparse.diags(np.sqrt(weights)), format="csc")
    scales = np.ones((len(differences), len(weights)))
    scales[len(times) - 1:] = np.sqrt(acceleration_multiplier)
    return sparse.diags(scales.ravel()) @ operator


def physical_motion_residuals(hand, states, operator, origin_rotation, *,
                              rotation_jacobian, coordinate_scale=1.0):
    values = [physical_coordinates(hand, state, origin_rotation, rotation_jacobian=rotation_jacobian)
              for state in states]
    coordinates = np.concatenate([item[0] for item in values])
    derivatives = sparse.block_diag([item[1] * coordinate_scale for item in values], format="csc")
    return operator @ coordinates, operator @ derivatives


def physical_coordinates(hand, state, origin_rotation, *, rotation_jacobian):
    position, rotation, position_jac, angular_jac = hand.palm_pose_jacobian(state)
    angle = Rotation.from_matrix(origin_rotation.T @ rotation).as_rotvec()
    angle_jac = rotation_jacobian(angle) @ origin_rotation.T @ angular_jac
    coordinates = np.concatenate((position, angle, state[hand.base_qpos_dim:]))
    joint_jac = np.eye(hand.qpos_dim)[hand.base_qpos_dim:]
    return coordinates, np.vstack((position_jac, angle_jac, joint_jac))
