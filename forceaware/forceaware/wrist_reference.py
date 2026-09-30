"""Remove equivalent Euler branch jumps before actuator-coordinate interpolation."""

import mujoco
import numpy as np


TAU = 2.0 * np.pi


def equivalent_branches(angles):
    other = angles * np.array([1.0, -1.0, 1.0]) + np.pi
    return np.stack((angles, other), axis=-2)


def continuous_angles(angles, lower, upper):
    candidates = []
    for first in equivalent_branches(angles[0]):
        path = np.empty_like(angles, dtype=float)
        path[0] = first
        for frame in range(1, len(angles)):
            branches = equivalent_branches(angles[frame])
            branches += TAU * np.rint((path[frame - 1] - branches) / TAU)
            path[frame] = branches[np.argmin(np.sum((branches - path[frame - 1]) ** 2, axis=-1))]
        minimum = np.ceil((lower - path.min(axis=0)) / TAU - 1e-10)
        maximum = np.floor((upper - path.max(axis=0)) / TAU + 1e-10)
        if np.any(minimum > maximum):
            continue
        path += TAU * np.clip(np.rint((angles[0] - path[0]) / TAU), minimum, maximum)
        candidates.append(path)
    if not candidates:
        raise ValueError('No continuous wrist chart fits the joint limits')
    return min(candidates, key=lambda path: np.sum((path[0] - angles[0]) ** 2))


def continuous_wrist_reference(model, hand_qpos, hand):
    addresses = np.asarray(hand_qpos[3:6])
    joints = np.array([np.flatnonzero(model.jnt_qposadr == address)[0]
                       for address in addresses])
    axes = model.jnt_axis[joints]
    order = np.argmax(np.abs(axes), axis=1)
    signs = axes[np.arange(3), order]
    bodies = model.jnt_bodyid[joints]
    same_body = len(set(bodies)) == 1
    serial_wrist = (
        len(set(bodies)) == 3
        and np.array_equal(model.body_parentid[bodies[1:]], bodies[:-1])
        and np.all(model.body_jntnum[bodies] == 1)
        and np.allclose(model.body_pos[bodies[1:]], 0.0)
        and np.allclose(model.body_quat[bodies[1:]], [1.0, 0.0, 0.0, 0.0])
    )
    if (np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)
            or not (same_body or serial_wrist)
            or len(set(order)) != 3
            or not np.allclose(axes, np.eye(3)[order] * signs[:, None])
            or not np.allclose(model.jnt_pos[joints], 0.0)):
        raise ValueError('Expected three orthogonal, co-located virtual-wrist hinges')
    origin = model.qpos0[addresses]
    limits = np.where(model.jnt_limited[joints, None], model.jnt_range[joints],
                      np.array([-np.inf, np.inf]))
    limits = (limits - origin[:, None]) * signs[:, None]
    raw = (hand[:, 3:6].astype(float) - origin) * signs
    lifted = continuous_angles(raw, limits.min(axis=1), limits.max(axis=1))
    corrected = hand.copy()
    corrected[:, 3:6] = (lifted * signs + origin).astype(hand.dtype)
    corrected[:, 3:6] = np.where(lifted == raw, hand[:, 3:6], corrected[:, 3:6])
    return corrected
