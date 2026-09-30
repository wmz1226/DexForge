"""Convert robot base poses through FK, respecting asset axes and static transforms."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

XYZ_DIM = 3
FIT_TOLERANCE = 1e-9
WORLD_FRAME_LENGTH_SCALE_M = 1.0


def translate_qpos_sequence(hand, states, translations):
    """Apply world translations through the asset's actual base slide axes."""
    states = np.asarray(states, dtype=np.float64)
    translations = np.broadcast_to(translations, (len(states), XYZ_DIM))
    joint_types = hand.model.jnt_type[:hand.base_qpos_dim]
    slide_ids = np.flatnonzero(joint_types == mujoco.mjtJoint.mjJNT_SLIDE)
    if len(slide_ids) != XYZ_DIM:
        raise ValueError("Rigid hand translation requires three base slide joints")
    result = states.copy()
    for index, state in enumerate(states):
        _, _, position_rows, _ = hand.palm_pose_jacobian(state)
        result[index, slide_ids] += np.linalg.solve(position_rows[:, slide_ids], translations[index])
    return result


def _least_squares(function, initial, **kwargs):
    result = least_squares(function, initial, ftol=FIT_TOLERANCE,
                           xtol=FIT_TOLERANCE, gtol=FIT_TOLERANCE, **kwargs)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Grasp initialization failed: {result.message}")
    return result


def _fit_base_pose(hand, qpos, position, rotation, length_scale):
    """Solve through FK, including static body transforms and actual joint axes."""
    def residual(delta):
        state = np.asarray(qpos, dtype=np.float64).copy()
        state[:hand.base_qpos_dim] += delta
        hand.forward(state)
        current_position = hand.data.xpos[hand.palm_body_id]
        current_rotation = hand.data.xmat[hand.palm_body_id].reshape(XYZ_DIM, XYZ_DIM)
        orientation = Rotation.from_matrix(rotation.T @ current_rotation).as_rotvec()
        return np.concatenate(((current_position - position) / length_scale, orientation))

    # Solve increments to avoid a vanishing TRF trust radius near zero.
    result = _least_squares(residual, np.zeros(hand.base_qpos_dim))
    output = np.asarray(qpos, dtype=np.float64).copy()
    output[:hand.base_qpos_dim] += result.x
    return output


def fit_palm_pose(hand, qpos, position, rotation):
    """Fit the calibrated palm frame used by tracking, rather than the raw body."""
    body_rotation = rotation @ hand.palm_basis_local.T
    body_position = position - body_rotation @ hand.palm_origin_local
    return _fit_base_pose(hand, qpos, body_position, body_rotation, WORLD_FRAME_LENGTH_SCALE_M)


def transform_qpos(hand, q_obj, object_pose, *, base_reference=None) -> np.ndarray:
    """Compose object-frame robot FK with an object pose (xyzw quaternion, xyz)."""
    pose = np.asarray(object_pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Object pose must be finite with shape (7,)")
    object_rotation = Rotation.from_quat(pose[:4]).as_matrix()
    hand.forward(q_obj)
    position = hand.data.xpos[hand.palm_body_id].copy()
    rotation = hand.data.xmat[hand.palm_body_id].reshape(XYZ_DIM, XYZ_DIM).copy()
    initial = np.array(q_obj, copy=True)
    if base_reference is not None:
        initial[:hand.base_qpos_dim] = base_reference
    return _fit_base_pose(hand, initial, object_rotation @ position + pose[4:],
                          object_rotation @ rotation, WORLD_FRAME_LENGTH_SCALE_M)


def transform_qpos_sequence(hand, states, object_poses):
    """Compose poses by continuation, retaining a continuous Euler branch."""
    if len(states) != len(object_poses) or not len(states):
        raise ValueError("Coordinate conversion needs matching nonempty pose sequences")
    result = []
    previous = None
    for state, pose in zip(states, object_poses):
        transformed = transform_qpos(hand, state, pose, base_reference=previous)
        if previous is None:
            angles = transformed[XYZ_DIM:hand.base_qpos_dim]
            transformed[XYZ_DIM:hand.base_qpos_dim] = (angles + np.pi) % (2. * np.pi) - np.pi
        result.append(transformed)
        previous = transformed[:hand.base_qpos_dim]
    return np.asarray(result, dtype=np.float32)


