"""Palm-relative MANO shape terms for contact-aware retargeting."""

from __future__ import annotations


import numpy as np

from contactaware.settings import MANO_CHAIN_JOINTS
from contactaware.solver.mano_regularization import (
    FINGER_ID_BY_LABEL,
    direction_jacobian,
    frame_mano_shape_weight,
    mano_chain_dirs,
)
from contactaware.solver.palm import human_palm_frame


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def local_direction_linearization(
    direction: np.ndarray,
    direction_jacobian_world: np.ndarray,
    palm_rotation: np.ndarray,
    palm_angular_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world direction and derivative in the moving palm frame."""
    correction = skew(direction) @ palm_angular_jacobian
    local_jacobian = palm_rotation.T @ (
        direction_jacobian_world + correction
    )
    return palm_rotation.T @ direction, local_jacobian


def palm_relative_shape_terms(hand, qpos, joints, *, fingers=None) -> dict[str, list]:
    selected = tuple(hand.profile.chain_points) if fingers is None else tuple(fingers)
    if not selected:
        return {}
    hand.forward(np.asarray(qpos, dtype=np.float64))
    _, robot_rotation, _, robot_angular_jacobian = hand.palm_pose_jacobian(qpos)
    _, human_rotation = human_palm_frame(joints, hand.profile.track_finger_ids)
    terms = {}
    for finger in selected:
        specs = hand.profile.chain_points[finger]
        points, point_jacobians = hand.current_chain_points_jacobian(specs)
        targets = mano_chain_dirs(joints, MANO_CHAIN_JOINTS[finger]) @ human_rotation
        terms[finger] = _finger_terms(
            points,
            point_jacobians,
            targets,
            robot_rotation,
            robot_angular_jacobian,
        )
    return terms


def _finger_terms(points, point_jacobians, targets, palm_rotation, palm_angular_jacobian):
    if len(targets) != len(points) - 1:
        raise ValueError("Robot and MANO finger chains have different segment counts")
    terms = []
    for index, target in enumerate(targets):
        vector = points[index + 1] - points[index]
        jacobian = point_jacobians[index + 1] - point_jacobians[index]
        direction, world_jacobian = direction_jacobian(vector, jacobian)
        local, local_jacobian = local_direction_linearization(
            direction,
            world_jacobian,
            palm_rotation,
            palm_angular_jacobian,
        )
        terms.append((local - target, local_jacobian))
    return terms


def palm_relative_shape_residual(hand, qpos, joints, fingers) -> tuple[np.ndarray, np.ndarray]:
    selected_fingers = tuple(fingers)
    by_finger = palm_relative_shape_terms(hand, qpos, joints, fingers=selected_fingers)
    selected = [term for finger in selected_fingers for term in by_finger[finger]]
    if not selected:
        return np.zeros(0), np.zeros((0, hand.qpos_dim))
    residuals, jacobians = zip(*selected)
    return np.concatenate(residuals), np.vstack(jacobians)


def unguided_shape_fingers(hand, active_finger_ids) -> tuple[str, ...]:
    active = set(np.asarray(active_finger_ids, dtype=np.int32))
    return tuple(
        finger
        for finger in hand.profile.chain_points
        if FINGER_ID_BY_LABEL[finger] not in active
    )


def frame_shape_weights(hand, *, cfg, runtime, frame_id):
    return {finger: frame_mano_shape_weight(finger, cfg=cfg, runtime=runtime, frame_id=frame_id)
            for finger in hand.profile.chain_points}


def active_shape_fingers(weights):
    # Exact zero is an absent objective term; no near-zero threshold is applied.
    return tuple(finger for finger, weight in weights.items() if weight != 0.0)


def weighted_shape_terms(hand, state, joints, *, cfg, runtime, frame_id):
    """Identical MANO shape residuals in compact, joint-key and trajectory solves."""
    weights = frame_shape_weights(hand, cfg=cfg, runtime=runtime, frame_id=frame_id)
    geometry = palm_relative_shape_terms(hand, state, joints, fingers=active_shape_fingers(weights))
    result = []
    for finger, specs in hand.profile.chain_points.items():
        if weights[finger] == 0.0:
            count = joints.shape[-1] * (len(specs) - 1)
            result.append((np.zeros(count), np.zeros((count, hand.qpos_dim))))
            continue
        residuals, jacobians = zip(*geometry[finger])
        scale = np.sqrt(weights[finger])
        result.append((scale * np.concatenate(residuals), scale * np.vstack(jacobians)))
    return result


