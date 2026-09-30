"""MANO palm-frame construction shared by initialization and QP tracking."""

from __future__ import annotations

import numpy as np

from contactaware.contact.surface import normalize_vector
from contactaware.settings import PALM_ANCHORS, mano_base_joint_ids, mano_tip_joint_ids


def target_palm_anchors(
    joints: np.ndarray,
    finger_ids: tuple[int, ...],
) -> np.ndarray:
    origin, basis = human_palm_frame(joints, finger_ids)
    return origin + PALM_ANCHORS @ basis.T


def human_palm_frame(
    joints: np.ndarray,
    finger_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    base_ids = mano_base_joint_ids(finger_ids)
    tip_ids = mano_tip_joint_ids(finger_ids)
    origin_ids = np.concatenate([np.asarray([0], dtype=np.int64), base_ids])
    origin = joints[origin_ids].mean(axis=0)
    left_id, right_id = palm_axis_joint_ids(finger_ids)
    x_axis = normalize_vector(
        joints[left_id] - joints[right_id], name="human palm lateral axis"
    )
    y_seed = normalize_vector(
        joints[tip_ids].mean(axis=0) - joints[0], name="human palm forward axis"
    )
    z_axis = normalize_vector(np.cross(x_axis, y_seed), name="human palm normal")
    y_axis = normalize_vector(
        np.cross(z_axis, x_axis), name="human palm orthogonal axis"
    )
    return origin, np.stack([x_axis, y_axis, z_axis], axis=1)


def palm_axis_joint_ids(finger_ids: tuple[int, ...]) -> tuple[int, int]:
    fingers = tuple(int(value) for value in finger_ids)
    if 0 not in fingers:
        raise ValueError(f"Palm frame requires index finger, got {fingers}")
    right_finger = 5 if 5 in fingers else 2
    if right_finger not in fingers:
        raise ValueError(f"Palm frame requires ring or pinky finger, got {fingers}")
    left_id = int(mano_base_joint_ids((0,))[0])
    right_id = int(mano_base_joint_ids((right_finger,))[0])
    return left_id, right_id
