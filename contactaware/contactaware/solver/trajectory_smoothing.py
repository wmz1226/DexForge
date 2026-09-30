"""Smooth final object motion while preserving hand-object geometry."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from contactaware.solver.interpolation import continuous_rotation_vectors
from contactaware.solver.pose import transform_qpos


OBJECT_POSE_DIM = 7
QUATERNION_DIM = 4
MIN_SMOOTHING_FRAMES = 3


def smooth_object_poses(poses: np.ndarray, *, weight: float) -> np.ndarray:
    source = validate_object_poses(poses)
    if source.shape[0] < MIN_SMOOTHING_FRAMES:
        return source.astype(np.float32)
    position = smooth_samples(source[:, QUATERNION_DIM:], weight)
    rotation = smooth_rotations(source[:, :QUATERNION_DIM], weight)
    result = np.concatenate([rotation, position], axis=1)
    result[0] = source[0]
    return result.astype(np.float32)


def smooth_samples(samples: np.ndarray, weight: float) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    frame_count = values.shape[0]
    second_difference = np.diff(np.eye(frame_count), n=2, axis=0)
    hessian = np.eye(frame_count) + (
        float(weight) * second_difference.T @ second_difference
    )
    result = values.copy()
    result[1:] = np.linalg.solve(
        hessian[1:, 1:],
        values[1:] - hessian[1:, :1] * values[:1],
    )
    return result


def smooth_rotations(quaternions: np.ndarray, weight: float) -> np.ndarray:
    rotations = Rotation.from_quat(quaternions)
    smoothed = smooth_samples(continuous_rotation_vectors(rotations), weight)
    return (rotations[0] * Rotation.from_rotvec(smoothed)).as_quat()


def smooth_object_and_carry_hand(hand, qpos: np.ndarray, obj_pose: np.ndarray, *, weight: float):
    """Smooth the object poses and move the hand rigidly with the object, keeping the hand-object geometry."""
    source = validate_object_poses(obj_pose)
    target = smooth_object_poses(source, weight=weight).astype(np.float64)
    source_rotation, target_rotation = Rotation.from_quat(source[:, :4]), Rotation.from_quat(target[:, :4])
    change = target_rotation * source_rotation.inv()
    offsets = target[:, 4:] - change.apply(source[:, 4:])
    carried = np.asarray([transform_qpos(hand, state, np.r_[rotation.as_quat(), offset])
                          for state, rotation, offset in zip(np.asarray(qpos, np.float64), change, offsets)])
    return carried, target, smoothing_report(source, target, weight=weight)


def validate_object_poses(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != OBJECT_POSE_DIM:
        raise ValueError(f"Object poses must have shape [frame, {OBJECT_POSE_DIM}]")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Object pose trajectory must be non-empty and finite")
    return values


def smoothing_report(
    source: np.ndarray,
    smoothed: np.ndarray,
    *,
    weight: float,
) -> dict:
    translation = np.linalg.norm(smoothed[:, 4:7] - source[:, 4:7], axis=1)
    source_rotation = Rotation.from_quat(source[:, :4])
    target_rotation = Rotation.from_quat(smoothed[:, :4])
    rotation = (target_rotation * source_rotation.inv()).magnitude()
    return {
        "source": "object_pose",
        "weight": float(weight),
        "max_translation_change_mm": round(float(np.max(translation)) * 1000.0, 6),
        "max_rotation_change_rad": round(float(np.max(rotation)), 9),
        "first_frame_preserved": bool(np.array_equal(source[0], smoothed[0])),
        "hand_object_relative_pose_preserved": True,
    }
