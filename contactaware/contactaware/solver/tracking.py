"""Phase-aware palm, fingertip, and contact tracking objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from contactaware.settings import (
    BASE_ROTATION_DIM,
    BASE_TRANSLATION_DIM,
    MM,
    mano_tip_joint_ids,
)
from contactaware.types import ContactRuntime, SafeConfig
from contactaware.solver.contact import anchor_constraint_linearization, anchor_normal_linearization
from contactaware.solver.hand import MujocoHand
from contactaware.solver.palm import human_palm_frame

SO3_SMALL_ANGLE = 1e-6
METRIC_DECIMALS = 3


@dataclass(frozen=True)
class TrackingResidual:
    cost_m: float
    palm_position_m: float
    palm_rotation_rad: float
    fingertip_m: float
    fingertip_finger_count: int
    contact_anchor_m: float


def palm_tracking_linearization(
    hand: MujocoHand,
    qpos: np.ndarray,
    joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_pos, target_rot = human_palm_frame(joints, hand.profile.track_finger_ids)
    return palm_pose_residual(hand, qpos, target_pos, target_rot)


def palm_pose_residual(hand, qpos, target_pos, target_rot):
    """Shared pose residual; callers supply either MANO or calibrated MANO targets."""
    position, rotation, pos_jac, angular_jac = hand.palm_pose_jacobian(qpos)
    rot_error = Rotation.from_matrix(target_rot.T @ rotation).as_rotvec()
    rot_jac = so3_left_jacobian_inverse(rot_error) @ target_rot.T @ angular_jac
    return position - target_pos, rot_error, pos_jac, rot_jac


def fingertip_tracking_linearization(
    hand: MujocoHand,
    qpos: np.ndarray,
    joints: np.ndarray,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track only fingers without a stable contact segment in the sequence."""
    selected = ~guided_finger_mask(hand, runtime, cfg=cfg)
    if not np.any(selected):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, hand.qpos_dim), dtype=np.float64),
        )
    positions, jacobians = hand.retarget_points_jacobian(qpos)
    target_ids = mano_tip_joint_ids(hand.profile.track_finger_ids)
    residual = positions[selected] - joints[target_ids[selected]]
    jacobian = jacobians[selected].reshape(-1, hand.qpos_dim)
    return residual, jacobian


def guided_finger_mask(
    hand: MujocoHand,
    runtime: ContactRuntime | None,
    *,
    cfg: SafeConfig,
) -> np.ndarray:
    """Tracked fingers that own at least one contact-guidance anchor column."""
    tracked = np.asarray(hand.profile.track_finger_ids, dtype=np.int32)
    if runtime is None or cfg.contact_anchor_weight <= 0.0:
        return np.zeros(tracked.size, dtype=bool)
    guided = np.unique(runtime.anchors.finger_ids).astype(np.int32)
    untracked = np.setdiff1d(guided, tracked)
    if untracked.size:
        raise ValueError(f"Guided fingers are not tracked: {untracked.tolist()}")
    return np.isin(tracked, guided)


def step_regularization_qp(
    hand: MujocoHand,
    cfg: SafeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    expected_base_dim = BASE_TRANSLATION_DIM + BASE_ROTATION_DIM
    if hand.base_qpos_dim != expected_base_dim:
        raise ValueError(
            f"Expected {expected_base_dim} base coordinates, got {hand.base_qpos_dim}"
        )
    weights = np.full(hand.qpos_dim, cfg.joint_step_weight, dtype=np.float64)
    weights[:BASE_TRANSLATION_DIM] = cfg.base_translation_step_weight
    weights[BASE_TRANSLATION_DIM:expected_base_dim] = cfg.base_rotation_step_weight
    if np.any(weights < 0.0) or cfg.min_hessian <= 0.0:
        raise ValueError("Step regularization weights must be non-negative")
    hessian = np.diag(weights + cfg.min_hessian)
    return hessian, np.zeros(hand.qpos_dim, dtype=np.float64)


def velocity_weights(hand: MujocoHand, cfg: SafeConfig) -> np.ndarray:
    values = np.asarray(
        [
            cfg.base_translation_velocity_weight,
            cfg.base_rotation_velocity_weight,
            cfg.joint_velocity_weight,
        ],
        dtype=np.float64,
    )
    if np.any(values < 0.0):
        raise ValueError("Velocity weights must be non-negative")
    weights = np.full(hand.qpos_dim, values[2], dtype=np.float64)
    weights[:BASE_TRANSLATION_DIM] = values[0]
    weights[BASE_TRANSLATION_DIM : hand.base_qpos_dim] = values[1]
    return weights


def frame_tracking_residual(
    hand: MujocoHand,
    qpos: np.ndarray,
    joints: np.ndarray,
    *,
    cfg: SafeConfig,
    obj_pose,
    runtime: ContactRuntime | None,
    frame_id: int,
) -> TrackingResidual:
    pos_error, rot_error, _, _ = palm_tracking_linearization(hand, qpos, joints)
    tip_error, _ = fingertip_tracking_linearization(
        hand, qpos, joints, cfg=cfg, runtime=runtime
    )
    contact_error = contact_tracking_residual(
        hand, qpos, obj_pose, cfg=cfg, runtime=runtime, frame_id=frame_id
    )
    normal_error = contact_normal_tracking_residual(
        hand, qpos, obj_pose, cfg=cfg, runtime=runtime, frame_id=frame_id
    )
    return TrackingResidual(
        cost_m=weighted_tracking_cost(
            pos_error,
            rot_error,
            tip_error,
            contact_anchor=np.vstack((contact_error, normal_error)),
            cfg=cfg,
        ),
        palm_position_m=float(np.linalg.norm(pos_error)),
        palm_rotation_rad=float(np.linalg.norm(rot_error)),
        fingertip_m=maximum_point_error(tip_error),
        fingertip_finger_count=int(tip_error.shape[0]),
        contact_anchor_m=maximum_point_error(contact_error),
    )


def trajectory_tracking_report(
    hand: MujocoHand,
    qpos: np.ndarray,
    joints: np.ndarray,
    *,
    cfg: SafeConfig,
    obj_pose: np.ndarray,
    runtime: ContactRuntime,
    runtime_ratio: int,
) -> dict:
    validate_trajectory_shapes(
        qpos, joints, obj_pose, runtime=runtime, runtime_ratio=runtime_ratio
    )
    residuals = []
    contact_frames = []
    for output_frame, (frame_qpos, frame_joints) in enumerate(zip(qpos, joints)):
        runtime_frame = output_frame * runtime_ratio
        residuals.append(
            frame_tracking_residual(
                hand,
                frame_qpos,
                frame_joints,
                cfg=cfg,
                obj_pose=obj_pose[output_frame],
                runtime=runtime,
                frame_id=runtime_frame,
            )
        )
        contact_frames.append(bool(runtime.anchors.mask[runtime_frame].any()))
    active = np.asarray(contact_frames, dtype=bool)
    return {
        "all_frames": tracking_residual_statistics(residuals),
        "noncontact_frames": tracking_residual_statistics(
            [item for item, is_active in zip(residuals, active) if not is_active]
        ),
        "contact_frames": tracking_residual_statistics(
            [item for item, is_active in zip(residuals, active) if is_active]
        ),
    }


def validate_trajectory_shapes(
    qpos: np.ndarray,
    joints: np.ndarray,
    obj_pose: np.ndarray,
    *,
    runtime: ContactRuntime,
    runtime_ratio: int,
) -> None:
    if not (len(qpos) == len(joints) == len(obj_pose)):
        raise ValueError("QP tracking report trajectory lengths do not match")
    expected_runtime_frames = (len(qpos) - 1) * runtime_ratio + 1
    if runtime.anchors.mask.shape[0] != expected_runtime_frames:
        raise ValueError("QP tracking report runtime length does not match output")


def tracking_residual_statistics(residuals: list[TrackingResidual]) -> dict:
    if not residuals:
        return {"frame_count": 0}
    tracked_fingers = max(item.fingertip_finger_count for item in residuals)
    return {
        "frame_count": len(residuals),
        "tracking_cost_mm": distribution([item.cost_m for item in residuals], MM),
        "palm_position_mm": distribution(
            [item.palm_position_m for item in residuals], MM
        ),
        "palm_rotation_deg": distribution(
            [item.palm_rotation_rad for item in residuals], 180.0 / np.pi
        ),
        # Report null when no unguided fingers remain.
        "fingertip_tracked_fingers": int(tracked_fingers),
        "fingertip_max_error_mm": (
            distribution([item.fingertip_m for item in residuals], MM)
            if tracked_fingers
            else None
        ),
        "contact_anchor_max_error_mm": distribution(
            [item.contact_anchor_m for item in residuals], MM
        ),
    }


def distribution(values: list[float], scale: float) -> dict:
    array = np.asarray(values, dtype=np.float64) * float(scale)
    return {
        "mean": round(float(np.mean(array)), METRIC_DECIMALS),
        "p95": round(float(np.percentile(array, 95.0)), METRIC_DECIMALS),
        "max": round(float(np.max(array)), METRIC_DECIMALS),
    }


def contact_tracking_residual(
    hand: MujocoHand,
    qpos: np.ndarray,
    obj_pose,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
    frame_id: int,
) -> np.ndarray:
    data = anchor_constraint_linearization(
        hand, qpos, obj_pose, cfg=cfg, runtime=runtime, frame_id=frame_id
    )
    if data is None:
        return np.zeros((0, 3), dtype=np.float64)
    positions, _, targets, participation = data
    return np.sqrt(participation)[:, None] * (positions - targets)


def contact_normal_tracking_residual(
    hand: MujocoHand,
    qpos: np.ndarray,
    obj_pose,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
    frame_id: int,
) -> np.ndarray:
    data = anchor_normal_linearization(
        hand, qpos, obj_pose, cfg=cfg, runtime=runtime, frame_id=frame_id
    )
    if data is None:
        return np.zeros((0, 3), dtype=np.float64)
    residual, _, participation = data
    return np.sqrt(participation)[:, None] * residual


def weighted_tracking_cost(
    palm_position: np.ndarray,
    palm_rotation: np.ndarray,
    fingertip: np.ndarray,
    *,
    contact_anchor: np.ndarray,
    cfg: SafeConfig,
) -> float:
    terms = (
        cfg.palm_position_weight * squared_norm(palm_position),
        cfg.palm_rotation_weight * squared_norm(palm_rotation),
        cfg.fingertip_weight * squared_norm(fingertip),
        cfg.contact_anchor_weight * squared_norm(contact_anchor),
    )
    return float(np.sqrt(sum(terms)))


def squared_norm(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.reshape(-1) @ array.reshape(-1))


def maximum_point_error(residual: np.ndarray) -> float:
    if residual.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(residual.reshape(-1, 3), axis=1)))


def so3_left_jacobian_inverse(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    skew = skew_matrix(vector)
    if theta < SO3_SMALL_ANGLE:
        return np.eye(3) - 0.5 * skew + (skew @ skew) / 12.0
    half_theta = 0.5 * theta
    coefficient = (
        1.0 - half_theta * np.cos(half_theta) / np.sin(half_theta)
    ) / (theta * theta)
    return np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)


def skew_matrix(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
