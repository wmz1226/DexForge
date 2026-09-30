"""Optimize the object ground pose and invoke initial resting correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, minimize

from contactaware.settings import GROUND_PLANE_NORMAL, GROUND_PLANE_POS, MM
from contactaware.types import ContactAnchorSet, ObjectModel, RetargetInputs
from contactaware.solver.pose import translate_qpos_sequence
from contactaware.solver.trajectory_smoothing import smooth_object_and_carry_hand


FIRST_FRAME = 0


def object_ground_phi(obj_pose: np.ndarray, obj: ObjectModel) -> np.ndarray:
    return obj.query_fn.ground_clearance(obj_pose, GROUND_PLANE_POS, GROUND_PLANE_NORMAL)


def apply_object_ground_qp(
    qpos: np.ndarray,
    obj_pose: np.ndarray,
    *,
    hand,
    obj: ObjectModel,
    args,
) -> tuple:
    before = object_ground_phi(obj_pose, obj)
    dz = solve_object_ground_qp(before, args)
    obj_aligned = np.asarray(obj_pose, dtype=np.float32).copy()
    shift = dz[:, None] * GROUND_PLANE_NORMAL[None, :]
    hand_aligned = translate_qpos_sequence(hand, qpos, shift).astype(np.float32)
    obj_aligned[:, 4:7] += shift
    return hand_aligned, obj_aligned


def solve_object_ground_qp(phi: np.ndarray, args) -> np.ndarray:
    target = float(args.object_ground_margin)
    lower = target - np.asarray(phi, dtype=np.float64)
    if lower.ndim != 1 or lower.size == 0:
        raise ValueError("Object-ground QP requires a non-empty 1D phi array")
    upper = np.full(lower.size, np.inf, dtype=np.float64)
    upper[FIRST_FRAME] = lower[FIRST_FRAME]
    initial = np.maximum(lower, 0.0)
    initial[FIRST_FRAME] = lower[FIRST_FRAME]
    hessian = object_ground_qp_hessian(
        lower.size,
        args.object_ground_qp_reg,
        smooth=args.object_ground_qp_smooth,
        acc_smooth=args.object_ground_qp_acc_smooth,
    )
    result = minimize(
        lambda x: float(0.5 * x @ hessian @ x),
        initial,
        jac=lambda x: hessian @ x,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        options={
            "ftol": float(args.qp_ftol),
            "maxiter": int(args.qp_maxiter),
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(f"Object-ground QP failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def object_ground_qp_hessian(
    frame_count: int,
    reg: float,
    *,
    smooth: float,
    acc_smooth: float,
) -> np.ndarray:
    hessian = float(reg) * np.eye(frame_count, dtype=np.float64)
    if frame_count > 1 and smooth > 0.0:
        diff = np.diff(np.eye(frame_count, dtype=np.float64), axis=0)
        hessian += float(smooth) * (diff.T @ diff)
    if frame_count > 2 and acc_smooth > 0.0:
        acc = np.diff(np.eye(frame_count, dtype=np.float64), n=2, axis=0)
        hessian += float(acc_smooth) * (acc.T @ acc)
    return hessian + np.eye(frame_count, dtype=np.float64) * 1e-9


def apply_precontact_ground_alignment(
    qpos: np.ndarray,
    obj_pose: np.ndarray,
    *,
    hand,
    obj: ObjectModel,
    anchors: ContactAnchorSet,
) -> tuple:
    offset = precontact_ground_offset(obj_pose, obj, anchors)
    obj_aligned = np.asarray(obj_pose, dtype=np.float32).copy()
    hand_aligned = translate_qpos_sequence(hand, qpos, -offset * GROUND_PLANE_NORMAL).astype(np.float32)
    obj_aligned[:, 4:7] -= offset * GROUND_PLANE_NORMAL[None, :]
    return hand_aligned, obj_aligned


def precontact_ground_offset(obj_pose, obj, anchors):
    mask, first_contact = precontact_mask(anchors, obj_pose.shape[0])
    if first_contact >= obj_pose.shape[0]:
        raise ValueError(
            "Cannot estimate ground alignment without any contact-anchor frames"
        )
    if not np.any(mask):
        raise ValueError("Cannot estimate ground alignment without pre-contact frames")
    before = object_ground_phi(obj_pose, obj)
    return float(np.mean(before[mask]))


def precontact_mask(anchors: ContactAnchorSet, frame_count: int) -> tuple:
    active = (
        np.any(anchors.mask, axis=1)
        if anchors.mask.size
        else np.zeros(frame_count, bool)
    )
    first_contact = int(np.argmax(active)) if np.any(active) else int(frame_count)
    return np.arange(frame_count, dtype=np.int32) < first_contact, first_contact


def postprocess_world_pose(
    args,
    inputs: RetargetInputs,
    *,
    qpos: np.ndarray,
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    policy = inputs.sequence_policy
    if policy.ground_stabilization:
        return optimize_ground_stability(
            args, inputs, qpos=qpos, xml_path=xml_path
        )
    print(f"  ground postprocessing skipped: {policy.world_pose_policy}; object trajectory smoothed")
    qpos, obj_pose, smoothing = smooth_object_and_carry_hand(
        inputs.hand, qpos, inputs.obj_pose, weight=float(args.object_trajectory_smooth_weight)
    )
    report = {
        "world_pose_policy": policy.world_pose_policy,
        "ground_alignment_applied": False,
        "object_ground_qp_applied": False,
        "initial_resting_applied": False,
        "resting_correction": None,
        "trajectory_smoothing": smoothing,
    }
    return qpos, obj_pose, report


def optimize_ground_stability(
    args,
    inputs: RetargetInputs,
    *,
    qpos: np.ndarray,
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from contactaware.solver.resting_process import apply_initial_resting_correction

    obj_pose = inputs.obj_pose
    qpos, obj_pose = apply_precontact_ground_alignment(
        qpos,
        obj_pose,
        hand=inputs.hand,
        obj=inputs.obj,
        anchors=inputs.anchors,
    )
    qpos, obj_pose = apply_object_ground_qp(qpos, obj_pose, hand=inputs.hand, obj=inputs.obj, args=args)
    initial_ground_gap_mm = float(
        object_ground_phi(obj_pose[:1], inputs.obj)[FIRST_FRAME] * MM
    )
    qpos, obj_pose, resting = apply_initial_resting_correction(
        qpos,
        obj_pose,
        args=args,
        xml_path=xml_path,
    )
    resting["initial_ground_gap_mm"] = initial_ground_gap_mm
    report = {
        "world_pose_policy": inputs.sequence_policy.world_pose_policy,
        "ground_alignment_applied": True,
        "object_ground_qp_applied": True,
        "initial_resting_applied": True,
        "resting_correction": resting,
    }
    return qpos, obj_pose, report
