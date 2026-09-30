"""Interpolated MANO/object references and the contact runtime shared by all stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from contactaware.contact.mapping import assign_anchor_queries, make_contact_runtime
from contactaware.settings import make_config
from contactaware.types import ContactRuntime, RetargetInputs, SafeConfig
from contactaware.solver.contact import (
    HandObjectSafeDistancePolicy,
)
from contactaware.solver.hand import MujocoHand
from contactaware.solver.palm import target_palm_anchors


@dataclass(frozen=True)
class FrameProblem:
    hand: MujocoHand
    cfg: SafeConfig
    obj: object | None
    runtime: ContactRuntime | None
    joints: np.ndarray
    obj_pose: np.ndarray | None
    frame_id: int
    motion_target: np.ndarray | None = None
    hand_object_safe_distance_policy: HandObjectSafeDistancePolicy | None = None
    additional_objective: FrameObjective | None = None


FrameObjective = Callable[[FrameProblem, np.ndarray], tuple[np.ndarray, np.ndarray]]


def interpolate_object_poses(
    poses: np.ndarray,
    source_fps: int,
    internal_fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = interpolation_ratio(source_fps, internal_fps)
    if ratio == 1:
        return poses.copy(), np.arange(len(poses), dtype=np.int64)
    out = [
        object_pose_lerp(poses[i], poses[i + 1], step / ratio)
        for i in range(len(poses) - 1)
        for step in range(ratio)
    ]
    out.append(poses[-1])
    frame_ids = np.arange(len(out), dtype=np.int64) // ratio
    return np.asarray(out, dtype=np.float32), frame_ids


def object_pose_lerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    right_quat = -right[:4] if float(left[:4] @ right[:4]) < 0.0 else right[:4]
    quat = (1.0 - alpha) * left[:4] + alpha * right_quat
    quat = quat / np.linalg.norm(quat)
    pos = (1.0 - alpha) * left[4:7] + alpha * right[4:7]
    return np.concatenate([quat, pos])


def build_retarget_runtime(inputs: RetargetInputs, args, *, adapter=None) -> tuple:
    """Interpolated references, contact runtime and per-frame problem data."""
    cfg = make_config(args)
    joints, output_indices = interpolate_mano_joints(
        inputs.mano_joints,
        args.video_fps,
        args.internal_fps,
    )
    objects, source_frame_ids = interpolate_object_poses(
        inputs.obj_pose,
        args.video_fps,
        args.internal_fps,
    )
    vertices, _ = interpolate_mano_joints(
        inputs.mano_vertices,
        args.video_fps,
        args.internal_fps,
    )
    runtime = make_contact_runtime(
        inputs.anchors,
        source_frame_ids,
        approach_steps=int(round(args.contact_approach_seconds * args.internal_fps)),
        release_steps=int(round(args.contact_release_seconds * args.internal_fps)),
        mano_vertices=vertices,
        obj_pose=objects,
    )
    runtime = assign_anchor_queries(
        runtime,
        inputs.hand,
        inputs.mano_surface_mapping,
        cfg=cfg,
        contact_query_region=inputs.sequence_policy.contact_query_region,
    )
    if adapter is not None:
        runtime = adapter(runtime)
    return cfg, joints, objects, output_indices, runtime


def initial_qpos(
    hand: MujocoHand,
    joints: np.ndarray,
    *,
    cfg: SafeConfig,
) -> np.ndarray:
    qpos = hand.profile.default_qpos.astype(np.float64, copy=True)
    return palm_aligned_qpos(hand, qpos, joints, cfg=cfg)


def interpolate_mano_joints(
    mano_joints: np.ndarray,
    source_fps: int,
    internal_fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = interpolation_ratio(source_fps, internal_fps)
    if ratio == 1:
        return mano_joints.copy(), np.arange(len(mano_joints), dtype=np.int64)
    frames = [
        (1.0 - step / ratio) * mano_joints[i] + (step / ratio) * mano_joints[i + 1]
        for i in range(len(mano_joints) - 1)
        for step in range(ratio)
    ]
    frames.append(mano_joints[-1])
    return np.asarray(frames, dtype=np.float32), np.arange(0, len(frames), ratio)


def interpolation_ratio(source_fps: int, internal_fps: int) -> int:
    if source_fps <= 0 or internal_fps <= 0:
        raise ValueError("source and internal FPS must be positive")
    ratio = internal_fps / source_fps
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-9:
        raise ValueError(
            f"internal_fps must be an integer multiple of video_fps: "
            f"{internal_fps}/{source_fps}"
        )
    return rounded


def palm_aligned_qpos(
    hand: MujocoHand,
    qpos: np.ndarray,
    joints: np.ndarray,
    *,
    cfg: SafeConfig,
) -> np.ndarray:
    target = target_palm_anchors(joints, hand.profile.track_finger_ids)
    aligned = qpos.copy()
    limit = float(cfg.base_align_step)
    for _ in range(int(cfg.base_align_iters)):
        positions, jacobians = hand.palm_anchors_jacobian(aligned)
        step = palm_alignment_step(hand, positions, jacobians, target=target)
        aligned[:hand.base_qpos_dim] += np.clip(step, -limit, limit)
        aligned = np.clip(aligned, hand.lower, hand.upper)
    return aligned


def palm_alignment_step(
    hand: MujocoHand,
    positions: np.ndarray,
    jacobians: np.ndarray,
    *,
    target: np.ndarray,
) -> np.ndarray:
    residual = (positions - target).reshape(-1)
    jac = jacobians.reshape(-1, hand.qpos_dim)[:, :hand.base_qpos_dim]
    hessian = jac.T @ jac + np.eye(hand.base_qpos_dim) * 1e-6
    return -np.linalg.solve(hessian, jac.T @ residual)


