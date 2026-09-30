"""Metrics written to metrics.json: tracking of MANO anchors, per-frame QP tracking and world-pose postprocess."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from contactaware.contact.object_model import obj_to_world
from contactaware.settings import MM, config_report, make_config
from contactaware.solver.contact import anchor_outward_normals_world
from contactaware.solver.retarget import interpolation_ratio
from contactaware.solver.tracking import trajectory_tracking_report
from contactaware.types import ContactRuntime, RetargetInputs

METRIC_DECIMALS = 3
RESTING_DRIFT_DECIMALS = 6
TIME_DECIMALS = 3


def build_metrics_report(
    args,
    inputs: RetargetInputs,
    *,
    mano_anchor_tracking: dict,
    guidance_report: dict,
    ground_report: dict,
    qp_tracking_report: dict,
    initialization_seconds: float,
    key_seconds: float,
    qp_seconds: float,
    world_pose_seconds: float,
) -> dict:
    return {
        "sequence_policy": asdict(inputs.sequence_policy),
        "anchor_tracking": {
            "mano_anchor_retargeting": mano_anchor_tracking,
            "optimized_hand_guidance": guidance_report["anchor_tracking"],
        },
        "contact_guidance_source": guidance_report["source"],
        "qp_tracking": qp_tracking_report,
        "world_pose_postprocessing": {
            "policy": ground_report["world_pose_policy"],
            "ground_alignment_applied": bool(
                ground_report["ground_alignment_applied"]
            ),
            "object_ground_qp_applied": bool(
                ground_report["object_ground_qp_applied"]
            ),
            "initial_resting_applied": bool(
                ground_report["initial_resting_applied"]
            ),
            "object_smoothing": ground_report.get("trajectory_smoothing"),
        },
        "initial_resting": initial_resting_metrics(ground_report),
        "stable_contacts": {
            "mano": inputs.anchors.summary,
            "retargeted_hand": guidance_report["stable_contacts"],
        },
        "optimization_time_seconds": {
            "total": round(initialization_seconds + key_seconds + qp_seconds + world_pose_seconds, TIME_DECIMALS),
            "initialization": round(initialization_seconds, TIME_DECIMALS),
            "joint_keyframes": round(key_seconds, TIME_DECIMALS),
            "qp_retarget": round(qp_seconds, TIME_DECIMALS),
            "world_pose_postprocessing": round(
                world_pose_seconds, TIME_DECIMALS
            ),
        },
        "config": {"hand": args.hand, **config_report(args)},
    }


def initial_resting_metrics(ground_report: dict) -> dict:
    if not ground_report["initial_resting_applied"]:
        return {"applied": False}
    resting = ground_report["resting_correction"]
    return {
        "applied": True,
        "source": resting["source"],
        "converged": bool(resting["converged"]),
        "settle_converged": bool(resting["settle_converged"]),
        "verification_converged": bool(resting["verification_converged"]),
        "settle_seconds": round(float(resting["settle_seconds"]), TIME_DECIMALS),
        "settle_steps": int(resting["settle_steps"]),
        "stable_window_seconds": round(
            float(resting["stable_window_seconds"]), TIME_DECIMALS
        ),
        "velocity_decay_rate_s": float(resting["velocity_decay_rate_s"]),
        "terminal_linear_speed_mm_s": round(
            float(resting["terminal_linear_speed_mm_s"]), RESTING_DRIFT_DECIMALS
        ),
        "terminal_angular_speed_rad_s": round(
            float(resting["terminal_angular_speed_rad_s"]),
            RESTING_DRIFT_DECIMALS,
        ),
        "initial_ground_gap_mm": round(
            float(resting["initial_ground_gap_mm"]), RESTING_DRIFT_DECIMALS
        ),
        "terminal_pos_drift_mm": round(
            float(resting["verified_terminal_pos_drift_mm"]),
            RESTING_DRIFT_DECIMALS,
        ),
        "delta_pos_mm": resting["delta_pos_mm"],
        "delta_rot_rad": resting["delta_rot_rad"],
        "trajectory_smoothing": resting["trajectory_smoothing"],
    }


def anchor_tracking_metrics(
    inputs: RetargetInputs,
    runtime: ContactRuntime,
    *,
    qpos: np.ndarray,
    obj_pose: np.ndarray,
    ratio: int,
) -> dict:
    expected_internal_frames = (qpos.shape[0] - 1) * ratio + 1
    if runtime.anchors.mask.shape[0] != expected_internal_frames:
        raise ValueError(
            "Contact runtime/output frame mismatch: "
            f"{runtime.anchors.mask.shape[0]} != {expected_internal_frames}"
        )
    errors, normal_errors = [], []
    for output_frame in range(qpos.shape[0]):
        runtime_frame = output_frame * ratio
        active = np.flatnonzero(runtime.anchors.mask[runtime_frame])
        if active.size == 0:
            continue
        query_ids = runtime.hand_query_ids[runtime_frame, active]
        if np.any(query_ids < 0):
            raise ValueError("Active MANO anchor has no mapped robot query")
        query_world = inputs.hand.query_positions(qpos[output_frame], query_ids)
        anchor_obj = runtime.anchors.anchor_pos_obj[runtime_frame, active]
        anchor_world = obj_to_world(anchor_obj, obj_pose[output_frame])
        errors.extend(np.linalg.norm(query_world - anchor_world, axis=1).tolist())
        query_normals, _ = inputs.hand.query_normals_jacobian(
            qpos[output_frame], query_ids
        )
        target_normals = -anchor_outward_normals_world(
            runtime, obj_pose[output_frame], active
        )
        alignment = np.einsum("ni,ni->n", query_normals, target_normals)
        normal_errors.extend(np.arccos(np.clip(alignment, -1.0, 1.0)).tolist())
    error_mm = np.asarray(errors, dtype=np.float64) * MM
    if error_mm.size == 0:
        raise RuntimeError("Cannot report anchor tracking without active anchor samples")
    return {
        "sample_count": int(error_mm.size),
        "mean_error_mm": round(float(np.mean(error_mm)), METRIC_DECIMALS),
        "p95_error_mm": round(
            float(np.percentile(error_mm, 95.0)), METRIC_DECIMALS
        ),
        "max_error_mm": round(float(np.max(error_mm)), METRIC_DECIMALS),
        "normal_alignment_deg": normal_alignment_statistics(normal_errors),
    }


def normal_alignment_statistics(errors_rad: list[float]) -> dict:
    """Zero means opposing outward normals: the two contact surfaces face."""
    angles = np.rad2deg(np.asarray(errors_rad, dtype=np.float64))
    return {
        "sample_count": int(angles.size),
        "mean": round(float(np.mean(angles)), METRIC_DECIMALS),
        "p95": round(float(np.percentile(angles, 95.0)), METRIC_DECIMALS),
        "max": round(float(np.max(angles)), METRIC_DECIMALS),
    }


def qp_tracking_metrics(
    inputs: RetargetInputs,
    qpos: np.ndarray,
    runtime: ContactRuntime,
    *,
    args,
) -> tuple[int, dict]:
    ratio = interpolation_ratio(args.video_fps, args.internal_fps)
    report = trajectory_tracking_report(
        inputs.hand,
        qpos,
        inputs.mano_joints,
        cfg=make_config(args),
        obj_pose=inputs.obj_pose,
        runtime=runtime,
        runtime_ratio=ratio,
    )
    return ratio, report
