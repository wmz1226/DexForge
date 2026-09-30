"""Output recording, rendering, and metrics for force-aware retargeting."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .metric_contract import control_stats
from .metric_contract import intended_contact_stats
from .metric_contract import marked_query_stats
from .validation import require_finite_array, require_finite_tree
from forceaware.render import (
    contactaware_camera_pose_path,
    sequence_camera_pose_and_fovy,
)
from tools.render import (
    TrajectoryRenderRequest,
    render_scene_trajectory_video,
    scene_mesh_xml_for,
)


def save_video(
    *,
    qpos_traj,
    qpos_frames,
    out_path: Path,
    seq_dir: Path,
    xml_path: Path,
    fps: float,
    hand_name: str | None = None,
    camera_pose_path: Path | None = None,
) -> None:
    camera_pose, camera_fovy = sequence_camera_pose_and_fovy(
        seq_dir,
        camera_pose_path=camera_pose_path or contactaware_camera_pose_path(
            seq_dir, xml_path, hand_name=hand_name
        ),
    )
    render_scene_trajectory_video(
        TrajectoryRenderRequest(
            scene_xml=scene_mesh_xml_for(xml_path),
            out_path=out_path,
            fps=fps,
            camera_pose=camera_pose,
            camera_fovy=camera_fovy,
            qpos=qpos_traj,
            qpos_frames=qpos_frames,
        )
    )


def metric_stats(values):
    values = require_finite_array("metric values", np.asarray(values, np.float64))
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
        "terminal": float(values[-1]),
    }


METERS_TO_MM = 1000.0


CONTACT_THRESHOLD_EPS = 1e-9


def empty_dynamic_contact_stats(candidate_count):
    return {
        "candidate_points_per_frame": int(candidate_count),
        "mean_active_points_per_frame": 0.0,
        "mean_penetrating_points_per_frame": 0.0,
        "mean_mm": 0.0,
        "max_mm": 0.0,
    }


def dynamic_contact_stats(phi, threshold):
    if phi.size == 0:
        return empty_dynamic_contact_stats(0)

    contact_threshold = np.asarray(threshold, dtype=np.float64)
    active = phi < contact_threshold - CONTACT_THRESHOLD_EPS
    penetrating = active & (phi < 0.0)
    active_counts = active.sum(axis=1)
    penetrating_counts = penetrating.sum(axis=1)
    stats = empty_dynamic_contact_stats(phi.shape[1])
    stats.update(
        {
            "mean_active_points_per_frame": float(active_counts.mean()),
            "mean_penetrating_points_per_frame": float(penetrating_counts.mean()),
        }
    )
    if not penetrating.any():
        return stats

    penetration = -phi[penetrating] * METERS_TO_MM
    stats.update(
        {
            "mean_mm": float(penetration.mean()),
            "max_mm": float(penetration.max()),
        }
    )
    return stats


def marked_query_metrics(xpos, xmat, target_frames, *, cg, idx):
    frame_ids = metric_frame_ids(target_frames, cg)
    target = query_anchor_world(xpos, xmat, frame_ids, cg, idx)
    query = query_world_points(xpos, xmat, frame_ids, cg)
    distance_mm = np.linalg.norm(query - target, axis=-1) * METERS_TO_MM
    return marked_query_stats(
        distance_mm,
        frame_ids,
        contact_mask=cg["contact_mask"][frame_ids],
        contact_weight=cg["contact_weight"][frame_ids],
        age_ramp=cg["contact_age_ramp"][frame_ids],
    )


def intended_contact_metrics(
    phi,
    contact_pos,
    kinematics,
    *,
    target_frames,
    cg,
    idx,
    contact_threshold,
):
    xpos, xmat = kinematics
    frame_ids = metric_frame_ids(target_frames, cg)
    output_indices = cg["gs_output_indices"][frame_ids].astype(np.int32)
    time_indices = np.arange(frame_ids.size, dtype=np.int32)[:, None]
    selected_phi = phi[time_indices, output_indices]
    selected_pos = contact_pos[time_indices, output_indices]
    thresholds = np.asarray(contact_threshold, dtype=np.float64)
    selected_threshold = (
        thresholds if thresholds.ndim == 0 else thresholds[output_indices]
    )
    target = query_anchor_world(xpos, xmat, frame_ids, cg, idx)
    distance_mm = np.linalg.norm(selected_pos - target, axis=-1) * METERS_TO_MM
    collision_active = selected_phi < selected_threshold - CONTACT_THRESHOLD_EPS
    return intended_contact_stats(
        distance_mm,
        collision_active,
        contact_mask=cg["contact_mask"][frame_ids],
        contact_weight=cg["contact_weight"][frame_ids],
        age_ramp=cg["contact_age_ramp"][frame_ids],
    )


def metric_frame_ids(target_frames, cg):
    return np.clip(
        np.rint(target_frames).astype(np.int32),
        0,
        cg["contact_mask"].shape[0] - 1,
    )


def query_anchor_world(xpos, xmat, frame_ids, cg, idx):
    pos_obj = cg["contact_pos_obj"][frame_ids].astype(np.float64)
    obj_pos = xpos[:, idx["obj_bodyid"]].astype(np.float64)
    obj_mat = xmat[:, idx["obj_bodyid"]].astype(np.float64)
    return np.einsum("tij,tnj->tni", obj_mat, pos_obj) + obj_pos[:, None, :]


def query_world_points(xpos, xmat, frame_ids, cg):
    body_ids = cg["hand_query_body_ids"][frame_ids].astype(np.int32)
    local = cg["hand_query_local_pos"][frame_ids].astype(np.float64)
    t_idx = np.arange(frame_ids.shape[0])[:, None]
    body_pos = xpos[t_idx, body_ids].astype(np.float64)
    body_mat = xmat[t_idx, body_ids].astype(np.float64)
    return body_pos + np.einsum("tnij,tnj->tni", body_mat, local)


def control_metrics(
    qpos0,
    ctrl_traj,
    idx,
    *,
    mpc_dt,
    action_substeps,
    knot_substeps=None,
):
    init_ctrl = np.concatenate(
        [
            qpos0[idx["base_qadr"]],
            qpos0[idx["finger_qadr"]],
        ]
    ).astype(np.float64)
    return control_stats(
        init_ctrl,
        ctrl_traj,
        mpc_dt=mpc_dt,
        action_substeps=action_substeps,
        knot_substeps=knot_substeps,
    )


def write_metrics_json(out_path, metrics):
    require_finite_tree("metrics", metrics)
    out_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
