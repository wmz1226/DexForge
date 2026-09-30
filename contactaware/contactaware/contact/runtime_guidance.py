"""Contact schedule, anchors and targets of the trajectory solve, exported for the dynamics stage."""

from __future__ import annotations

import numpy as np

from contactaware.settings import MM
from contactaware.types import ContactRuntime
from contactaware.contact.object_model import world_to_obj
from contactaware.contact.guidance import (
    METRIC_DECIMALS,
    POSITION_DECIMALS,
    HandContactSequence,
    empty_contact_guidance,
    link_label,
)
from contactaware.contact.mapping import query_groups

GUIDANCE_SOURCE = "qp_runtime_contact_targets"


def build_runtime_contact_guidance(
    sequence: HandContactSequence,
    runtime: ContactRuntime,
    interpolation_ratio: int,
) -> tuple[dict, dict]:
    """Contact guidance of the final trajectory from the solver runtime, without rescanning contacts."""
    output_indices = runtime_output_indices(
        sequence.qpos.shape[0],
        runtime.anchors.mask.shape[0],
        interpolation_ratio,
    )
    sampled_mask = runtime.anchors.mask[output_indices]
    sampled_anchors = runtime.contact_target_pos_obj[output_indices]
    sampled_queries = runtime.hand_query_ids[output_indices]
    validate_runtime_contacts(sequence, sampled_mask, sampled_anchors, sampled_queries)
    query = sequence.hand.query_points
    groups = query_groups(query)
    unique_groups = np.unique(groups, axis=0)
    labels = np.asarray(
        [link_label(int(finger), int(link)) for finger, link in unique_groups]
    )
    guidance = empty_contact_guidance(
        sequence.qpos.shape[0],
        unique_groups.shape[0],
        labels,
    )
    group_to_column = {
        (int(finger), int(link)): index
        for index, (finger, link) in enumerate(unique_groups)
    }
    write_runtime_contacts(
        guidance,
        sampled_mask,
        sampled_anchors,
        sampled_queries,
        query=query,
        groups=groups,
        group_to_column=group_to_column,
    )
    return guidance, {
        "source": GUIDANCE_SOURCE,
        "anchor_tracking": runtime_guidance_anchor_metrics(sequence, guidance),
        "stable_contacts": runtime_stable_contact_report(
            sampled_mask,
            sampled_anchors,
            sampled_queries,
            query=query,
            groups=groups,
        ),
    }


def runtime_output_indices(
    output_frame_count: int,
    runtime_frame_count: int,
    interpolation_ratio: int,
) -> np.ndarray:
    if output_frame_count <= 0 or interpolation_ratio <= 0:
        raise ValueError("Output frame count and interpolation ratio must be positive")
    expected = (output_frame_count - 1) * interpolation_ratio + 1
    if runtime_frame_count != expected:
        raise ValueError(
            "Contact runtime/output frame mismatch: "
            f"{runtime_frame_count} != {expected}"
        )
    return np.arange(output_frame_count, dtype=np.int64) * interpolation_ratio


def validate_runtime_contacts(
    sequence: HandContactSequence,
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
) -> None:
    frame_count = sequence.qpos.shape[0]
    if mask.ndim != 2 or mask.shape[0] != frame_count:
        raise ValueError("Sampled contact mask must have one row per output frame")
    expected = mask.shape + (3,)
    if anchors.shape != expected or query_ids.shape != mask.shape:
        raise ValueError("Sampled anchors and query IDs do not match the contact mask")
    active_queries = query_ids[mask]
    query_count = len(sequence.hand.query_points.local_pos)
    if active_queries.size == 0:
        raise RuntimeError("Cannot export QP guidance without active contacts")
    if np.any((active_queries < 0) | (active_queries >= query_count)):
        raise IndexError("QP contact contains an invalid robot query ID")
    if not np.all(np.isfinite(anchors[mask])):
        raise ValueError("QP contact contains a non-finite object target")


def write_runtime_contacts(
    guidance: dict,
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
    *,
    query,
    groups: np.ndarray,
    group_to_column: dict,
) -> None:
    for frame, anchor_column in np.argwhere(mask):
        query_id = int(query_ids[frame, anchor_column])
        group = tuple(int(value) for value in groups[query_id])
        output_column = group_to_column[group]
        if guidance["contact_mask"][frame, output_column] != 0.0:
            raise ValueError(
                "Two simultaneous QP contacts map to one robot finger-link group"
            )
        guidance["contact_mask"][frame, output_column] = 1.0
        guidance["contact_weight"][frame, output_column] = 1.0
        guidance["contact_pos_obj"][frame, output_column] = anchors[
            frame, anchor_column
        ]
        guidance["hand_query_indices"][frame, output_column] = query_id
        guidance["hand_query_body_ids"][frame, output_column] = query.body_ids[
            query_id
        ]
        guidance["hand_query_local_pos"][frame, output_column] = query.local_pos[
            query_id
        ]


def runtime_guidance_anchor_metrics(
    sequence: HandContactSequence,
    guidance: dict,
) -> dict:
    errors = []
    for frame in range(sequence.qpos.shape[0]):
        active = np.flatnonzero(guidance["contact_mask"][frame])
        if active.size == 0:
            continue
        query_ids = guidance["hand_query_indices"][frame, active]
        query_world = sequence.hand.query_positions(sequence.qpos[frame], query_ids)
        query_obj = world_to_obj(query_world, sequence.obj_pose[frame])
        anchors = guidance["contact_pos_obj"][frame, active]
        errors.extend(np.linalg.norm(query_obj - anchors, axis=1).tolist())
    return error_statistics_mm(errors)


def error_statistics_mm(errors_m: list[float]) -> dict:
    values = np.asarray(errors_m, dtype=np.float64) * MM
    if values.size == 0:
        raise RuntimeError("Cannot report contact guidance without active samples")
    return {
        "sample_count": int(values.size),
        "query_anchor_error_mean_mm": round(float(np.mean(values)), METRIC_DECIMALS),
        "query_anchor_error_p95_mm": round(
            float(np.percentile(values, 95.0)), METRIC_DECIMALS
        ),
        "query_anchor_error_max_mm": round(float(np.max(values)), METRIC_DECIMALS),
    }


def runtime_stable_contact_report(
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
    *,
    query,
    groups: np.ndarray,
) -> dict:
    segments = [
        runtime_contact_segment(
            column,
            mask=mask,
            anchors=anchors,
            query_ids=query_ids,
            query=query,
            groups=groups,
        )
        for column in range(mask.shape[1])
    ]
    return {"segment_count": len(segments), "segments": segments}


def runtime_contact_segment(
    column: int,
    *,
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
    query,
    groups: np.ndarray,
) -> dict:
    frames = np.flatnonzero(mask[:, column])
    if frames.size == 0 or not np.array_equal(
        frames, np.arange(frames[0], frames[-1] + 1)
    ):
        raise ValueError("Every QP contact must be one contiguous segment")
    stable_queries = np.unique(query_ids[frames, column])
    stable_anchors = anchors[frames, column]
    if stable_queries.size != 1 or not np.all(
        stable_anchors == stable_anchors[0]
    ):
        raise ValueError("QP contact must have one fixed target/query pair")
    query_id = int(stable_queries[0])
    finger_id, link_id = (int(value) for value in groups[query_id])
    return {
        "link": link_label(finger_id, link_id),
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "frame_count": int(frames.size),
        "anchor_pos_obj": np.round(
            stable_anchors[0].astype(float), POSITION_DECIMALS
        ).tolist(),
        "hand_query_index": query_id,
        "hand_query_body_id": int(query.body_ids[query_id]),
        "hand_query_local_pos": np.round(
            query.local_pos[query_id].astype(float), POSITION_DECIMALS
        ).tolist(),
    }
