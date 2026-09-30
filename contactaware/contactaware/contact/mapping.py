"""Map stable MANO anchors to fixed robot-hand surface query IDs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from contactaware.contact.object_model import world_to_obj

from contactaware.types import (
    FULL_TERMINAL_SURFACE,
    TERMINAL_PAD,
    ContactAnchorSet,
    ContactRuntime,
    ManoSurfaceMapping,
    QueryPointSet,
    SafeConfig,
    SurfaceDescriptorSet,
)
from contactaware.contact.mano_surface import mano_reference_descriptor
from contactaware.contact.surface import (
    AXIAL_COMPONENT,
    PALMAR_AZIMUTH_COMPONENT,
    PALMAR_NORMAL_COMPONENT,
    subset_descriptors,
    contact_surface_descriptor_cost,
)

if TYPE_CHECKING:
    from contactaware.solver.hand import MujocoHand


def tip_query_ids(query: QueryPointSet, finger_id: int) -> np.ndarray:
    query_ids = np.flatnonzero((query.finger_ids == int(finger_id)) & query.is_tip)
    if query_ids.size == 0:
        raise ValueError(f"No tip query points for finger_id={finger_id}")
    return query_ids.astype(np.int32)


def contact_query_ids(
    hand: MujocoHand,
    finger_id: int,
    query_region: str,
    *,
    cfg: SafeConfig,
) -> np.ndarray:
    query_ids = tip_query_ids(hand.query_points, finger_id)
    descriptors = subset_descriptors(hand.query_surface_descriptors, query_ids)
    values = descriptors.values
    axial_mask = (
        (values[:, AXIAL_COMPONENT] >= cfg.pad_min_axial)
        & (values[:, AXIAL_COMPONENT] <= cfg.pad_max_axial)
    )
    if query_region == FULL_TERMINAL_SURFACE:
        mask = axial_mask
    elif query_region == TERMINAL_PAD:
        mask = (
            axial_mask
            & descriptors.azimuth_valid
            & (values[:, PALMAR_AZIMUTH_COMPONENT] >= cfg.pad_min_palmar)
            & (values[:, PALMAR_NORMAL_COMPONENT] >= cfg.pad_min_palmar)
        )
    else:
        raise ValueError(f"Unsupported contact query region: {query_region!r}")
    selected = query_ids[mask]
    if selected.size == 0:
        raise ValueError(
            f"No {query_region} queries for finger_id={finger_id} "
            f"on hand={hand.profile.name}"
        )
    return selected.astype(np.int32)


def query_groups(query: QueryPointSet) -> np.ndarray:
    return np.stack(
        [
            query.finger_ids.astype(np.int32),
            query.link_ids.astype(np.int32),
        ],
        axis=1,
    )


def make_contact_runtime(
    anchors: ContactAnchorSet,
    frame_ids: np.ndarray,
    *,
    approach_steps: int,
    release_steps: int,
    mano_vertices: np.ndarray,
    obj_pose: np.ndarray,
) -> ContactRuntime:
    expanded = ContactAnchorSet(
        anchors.mask[frame_ids].copy(),
        anchors.anchor_pos_obj[frame_ids].copy(),
        anchors.anchor_normal_obj[frame_ids].copy(),
        anchors.finger_ids,
        anchors.mano_vertex_ids[frame_ids].copy(),
        anchors.summary,
    )
    blend = contact_target_blend(
        expanded.mask,
        approach_steps,
        release_steps=release_steps,
    )
    targets = scheduled_anchor_targets(
        expanded,
        blend=blend,
        mano_vertices=mano_vertices,
        obj_pose=obj_pose,
    )
    ids = np.full(expanded.mano_vertex_ids.shape, -1, dtype=np.int32)
    return ContactRuntime(
        expanded,
        ids,
        finger_participation(expanded),
        targets,
        blend,
    )


def contact_target_blend(
    mask: np.ndarray,
    approach_steps: int,
    *,
    release_steps: int,
) -> np.ndarray:
    """Blend MANO vertices into object anchors with linear approach and release ramps."""
    if mask.ndim != 2 or min(approach_steps, release_steps) < 0:
        raise ValueError("Contact mask must be 2-D and ramp steps must be non-negative")
    blend = np.asarray(mask, dtype=np.float64).copy()
    for column in range(mask.shape[1]):
        active = np.flatnonzero(mask[:, column])
        if active.size == 0:
            raise ValueError(f"Contact column {column} has no active frames")
        start = int(active[0])
        ramp_start = max(0, start - approach_steps)
        if ramp_start < start:
            frames = np.arange(ramp_start, start, dtype=np.int64)
            progress = (frames - ramp_start) / float(start - ramp_start)
            blend[frames, column] = progress
        end = int(active[-1]) + 1
        release_stop = min(mask.shape[0], end + release_steps)
        frames = np.arange(end, release_stop, dtype=np.int64)
        progress = (frames - end + 1) / float(release_steps)
        blend[frames, column] = 1.0 - progress
    return blend.astype(np.float32)


def finger_participation(anchors: ContactAnchorSet) -> np.ndarray:
    """Assign every frame of a guided finger to its temporally closest contact column, with weight one."""
    frames = np.arange(anchors.mask.shape[0], dtype=np.int64)
    result = np.zeros(anchors.mask.shape, dtype=np.float64)
    for finger_id in np.unique(anchors.finger_ids):
        columns = ordered_finger_columns(anchors, int(finger_id))
        distance = np.stack([
            column_frame_distance(anchors.mask[:, column], frames)
            for column in columns
        ])
        result[frames, columns[np.argmin(distance, axis=0)]] = 1.0
    return result.astype(np.float32)


def column_frame_distance(
    mask_column: np.ndarray,
    frames: np.ndarray,
) -> np.ndarray:
    active = np.flatnonzero(mask_column)
    if active.size == 0:
        raise ValueError("Contact column has no active frames")
    before = np.maximum(int(active[0]) - frames, 0)
    after = np.maximum(frames - int(active[-1]), 0)
    return (before + after).astype(np.int64)


def scheduled_anchor_targets(
    anchors: ContactAnchorSet,
    *,
    blend: np.ndarray,
    mano_vertices: np.ndarray,
    obj_pose: np.ndarray,
) -> np.ndarray:
    """Contact target per column: the tracked MANO vertex blended toward the object anchor by alpha."""
    validate_approach_inputs(anchors, mano_vertices, obj_pose)
    targets = anchors.anchor_pos_obj.copy()
    for column in range(anchors.mask.shape[1]):
        active = np.flatnonzero(anchors.mask[:, column])
        scheduled = ~anchors.mask[:, column]
        anchor = targets[active[0], column]
        vertex_id = fixed_contact_vertex(
            anchors.mano_vertex_ids[active, column], column
        )
        if not np.any(scheduled):
            continue
        mano_target = mano_vertex_positions_obj(
            mano_vertices[scheduled, vertex_id], obj_pose[scheduled]
        )
        alpha = blend[scheduled, column, None]
        targets[scheduled, column] = (
            (1.0 - alpha) * mano_target + alpha * anchor
        )
    return targets


def ordered_finger_columns(anchors: ContactAnchorSet, finger_id: int) -> np.ndarray:
    columns = np.flatnonzero(anchors.finger_ids == finger_id)
    starts = np.asarray(
        [np.flatnonzero(anchors.mask[:, column])[0] for column in columns],
        dtype=np.int64,
    )
    return columns[np.argsort(starts)].astype(np.int32)


def validate_approach_inputs(
    anchors: ContactAnchorSet,
    mano_vertices: np.ndarray,
    obj_pose: np.ndarray,
) -> None:
    frame_count = anchors.mask.shape[0]
    if mano_vertices.ndim != 3 or mano_vertices.shape[0] != frame_count:
        raise ValueError("MANO approach vertices must have shape (frames, vertices, 3)")
    if obj_pose.shape != (frame_count, 7):
        raise ValueError("Object approach poses must have shape (frames, 7)")


def mano_vertex_positions_obj(
    positions_world: np.ndarray,
    obj_pose: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [world_to_obj(point[None], pose)[0] for point, pose in zip(positions_world, obj_pose)],
        dtype=np.float64,
    )


def fixed_contact_vertex(vertex_ids: np.ndarray, column: int) -> int:
    unique = np.unique(vertex_ids)
    if unique.size != 1 or unique[0] < 0:
        raise ValueError(f"Contact column {column} has no fixed MANO vertex")
    return int(unique[0])


def assign_anchor_queries(
    runtime: ContactRuntime,
    hand: MujocoHand,
    mano_mapping: ManoSurfaceMapping,
    *,
    cfg: SafeConfig,
    contact_query_region: str = TERMINAL_PAD,
) -> ContactRuntime:
    hand_query_ids = runtime.hand_query_ids.copy()
    cache: dict[tuple[int, int], int] = {}
    for anchor_id in range(runtime.anchors.mask.shape[1]):
        active = runtime.anchors.mask[:, anchor_id]
        vertex_id = fixed_contact_vertex(
            runtime.anchors.mano_vertex_ids[active, anchor_id], anchor_id
        )
        finger_id = int(runtime.anchors.finger_ids[anchor_id])
        key = (finger_id, vertex_id)
        if key not in cache:
            reference = mano_reference_descriptor(
                mano_mapping,
                vertex_id=vertex_id,
                finger_id=finger_id,
                merged_terminal=finger_id in hand.profile.merged_terminal_finger_ids,
            )
            cache[key] = map_surface_query(
                hand,
                reference,
                finger_id,
                cfg=cfg,
                contact_query_region=contact_query_region,
            )
        hand_query_ids[:, anchor_id] = cache[key]
    mapped = ContactRuntime(
        runtime.anchors,
        hand_query_ids,
        runtime.participation,
        runtime.contact_target_pos_obj,
        runtime.guidance_blend,
    )
    return mapped


def map_surface_query(
    hand: MujocoHand,
    reference: SurfaceDescriptorSet,
    finger_id: int,
    *,
    cfg: SafeConfig,
    contact_query_region: str = TERMINAL_PAD,
) -> int:
    query_ids = contact_query_ids(
        hand, int(finger_id), FULL_TERMINAL_SURFACE, cfg=cfg)
    candidates = subset_descriptors(hand.query_surface_descriptors, query_ids)
    cost = contact_surface_descriptor_cost(
        reference,
        candidates,
        minimum_palmar=cfg.pad_min_palmar,
        axial_weight=cfg.descriptor_axial_cost_weight,
        azimuth_weight=cfg.descriptor_azimuth_cost_weight,
    )
    if not np.any(np.isfinite(cost)):
        raise ValueError(
            f"No finite surface mapping candidate for finger_id={finger_id}"
        )
    selected = int(np.argmin(cost))
    return int(query_ids[selected])
