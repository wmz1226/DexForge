"""Stable-contact targets, normals and query selection per frame."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from contactaware.contact.object_model import obj_to_world, object_rotation
from contactaware.contact.surface import SURFACE_EPS
from contactaware.types import ContactRuntime, SafeConfig
from contactaware.solver.hand import MujocoHand


HandObjectSafeDistancePolicy = Callable[
    [MujocoHand, np.ndarray, Optional[ContactRuntime], int, SafeConfig],
    np.ndarray,
]


def include_anchor_query_ids(
    selected: np.ndarray,
    anchor_ids: np.ndarray,
) -> np.ndarray:
    if anchor_ids.size == 0:
        return selected.astype(np.int32)
    return np.unique(
        np.concatenate([selected.astype(np.int32), anchor_ids.astype(np.int32)])
    )


def active_query_ids(
    hand: MujocoHand,
    phi: np.ndarray,
    cfg: SafeConfig,
) -> np.ndarray:
    if cfg.hand_object_topk_per_link <= 0:
        return np.flatnonzero(phi < cfg.hand_object_activation).astype(np.int32)
    topk = int(cfg.hand_object_topk_per_link)
    chosen = []
    for group_ids in hand.query_ids_by_group:
        active = group_ids[phi[group_ids] < cfg.hand_object_activation]
        if active.size == 0:
            continue
        if active.size > topk:
            order = np.argpartition(phi[active], topk - 1)[:topk]
            active = active[order]
        chosen.append(active.astype(np.int32, copy=False))
    if not chosen:
        return np.zeros(0, dtype=np.int32)
    return np.sort(np.concatenate(chosen)).astype(np.int32)


def anchor_outward_normals_world(
    runtime: ContactRuntime,
    obj_pose: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Extend each fixed anchor's normal across its approach/release interval."""
    masks = runtime.anchors.mask[:, columns]
    if not np.all(np.any(masks, axis=0)):
        raise ValueError("Cannot orient a contact column without a stable anchor")
    first_frames = np.argmax(masks, axis=0)
    normals = np.asarray(
        runtime.anchors.anchor_normal_obj[first_frames, columns], dtype=np.float64
    )
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= SURFACE_EPS):
        raise ValueError("Object anchor normals must be finite and nonzero")
    return (normals / lengths[:, None]) @ object_rotation(obj_pose).T


def anchor_normal_linearization(
    hand: MujocoHand,
    qpos: np.ndarray,
    obj_pose: np.ndarray | None,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
    frame_id: int,
):
    """Contact-normal residual scaled to metres by the terminal surface radius."""
    if runtime is None or obj_pose is None or cfg.contact_anchor_weight <= 0.0:
        return None
    scale = float(cfg.contact_normal_weight_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("Contact normal weight scale must be finite and nonnegative")
    columns, query_ids, _, participation = frame_anchor_mapping(runtime, frame_id)
    strength = scale * participation * runtime.guidance_blend[frame_id, columns]
    selected = strength > 0.0
    if not np.any(selected):
        return None
    columns, query_ids = columns[selected], query_ids[selected]
    outward = anchor_outward_normals_world(runtime, obj_pose, columns)
    normals, jacobians = hand.query_normals_jacobian(qpos, query_ids)
    radii = hand.query_surface_radii[query_ids]
    if not np.isfinite(radii).all():
        raise ValueError("Normal guidance requires terminal-surface queries")
    residual, rows = pad_normal_residuals(normals, jacobians, outward, radii)
    return residual, rows, strength[selected]


def pad_normal_residuals(normals, jacobians, object_normals, radii, *, object_jacobians=0.0):
    """The same pad-facing residual with variable or fixed object-anchor normals."""
    residual = radii[:, None] * (normals + object_normals)
    rows = radii[:, None, None] * (jacobians + object_jacobians)
    return residual, rows


def anchor_constraint_linearization(
    hand: MujocoHand,
    qpos: np.ndarray,
    obj_pose: np.ndarray | None,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
    frame_id: int,
):
    if runtime is None or obj_pose is None or cfg.contact_anchor_weight <= 0.0:
        return None
    active, query_ids, _, participation = frame_anchor_mapping(runtime, frame_id)
    if active.size == 0:
        return None
    positions, jacobians = hand.query_positions_jacobian(qpos, query_ids)
    targets = obj_to_world(
        runtime.contact_target_pos_obj[frame_id, active], obj_pose
    )
    return positions, jacobians, targets, participation


def frame_anchor_ids(runtime: ContactRuntime, frame_id: int) -> np.ndarray:
    return np.flatnonzero(runtime.participation[frame_id] > 0.0).astype(np.int32)


def frame_anchor_mapping(
    runtime: ContactRuntime,
    frame_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = frame_anchor_ids(runtime, frame_id)
    if active.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty.copy(), empty.copy(), empty.astype(np.float64)
    query_ids = runtime.hand_query_ids[frame_id, active].astype(np.int32)
    invalid = active[query_ids < 0]
    if invalid.size:
        raise RuntimeError(
            f"Active contact anchors have no mapped robot query at frame {frame_id}: "
            f"{invalid.tolist()}"
        )
    finger_ids = runtime.anchors.finger_ids[active].astype(np.int32)
    participation = runtime.participation[frame_id, active].astype(np.float64)
    return active, query_ids, finger_ids, participation


def frame_anchor_query_ids(cfg: SafeConfig, runtime, frame_id: int) -> np.ndarray:
    if runtime is None or cfg.contact_anchor_weight <= 0.0:
        return np.zeros(0, dtype=np.int32)
    _, query_ids, _, _ = frame_anchor_mapping(runtime, frame_id)
    return query_ids
