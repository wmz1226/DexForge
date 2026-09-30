"""MANO finger chain directions and their Jacobians."""

from __future__ import annotations

import numpy as np

from contactaware.settings import FINGER_LABELS
from contactaware.types import ContactRuntime, SafeConfig

FINGER_ID_BY_LABEL = {label: finger_id for finger_id, label in FINGER_LABELS.items()}


def frame_mano_shape_weight(
    finger: str,
    *,
    cfg: SafeConfig,
    runtime: ContactRuntime | None,
    frame_id: int,
) -> float:
    base_weight = float(cfg.mano_shape_weight)
    if runtime is None or cfg.contact_anchor_weight <= 0.0:
        return base_weight
    finger_id = FINGER_ID_BY_LABEL[finger]
    columns = np.flatnonzero(runtime.anchors.finger_ids == finger_id)
    if columns.size == 0:
        return base_weight
    progress = float(
        runtime.participation[frame_id, columns]
        @ runtime.guidance_blend[frame_id, columns]
    )
    return base_weight * (1.0 - progress)


def mano_chain_dirs(joints: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    points = joints[np.asarray(ids, dtype=np.int64)]
    vec = points[1:] - points[:-1]
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


def chain_direction_terms(
    points: np.ndarray,
    jacobians: np.ndarray,
    target_dirs: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    terms = []
    for idx, target in enumerate(target_dirs):
        vec = points[idx + 1] - points[idx]
        jac = jacobians[idx + 1] - jacobians[idx]
        direction, dir_jac = direction_jacobian(vec, jac)
        terms.append((direction - target, dir_jac))
    return terms


def direction_jacobian(
    vec: np.ndarray,
    jac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError("Cannot build MANO shape loss from zero-length hand segment")
    direction = vec / length
    projector = np.eye(3) - np.outer(direction, direction)
    return direction, (projector @ jac) / length


