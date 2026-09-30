"""Semantic mapping between contact guidance and GS collision outputs."""

from __future__ import annotations

import numpy as np


def map_guidance_to_gs_outputs(
    output_body_ids: np.ndarray,
    query_body_ids: np.ndarray,
    contact_mask: np.ndarray,
) -> np.ndarray:
    """Return the GS output index for every guidance entry."""
    output_bodies = np.asarray(output_body_ids, dtype=np.int32)
    query_bodies = np.asarray(query_body_ids, dtype=np.int32)
    active = np.asarray(contact_mask) > 0.0
    if output_bodies.ndim != 1 or output_bodies.size == 0:
        raise ValueError("GS output body IDs must be a non-empty vector")
    if query_bodies.shape != active.shape:
        raise ValueError(
            "Guidance query body IDs and contact mask must have equal shape: "
            f"{query_bodies.shape} vs {active.shape}"
        )

    _validate_guidance_columns(query_bodies, active)
    active_bodies = query_bodies[active]
    if np.any(output_bodies < 0) or np.any(active_bodies < 0):
        raise ValueError("GS output and active guidance body IDs must be nonnegative")

    mapped = np.zeros(query_bodies.shape, dtype=np.int32)
    for body_id in np.unique(active_bodies):
        outputs = np.flatnonzero(output_bodies == body_id)
        if outputs.size == 0:
            raise ValueError(
                f"Active guidance body {int(body_id)} is missing from GS collision outputs"
            )
        if outputs.size > 1:
            raise ValueError(
                f"Active guidance body {int(body_id)} has ambiguous GS outputs: "
                f"{outputs.tolist()}"
            )
        mapped[active & (query_bodies == body_id)] = int(outputs[0])
    return mapped


def _validate_guidance_columns(query_body_ids: np.ndarray, active: np.ndarray) -> None:
    for column in range(query_body_ids.shape[1]):
        bodies = np.unique(query_body_ids[active[:, column], column])
        if bodies.size > 1:
            raise ValueError(
                f"Guidance column {column} crosses query bodies: {bodies.tolist()}"
            )
