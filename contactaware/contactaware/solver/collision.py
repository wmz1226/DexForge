"""Self-collision and joint-limit constraints for hand retargeting."""

from __future__ import annotations

import numpy as np

from contactaware.types import SafeConfig
from contactaware.solver.hand import MujocoHand


NO_WORSENING_STEP = 0.0


def capsule_collision_constraints(hand: MujocoHand, qpos: np.ndarray, cfg: SafeConfig,
                                  *, allow_initial_penetration: bool):
    starts, ends = hand.capsule_segments(qpos)
    point_left, point_right, normals, clearances, groups = near_capsule_pair_states(
        hand, starts, ends, float(cfg.activation_distance))
    selected = np.flatnonzero(clearances < float(cfg.activation_distance))
    active = groups[selected]
    rows, rhs = capsule_cbf_rows(
        hand, active, point_left[selected],
        point_right=point_right[selected],
        normals=normals[selected],
        clearances=clearances[selected],
        cfg=cfg,
        allow_initial_penetration=allow_initial_penetration,
    )
    return rows, rhs


def pair_clearance_bounds(hand, starts, ends):
    """Lower bound of every component pair's clearance from the capsules' bounding spheres."""
    left, right = hand.capsule_component_left, hand.capsule_component_right
    middle, half = .5 * (starts + ends), .5 * np.linalg.norm(ends - starts, axis=1)
    reach = half + hand.capsule_radii
    return np.linalg.norm(middle[left] - middle[right], axis=1) - reach[left] - reach[right]


def near_capsule_pair_states(hand, starts, ends, threshold):
    """capsule_pair_states restricted to link groups whose clearance may be below `threshold`.

    A group's closest pair always has a bound below the threshold when its clearance is, so the
    returned rows of those groups equal capsule_pair_states' rows exactly.
    """
    groups = hand.capsule_pair_groups
    candidates = pair_clearance_bounds(hand, starts, ends) < threshold
    kept = np.flatnonzero(candidates[groups].any(axis=1)) if len(groups) else np.empty(0, dtype=np.int64)
    if not len(kept):
        empty = np.empty((0, 3))
        return empty, empty, empty, np.empty(0), kept
    left, right = hand.capsule_component_left, hand.capsule_component_right
    members = groups[kept]
    pairs = np.unique(members)
    point_left, point_right = closest_segment_points_batch(
        starts[left[pairs]], ends[left[pairs]], starts[right[pairs]], b1=ends[right[pairs]])
    position = np.searchsorted(pairs, members)
    clearances = (np.linalg.norm(point_left - point_right, axis=1)
                  - hand.capsule_radii[left[pairs]] - hand.capsule_radii[right[pairs]])[position]
    best = position[np.arange(len(kept)), np.argmin(clearances, axis=1)]
    selected = pairs[best]
    delta = point_left[best] - point_right[best]
    normals = capsule_normals(delta, starts[left[selected]], ends[left[selected]],
                              b0=starts[right[selected]], b1=ends[right[selected]])
    return point_left[best], point_right[best], normals, clearances.min(axis=1), kept


def capsule_pair_states(hand, starts, ends):
    left, right = hand.capsule_component_left, hand.capsule_component_right
    point_left, point_right = closest_segment_points_batch(
        starts[left], ends[left], starts[right], b1=ends[right]
    )
    delta = point_left - point_right
    dist = np.linalg.norm(delta, axis=1)
    clearances = dist - hand.capsule_radii[left] - hand.capsule_radii[right]
    groups = hand.capsule_pair_groups
    selected = groups[np.arange(len(groups)), np.argmin(clearances[groups], axis=1)] if len(groups) else np.empty(0, dtype=np.int64)
    normals = capsule_normals(delta[selected], starts[left[selected]], ends[left[selected]],
                              b0=starts[right[selected]], b1=ends[right[selected]])
    return point_left[selected], point_right[selected], normals, clearances[selected]


def capsule_cbf_rows(hand, active, point_left, *, point_right,
                     normals, clearances, cfg: SafeConfig,
                     allow_initial_penetration: bool):
    if active.size == 0:
        return as_rows([], hand.qpos_dim), np.zeros(0, dtype=np.float64)
    left = hand.capsule_pair_left[active]
    right = hand.capsule_pair_right[active]
    left_ids = hand.capsule_body_ids_np[left]
    right_ids = hand.capsule_body_ids_np[right]
    jac = hand.point_jacobians(left_ids, point_left) - hand.point_jacobians(
        right_ids, point_right
    )
    rows = np.einsum("ni,nij->nj", normals, jac)
    rhs = cfg.gamma * (float(cfg.safe_distance) - clearances)
    rhs = initial_self_collision_rhs(
        rhs, clearances, allow_initial_penetration,
        safe_distance=float(cfg.safe_distance),
    )
    return rows.astype(np.float64), rhs.astype(np.float64)


def initial_self_collision_rhs(rhs: np.ndarray, clearances: np.ndarray,
                               enabled: bool, *, safe_distance: float) -> np.ndarray:
    if not enabled:
        return rhs
    unsafe = clearances < float(safe_distance)
    return np.where(unsafe, NO_WORSENING_STEP, rhs)


def closest_segment_points_batch(a0, a1, b0, *, b1) -> tuple[np.ndarray, np.ndarray]:
    da, db, r = a1 - a0, b1 - b0, a0 - b0
    aa = np.einsum("ni,ni->n", da, da)
    bb = np.einsum("ni,ni->n", db, db)
    ab = np.einsum("ni,ni->n", da, db)
    ar = np.einsum("ni,ni->n", da, r)
    br = np.einsum("ni,ni->n", db, r)
    denom = aa * bb - ab * ab
    raw_s = np.divide(
        ab * br - bb * ar,
        denom,
        out=np.zeros_like(denom),
        where=denom >= 1e-12,
    )
    s = np.clip(raw_s, 0.0, 1.0)
    t = segment_parameters((a0 + s[:, None] * da) - b0, db, bb)
    s = segment_parameters((b0 + t[:, None] * db) - a0, da, aa)
    return a0 + s[:, None] * da, b0 + t[:, None] * db


def segment_parameters(offset: np.ndarray, direction: np.ndarray,
                       length_sq: np.ndarray) -> np.ndarray:
    dot = np.einsum("ni,ni->n", direction, offset)
    raw = np.divide(
        dot,
        length_sq,
        out=np.zeros_like(length_sq),
        where=length_sq >= 1e-12,
    )
    return np.clip(raw, 0.0, 1.0)


def capsule_normals(delta, a0, a1, *, b0, b1) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=1)
    normals = np.zeros_like(delta)
    regular = norm >= 1e-9
    normals[regular] = delta[regular] / norm[regular, None]
    if np.all(regular):
        return normals
    midpoint_delta = 0.5 * (a0 + a1 - b0 - b1)
    midpoint_norm = np.linalg.norm(midpoint_delta, axis=1)
    fallback = ~regular
    if np.any(midpoint_norm[fallback] < 1e-9):
        raise RuntimeError("Degenerate capsule witness points")
    normals[fallback] = midpoint_delta[fallback] / midpoint_norm[fallback, None]
    return normals


def as_rows(rows: list[np.ndarray], qpos_dim: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, qpos_dim), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
