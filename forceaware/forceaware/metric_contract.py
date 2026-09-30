"""Pure NumPy aggregation for the shared ForceAware metric schema."""

from __future__ import annotations

import numpy as np


BASE_POSITION_DIM = 3
BASE_ACTION_DIM = 6
PERCENTILE_95 = 95.0
METERS_TO_MM = 1000.0


def marked_query_stats(
    distance_mm: np.ndarray,
    frame_ids: np.ndarray,
    *,
    contact_mask: np.ndarray,
    contact_weight: np.ndarray,
    age_ramp: np.ndarray,
) -> dict:
    requested, objective_weight = _contact_weights(
        contact_mask, contact_weight, age_ramp
    )
    terminal = requested[-1] if requested.shape[0] else requested
    first = int(frame_ids[np.argmax(requested.any(axis=1))]) if requested.any() else -1
    return {
        "requested_count": int(requested.sum()),
        "first_requested_frame": first,
        "anchor_error_mm": _sample_stats(distance_mm[requested]),
        "terminal_anchor_error_mm": _sample_stats(distance_mm[-1][terminal]),
        "objective_weighted": {
            "requested_weight": float(objective_weight.sum()),
            "anchor_error_mm": _weighted_stats(distance_mm, objective_weight),
        },
    }


def intended_contact_stats(
    distance_mm: np.ndarray,
    collision_active: np.ndarray,
    *,
    contact_mask: np.ndarray,
    contact_weight: np.ndarray,
    age_ramp: np.ndarray,
) -> dict:
    requested, objective_weight = _contact_weights(
        contact_mask, contact_weight, age_ramp
    )
    matched = requested & np.asarray(collision_active, bool)
    missing = requested & ~matched
    requested_count = int(requested.sum())
    matched_count = int(matched.sum())
    matched_weight = np.where(matched, objective_weight, 0.0)
    requested_weight = float(objective_weight.sum())
    return {
        "requested_count": requested_count,
        "matched_count": matched_count,
        "missing_count": int(missing.sum()),
        "coverage": _ratio(matched_count, requested_count),
        "matched_point_error_mm": _sample_stats(distance_mm[matched]),
        "objective_weighted": {
            "requested_weight": requested_weight,
            "matched_weight": float(matched_weight.sum()),
            "missing_weight": float(np.where(missing, objective_weight, 0.0).sum()),
            "coverage": _ratio(float(matched_weight.sum()), requested_weight),
            "matched_point_error_mm": _weighted_stats(distance_mm, matched_weight),
        },
    }


def control_stats(
    initial_control: np.ndarray,
    control_trajectory: np.ndarray,
    *,
    mpc_dt: float,
    action_substeps: int,
    knot_substeps: int | None = None,
) -> dict:
    initial, controls = _validate_controls(initial_control, control_trajectory)
    knot_steps = action_substeps if knot_substeps is None else knot_substeps
    if mpc_dt <= 0.0 or min(action_substeps, knot_steps) < 1:
        raise ValueError("mpc_dt and action/knot substeps must be positive")
    dense_delta = np.diff(np.vstack((initial[None], controls)), axis=0)
    dense_velocity = dense_delta / mpc_dt
    dense_acceleration = np.diff(dense_velocity, axis=0) / mpc_dt
    knot_controls = controls[knot_steps - 1 :: knot_steps]
    knot_delta = np.diff(np.vstack((initial[None], knot_controls)), axis=0)
    knot_dt = mpc_dt * knot_steps
    knot_velocity = knot_delta / knot_dt
    knot_acceleration = np.diff(knot_velocity, axis=0) / knot_dt
    boundary_rows = np.arange(action_substeps, controls.shape[0], action_substeps)
    boundary_delta = dense_delta[boundary_rows]
    boundary_velocity = dense_velocity[boundary_rows]
    boundary_acceleration = dense_acceleration[boundary_rows - 1]
    return {
        "dense": {
            "dt_s": float(mpc_dt),
            "delta": _delta_stats(dense_delta),
            "si_rate": _rate_stats(dense_delta, mpc_dt),
            "velocity": _motion_stats(dense_velocity, "m_per_s", "rad_per_s"),
            "acceleration": _motion_stats(dense_acceleration, "m_per_s2", "rad_per_s2"),
        },
        "knot": {
            "dt_s": float(knot_dt),
            "delta": _delta_stats(knot_delta),
            "velocity": _motion_stats(knot_velocity, "m_per_s", "rad_per_s"),
            "acceleration": _motion_stats(knot_acceleration, "m_per_s2", "rad_per_s2"),
            "bounded": False,
        },
        "window_boundary": {
            "delta": _delta_stats(boundary_delta),
            "velocity": _motion_stats(boundary_velocity, "m_per_s", "rad_per_s"),
            "acceleration": _motion_stats(
                boundary_acceleration, "m_per_s2", "rad_per_s2"
            ),
        },
    }


def hand_state_stats(
    qvel_trajectory: np.ndarray,
    hand_qvel_indices: np.ndarray,
    *,
    dt: float,
) -> dict:
    velocity = np.asarray(qvel_trajectory, np.float64)
    indices = np.asarray(hand_qvel_indices, np.int64)
    if dt <= 0.0:
        raise ValueError("hand-state metric dt must be positive")
    if velocity.ndim != 2 or velocity.shape[0] < 1:
        raise ValueError("qvel trajectory must be a non-empty rank-2 array")
    if indices.ndim != 1 or indices.size < BASE_ACTION_DIM:
        raise ValueError("hand qvel indices must contain the 6-DoF base")
    if (indices < 0).any() or (indices >= velocity.shape[1]).any():
        raise IndexError("hand qvel index exceeds the trajectory width")
    if not np.isfinite(velocity).all():
        raise ValueError("hand-state metrics require finite qvel values")
    hand_velocity = velocity[:, indices]
    acceleration = np.diff(hand_velocity, axis=0) / dt
    return {
        "dt_s": float(dt),
        "velocity": _motion_stats(hand_velocity[1:], "m_per_s", "rad_per_s"),
        "acceleration": _motion_stats(acceleration, "m_per_s2", "rad_per_s2"),
        "acceleration_method": "finite_difference_of_qvel",
    }


def _contact_weights(mask, weight, ramp) -> tuple[np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, np.float64) for value in (mask, weight, ramp))
    if arrays[0].ndim != 2 or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("contact mask, weight, and age ramp must share a rank-2 shape")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("contact metric inputs must be finite")
    objective_weight = arrays[0] * arrays[1] * arrays[2]
    if (objective_weight < 0.0).any():
        raise ValueError("objective contact weights must be non-negative")
    return arrays[0] > 0.0, objective_weight


def _sample_stats(values: np.ndarray) -> dict[str, float]:
    samples = np.asarray(values, np.float64).reshape(-1)
    if not samples.size:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(samples).all():
        raise ValueError("metric samples must be finite")
    return {
        "mean": float(samples.mean()),
        "p95": float(np.percentile(samples, PERCENTILE_95)),
        "max": float(samples.max()),
    }


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64).reshape(-1)
    weights = np.asarray(weights, np.float64).reshape(-1)
    positive = weights > 0.0
    if not positive.any():
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    samples, sample_weights = values[positive], weights[positive]
    if not np.isfinite(samples).all():
        raise ValueError("weighted metric samples must be finite")
    order = np.argsort(samples)
    cumulative = np.cumsum(sample_weights[order])
    index = min(
        int(np.searchsorted(cumulative, PERCENTILE_95 / 100.0 * cumulative[-1])),
        order.size - 1,
    )
    return {
        "mean": float(np.average(samples, weights=sample_weights)),
        "p95": float(samples[order[index]]),
        "max": float(samples.max()),
    }


def _validate_controls(initial, controls) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial, np.float64)
    controls = np.asarray(controls, np.float64)
    if initial.ndim != 1 or initial.size < BASE_ACTION_DIM:
        raise ValueError("initial control must contain a 6-DoF base")
    if controls.ndim != 2 or controls.shape[1] != initial.size:
        raise ValueError("control trajectory width must match initial control")
    if not np.isfinite(initial).all() or not np.isfinite(controls).all():
        raise ValueError("control metrics require finite inputs")
    return initial, controls


def _delta_stats(delta: np.ndarray) -> dict[str, float]:
    return {
        "base_position_component_max_mm": _max_abs(delta[:, :BASE_POSITION_DIM])
        * METERS_TO_MM,
        "base_position_norm_max_mm": _max_norm(delta[:, :BASE_POSITION_DIM])
        * METERS_TO_MM,
        "base_rotation_component_max_rad": _max_abs(
            delta[:, BASE_POSITION_DIM:BASE_ACTION_DIM]
        ),
        "finger_component_max_rad": _max_abs(delta[:, BASE_ACTION_DIM:]),
    }


def _rate_stats(delta: np.ndarray, dt: float) -> dict[str, float]:
    rate = delta / dt
    return {
        "base_position_component_max_m_per_s": _max_abs(rate[:, :BASE_POSITION_DIM]),
        "base_position_norm_max_m_per_s": _max_norm(rate[:, :BASE_POSITION_DIM]),
        "base_rotation_component_max_rad_per_s": _max_abs(
            rate[:, BASE_POSITION_DIM:BASE_ACTION_DIM]
        ),
        "finger_component_max_rad_per_s": _max_abs(rate[:, BASE_ACTION_DIM:]),
    }


def _motion_stats(
    values: np.ndarray,
    position_unit: str,
    angular_unit: str,
) -> dict[str, dict[str, float]]:
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] < BASE_ACTION_DIM:
        raise ValueError("motion samples must contain the 6-DoF base")
    return {
        f"base_position_norm_{position_unit}": _distribution_stats(
            np.linalg.norm(values[:, :BASE_POSITION_DIM], axis=1)
        ),
        f"base_rotation_norm_{angular_unit}": _distribution_stats(
            np.linalg.norm(values[:, BASE_POSITION_DIM:BASE_ACTION_DIM], axis=1)
        ),
        f"finger_component_{angular_unit}": _distribution_stats(
            values[:, BASE_ACTION_DIM:]
        ),
    }


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    magnitudes = np.abs(np.asarray(values, np.float64)).reshape(-1)
    if not magnitudes.size:
        return {"mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(magnitudes).all():
        raise ValueError("motion metric samples must be finite")
    return {
        "mean": float(magnitudes.mean()),
        "rms": float(np.sqrt(np.mean(magnitudes * magnitudes))),
        "p95": float(np.percentile(magnitudes, PERCENTILE_95)),
        "max": float(magnitudes.max()),
    }


def _max_abs(values: np.ndarray) -> float:
    return float(np.abs(values).max()) if values.size else 0.0


def _max_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values, axis=1).max()) if values.shape[0] else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0
