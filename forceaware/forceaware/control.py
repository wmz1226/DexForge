"""Control-knot operations shared by ForceAware rollout paths."""

from __future__ import annotations


import numpy as np


SMOOTHING_PROFILE = "forceaware_si_v1"
CONTROL_VELOCITY_BASE_POSITION_SCALE_M_S = 10.0
CONTROL_VELOCITY_BASE_ROTATION_SCALE_RAD_S = 3.0
CONTROL_VELOCITY_FINGER_SCALE_RAD_S = 10.0
FINGER_PALM_SMOOTHING_RATIO = 0.3
CONTROL_ACCELERATION_BASE_POSITION_SCALE_M_S2 = 500.0
CONTROL_ACCELERATION_BASE_ROTATION_SCALE_RAD_S2 = 500.0
CONTROL_ACCELERATION_FINGER_SCALE_RAD_S2 = 5000.0
HAND_STATE_BASE_POSITION_VELOCITY_SCALE_M_S = 1.0
HAND_STATE_BASE_ROTATION_VELOCITY_SCALE_RAD_S = 3.0
HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S = 4.0
HAND_STATE_BASE_POSITION_ACCELERATION_SCALE_M_S2 = 50.0
HAND_STATE_BASE_ROTATION_ACCELERATION_SCALE_RAD_S2 = 100.0
HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2 = 100.0
HAND_STATE_ACCELERATION_RATIO = 2.0


def action_knot_interpolation_map(
    horizon: int,
    substeps: int,
) -> tuple:
    """Map dense steps to adjacent absolute action knots."""
    if horizon < 1 or substeps < 1:
        raise ValueError("horizon and substeps must be positive")
    return tuple(
        (
            interval - 1,
            interval,
            float(substep + 1) / float(substeps),
        )
        for interval in range(horizon)
        for substep in range(substeps)
    )


def resample_warm_corrections(
    raw: np.ndarray,
    *,
    action_dim: int,
    horizon: int,
    executed_dense_steps: int,
    physics_steps_per_knot: int,
    warm_scale: float,
) -> np.ndarray:
    """Shift additive knot corrections by the actually executed duration."""
    expected = action_dim * (1 + horizon)
    values = np.asarray(raw)
    if values.shape != (expected,):
        raise ValueError(
            f"raw warm-start shape must be {(expected,)}, got {values.shape}"
        )
    if executed_dense_steps < 1 or executed_dense_steps > physics_steps_per_knot:
        raise ValueError(
            "executed_dense_steps must be within one control-knot interval: "
            f"{executed_dense_steps} not in [1, {physics_steps_per_knot}]"
        )
    controls = values[action_dim:].reshape(horizon, action_dim)
    shifted = _shifted_corrections(
        controls,
        executed_dense_steps=executed_dense_steps,
        physics_steps_per_knot=physics_steps_per_knot,
    )
    output = np.zeros_like(values)
    output[action_dim:] = warm_scale * shifted.reshape(-1)
    return output


def _shifted_corrections(
    controls: np.ndarray,
    *,
    executed_dense_steps: int,
    physics_steps_per_knot: int,
) -> np.ndarray:
    if executed_dense_steps == physics_steps_per_knot:
        return np.concatenate((controls[1:], controls[-1:]), axis=0)
    advance = float(executed_dense_steps) / float(physics_steps_per_knot)
    query = advance + np.arange(controls.shape[0], dtype=np.float64)
    left = np.minimum(np.floor(query).astype(np.int32), controls.shape[0] - 1)
    right = np.minimum(left + 1, controls.shape[0] - 1)
    fraction = (query - left.astype(np.float64))[:, None]
    shifted = (1.0 - fraction) * controls[left] + fraction * controls[right]
    return shifted.astype(controls.dtype)
