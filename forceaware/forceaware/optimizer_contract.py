"""Backend-neutral ForceAware optimizer diagnostics."""

from typing import NamedTuple

import numpy as np


CANDIDATE_FAILURE_NONE = 0
CANDIDATE_FAILURE_LOSS = 1
CANDIDATE_FAILURE_GRADIENT = 2
CANDIDATE_FAILURE_UPDATE = 3
CANDIDATE_FAILURE_HARD_ROLLOUT = 4
CANDIDATE_FAILURE_SIMULATOR = 5
CANDIDATE_FAILURE_NO_BEST = 6
NO_FAILURE_ITERATION = -1

CANDIDATE_FAILURE_LABELS = (
    "none",
    "loss_nonfinite",
    "gradient_nonfinite",
    "optimizer_update_nonfinite",
    "hard_rollout_nonfinite",
    "simulator_failure",
    "no_finite_best",
)


class CandidateEvaluation(NamedTuple):
    scores: np.ndarray
    drifts: np.ndarray
    selection_scores: np.ndarray
    has_finite_best: np.ndarray
    valid: np.ndarray
    training_failure_iteration: np.ndarray
    training_failure_code: np.ndarray
    rejection_code: np.ndarray


def finite_best_state(
    best_loss,
    best_raw,
    best_grad_norm,
    *,
    available,
    array_module,
):
    """Return candidates whose complete best-state snapshot is finite."""
    raw_axes = tuple(range(1, best_raw.ndim))
    raw_finite = array_module.all(array_module.isfinite(best_raw), axis=raw_axes)
    return (
        available
        & array_module.isfinite(best_loss)
        & raw_finite
        & array_module.isfinite(best_grad_norm)
    )


def finalize_candidate_evaluation(
    scores,
    drifts,
    *,
    has_finite_best,
    hard_simulator_valid,
    training_failure_iteration,
    training_failure_code,
) -> CandidateEvaluation:
    """Apply the shared post-training candidate-selection contract."""
    scores = _candidate_vector("scores", scores, np.float32)
    count = scores.size
    drifts = _candidate_vector("drifts", drifts, np.float32, count=count)
    has_best = _candidate_vector("has_finite_best", has_finite_best, bool, count=count)
    simulator_valid = _candidate_vector(
        "hard_simulator_valid", hard_simulator_valid, bool, count=count
    )
    failure_iteration = _candidate_vector(
        "training_failure_iteration", training_failure_iteration, np.int32, count=count
    )
    failure_code = _candidate_vector(
        "training_failure_code", training_failure_code, np.int32, count=count
    )
    hard_finite = np.isfinite(scores) & np.isfinite(drifts)
    valid = has_best & simulator_valid & hard_finite
    rejection = _candidate_rejection_codes(has_best, simulator_valid, hard_finite)
    finite_scores = np.where(valid, scores, np.inf).astype(np.float32)
    return CandidateEvaluation(
        finite_scores,
        drifts.astype(np.float32, copy=True),
        finite_scores.copy(),
        has_best,
        valid,
        failure_iteration,
        failure_code,
        rejection,
    )


def _candidate_vector(name, values, dtype, *, count=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if count is not None and array.size != count:
        raise ValueError(f"{name} has {array.size} candidates; expected {count}")
    return array


def _candidate_rejection_codes(has_best, simulator_valid, hard_finite):
    rejection = np.full(has_best.shape, CANDIDATE_FAILURE_NONE, np.int32)
    rejection[~has_best] = CANDIDATE_FAILURE_NO_BEST
    rejection[has_best & ~simulator_valid] = CANDIDATE_FAILURE_SIMULATOR
    rejection[has_best & simulator_valid & ~hard_finite] = (
        CANDIDATE_FAILURE_HARD_ROLLOUT
    )
    return rejection


def multistart_raws(
    warm,
    *,
    candidate_count: int,
    initial_dimension: int,
    initial_std: float,
    control_std: float,
    initial_raw_limit: float,
    seed: int,
) -> np.ndarray:
    """Build identical multistart candidates for both simulator backends."""
    warm_array = np.asarray(warm, np.float32)
    invalid = np.argwhere(~np.isfinite(warm_array))
    if invalid.size:
        first = tuple(int(index) for index in invalid[0])
        raise FloatingPointError(
            f"warm parameters must be finite; first invalid value at {first}"
        )
    if candidate_count == 1:
        return warm_array[None]
    exploration_center = warm_array.copy()
    exploration_center[:initial_dimension] = np.clip(
        exploration_center[:initial_dimension],
        -initial_raw_limit,
        initial_raw_limit,
    )
    scale = np.full(warm_array.size, control_std, np.float32)
    scale[:initial_dimension] = initial_std
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(candidate_count - 1, warm_array.size)).astype(np.float32)
    exploration = exploration_center[None] + noise * scale[None]
    exploration[:, :initial_dimension] = np.clip(
        exploration[:, :initial_dimension],
        -initial_raw_limit,
        initial_raw_limit,
    )
    return np.concatenate((warm_array[None], exploration), axis=0)
