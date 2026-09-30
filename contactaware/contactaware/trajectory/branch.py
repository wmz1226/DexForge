"""Continuous base-rotation coordinates: same palm pose, nearest equivalent hinge angles."""

from itertools import product

import numpy as np

ROTATION = slice(3, 6)
POSE_TOLERANCE = 1e-9


def equivalent_rotations(hand, state):
    """All hinge triples inside the joint limits that give the same three-axis rotation."""
    a, b, c = state[ROTATION]
    lower, upper = hand.lower[ROTATION], hand.upper[ROTATION]
    candidates = []
    for base in ((a, b, c), (a + np.pi, np.pi - b, c + np.pi)):
        first = np.ceil((lower - base) / (2 * np.pi)).astype(int)
        last = np.floor((upper - base) / (2 * np.pi)).astype(int)
        for turns in product(*(range(start, end + 1) for start, end in zip(first, last))):
            value = np.asarray(base) + 2 * np.pi * np.asarray(turns)
            if np.all(value >= lower) and np.all(value <= upper):
                candidates.append(value)
    return np.asarray(candidates)


def unwrap_base_rotation(hand, states, *, fixed_frames=()):
    """Re-express every state on the branch nearest its predecessor; FK is verified unchanged."""
    result = np.array(states, dtype=np.float64, copy=True)
    fixed = frozenset(map(int, fixed_frames))
    changed = []
    for t in range(len(result)):
        if t in fixed:
            continue
        if t == 0 and np.all(result[t, ROTATION] >= hand.lower[ROTATION]) and np.all(
            result[t, ROTATION] <= hand.upper[ROTATION]
        ):
            continue
        candidates = equivalent_rotations(hand, result[t])
        if not len(candidates):
            raise ValueError(f"No bounded equivalent base rotation at frame {t}")
        reference = result[max(t - 1, 0), ROTATION]
        best = candidates[np.argmin(np.linalg.norm(candidates - reference, axis=1))]
        if np.allclose(best, result[t, ROTATION]):
            continue
        before = hand.palm_pose_jacobian(result[t])[:2]
        trial = result[t].copy()
        trial[ROTATION] = best
        after = hand.palm_pose_jacobian(trial)[:2]
        if (
            max(np.abs(before[0] - after[0]).max(), np.abs(before[1] - after[1]).max())
            > POSE_TOLERANCE
        ):
            raise RuntimeError(f"Equivalent base rotation changed the palm pose at frame {t}")
        result[t] = trial
        changed.append(t)
    return result, changed
