"""Initialize free-motion frames by MANO fitting and interpolated key residuals."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from contactaware.initialization.fixed_key_trajectory import (
    KeyPalmObjective,
)
from contactaware.initialization.mano_fit import fit_mano_posture
from contactaware.solver.pose import fit_palm_pose
from contactaware.trajectory.branch import ROTATION, unwrap_base_rotation
from contactaware.solver.retarget import (
    interpolation_ratio,
    palm_aligned_qpos,
)


def warp_free_motion(case):
    """Return initial states with warped free-motion frames, and those frames."""
    hand, states, base = case.hand, case.initial.copy(), case.hand.base_qpos_dim
    keys = np.sort(np.asarray(case.frames))
    free = np.ones(len(states), dtype=bool)
    free[keys] = False
    free &= ~np.asarray(case.inputs.anchors.mask).reshape(len(states), -1).any(axis=1)
    frames = np.flatnonzero(free)
    if not len(frames):
        return states, []
    ratio = interpolation_ratio(case.args.video_fps, case.args.internal_fps)
    palm = KeyPalmObjective.from_keys(case.inputs, keys, states[keys], ratio)
    reference, posture = {}, None
    for frame in np.union1d(frames, keys):
        problem = case.problems[frame]
        posture, _ = fit_mano_posture(
            hand, problem.joints, cfg=problem.cfg, align_palm=palm_aligned_qpos, start=posture
        )
        reference[frame] = posture[base:]
    delta = states[keys, base:] - np.array([reference[key] for key in keys])
    offset = PchipInterpolator(keys, delta) if len(keys) > 1 else (lambda frame: delta[0])
    for frame in frames:
        problem = case.problems[frame]
        fingers = reference[frame] + offset(np.clip(frame, keys[0], keys[-1]))
        fingers = np.clip(fingers, hand.lower[base:], hand.upper[base:])
        guess = np.r_[states[max(frame - 1, 0), :base], fingers]
        target = palm.target_at(hand, problem.joints, problem.frame_id)
        states[frame] = fit_palm_pose(hand, guess, *target)
    outside = np.any((states[:, ROTATION] < hand.lower[ROTATION])
                     | (states[:, ROTATION] > hand.upper[ROTATION]), axis=1)
    states, _ = unwrap_base_rotation(hand, states, fixed_frames=np.flatnonzero(~(free & outside)))
    return states, frames.tolist()
