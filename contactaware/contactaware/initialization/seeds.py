"""Starting postures for the seed solve."""

from dataclasses import dataclass

import numpy as np

OPEN_FRACTIONS = (0.25, 0.5)
MM_PER_M = 1000.0


@dataclass(frozen=True)
class SeedOps:
    align_palm: object
    query_world: object
    object_rotation: object
    alignment_step: object


def starting_postures(hand, seed):
    """Opened interpolations of the MANO-fit posture, plus the hand's default posture."""
    opened = np.clip(np.zeros(hand.qpos_dim), hand.lower, hand.upper)
    candidates = {f"mano_open_{fraction:.2f}": (1.0 - fraction) * seed + fraction * opened
                  for fraction in OPEN_FRACTIONS}
    candidates["profile_default"] = hand.profile.default_qpos.copy()
    return candidates


def exterior_start(problem, qpos, *, ops):
    """Translate an opened posture outward until all object queries are exterior."""
    hand, cfg, obj = problem.hand, problem.cfg, problem.obj
    points = hand.query_positions(qpos)
    phi = ops.query_world(points, problem.obj_pose, obj)[0]
    if np.min(phi) >= cfg.hand_object_safe_distance:
        return qpos.copy(), 0.0
    rotation = ops.object_rotation(problem.obj_pose)
    object_center, radius = obj.query_fn.bounding_sphere()
    center = object_center @ rotation.T + problem.obj_pose[4:7]
    palm, _, pos_jac, _ = hand.palm_pose_jacobian(qpos)
    direction = palm - center
    direction = direction / np.linalg.norm(direction)
    low = 0.0
    high = float(radius + cfg.hand_object_safe_distance - np.min((points - center) @ direction))
    outer_phi = ops.query_world(points + high * direction, problem.obj_pose, obj)[0]
    if np.min(outer_phi) < cfg.hand_object_safe_distance:
        raise ValueError("Object bounding sphere did not produce an exterior starting pose")
    while high - low > cfg.geometry_feasibility_tolerance_m:
        distance = (low + high) / 2.0
        phi = ops.query_world(points + distance * direction, problem.obj_pose, obj)[0]
        if np.min(phi) >= cfg.hand_object_safe_distance:
            high = distance
        else:
            low = distance
    result = qpos.copy()
    result[:3] += np.linalg.solve(pos_jac[:, :3], high * direction)
    if np.any(result < hand.lower) or np.any(result > hand.upper):
        raise ValueError("Exterior starting pose exceeds base workspace limits")
    return result, high * MM_PER_M


def prepare_start(name, posture, problem, *, ops):
    aligned = ops.align_palm(problem.hand, posture, problem.joints, cfg=problem.cfg)
    return exterior_start(problem, aligned, ops=ops)
