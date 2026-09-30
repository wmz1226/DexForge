"""Physical feasibility in metres and radians, independent of QP accuracy."""


import numpy as np


def key_joint_violation(problem, vector):
    # Finite key bounds are angular; base poses and anchor coordinates are free.
    lower, upper = problem.bounds()
    return float(np.maximum(np.r_[lower-vector, vector-upper], 0.).max(initial=0.))


def key_violation_ratio(problem, vector):
    geometry = problem.constraint_evaluation(vector).constraints
    geometric_ratio = np.maximum(-geometry, 0.).max() / problem.feasibility_tolerance
    joint_ratio = key_joint_violation(problem, vector) / problem.cfg.joint_limit_tolerance_rad
    return max(float(geometric_ratio), joint_ratio)
