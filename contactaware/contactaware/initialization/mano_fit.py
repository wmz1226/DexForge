"""Fit the robot hand to the MANO hand at one frame: palm alignment plus per-finger chain directions."""

from scipy.optimize import least_squares
import numpy as np

from contactaware.settings import MANO_CHAIN_JOINTS
from contactaware.solver.mano_regularization import chain_direction_terms, mano_chain_dirs

FIT_MAX_EVALUATIONS = 64


def chain_residual(hand, qpos, joints):
    """Robot-minus-MANO chain directions and their Jacobian at one posture."""
    hand.forward(np.asarray(qpos, dtype=np.float64))
    residuals, jacobians = [], []
    for finger, specs in hand.profile.chain_points.items():
        points, jac = hand.current_chain_points_jacobian(specs)
        targets = mano_chain_dirs(joints, MANO_CHAIN_JOINTS[finger])
        for residual, jacobian in chain_direction_terms(points, jac, targets):
            residuals.append(residual)
            jacobians.append(jacobian)
    if not residuals:
        raise ValueError("Hand profile defines no MANO chain points to fit")
    return np.concatenate(residuals), np.vstack(jacobians)


def fit_mano_posture(hand, joints, *, cfg, align_palm, start=None):
    """Place the palm on the MANO palm and fit the finger joints to MANO chain directions."""
    initial = hand.profile.default_qpos.copy() if start is None else np.asarray(start, np.float64)
    aligned = align_palm(hand, np.clip(initial, hand.lower, hand.upper), joints, cfg=cfg)
    articulated = slice(hand.base_qpos_dim, hand.qpos_dim)
    base = aligned.copy()

    def evaluate(values):
        qpos = base.copy()
        qpos[articulated] = values
        return chain_residual(hand, qpos, joints)

    result = least_squares(
        lambda values: evaluate(values)[0], aligned[articulated],
        jac=lambda values: evaluate(values)[1][:, articulated],
        bounds=(hand.lower[articulated], hand.upper[articulated]), x_scale="jac",
        ftol=cfg.qp_validation_tolerance, xtol=cfg.qp_validation_tolerance,
        gtol=cfg.qp_validation_tolerance, max_nfev=FIT_MAX_EVALUATIONS)
    fitted = base.copy()
    fitted[articulated] = np.clip(result.x, hand.lower[articulated], hand.upper[articulated])
    report = {"chain_direction_rms": float(np.sqrt(np.mean(result.fun ** 2))),
              "chain_direction_max": float(np.max(np.abs(result.fun))),
              "evaluations": int(result.nfev), "status": int(result.status)}
    return fitted, report
