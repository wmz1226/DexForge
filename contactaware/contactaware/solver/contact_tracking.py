"""Direct residuals for calibrated palm, finger shape and shared contact tracking."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from contactaware.solver.contact import anchor_constraint_linearization, anchor_normal_linearization, frame_anchor_ids
from contactaware.solver.contact_position import contact_position_weights, validate_position_multiplier
from contactaware.solver.palm_relative_shape import weighted_shape_terms
from contactaware.solver.tracking import fingertip_tracking_linearization, palm_pose_residual, step_regularization_qp


def weighted_residual(residual, jacobian, weight):
    scale = np.sqrt(weight)
    return scale * residual.ravel(), scale * jacobian.reshape(-1, jacobian.shape[-1])


def weighted_points(residual, jacobian, weights):
    scale = np.sqrt(weights)
    return ((scale[:, None] * residual).ravel(),
            (scale[:, None, None] * jacobian).reshape(-1, jacobian.shape[-1]))


def palm_terms(problem, state, target):
    target_position, target_rotation = target(problem)
    position, rotation, position_rows, rotation_rows = palm_pose_residual(
        problem.hand, state, target_position, target_rotation)
    return [weighted_residual(position, position_rows, problem.cfg.palm_position_weight),
            weighted_residual(rotation, rotation_rows, problem.cfg.palm_rotation_weight)]


def shape_terms(problem, state):
    if problem.cfg.mano_shape_weight <= 0.0:
        return []
    return weighted_shape_terms(problem.hand, state, problem.joints, cfg=problem.cfg,
                                 runtime=problem.runtime, frame_id=problem.frame_id)


def contact_terms(problem, state, position_multiplier):
    arguments = dict(cfg=problem.cfg, runtime=problem.runtime, frame_id=problem.frame_id)
    data = anchor_constraint_linearization(problem.hand, state, problem.obj_pose, **arguments)
    terms = []
    if data is not None:
        points, jacobians, targets, participation = data
        columns = frame_anchor_ids(problem.runtime, problem.frame_id)
        blend = np.asarray(problem.runtime.guidance_blend[problem.frame_id, columns], dtype=np.float64)
        weights = contact_position_weights(problem.cfg, participation, blend, position_multiplier)
        terms.append(weighted_points(points - targets, jacobians, weights))
    normals = anchor_normal_linearization(problem.hand, state, problem.obj_pose, **arguments)
    if normals is not None:
        residuals, jacobians, strength = normals
        terms.append(weighted_points(residuals, jacobians, problem.cfg.contact_anchor_weight * strength))
    return terms


@dataclass(frozen=True)
class ContactTrackingObjective:
    """Per-frame palm, shape, fingertip, contact-position and pad-normal residuals."""

    palm_target: Callable
    position_multiplier: float

    def __post_init__(self):
        validate_position_multiplier(self.position_multiplier)

    def residuals(self, problem, state):
        terms = palm_terms(problem, state, self.palm_target) + shape_terms(problem, state)
        tip_residual, tip_jacobian = fingertip_tracking_linearization(
            problem.hand, state, problem.joints, cfg=problem.cfg, runtime=problem.runtime)
        terms.append(weighted_residual(tip_residual, tip_jacobian, problem.cfg.fingertip_weight))
        terms.extend(contact_terms(problem, state, self.position_multiplier))
        return np.concatenate([value[0] for value in terms]), np.vstack([value[1] for value in terms])

    def linearize(self, problem, state):
        residual, jacobian = self.residuals(problem, state)
        proximal_hessian, _ = step_regularization_qp(problem.hand, problem.cfg)
        # Step regularization stabilizes the local solve, not the physical state cost.
        return (.5 * (residual @ residual), jacobian.T @ jacobian + proximal_hessian,
                jacobian.T @ residual)

