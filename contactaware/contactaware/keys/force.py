"""Joint key refinement with a force-closure residual and bounded derivative."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from contactaware.contact.force_margin import make_force_margin_solver
from contactaware.contact.wrench_residual import ContactWrenchResidual
from contactaware.initialization.actual_contact_costs import (
    ActualContactCosts,
    reference_terms,
)
from contactaware.initialization.temporal_contact_keys import (
    KeyMotion,
    ScheduledKeyCosts,
    TemporalContactKeys,
    completed_contact_sets,
    contact_state_weights,
)
from contactaware.keys.surface_anchors import (
    SurfaceAnchorMotion,
    SurfaceAnchorProblem,
)
from contactaware.solver.qp import SolverFailure
from contactaware.solver.temporal import physical_motion_operator

LINEAR_BELOW = 3.0


@dataclass(frozen=True)
class BoundedCircularForceResidual:
    solve: object
    friction: float

    def __call__(self, points, normals, point_rows, normal_rows):
        result = self.solve(points, normals)
        # AlmostSolved meets Clarabel's reduced accuracy tolerances (see force_margin).
        if result.status not in ("Solved", "AlmostSolved"):
            raise SolverFailure(f"Multi-key circular force problem: {result.status}")
        gradient = np.einsum("ni,nij->j", result.point_gradient, point_rows)
        gradient += np.einsum("ni,nij->j", result.normal_gradient, normal_rows)
        x = result.reserve / (2.0 * self.friction)
        if x >= -LINEAR_BELOW:
            residual, slope = np.exp(-x), -np.exp(-x)
        else:
            edge = np.exp(LINEAR_BELOW)
            residual, slope = edge * (1.0 - (x + LINEAR_BELOW)), -edge
        return np.array([residual]), (slope * gradient / (2.0 * self.friction))[None]


def bounded_temporal_problem(
    base,
    inputs,
    args,
    runtime,
    indices,
    *,
    operator_factory=physical_motion_operator,
    origin_rotation=None,
):
    """Key costs and inter-key motion with the bounded force-closure residual."""
    weights = contact_state_weights(inputs.anchors.mask, base.schedule.frames)
    scheduled = ScheduledKeyCosts(
        base,
        runtime,
        indices[base.schedule.frames],
        position_multiplier=args.trajectory_contact_position_multiplier,
        anchor_costs=partial(reference_terms, base),
    )
    friction = args.force_closure_friction
    costs = ActualContactCosts(
        base,
        scheduled,
        weights,
        completed_contact_sets(base.schedule.columns),
        ContactWrenchResidual(
            base.center, base.radius, friction, args.force_closure_force_limit
        ),
        BoundedCircularForceResidual(
            make_force_margin_solver(center=base.center, radius=base.radius, friction=friction),
            friction,
        ),
    )
    return TemporalContactKeys(
        base,
        motion=KeyMotion(
            base,
            fps=args.video_fps,
            seconds=args.trajectory_smoothing_seconds,
            operator_factory=operator_factory,
            origin_rotation=origin_rotation,
        ),
        key_costs=costs,
    )


def bounded_continuous_temporal_problem(
    base, inputs, args, runtime, indices, *, operator_factory, origin_rotation
):
    """The same problem with the shared surface anchors as variables."""
    lifted = SurfaceAnchorProblem(base)
    scheduled = bounded_temporal_problem(
        lifted,
        inputs,
        args,
        runtime,
        indices,
        operator_factory=operator_factory,
        origin_rotation=origin_rotation,
    )
    return TemporalContactKeys(
        lifted,
        motion=SurfaceAnchorMotion(scheduled.motion, lifted.anchor_dimension),
        key_costs=scheduled.key_costs,
    )
