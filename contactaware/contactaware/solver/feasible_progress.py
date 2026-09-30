"""Track the best computed iterate; feasibility alone is not convergence."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class FeasibleProgress:
    patience: int
    relative_tolerance: float
    state: object = None
    cost: float = np.inf
    iteration: int = -1
    violation: float = np.inf
    reference_quality: float = np.inf
    reference_cost: float = np.inf
    stale_iterations: int = 0
    has_update: bool = False

    def __post_init__(self):
        if int(self.patience) != self.patience or self.patience < 0:
            raise ValueError('SQP stagnation patience must be a nonnegative integer')
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.:
            raise ValueError('SQP progress tolerance must be finite and positive')

    @property
    def feasible(self):
        return self.violation <= 1.0

    @property
    def quality(self):
        return self.cost if self.feasible else self.violation

    def observe(self, state, cost, violation, tolerance, iteration):
        finite = np.isfinite(cost) and np.isfinite(violation) and np.isfinite(state).all()
        if not finite:
            return replace(self, stale_iterations=self.stale_iterations + 1)
        ratio = float(violation / tolerance)
        # The initial trajectory is a guess, not an SQP update.
        first_update = iteration >= 0 and not self.has_update
        rank = (ratio > 1.0, max(ratio - 1.0, 0.0), float(cost))
        old_rank = (not self.feasible, max(self.violation - 1.0, 0.0), self.cost)
        best = self
        if self.state is None or first_update or rank < old_rank:
            saved = np.array(state, copy=True)
            saved.setflags(write=False)
            best = replace(self, state=saved, cost=float(cost), violation=ratio, iteration=iteration)
        best = replace(best, has_update=self.has_update or iteration >= 0)
        scale = max(abs(self.reference_quality), np.finfo(float).eps)
        improved = self.reference_quality - best.quality > self.relative_tolerance * scale
        # Before feasibility the iterates may still reduce the objective at a sub-tolerance violation level.
        descending = (not best.feasible and np.isfinite(self.reference_cost)
                      and self.reference_cost - cost > self.relative_tolerance * max(abs(self.reference_cost), 1e-12))
        phase_changed = best.feasible != self.feasible
        if first_update or phase_changed or not np.isfinite(self.reference_quality) or improved or descending:
            return replace(best, reference_quality=best.quality, reference_cost=float(cost), stale_iterations=0)
        return replace(best, stale_iterations=self.stale_iterations + 1)

    @property
    def stalled(self):
        return self.patience > 0 and self.stale_iterations >= self.patience

    def summary(self):
        return dict(best_cost=self.cost, best_iteration=self.iteration,
                    best_constraint_violation_ratio=self.violation,
                    stale_iterations=self.stale_iterations, stagnation_patience=self.patience,
                    progress_relative_tolerance=self.relative_tolerance)

    def ending(self, status=None):
        if self.stalled:
            return 'best_feasible_stagnation' if self.feasible else 'best_available_stagnation'
        return status

    def result(self, current, status):
        if self.state is None:
            return current, status
        if not np.array_equal(current, self.state):
            status = 'best_feasible_return' if self.feasible else 'best_available_return'
        return np.array(self.state, copy=True), self.ending(status)


