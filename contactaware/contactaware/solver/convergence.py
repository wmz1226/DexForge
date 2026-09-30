"""Relative convergence of an accepted nonlinear least-squares SQP step."""

import numpy as np


def relative_model_converged(*, cost, actual_reduction, predicted_reduction,
                             step_fraction, max_constraint_violation,
                             relative_tolerance, feasibility_tolerance):
    """Test both reductions directly; their ratio is unreliable near zero."""
    reductions = np.asarray((cost, actual_reduction, predicted_reduction))
    limit = relative_tolerance * max(abs(cost), np.finfo(np.float64).eps)
    return bool(np.isfinite(reductions).all()
        and step_fraction == 1.0
        and max_constraint_violation <= feasibility_tolerance
        and 0.0 <= actual_reduction <= limit
        and abs(predicted_reduction) <= limit)
