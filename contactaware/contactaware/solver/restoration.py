"""SLSQP restoration with the same bounded-stagnation policy as key SQP."""
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from contactaware.solver.feasible_progress import FeasibleProgress


class _RestorationStalled(Exception):
    pass


def minimize_restoration(fun, x0, *, constraints, jac, bounds, method, options,
                         patience, relative_tolerance, feasibility_tolerance,
                         callback=None):
    """Keep the best feasible iterate; stagnation is reported, not called convergence.

    Both contact modes supply their own consistent constraint function. The
    existing physical feasibility tolerance is unchanged. An infeasible best
    iterate remains a failure for the caller's normal exterior-start fallback.
    """
    progress = FeasibleProgress(patience, relative_tolerance)
    iterations = 0
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        return fun(x)

    def observe(x, iteration):
        nonlocal progress
        violation = max(0., -float(np.min(constraints['fun'](x))))
        progress = progress.observe(x, float(fun(x)[0]), violation, feasibility_tolerance, iteration)

    # Restoration starts from a real candidate, which may already be feasible.
    # Mark it as an observed iterate so the first failed SLSQP step cannot
    # unconditionally replace it under the key solver's initial-guess policy.
    observe(x0, 0)

    def check(x):
        nonlocal iterations
        iterations += 1
        observe(x, iterations)
        if callback is not None:
            callback(x)
        if progress.stalled:
            raise _RestorationStalled

    try:
        result = minimize(evaluate, x0, constraints=constraints, jac=jac, bounds=bounds,
                          method=method, options=options, callback=check)
    except _RestorationStalled:
        result = OptimizeResult(x=np.array(progress.state, copy=True),
                                success=False, status=11, message=progress.ending(),
                                nit=iterations, nfev=evaluations)
    else:
        observe(result.x, int(result.nit))
    solver_success = bool(result.success)
    selected, _ = progress.result(result.x, str(result.message))
    if not np.array_equal(selected, result.x):
        result.success = False
        result.message = f"best iterate returned after: {result.message}"
    result.x = selected
    result.fun = float(fun(selected)[0])
    result.feasible = bool(np.isfinite(selected).all() and
                           np.min(constraints['fun'](selected)) >= -feasibility_tolerance)
    result.solver_success = solver_success
    result.progress = progress.summary()
    return result
