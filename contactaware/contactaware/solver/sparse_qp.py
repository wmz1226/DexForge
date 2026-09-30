"""Sparse interior-point QP for shared anchors and elastic contact feasibility."""

from functools import partial
import numpy as np
from scipy import sparse

from contactaware.solver.qp import QPResult, SolverFailure, primal_violation_details, validate_problem


def cone_constraints(problem, *, equality_cone, inequality_cone):
    rows = sparse.vstack((sparse.csc_matrix(problem.rows), sparse.eye(len(problem.gradient))), format="csc")
    lower, upper = np.r_[problem.lower_rows, problem.lower], np.r_[problem.upper_rows, problem.upper]
    equality = np.isfinite(lower) & (lower == upper)
    below, above = np.isfinite(lower) & ~equality, np.isfinite(upper) & ~equality
    matrix = sparse.vstack((rows[equality], -rows[below], rows[above]), format="csc")
    bounds = np.r_[upper[equality], -lower[below], upper[above]]
    cones = [equality_cone(int(equality.sum())), inequality_cone(int(below.sum() + above.sum()))]
    return matrix, bounds, cones


def solve_interior_point(problem, options, *, solver_factory, settings_factory, equality_cone, inequality_cone):
    matrix, bounds, cones = cone_constraints(problem, equality_cone=equality_cone, inequality_cone=inequality_cone)
    settings = settings_factory()
    settings.verbose, settings.max_threads = False, 1
    settings.max_iter = options.max_iterations
    settings.tol_gap_abs = options.absolute_tolerance
    settings.tol_gap_rel = options.absolute_tolerance
    settings.tol_feas = options.absolute_tolerance
    solver = solver_factory(sparse.triu(sparse.csc_matrix(problem.hessian), format="csc"),
                             problem.gradient, matrix, bounds, cones, settings)
    result = solver.solve()
    if str(result.status) != "Solved":
        raise SolverFailure(f"Joint contact interior-point QP failed: {result.status}; "
                           f"primal={result.r_prim:g}, dual={result.r_dual:g}, "
                           f"gap={abs(result.obj_val - result.obj_val_dual):g}, "
                           f"cost={result.obj_val:g}, iterations={result.iterations}")
    return np.asarray(result.x, dtype=np.float64)


class JointContactQP:
    def __init__(self, options, *, solve):
        self.options, self.solve_convex = options, solve

    def solve(self, problem, *, label):
        validate_problem(problem)
        solution = self.solve_convex(problem, self.options)
        details = primal_violation_details(problem, solution)
        violation = max(details.values())
        if violation > self.options.validation_tolerance:
            raise SolverFailure(f"Invalid joint contact QP for {label}: {details}")
        return QPResult(solution, violation)


def make_joint_contact_qp(options):
    import clarabel
    return JointContactQP(options, solve=partial(
        solve_interior_point, solver_factory=clarabel.DefaultSolver, settings_factory=clarabel.DefaultSettings,
        equality_cone=clarabel.ZeroConeT, inequality_cone=clarabel.NonnegativeConeT))
