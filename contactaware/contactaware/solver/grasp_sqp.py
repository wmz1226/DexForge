"""Elastic trust-region SQP with anchors kept on the object surface."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from contactaware.solver.qp import QPProblem, SolverFailure

BACKTRACK_FACTOR = 0.5
PENALTY_GROWTH = 10.
ARMIJO_FRACTION = 1.e-4


@dataclass(frozen=True)
class JointEvaluation:
    residual: np.ndarray
    constraints: np.ndarray
    groups: np.ndarray

    @property
    def cost(self):
        return 0.5 * float(self.residual @ self.residual)

    @property
    def violation(self):
        result = np.zeros(int(self.groups.max()) + 1)
        np.maximum.at(result, self.groups, np.maximum(-self.constraints, 0.))
        return result


@dataclass(frozen=True)
class ShrinkingTrustRegion:
    line_search_trigger: float
    shrink_factor: float
    minimum_fraction: float

    def __call__(self, fraction, line_search_factor):
        if line_search_factor > self.line_search_trigger:
            return fraction
        return max(self.minimum_fraction, self.shrink_factor * fraction)


def merit(value, penalty):
    return value.cost + penalty * float(value.violation.sum())


def unchanged_constraint_set(vector):
    """Static constraint providers need no working-set update."""
    del vector


def elastic_step(value, jacobian, rows, tangent, lower, upper, penalty, regularization, solver, gram=None):
    dimension, phases = jacobian.shape[1], len(value.violation)
    gram = jacobian.T @ jacobian if gram is None else gram
    hessian = sparse.block_diag((gram + regularization * np.eye(dimension),
                                 regularization * np.eye(phases)), format="csc")
    hessian.eliminate_zeros()
    gradient = np.r_[jacobian.T @ value.residual, np.full(phases, penalty)]
    incidence = sparse.csr_matrix(np.eye(phases)[value.groups])
    constraints = sparse.bmat([[sparse.csr_matrix(rows), incidence],
                               [sparse.csr_matrix(tangent, shape=(len(tangent), dimension)), None]], format="csr")
    constraints.eliminate_zeros()
    bounds = np.r_[-value.constraints, np.zeros(len(tangent))]
    maxima = np.r_[np.full(len(value.constraints), np.inf), np.zeros(len(tangent))]
    problem = QPProblem(hessian, gradient, constraints, bounds, maxima,
                        np.r_[lower, np.zeros(phases)], np.r_[upper, np.full(phases, np.inf)])
    result = solver.solve(problem, label="joint grasp elastic feasibility").solution
    return result[:dimension], result[dimension:]


def correct_constraint_curvature(candidate, *, constraints, surface, lower, upper, tolerance):
    """Second-order correction of the same constraints after surface retraction."""
    current = np.array(candidate, copy=True)
    values = constraints(current)
    violation = float(np.maximum(-values, 0.).max())
    iterations = 0
    while violation > tolerance:
        active = values < 0.
        tangent = surface.tangent_rows(current)
        rows = np.vstack((constraints.jacobian(current)[active], tangent))
        free = lower < upper
        correction = np.zeros_like(current)
        correction[free] = np.linalg.lstsq(
            rows[:, free], np.r_[-values[active], np.zeros(len(tangent))], rcond=None)[0]
        adjusted = surface.project_initial(np.clip(current + correction, lower, upper))
        next_values = constraints(adjusted)
        next_violation = float(np.maximum(-next_values, 0.).max())
        iterations += 1
        if next_violation >= violation:
            return current, iterations
        current, values, violation = adjusted, next_values, next_violation
    return current, iterations


def projected_search(current, step, before, *, evaluate, project, penalty, predicted, tolerance,
                     feasibility_tolerance, correct):
    factor, calls = 1., 0
    corrections = 0
    while calls == 0 or np.max(np.abs(factor * step)) > tolerance:
        candidate = project(current + factor * step)
        candidate, count = correct(candidate)
        corrections += count
        after = evaluate(candidate)
        calls += 1
        feasibility_progress = (before.violation.max() > feasibility_tolerance and
                                after.violation.sum() < (1. - ARMIJO_FRACTION * factor) * before.violation.sum())
        merit_progress = merit(after, penalty) < merit(before, penalty) - ARMIJO_FRACTION * factor * max(predicted, 0.)
        if feasibility_progress or merit_progress:
            return candidate, after, factor, calls, corrections
        factor *= BACKTRACK_FACTOR
    return current, before, 0., calls, corrections


def _identity_correction(candidate):
    return candidate, 0


def _curvature_corrector(
    slack,
    *,
    constraints,
    surface,
    lower,
    upper,
    tolerance,
):
    if slack.max() > tolerance:
        return _identity_correction
    return lambda candidate: correct_constraint_curvature(
        candidate,
        constraints=constraints,
        surface=surface,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
    )


def _post_iteration(
    current,
    before,
    after,
    factor,
    slack,
    previous_failure,
    penalty,
    *,
    accept,
    tolerance,
    feasibility_tolerance,
):
    if factor == 0.0 and repeated_failed_feasibility(
        previous_failure,
        current,
        slack,
        tolerance,
        feasibility_tolerance,
    ):
        return penalty, previous_failure, "infeasible stationary feasibility after penalty update"
    failure = (current.copy(), slack.copy()) if factor == 0.0 else None
    if accept(current):
        return penalty, failure, "accepted physical accuracy"
    if factor == 0.0 and after.violation.max() <= feasibility_tolerance:
        return penalty, failure, "feasible stationary merit"
    if (
        after.violation.sum() >= before.violation.sum()
        and slack.max() > feasibility_tolerance
    ):
        penalty *= PENALTY_GROWTH
    return penalty, failure, None


def _solve_report(status, accepted, first, final, history, started, tolerance):
    target_feasible = bool(final.violation.max() <= tolerance)
    # A feasible zero-step line search does not establish stationarity.
    converged = accepted or status in {"feasible relative objective", "feasible step tolerance"}
    evaluations = sum(record["search_calls"] for record in history)
    return {
        "message": status,
        "converged": converged,
        "feasible": accepted or target_feasible,
        "physical_accuracy_accepted": accepted,
        "target_constraints_satisfied": target_feasible,
        "max_normalized_constraint_violation": float(final.violation.max()),
        "initial_objective": first.cost,
        "final_objective": final.cost,
        "iterations": len(history),
        "evaluations": evaluations,
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def solve_joint_grasp(objective, constraints, surface, initial, lower, upper, trust, *,
                       solver, regularization, tolerance, feasibility_tolerance,
                       max_iterations, accept, update_trust,
                       prepare_iteration=unchanged_constraint_set):
    current = surface.project_initial(initial)
    evaluate = lambda vector: JointEvaluation(objective(vector), constraints(vector), constraints.groups(vector))
    first = evaluate(current)
    gradient = objective.jacobian(current).T @ first.residual
    penalty = max(1., float(np.linalg.norm(gradient, ord=np.inf)))
    started, history, status = time.perf_counter(), [], "iteration limit"
    trust_fraction = 1.0
    failed_feasibility = None
    for iteration in range(max_iterations):
        try:
            current, transfers = constraints.transfer_phase_initials(current)
            prepare_iteration(current)
            if accept(current):
                status = "accepted physical accuracy"
                break
            before = evaluate(current)
            jacobian = objective.jacobian(current)
            rows = constraints.jacobian(current)
            tangent = surface.tangent_rows(current)
            step, slack = elastic_step(
                before,
                jacobian,
                rows,
                tangent,
                np.maximum(lower - current, -trust_fraction * trust),
                np.minimum(upper - current, trust_fraction * trust),
                penalty,
                regularization,
                solver,
            )
            if (
                np.max(np.abs(step)) <= tolerance
                and before.violation.max() <= feasibility_tolerance
            ):
                status = "feasible step tolerance"
                break
            linear = before.residual + jacobian @ step
            predicted = (
                merit(before, penalty)
                - 0.5 * float(linear @ linear)
                - penalty * slack.sum()
            )
            candidate, after, factor, calls, corrections = projected_search(
                current,
                step,
                before,
                evaluate=evaluate,
                project=surface.project_initial,
                penalty=penalty,
                predicted=predicted,
                tolerance=tolerance,
                feasibility_tolerance=feasibility_tolerance,
                correct=_curvature_corrector(
                    slack,
                    constraints=constraints,
                    surface=surface,
                    lower=lower,
                    upper=upper,
                    tolerance=feasibility_tolerance,
                ),
            )
        except SolverFailure as error:
            # Keep the current iterate rather than aborting the whole retargeting.
            status = f"solver failure: {error}"
            print(f"[joint-grasp] {status}", flush=True)
            break
        trust_fraction = update_trust(trust_fraction, factor)
        record = {
            "iteration": iteration,
            "cost": after.cost,
            "violation": float(after.violation.max()),
            "elastic": float(slack.max()),
            "penalty": penalty,
            "step": factor,
            "search_calls": calls,
            "seconds": time.perf_counter() - started,
            "trust_fraction": trust_fraction,
            "curvature_corrections": corrections,
            "phase_initial_transfers": transfers,
            "phases": constraints.diagnostics(candidate),
        }
        history.append(record)
        print(f"[joint-grasp] {record}", flush=True)
        current = candidate
        penalty, failed_feasibility, ending = _post_iteration(
            current,
            before,
            after,
            factor,
            slack,
            failed_feasibility,
            penalty,
            accept=accept,
            tolerance=tolerance,
            feasibility_tolerance=feasibility_tolerance,
        )
        if ending is not None:
            status = ending
            break
    final = evaluate(current)
    accepted = bool(accept(current))
    return current, _solve_report(
        status,
        accepted,
        first,
        final,
        history,
        started,
        feasibility_tolerance,
    )


def repeated_failed_feasibility(previous, state, slack, tolerance, feasibility_tolerance):
    """Stop explicit failure when larger penalty improves neither model nor actual feasibility."""
    if previous is None:
        return False
    old_state, old_slack = previous
    return (np.max(np.abs(state - old_state)) <= tolerance
            and np.max(np.abs(slack - old_slack)) <= feasibility_tolerance)


