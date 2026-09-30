"""Joint key SQP with object-component separation and consistent trial correction."""

from dataclasses import replace
from functools import partial
import time

import numpy as np

from contactaware.solver.feasible_progress import FeasibleProgress
from contactaware.solver.feasibility import key_violation_ratio, key_joint_violation

from contactaware.keys.component_constraints import separated_step
from contactaware.solver.grasp_sqp import (
    PENALTY_GROWTH,
    JointEvaluation, _post_iteration, _solve_report,
    ARMIJO_FRACTION, BACKTRACK_FACTOR,
)
from contactaware.solver.least_norm import structural_least_norm
from contactaware.solver.qp import SolverFailure
from contactaware.solver.convergence import relative_model_converged


PENALTY_ESCALATIONS = 3
ESCALATION_MAX_VIOLATION_RATIO = 2.0


def objective_settled(before, after, jacobian, displacement, factor, cfg, tolerance):
    predicted = before.cost - .5*np.linalg.norm(before.residual + jacobian@displacement)**2
    return relative_model_converged(cost=after.cost, actual_reduction=before.cost-after.cost,
        predicted_reduction=predicted, step_fraction=factor,
        max_constraint_violation=float(after.violation.max()),
        relative_tolerance=cfg.tracking_relative_tolerance, feasibility_tolerance=tolerance)


def correct_key_curvature(candidate, *, constraints, surface, lower, upper, tolerance):
    current = np.array(candidate,copy=True)
    values = constraints(current)
    violation = float(np.maximum(-values,0.).max())
    iterations = 0
    while violation > tolerance:
        active = values<0.
        tangent = surface.tangent_rows(current)
        rows = np.vstack((constraints.selected_jacobian(current,active),tangent))
        rhs = np.r_[-values[active],np.zeros(len(tangent))]
        correction = structural_least_norm(rows,rhs,lower<upper)
        adjusted = surface.project_initial(np.clip(current+correction,lower,upper))
        after = constraints(adjusted)
        next_violation = float(np.maximum(-after,0.).max())
        iterations += 1
        if next_violation >= violation:
            return current,iterations
        current,values,violation = adjusted,after,next_violation
    return current,iterations


def curvature_corrector(slack, **kwargs):
    if slack.max() > kwargs['tolerance']:
        return lambda candidate:(candidate,0)
    return partial(correct_key_curvature,**kwargs)


def predicted_key_reduction(before, jacobian, step, slack, penalty):
    linear = before.residual + jacobian @ step
    return before.cost - .5 * float(linear @ linear) + penalty * float((before.violation - slack).sum())


def joint_progress(before, after, *, penalty, predicted, factor, tolerance):
    # Cancel unchanged constraint residuals before applying a large penalty.
    reduction = before.cost - after.cost + penalty * float((before.violation - after.violation).sum())
    descent = reduction > ARMIJO_FRACTION * factor * max(predicted, 0.)
    # Restoration may trade objective for feasibility, but never at a higher merit.
    restoration = (before.violation.max() > tolerance and reduction >= 0. and
        after.violation.sum() < (1. - ARMIJO_FRACTION * factor) * before.violation.sum())
    return restoration or descent


def conditional_search(current, step, before, *, evaluate, project, penalty,
                       predicted, tolerance, feasibility_tolerance, correct,
                       progress=joint_progress):
    factor, calls, corrections = 1., 0, 0
    accepts = partial(progress, before, penalty=penalty, predicted=predicted,
                      tolerance=feasibility_tolerance)
    while calls == 0 or np.max(np.abs(factor * step)) > tolerance:
        candidate = project(current + factor * step)
        after = evaluate(candidate)
        calls += 1
        if accepts(after, factor=factor):
            return candidate, after, factor, calls, corrections
        candidate, count = correct(candidate)
        after = evaluate(candidate)
        calls, corrections = calls + 1, corrections + count
        if accepts(after, factor=factor):
            return candidate, after, factor, calls, corrections
        factor *= BACKTRACK_FACTOR
    return current, before, 0., calls, corrections


def solve_conditional_keys(objective, constraints, surface, initial, lower, upper, trust, *,
                           solver, regularization, tolerance, feasibility_tolerance,
                           max_iterations, accept, update_trust, prepare_iteration,
                           output=None, progress_factory=FeasibleProgress,
                           search=conditional_search):
    current = surface.project_initial(initial)
    original_constraints = constraints
    evaluate = lambda vector: JointEvaluation(objective(vector), constraints(vector), constraints.groups(vector))
    first = evaluate(current)
    penalty = max(1., float(np.linalg.norm(objective.jacobian(current).T @ first.residual, ord=np.inf)))
    started, history, status = time.perf_counter(), [], 'iteration limit'
    fraction, failed = 1., None
    cfg = original_constraints.problem.cfg
    physical_violation = partial(key_violation_ratio, original_constraints.problem)
    progress = progress_factory(cfg.sqp_stagnation_patience, cfg.sqp_progress_relative_tolerance)
    progress = progress.observe(current, first.cost, physical_violation(current), 1.0, -1)
    escalations = 0
    for iteration in range(max_iterations):
        try:
            prepare_iteration(current)
            before = evaluate(current)
            jacobian = objective.jacobian(current)
            step, slack, constraints, separated = separated_step(objective, original_constraints, surface,
                current, lower, upper, fraction*trust, penalty=penalty,
                regularization=regularization, solver=solver)
            before = evaluate(current)
            predicted = predicted_key_reduction(before, jacobian, step, slack, penalty)
            candidate,after,factor,calls,corrections=search(current,step,before,evaluate=evaluate,
                project=surface.project_initial,penalty=penalty,predicted=predicted,tolerance=tolerance,
                feasibility_tolerance=feasibility_tolerance,
                correct=curvature_corrector(slack,constraints=constraints,surface=surface,
                    lower=lower,upper=upper,tolerance=feasibility_tolerance))
        except SolverFailure as error:
            # Keep the best iterate rather than aborting the whole retargeting.
            status = f'solver failure: {error}'
            print('[conditional-key]', status, flush=True)
            break
        settled = objective_settled(before, after, jacobian, candidate-current, factor,
            original_constraints.problem.cfg, feasibility_tolerance)
        candidate_violation = physical_violation(candidate)
        settled = settled and candidate_violation <= 1.0
        fraction=update_trust(fraction,factor)
        current=candidate
        record=dict(iteration=iteration,cost=after.cost,violation=float(after.violation.max()),
            elastic=float(slack.max()),penalty=penalty,step=factor,search_calls=calls,
            seconds=time.perf_counter()-started,trust_fraction=fraction,curvature_corrections=corrections,inner_models=len(separated))
        progress = progress.observe(current, after.cost, candidate_violation, 1.0,
                                    iteration if factor > 0.0 else -1)
        record.update(progress.summary())
        history.append(record);print('[conditional-key]',record,flush=True)
        ending = progress.ending('no_nonlinear_progress' if factor == 0. else None)
        if (ending is not None and 1.0 < candidate_violation <= ESCALATION_MAX_VIOLATION_RATIO
                and escalations < PENALTY_ESCALATIONS):
            # The elastic penalty is exact only when large enough: raise it before giving up on a nearly
            # feasible key. Far from feasibility a larger penalty only makes the QP ill-conditioned.
            penalty, escalations = penalty * PENALTY_GROWTH, escalations + 1
            progress = replace(progress, stale_iterations=0)
            continue
        if ending is not None:
            status = ending
            break
        if settled:
            status = 'feasible relative objective'
            break
        penalty,failed,ending=_post_iteration(current,before,after,factor,slack,failed,penalty,
            accept=accept,tolerance=tolerance,feasibility_tolerance=feasibility_tolerance)
        if ending is not None:
            status=ending
            break
        if factor == 1.0 and np.max(np.abs(step)) <= tolerance and candidate_violation <= 1.0:
            status='feasible step tolerance'
            break
    current, status = progress.result(current, status)
    report = _solve_report(status,bool(accept(current)),first,evaluate(current),history,started,feasibility_tolerance)
    report.update(progress.summary())
    ratio = physical_violation(current)
    report.update(feasible=ratio <= 1.0, converged=report['converged'] and ratio <= 1.0,
        max_constraint_violation_ratio=ratio,
        joint_limit_violation_rad=key_joint_violation(original_constraints.problem, current),
        geometry_feasibility_tolerance_m=cfg.geometry_feasibility_tolerance_m,
        joint_limit_tolerance_rad=cfg.joint_limit_tolerance_rad)
    return current, report
