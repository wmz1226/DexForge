"""Feasible-start trajectory SQP with explicit swept-interval checks."""

import time
import numpy as np
from scipy import sparse

from contactaware.trajectory.model import (
    ARMIJO,
    MIN_ALPHA,
    GLOBAL_ITERATIONS,
    DISCOVERY_ROUNDS,
    TRUST_MIN,
    Trajectory,
    update_trust,
    trapezoid_cost,
    motion_energy,
    all_coordinates,
)
from contactaware.solver.contact import anchor_constraint_linearization, frame_anchor_ids
from contactaware.solver.qp import SolverFailure
from contactaware.trajectory.repair import (
    repair,
)


def global_sqp(trajectory, states):
    """Optimize all free frames together, discovering constraints during line search."""
    stats, state_dim = trajectory.stats, trajectory.n
    free_frames = np.flatnonzero(~trajectory.fixed)
    if not len(free_frames):
        return states, dict(status="all_frames_fixed", iterations=0,
                            unchecked_swept_intervals=[])
    free_block_index = {int(frame): block_index for block_index, frame in enumerate(free_frames)}
    intervals = [
        interval
        for interval in range(trajectory.count - 1)
        if not (trajectory.fixed[interval] and trajectory.fixed[interval + 1])
    ]
    feasible, _, swept_bad = trajectory.check(states, free_frames, intervals)
    ignored_intervals = set(swept_bad)
    checked_intervals = [interval for interval in intervals if interval not in ignored_intervals]
    frame_pairs = {int(frame): set() for frame in free_frames}
    swept_samples = {}
    coordinates, jacobians = all_coordinates(trajectory, states)
    value = trapezoid_cost(trajectory, states, free_frames) + motion_energy(trajectory, coordinates)
    trust_factor, status = 1.0, "iteration_limit"
    for iteration in range(GLOBAL_ITERATIONS):
        started = time.perf_counter()
        frame_pairs = {
            frame: trajectory.near_pairs(states[frame], discovered_entries)
            for frame, discovered_entries in frame_pairs.items()
        }
        swept_samples = {
            interval: trajectory.near_swept(states, interval, discovered_entries)
            for interval, discovered_entries in swept_samples.items()
        }
        tracking_models = [
            trajectory.case.tracking.linearize(trajectory.case.problems[frame], states[frame])
            for frame in free_frames
        ]
        tracking_hessian = sparse.block_diag(
            [
                trajectory.dt[frame] * tracking_model[1]
                for frame, tracking_model in zip(free_frames, tracking_models)
            ],
            format="csr",
        )
        tracking_gradient = np.concatenate(
            [
                trajectory.dt[frame] * tracking_model[2]
                for frame, tracking_model in zip(free_frames, tracking_models)
            ]
        )
        coordinate_jacobian = sparse.block_diag(jacobians, format="csr")
        columns = np.concatenate(
            [np.arange(frame * state_dim, (frame + 1) * state_dim) for frame in free_frames]
        )
        motion_gradient = trajectory.M @ coordinates.ravel()
        hessian = (
            tracking_hessian
            + (coordinate_jacobian.T @ trajectory.M @ coordinate_jacobian)[columns][:, columns]
        ).tocsc()
        gradient = tracking_gradient + (coordinate_jacobian.T @ motion_gradient)[columns]
        stats.model_seconds += time.perf_counter() - started
        bases = trajectory.batch_constraints({int(frame): states[frame] for frame in free_frames})
        discovered_rounds = 0
        failure = None
        while True:
            model_started = time.perf_counter()
            row_indices, column_indices, row_values, rhs, lower, upper, constraint_count = (
                [],
                [],
                [],
                [],
                [],
                [],
                0,
            )
            for frame in free_frames:
                rows, local_rhs, local_lower, local_upper = trajectory.frame_constraints(
                    states[frame], bases[int(frame)], frame_pairs[int(frame)], trust_factor,
                    frame=int(frame),
                )
                local_row_indices, local_column_indices = np.nonzero(rows)
                row_indices.append(local_row_indices + constraint_count)
                column_indices.append(
                    local_column_indices + free_block_index[int(frame)] * state_dim
                )
                row_values.append(rows[local_row_indices, local_column_indices])
                constraint_count += len(rows)
                rhs.append(local_rhs)
                lower.append(local_lower)
                upper.append(local_upper)
            for interval, discovered_entries in swept_samples.items():
                rows, local_rhs = trajectory.swept_rows(
                    states, interval, discovered_entries, free_block_index
                )
                for endpoint_rows, sample_rhs in zip(rows, local_rhs):
                    for block_index, endpoint_row in endpoint_rows.items():
                        row_indices.append(np.full(state_dim, constraint_count))
                        column_indices.append(
                            np.arange(block_index * state_dim, (block_index + 1) * state_dim)
                        )
                        row_values.append(endpoint_row)
                    constraint_count += 1
                    rhs.append([sample_rhs])
            matrix = sparse.csr_matrix(
                (
                    np.concatenate(row_values),
                    (np.concatenate(row_indices), np.concatenate(column_indices)),
                ),
                shape=(constraint_count, len(free_frames) * state_dim),
            )
            stats.model_seconds += time.perf_counter() - model_started
            try:
                step = trajectory.solve_qp(
                    hessian,
                    gradient,
                    matrix,
                    np.concatenate(rhs),
                    np.concatenate(lower),
                    np.concatenate(upper),
                )
            except SolverFailure as error:
                failure = f"solver_failure: {error}"
                break
            predicted = -float(gradient @ step + 0.5 * step @ (hessian @ step))
            alpha, retry = 1.0, False
            while alpha >= MIN_ALPHA:
                trial = states.copy()
                trial[free_frames] += alpha * step.reshape(len(free_frames), state_dim)
                feasible, frame_bad, swept_bad = trajectory.check(
                    trial, free_frames, checked_intervals
                )
                if not feasible:
                    new_frame_pairs = {
                        frame: violations - frame_pairs[frame]
                        for frame, violations in frame_bad.items()
                    }
                    new_swept_samples = {
                        interval: violations - swept_samples.get(interval, set())
                        for interval, violations in swept_bad.items()
                    }
                    if discovered_rounds < DISCOVERY_ROUNDS and (
                        any(new_frame_pairs.values()) or any(new_swept_samples.values())
                    ):
                        for frame, violations in new_frame_pairs.items():
                            frame_pairs[frame] |= violations
                        for interval, violations in new_swept_samples.items():
                            swept_samples.setdefault(interval, set()).update(violations)
                        discovered_rounds += 1
                        retry = True
                        break
                    alpha *= 0.5
                    continue
                trial_coordinates, trial_jacobians = all_coordinates(trajectory, trial)
                trial_value = trapezoid_cost(trajectory, trial, free_frames) + motion_energy(
                    trajectory, trial_coordinates
                )
                if trial_value <= value - ARMIJO * alpha * max(predicted, 0.0):
                    break
                alpha *= 0.5
            if not retry:
                break
        if failure is not None:
            # States are the last accepted feasible iterate.
            status = failure
            break
        if alpha < MIN_ALPHA:
            stats.history.append(
                dict(iteration=iteration, value=value, alpha=0.0, rho=trust_factor)
            )
            if trust_factor > TRUST_MIN:
                trust_factor = max(trust_factor * 0.25, TRUST_MIN)
                continue
            status = "no_feasible_descent"
            break
        reduction = value - trial_value
        states, coordinates, jacobians, value = (
            trial,
            trial_coordinates,
            trial_jacobians,
            trial_value,
        )
        stats.history.append(
            dict(
                iteration=iteration,
                value=value,
                alpha=alpha,
                rho=trust_factor,
                predicted=predicted,
                reduction=reduction,
                rows=int(sum(len(x) for x in rhs)),
                pairs=int(sum(map(len, frame_pairs.values()))),
                swept=int(sum(map(len, swept_samples.values()))),
            )
        )
        print("[fsqp]", stats.history[-1], flush=True)
        trust_factor = update_trust(trust_factor, alpha)
        if alpha == 1.0 and reduction <= trajectory.cfg.tracking_relative_tolerance * value:
            status = "relative_objective_settled"
            break
    return states, dict(
        status=status,
        iterations=len(stats.history),
        unchecked_swept_intervals=sorted(ignored_intervals),
    )


def key_contact_error(trajectory, t, state):
    """Largest distance of the key's stable-contact queries from their anchors."""
    problem = trajectory.case.problems[t]
    data = anchor_constraint_linearization(problem.hand, state, problem.obj_pose, cfg=problem.cfg,
                                           runtime=problem.runtime, frame_id=problem.frame_id)
    if data is None:
        return 0.0
    positions, _, targets, _ = data
    columns = frame_anchor_ids(problem.runtime, problem.frame_id)
    active = problem.runtime.guidance_blend[problem.frame_id, columns] >= 1.0
    return float(np.linalg.norm(positions - targets, axis=1)[active].max(initial=0.0))


def release_infeasible_keys(trajectory, states):
    """Keep feasible, accurately contacting keys fixed; release the others before trajectory repair."""
    released, limit = [], trajectory.cfg.key_release_contact_error
    for t in np.flatnonzero(trajectory.fixed):
        trajectory.fixed[t] = False
        if trajectory.check(states, [t], [])[0] and key_contact_error(trajectory, t, states[t]) <= limit:
            trajectory.fixed[t] = True
        else:
            released.append(int(t))
    return released


def solve(case):
    """Release keys, repair the start and run global SQP; return states and diagnostics."""
    trajectory = Trajectory(case)
    started = time.perf_counter()
    released = release_infeasible_keys(trajectory, case.initial)
    states, repair_report = repair(trajectory, case.initial.copy())
    repair_seconds = time.perf_counter() - started
    print("[fsqp-repair]", repair_report, f"{repair_seconds:.2f}s", flush=True)
    states, report = global_sqp(trajectory, states)
    stats = trajectory.stats
    report.update(
        repair_report,
        released_keys=released,
        repair_seconds=repair_seconds,
        seconds=time.perf_counter() - started,
        qp_seconds=stats.qp_seconds,
        model_seconds=stats.model_seconds,
        check_seconds=stats.check_seconds,
        value_seconds=stats.value_seconds,
        qps=stats.qps,
        checks=stats.checks,
    )
    return states, report
