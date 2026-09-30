"""Restore a trajectory by projection and continuation from feasible neighbours."""

import numpy as np

from contactaware.solver.qp import SolverFailure
from contactaware.trajectory.model import (
    ARMIJO,
    DISCOVERY_ROUNDS,
    MIN_ALPHA,
    PROJECTION_ITERATIONS,
    PROJECTION_TRUST,
    REPAIR_ITERATIONS,
    all_coordinates,
    update_trust,
)


def repair(trajectory, states):
    """Repair consecutive infeasible frames and record unresolved boundary intervals."""
    count, fixed = trajectory.count, trajectory.fixed
    intervals = list(range(count - 1))
    ok, frame_bad, swept_bad = trajectory.check(states, range(count), intervals)
    bad = np.zeros(count, dtype=bool)
    bad[list(frame_bad)] = True
    for frame in range(count):
        if not fixed[frame] and not trajectory.self_ok(states[frame]):
            bad[frame] = True
    for interval in swept_bad:
        free = [frame for frame in (interval, interval + 1) if not fixed[frame]]
        if free and not bad[interval] and not bad[interval + 1]:
            bad[max(free)] = True
    coordinates, jacobians = all_coordinates(trajectory, states)
    repaired, unresolved, projected = [], [], []
    if bad.all():
        # Continuation needs a feasible frame: project frames in order until one becomes feasible.
        for seed in range(count):
            state = project_frame(trajectory, states, seed, [])
            if state is not None:
                break
        else:
            return states, dict(repaired_frames=[], projected_frames=[], unresolved_swept_intervals=[],
                                unrepaired_frames=list(range(count)))
        states[seed], bad[seed] = state, False
        coordinates[seed], jacobians[seed] = trajectory.coordinates(state)
        repaired.append(seed)
        projected.append(seed)
    for start, end in infeasible_runs(bad):
        if not bad[start]:
            continue
        direction = march_direction(trajectory, states, start, end)
        frame, previous_frame = (start, start - 1) if direction == 1 else (end, end + 1)
        while True:
            if march_frame(trajectory, states, coordinates, jacobians, frame, previous_frame, bad):
                projected.append(int(frame))
            bad[frame] = False
            repaired.append(int(frame))
            next_frame = frame + direction
            if not 0 <= next_frame < count:
                break
            if bad[next_frame]:
                frame, previous_frame = next_frame, frame
                continue
            if trajectory.check(states, [], [min(frame, next_frame)])[0]:
                break
            if fixed[next_frame]:
                unresolved.append([int(min(frame, next_frame)), int(max(frame, next_frame))])
                break
            frame, previous_frame = next_frame, frame
    return states, dict(
        repaired_frames=repaired, projected_frames=projected, unresolved_swept_intervals=unresolved
    )


def infeasible_runs(bad):
    runs, t = [], 0
    while t < len(bad):
        if bad[t]:
            end = t
            while end + 1 < len(bad) and bad[end + 1]:
                end += 1
            runs.append((t, end))
            t = end
        t += 1
    return runs


def march_direction(trajectory, states, start, end):
    """March away from the more constrained feasible neighbour: a key, else the smaller clearance."""
    sides = [(1, start - 1), (-1, end + 1)]
    sides = [(step, u) for step, u in sides if 0 <= u < trajectory.count]
    if not sides:
        raise ValueError("Continuation needs at least one feasible frame")

    def rank(side):
        u = side[1]
        return (
            not trajectory.fixed[u],
            float(trajectory.phi(trajectory.hand.query_positions(states[u])).min()),
        )

    return min(sides, key=rank)[0]


def march_frame(trajectory, states, coordinates, jacobians, t, previous, bad):
    """Project the frame's own state when the violation is shallow; else continue from the feasible predecessor."""
    interval = min(t, previous)
    far = 2 * t - previous
    intervals = [interval]
    if (
        0 <= far < trajectory.count
        and not bad[far]
        and trajectory.check(states, [], [min(t, far)])[0]
    ):
        intervals.append(min(t, far))
    projected = project_frame(trajectory, states, t, intervals)
    if projected is None:
        states[t] = states[previous]
        intervals = [interval] + [i for i in intervals[1:] if trajectory.check(states, [], [i])[0]]
    else:
        states[t] = projected
    coordinates[t], jacobians[t] = trajectory.coordinates(states[t])
    block_descent(trajectory, states, coordinates, jacobians, t, intervals)
    return projected is not None


def project_frame(trajectory, states, t, intervals):
    """Project one frame towards feasibility; return None if checks still fail."""
    target = states[t].copy()
    trial = states.copy()
    weight = np.diag(1.0 / trajectory.scale**2)
    pairs, swept = set(), {}
    for _ in range(PROJECTION_ITERATIONS):
        ok, frame_bad, swept_bad = trajectory.check(trial, [t], intervals)
        if ok:
            return trial[t].copy()
        pairs |= frame_bad.get(t, set())
        for i, found in swept_bad.items():
            swept.setdefault(i, set()).update(found)
        base = trajectory.base_constraints(trial[t], t, push=True)
        rows, rhs, lower, upper = trajectory.frame_constraints(
            trial[t], base, pairs, PROJECTION_TRUST, push=True, frame=t
        )
        extra = [
            trajectory.swept_rows(trial, i, found, {t: 0}, push=True) for i, found in swept.items()
        ]
        extra_rows = [sample_row[0] for sample_rows, _ in extra for sample_row in sample_rows]
        extra_rhs = [
            sample_rhs for _, sample_rhs_values in extra for sample_rhs in sample_rhs_values
        ]
        if extra_rows:
            rows, rhs = np.vstack([rows, extra_rows]), np.r_[rhs, extra_rhs]
        try:
            step = trajectory.solve_qp(
                weight, weight @ (trial[t] - target), rows, rhs, lower, upper
            )
        except SolverFailure:
            return None
        trial[t] = trial[t] + step
    return trial[t].copy() if trajectory.check(trial, [t], intervals)[0] else None


def block_descent(trajectory, states, coordinates, jacobians, t, intervals):
    """Take feasible single-frame steps of the complete tracking and motion objective."""
    rho, pairs, swept = 1.0, set(), {}
    block = trajectory.block(t)
    motion_block = trajectory.M[block, block].toarray()
    for _ in range(REPAIR_ITERATIONS):
        _, tracking_hessian, tracking_gradient = trajectory.case.tracking.linearize(
            trajectory.case.problems[t], states[t]
        )
        motion_gradient = trajectory.M[block] @ coordinates.ravel()
        coordinate_jacobian = jacobians[t]
        hessian = (
            trajectory.dt[t] * tracking_hessian
            + coordinate_jacobian.T @ motion_block @ coordinate_jacobian
        )
        gradient = trajectory.dt[t] * tracking_gradient + coordinate_jacobian.T @ motion_gradient
        before = trajectory.dt[t] * trajectory.tracking_value(t, states[t])
        base = trajectory.base_constraints(states[t], t)
        accepted = False
        for _ in range(DISCOVERY_ROUNDS + 1):
            rows, rhs, lower, upper = trajectory.frame_constraints(states[t], base, pairs, rho, frame=t)
            extra_rows, extra_rhs = [], []
            for interval, found in swept.items():
                sample_rows, sample_rhs = trajectory.swept_rows(states, interval, found, {t: 0})
                extra_rows += [x[0] for x in sample_rows]
                extra_rhs += sample_rhs
            if extra_rows:
                rows, rhs = np.vstack([rows, extra_rows]), np.r_[rhs, extra_rhs]
            try:
                step = trajectory.solve_qp(hessian, gradient, rows, rhs, lower, upper)
            except SolverFailure:
                return  # the frame keeps its last accepted state
            predicted = -float(gradient @ step + 0.5 * step @ hessian @ step)
            alpha, constraint_discovered = 1.0, False
            while alpha >= MIN_ALPHA:
                trial = states.copy()
                trial[t] = states[t] + alpha * step
                ok, frame_bad, swept_bad = trajectory.check(trial, [t], intervals)
                if not ok:
                    new_pairs = frame_bad.get(t, set()) - pairs
                    new_swept = {
                        interval: new_samples - swept.get(interval, set())
                        for interval, new_samples in swept_bad.items()
                    }
                    if alpha == 1.0 and (new_pairs or any(new_swept.values())):
                        pairs |= new_pairs
                        for interval, new_samples in new_swept.items():
                            swept.setdefault(interval, set()).update(new_samples)
                        constraint_discovered = True
                        break
                    alpha *= 0.5
                    continue
                trial_coordinates, trial_jacobian = trajectory.coordinates(trial[t])
                delta = trial_coordinates - coordinates[t]
                change = trajectory.dt[t] * trajectory.tracking_value(t, trial[t]) - before
                change += float(delta @ motion_gradient + 0.5 * delta @ motion_block @ delta)
                if change <= -ARMIJO * alpha * max(predicted, 0.0):
                    states[t], coordinates[t], jacobians[t] = (
                        trial[t],
                        trial_coordinates,
                        trial_jacobian,
                    )
                    accepted = True
                    break
                alpha *= 0.5
            if not constraint_discovered:
                break
        if not accepted:
            break
        rho = update_trust(rho, alpha)
