"""Trajectory objective, geometry checks and QP assembly in object coordinates."""

from dataclasses import dataclass, field, replace
import time
import clarabel
import quadprog
import numpy as np
from scipy import sparse
from contactaware.solver.temporal import (
    JOINT_ACCELERATION_MULTIPLIER,
    PALM_ACCELERATION_MULTIPLIER,
    physical_motion_operator,
)
from contactaware.solver.pre_contact import PreContactClearance
from contactaware.solver.qp import SolverFailure
from contactaware.solver.tracking import velocity_weights
from contactaware.contact.object_model import object_distance_obj
from contactaware.solver.collision import (
    capsule_collision_constraints,
    capsule_pair_states,
    near_capsule_pair_states,
)
from contactaware.solver.contact import (
    active_query_ids,
    frame_anchor_query_ids,
    include_anchor_query_ids,
)
from contactaware.solver.temporal import (
    physical_coordinates,
    time_sample_weights,
)
from contactaware.solver.tracking import so3_left_jacobian_inverse

TRANSLATION_SCALE_M = 0.1
SUBSTEPS = 16
ARMIJO = 1e-4
MIN_ALPHA = 1e-3
SWEPT_FILTER = 1.5
REPAIR_ITERATIONS = 3
GLOBAL_ITERATIONS = 40
DISCOVERY_ROUNDS = 2
TRUST_MIN = 0.25
TRUST_MAX = 16.0
PROJECTION_ITERATIONS = 6
PROJECTION_TRUST = 4.0
MARGIN_M = 5e-4
SWEPT_TOLERANCE_M = 1e-3
PRUNE_DISTANCE_M = 5e-3
ACTIVATION_M = 0.003
TOPK_PER_LINK = 16
DENSE_QP_MAX_VARIABLES = 64


@dataclass
class Stats:
    qp_seconds: float = 0.0
    model_seconds: float = 0.0
    check_seconds: float = 0.0
    value_seconds: float = 0.0
    qps: int = 0
    checks: int = 0
    history: list = field(default_factory=list)


def make_cfg(case):
    return case.problems[0].cfg


def motion_operator(case, count=None):
    """Velocity and acceleration operator on the frame grid (the same model as for the keys)."""
    hand = case.hand
    acceleration = np.full(hand.qpos_dim, JOINT_ACCELERATION_MULTIPLIER)
    acceleration[: hand.base_qpos_dim] = PALM_ACCELERATION_MULTIPLIER
    count = len(case.initial) if count is None else count
    times = np.arange(count, dtype=float) / case.args.video_fps
    return physical_motion_operator(
        times,
        case.args.trajectory_smoothing_seconds,
        velocity_weights(hand, make_cfg(case)),
        acceleration_multiplier=acceleration,
    )


class Trajectory:
    """Trajectory objective and constraints; states have shape (frames, qpos_dim)."""

    def __init__(self, case):
        self.case, self.cfg = case, make_cfg(case)
        self.hand, self.obj = case.hand, case.inputs.obj
        self.n = self.hand.qpos_dim
        self.count = len(case.initial)
        self.fixed = np.zeros(self.count, dtype=bool)
        self.fixed[case.frames] = True
        self.dt = time_sample_weights(
            np.arange(self.count, dtype=float) / case.args.video_fps,
            single_frame_seconds=1.0 / case.args.video_fps,
        )
        motion_operator_matrix = motion_operator(case)
        self.M = (motion_operator_matrix.T @ motion_operator_matrix).tocsr()
        self.scale = np.ones(self.n)
        self.scale[:3] = TRANSLATION_SCALE_M
        self.tol = self.cfg.geometry_feasibility_tolerance_m
        self.pre_contact = PreContactClearance(self.hand, case.inputs.anchors, self.cfg.pre_contact_safe_distance,
                                               self.cfg.hand_object_safe_distance)
        self.safe = self.pre_contact.maximum  # conservative bound for pruning discovered pairs
        self.stats = Stats()
        self.active_cfg = replace(
            self.cfg, hand_object_activation=ACTIVATION_M + self.pre_contact.maximum,
            hand_object_topk_per_link=TOPK_PER_LINK
        )

    def coordinates(self, state):
        """Physical translation, rotation and finger coordinates and their Jacobian."""
        return physical_coordinates(
            self.hand, state, self.case.origin, rotation_jacobian=so3_left_jacobian_inverse
        )

    def tracking_value(self, t, state):
        residual, _ = self.case.tracking.residuals(self.case.problems[t], state)
        return 0.5 * float(residual @ residual)

    def block(self, t):
        return slice(t * self.n, (t + 1) * self.n)

    def phi(self, points):
        return self.obj.query_fn.distance_bounds(points)[:, 0]

    def component_rows(self, state, queries, components):
        """Per-query object-component clearance and outward Jacobian in object coordinates."""
        points, jacobians = self.hand.query_positions_jacobian(state, np.asarray(queries))
        value, gradient = self.obj.query_fn.constraint_values(points, components)
        return value, np.einsum("ni,nij->nj", gradient, jacobians)

    def self_ok(self, state):
        clearance = capsule_pair_states(self.hand, *self.hand.capsule_segments(state))[-1]
        return float(clearance.min()) >= self.cfg.safe_distance - self.tol

    def check(self, states, frames, intervals):
        """Check requested frames and intervals; return newly intersected object-component rows."""
        started = time.perf_counter()
        hand, object_query = self.hand, self.obj.query_fn
        frames = sorted(
            set(frames) | {frame for interval in intervals for frame in (interval, interval + 1)}
        )
        if not frames:
            return True, {}, {}
        kinematics = {frame: hand.points_and_capsules(states[frame]) for frame in frames}
        points = {frame: item[0] for frame, item in kinematics.items()}
        limit = self.cfg.safe_distance - self.tol
        self_ok = {frame: bool(np.all(near_capsule_pair_states(hand, *item[1:], limit)[3] >= limit))
                   for frame, item in kinematics.items()}
        field = object_query.distance_bounds(np.concatenate([points[frame] for frame in frames]))
        values, bounds = field[:, 0], field[:, 1]
        frame_distances = dict(zip(frames, np.split(values, len(frames))))
        frame_bounds = dict(zip(frames, np.split(bounds, len(frames))))
        frame_violations, swept_violations, feasible = {}, {}, True
        for frame in frames:
            violating_queries = np.flatnonzero(frame_distances[frame] < self.pre_contact(frame) - self.tol)
            if not self.fixed[frame] and (
                len(violating_queries) or not self_ok[frame]
            ):
                feasible = False
                if len(violating_queries):
                    rows, components = object_query.select_constraints(points[frame][violating_queries])
                    frame_violations[frame] = set(
                        zip(violating_queries[rows].tolist(), components.tolist())
                    )
        samples, owners = [], []
        for interval in intervals:
            reach = (
                SWEPT_FILTER * np.linalg.norm(points[interval + 1] - points[interval], axis=1)
                + self.tol
            )
            swept_safe = self.pre_contact(interval + 1)
            ids = np.flatnonzero(
                np.minimum(frame_bounds[interval], frame_bounds[interval + 1])
                - swept_safe
                + SWEPT_TOLERANCE_M
                < reach
            )
            # A failed broad-phase bound only schedules a soft-field query;
            # frame/swept acceptance and discovered rows use that field alone.
            if not len(ids):
                continue
            for fraction in np.arange(1, SUBSTEPS) / SUBSTEPS:
                samples.append(
                    hand.query_positions(
                        (1 - fraction) * states[interval] + fraction * states[interval + 1], ids
                    )
                )
                owners.append((interval, fraction, ids, swept_safe[ids]))
        if samples:
            values = self.phi(np.concatenate(samples))
            deepest = {}
            for (interval, fraction, ids, sample_safe), sample_distances, sample in zip(
                owners, np.split(values, np.cumsum([len(x) for x in samples])[:-1]), samples
            ):
                for query_index in np.flatnonzero(sample_distances < sample_safe - SWEPT_TOLERANCE_M):
                    key = (interval, int(ids[query_index]))
                    if key not in deepest or sample_distances[query_index] < deepest[key][0]:
                        deepest[key] = (
                            sample_distances[query_index],
                            fraction,
                            sample[query_index],
                        )
            if deepest:
                feasible = False
                violations = list(deepest.items())
                rows, components = object_query.select_constraints(
                    np.asarray([violation[2] for _, violation in violations])
                )
                for row, component in zip(rows, components):
                    (interval, query_id), (_, fraction, _) = violations[row]
                    swept_violations.setdefault(interval, set()).add(
                        (fraction, query_id, int(component))
                    )
        self.stats.check_seconds += time.perf_counter() - started
        self.stats.checks += 1
        return feasible, frame_violations, swept_violations

    def solve_qp(self, hessian, gradient, rows, rhs, lower, upper):
        """Solve the scaled QP: dense active set for one frame, Clarabel for the sparse trajectory."""
        started = time.perf_counter()
        size = len(gradient)
        scale = np.tile(self.scale, size // self.n)
        if size <= DENSE_QP_MAX_VARIABLES:
            step = dense_qp(hessian, gradient, rows, rhs, lower, upper, scale)
            if step is not None:
                self.stats.qp_seconds += time.perf_counter() - started
                self.stats.qps += 1
                return step
        scaling_matrix = sparse.diags(scale)
        quadratic_matrix = sparse.triu(
            scaling_matrix @ sparse.csc_matrix(hessian) @ scaling_matrix, format="csc"
        )
        constraint_matrix = sparse.vstack(
            [-sparse.csr_matrix(rows) @ scaling_matrix, sparse.eye(size), -sparse.eye(size)],
            format="csc",
        )
        constraint_bounds = np.r_[-rhs, upper / scale, -lower / scale]
        settings = clarabel.DefaultSettings()
        settings.verbose, settings.max_threads = False, 1
        solver = clarabel.DefaultSolver(
            quadratic_matrix,
            gradient * scale,
            constraint_matrix,
            constraint_bounds,
            [clarabel.NonnegativeConeT(len(constraint_bounds))],
            settings,
        )
        result = solver.solve()
        self.stats.qp_seconds += time.perf_counter() - started
        self.stats.qps += 1
        if str(result.status) not in ("Solved", "AlmostSolved"):
            raise SolverFailure(f"trajectory QP: {result.status}")
        return np.asarray(result.x) * scale

    def backoff(self, value, safe, push=False):
        """Keep a curvature margin; only projection (push) may demand an outward move."""
        demand = safe + MARGIN_M - value
        return demand if push else np.minimum(demand, 0.0)

    def base_constraints(self, state, frame, push=False):
        return self.batch_constraints({frame: state}, push)[frame]

    def batch_constraints(self, states, push=False):
        """Batch object queries; combine capsule, object-component and joint-limit rows."""
        started = time.perf_counter()
        hand, cfg, frames = self.hand, self.active_cfg, list(states)
        points = [hand.query_positions(states[frame]) for frame in frames]
        distances, distance_gradients = object_distance_obj(np.concatenate(points), self.obj)
        batch_offsets = np.cumsum([len(x) for x in points])[:-1]
        result = {}
        for frame, frame_points, frame_distances, frame_gradients in zip(
            frames,
            points,
            np.split(distances, batch_offsets),
            np.split(distance_gradients, batch_offsets),
        ):
            problem = self.case.problems[frame]
            ids = include_anchor_query_ids(
                active_query_ids(hand, frame_distances, cfg),
                frame_anchor_query_ids(cfg, problem.runtime, problem.frame_id),
            )
            self_rows, self_rhs = capsule_collision_constraints(
                hand, states[frame], cfg, allow_initial_penetration=False
            )
            object_rows = np.einsum(
                "ni,nij->nj", frame_gradients[ids], hand.query_jacobians(ids, frame_points[ids])
            )
            object_rhs = cfg.hand_object_gamma * (self.pre_contact(frame, ids) - frame_distances[ids])
            backoff = np.r_[self_rhs, object_rhs] + MARGIN_M
            backoff = backoff if push else np.minimum(backoff, 0.0)
            # Joint limits are variable bounds (frame_constraints), not rows.
            result[frame] = (np.vstack([self_rows, object_rows]), backoff)
        self.stats.model_seconds += time.perf_counter() - started
        return result

    def frame_constraints(self, state, base, pairs, rho, push=False, *, frame):
        """Add discovered object-component rows and the scaled trust region."""
        rows, rhs = base
        if pairs:
            query_ids, component_ids = np.asarray(sorted(pairs)).T
            distances, component_rows = self.component_rows(state, query_ids, component_ids)
            rows = np.vstack([rows, component_rows])
            rhs = np.r_[rhs, self.backoff(distances, self.pre_contact(frame, query_ids), push)]
        trust = np.full(self.n, self.cfg.joint_trust_rad)
        trust[:3], trust[3 : self.hand.base_qpos_dim] = (
            self.cfg.base_translation_trust_m,
            self.cfg.base_rotation_trust_rad,
        )
        gamma = np.ones(self.n)
        gamma[self.hand.base_qpos_dim:] = self.cfg.joint_gamma
        lower = np.maximum(np.maximum(self.hand.lower - state, gamma * (self.hand.lower - state)), -rho * trust)
        upper = np.minimum(np.minimum(self.hand.upper - state, gamma * (self.hand.upper - state)), rho * trust)
        return rows, rhs, lower, upper

    def near_pairs(self, state, pairs):
        """Drop discovered object components that are now farther than the prune distance."""
        if not pairs:
            return pairs
        query_ids, component_ids = np.asarray(sorted(pairs)).T
        distances, _ = self.component_rows(state, query_ids, component_ids)
        return {
            pair
            for pair, distance in zip(sorted(pairs), distances)
            if distance < self.safe + PRUNE_DISTANCE_M
        }

    def near_swept(self, states, interval, discovered):
        kept = set()
        for fraction, query_id, component_id in discovered:
            distances, _ = self.component_rows(
                (1 - fraction) * states[interval] + fraction * states[interval + 1],
                [query_id],
                [component_id],
            )
            if distances[0] < self.safe + PRUNE_DISTANCE_M:
                kept.add((fraction, query_id, component_id))
        return kept

    def swept_rows(self, states, interval, discovered, free_index, push=False):
        """Per-component rows at discovered swept samples, split over the two endpoint blocks."""
        rows, rhs = [], []
        for fraction, query_id, component_id in sorted(discovered):
            state = (1 - fraction) * states[interval] + fraction * states[interval + 1]
            distances, row = self.component_rows(state, [query_id], [component_id])
            endpoint_rows = {}
            for frame, weight in ((interval, 1 - fraction), (interval + 1, fraction)):
                if frame in free_index:
                    endpoint_rows[free_index[frame]] = weight * row[0]
            if endpoint_rows:
                rows.append(endpoint_rows)
                rhs.append(self.backoff(distances, self.pre_contact(interval + 1, [query_id]), push)[0])
        return rows, rhs


def dense_qp(hessian, gradient, rows, rhs, lower, upper, scale):
    """Goldfarb-Idnani active set on the scaled single-frame QP; None if it reports inconsistency."""
    hessian = np.asarray(hessian.toarray() if sparse.issparse(hessian) else hessian) * np.outer(scale, scale)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, len(scale)) * scale
    eye = np.eye(len(scale))
    constraints = np.vstack([rows, eye, -eye])
    bounds = np.r_[rhs, lower / scale, -upper / scale]
    try:
        solution = quadprog.solve_qp(.5 * (hessian + hessian.T), -gradient * scale, constraints.T, bounds)[0]
    except ValueError:
        return None
    return solution * scale


def update_trust(rho, alpha):
    if alpha == 1.0:
        return min(2 * rho, TRUST_MAX)
    return rho if alpha >= 0.25 else max(0.5 * rho, TRUST_MIN)


def trapezoid_cost(trajectory, states, frames):
    started = time.perf_counter()
    value = sum(trajectory.dt[t] * trajectory.tracking_value(t, states[t]) for t in frames)
    trajectory.stats.value_seconds += time.perf_counter() - started
    return value


def motion_energy(trajectory, coordinates):
    flat = coordinates.ravel()
    return 0.5 * float(flat @ (trajectory.M @ flat))


def all_coordinates(trajectory, states):
    values = [trajectory.coordinates(state) for state in states]
    return np.asarray([coordinate_result[0] for coordinate_result in values]), [
        coordinate_result[1] for coordinate_result in values
    ]
