"""Discover geometry constraints on linearized material-point motion inside SQP.

Hard uses fixed feature cuts; soft uses only the fused distance field.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from contactaware.initialization.multi_contact_keys import CONSTRAINT_GROUPS_PER_KEY
from contactaware.solver.grasp_sqp import JointEvaluation, elastic_step


def safe_distances(problem, key, queries):
    active = problem.query_ids[problem.schedule.columns[key]]
    return np.where(np.isin(queries, active), 0., problem.pre_contact(problem.schedule.frames[key], queries))


@dataclass(frozen=True)
class ComponentConstraints:
    original: object
    pairs: tuple

    @property
    def problem(self):
        return self.original.problem

    def component_terms(self, vector, *, derivatives=False):
        problem = self.problem
        geometry = problem.constraint_evaluation(vector)
        values, rows, groups = [], [], []
        for key, pairs in enumerate(self.pairs):
            if not pairs:
                continue
            queries, features = np.asarray(sorted(pairs), dtype=np.int64).T
            robot = geometry.robots[key]
            phi, normals = problem.geometry.component_rows(robot.points[queries], features)
            values.append((phi - safe_distances(problem, key, queries)) / problem.radius)
            groups.append(np.full(len(pairs), CONSTRAINT_GROUPS_PER_KEY * key, dtype=np.int64))
            if derivatives:
                local = np.einsum('ni,nij->nj', normals, robot.point_jacobians[queries])
                rows.append(problem.normalized_rows(local, key))
        return (np.concatenate(values), sparse.vstack(rows, format="csr") if derivatives else None,
                np.concatenate(groups))

    def __call__(self, vector):
        return np.r_[self.original(vector), self.component_terms(vector)[0]]

    def groups(self, vector):
        return np.r_[self.original.groups(vector), self.component_terms(vector)[2]]

    def jacobian(self, vector):
        return sparse.vstack((self.original.jacobian(vector), self.component_terms(vector, derivatives=True)[1]),
                             format="csr")

    def selected_jacobian(self, vector, mask):
        base_count = len(self.original(vector))
        original = self.original.selected_jacobian(vector, mask[:base_count])
        extra = self.component_terms(vector, derivatives=True)[1][mask[base_count:]].toarray()
        return np.vstack((original, extra))


def linearized_points(problem, geometry, step):
    state_step = step[:problem.state_dimension].reshape(problem.initial.shape) * problem.scale
    return tuple(robot.points + robot.point_jacobians.apply(delta)
                 for robot, delta in zip(geometry.robots, state_step))


def violated_pairs(problem, geometry, step, slack):
    points = linearized_points(problem, geometry, step)
    cuts = np.cumsum([len(frame) for frame in points])[:-1]
    distances = np.split(problem.geometry.distances(np.concatenate(points)), cuts)
    additions, worst = [], 0.
    for key, (frame, phi) in enumerate(zip(points, distances)):
        safe = safe_distances(problem, key, np.arange(len(phi)))
        residual = (safe - phi) / problem.radius - slack[CONSTRAINT_GROUPS_PER_KEY * key]
        worst = max(worst, float(residual.max()))
        active = problem.query_ids[problem.schedule.columns[key]]
        queries = np.union1d(active, problem.collision_queries(phi - safe, active))
        queries = queries[residual[queries] > problem.feasibility_tolerance]
        if not len(queries):
            additions.append(frozenset())
            continue
        rows, components = problem.geometry.select(frame[queries])
        additions.append(frozenset(zip(queries[rows].tolist(), components.tolist())))
    return tuple(additions), worst


def separated_step(objective, original, surface, vector, lower, upper, trust, *,
                   penalty, regularization, solver):
    problem = original.problem
    geometry = problem.constraint_evaluation(vector)
    residual = objective(vector)
    jacobian = objective.jacobian(vector)
    gram = jacobian.T @ jacobian
    pairs = tuple(frozenset() for _ in problem.initial)
    constraints = original
    history, started = [], time.perf_counter()
    while True:
        before = JointEvaluation(residual, constraints(vector), constraints.groups(vector))
        step, slack = elastic_step(before, jacobian, constraints.jacobian(vector), surface.tangent_rows(vector),
            np.maximum(lower - vector, -trust), np.minimum(upper - vector, trust),
            penalty, regularization, solver, gram=gram)
        additions, worst = violated_pairs(problem, geometry, step, slack)
        updated = tuple(previous | new for previous, new in zip(pairs, additions))
        record = dict(iteration=len(history), components=sum(map(len, updated)),
                      worst_linear_point_violation=worst, seconds=time.perf_counter() - started)
        history.append(record)
        print('[key-separation]', record, flush=True)
        if worst <= problem.feasibility_tolerance:
            return step, slack, constraints, history
        if updated == pairs:
            # All discovered constraints are present. Let the outer nonlinear
            # line search/corrector handle their remaining curvature.
            return step, slack, constraints, history
        pairs = updated
        constraints = ComponentConstraints(original, pairs)
