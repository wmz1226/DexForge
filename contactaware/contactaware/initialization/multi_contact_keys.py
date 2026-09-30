"""Joint contact states with one implicit object-surface anchor per stable segment."""

from dataclasses import dataclass
from functools import cached_property, partial
from itertools import combinations

import numpy as np
from scipy import sparse

from contactaware.contact.phases import contact_mask_runs
from contactaware.contact.topology import (
    DISTANCE_FEASIBILITY_MARGIN_M, MAX_RELATIVE_DIRECTION_DEGREES,
    MIN_PAIR_DISTANCE_RATIO,
)

XYZ_DIM = 3
CONSTRAINT_GROUPS_PER_KEY = 4
NUMERICAL_LENGTH_M = 1.0e-12
MM_PER_M = 1000.0
ACCELERATION_NEIGHBORS = (-1, 0, 1)


@dataclass(frozen=True)
class ContactSchedule:
    frames: np.ndarray
    columns: tuple
    owners: np.ndarray


def select_contact_states(mask):
    """Keep every distinct contact-mask run and frame zero, without pruning subsets."""
    active = np.asarray(mask, dtype=bool)
    keys = {0: np.flatnonzero(active[0])}
    runs = contact_mask_runs(active) if active.shape[1] else ()
    for start, end, columns in runs:
        frame = 0 if start == 0 else (start + end) // 2
        keys[frame] = columns
    frames = np.asarray(sorted(keys), dtype=np.int64)
    columns = tuple(np.asarray(keys[frame], dtype=np.int64) for frame in frames)
    return ContactSchedule(frames, columns, anchor_owners(columns, active.shape[1]))


def anchor_owners(columns, count):
    owners = np.empty(count, dtype=np.int64)
    for column in range(count):
        candidates = [index for index, ids in enumerate(columns) if column in ids]
        owners[column] = max(candidates, key=lambda index: len(columns[index]))
    return owners


def select_transition_states(mask):
    """Contact-state keys plus frames t-1, t, t+1 around every contact change; anchors keep their owner key."""
    original = select_contact_states(mask)
    changes = np.flatnonzero(np.any(mask[1:] != mask[:-1], axis=1)) + 1
    neighbors = (changes[:, None] + np.asarray(ACCELERATION_NEIGHBORS)).ravel()
    neighbors = neighbors[(neighbors >= 0) & (neighbors < len(mask))]
    frames = np.union1d(original.frames, neighbors)
    columns = tuple(np.flatnonzero(mask[frame]) for frame in frames)
    owners = np.searchsorted(frames, original.frames[original.owners])
    return ContactSchedule(frames, columns, owners)


def embed_sparse_rows(local, state_index, state_dimension, total_dimension):
    """Rows of one key's state block as a CSR matrix without explicit zeros."""
    local = np.asarray(local, dtype=np.float64).reshape(-1, state_dimension)
    rows, columns = np.nonzero(local)
    return sparse.csr_matrix((local[rows, columns], (rows, columns + state_index * state_dimension)),
                             shape=(len(local), total_dimension))


def as_sparse_rows(rows):
    matrix = sparse.csr_matrix(rows)
    matrix.eliminate_zeros()
    return matrix


def embed_rows(local, state_index, state_dimension, total_dimension):
    result = np.zeros((*local.shape[:-1], total_dimension), dtype=np.float64)
    start = state_index * state_dimension
    result[..., start:start + state_dimension] = local
    return result


def shared_projections(robots, surfaces, query_ids, owners, scale):
    count, dimension = len(robots), len(scale)
    points, normals, point_rows, normal_rows = [], [], [], []
    for column, owner in enumerate(owners):
        query = query_ids[column]
        values, local = surfaces[owner]
        chained = local[query] @ robots[owner].point_jacobians[query] * scale
        points.append(values[query, 1:4])
        normals.append(values[query, 4:7])
        point_rows.append(embed_rows(chained[1:4], owner, dimension, count * dimension))
        normal_rows.append(embed_rows(chained[4:7], owner, dimension, count * dimension))
    shape = (len(owners), XYZ_DIM)
    return (np.asarray(points).reshape(shape), np.asarray(normals).reshape(shape),
            np.asarray(point_rows).reshape(*shape, count * dimension),
            np.asarray(normal_rows).reshape(*shape, count * dimension))


@dataclass(frozen=True)
class TopologyReference:
    pairs: np.ndarray
    vectors: np.ndarray
    distances: np.ndarray

    @classmethod
    def from_points(cls, points):
        pairs = np.asarray(tuple(combinations(range(len(points)), 2)), dtype=np.int64).reshape(-1, 2)
        vectors = points[pairs[:, 0]] - points[pairs[:, 1]]
        distances = np.linalg.norm(vectors, axis=1)
        if np.any(distances <= NUMERICAL_LENGTH_M):
            raise ValueError("The initial robot contact distribution contains coincident active anchors")
        return cls(pairs, vectors, distances)

    def linearize(self, points, rows, radius):
        left, right = self.pairs.T
        delta, jacobian = points[left] - points[right], rows[left] - rows[right]
        distances = np.sqrt(np.sum(delta * delta, axis=1) + NUMERICAL_LENGTH_M**2)
        units = delta / distances[:, None]
        original_units = self.vectors / self.distances[:, None]
        cosine = np.cos(np.deg2rad(MAX_RELATIVE_DIRECTION_DEGREES))
        spacing = (distances - MIN_PAIR_DISTANCE_RATIO * self.distances - DISTANCE_FEASIBILITY_MARGIN_M) / radius
        direction = (np.sum(delta * original_units, axis=1) - cosine * distances) / radius
        spacing_rows = np.einsum("ni,nij->nj", units, jacobian) / radius
        direction_rows = np.einsum("ni,nij->nj", original_units - cosine * units, jacobian) / radius
        return np.r_[spacing, direction], np.vstack((spacing_rows, direction_rows))


@dataclass(frozen=True)
class KeyEvaluation:
    states: np.ndarray
    anchors: np.ndarray
    normals: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    diagnostics: tuple


@dataclass(frozen=True)
class ConstraintRowBlock:
    count: int
    dimension: int
    linearize: object

    @cached_property
    def rows(self):
        return self.linearize()

    def __call__(self):
        return self.rows


@dataclass(frozen=True)
class KeyGeometry:
    states: np.ndarray
    robots: tuple
    shared: tuple
    constraints: np.ndarray
    row_factories: tuple
    groups: np.ndarray
    diagnostics: tuple

    @cached_property
    def rows(self):
        """Sparse constraint Jacobian; line-search values do not require it."""
        return sparse.vstack([as_sparse_rows(factory()) for factory in self.row_factories], format="csr")

    def selected_rows(self, mask):
        """Dense rows of the requested constraints, building only the blocks they intersect."""
        if mask.shape != self.constraints.shape:
            raise ValueError("Constraint row mask must match the constraint vector")
        if "rows" in self.__dict__:
            return self.rows[mask].toarray()
        selected, start = [], 0
        for block in self.row_factories:
            local = mask[start:start + block.count]
            if np.any(local):
                rows = block()
                selected.append((rows.toarray() if sparse.issparse(rows) else rows)[local])
            start += block.count
        dimension = self.row_factories[0].dimension
        return np.vstack(selected) if selected else np.empty((0, dimension))


def weighted(residual, jacobian, weight):
    scale = np.sqrt(weight)
    return scale * np.asarray(residual).ravel(), scale * jacobian.reshape(-1, jacobian.shape[-1])


class MultiContactProblem:
    def __init__(self, *, hand, cfg, settings, schedule, query_ids, initial,
                 references, joints_obj, geometry, center, kinematics, score,
                 palm_terms, shape_terms, finger_ids, distribution_weights, pre_contact,
                 topology_reference=None):
        self.hand, self.cfg, self.settings = hand, cfg, settings
        self.schedule, self.query_ids = schedule, np.array(query_ids, copy=True)
        self.initial, self.references = np.array(initial, copy=True), np.array(references, copy=True)
        self.joints_obj, self.finger_ids = joints_obj, finger_ids
        self.geometry, self.center, self.kinematics, self.score = geometry, center, kinematics, score
        self.palm_terms, self.shape_terms = palm_terms, shape_terms
        self.distribution_weights, self.pre_contact = distribution_weights, pre_contact
        self.radius = geometry.bounding_radius(center)
        self.scale = np.ones(hand.qpos_dim)
        self.scale[:XYZ_DIM] = self.radius
        self.dimension = initial.size
        self.topologies = self.make_topologies(topology_reference)
        self._geometry_x, self._geometry = None, None
        self._cache_x, self._cache = None, None
        self.collision_working_set = tuple(np.empty(0, dtype=np.int64) for _ in initial)
        self._state_values, self._state_robots, self._state_surfaces = None, None, None


    def query_surfaces(self, robots):
        values, derivatives = self.geometry.compact(np.concatenate([robot.points for robot in robots]))
        sizes = np.cumsum([len(robot.points) for robot in robots])[:-1]
        return tuple(zip(np.split(values, sizes), derivatives.split(sizes)))

    def state_geometry(self, states):
        """Reuse only exactly unchanged FK/surfaces during constraint correction."""
        if self._state_values is None:
            robots = tuple(self.kinematics(state) for state in states)
            surfaces = self.query_surfaces(robots)
        else:
            changed = np.flatnonzero(np.any(states != self._state_values, axis=1))
            robots, surfaces = list(self._state_robots), list(self._state_surfaces)
            replacements = tuple(self.kinematics(states[index]) for index in changed)
            replacement_surfaces = self.query_surfaces(replacements) if len(changed) else ()
            for index, robot, surface in zip(changed, replacements, replacement_surfaces):
                robots[index], surfaces[index] = robot, surface
        self._state_values = np.array(states, copy=True)
        self._state_robots, self._state_surfaces = tuple(robots), tuple(surfaces)
        return self._state_robots, self._state_surfaces

    def make_topologies(self, reference=None):
        if reference is not None:
            self.topology_reference = np.array(reference, copy=True)
            return tuple(TopologyReference.from_points(self.topology_reference[columns]) for columns in self.schedule.columns)
        robots = tuple(self.kinematics(q) for q in self.initial)
        surfaces = self.query_surfaces(robots)
        points, _, _, _ = shared_projections(robots, surfaces, self.query_ids, self.schedule.owners, self.scale)
        self.topology_reference = points
        return tuple(TopologyReference.from_points(points[columns]) for columns in self.schedule.columns)

    def states(self, vector):
        return self.initial + np.asarray(vector).reshape(self.initial.shape) * self.scale

    def bounds(self):
        lower, upper = self.hand.lower.copy(), self.hand.upper.copy()
        lower[:self.hand.base_qpos_dim], upper[:self.hand.base_qpos_dim] = -np.inf, np.inf
        return ((lower - self.initial) / self.scale).ravel(), ((upper - self.initial) / self.scale).ravel()

    def trust(self):
        values = np.full(self.hand.qpos_dim, self.cfg.joint_trust_rad)
        values[:XYZ_DIM] = self.cfg.base_translation_trust_m
        values[XYZ_DIM:self.hand.base_qpos_dim] = self.cfg.base_rotation_trust_rad
        return np.tile(values / self.scale, len(self.initial))

    @property
    def feasibility_tolerance(self):
        return self.cfg.geometry_feasibility_tolerance_m / self.radius

    def embed(self, local, index):
        return embed_rows(local * self.scale, index, self.hand.qpos_dim, self.dimension)

    def shared_geometry(self, vector, robots, surfaces):
        del vector
        return shared_projections(robots, surfaces, self.query_ids, self.schedule.owners, self.scale)

    def constraint_evaluation(self, vector):
        """Constraint values and rows only; force-closure solves belong to the objective."""
        if self._geometry_x is not None and np.array_equal(vector, self._geometry_x):
            return self._geometry
        states = self.states(vector)
        robots, surfaces = self.state_geometry(states)
        shared = self.shared_geometry(vector, robots, surfaces)
        inequalities, groups, reports = [], [], []
        for index, robot in enumerate(robots):
            physical, category, report = self.physical(index, robot, surfaces[index], shared)
            inequalities.extend(physical)
            groups.extend(category)
            reports.append(report)
        value = KeyGeometry(states, robots, shared,
            np.concatenate([x[0] for x in inequalities]),
            tuple(ConstraintRowBlock(len(values), self.dimension, rows) for values, rows in inequalities),
            np.concatenate(groups), tuple(reports))
        self._geometry_x, self._geometry = np.array(vector, copy=True), value
        self._surfaces = surfaces
        return value

    def prepare_iteration(self, vector):
        """Keep discovered link contacts across SQP steps; trial points add no rows."""
        self.constraint_evaluation(vector)
        discovered = tuple(self.collision_queries(surface[0][:, 0], self.query_ids[columns])
            for surface, columns in zip(self._surfaces, self.schedule.columns))
        updated = tuple(np.union1d(previous, found)
                        for previous, found in zip(self.collision_working_set, discovered))
        if all(np.array_equal(a, b) for a, b in zip(updated, self.collision_working_set)):
            return
        self.collision_working_set = updated
        self._geometry_x, self._cache_x = None, None


    def anchor_reference_objectives(self, columns, points, point_rows):
        center_weight, relative_weight, _ = self.distribution_weights
        reference = self.references[columns]
        centroid = (points.mean(axis=0) - reference.mean(axis=0)) / self.radius
        center_rows = point_rows.mean(axis=0) / self.radius
        relative = (points - points.mean(axis=0) - reference + reference.mean(axis=0)) / self.radius
        relative_rows = (point_rows - point_rows.mean(axis=0)) / self.radius
        return [weighted(centroid, center_rows, center_weight),
            weighted(relative, relative_rows, relative_weight / len(columns))]


    def collision_queries(self, phi, contact_ids):
        bodies = np.asarray(self.hand.query_points.body_ids)
        eligible = np.ones(len(bodies), dtype=bool)
        eligible[contact_ids] = False
        groups = [np.flatnonzero(eligible & (bodies == body)) for body in np.unique(bodies)]
        return np.asarray([ids[np.argmin(phi[ids])] for ids in groups if len(ids)], dtype=np.int64)

    def physical(self, index, robot, surface, shared):
        columns = self.schedule.columns[index]
        ids = self.query_ids[columns]
        values, local = surface
        collision_ids = np.union1d(self.collision_queries(values[:, 0], ids),
                                   self.collision_working_set[index])
        checked = np.r_[ids, collision_ids]
        frame = self.schedule.frames[index]
        safe = np.r_[np.zeros(len(ids)), self.pre_contact(frame, collision_ids)]
        object_term = ((values[checked, 0] - safe) / self.radius,
                       partial(self.object_constraint_rows, index, robot, local, checked))
        self_term = ((robot.self_clearances - self.cfg.safe_distance) / self.radius,
                     partial(self.normalized_rows, robot.self_jacobians, index))
        contact_terms, facing = self.contact_constraints(index, robot, shared)
        terms = [object_term, self_term, *contact_terms]
        categories = [np.full(len(term[0]), CONSTRAINT_GROUPS_PER_KEY * index + group, dtype=np.int64)
                      for group, term in enumerate(terms)]
        gap = np.linalg.norm(robot.points[ids] - shared[0][columns], axis=1)
        report = {"source_frame": int(self.schedule.frames[index]), "active_segments": columns.tolist(),
                  "contact_max_mm": float(gap.max(initial=0.0) * MM_PER_M),
                  "penetration_mm": float(np.maximum(-values[:, 0], 0.0).max() * MM_PER_M),
                  "self_clearance_mm": float(robot.self_clearances.min() * MM_PER_M),
                  "pad_surface_facing": float(facing.min()) if len(facing) else None}
        return terms, categories, report

    def normalized_rows(self, rows, index):
        return embed_sparse_rows(rows * self.scale / self.radius, index, self.hand.qpos_dim, self.dimension)

    def object_constraint_rows(self, index, robot, local, checked):
        rows = np.einsum("ni,nij->nj", local[checked, 0], robot.point_jacobians[checked])
        return self.normalized_rows(rows, index)

    def contact_constraints(self, index, robot, shared):
        columns = self.schedule.columns[index]
        points, normals, point_rows, _ = (item[columns] for item in shared)
        ids = self.query_ids[columns]
        pad = robot.pad_normals[ids]
        facing = -np.einsum("ni,ni->n", pad, normals)
        # Normal switches between object components require a soft alignment penalty.
        topology, topology_rows = self.topologies[index].linearize(points, point_rows, self.radius)
        pairs = len(self.topologies[index].pairs)
        return [(topology[:pairs], lambda: topology_rows[:pairs]),
                (topology[pairs:], lambda: topology_rows[pairs:])], facing


@dataclass(frozen=True)
class MultiObjective:
    problem: MultiContactProblem

    def __call__(self, vector):
        return self.problem.evaluate(vector).residual

    def jacobian(self, vector):
        return self.problem.evaluate(vector).jacobian


@dataclass(frozen=True)
class MultiConstraints:
    problem: MultiContactProblem

    def __call__(self, vector):
        return self.problem.constraint_evaluation(vector).constraints

    def jacobian(self, vector):
        return self.problem.constraint_evaluation(vector).rows

    def selected_jacobian(self, vector, mask):
        return self.problem.constraint_evaluation(vector).selected_rows(mask)

    def groups(self, vector):
        return self.problem.constraint_evaluation(vector).groups

    def diagnostics(self, vector):
        return list(self.problem.constraint_evaluation(vector).diagnostics)

    def transfer_phase_initials(self, vector):
        return np.asarray(vector), []


