"""Compact seed refinement: pad fit, anchor layout and palm terms on the object surface."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from contactaware.contact.object_model import object_center
from contactaware.initialization.key_geometry import GraspSeed
from contactaware.initialization.key_geometry import robot_kinematics
from contactaware.initialization.contact_priority_frame import FIRST_FRAME_PALM_WEIGHT_SCALE
from contactaware.contact.object_model import object_query_world, object_rotation
from contactaware.initialization.contact_priority_frame import solve_first_frame
from contactaware.initialization.seeds import SeedOps
from contactaware.solver.retarget import palm_aligned_qpos, palm_alignment_step
from contactaware.settings import make_config
from contactaware.solver.qp import QPSolverOptions
from contactaware.solver.tracking import palm_tracking_linearization
from contactaware.types import RetargetInputs
from contactaware.initialization.centered_pad import (
    centered_geometry_transform,
)
from comfree_warp.geometry import ObjectGeometry
from contactaware.solver.pose import _fit_base_pose, transform_qpos
from contactaware.solver.sparse_qp import make_joint_contact_qp
from contactaware.solver.grasp_sqp import ShrinkingTrustRegion, solve_joint_grasp
from contactaware.contact.topology import (
    DISTANCE_FEASIBILITY_MARGIN_M,
    MIN_PAIR_DISTANCE_RATIO,
    MAX_RELATIVE_DIRECTION_DEGREES,
    PairwiseTopologyConstraint,
)
from contactaware.solver.palm_relative_shape import palm_relative_shape_residual, unguided_shape_fingers

BASE_TRANSLATION_DIM = 3
MAX_KEY_SQP_ITERATIONS = 20
MM_PER_M = 1000.0
OBJECT_NONPENETRATION_GROUP = 0
SELF_COLLISION_GROUP = 1
PAD_SURFACE_FACING_GROUP = 2
TOPOLOGY_DISTANCE_GROUP = 3
TOPOLOGY_DIRECTION_GROUP = 4
SQP_STEP_TOLERANCE = 1.0e-5
TOPOLOGY_NORM_EPSILON_M = 1.0e-12
XYZ_DIM = 3


@dataclass(frozen=True)
class RobotAwareKeySettings:
    max_iterations: int = MAX_KEY_SQP_ITERATIONS
    minimum_pad_surface_facing: float = 0.5
    state_prior_weight: float = 0.001
    trust_shrink_trigger: float = 0.25
    trust_shrink_factor: float = 0.5
    minimum_trust_fraction: float = 1.0 / 64.0
    device: str = "cuda"


@dataclass(frozen=True)
class ImplicitKeyEvaluation:
    qpos: np.ndarray
    query_points: np.ndarray
    query_jacobians: np.ndarray
    query_normals: np.ndarray
    query_normal_jacobians: np.ndarray
    anchors: np.ndarray
    normals: np.ndarray
    anchor_jacobians: np.ndarray
    normal_jacobians: np.ndarray
    inequality: np.ndarray
    inequality_jacobian: np.ndarray
    metrics: dict


@dataclass(frozen=True)
class ObjectiveBlock:
    name: str
    residual: np.ndarray
    jacobian: np.ndarray

    @property
    def energy(self) -> float:
        return float(self.residual @ self.residual)


def key_frame_inputs(
    inputs: RetargetInputs,
    source_frame: int,
    anchor_columns: np.ndarray,
) -> RetargetInputs:
    """Build a one-frame, active-contact-only problem without changing the source."""
    columns = np.asarray(anchor_columns, dtype=np.int32)
    if columns.ndim != 1 or not len(columns):
        raise ValueError("Robot-aware contact key requires active anchor columns")
    if not np.all(inputs.anchors.mask[source_frame, columns]):
        raise ValueError("Every selected contact-key anchor must be active")
    anchors = replace(
        inputs.anchors,
        mask=np.ones((1, len(columns)), dtype=bool),
        anchor_pos_obj=inputs.anchors.anchor_pos_obj[source_frame:source_frame + 1, columns].copy(),
        anchor_normal_obj=inputs.anchors.anchor_normal_obj[source_frame:source_frame + 1, columns].copy(),
        finger_ids=inputs.anchors.finger_ids[columns].copy(),
        mano_vertex_ids=inputs.anchors.mano_vertex_ids[source_frame:source_frame + 1, columns].copy(),
    )
    return replace(
        inputs,
        mano_joints=inputs.mano_joints[source_frame:source_frame + 1].copy(),
        mano_vertices=inputs.mano_vertices[source_frame:source_frame + 1].copy(),
        obj_pose=inputs.obj_pose[source_frame:source_frame + 1].copy(),
        anchors=anchors,
    )


class ImplicitContactKeyProblem:
    """Optimize q on the same object surface used by hand/object collision."""

    def __init__(self, inputs, q_reference, query_ids, reference_points, args, settings, *, geometry):
        self.hand = inputs.hand
        self.q_reference = np.asarray(q_reference, dtype=np.float64).copy()
        self.query_ids = np.asarray(query_ids, dtype=np.int64).copy()
        self.reference = np.asarray(reference_points, dtype=np.float64).copy()
        self.settings = settings
        self.cfg = make_config(args)
        self.center_weight = float(args.anchor_center_weight)
        self.relative_weight = float(args.anchor_relative_weight)
        self.mano_joints_obj = _world_points_to_object(
            inputs.mano_joints[0], inputs.obj_pose[0]
        )
        self.shape_fingers = unguided_shape_fingers(
            self.hand,
            inputs.anchors.finger_ids,
        )
        self.obj = geometry
        self.center = object_center(inputs.obj.xml_path).astype(np.float64)
        self.radius = self.obj.bounding_radius(self.center)
        self.topology_reference = self._initial_contact_anchors()
        self.topology = PairwiseTopologyConstraint.from_reference(
            self.topology_reference,
            np.ones(len(self.reference), dtype=bool),
            max_direction_degrees=MAX_RELATIVE_DIRECTION_DEGREES,
        )
        self.scale = np.ones_like(self.q_reference)
        self.scale[:BASE_TRANSLATION_DIM] = self.radius
        self.body_groups = self._noncontact_body_groups()
        self.constraint_groups = self._constraint_group_ids()
        self._last_x = None
        self._last_value = None

    def _initial_contact_anchors(self) -> np.ndarray:
        robot = robot_kinematics(self.q_reference, hand=self.hand)
        values, _ = self.obj(robot.points[self.query_ids])
        return values[:, 1:4].astype(np.float64, copy=True)

    def _noncontact_body_groups(self) -> tuple[np.ndarray, ...]:
        body_ids = np.asarray(self.hand.query_points.body_ids)
        eligible = np.ones(len(body_ids), dtype=bool)
        eligible[self.query_ids] = False
        groups = []
        for body_id in np.unique(body_ids):
            ids = np.flatnonzero((body_ids == body_id) & eligible)
            if len(ids):
                groups.append(ids)
        return tuple(groups)

    def _constraint_group_ids(self) -> np.ndarray:
        contact_count = len(self.query_ids)
        topology_count = len(self.topology.pair_ids)
        counts = (
            (OBJECT_NONPENETRATION_GROUP, contact_count + len(self.body_groups)),
            (SELF_COLLISION_GROUP, len(self.hand.capsule_pairs)),
            (PAD_SURFACE_FACING_GROUP, contact_count),
            (TOPOLOGY_DISTANCE_GROUP, topology_count),
            (TOPOLOGY_DIRECTION_GROUP, topology_count),
        )
        return np.concatenate(
            [np.full(count, group, dtype=np.int32) for group, count in counts]
        )

    def qpos(self, x: np.ndarray) -> np.ndarray:
        return self.q_reference + self.scale * np.asarray(x, dtype=np.float64)

    def scaled_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.hand.lower.copy()
        upper = self.hand.upper.copy()
        lower[:self.hand.base_qpos_dim] = -np.inf
        upper[:self.hand.base_qpos_dim] = np.inf
        scaled_lower = (lower - self.q_reference) / self.scale
        scaled_upper = (upper - self.q_reference) / self.scale
        return scaled_lower, scaled_upper

    @property
    def feasibility_tolerance(self) -> float:
        return self.cfg.geometry_feasibility_tolerance_m / self.radius

    def constraint_violation(self, x: np.ndarray) -> float:
        inequality = self.evaluate(x).inequality
        return float(np.max(np.maximum(-inequality, 0.0)))

    def is_feasible(self, x: np.ndarray) -> bool:
        lower, upper = self.scaled_bounds()
        joint_violation = np.maximum(np.r_[lower-x, x-upper], 0.).max(initial=0.)
        return bool(self.constraint_violation(x) <= self.feasibility_tolerance
                    and joint_violation <= self.cfg.joint_limit_tolerance_rad)

    def evaluate(self, x: np.ndarray) -> ImplicitKeyEvaluation:
        coordinates = np.asarray(x, dtype=np.float64)
        if self._last_x is not None and np.array_equal(coordinates, self._last_x):
            return self._last_value
        qpos = self.qpos(coordinates)
        robot = robot_kinematics(qpos, hand=self.hand)
        contact_points = robot.points[self.query_ids]
        object_values, object_local_jac = self.obj(robot.points)
        contact_values = object_values[self.query_ids]
        contact_local_jac = object_local_jac[self.query_ids]
        contact_jac = self._chain(
            contact_local_jac,
            robot.point_jacobians[self.query_ids],
        )
        collision_ids = self._minimum_body_queries(object_values[:, 0])
        collision_jac = self._chain(
            object_local_jac[collision_ids, :1],
            robot.point_jacobians[collision_ids],
        )[:, 0]
        anchors = contact_values[:, 1:4]
        normals = contact_values[:, 4:7]
        anchor_jac = contact_jac[:, 1:4]
        normal_jac = contact_jac[:, 4:7]
        pad_normals = robot.pad_normals[self.query_ids]
        pad_normal_jac = robot.pad_jacobians[self.query_ids] * self.scale
        pad_surface_facing, pad_surface_facing_jac = self._pad_surface_facing(
            pad_normals,
            pad_normal_jac,
            normals,
            normal_jac,
        )
        topology, topology_jac = self._topology(anchors, anchor_jac)
        contact_gap = contact_values[:, 0] / self.radius
        contact_gap_jac = contact_jac[:, 0] / self.radius
        inequality = np.concatenate(
            (
                contact_gap,
                (object_values[collision_ids, 0] - self.cfg.hand_object_safe_distance)
                / self.radius,
                (robot.self_clearances - self.cfg.safe_distance) / self.radius,
                pad_surface_facing - self.settings.minimum_pad_surface_facing,
                topology,
            )
        )
        inequality_jac = np.concatenate(
            (
                contact_gap_jac,
                collision_jac / self.radius,
                robot.self_jacobians * self.scale / self.radius,
                pad_surface_facing_jac,
                topology_jac,
            )
        )
        metrics = self._metrics(
            contact_values[:, 0], object_values[:, 0], collision_ids,
            robot.self_clearances, pad_surface_facing, anchors,
        )
        value = ImplicitKeyEvaluation(
            qpos=qpos.copy(),
            query_points=contact_points.copy(),
            query_jacobians=(
                robot.point_jacobians[self.query_ids] * self.scale
            ).copy(),
            query_normals=pad_normals.copy(),
            query_normal_jacobians=pad_normal_jac.copy(),
            anchors=anchors.copy(),
            normals=normals.copy(),
            anchor_jacobians=anchor_jac.copy(),
            normal_jacobians=normal_jac.copy(),
            inequality=inequality,
            inequality_jacobian=inequality_jac,
            metrics=metrics,
        )
        self._last_x = coordinates.copy()
        self._last_value = value
        return value

    def _chain(self, local: np.ndarray, point_jac: np.ndarray) -> np.ndarray:
        return np.einsum("nki,nij->nkj", local, point_jac) * self.scale

    def _minimum_body_queries(self, distances: np.ndarray) -> np.ndarray:
        return np.asarray(
            [group[np.argmin(distances[group])] for group in self.body_groups],
            dtype=np.int64,
        )

    def _pad_surface_facing(self, pad_normals, pad_jac, normals, normal_jac):
        values = -np.einsum("ni,ni->n", pad_normals, normals)
        derivatives = -np.einsum("ni,nij->nj", normals, pad_jac)
        derivatives -= np.einsum("ni,nij->nj", pad_normals, normal_jac)
        return values, derivatives

    def _topology(self, anchors, anchor_jac):
        pairs = self.topology.pair_ids
        delta = anchors[pairs[:, 0]] - anchors[pairs[:, 1]]
        delta_jac = anchor_jac[pairs[:, 0]] - anchor_jac[pairs[:, 1]]
        distances = np.sqrt(
            np.einsum("ni,ni->n", delta, delta) + TOPOLOGY_NORM_EPSILON_M**2
        )
        units = delta / distances[:, None]
        reference_units = self.topology.reference_vectors / self.topology.reference_distances[:, None]
        # Match the scaled bound used by accept().
        distance = (
            distances
            - MIN_PAIR_DISTANCE_RATIO * self.topology.reference_distances
            - DISTANCE_FEASIBILITY_MARGIN_M
        ) / self.radius
        distance_jac = np.einsum("ni,nij->nj", units, delta_jac) / self.radius
        direction = (
            np.einsum("ni,ni->n", delta, reference_units)
            - self.topology.minimum_cosine * distances
        ) / self.radius
        direction_gradient = reference_units - self.topology.minimum_cosine * units
        direction_jac = np.einsum("ni,nij->nj", direction_gradient, delta_jac) / self.radius
        return np.concatenate((distance, direction)), np.concatenate((distance_jac, direction_jac))


    def _palm_terms(self, qpos):
        position, rotation, position_jac, rotation_jac = palm_tracking_linearization(
            self.hand,
            qpos,
            self.mano_joints_obj,
        )
        return (
            position,
            rotation,
            position_jac * self.scale,
            rotation_jac * self.scale,
        )


    def shape_terms(self, qpos):
        residual, jacobian = palm_relative_shape_residual(
            self.hand,
            qpos,
            self.mano_joints_obj,
            self.shape_fingers,
        )
        return residual, jacobian * self.scale

    def objective_blocks(self, x: np.ndarray) -> tuple[ObjectiveBlock, ...]:
        value = self.evaluate(x)
        contact_delta = value.query_points - value.anchors
        contact_jacobian = value.query_jacobians - value.anchor_jacobians
        normal_delta, normal_jacobian = self._normal_fit(value)
        center_delta, center_jacobian = self._anchor_center_terms(value)
        relative_delta, relative_jacobian = self._anchor_relative_terms(value)
        position, rotation, position_jacobian, rotation_jacobian = self._palm_terms(
            value.qpos
        )
        normal_weight = (
            self.cfg.contact_anchor_weight * self.cfg.contact_normal_weight_scale
        )
        return (
            # Keep the zero-weight row to preserve reduction order.
            ObjectiveBlock("force", np.zeros(1), np.zeros((1, len(x)))),
            _weighted_block(
                "contact_fit",
                contact_delta,
                contact_jacobian,
                self.cfg.contact_anchor_weight,
            ),
            _weighted_block(
                "contact_normal_fit",
                normal_delta,
                normal_jacobian,
                normal_weight,
            ),
            _weighted_block(
                "mano_center",
                center_delta,
                center_jacobian,
                self.center_weight,
            ),
            _weighted_block(
                "mano_relative",
                relative_delta,
                relative_jacobian,
                self.relative_weight / len(relative_delta),
            ),
            _weighted_block(
                "mano_palm_position",
                position,
                position_jacobian,
                FIRST_FRAME_PALM_WEIGHT_SCALE * self.cfg.palm_position_weight,
            ),
            _weighted_block(
                "mano_palm_rotation",
                rotation,
                rotation_jacobian,
                FIRST_FRAME_PALM_WEIGHT_SCALE * self.cfg.palm_rotation_weight,
            ),
            _weighted_block(
                "state_prior",
                np.asarray(x),
                np.eye(len(x)),
                self.settings.state_prior_weight,
            ),
        )


    def _anchor_center_terms(self, value):
        delta = (value.anchors.mean(axis=0) - self.reference.mean(axis=0))
        return delta / self.radius, value.anchor_jacobians.mean(axis=0) / self.radius

    def _anchor_relative_terms(self, value):
        relative = value.anchors - value.anchors.mean(axis=0)
        reference_relative = self.reference - self.reference.mean(axis=0)
        delta = (relative - reference_relative) / self.radius
        jacobian = (
            value.anchor_jacobians - value.anchor_jacobians.mean(axis=0)
        ) / self.radius
        return delta, jacobian

    def objective_components(self, x: np.ndarray) -> tuple[dict, np.ndarray]:
        blocks = self.objective_blocks(x)
        components = {block.name: block.energy for block in blocks}
        gradient = np.zeros_like(x, dtype=np.float64)
        for block in blocks:
            gradient += 2.0 * block.jacobian.T @ block.residual
        return components, gradient

    def objective(self, x: np.ndarray) -> float:
        components, _ = self.objective_components(x)
        return float(sum(components.values()))

    def least_squares_objective(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blocks = self.objective_blocks(x)
        return (
            np.concatenate([block.residual.ravel() for block in blocks]),
            np.vstack([block.jacobian.reshape(-1, len(x)) for block in blocks]),
        )

    def _normal_fit(self, value):
        radii = self.hand.query_surface_radii[self.query_ids]
        residual = radii[:, None] * (value.query_normals + value.normals)
        jacobian = radii[:, None, None] * (
            value.query_normal_jacobians + value.normal_jacobians
        )
        return residual, jacobian

    def _metrics(
        self,
        contact_phi,
        object_phi,
        collision_ids,
        self_clearance,
        pad_surface_facing,
        anchors,
    ):
        displacement = np.linalg.norm(anchors - self.reference, axis=1)
        pair_report = self.topology.report(anchors)
        return {
            "object_contact_gap_abs_mean_mm": float(
                np.abs(contact_phi).mean() * MM_PER_M
            ),
            "object_contact_gap_abs_max_mm": float(
                np.abs(contact_phi).max() * MM_PER_M
            ),
            "object_contact_penetration_max_mm": float(
                np.maximum(-contact_phi, 0.0).max() * MM_PER_M
            ),
            "mano_anchor_displacement_mean_mm": float(displacement.mean() * MM_PER_M),
            "mano_anchor_displacement_max_mm": float(displacement.max() * MM_PER_M),
            "all_query_object_penetration_max_mm": float(np.maximum(-object_phi, 0.0).max() * MM_PER_M),
            "noncontact_object_penetration_max_mm": float(
                np.maximum(-object_phi[collision_ids], 0.0).max() * MM_PER_M
            ),
            "self_clearance_min_mm": float(self_clearance.min() * MM_PER_M),
            "minimum_pad_surface_facing": float(pad_surface_facing.min()),
            "pairwise_topology": pair_report,
        }


def _weighted_block(name, residual, jacobian, weight) -> ObjectiveBlock:
    if weight < 0.0:
        raise ValueError(f"Objective weight must be nonnegative: {name}={weight}")
    scale = np.sqrt(weight)
    values = np.asarray(residual, dtype=np.float64)
    derivatives = np.asarray(jacobian, dtype=np.float64)
    return ObjectiveBlock(
        name,
        scale * values.ravel(),
        scale * derivatives.reshape(-1, derivatives.shape[-1]),
    )


@dataclass(frozen=True)
class ImplicitLeastSquaresObjective:
    problem: ImplicitContactKeyProblem

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.problem.least_squares_objective(x)[0]

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.problem.least_squares_objective(x)[1]


@dataclass(frozen=True)
class ImplicitPhysicalConstraints:
    problem: ImplicitContactKeyProblem

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.problem.evaluate(x).inequality

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.problem.evaluate(x).inequality_jacobian

    def groups(self, x: np.ndarray) -> np.ndarray:
        if len(self.problem.constraint_groups) != len(self(x)):
            raise RuntimeError("Contact-key constraint groups do not match constraint rows")
        return self.problem.constraint_groups.copy()

    def diagnostics(self, x: np.ndarray) -> list[dict]:
        value = self.problem.evaluate(x)
        return [{
            "surface_max_mm": value.metrics["object_contact_gap_abs_max_mm"],
            "penetration_mm": value.metrics["noncontact_object_penetration_max_mm"],
            "self_clearance_mm": value.metrics["self_clearance_min_mm"],
            "pad_surface_facing": value.metrics["minimum_pad_surface_facing"],
            "topology": value.metrics["pairwise_topology"]["all_constraints_satisfied"],
        }]

    def transfer_phase_initials(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        return np.asarray(x, dtype=np.float64), []


@dataclass(frozen=True)
class ImplicitContactSurface:
    problem: ImplicitContactKeyProblem

    def project_initial(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).copy()

    def tangent_rows(self, x: np.ndarray) -> np.ndarray:
        return np.empty((0, len(x)), dtype=np.float64)


@dataclass(frozen=True)
class InitialProblemCandidate:
    name: str
    rank: tuple[float, float]
    problem: ImplicitContactKeyProblem
    metrics: dict


def _select_initial_problem(
    inputs,
    centered_seed,
    query_ids,
    reference_points,
    args,
    settings,
):
    geometry = ObjectGeometry(inputs.obj.query_fn)
    candidates = [("centered_contact_seed", np.asarray(centered_seed, dtype=np.float64))]
    evaluated = [
        _initial_problem_candidate(
            name,
            qpos,
            inputs,
            query_ids,
            reference_points,
            args,
            settings,
            geometry,
        )
        for name, qpos in candidates
    ]
    selected = min(evaluated, key=lambda item: item.rank)
    report = {
        "policy": (
            "minimum hard-constraint violation, then minimum normalized "
            "contact gap"
        ),
        "selected": selected.name,
        "candidates": {
            candidate.name: candidate.metrics for candidate in evaluated
        },
    }
    return selected.problem, report


def _initial_problem_candidate(
    name,
    qpos,
    inputs,
    query_ids,
    reference_points,
    args,
    settings,
    geometry,
) -> InitialProblemCandidate:
    problem = ImplicitContactKeyProblem(
        inputs,
        qpos,
        query_ids,
        reference_points,
        args,
        settings,
        geometry=geometry,
    )
    origin = np.zeros_like(qpos)
    value = problem.evaluate(origin)
    violation = problem.constraint_violation(origin)
    normalized_gap = value.metrics["object_contact_gap_abs_mean_mm"] / (
        problem.radius * MM_PER_M
    )
    metrics = {
        "normalized_hard_constraint_violation": violation,
        "normalized_contact_gap_mean": normalized_gap,
        **value.metrics,
    }
    return InitialProblemCandidate(
        name=name,
        rank=(violation, normalized_gap),
        problem=problem,
        metrics=metrics,
    )


def _centered_seed(inputs, args, query_ids, points, *, axial_center, cfg):
    columns = np.arange(len(query_ids), dtype=np.int32)
    mapped_ids, transform = centered_geometry_transform(
        inputs,
        columns,
        axial_center=axial_center,
        cfg=cfg,
    )
    if not np.array_equal(mapped_ids, query_ids):
        raise RuntimeError("Centered query mapping changed between selection and seed construction")
    ops = SeedOps(palm_aligned_qpos, object_query_world, object_rotation, palm_alignment_step)
    qpos_world, solved_ids, initialization_report = solve_first_frame(
        args, inputs, initialization_ops=ops, geometry_transform=transform)
    if not np.array_equal(solved_ids, query_ids):
        raise RuntimeError("Fixed center-pad seed returned different query IDs")
    qpos = _world_to_object_qpos(inputs.hand, qpos_world, inputs.obj_pose[0])
    robot_points = inputs.hand.query_positions(qpos, query_ids)
    report = {
        "initialization": initialization_report,
        "strict_center_query_ids": query_ids.tolist(),
        "strict_center_seed_mean_mm": float(
            np.linalg.norm(robot_points - points, axis=1).mean() * MM_PER_M
        ),
        "object_frame_roundtrip_max_mm": _roundtrip_error_mm(
            inputs.hand,
            qpos_world,
            qpos,
            inputs.obj_pose[0],
        ),
    }
    return GraspSeed(qpos, query_ids, inputs.anchors.finger_ids.copy(), robot_points, None), report


def _solve_problem(problem, settings):
    origin = np.zeros_like(problem.q_reference)
    lower, upper = problem.scaled_bounds()
    trust = np.full(len(origin), problem.cfg.joint_trust_rad, dtype=np.float64)
    trust[:BASE_TRANSLATION_DIM] = problem.cfg.base_translation_trust_m
    trust[BASE_TRANSLATION_DIM:problem.hand.base_qpos_dim] = problem.cfg.base_rotation_trust_rad
    trust /= problem.scale
    solver = make_joint_contact_qp(
        QPSolverOptions(
            problem.cfg.qp_absolute_tolerance,
            problem.cfg.qp_relative_tolerance,
            problem.cfg.qp_validation_tolerance,
            problem.cfg.qp_maxiter,
        )
    )
    objective = ImplicitLeastSquaresObjective(problem)
    surface = ImplicitContactSurface(problem)
    physical = ImplicitPhysicalConstraints(problem)
    selected, result = solve_joint_grasp(
        objective,
        physical,
        surface,
        origin,
        lower,
        upper,
        trust,
        solver=solver,
        regularization=problem.cfg.min_hessian,
        tolerance=SQP_STEP_TOLERANCE,
        feasibility_tolerance=problem.feasibility_tolerance,
        max_iterations=settings.max_iterations,
        accept=lambda _: False,
        update_trust=ShrinkingTrustRegion(
            settings.trust_shrink_trigger,
            settings.trust_shrink_factor,
            settings.minimum_trust_fraction,
        ),
    )
    value = problem.evaluate(selected)
    report = {
        **result,
        "objective": problem.objective(selected),
        "objective_components": problem.objective_components(selected)[0],
        "normalized_constraint_violation": problem.constraint_violation(selected),
        "metrics": value.metrics,
    }
    return selected, report


def _world_points_to_object(points, object_pose):
    pose = np.asarray(object_pose, dtype=np.float64)
    rotation = Rotation.from_quat(pose[:4]).as_matrix()
    return (np.asarray(points, dtype=np.float64) - pose[4:]) @ rotation


def _world_to_object_qpos(hand, qpos_world, object_pose):
    pose = np.asarray(object_pose, dtype=np.float64)
    rotation = Rotation.from_quat(pose[:4]).as_matrix()
    hand.forward(qpos_world)
    palm_position = hand.data.xpos[hand.palm_body_id].copy()
    palm_rotation = hand.data.xmat[hand.palm_body_id].reshape(XYZ_DIM, XYZ_DIM).copy()
    target_position = rotation.T @ (palm_position - pose[4:])
    target_rotation = rotation.T @ palm_rotation
    return _fit_base_pose(hand, qpos_world, target_position, target_rotation, 1.0)


def _roundtrip_error_mm(hand, qpos_world, qpos_obj, object_pose):
    query_ids = np.arange(len(hand.query_points.local_pos), dtype=np.int32)
    expected = hand.query_positions(qpos_world, query_ids)
    reconstructed = transform_qpos(hand, qpos_obj, object_pose)
    actual = hand.query_positions(reconstructed, query_ids)
    return float(np.linalg.norm(actual - expected, axis=1).max() * MM_PER_M)

