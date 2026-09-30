"""Circular-cone force-closure margin under zero net wrench and a unit normal-force budget."""

from dataclasses import dataclass
from functools import partial
import time
from typing import Optional

import numpy as np
from scipy import sparse

XYZ_DIM = 3
TANGENT_DIM = XYZ_DIM - 1
CONE_DIM = TANGENT_DIM + 1
WRENCH_DIM = 6
FORCE_SOLVER_TOLERANCE = 1.e-9


@dataclass(frozen=True)
class ForceMarginResult:
    status: str
    reserve: Optional[float]
    forces: Optional[np.ndarray]
    wrench_rank: int
    max_balance_error: Optional[float]
    minimum_cone_slack: Optional[float]
    iterations: int
    seconds: float
    point_gradient: Optional[np.ndarray] = None
    normal_gradient: Optional[np.ndarray] = None


def force_balance_rows(points, normals, center, radius):
    count = len(points)
    rows = np.zeros((WRENCH_DIM + 1, XYZ_DIM * count + 1))
    for index, (point, normal) in enumerate(zip(points, normals)):
        columns = slice(XYZ_DIM * index, XYZ_DIM * (index + 1))
        lever = (point - center) / radius
        rows[:XYZ_DIM, columns] = np.eye(XYZ_DIM)
        rows[XYZ_DIM:WRENCH_DIM, columns] = np.cross(lever, np.eye(XYZ_DIM)).T
        rows[WRENCH_DIM, columns] = normal
    return rows


def circular_cone_rows(normals, friction, tangent_frames):
    dimension = XYZ_DIM * len(normals) + 1
    blocks = []
    for index, (normal, tangents) in enumerate(zip(normals, tangent_frames)):
        block = np.zeros((CONE_DIM, dimension))
        columns = slice(XYZ_DIM * index, XYZ_DIM * (index + 1))
        block[0, columns], block[0, -1] = -friction * normal, 1.
        block[1:, columns] = -tangents
        blocks.append(block)
    return np.vstack(blocks)


def margin_sensitivity(forces, normals, dual, radius, friction, tangent_frames):
    """Envelope derivative of the optimized reserve, including force redistribution."""
    cone_dual = dual[WRENCH_DIM + 1:].reshape(-1, CONE_DIM)
    tangent_dual = np.einsum('nki,nk->ni', tangent_frames, cone_dual[:, 1:])
    normal_force = np.einsum('ni,ni->n', normals, forces)
    points = -np.cross(forces, dual[XYZ_DIM:WRENCH_DIM]) / radius
    coefficient = (friction * cone_dual[:, 0] - dual[WRENCH_DIM]
                   - np.einsum('ni,ni->n', tangent_dual, normals))
    normals_gradient = coefficient[:, None] * forces - normal_force[:, None] * tangent_dual
    return points, normals_gradient


def balance_coordinates(balance):
    """Whiten equality rows without dropping dependent or inconsistent equations."""
    left, singular, _ = np.linalg.svd(balance, full_matrices=True)
    cutoff = np.finfo(balance.dtype).eps * max(balance.shape) * singular[0]
    scale = np.ones(len(balance))
    scale[:len(singular)] = np.divide(1., singular, out=np.ones_like(singular), where=singular > cutoff)
    return scale[:, None] * left.T


def solve_force_margin(points, normals, *, center, radius, friction,
                       solver_factory, settings_factory, equality_cone, circular_cone):
    """Maximize normalized friction reserve; positive reserve and full rank certify force closure."""
    started = time.perf_counter()
    balance = force_balance_rows(points, normals, center, radius)
    # Two tangential coordinates; a third would add a redundant cone direction.
    tangent_frames = np.linalg.svd(normals[:, None, :], full_matrices=True)[2][:, 1:, :]
    cone_rows = circular_cone_rows(normals, friction, tangent_frames)
    equality_transform = balance_coordinates(balance)
    balance_target = np.r_[np.zeros(WRENCH_DIM), 1.]
    rows = sparse.csc_matrix(np.vstack([equality_transform @ balance, cone_rows]))
    bounds = np.r_[equality_transform @ balance_target, np.zeros(len(cone_rows))]
    dimension = balance.shape[1]
    objective = np.r_[np.zeros(dimension - 1), -1.]
    settings = settings_factory()
    settings.verbose, settings.max_threads = False, 1
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = FORCE_SOLVER_TOLERANCE
    cones = [equality_cone(WRENCH_DIM + 1)] + [circular_cone(CONE_DIM) for _ in points]
    solver = solver_factory(sparse.csc_matrix((dimension, dimension)), objective, rows, bounds, cones, settings)
    result = solver.solve()
    rank = int(np.linalg.matrix_rank(balance[:WRENCH_DIM, :-1]))
    seconds = time.perf_counter() - started
    # AlmostSolved meets Clarabel's reduced accuracy tolerances; its point and duals remain usable.
    if str(result.status) not in ('Solved', 'AlmostSolved'):
        return ForceMarginResult(str(result.status), None, None, rank, None, None, result.iterations, seconds)
    vector = np.asarray(result.x)
    forces = vector[:-1].reshape(-1, XYZ_DIM)
    normal_force = np.einsum('ni,ni->n', normals, forces)
    tangent = forces - normal_force[:, None] * normals
    slack = friction * normal_force - np.linalg.norm(tangent, axis=1) - vector[-1]
    error = balance @ vector - balance_target
    dual = np.asarray(result.z).copy()
    dual[:WRENCH_DIM + 1] = equality_transform.T @ dual[:WRENCH_DIM + 1]
    point_gradient, normal_gradient = margin_sensitivity(
        forces, normals, dual, radius, friction, tangent_frames)
    return ForceMarginResult('Solved', float(vector[-1]), forces, rank,
        float(np.abs(error).max()), float(slack.min()), result.iterations, seconds,
        point_gradient, normal_gradient)


def make_force_margin_solver(*, center, radius, friction):
    import clarabel
    return partial(solve_force_margin, center=np.asarray(center), radius=radius, friction=friction,
        solver_factory=clarabel.DefaultSolver, settings_factory=clarabel.DefaultSettings,
        equality_cone=clarabel.ZeroConeT, circular_cone=clarabel.SecondOrderConeT)
