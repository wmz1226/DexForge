"""Net-wrench residual of bounded friction-cone forces (bounded least squares) and its exact derivative."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from contactaware.solver.qp import SolverFailure

XYZ_DIM = 3
CONE_EDGES = 4
LEAST_SQUARES_TOLERANCE = 1.0e-12


def unit_vector_rows(values, rows):
    lengths = np.linalg.norm(values, axis=1)
    units = values / lengths[:, None]
    tangent = np.eye(XYZ_DIM) - units[:, :, None] * units[:, None, :]
    return units, np.einsum("nij,njk->nik", tangent / lengths[:, None, None], rows)


def cross_rows(left, right_rows):
    return np.cross(left[:, None], right_rows.swapaxes(1, 2)).swapaxes(1, 2)


def cone_force_rows(normals, normal_rows, friction):
    """Four-edge linearized friction cone and its derivative."""
    normals, normal_rows = unit_vector_rows(normals, normal_rows)
    basis = np.eye(XYZ_DIM)[np.argmin(np.abs(normals), axis=1)]
    tangent, tangent_rows = unit_vector_rows(np.cross(normals, basis),
                                            -cross_rows(basis, normal_rows))
    second = np.cross(normals, tangent)
    second_rows = cross_rows(normals, tangent_rows) - cross_rows(tangent, normal_rows)
    tangents = np.stack([tangent, second, -tangent, -second], axis=1)
    derivatives = np.stack([tangent_rows, second_rows, -tangent_rows, -second_rows], axis=1)
    forces = (normals[:, None] + friction * tangents) / CONE_EDGES
    rows = (normal_rows[:, None] + friction * derivatives) / CONE_EDGES
    force_count = len(normals) * CONE_EDGES
    return forces.reshape(force_count, XYZ_DIM), rows.reshape(force_count, XYZ_DIM, rows.shape[-1])


def wrench_rows(points, normals, point_rows, normal_rows, *, center, radius, friction):
    forces, derivatives = cone_force_rows(normals, normal_rows, friction)
    lever = np.repeat(points - center, CONE_EDGES, axis=0)
    lever_rows = np.repeat(point_rows, CONE_EDGES, axis=0)
    torques = np.cross(lever, forces) / radius
    torque_rows = (cross_rows(lever, derivatives) - cross_rows(forces, lever_rows)) / radius
    matrix = np.concatenate([forces, torques], axis=1).T
    rows = np.concatenate([derivatives, torque_rows], axis=1).transpose(1, 0, 2)
    return matrix, rows


def bounded_balance(matrix, rows, force_limit):
    """Solve the force block exactly; differentiate its free-variable KKT system."""
    solved = lsq_linear(matrix, np.zeros(len(matrix)), bounds=(1.0, force_limit + 1.0),
                        method="bvls", tol=LEAST_SQUARES_TOLERANCE)
    if not solved.success:
        raise SolverFailure(f"Contact-force least squares: {solved.message}")
    coefficients = solved.x
    residual = matrix @ coefficients
    direct = np.einsum("mnd,n->md", rows, coefficients)
    free = solved.active_mask == 0
    if not np.any(free):
        return residual, direct
    columns = matrix[:, free]
    hessian = columns.T @ columns
    rhs = np.einsum("mnd,m->nd", rows[:, free], residual) + columns.T @ direct
    coefficient_rows = -np.linalg.lstsq(hessian, rhs, rcond=None)[0]
    return residual, direct + columns @ coefficient_rows


@dataclass(frozen=True)
class ContactWrenchResidual:
    center: np.ndarray
    radius: float
    friction: float
    force_limit: float

    def __call__(self, points, normals, point_rows, normal_rows):
        matrix, rows = wrench_rows(points, normals, point_rows, normal_rows,
            center=self.center, radius=self.radius, friction=self.friction)
        return bounded_balance(matrix, rows, self.force_limit)
