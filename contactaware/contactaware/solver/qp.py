"""QP problem and result types with validation, shared by the sparse QP solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


class SolverFailure(RuntimeError):
    """A numerical solver returned no usable solution; iterative callers stop and keep their best iterate."""


@dataclass(frozen=True)
class QPSolverOptions:
    absolute_tolerance: float
    relative_tolerance: float
    validation_tolerance: float
    max_iterations: int


@dataclass(frozen=True)
class QPProblem:
    hessian: np.ndarray
    gradient: np.ndarray
    rows: np.ndarray
    lower_rows: np.ndarray
    upper_rows: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class QPResult:
    solution: np.ndarray
    max_violation: float


def validate_problem(problem: QPProblem) -> None:
    variable_count = problem.gradient.size
    expected = {
        "hessian": (variable_count, variable_count),
        "rows": (problem.lower_rows.size, variable_count),
        "upper_rows": problem.lower_rows.shape,
        "lower": (variable_count,),
        "upper": (variable_count,),
    }
    arrays = {
        "hessian": problem.hessian,
        "rows": problem.rows,
        "upper_rows": problem.upper_rows,
        "lower": problem.lower,
        "upper": problem.upper,
    }
    malformed = [name for name, shape in expected.items() if arrays[name].shape != shape]
    if malformed:
        raise ValueError(f"Malformed QP arrays: {malformed}")
    hessian = problem.hessian.data if sparse.issparse(problem.hessian) else problem.hessian
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(problem.gradient)):
        raise ValueError("QP objective contains non-finite values")


def primal_violation_details(
    problem: QPProblem,
    solution: np.ndarray,
) -> dict[str, float]:
    if not np.all(np.isfinite(solution)):
        raise SolverFailure("QP solver returned a non-finite solution")
    details = {
        "lower_bound": positive_max(problem.lower - solution),
        "upper_bound": positive_max(solution - problem.upper),
        "lower_linear_row": 0.0,
        "upper_linear_row": 0.0,
    }
    if problem.lower_rows.size:
        row_values = problem.rows @ solution
        details["lower_linear_row"] = positive_max(problem.lower_rows - row_values)
        details["upper_linear_row"] = positive_max(row_values - problem.upper_rows)
    return details


def positive_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return max(float(np.max(values)), 0.0)
