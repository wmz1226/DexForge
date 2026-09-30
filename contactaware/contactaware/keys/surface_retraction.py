"""Retract shared anchors onto the selected object's zero-distance surface."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorSurfaceRetraction:
    problem: object

    def project_initial(self, vector):
        problem = self.problem
        points = problem.anchor_coordinates(vector)
        projected = problem.geometry.project_surface(points)
        result = np.array(vector, copy=True)
        result[problem.state_dimension:] = ((projected - problem.initial_anchors) / problem.radius).ravel()
        return result

    def tangent_rows(self, vector):
        # The surface equalities are already QP constraint rows.
        return np.empty((0, len(vector)))


def solve_surface_keys(objective, constraints, surface, initial, lower, upper, trust, *, solve, **kwargs):
    return solve(objective, constraints, AnchorSurfaceRetraction(constraints.problem),
                 initial, lower, upper, trust, **kwargs)
