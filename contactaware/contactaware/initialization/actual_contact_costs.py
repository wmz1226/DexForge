"""Force-closure costs at the robot pad points of each key."""

from dataclasses import dataclass

import numpy as np


MIN_SPATIAL_CLOSURE_CONTACTS = 3


def reference_terms(problem, index, columns, shared):
    del index
    return problem.anchor_reference_objectives(columns, shared[0], shared[2])


def actual_contact_geometry(problem, index, robot, columns):
    """Robot pad points and the object normals at those points, with their derivatives."""
    ids = problem.query_ids[columns]
    points = robot.points[ids]
    surface, derivatives = problem.geometry(points)
    point_rows = problem.embed(robot.point_jacobians[ids], index)
    normal_rows = np.einsum('nki,nij->nkj', derivatives[:, 4:7], point_rows)
    return points, surface[:, 4:7], point_rows, normal_rows


@dataclass(frozen=True)
class ActualContactCosts:
    problem: object
    scheduled: object
    weights: np.ndarray
    complete: tuple
    partial_force: object
    circular_force: object

    def __call__(self, index, geometry):
        terms = self.scheduled(index, geometry)
        columns = self.problem.schedule.columns[index]
        if not len(columns) or self.problem.distribution_weights[2] == 0.0:
            return terms
        actual = actual_contact_geometry(self.problem, index, geometry.robots[index], columns)
        # One or two point contacts cannot span a six-dimensional grasp wrench.
        complete = (len(columns) >= MIN_SPATIAL_CLOSURE_CONTACTS
                    and any(np.array_equal(columns, ids) for ids in self.complete))
        residual = self.circular_force(*actual) if complete else self.partial_force(*actual)
        scale = np.sqrt(self.weights[index] * self.problem.distribution_weights[2])
        return [*terms, (scale * residual[0], scale * residual[1])]


