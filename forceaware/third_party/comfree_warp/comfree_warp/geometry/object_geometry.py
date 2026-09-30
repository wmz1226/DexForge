"""Object geometry for the key and trajectory solvers: values, Jacobians and separation components."""

from __future__ import annotations

from .source_query import DenseRows


class ObjectGeometry:
    """``[distance, surface point, outward normal]`` of points and their 7x3 Jacobians.

    Every mode uses the simulator's BVH, contact-fusion functions and VJP.
    Derivative blocks are requested only for selected solver rows.
    """

    def __init__(self, query):
        self.query = self.query_fn = query
        self.offset = query.distance_offset

    def distances(self, points):
        return self.query.distance_bounds(points)[:, 0]

    def __call__(self, points):
        values, derivatives = self.compact(points)
        return values, derivatives[:]

    def compact(self, points):
        return self.query.solver_compact(points)

    def project_surface(self, points):
        from .projection import project_surface
        return project_surface(self, points)

    def distance_derivatives(self, points):
        features = self.query.distance_features(points)
        return features[:, :1], DenseRows(features[:, None, 1:4])

    def select(self, points):
        """Query constraints: fixed nearest features in hard, the fused field in soft."""
        return self.query.select_constraints(points, key=True)

    def component_rows(self, points, components):
        return self.query.constraint_values(points, components, key=True)

    @property
    def fused(self):
        return self.query.fused

    def bounding_radius(self, center):
        return self.query.bounding_radius(center)

    def clearance_retreat(self, origin, direction, padding, limit):
        return self.query.ray_exit(origin, direction, padding, limit)
