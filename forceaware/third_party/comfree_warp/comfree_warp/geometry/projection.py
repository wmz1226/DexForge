"""Retraction onto the configured collision surface."""
import numpy as np

SURFACE_TOLERANCE_M = 1e-9
SURFACE_ITERATIONS = 24
SURFACE_BACKTRACKS = 16


def retract_fused_surface(geometry, points):
    """Damped distance-gradient projection, evaluated entirely in the soft field.

    A fused distance is not a signed Euclidean distance: neither its unit contact
    normal nor a hard sphere-union ray defines a projection onto its zero set.
    Retain the best residual if Newton stalls; the outer solver still checks the
    same surface equality and cannot mistake a stalled projection for feasibility.
    """
    projected = np.array(points, dtype=np.float64, copy=True)
    for _ in range(SURFACE_ITERATIONS):
        values, rows = geometry.distance_derivatives(projected)
        phi, gradient = values[:, 0], rows[:, 0]
        norm2 = np.einsum('ni,ni->n', gradient, gradient)
        active = np.flatnonzero((np.abs(phi) > SURFACE_TOLERANCE_M) & (norm2 > 1e-20))
        if not len(active):
            break
        step = -phi[active, None] * gradient[active] / norm2[active, None]
        accepted = np.zeros(len(active), dtype=bool)
        for backtrack in range(SURFACE_BACKTRACKS):
            pending = np.flatnonzero(~accepted)
            if not len(pending):
                break
            ids = active[pending]
            trial = projected[ids] + (0.5 ** backtrack) * step[pending]
            better = np.abs(geometry.distances(trial)) < np.abs(phi[ids])
            projected[ids[better]] = trial[better]
            accepted[pending[better]] = True
        if not accepted.any():
            break
    return projected


def project_surface(geometry, points):
    """Project to the configured field: hard feature/ray or soft distance gradient."""
    if geometry.fused:
        return retract_fused_surface(geometry, points)
    values, _ = geometry(points)
    distances, normals = values[:, 0], values[:, 4:7]
    projected = points - distances[:, None] * normals
    for index in np.flatnonzero(distances < 0.):
        origin, direction = points[index], normals[index]
        limit = geometry.bounding_radius(origin) - geometry.offset
        distance = geometry.clearance_retreat(origin, direction, -geometry.offset, limit)
        projected[index] = origin + distance * direction
    return projected
