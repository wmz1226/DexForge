"""Candidate selection for point queries against a collision target.

A query point is a zero-radius source. ``query_points`` returns its ``ranks`` nearest target
components with their ids and features (``collision_targets``: Gaussian spheres as point features;
a mesh's nearest triangle as its closest face, edge or vertex), in the selection order of
``collision_targets.nearest_components``. Callers evaluate distances and fuse candidates from these
features; ContactAware does so in float64.
"""

import warp as wp

from comfree_warp import collision_targets as targets


@wp.kernel
def query_points(target: targets.CollisionTarget, points: wp.array(dtype=wp.vec3), ranks: int,
                 anchors: wp.array2d(dtype=wp.vec4), axes: wp.array2d(dtype=wp.vec4),
                 ids: wp.array2d(dtype=int)):
    index = wp.tid()
    point = points[index]
    # Previous candidates only bound the search; the shared BVH still performs
    # the exact global selection, including deterministic ties.
    reach = float(targets.UNBOUNDED)
    if target.kind == targets.SPHERE_TARGET:
        warm_reach = float(-targets.UNBOUNDED)
        valid = int(0)
        for rank in range(targets.MAX_RANKS):
            if rank < ranks:
                seed = ids[index, rank]
                if seed >= 0 and seed < target.spheres.shape[0]:
                    sphere = target.spheres[seed]
                    distance = wp.length(point-wp.vec3(sphere[0], sphere[1], sphere[2]))-sphere[3]
                    warm_reach = wp.max(warm_reach, distance)
                    valid += 1
        if valid == ranks:
            reach = warm_reach + 1.0e-6
    nearest, components = targets.nearest_components(target, point, reach, ranks)
    for rank in range(targets.MAX_RANKS):
        if rank < ranks:
            anchor = wp.vec4(0.0)
            axis = wp.vec4(0.0)
            if components[rank] >= 0:
                anchor, axis = targets.component_feature(target, point, components[rank])
            anchors[index, rank] = anchor
            axes[index, rank] = axis
            ids[index, rank] = components[rank]
