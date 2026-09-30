"""Object-frame adapter to the simulator's shared collision target, BVH and VJP."""
from __future__ import annotations
from pathlib import Path
import mujoco
import numpy as np
import comfree_warp
from comfree_warp import collision_targets
from comfree_warp.collision_config import CollisionConfig, configure_collision
from .source_query import SourceQuery, QueryRows
from .sphere_bvh import make_bvh_sphere_selector

OBJECT_BODY = "obj"
XYZ_DIM = 3
VALUE_COLUMNS = 7
RAY_ITERATIONS = 256
RAY_EPS_M = 1e-9
FUSED_CONSTRAINT = -1


class ObjectCollisionQuery:
    """Object-frame point queries; GPU candidate batches are recorded per size and own their buffers."""

    def __init__(self, xml_path: Path, *, topk: int = 1, distance_offset: float = 0.,
                 config: CollisionConfig | None = None, device="cuda"):
        config = config or CollisionConfig(topk, distance_offset)
        topk, distance_offset = config.contact_topk, config.distance_offset
        device_model, cpu_model = comfree_warp.load_model(str(xml_path))
        configure_collision(device_model, config)
        body = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        batches = device_model.gaussian_collision
        batches = batches if isinstance(batches, tuple) else (batches,)
        batch = next((b for b in batches if int(b.target_body_id) == body), None)
        if batch is None:
            raise ValueError(f"Scene has no collision target on body {OBJECT_BODY!r}: {xml_path}")
        meshes = getattr(device_model, "mesh_targets", {}) or {}
        self._resources = collision_targets.TargetResources(batch, meshes.get(body), device)
        self.kind = "mesh" if body in meshes else "spheres"
        self.topk = int(topk)
        self.fusion_boundary_fallback = float(batch.threshold)
        self.distance_offset = float(distance_offset)
        self.support_spheres = batch.target_spheres.numpy()
        self._points = SourceQuery(self._resources.target, config, batch.threshold, device)
        self.ranks = self._points.ranks
        self._components = {}
        self._component_features = []
        self._rays = (make_bvh_sphere_selector(self.support_spheres[:, :3], self.support_spheres[:, 3])
                      if self.exact_spheres else None)

    @property
    def exact_spheres(self) -> bool:
        """Nearest Gaussian over the whole cloud (``topk = 1`` on a Gaussian target)."""
        return self.kind == "spheres" and self.topk == 1

    @property
    def fused(self) -> bool:
        return self.topk > 1

    def solver_compact(self, points):
        return self.compact(points)

    def candidates(self, points):
        return self._points.candidates(points)

    def distance_bounds(self, points, source_radii=None):
        """Configured distance and a conservative bound used only to schedule queries."""
        return self._points.evaluate(points, bounds=True, source_radii=source_radii)

    def distance_features(self, points, source_radii=None):
        """Distance and its true position gradient, from the simulator's VJP."""
        return self._points.evaluate(points, distance=True, source_radii=source_radii)[:, :4]

    def features(self, points, source_radii=None):
        """``(N, 28)``: distance, surface point, outward normal, then their 7x3 Jacobians."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, XYZ_DIM)
        if not len(points):
            return np.empty((0, VALUE_COLUMNS * 4))
        values, rows = self.compact(points, source_radii)
        return np.concatenate([values, rows[:].reshape(len(points), -1)], axis=1)

    def compact(self, points, source_radii=None):
        points = np.array(points, dtype=np.float64, copy=True).reshape(-1, XYZ_DIM)
        radii = (None if source_radii is None else
                 np.array(np.broadcast_to(source_radii, (len(points),)), copy=True))
        return self._points.evaluate(points, source_radii=radii), QueryRows(self._points, points, radii)

    def query_clouds(self, clouds, source_radii=None):
        """Batch separate clouds through one query, preserving per-point results.

        Each returned pair contains a cloud's values and lazy position Jacobians.
        This is a layout adapter; it does not aggregate independent point distances
        into a simulation contact slot. Omitted radii are zero for every point.
        """
        clouds = tuple(np.asarray(p, dtype=np.float64).reshape(-1, XYZ_DIM) for p in clouds)
        if not clouds:
            return ()
        sizes = [len(p) for p in clouds]
        radii = None
        if source_radii is not None:
            if np.isscalar(source_radii):
                radii = source_radii
            else:
                if len(source_radii) != len(clouds):
                    raise ValueError("One radius array per cloud is required")
                radii = np.concatenate([np.broadcast_to(r, (n,)) for r, n in zip(source_radii, sizes)])
        values, rows = self.compact(np.concatenate(clouds, axis=0), source_radii=radii)
        boundaries = np.cumsum(sizes)[:-1]
        return tuple(zip(np.split(values, boundaries), rows.split(boundaries)))

    def __call__(self, points, source_radii=None):
        """Configured signed distance, contact position and inward normal."""
        values = self._points.evaluate(points, source_radii=source_radii)
        return values[:, 0], values[:, 1:4], -values[:, 4:7]

    def select_constraints(self, points, *, key=False):
        """One geometry constraint per point, using this query's contact mode.

        Hard keeps the discovered nearest feature fixed for separation cuts.
        Soft records the fused field, reselecting candidates and differentiating
        their weights whenever the constraint is evaluated. Individual soft
        candidates are never separate inequalities.
        ``key`` is retained for caller compatibility; every query shares selection.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, XYZ_DIM)
        if not len(points):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if self.fused:
            return np.arange(len(points), dtype=np.int64), np.full(len(points), FUSED_CONSTRAINT, dtype=np.int64)
        anchors, axes, ids = self.candidates(points)
        rows, ranks = np.nonzero(ids[:, :self.topk] >= 0)
        if self.kind == "spheres":
            return rows.astype(np.int64), ids[rows, ranks].astype(np.int64)
        selected = np.empty(len(rows), dtype=np.int64)
        base = len(self.support_spheres)
        for index, (row, rank) in enumerate(zip(rows, ranks)):
            feature = np.concatenate([anchors[row, rank], axes[row, rank]])
            key_bytes = feature.tobytes()
            component = self._components.get(key_bytes)
            if component is None:
                component = self._components[key_bytes] = base + len(self._component_features)
                self._component_features.append(feature)
            selected[index] = component
        return rows.astype(np.int64), selected

    def _features_of(self, components):
        base = len(self.support_spheres)
        anchors, axes = np.zeros((len(components), 4)), np.zeros((len(components), 4))
        spheres = components < base
        anchors[spheres] = self.support_spheres[components[spheres]]
        axes[spheres, 3] = collision_targets.POINT_FEATURE
        for index in np.flatnonzero(~spheres):
            feature = self._component_features[components[index] - base]
            anchors[index], axes[index] = feature[:4], feature[4:]
        return anchors, axes

    def constraint_values(self, points, components, *, key=False):
        """Constraint distance and its actual position derivative in the selected mode."""
        components = np.asarray(components, dtype=np.int64).reshape(-1)
        points = np.asarray(points, dtype=np.float64).reshape(-1, XYZ_DIM)
        if components.shape != (len(points),):
            raise ValueError("One component per point is required")
        if not len(points):
            return np.empty(0), np.empty((0, XYZ_DIM))
        if self.fused:
            if np.any(components != FUSED_CONSTRAINT):
                raise ValueError("Soft geometry constraints must reference the fused field")
            features = self.distance_features(points)
            return features[:, 0], features[:, 1:4]
        if np.any(components < 0):
            raise ValueError("Hard geometry constraints require a feature id")
        anchors, axes = self._features_of(components)
        features = self._points.evaluate(points, distance=True,
                                         fixed=(anchors[:, None], axes[:, None]))
        return features[:, 0], features[:, 1:4]

    def bounding_radius(self, center):
        """Enclose the unshifted target field, including smooth-min expansion."""
        spheres = self.support_spheres.astype(np.float64)
        center = np.asarray(center, dtype=np.float64)
        radius = float(np.max(np.linalg.norm(spheres[:, :3] - center, axis=1) + spheres[:, 3]))
        return radius + self._smooth_bound_padding()

    def _smooth_bound_padding(self):
        if self.kind == "spheres" and self.fused:
            from comfree_warp.smooth_contact import WIDTH
            # D >= min(d) - 2h/3, so this padding encloses the soft zero set.
            return 2. * float(WIDTH) / 3.
        return 0.

    def bounding_sphere(self):
        spheres = self.support_spheres
        center = np.mean(spheres[:, :3], axis=0)
        radius = np.max(np.linalg.norm(spheres[:, :3]-center, axis=1)) + np.max(spheres[:, 3])
        return center, float(radius) + self._smooth_bound_padding()

    def ground_clearance(self, poses, plane_position, plane_normal):
        """The simulator's support-sphere/plane contact rule for an object pose batch."""
        from scipy.spatial.transform import Rotation
        values = []
        for pose in poses:
            centers = self.support_spheres[:, :3] @ Rotation.from_quat(pose[:4]).as_matrix().T + pose[4:7]
            phi = (centers-plane_position) @ plane_normal - self.support_spheres[:, 3] + self.distance_offset
            values.append(float(np.min(phi)))
        return np.asarray(values, dtype=np.float64)

    def ray_exit(self, origin, direction, padding, limit):
        """Distance along ``direction`` at which an interior point first clears the object by ``padding``."""
        if self.fused:
            raise ValueError("Fused surfaces use distance-gradient retraction, not hard ray intervals")
        origin = np.asarray(origin, dtype=np.float64).reshape(XYZ_DIM)
        direction = np.asarray(direction, dtype=np.float64).reshape(XYZ_DIM)
        if self.kind == "spheres":
            # End of the connected ray interval through the spheres grown by the padding.
            return self._rays.clearance_retreat(origin[None], direction, padding, limit)
        # Hard mesh distance; soft retraction never uses sphere tracing.
        distance = 0.0
        for _ in range(RAY_ITERATIONS):
            phi = float(self.distance_bounds((origin + distance * direction)[None])[0, 0])
            if phi >= padding or distance >= limit:
                break
            distance = min(distance + (padding - phi) + RAY_EPS_M, float(limit))
        return float(np.nextafter(distance, np.inf))
