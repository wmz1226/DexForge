"""Seed contact geometry with lazily cached Jacobians."""

from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from typing import Optional

import numpy as np

from contactaware.initialization.contact_priority_frame import ContactGeometry
from contactaware.initialization.first_frame_jacobian import load_point_jacobian_kernel, point_jacobians_from_bodies

LINE_SEARCH_CACHE_SIZE = 3  # Accepted state, trial state and callback state.


@dataclass(frozen=True)
class LazyGeometryValues:
    geometry: object
    qpos: np.ndarray
    points: np.ndarray
    phi: np.ndarray
    distance_gradient: np.ndarray
    witnesses: tuple
    collision_query_ids: np.ndarray
    query_count: int

    def __getitem__(self, name):
        return getattr(self, name)

    @cached_property
    def error(self):
        return (self.points[self.contact_indices] - self.geometry.targets) / self.geometry.length_scale

    @cached_property
    def contact_indices(self):
        indices = np.searchsorted(self.collision_query_ids, self.geometry.query_ids)
        if np.any(self.collision_query_ids[indices] != self.geometry.query_ids):
            raise RuntimeError("Active collision queries do not contain every contact query")
        return indices

    @cached_property
    def constraints(self):
        cfg = self.geometry.problem.cfg
        return np.concatenate((self.phi - cfg.hand_object_safe_distance,
                               self.self_clearance - cfg.safe_distance)) / self.geometry.length_scale

    @property
    def self_clearance(self):
        return self.witnesses[-1]

    @cached_property
    def contact_frame_error(self):
        normals, _ = self.normal_geometry
        directions, _ = self.geometry.pad_aim_geometry(
            self.qpos, self.geometry.query_ids, self.geometry.targets)
        return self.geometry.contact_frame_errors(
            self.points[self.contact_indices], normals, directions)

    @cached_property
    def normal_geometry(self):
        return self.geometry.problem.hand.query_normals_jacobian(self.qpos, self.geometry.query_ids)

    @cached_property
    def body_derivatives(self):
        hand = self.geometry.problem.hand
        hand.forward(self.qpos)
        jac = np.asarray([hand.body_jacobian(body) for body in range(hand.model.nbody)])
        return jac, hand.data.xpos.copy()

    def points_jacobian(self, body_ids, points):
        jac, origins = self.body_derivatives
        return point_jacobians_from_bodies(jac, origins, body_ids, points,
                                           kernel=self.geometry.jacobian_kernel)

    @cached_property
    def point_jac(self):
        geometry, hand = self.geometry, self.geometry.problem.hand
        body_ids = hand.query_points.body_ids[self.collision_query_ids]
        jac = self.points_jacobian(body_ids, self.points[:self.query_count])
        if geometry.extra_samples is None:
            return jac
        extra = self.points_jacobian(
            geometry.extra_samples.body_ids, self.points[self.query_count:])
        return np.concatenate((jac, extra))

    @cached_property
    def constraint_jac(self):
        geometry, hand = self.geometry, self.geometry.problem.hand
        object_jac = np.einsum("ni,nij->nj", self.distance_gradient, self.point_jac)
        left, right, normals, _ = self.witnesses
        left_ids = hand.capsule_body_ids_np[hand.capsule_pair_left]
        right_ids = hand.capsule_body_ids_np[hand.capsule_pair_right]
        delta_jac = self.points_jacobian(left_ids, left) - self.points_jacobian(right_ids, right)
        self_jac = np.einsum("ni,nij->nj", normals, delta_jac)
        return np.vstack((object_jac, self_jac)) / geometry.length_scale

    @cached_property
    def normal_alignment(self):
        normals, _ = self.normal_geometry
        directions, _ = self.geometry.pad_aim_geometry(
            self.qpos, self.geometry.query_ids, self.geometry.targets)
        return np.sum(normals * directions, axis=1)


@dataclass(frozen=True)
class FastContactGeometry(ContactGeometry):
    jacobian_kernel: object = None
    collision_query_ids: Optional[np.ndarray] = None

    @property
    def evaluation_cache_size(self):
        return LINE_SEARCH_CACHE_SIZE

    @cached_property
    def state_evaluator(self):
        return lru_cache(maxsize=LINE_SEARCH_CACHE_SIZE)(self.evaluate)


    def evaluate(self, coordinates):
        qpos = np.asarray(coordinates)
        hand, problem = self.problem.hand, self.problem
        ids = (np.arange(len(hand.query_points.body_ids), dtype=np.int32)
               if self.collision_query_ids is None else self.collision_query_ids)
        ids = np.union1d(ids, self.query_ids).astype(np.int32)
        points = hand.query_positions(qpos, ids)
        query_count = len(points)
        if self.extra_samples is not None:
            points = np.vstack((points, self.extra_samples.positions(hand, qpos)))
        phi, distance_gradient = self.ops.distance_world(points, problem.obj_pose, problem.obj)
        starts, ends = hand.capsule_segments(qpos)
        witnesses = self.ops.capsule_states(hand, starts, ends)
        return LazyGeometryValues(
            self, qpos, points, phi, distance_gradient, witnesses, ids, query_count)


    def full_collision_violations(self, qpos):
        hand, problem = self.problem.hand, self.problem
        ids = np.arange(len(hand.query_points.body_ids), dtype=np.int32)
        points = hand.query_positions(np.asarray(qpos), ids)
        phi = self.ops.distance_world(points, problem.obj_pose, problem.obj)[0]
        return ids[phi < problem.cfg.hand_object_safe_distance]

    def add_collision_queries(self, query_ids):
        """Widen the active collision set; ``None`` already means every query."""
        if self.collision_query_ids is None:
            return self
        return replace(self, collision_query_ids=np.union1d(
            self.collision_query_ids, query_ids).astype(np.int32))


def accelerated_geometry(geometry):
    return FastContactGeometry(**{**vars(geometry),
                                  "jacobian_kernel": load_point_jacobian_kernel()})
