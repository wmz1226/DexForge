"""Robot query points, normals and kinematics at a key."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contactaware.solver.collision import capsule_pair_states

XYZ_DIM = 3
GEOMETRY_EPSILON = 1e-12


@dataclass(frozen=True)
class RobotGeometry:
    points: np.ndarray
    point_jacobians: np.ndarray
    pad_normals: np.ndarray
    pad_jacobians: np.ndarray
    self_clearances: np.ndarray
    self_jacobians: np.ndarray


@dataclass(frozen=True)
class GraspSeed:
    qpos_obj: np.ndarray
    query_ids: np.ndarray
    finger_ids: np.ndarray
    contact_points_obj: np.ndarray
    source: Path


def _unit_vectors(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= GEOMETRY_EPSILON):
        raise ValueError("Contact normals must be finite and nonzero")
    return values / norms


def query_normals_jacobian(hand, ids) -> tuple[np.ndarray, np.ndarray]:
    """Use the hand's current FK state; normals are outward in that world frame."""
    ids = np.asarray(ids, dtype=np.int64)
    bodies = hand.query_points.body_ids[ids]
    local = _unit_vectors(hand.query_points.local_normal[ids])
    rotations = hand.data.xmat[bodies].reshape(-1, XYZ_DIM, XYZ_DIM)
    normals = np.einsum("nij,nj->ni", rotations, local)
    jacobians = np.empty((len(ids), XYZ_DIM, hand.qpos_dim), dtype=np.float64)
    for body in np.unique(bodies):
        mask = bodies == body
        _, angular = hand.body_jacobian(int(body))
        jacobians[mask] = np.cross(angular.T[None], normals[mask, None]).swapaxes(1, 2)
    return normals, jacobians


def robot_kinematics(qpos, *, hand):
    all_ids = np.arange(len(hand.query_points.local_pos), dtype=np.int64)
    positions, jacobians = hand.query_positions_jacobian(qpos, all_ids)
    normals, normal_jacobians = query_normals_jacobian(hand, all_ids)
    starts, ends = hand.capsule_segments(qpos)
    left_points, right_points, directions, clearance = capsule_pair_states(hand, starts, ends)
    left_bodies = hand.capsule_body_ids_np[hand.capsule_pair_left]
    right_bodies = hand.capsule_body_ids_np[hand.capsule_pair_right]
    relative = (hand.point_jacobians(left_bodies, left_points)
                - hand.point_jacobians(right_bodies, right_points))
    self_jac = np.einsum("ni,nij->nj", directions, relative)
    return RobotGeometry(positions, jacobians, normals, normal_jacobians, clearance, self_jac)


