"""Evaluate only the point Jacobian rows actually used by key constraints."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from contactaware.initialization.key_geometry import RobotGeometry
from contactaware.initialization.key_kinematics import KeyKinematics, XYZ_DIM
from contactaware.solver.collision import capsule_pair_states
from contactaware.solver.hand import cross3


@dataclass(frozen=True)
class PointJacobianRows:
    body_rows: np.ndarray
    points: np.ndarray
    origins: np.ndarray
    linear: np.ndarray
    angular: np.ndarray

    def apply(self, step):
        """Apply a pose increment using body twists without dense point Jacobians."""
        linear = self.linear @ step
        angular = self.angular @ step
        offsets = self.points - self.origins[self.body_rows]
        return linear[self.body_rows] + cross3(angular[self.body_rows], offsets)

    def __getitem__(self, indices):
        selected = np.atleast_1d(indices)
        rows = KeyKinematics.point_rows(self.body_rows[selected], self.points[selected],
            self.origins, self.linear, self.angular)
        return rows[0] if np.ndim(indices) == 0 else rows


@dataclass(frozen=True)
class NormalJacobianRows:
    body_rows: np.ndarray
    normals: np.ndarray
    angular: np.ndarray

    def __getitem__(self, indices):
        selected = np.atleast_1d(indices)
        rows = cross3(self.angular[self.body_rows[selected]].swapaxes(1, 2),
                      self.normals[selected, None]).swapaxes(1, 2)
        return rows[0] if np.ndim(indices) == 0 else rows


@dataclass(frozen=True)
class CapsuleJacobianRows:
    """Keep capsule values cheap until a constraint block requests derivatives."""
    left: PointJacobianRows
    right: PointJacobianRows
    direction: np.ndarray

    @cached_property
    def rows(self):
        indices = np.arange(len(self.direction))
        relative = self.left[indices] - self.right[indices]
        return np.einsum("ni,nij->nj", self.direction, relative)

    def __array__(self, dtype=None):
        return np.asarray(self.rows, dtype=dtype)


class CompactKeyKinematics(KeyKinematics):
    def __call__(self, state):
        hand = self.hand
        points = hand.query_positions(state)
        linear, angular = (np.stack(items) for items in zip(*(hand.body_jacobian(int(body)) for body in self.bodies)))
        origins = hand.data.xpos[self.bodies].copy()
        rows = PointJacobianRows(self.query_rows, points, origins, linear, angular)
        rotations = hand.data.xmat[self.query_bodies].reshape(-1, XYZ_DIM, XYZ_DIM)
        normals = np.einsum('nij,nj->ni', rotations, self.local_normals)
        normal_rows = NormalJacobianRows(self.query_rows, normals, angular)
        starts, ends = hand.capsule_segments(state)
        left, right, direction, clearance = capsule_pair_states(hand, starts, ends)
        self_rows = CapsuleJacobianRows(
            PointJacobianRows(self.left_rows, left, origins, linear, angular),
            PointJacobianRows(self.right_rows, right, origins, linear, angular), direction)
        return RobotGeometry(points, rows, normals, normal_rows, clearance, self_rows)
