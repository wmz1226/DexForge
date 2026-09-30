"""Reuse body Jacobians across the collision points and contact normals of a key."""

import numpy as np

from contactaware.solver.hand import cross3

XYZ_DIM = 3


class KeyKinematics:
    def __init__(self, hand):
        self.hand = hand
        self.query_bodies = np.asarray(hand.query_points.body_ids, dtype=np.int64)
        self.left_bodies = hand.capsule_body_ids_np[hand.capsule_pair_left]
        self.right_bodies = hand.capsule_body_ids_np[hand.capsule_pair_right]
        self.bodies = np.unique(np.r_[self.query_bodies, self.left_bodies, self.right_bodies])
        self.query_rows = np.searchsorted(self.bodies, self.query_bodies)
        self.left_rows = np.searchsorted(self.bodies, self.left_bodies)
        self.right_rows = np.searchsorted(self.bodies, self.right_bodies)
        local = np.asarray(hand.query_points.local_normal, dtype=np.float64)
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        if not np.isfinite(local).all() or np.any(norms == 0.0):
            raise ValueError("Contact query normals must be finite and nonzero")
        self.local_normals = local / norms

    @staticmethod
    def point_rows(body_rows, points, origins, linear, angular):
        offsets = points - origins[body_rows]
        rotation = cross3(angular[body_rows].swapaxes(1, 2), offsets[:, None])
        return linear[body_rows] + rotation.swapaxes(1, 2)

