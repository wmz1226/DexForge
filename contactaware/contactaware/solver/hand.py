"""MuJoCo hand adapter, query-point assets, and geometric Jacobians."""

from __future__ import annotations

import mujoco
import numpy as np

from contactaware.contact.mapping import (
    query_groups,
    tip_query_ids,
)
from contactaware.contact.surface import (
    SURFACE_EPS,
    SURFACE_DESCRIPTOR_SIZE,
    TerminalFrame,
    make_terminal_frame,
    normalize_vector,
    surface_descriptors,
)
from contactaware.settings import FINGER_LABELS, PALM_ANCHORS, XYZ_DIM
from contactaware.types import (
    Capsule,
    HandProfile,
    QueryBodySpec,
    QueryPointSet,
    SurfaceDescriptorSet,
)

def cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Broadcast cross product, identical to ``np.cross`` without its per-call overhead."""
    return np.stack(
        (left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
         left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
         left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]),
        axis=-1)

class MujocoHand:
    def __init__(self, profile: HandProfile, radius_scale: float):
        self.profile = profile
        self.model = mujoco.MjModel.from_xml_path(str(profile.mesh_xml))
        self.data = mujoco.MjData(self.model)
        self.qpos_dim = int(profile.default_qpos.shape[0])
        self.base_qpos_dim = int(profile.base_qpos_dim)
        self.validate_qpos_layout()
        self.qadr = np.arange(self.qpos_dim, dtype=np.int32)
        self.lower, self.upper = self.joint_limits()
        self.palm_body_id = self.body_id(profile.palm_body)
        self.palm_origin_local, self.palm_basis_local = self.calibrated_palm_frame_local()
        self.capsules = scale_capsules(profile.capsules, radius_scale)
        self.capsule_body_ids = [self.body_id(c.body) for c in self.capsules]
        self.capsule_body_ids_np = np.asarray(
            self.capsule_body_ids, dtype=np.int64
        )
        self.capsule_starts_local = np.asarray(
            [c.start for c in self.capsules], dtype=np.float64
        )
        self.capsule_ends_local = np.asarray(
            [c.end for c in self.capsules], dtype=np.float64
        )
        self.capsule_radii = np.asarray([c.radius for c in self.capsules], dtype=np.float64)
        component_pairs = self_collision_capsule_pairs(
            self.model, self.capsules, self.capsule_body_ids
        )
        self.capsule_component_left = np.asarray([p[0] for p in component_pairs], dtype=np.int64)
        self.capsule_component_right = np.asarray([p[1] for p in component_pairs], dtype=np.int64)
        self.capsule_pair_groups = group_capsule_pairs(component_pairs, self.capsule_body_ids)
        self.capsule_pairs = [component_pairs[group[0]] for group in self.capsule_pair_groups]
        self.capsule_pair_left = np.asarray(
            [pair[0] for pair in self.capsule_pairs], dtype=np.int64
        )
        self.capsule_pair_right = np.asarray(
            [pair[1] for pair in self.capsule_pairs], dtype=np.int64
        )
        self.query_points = load_query_points(self.model, profile)
        self.query_surface_radii = terminal_surface_radii(self.query_points)
        self.query_surface_descriptors = self.build_query_surface_descriptors()
        self.query_ids_by_group = grouped_query_ids(query_groups(self.query_points))

    def validate_qpos_layout(self) -> None:
        if self.base_qpos_dim <= 0 or self.base_qpos_dim > self.qpos_dim:
            raise ValueError(
                f"Invalid base_qpos_dim={self.base_qpos_dim} for {self.profile.name}"
            )
        if self.model.nq != self.qpos_dim or self.model.nv != self.qpos_dim:
            raise ValueError(
                f"{self.profile.name} profile default_qpos has dim {self.qpos_dim}, "
                f"but MuJoCo model has nq={self.model.nq}, nv={self.model.nv}"
            )
        if self.model.njnt != self.qpos_dim:
            raise ValueError(
                f"{self.profile.name} expects one scalar joint per qpos; "
                f"got njnt={self.model.njnt}, qpos_dim={self.qpos_dim}"
            )

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = [], []
        for joint_id in range(self.model.njnt):
            low.append(self.model.jnt_range[joint_id, 0])
            high.append(self.model.jnt_range[joint_id, 1])
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)

    def body_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"Missing MuJoCo body: {name}")
        return int(idx)

    def site_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"Missing MuJoCo site: {name}")
        return int(idx)

    def forward(self, qpos: np.ndarray) -> None:
        if qpos.shape[0] != self.qpos_dim:
            raise ValueError(
                f"{self.profile.name} qpos dim mismatch: expected {self.qpos_dim}, "
                f"got {qpos.shape[0]}"
            )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        # Refresh COM-frame motion axes for mj_jac.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def capsule_segments(self, qpos: np.ndarray):
        self.forward(qpos)
        body_pos = self.data.xpos[self.capsule_body_ids_np]
        body_mat = self.data.xmat[self.capsule_body_ids_np].reshape(-1, 3, 3)
        starts = body_pos + np.einsum("nij,nj->ni", body_mat, self.capsule_starts_local)
        ends = body_pos + np.einsum("nij,nj->ni", body_mat, self.capsule_ends_local)
        return starts, ends

    def points_and_capsules(self, qpos: np.ndarray):
        """All query positions and capsule segments from one kinematics pass."""
        self.forward(qpos)
        local = self.query_points.local_pos.astype(np.float64, copy=False)
        body_ids = self.query_points.body_ids.astype(np.int64)
        points = self.data.xpos[body_ids] + np.einsum(
            "nij,nj->ni", self.data.xmat[body_ids].reshape(-1, 3, 3), local)
        body_pos = self.data.xpos[self.capsule_body_ids_np]
        body_mat = self.data.xmat[self.capsule_body_ids_np].reshape(-1, 3, 3)
        starts = body_pos + np.einsum("nij,nj->ni", body_mat, self.capsule_starts_local)
        ends = body_pos + np.einsum("nij,nj->ni", body_mat, self.capsule_ends_local)
        return points, starts, ends

    def palm_anchors_jacobian(self, qpos: np.ndarray):
        self.forward(qpos)
        positions, jacobians = [], []
        for offset in PALM_ANCHORS:
            local_offset = self.palm_origin_local + self.palm_basis_local @ offset
            positions.append(self.local_point(self.palm_body_id, local_offset))
            jacobians.append(self.point_jacobian(self.palm_body_id, positions[-1]))
        return np.stack(positions), np.stack(jacobians)

    def palm_pose_jacobian(
        self,
        qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.forward(qpos)
        origin = self.local_point(self.palm_body_id, self.palm_origin_local)
        body_rotation = self.data.xmat[self.palm_body_id].reshape(3, 3)
        rotation = body_rotation @ self.palm_basis_local
        position_jacobian = self.point_jacobian(self.palm_body_id, origin)
        _, rotation_jacobian = self.body_jacobian(self.palm_body_id)
        return origin, rotation, position_jacobian, rotation_jacobian

    def calibrated_palm_frame_local(self) -> tuple[np.ndarray, np.ndarray]:
        self.forward(self.profile.default_qpos)
        finger_ids = tuple(int(value) for value in self.profile.track_finger_ids)
        right_finger = 5 if 5 in finger_ids else 2
        if 0 not in finger_ids or right_finger not in finger_ids:
            raise ValueError(f"Cannot build palm frame from fingers {finger_ids}")
        roots = {
            finger_id: self.finger_root_position(FINGER_LABELS[finger_id])
            for finger_id in finger_ids
        }
        palm_reference = self.local_point(
            self.palm_body_id, self.profile.palm_anchor_offset
        )
        origin = np.mean([palm_reference, *roots.values()], axis=0)
        left, right = roots[0], roots[right_finger]
        tips = [self.data.site_xpos[self.site_id(name)] for name in self.profile.track_sites]
        x_axis = normalize_vector(left - right, name="robot palm lateral axis")
        y_seed = normalize_vector(
            np.mean(tips, axis=0) - origin, name="robot palm forward axis"
        )
        z_axis = normalize_vector(np.cross(x_axis, y_seed), name="robot palm normal")
        y_axis = normalize_vector(np.cross(z_axis, x_axis), name="robot palm orthogonal axis")
        basis_world = np.stack([x_axis, y_axis, z_axis], axis=1)
        body_rotation = self.data.xmat[self.palm_body_id].reshape(3, 3)
        body_position = self.data.xpos[self.palm_body_id]
        origin_local = body_rotation.T @ (origin - body_position)
        return origin_local, body_rotation.T @ basis_world

    def finger_root_position(self, finger: str) -> np.ndarray:
        specs = self.profile.chain_points.get(finger)
        if not specs:
            raise ValueError(f"Missing {self.profile.name} chain points for {finger}")
        for spec in specs:
            if spec[0] == "body" and self.body_id(spec[1]) == self.palm_body_id:
                continue
            position, _ = self.named_point_jacobian(*spec)
            return position
        raise ValueError(f"Cannot locate {self.profile.name} finger root for {finger}")

    def retarget_points_jacobian(self, qpos: np.ndarray):
        self.forward(qpos)
        positions, jacobians = [], []
        for site_name in self.profile.track_sites:
            site_id = self.site_id(site_name)
            positions.append(self.data.site_xpos[site_id].copy())
            jacobians.append(self.site_jacobian(site_id))
        return np.stack(positions), np.stack(jacobians)

    def current_chain_points_jacobian(self, specs: tuple[tuple[str, str], ...]):
        positions, jacobians = [], []
        for kind, name in specs:
            pos, jac = self.named_point_jacobian(kind, name)
            positions.append(pos)
            jacobians.append(jac)
        return np.stack(positions), np.stack(jacobians)

    def named_point_jacobian(self, kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        if kind == "site":
            site_id = self.site_id(name)
            return self.data.site_xpos[site_id].copy(), self.site_jacobian(site_id)
        if kind == "body":
            body_id = self.body_id(name)
            point = self.data.xpos[body_id].copy()
            return point, self.point_jacobian(body_id, point)
        raise ValueError(f"Unknown point spec kind: {kind}")

    def query_positions(
        self,
        qpos: np.ndarray,
        query_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        self.forward(qpos)
        ids = (
            np.arange(self.query_points.local_pos.shape[0], dtype=np.int64)
            if query_ids is None
            else query_ids.astype(np.int64)
        )
        body_ids = self.query_points.body_ids[ids].astype(np.int64)
        local_pos = self.query_points.local_pos[ids].astype(np.float64, copy=False)
        body_pos = self.data.xpos[body_ids]
        body_mat = self.data.xmat[body_ids].reshape(-1, 3, 3)
        return body_pos + np.einsum("nij,nj->ni", body_mat, local_pos)

    def build_query_surface_descriptors(self) -> SurfaceDescriptorSet:
        count = len(self.query_points.local_pos)
        values = np.full((count, SURFACE_DESCRIPTOR_SIZE), np.nan, dtype=np.float64)
        azimuth_valid = np.zeros(count, dtype=bool)
        self.forward(self.profile.default_qpos)
        for finger_id in self.profile.track_finger_ids:
            query_ids = tip_query_ids(self.query_points, int(finger_id))
            frame = self.query_terminal_frame(int(finger_id), query_ids)
            descriptors = surface_descriptors(
                self.query_points.local_pos[query_ids],
                self.query_points.local_normal[query_ids],
                frame,
            )
            values[query_ids] = descriptors.values
            azimuth_valid[query_ids] = descriptors.azimuth_valid
        return SurfaceDescriptorSet(values, azimuth_valid)

    def terminal_joint_id(self, body_id: int) -> int:
        current = int(body_id)
        while current > 0:
            joint_ids = np.flatnonzero(self.model.jnt_bodyid == current)
            if joint_ids.size > 1:
                raise ValueError(
                    f"{self.profile.name} terminal ancestor body {current} has multiple joints: "
                    f"{joint_ids.tolist()}"
                )
            if joint_ids.size == 1:
                return int(joint_ids[0])
            current = int(self.model.body_parentid[current])
        raise ValueError(f"{self.profile.name} tip body {body_id} has no articulated ancestor")


    def query_terminal_frame(
        self,
        finger_id: int,
        query_ids: np.ndarray,
    ) -> TerminalFrame:
        body_ids = np.unique(self.query_points.body_ids[query_ids])
        if body_ids.size != 1:
            raise ValueError(
                f"{self.profile.name} finger {finger_id} tip queries span bodies: "
                f"{body_ids.tolist()}"
            )
        body_id = int(body_ids[0])
        joint_id = self.terminal_joint_id(body_id)
        tip_body, tip_local = self.profile.tip_references[int(finger_id)]
        if self.body_id(tip_body) != body_id:
            raise ValueError(f"{self.profile.name} finger {finger_id} tip site/query body mismatch")
        body_rotation = self.data.xmat[body_id].reshape(3, 3)
        body_position = self.data.xpos[body_id]
        root = body_rotation.T @ (self.data.xanchor[joint_id] - body_position)
        joint_axis = body_rotation.T @ self.data.xaxis[joint_id]
        tip_reference = np.asarray(tip_local, dtype=np.float64)
        longitudinal = normalize_vector(tip_reference - root, name="robot terminal axis")
        surface_offsets = self.query_points.local_pos[query_ids] - root
        terminal_length = float(np.max(surface_offsets @ longitudinal))
        if terminal_length <= SURFACE_EPS:
            raise ValueError(
                f"{self.profile.name} finger {finger_id} has non-positive terminal length"
            )
        tip = root + terminal_length * longitudinal
        palmar = normalize_vector(
            self.profile.terminal_flexion_signs.get(int(finger_id), 1.0)
            * np.cross(joint_axis, longitudinal),
            name="robot flexion direction",
        )
        return make_terminal_frame(root, tip, palmar)

    def query_positions_jacobian(
        self,
        qpos: np.ndarray,
        query_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = query_ids.astype(np.int64)
        points = self.query_positions(qpos, ids)
        body_ids = self.query_points.body_ids[ids].astype(np.int64)
        return points, self.point_jacobians(body_ids, points)

    def query_jacobians(
        self,
        query_ids: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        ids = query_ids.astype(np.int64)
        body_ids = self.query_points.body_ids[ids].astype(np.int64)
        return self.point_jacobians(body_ids, np.asarray(points, dtype=np.float64))

    def query_normals_jacobian(
        self,
        qpos: np.ndarray,
        query_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normals and dn/dq = angular_jacobian × normal."""
        self.forward(qpos)
        ids = np.asarray(query_ids, dtype=np.int64)
        body_ids = self.query_points.body_ids[ids]
        local = np.asarray(self.query_points.local_normal[ids], dtype=np.float64)
        lengths = np.linalg.norm(local, axis=1)
        if not np.isfinite(local).all() or np.any(lengths <= SURFACE_EPS):
            raise ValueError("Contact query normals must be finite and nonzero")
        rotations = self.data.xmat[body_ids].reshape(-1, XYZ_DIM, XYZ_DIM)
        normals = np.einsum("nij,nj->ni", rotations, local / lengths[:, None])
        jacobians = np.empty((len(ids), XYZ_DIM, self.qpos_dim), dtype=np.float64)
        for body_id in np.unique(body_ids):
            selected = body_ids == body_id
            _, angular = self.body_jacobian(int(body_id))
            jacobians[selected] = cross3(
                angular.T[None], normals[selected, None]
            ).swapaxes(1, 2)
        return normals, jacobians

    def site_jacobian(self, site_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp[:, :self.qpos_dim].copy()

    def local_point(self, body_id: int, offset) -> np.ndarray:
        rot = self.data.xmat[body_id].reshape(3, 3)
        return self.data.xpos[body_id] + rot @ np.asarray(offset, dtype=np.float64)

    def point_jacobian(self, body_id: int, point: np.ndarray) -> np.ndarray:
        return self.point_jacobians(
            np.asarray([body_id], dtype=np.int64),
            np.asarray([point], dtype=np.float64),
        )[0]

    def point_jacobians(self, body_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
        body_ids = np.asarray(body_ids, dtype=np.int64)
        points = np.asarray(points, dtype=np.float64)
        out = np.empty((body_ids.shape[0], 3, self.qpos_dim), dtype=np.float64)
        for body_id in np.unique(body_ids):
            mask = body_ids == int(body_id)
            jacp, jacr = self.body_jacobian(int(body_id))
            offsets = points[mask] - self.data.xpos[int(body_id)]
            angular = cross3(jacr.T[None, :, :], offsets[:, None, :])
            out[mask] = jacp[None, :, :] + np.swapaxes(angular, 1, 2)
        return out

    def body_jacobian(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(body_id))
        return jacp[:, :self.qpos_dim].copy(), jacr[:, :self.qpos_dim].copy()


def terminal_surface_radii(query: QueryPointSet) -> np.ndarray:
    """RMS terminal-surface radius, converting normal error to displacement."""
    radii = np.full(len(query.local_pos), np.nan, dtype=np.float64)
    for body_id in np.unique(query.body_ids[query.is_tip]):
        selected = query.is_tip & (query.body_ids == body_id)
        points = np.asarray(query.local_pos[selected], dtype=np.float64)
        centered = points - np.mean(points, axis=0)
        radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if not np.isfinite(radius) or radius <= SURFACE_EPS:
            raise ValueError(f"Terminal query surface has no finite extent: body {body_id}")
        radii[selected] = radius
    return radii


def scale_capsules(capsules: tuple[Capsule, ...], scale: float) -> list[Capsule]:
    if scale <= 0.0:
        raise ValueError(f"sphere_radius_scale must be positive, got {scale}")
    return [
        Capsule(c.geom, c.body, c.finger, c.start, c.end, float(c.radius) * float(scale))
        for c in capsules
    ]


def self_collision_capsule_pairs(model: mujoco.MjModel, capsules: list[Capsule],
                                 body_ids: list[int]) -> list[tuple[int, int]]:
    # Include cross-finger pairs beyond those explicitly listed in XML.
    excluded = excluded_body_pairs(model)
    for i, left in enumerate(body_ids):
        for right in body_ids[i + 1:]:
            left_weld, right_weld = int(model.body_weldid[left]), int(model.body_weldid[right])
            if (left_weld == right_weld
                    or model.body_weldid[model.body_parentid[left_weld]] == right_weld
                    or model.body_weldid[model.body_parentid[right_weld]] == left_weld):
                excluded.add(tuple(sorted((int(left), int(right)))))
    return default_cross_finger_capsule_pairs(
        capsules, body_ids, excluded)


def group_capsule_pairs(pairs, body_ids):
    """Closest capsule-pair constraints per link pair; repeated indices pad groups."""
    groups = {}
    for index, (left, right) in enumerate(pairs):
        groups.setdefault((body_ids[left], body_ids[right]), []).append(index)
    width = max(map(len, groups.values()), default=0)
    return np.asarray([group + [group[0]] * (width - len(group))
                       for group in groups.values()], dtype=np.int64).reshape(len(groups), width)


def default_cross_finger_capsule_pairs(capsules: list[Capsule], body_ids: list[int],
                                       excluded: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (i, j) for i, cap in enumerate(capsules)
        for j in range(i + 1, len(capsules))
        if cap.finger != capsules[j].finger
        and tuple(sorted((int(body_ids[i]), int(body_ids[j])))) not in excluded
    ]


def excluded_body_pairs(model: mujoco.MjModel) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(sig) & 0xFFFF, int(sig) >> 16)))
        for sig in model.exclude_signature
    }


def load_query_points(model: mujoco.MjModel, profile: HandProfile) -> QueryPointSet:
    local_pos, local_normal = [], []
    body_ids, finger_ids, link_ids, is_tip = [], [], [], []
    for spec in profile.query_specs:
        with np.load(profile.qp_dir / f"{spec.file_stem}.npz") as npz:
            pos = npz["local_pos"].astype(np.float64)
            normal = load_query_normals(npz, spec, len(pos))
            finger = query_metadata(
                npz, "finger_idx", pos.shape[0], fallback=spec.finger_id
            )
            link = query_metadata(npz, "link_id", pos.shape[0], fallback=spec.link_id)
        if pos.size == 0:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
        if body_id < 0:
            raise ValueError(f"Missing query-point body: {spec.body}")
        local_pos.append(pos)
        local_normal.append(normal)
        body_ids.append(np.full(pos.shape[0], body_id, dtype=np.int32))
        finger_ids.append(finger)
        link_ids.append(link)
        is_tip.append(np.full(pos.shape[0], spec.is_tip, dtype=bool))
    if not local_pos:
        raise ValueError(f"No query points loaded for hand profile: {profile.name}")
    return QueryPointSet(
        local_pos=np.concatenate(local_pos),
        local_normal=np.concatenate(local_normal),
        body_ids=np.concatenate(body_ids),
        finger_ids=np.concatenate(finger_ids),
        link_ids=np.concatenate(link_ids),
        is_tip=np.concatenate(is_tip),
    )


def load_query_normals(npz, spec: QueryBodySpec, count: int) -> np.ndarray:
    if "local_normal" not in npz.files:
        if spec.is_tip:
            raise KeyError(f"Tip query file {spec.file_stem}.npz is missing local_normal")
        return np.full((count, 3), np.nan, dtype=np.float64)
    normals = npz["local_normal"].astype(np.float64)
    if normals.shape != (count, 3):
        raise ValueError(
            f"{spec.file_stem}.npz local_normal must have shape {(count, 3)}, "
            f"got {normals.shape}"
        )
    return normals


def query_metadata(npz, key: str, count: int, *, fallback: int) -> np.ndarray:
    if key in npz.files:
        return npz[key].astype(np.int32)
    return np.full(count, int(fallback), dtype=np.int32)


def grouped_query_ids(groups: np.ndarray) -> tuple[np.ndarray, ...]:
    unique_groups = np.unique(groups, axis=0)
    return tuple(
        np.flatnonzero(np.all(groups == group, axis=1)).astype(np.int32)
        for group in unique_groups
    )
