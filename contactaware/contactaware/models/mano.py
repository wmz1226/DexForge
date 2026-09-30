"""MANO model, quaternion, and trajectory utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import Module

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANOPTH_ROOT = ROOT / "third_party/manopth"
DEFAULT_MANO_MODEL_ROOT = ROOT.parent / "assets/mano"
VALID_MANO_SIDES = ("left", "right")
LEFT_SHAPEDIRS_MATCH_THRESHOLD = 1.0

MANOPTH_ROOT = Path(os.environ.get("MANOPTH_ROOT", DEFAULT_MANOPTH_ROOT))
MANO_MODEL_ROOT = Path(os.environ.get("MANO_MODEL_ROOT", DEFAULT_MANO_MODEL_ROOT))
if not MANOPTH_ROOT.exists():
    raise FileNotFoundError(f"Missing manopth package root: {MANOPTH_ROOT}")
if not (MANO_MODEL_ROOT / "MANO_RIGHT.pkl").exists():
    raise FileNotFoundError(f"Missing MANO_RIGHT.pkl in {MANO_MODEL_ROOT}")
sys.path.insert(0, str(MANOPTH_ROOT))

np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_

from manopth.manolayer import ManoLayer as TorchManoLayer  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from contactaware.contact.mano_surface import canonical_mano_surface_mapping
from contactaware.settings import MANO_DISTAL_WEIGHT_JOINT_BY_FINGER
from contactaware.types import ManoTrajectory


class MANOLayer(Module):
    """Wrap manopth and return vertices/joints in meters."""

    def __init__(self, side: str, betas) -> None:
        super().__init__()
        if side not in VALID_MANO_SIDES:
            raise ValueError(f"Unsupported MANO side: {side!r}")
        self.side = side
        self._mano_layer = self._new_mano_layer(side)
        self._correct_left_shapedirs(side)

        b = torch.from_numpy(betas).unsqueeze(0)
        f = self._mano_layer.th_faces
        self.register_buffer("b", b)
        self.register_buffer("f", f)

        shaped_vertices = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1))
            .permute(2, 0, 1)
            + self._mano_layer.th_v_template
        )
        canonical_joints = torch.matmul(
            self._mano_layer.th_J_regressor,
            shaped_vertices,
        )
        self.register_buffer("canonical_vertices", shaped_vertices[0])
        self.register_buffer("canonical_joints", canonical_joints[0])

    @staticmethod
    def _new_mano_layer(side: str):
        return TorchManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=side,
            mano_root=str(MANO_MODEL_ROOT),
            use_pca=True,
        )

    def _correct_left_shapedirs(self, side: str) -> None:
        if side != "left":
            return
        right_layer = self._new_mano_layer("right")
        difference = torch.sum(
            torch.abs(
                self._mano_layer.th_shapedirs[:, 0, :]
                - right_layer.th_shapedirs[:, 0, :]
            )
        )
        if float(difference) >= LEFT_SHAPEDIRS_MATCH_THRESHOLD:
            return
        with torch.no_grad():
            self._mano_layer.th_shapedirs[:, 0, :] *= -1

    def forward(self, pose: torch.Tensor, trans: torch.Tensor):
        verts, joints = self._mano_layer(pose, self.b.expand(pose.size(0), -1), trans)
        return verts / 1000.0, joints / 1000.0

    def canonical_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.canonical_vertices.detach().cpu().numpy().copy(),
            self.canonical_joints.detach().cpu().numpy().copy(),
            self.f.detach().cpu().numpy().copy(),
        )


def canonicalize_quaternion_sequence(quaternions: np.ndarray) -> np.ndarray:
    """Flip signs so adjacent quaternions remain on the same hemisphere."""
    canonical = quaternions.copy()
    for frame_id in range(1, len(canonical)):
        if np.dot(canonical[frame_id], canonical[frame_id - 1]) < 0:
            canonical[frame_id] = -canonical[frame_id]
    return canonical


def prepare_mano(data: dict):
    hand_pose = data["hand_pose"]
    if hand_pose.ndim == 2:
        hand_pose = hand_pose[:, np.newaxis, :]
    camera_mat = np.linalg.inv(data["extrinsics"])
    mano = MANOLayer(data["hand_side"], data["hand_shape"].astype(np.float32))
    start = next(
        (i for i in range(hand_pose.shape[0]) if np.abs(hand_pose[i]).sum() > 1e-5),
        0,
    )
    return mano, camera_mat[:3, :3], camera_mat[:3, 3], int(start)


def mano_refs_vertices(data: dict, args) -> ManoTrajectory:
    mano, r_wc, t_wc, start = prepare_mano(data)
    hand_pose = (
        data["hand_pose"][:, np.newaxis, :]
        if data["hand_pose"].ndim == 2
        else data["hand_pose"]
    )
    frame_ids = tuple(range(start, hand_pose.shape[0], int(args.subsample)))
    if "hand_valid" in data:
        frame_ids = tuple(i for i in frame_ids if data["hand_valid"][i])
    frame_ids = frame_ids[:args.max_frames] if args.max_frames else frame_ids
    refs, vertices = [], []
    for frame_id in frame_ids:
        verts, joints = mano_vertices_joints_world(
            mano,
            hand_pose[frame_id],
            r_wc=r_wc,
            t_wc=t_wc,
            mesh_scale=float(data["hand_scale"][frame_id]) if "hand_scale" in data else 1.0,
        )
        refs.append(joints)
        vertices.append(verts)
    refs = np.asarray(refs, dtype=np.float32)
    vertices = np.asarray(vertices, dtype=np.float32)
    vertex_fingers = mano_distal_vertex_finger_ids(mano)
    surface_mapping = canonical_mano_surface_mapping(mano, vertex_fingers)
    return ManoTrajectory(
        joints=refs,
        vertices=vertices,
        faces=mano.f.detach().cpu().numpy().astype(np.int32, copy=True),
        vertex_fingers=vertex_fingers,
        surface_mapping=surface_mapping,
        frame_ids=frame_ids,
    )


def mano_vertices_joints_world(
    mano,
    hand_pose_frame,
    *,
    r_wc,
    t_wc,
    mesh_scale=1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pose = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    trans = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    verts, joints = mano(pose, trans)
    verts_np = verts.cpu().numpy()[0]
    joints_np = joints.cpu().numpy()[0]
    if mesh_scale != 1.0:
        # V2D applies depth-alignment scale about the posed mesh centroid.
        # Keep MANO betas and joint/vertex geometry in the same metric frame.
        if not np.isfinite(mesh_scale) or mesh_scale <= 0:
            raise ValueError("MANO mesh_scale must be finite and positive")
        center = verts_np.mean(axis=0, keepdims=True)
        verts_np = center + (verts_np - center) * mesh_scale
        joints_np = center + (joints_np - center) * mesh_scale
    verts_np = verts_np @ r_wc.T + t_wc
    joints_np = joints_np @ r_wc.T + t_wc
    return verts_np, joints_np


def mano_distal_vertex_finger_ids(mano) -> np.ndarray:
    weights = mano._mano_layer.th_weights.detach().cpu().numpy()
    dominant = np.argmax(weights, axis=1)
    fingers = np.full(dominant.shape, -1, dtype=np.int32)
    for finger_id, mano_joint in MANO_DISTAL_WEIGHT_JOINT_BY_FINGER.items():
        fingers[dominant == int(mano_joint)] = int(finger_id)
    return fingers


def select_mano_vertex_fingers(
    vertex_fingers: np.ndarray,
    track_finger_ids: tuple[int, ...],
) -> np.ndarray:
    allowed = np.asarray(track_finger_ids, dtype=np.int32)
    selected = np.asarray(vertex_fingers, dtype=np.int32).copy()
    selected[~np.isin(selected, allowed)] = -1
    return selected


def object_sequence(data: dict, frame_ids: Sequence[int]) -> np.ndarray:
    obj_pose = data["object_pose"]
    motion = np.linalg.norm(obj_pose[-1, :, 4:] - obj_pose[0, :, 4:], axis=1)
    obj_idx = int(np.argmax(motion))
    camera_mat = np.linalg.inv(data["extrinsics"])
    r_wc, t_wc = camera_mat[:3, :3], camera_mat[:3, 3]
    poses = []
    for i in frame_ids:
        pose = obj_pose[i, obj_idx]
        rot = r_wc @ Rotation.from_quat(pose[:4]).as_matrix()
        poses.append(
            np.concatenate(
                [Rotation.from_matrix(rot).as_quat(), r_wc @ pose[4:] + t_wc]
            )
        )
    out = np.asarray(poses, dtype=np.float32)
    out[:, :4] = canonicalize_quaternion_sequence(out[:, :4])
    return out
