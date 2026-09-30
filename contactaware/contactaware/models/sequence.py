"""Load and validate processed DexYCB and HOT3D sequences."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from contactaware.types import (
    TERMINAL_PAD,
    ManoTrajectory,
    SequencePolicy,
)


REQUIRED_META_KEYS = ("object", "camera")
VALID_HAND_SIDES = ("left", "right")
DEXYCB_SOURCE = "dexycb_mano_raw"
HOT3D_SOURCE = "hot3d_mano_raw"
V2D_SOURCE = "v2d_mano_raw"
DEXYCB_DATASET = "DexYCB"
HOT3D_DATASET = "HOT3D"
ARIA_HEADSET = "aria"
QUEST3_HEADSET = "quest3"
IDENTITY_TRANSFORM = "identity"
QUEST3_TO_Z_UP_TRANSFORM = "quest3_y_up_to_z_up"
GROUND_STABILIZED_POLICY = "ground_stabilized"
PRESERVE_OBSERVED_WORLD_POLICY = "preserve_observed_world"
QUEST3_TO_Z_UP_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
POSE_QUATERNION = slice(0, 4)
POSE_TRANSLATION = slice(4, 7)


def load_sequence_data(raw_dir: Path) -> tuple[dict, dict]:
    meta_path = raw_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing contactaware MANO input: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            raise KeyError(f"{meta_path} is missing required key: {key}")
    hand_side = str(meta.get("hand_side", "right"))
    if hand_side not in VALID_HAND_SIDES:
        raise ValueError(f"{meta_path} has invalid hand_side: {hand_side!r}")
    policy = resolve_sequence_policy(meta)
    data = {
        "hand_pose": np.load(raw_dir / "hand_pose_51.npy"),
        "object_pose": np.load(raw_dir / "object_pose_camera_7.npy"),
        "extrinsics": np.load(raw_dir / "extrinsics_4x4.npy"),
        "hand_shape": np.load(raw_dir / "hand_shape_10.npy"),
        "camera_pose": np.load(raw_dir / "camera_pose_7.npy"),
        "camera_info": meta["camera"],
        "hand_side": hand_side,
        "sequence_policy": policy,
    }
    if meta.get("source") == V2D_SOURCE:
        if meta.get("hand_scale_convention") != "posed_vertex_centroid":
            raise ValueError("V2D hand_scale must use the posed_vertex_centroid convention")
        count = len(data["hand_pose"])
        scale = np.asarray(meta["hand_scale"], dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(count, float(scale))
        valid = np.asarray(meta.get("hand_valid", [True] * count), dtype=bool)
        if scale.shape != (count,) or valid.shape != (count,):
            raise ValueError("V2D hand_scale/hand_valid must match the hand frame count")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("V2D hand_scale must be finite and positive")
        data.update(hand_scale=scale, hand_valid=valid)
    return data, meta


def resolve_sequence_policy(meta: dict) -> SequencePolicy:
    source = str(meta.get("source", "")).strip()
    dataset = str(meta.get("dataset", "")).strip()
    if source == V2D_SOURCE:
        if dataset.casefold() != "v2d" or meta.get("pose_coordinate_frame") != "fixed_first_camera":
            raise ValueError("V2D input requires dataset=V2D and fixed_first_camera poses")
        return SequencePolicy(
            dataset="V2D",
            source=source,
            coordinate_transform=IDENTITY_TRANSFORM,
            world_pose_policy=PRESERVE_OBSERVED_WORLD_POLICY,
            ground_stabilization=False,
            contact_query_region=TERMINAL_PAD,
        )
    if source == DEXYCB_SOURCE:
        if dataset and dataset.casefold() != DEXYCB_DATASET.casefold():
            raise ValueError(
                f"Conflicting sequence metadata: source={source!r}, dataset={dataset!r}"
            )
        return SequencePolicy(
            dataset=DEXYCB_DATASET,
            source=source,
            coordinate_transform=IDENTITY_TRANSFORM,
            world_pose_policy=GROUND_STABILIZED_POLICY,
            ground_stabilization=True,
            contact_query_region=TERMINAL_PAD,
        )
    if source != HOT3D_SOURCE:
        raise ValueError(f"Unsupported sequence source: {source!r}")
    if dataset.casefold() != HOT3D_DATASET.casefold():
        raise ValueError(
            f"Conflicting sequence metadata: source={source!r}, dataset={dataset!r}"
        )
    headset = str(meta.get("headset", "")).strip().casefold()
    hot3d_world_rotation(headset)
    if headset == ARIA_HEADSET:
        coordinate_transform = IDENTITY_TRANSFORM
    elif headset == QUEST3_HEADSET:
        coordinate_transform = QUEST3_TO_Z_UP_TRANSFORM
    else:
        raise ValueError(f"Unsupported HOT3D headset: {meta.get('headset')!r}")
    return SequencePolicy(
        dataset=HOT3D_DATASET,
        source=source,
        coordinate_transform=coordinate_transform,
        world_pose_policy=PRESERVE_OBSERVED_WORLD_POLICY,
        ground_stabilization=False,
        contact_query_region=TERMINAL_PAD,
    )


def hot3d_world_rotation(headset: str) -> np.ndarray:
    normalized = str(headset).strip().casefold()
    if normalized == ARIA_HEADSET:
        return np.eye(3, dtype=np.float64)
    if normalized == QUEST3_HEADSET:
        return np.asarray(QUEST3_TO_Z_UP_ROTATION, dtype=np.float64)
    raise ValueError(f"Unsupported HOT3D headset: {headset!r}")


def canonicalize_world_trajectory(
    mano: ManoTrajectory,
    object_pose: np.ndarray,
    camera_pose: np.ndarray,
    policy: SequencePolicy,
) -> tuple[ManoTrajectory, np.ndarray, np.ndarray]:
    if policy.coordinate_transform == IDENTITY_TRANSFORM:
        return mano, object_pose, camera_pose
    rotation = world_rotation(policy)
    transformed_mano = replace(
        mano,
        joints=transform_world_points(mano.joints, rotation),
        vertices=transform_world_points(mano.vertices, rotation),
    )
    return (
        transformed_mano,
        transform_world_poses(object_pose, rotation),
        transform_world_poses(camera_pose, rotation),
    )


def world_rotation(policy: SequencePolicy) -> np.ndarray:
    if policy.coordinate_transform == IDENTITY_TRANSFORM:
        return np.eye(3, dtype=np.float64)
    if policy.coordinate_transform == QUEST3_TO_Z_UP_TRANSFORM:
        return np.asarray(QUEST3_TO_Z_UP_ROTATION, dtype=np.float64)
    raise ValueError(
        f"Unsupported coordinate transform: {policy.coordinate_transform!r}"
    )


def transform_world_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source = np.asarray(points)
    if source.shape[-1] != 3 or not np.isfinite(source).all():
        raise ValueError("World points must be finite XYZ vectors")
    return (source @ np.asarray(rotation).T).astype(source.dtype, copy=False)


def transform_world_poses(poses: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source = np.asarray(poses)
    if source.ndim != 2 or source.shape[1] != 7 or not np.isfinite(source).all():
        raise ValueError("World poses must be a finite (frames, 7) XYZW+XYZ array")
    frame_rotation = Rotation.from_quat(source[:, POSE_QUATERNION])
    world_rotation = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64))
    quaternions = (world_rotation * frame_rotation).as_quat()
    for frame_id in range(1, len(quaternions)):
        if float(quaternions[frame_id] @ quaternions[frame_id - 1]) < 0.0:
            quaternions[frame_id] *= -1.0
    translations = source[:, POSE_TRANSLATION] @ np.asarray(rotation).T
    transformed = np.concatenate((quaternions, translations), axis=1)
    return transformed.astype(source.dtype, copy=False)


def object_name(meta: dict) -> str:
    raw_name = str(meta["object"])
    prefix, _, suffix = raw_name.partition("_")
    return suffix if prefix.isdigit() and suffix else raw_name
