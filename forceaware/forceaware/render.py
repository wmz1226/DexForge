"""ForceAware sequence camera lookup."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from tools.render import (
    OBJECT_POSE_DIM,
    require_file,
)

def sequence_camera_pose_and_fovy(
    seq: Path, *, camera_pose_path: Path | None = None
) -> tuple[np.ndarray, float]:
    pose_path = (
        Path(seq) / "mano_raw/camera_pose_7.npy"
        if camera_pose_path is None
        else Path(camera_pose_path)
    )
    meta_path = Path(seq) / "mano_raw/meta.json"
    require_file(pose_path, "camera pose")
    poses = np.load(pose_path)
    if poses.ndim != 2 or poses.shape[1] != OBJECT_POSE_DIM or poses.shape[0] == 0:
        raise ValueError(
            f"Expected camera poses with shape (frames, 7), got {poses.shape}"
        )
    pose = poses[0]
    meta = json.loads(meta_path.read_text())
    fovy_env = os.environ.get("FORCEAWARE_CAMERA_FOVY")
    fovy = float(fovy_env) if fovy_env is not None else float(meta["camera"]["fovy"])
    return pose, fovy

def contactaware_camera_pose_path(
    seq: Path, scene_xml: Path, *, hand_name: str | None = None
) -> Path:
    scene_xml = Path(scene_xml).resolve()
    scene_dir = Path(seq).resolve() / "scene"
    if hand_name is None:
        if scene_xml.parent.parent != scene_dir:
            raise ValueError(f"Expected scene XML under {scene_dir}, got {scene_xml}")
        hand_name = scene_xml.parent.name
    if not hand_name or Path(hand_name).name != hand_name:
        raise ValueError(f"Invalid hand name for camera lookup: {hand_name!r}")
    path = Path(seq) / "retarget" / hand_name / "contactaware/camera_pose_7.npy"
    require_file(path, "ContactAware camera pose")
    return path
