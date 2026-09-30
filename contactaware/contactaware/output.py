"""Persist retargeting arrays, reports, and the final trajectory video."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation

from contactaware.utils.render import (
    ManoRenderRequest,
    label_comparison,
    load_mano_scene_model,
    normalize_mano_request,
    stream_mano_frames,
)
from tools.render import (
    TrajectoryRenderRequest,
    load_scene_model_for_render,
    normalize_render_request,
    render_qpos_frames,
    request_qpos_samples,
    scene_mesh_xml_for,
)

from contactaware.settings import CONTACT_GUIDANCE_NAME, config_report
from contactaware.types import RetargetInputs


def save_contactaware_result(
    args,
    inputs: RetargetInputs,
    *,
    object_pose: np.ndarray,
    hand_qpos: np.ndarray,
    guidance: dict,
    report: Mapping[str, object],
) -> Path:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = write_result_arrays(
        out,
        args,
        inputs,
        object_pose=object_pose,
        hand_qpos=hand_qpos,
        guidance=guidance,
    )
    paths["config"].write_text(
        json.dumps(run_config(args, inputs), indent=2, default=json_default)
    )
    paths["metrics"].write_text(
        json.dumps(report, indent=2, default=json_default)
    )
    render_retarget_video(
        args,
        inputs,
        hand_qpos=hand_qpos,
        object_pose=object_pose,
        out_path=paths["video"],
        fps=args.video_fps,
    )
    return out


def write_result_arrays(
    out: Path,
    args,
    inputs: RetargetInputs,
    *,
    object_pose: np.ndarray,
    hand_qpos: np.ndarray,
    guidance: dict,
) -> dict[str, Path]:
    paths = {
        "config": out / "config.json",
        "camera_pose": out / "camera_pose_7.npy",
        "metrics": out / "metrics.json",
        "contact_guidance": out / CONTACT_GUIDANCE_NAME,
        "hand_qpos": out / f"{args.hand}_qpos.npy",
        "object_pose": out / "object_pose_7.npy",
        "video": out / "retarget.mp4",
    }
    np.save(paths["camera_pose"], inputs.camera_pose.astype(np.float32))
    np.save(paths["hand_qpos"], hand_qpos.astype(np.float32))
    np.save(paths["object_pose"], object_pose.astype(np.float32))
    np.savez(paths["contact_guidance"], **guidance)
    return paths


def run_config(args, inputs: RetargetInputs) -> dict:
    return {
        "sequence_dir": str(Path(args.sequence_dir).resolve()),
        "hand": args.hand,
        "hand_qpos_dim": int(inputs.hand.qpos_dim),
        "base_qpos_dim": int(inputs.hand.base_qpos_dim),
        "output_dir": str(Path(getattr(args, "result_output_dir", args.output_dir)).resolve()),
        "scene_xml": str(inputs.obj.xml_path),
        "render_xml": str(scene_mesh_xml_for(inputs.obj.xml_path)),
        "collision_target": inputs.obj.query_fn.kind,
        "sequence_policy": asdict(inputs.sequence_policy),
        "hyperparameter_source": args.hyperparameter_source,
        "hyperparameters": config_report(args),
    }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _align_to_object(
    vertices: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    """Move MANO vertices with the rigid motion the object underwent in the world-pose postprocess."""
    vertices = np.asarray(vertices)
    source_pose = np.asarray(source_pose)
    target_pose = np.asarray(target_pose)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected vertices shaped (T,V,3), got {vertices.shape}")
    frame_count = vertices.shape[0]
    expected_pose_shape = (frame_count, 7)
    if source_pose.shape != expected_pose_shape:
        raise ValueError(
            f"Expected source object poses shaped {expected_pose_shape}, "
            f"got {source_pose.shape}"
        )
    if target_pose.shape != expected_pose_shape:
        raise ValueError(
            f"Expected target object poses shaped {expected_pose_shape}, "
            f"got {target_pose.shape}"
        )

    # Object poses use SciPy's XYZW convention throughout ContactAware.
    source_rotation = Rotation.from_quat(source_pose[:, :4]).as_matrix()
    target_rotation = Rotation.from_quat(target_pose[:, :4]).as_matrix()
    rotation = target_rotation @ source_rotation.transpose(0, 2, 1)
    local = vertices - source_pose[:, None, 4:7]
    aligned = np.einsum("tij,tvj->tvi", rotation, local)
    aligned += target_pose[:, None, 4:7]
    return aligned.astype(vertices.dtype, copy=False)


def render_retarget_video(
    args,
    inputs: RetargetInputs,
    *,
    hand_qpos: np.ndarray,
    object_pose: np.ndarray,
    out_path: Path,
    fps: int,
) -> None:
    scene_xml = scene_mesh_xml_for(inputs.obj.xml_path)
    mano_request = normalize_mano_request(ManoRenderRequest(
        scene_xml=scene_xml, out_path=out_path, fps=fps,
        camera_pose=inputs.camera_pose[0], camera_fovy=float(inputs.camera_fovy),
        vertices=_align_to_object(inputs.mano_vertices, inputs.obj_pose, object_pose),
        faces=inputs.mano_faces, object_pose=object_pose))
    robot_request = normalize_render_request(TrajectoryRenderRequest(
        scene_xml=scene_xml, out_path=out_path, fps=fps,
        camera_pose=inputs.camera_pose[0], camera_fovy=float(inputs.camera_fovy),
        hand_qpos=hand_qpos, object_pose=object_pose))
    model = load_scene_model_for_render(robot_request)
    samples = request_qpos_samples(model, robot_request)
    frames = iter(render_qpos_frames(model, samples, robot_request))
    mano_model = load_mano_scene_model(mano_request)
    with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as writer:
        def append_frame(frame):
            writer.append_data(label_comparison(frame, next(frames),
                left_label="MANO", right_label=f"contactaware-{args.hand}"))
        stream_mano_frames(mano_model, mano_request, append_frame)
