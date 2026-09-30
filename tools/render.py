"""Scene, trajectory and camera rendering shared by ContactAware and ForceAware."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

OBJECT_POSE_DIM = 7
DEFAULT_CAMERA_NAME = "real_cam"
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640

@dataclass(frozen=True)
class TrajectoryRenderRequest:
    scene_xml: Path
    out_path: Path
    fps: float
    camera_pose: np.ndarray
    camera_fovy: float
    qpos: np.ndarray | None = None
    qpos_frames: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    object_pose: np.ndarray | None = None
    camera_name: str = DEFAULT_CAMERA_NAME
    height: int = VIDEO_HEIGHT
    width: int = VIDEO_WIDTH
    overlay_labels: list[str] | None = None

def joint_qadr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Missing joint: {name}")
    return int(model.jnt_qposadr[jid])

def qpos_indices(model: mujoco.MjModel) -> tuple[np.ndarray, int]:
    obj_qadr = joint_qadr(model, "obj_joint")
    if obj_qadr <= 0:
        raise ValueError(
            f"Expected hand qpos before obj_joint, got nq={model.nq}, obj_qadr={obj_qadr}"
        )
    return np.arange(obj_qadr, dtype=np.int32), obj_qadr

def build_qpos(model: mujoco.MjModel, hand: np.ndarray, obj: np.ndarray) -> np.ndarray:
    if obj.shape[0] < OBJECT_POSE_DIM:
        raise ValueError(
            f"Expected object pose with {OBJECT_POSE_DIM} values, got {obj.shape[0]}"
        )
    hand_qadr, obj_qadr = qpos_indices(model)
    if hand.shape[0] != hand_qadr.size:
        raise ValueError(
            f"Hand qpos dim mismatch: scene expects {hand_qadr.size}, got {hand.shape[0]}"
        )
    qpos = np.zeros(model.nq, dtype=np.float32)
    qpos[hand_qadr] = hand
    qpos[obj_qadr : obj_qadr + 3] = obj[4:7]
    qpos[obj_qadr + 3 : obj_qadr + 7] = obj[[3, 0, 1, 2]]
    return qpos

def scene_mesh_xml_for(scene_xml: Path) -> Path:
    """Visual mesh scene of a physics scene variant: ``<object>_<variant>.xml`` -> ``<object>_mesh.xml``."""
    scene_xml = Path(scene_xml)
    if scene_xml.name.endswith(("_mesh.xml", "_mesh_real.xml")):
        require_file(scene_xml, "scene mesh XML")
        return scene_xml
    stem, real = scene_xml.name[: -len(".xml")], ""
    if stem.endswith("_real"):
        stem, real = stem[: -len("_real")], "_real"
    if "_" not in stem:
        raise ValueError(f"Expected <object>_<variant>.xml scene XML, got {scene_xml}")
    mesh_xml = scene_xml.with_name(f"{stem.rsplit('_', 1)[0]}_mesh{real}.xml")
    require_file(mesh_xml, "scene mesh XML")
    return mesh_xml

def camera_xml_from_pose(
    pose: np.ndarray, fovy: float, name: str = DEFAULT_CAMERA_NAME
) -> str:
    from scipy.spatial.transform import Rotation

    pose = np.asarray(pose, dtype=np.float64)
    r_wc = Rotation.from_quat(pose[:4]).as_matrix()
    r_mujoco = r_wc @ np.diag([1.0, -1.0, -1.0])
    q_xyzw = Rotation.from_matrix(r_mujoco).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    pos = pose[4:7]
    return (
        f'<camera name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" '
        f'quat="{q_wxyz[0]} {q_wxyz[1]} {q_wxyz[2]} {q_wxyz[3]}" '
        f'fovy="{float(fovy)}"/>'
    )

def load_scene_model_for_render(request: TrajectoryRenderRequest) -> mujoco.MjModel:
    root = render_scene_root(request)
    return load_temp_scene_model(root, Path(request.scene_xml))

def render_scene_root(request) -> ET.Element:
    root = ET.fromstring(Path(request.scene_xml).read_text())
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Render XML has no worldbody: {request.scene_xml}")
    remove_named_camera(root, request.camera_name)
    camera_xml = camera_xml_from_pose(
        request.camera_pose, request.camera_fovy, request.camera_name
    )
    worldbody.insert(0, ET.fromstring(camera_xml))
    return root

def remove_named_camera(root: ET.Element, name: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child.attrib.get("name") == name:
                parent.remove(child)

def load_temp_scene_model(
    root: ET.Element, source_xml: Path, *, assets=None
) -> mujoco.MjModel:
    """Compile edited XML in memory, retaining the original asset directory."""
    root = deepcopy(root)
    directory = Path(source_xml).resolve().parent
    compiler = root.find("compiler")
    directories = {} if compiler is None else compiler.attrib
    for element in root.iter():
        filename = element.get("file")
        if filename is None or (assets is not None and filename in assets):
            continue
        asset_dir = directories.get(
            f"{element.tag}dir", directories.get("assetdir", "")
        )
        element.set("file", str((directory / asset_dir / filename).resolve()))
    return mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=assets
    )

def draw_video_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame.copy())
    cv2.putText(
        out, text, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        out,
        text,
        (10, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out

def render_scene_trajectory_video(request: TrajectoryRenderRequest) -> None:
    request = normalize_render_request(request)
    request.out_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_scene_model_for_render(request)
    qpos_samples = request_qpos_samples(model, request)
    frames = render_qpos_frames(model, qpos_samples, request)
    imageio.mimsave(str(request.out_path), frames, fps=request.fps, macro_block_size=1)

def normalize_render_request(
    request: TrajectoryRenderRequest,
) -> TrajectoryRenderRequest:
    return TrajectoryRenderRequest(
        scene_xml=scene_mesh_xml_for(request.scene_xml),
        out_path=Path(request.out_path),
        fps=float(request.fps),
        camera_pose=np.asarray(request.camera_pose, dtype=np.float64).reshape(-1, 7)[0],
        camera_fovy=float(request.camera_fovy),
        qpos=None
        if request.qpos is None
        else np.asarray(request.qpos, dtype=np.float32),
        qpos_frames=None
        if request.qpos_frames is None
        else np.asarray(request.qpos_frames, dtype=np.float32),
        hand_qpos=None
        if request.hand_qpos is None
        else np.asarray(request.hand_qpos, dtype=np.float32),
        object_pose=None
        if request.object_pose is None
        else np.asarray(request.object_pose, dtype=np.float32),
        camera_name=request.camera_name,
        height=int(request.height),
        width=int(request.width),
        overlay_labels=request.overlay_labels,
    )

def request_qpos_samples(
    model: mujoco.MjModel, request: TrajectoryRenderRequest
) -> np.ndarray:
    if request.qpos is not None:
        return full_qpos_samples(model, request.qpos, request.qpos_frames)
    if request.hand_qpos is None or request.object_pose is None:
        raise ValueError(
            "Trajectory render requires either qpos or hand_qpos/object_pose"
        )
    return hand_object_qpos_samples(model, request.hand_qpos, request.object_pose)

def full_qpos_samples(
    model: mujoco.MjModel, qpos: np.ndarray, qpos_frames: np.ndarray | None
) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"Full qpos trajectory shape {qpos.shape} does not match model.nq={model.nq}"
        )
    if qpos_frames is None:
        return qpos
    return video_qpos_samples(qpos, qpos_frames, joint_qadr(model, "obj_joint"))

def trajectory_time_frame_coordinates(
    sample_count: int,
    sample_dt: float,
    video_fps: float,
) -> np.ndarray:
    if sample_count < 1:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError(f"sample_dt must be positive and finite, got {sample_dt}")
    if not np.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError(f"video_fps must be positive and finite, got {video_fps}")
    sample_times = np.arange(sample_count, dtype=np.float64) * sample_dt
    return sample_times * video_fps

def hand_object_qpos_samples(
    model: mujoco.MjModel, hand: np.ndarray, obj: np.ndarray
) -> np.ndarray:
    if hand.shape[0] != obj.shape[0]:
        raise ValueError(
            f"Hand/object trajectory length mismatch: {hand.shape[0]} vs {obj.shape[0]}"
        )
    return np.stack([build_qpos(model, hand[t], obj[t]) for t in range(hand.shape[0])])

def render_qpos_frames(
    model: mujoco.MjModel, qpos_samples: np.ndarray, request: TrajectoryRenderRequest
) -> list[np.ndarray]:
    labels = request.overlay_labels
    if labels is not None and len(labels) != qpos_samples.shape[0]:
        raise ValueError("Overlay label count must match rendered frame count")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=request.height, width=request.width)
    frames = []
    try:
        scene_opt = scene_options(model)
        cam = fixed_camera(model, request.camera_name)
        for frame_id, qpos in enumerate(qpos_samples):
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=scene_opt)
            frame = renderer.render().copy()
            if labels is not None:
                label = f"f {frame_id:03d}  anchor: {labels[frame_id]}"
                frame = draw_video_overlay(frame, label)
            frames.append(frame)
        return frames
    finally:
        renderer.close()

def quat_nlerp_wxyz(q0, q1, frac: float) -> np.ndarray:
    q0 = np.asarray(q0, np.float64)
    q1 = np.asarray(q1, np.float64)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    quat = (1.0 - frac) * q0 + frac * q1
    return (quat / (np.linalg.norm(quat) + 1e-12)).astype(np.float32)

def interp_qpos_at_frame(
    qpos_traj,
    qpos_frames,
    frame: float,
    *,
    obj_qadr: int,
) -> np.ndarray:
    frames = np.asarray(qpos_frames, np.float32)
    if frame <= float(frames[0]):
        return np.asarray(qpos_traj[0], np.float32).copy()
    if frame >= float(frames[-1]):
        return np.asarray(qpos_traj[-1], np.float32).copy()
    return interpolate_between_frames(qpos_traj, frames, frame, obj_qadr=obj_qadr)

def interpolate_between_frames(
    qpos_traj,
    frames,
    frame: float,
    *,
    obj_qadr: int,
) -> np.ndarray:
    hi = int(np.searchsorted(frames, frame, side="left"))
    lo = max(0, hi - 1)
    denom = float(frames[hi] - frames[lo])
    frac = 0.0 if abs(denom) < 1e-9 else float((frame - frames[lo]) / denom)
    qpos = ((1.0 - frac) * qpos_traj[lo] + frac * qpos_traj[hi]).astype(np.float32)
    qpos[obj_qadr + 3 : obj_qadr + 7] = quat_nlerp_wxyz(
        qpos_traj[lo, obj_qadr + 3 : obj_qadr + 7],
        qpos_traj[hi, obj_qadr + 3 : obj_qadr + 7],
        frac,
    )
    return qpos

def video_qpos_samples(qpos_traj, qpos_frames, obj_qadr: int) -> np.ndarray:
    start = int(np.ceil(float(qpos_frames[0]) - 1e-6))
    end = int(np.floor(float(qpos_frames[-1]) + 1e-6))
    frames = np.arange(start, end + 1, dtype=np.float32)
    return np.stack(
        [
            interp_qpos_at_frame(
                qpos_traj,
                qpos_frames,
                float(frame),
                obj_qadr=obj_qadr,
            )
            for frame in frames
        ]
    )

def scene_options(model: mujoco.MjModel) -> mujoco.MjvOption:
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        ungrouped_visual = (
            int(model.geom_group[gid]) == 0
            and int(model.geom_contype[gid]) == 0
            and int(model.geom_conaffinity[gid]) == 0
        )
        named_visual = (
            name == "floor"
            or name == "obj_visual"
            or name.endswith("_visual")
            or name.endswith("_tip")
        )
        if ungrouped_visual or named_visual:
            model.geom_group[gid] = 1
    scene_opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_opt)
    scene_opt.geomgroup[0] = 0
    scene_opt.geomgroup[1] = 1
    return scene_opt

def fixed_camera(model: mujoco.MjModel, name: str) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"Missing camera: {name}")
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    return cam

def require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
