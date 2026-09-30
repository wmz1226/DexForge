"""MANO mesh rendering for ContactAware."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.render import (
    DEFAULT_CAMERA_NAME,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    joint_qadr,
    scene_mesh_xml_for,
    render_scene_root,
    load_temp_scene_model,
    scene_options,
    fixed_camera,
)

import mujoco
import numpy as np

MANO_MESH_NAME = "mano_render_mesh"
MANO_MATERIAL_NAME = "mano_silver_material"
MANO_GEOM_NAME = "mano_visual"
MANO_BODY_NAME = "mano_render_body"
MANO_RGBA = "0.82 0.85 0.90 1"
MANO_SPECULAR = "0.9"
MANO_SHININESS = "0.8"

@dataclass(frozen=True)
class ManoRenderRequest:
    scene_xml: Path
    out_path: Path
    fps: float
    camera_pose: np.ndarray
    camera_fovy: float
    vertices: np.ndarray
    faces: np.ndarray
    object_pose: np.ndarray
    camera_name: str = DEFAULT_CAMERA_NAME
    height: int = VIDEO_HEIGHT
    width: int = VIDEO_WIDTH

def normalize_mano_request(request: ManoRenderRequest) -> ManoRenderRequest:
    vertices = np.asarray(request.vertices, dtype=np.float32)
    faces = np.asarray(request.faces, dtype=np.int32)
    object_pose = np.asarray(request.object_pose, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected MANO vertices shaped (T,V,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected MANO faces shaped (F,3), got {faces.shape}")
    if vertices.shape[0] != object_pose.shape[0]:
        raise ValueError("MANO/object trajectory length mismatch")
    return ManoRenderRequest(
        scene_xml=scene_mesh_xml_for(request.scene_xml),
        out_path=Path(request.out_path),
        fps=float(request.fps),
        camera_pose=np.asarray(request.camera_pose, dtype=np.float64).reshape(-1, 7)[0],
        camera_fovy=float(request.camera_fovy),
        vertices=vertices,
        faces=faces,
        object_pose=object_pose,
        camera_name=request.camera_name,
        height=int(request.height),
        width=int(request.width),
    )

def mano_obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    normals = vertex_normals(vertices, faces)
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines.extend(f"vn {x} {y} {z}" for x, y, z in normals)
    lines.extend(
        "f " + " ".join(f"{vertex + 1}//{vertex + 1}" for vertex in face)
        for face in faces
    )
    return ("\n".join(lines) + "\n").encode()

def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("MANO mesh contains a vertex without a valid normal")
    return normals / lengths

def load_mano_scene_model(
    request: ManoRenderRequest,
    mesh_path: Path | None = None,
) -> mujoco.MjModel:
    root = render_scene_root(request)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    mesh_file = str(mesh_path) if mesh_path is not None else f"{MANO_MESH_NAME}.obj"
    assets = None if mesh_path is not None else {
        mesh_file: mano_obj_bytes(request.vertices[0], request.faces)}
    ET.SubElement(asset, "mesh", name=MANO_MESH_NAME, file=mesh_file)
    ET.SubElement(
        asset, "material", name=MANO_MATERIAL_NAME, rgba=MANO_RGBA,
        specular=MANO_SPECULAR, shininess=MANO_SHININESS,
    )
    worldbody = root.find("worldbody")
    body = ET.SubElement(worldbody, "body", name=MANO_BODY_NAME)
    ET.SubElement(
        body, "geom", name=MANO_GEOM_NAME, type="mesh", mesh=MANO_MESH_NAME,
        material=MANO_MATERIAL_NAME, contype="0", conaffinity="0", group="1",
    )
    return load_temp_scene_model(root, request.scene_xml, assets=assets)

def stream_mano_frames(
    model: mujoco.MjModel,
    request: ManoRenderRequest,
    append_frame,
) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=request.height, width=request.width)
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, MANO_MESH_NAME)
    if mesh_id < 0:
        renderer.close()
        raise ValueError(f"Missing render mesh: {MANO_MESH_NAME}")
    qpos_samples = object_qpos_samples(model, request.object_pose)
    scene_opt = mano_scene_options(model)
    camera = fixed_camera(model, request.camera_name)
    try:
        for qpos, vertices in zip(qpos_samples, request.vertices):
            update_mano_mesh(model, mesh_id, vertices, faces=request.faces)
            mujoco.mjr_uploadMesh(model, renderer._mjr_context, mesh_id)
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_opt)
            append_frame(renderer.render().copy())
    finally:
        renderer.close()

def object_qpos_samples(model: mujoco.MjModel, poses: np.ndarray) -> np.ndarray:
    obj_qadr = joint_qadr(model, "obj_joint")
    samples = np.repeat(model.qpos0[None], poses.shape[0], axis=0)
    samples[:, obj_qadr:obj_qadr + 3] = poses[:, 4:7]
    samples[:, obj_qadr + 3:obj_qadr + 7] = poses[:, [3, 0, 1, 2]]
    return samples.astype(np.float32)

def update_mano_mesh(
    model: mujoco.MjModel,
    mesh_id: int,
    vertices: np.ndarray,
    *,
    faces: np.ndarray,
) -> None:
    vertex_adr = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    if vertices.shape[0] != vertex_count:
        raise ValueError(f"MANO vertex count mismatch: {vertices.shape[0]} vs {vertex_count}")
    rotation = mesh_rotation(model, mesh_id)
    position = np.asarray(model.mesh_pos[mesh_id])
    model.mesh_vert[vertex_adr:vertex_adr + vertex_count] = (vertices - position) @ rotation
    update_mano_normals(model, mesh_id, vertex_normals(vertices, faces) @ rotation)

def mesh_rotation(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(model.mesh_quat[mesh_id])
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()

def update_mano_normals(
    model: mujoco.MjModel,
    mesh_id: int,
    normals: np.ndarray,
) -> None:
    normal_adr = int(model.mesh_normaladr[mesh_id])
    normal_count = int(model.mesh_normalnum[mesh_id])
    if normal_count != normals.shape[0]:
        raise ValueError(f"MANO normal count mismatch: {normal_count} vs {normals.shape[0]}")
    model.mesh_normal[normal_adr:normal_adr + normal_count] = normals

def mano_scene_options(model: mujoco.MjModel) -> mujoco.MjvOption:
    options = scene_options(model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    object_body = int(model.jnt_bodyid[object_joint])
    mano_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MANO_BODY_NAME)
    object_root = int(model.body_rootid[object_body])
    mano_root = int(model.body_rootid[mano_body])
    visible_roots = {0, object_root, mano_root}
    for geom_id, body_id in enumerate(model.geom_bodyid):
        root_id = int(model.body_rootid[int(body_id)])
        if root_id not in visible_roots:
            model.geom_group[geom_id] = 0
    return options

def label_comparison(
    frame_o: np.ndarray,
    frame_p: np.ndarray,
    *,
    left_label: str,
    right_label: str,
) -> np.ndarray:
    import cv2

    frame = np.ascontiguousarray(np.concatenate([frame_o, frame_p], axis=1))
    cv2.putText(frame, left_label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, right_label, (frame_o.shape[1] + 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
