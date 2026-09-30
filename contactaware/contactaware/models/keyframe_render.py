"""Raw MANO and solved robot keys, one source-frame-numbered PNG per key."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from contactaware.utils.render import (
    MANO_MATERIAL_NAME,
    ManoRenderRequest,
    load_mano_scene_model,
    stream_mano_frames,
)
from tools.render import (
    TrajectoryRenderRequest,
    build_qpos,
    draw_video_overlay,
    load_scene_model_for_render,
    render_qpos_frames,
    scene_mesh_xml_for,
)
from contactaware.solver.pose import transform_qpos_sequence
from contactaware.models.mano import mano_refs_vertices, object_sequence
from contactaware.models.sequence import canonicalize_world_trajectory, load_sequence_data

PANEL_WIDTH = 1024
PANEL_HEIGHT = 768
MANO_GRAY = (0.42, 0.42, 0.42, 1.0)
MANO_SPECULAR = 0.2
MANO_SHININESS = 0.3
KEYFRAME_DIRECTORY = "keyframe"


def raw_key_request(args, frames, scene_xml, output):
    data, meta = load_sequence_data(Path(args.sequence_dir) / "mano_raw")
    mano = mano_refs_vertices(data, args)
    objects = object_sequence(data, mano.frame_ids)
    mano, objects, cameras = canonicalize_world_trajectory(
        mano, objects, data["camera_pose"], data["sequence_policy"])
    request = ManoRenderRequest(
        scene_xml=scene_mesh_xml_for(Path(scene_xml)), out_path=output,
        fps=args.video_fps, camera_pose=cameras[0],
        camera_fovy=float(meta["camera"]["fovy"]),
        vertices=mano.vertices[frames], faces=mano.faces,
        object_pose=objects[frames], width=PANEL_WIDTH, height=PANEL_HEIGHT)
    return request, np.asarray(mano.frame_ids)[frames]


def robot_panels(request, hand_qpos):
    robot_request = TrajectoryRenderRequest(
        scene_xml=request.scene_xml, out_path=request.out_path,
        fps=request.fps, camera_pose=request.camera_pose, camera_fovy=request.camera_fovy,
        width=request.width, height=request.height)
    model = load_scene_model_for_render(robot_request)
    model.vis.global_.offwidth = request.width
    model.vis.global_.offheight = request.height
    samples = np.asarray([build_qpos(model, state, pose)
                          for state, pose in zip(hand_qpos, request.object_pose)])
    return render_qpos_frames(model, samples, robot_request)


def write_comparisons(request, source_frames, panels, hand_name):
    pairs = iter(zip(source_frames, panels))

    def save_frame(mano_frame):
        frame_id, robot_frame = next(pairs)
        left = draw_video_overlay(mano_frame, "Raw MANO")
        right = draw_video_overlay(robot_frame, f"ContactAware {hand_name}")
        imageio.imwrite(request.out_path / f"{int(frame_id)}.png",
                        np.concatenate((left, right), axis=1))

    model = load_mano_scene_model(request)
    model.vis.global_.offwidth = request.width
    model.vis.global_.offheight = request.height
    material = model.material(MANO_MATERIAL_NAME)
    material.rgba, material.specular, material.shininess = MANO_GRAY, MANO_SPECULAR, MANO_SHININESS
    stream_mano_frames(model, request, save_frame)


def render_keyframes(args, hand, frames, qpos_obj, scene_xml, *, output=None):
    """Reuse exact solved states; express them at the raw object's pose for comparison."""
    destination = Path(args.output_dir) / KEYFRAME_DIRECTORY if output is None else Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    request, source_frames = raw_key_request(args, frames, scene_xml, destination)
    world_states = transform_qpos_sequence(hand, qpos_obj, request.object_pose)
    panels = robot_panels(request, world_states)
    write_comparisons(request, source_frames, panels, args.hand)
    names = {f"{int(frame)}.png" for frame in source_frames}
    for old_image in destination.glob("*.png"):
        if old_image.name not in names:
            old_image.unlink()
    print(f"[keyframes] {destination}: {len(source_frames)} images; raw frames {source_frames.tolist()}", flush=True)
    return destination


