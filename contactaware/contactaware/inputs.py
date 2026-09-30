"""Load one sequence and extract its stable MANO contacts on the object."""

from __future__ import annotations

from pathlib import Path

import numpy as np


from contactaware.contact.anchors import make_mano_contact_anchors
from contactaware.contact.object_model import load_object_model
from contactaware.contact.surface import terminal_face_masks
from contactaware.models.hand_profiles import get_hand_profile
from contactaware.models.mano import (
    mano_refs_vertices,
    object_sequence,
    select_mano_vertex_fingers,
)
from contactaware.models.sequence import (
    canonicalize_world_trajectory,
    load_sequence_data,
    object_name,
)
from contactaware.solver.hand import MujocoHand
from contactaware.types import RetargetInputs
from contactaware.wrist_mount import mounted_robot_xml

def prepare_inputs(args) -> RetargetInputs:
    """MANO trajectory, object poses, robot hand, object collision model and stable MANO contacts of one sequence."""
    data, metadata = load_sequence_data(args.sequence_dir / "mano_raw")
    raw_mano = mano_refs_vertices(data, args)
    raw_obj_pose = object_sequence(data, raw_mano.frame_ids)
    raw_camera_pose = data["camera_pose"].astype(np.float32)
    mano, obj_pose, camera_pose = canonicalize_world_trajectory(
        raw_mano,
        raw_obj_pose,
        raw_camera_pose,
        data["sequence_policy"],
    )
    print(
        "  sequence policy: "
        f"{data['sequence_policy'].world_pose_policy}, "
        f"coordinates={data['sequence_policy'].coordinate_transform}"
    )
    hand = MujocoHand(get_hand_profile(args.hand, mounted_robot_xml(args.sequence_dir, args.hand)),
                      args.sphere_radius_scale)
    vertex_fingers = select_mano_vertex_fingers(
        mano.vertex_fingers,
        hand.profile.track_finger_ids,
    )
    pad, _ = terminal_face_masks(mano.surface_mapping.descriptors,
                                 args.pad_min_palmar)
    vertex_fingers = np.where(pad, vertex_fingers, -1)
    scene_xml = contactaware_scene_xml(args.sequence_dir, args.hand, object_name(metadata), args.scene)
    obj = load_object_model(scene_xml, topk=args.contact_topk,
                            distance_offset=args.contact_distance_offset)
    anchors = make_mano_contact_anchors(mano.vertices, obj_pose, vertex_fingers, obj=obj, args=args)
    print(f"  stable MANO contacts: {anchors.summary['segment_count']}")
    return RetargetInputs(
        mano_joints=mano.joints,
        mano_vertices=mano.vertices,
        mano_faces=mano.faces,
        mano_surface_mapping=mano.surface_mapping,
        obj_pose=obj_pose,
        camera_pose=camera_pose,
        camera_fovy=float(data["camera_info"]["fovy"]),
        hand=hand,
        obj=obj,
        anchors=anchors,
        sequence_policy=data["sequence_policy"],
    )


def contactaware_scene_xml(sequence_dir: Path, hand: str, object_label: str, scene: str) -> Path:
    candidate = Path(sequence_dir) / "scene" / hand / f"{object_label}_{scene}.xml"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing existing scene XML: {candidate}")
    return candidate.resolve()
