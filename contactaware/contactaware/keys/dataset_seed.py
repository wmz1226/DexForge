"""Compact contact seed for the joint key problem; HOT3D transports its palm to every key's MANO palm."""

import time

from contactaware.contact.object_model import world_to_obj
from contactaware.initialization.key_state import select_contact_key
from contactaware.initialization.multi_contact_workflow import make_problem
from contactaware.keys.initialization import refined_contact_seed
from contactaware.solver.palm import human_palm_frame
from contactaware.solver.pose import fit_palm_pose
from contactaware.solver.retarget import interpolation_ratio


def initialize_raw(inputs, args):
    """Initialize keys from the compact seed; transport the palm for HOT3D."""
    ratio = interpolation_ratio(args.video_fps, args.internal_fps)
    selection = select_contact_key(inputs.anchors, interpolation_ratio=ratio)
    started = time.perf_counter()
    initial, report = refined_contact_seed(args, inputs, selection)
    seconds = time.perf_counter() - started
    print("[raw-seed]", dict(seconds=seconds, source_frame=selection.source_frame), flush=True)
    nominal = make_problem(inputs, args, initial)
    dataset = inputs.sequence_policy.dataset
    transport = dataset.casefold() != "dexycb"
    if transport:
        fingers = inputs.hand.profile.track_finger_ids
        source_joints = world_to_obj(inputs.mano_joints[selection.source_frame],
                                     inputs.obj_pose[selection.source_frame])
        human_position, human_rotation = human_palm_frame(source_joints, fingers)
        robot_position, robot_rotation, _, _ = inputs.hand.palm_pose_jacobian(initial)
        offset = human_rotation.T @ (robot_position - human_position)
        rotation_offset = human_rotation.T @ robot_rotation
        for index, joints in enumerate(nominal.joints_obj):
            palm_position, palm_rotation = human_palm_frame(joints, fingers)
            nominal.initial[index] = fit_palm_pose(inputs.hand, nominal.initial[index],
                palm_position + palm_rotation @ offset, palm_rotation @ rotation_offset)
    nominal.topologies = nominal.make_topologies(nominal.references)
    return nominal, dict(seconds=seconds, source="Current MANO", palm_transport=transport,
                         palm_transport_dataset=dataset,
                         joint_force_weight=args.force_closure_weight, report=report)
