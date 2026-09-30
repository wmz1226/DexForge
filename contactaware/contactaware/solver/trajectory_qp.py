"""Per-frame tracking problems of the trajectory solve."""


import numpy as np

from contactaware.contact.object_model import world_to_obj
from contactaware.solver.retarget import FrameProblem

IDENTITY_OBJECT_POSE = (0., 0., 0., 1., 0., 0., 0.)


def source_problems(inputs, cfg, joints, poses, indices, runtime, objective, safe_policy):
    return tuple(FrameProblem(
        hand=inputs.hand, cfg=cfg, obj=inputs.obj, runtime=runtime,
        joints=world_to_obj(joints[frame], poses[frame]), obj_pose=np.asarray(IDENTITY_OBJECT_POSE),
        frame_id=int(frame), additional_objective=objective,
        hand_object_safe_distance_policy=safe_policy,
    ) for frame in indices)


