"""Build and solve the joint key problem; interpolate the keys into the initial trajectory."""

from functools import partial

import numpy as np
from scipy.interpolate import PchipInterpolator

from contactaware.contact.object_model import object_center
from contactaware.contact.object_model import world_to_obj
from comfree_warp.geometry import ObjectGeometry
from contactaware.initialization.centered_pad import centered_contact_query_ids
from contactaware.initialization.key_problem import RobotAwareKeySettings, SQP_STEP_TOLERANCE
from contactaware.settings import make_config
from contactaware.solver.grasp_sqp import ShrinkingTrustRegion
from contactaware.solver.palm_relative_shape import palm_relative_shape_residual, unguided_shape_fingers
from contactaware.solver.pre_contact import PreContactClearance
from contactaware.solver.interpolation import position_interpolator, rotation_interpolator
from contactaware.solver.pose import _fit_base_pose, WORLD_FRAME_LENGTH_SCALE_M
from contactaware.solver.qp import QPSolverOptions
from contactaware.solver.retarget import initial_qpos
from contactaware.solver.sparse_qp import make_joint_contact_qp
from contactaware.solver.tracking import palm_tracking_linearization
from contactaware.initialization.compact_kinematics import CompactKeyKinematics
from contactaware.initialization.multi_contact_keys import (
    MultiConstraints, MultiContactProblem, MultiObjective, select_transition_states,
)

PAD_CENTER = 0.5
XYZ_DIM = 3


def shape_terms(hand, state, joints, fingers):
    labels = unguided_shape_fingers(hand, fingers)
    return palm_relative_shape_residual(hand, state, joints, labels)


def make_problem(inputs, args, initial_state):
    cfg = make_config(args)
    settings = RobotAwareKeySettings(max_iterations=args.first_frame_max_iterations)
    schedule = select_transition_states(inputs.anchors.mask)
    columns = np.arange(inputs.anchors.mask.shape[1])
    query_ids = np.asarray([centered_contact_query_ids(inputs, np.asarray([column]),
        source_frame=int(schedule.frames[schedule.owners[column]]), axial_center=PAD_CENTER, cfg=cfg)[0]
        for column in columns])
    references = np.asarray([inputs.anchors.anchor_pos_obj[
        schedule.frames[schedule.owners[column]], column] for column in columns])
    states = np.repeat(initial_state[None], len(schedule.frames), axis=0)
    joints_obj = np.asarray([world_to_obj(inputs.mano_joints[frame], inputs.obj_pose[frame])
                             for frame in schedule.frames])
    if not len(schedule.columns[0]):
        states[0] = initial_qpos(inputs.hand, joints_obj[0], cfg=cfg)
    center = object_center(inputs.obj.xml_path)
    geometry = ObjectGeometry(inputs.obj.query_fn)
    return MultiContactProblem(hand=inputs.hand, cfg=cfg, settings=settings, schedule=schedule,
        query_ids=query_ids, initial=states, references=references, joints_obj=joints_obj,
        geometry=geometry, center=center, kinematics=CompactKeyKinematics(inputs.hand), score=None,
        palm_terms=partial(palm_tracking_linearization, inputs.hand),
        shape_terms=partial(shape_terms, inputs.hand), finger_ids=inputs.anchors.finger_ids,
        distribution_weights=(args.anchor_center_weight,
                              args.anchor_relative_weight,
                              args.force_closure_weight),
        pre_contact=PreContactClearance(inputs.hand, inputs.anchors, cfg.pre_contact_safe_distance,
                                        cfg.hand_object_safe_distance))


def solve_keys(problem, *, joint_solver, report_callback):
    cfg, settings = problem.cfg, problem.settings
    origin = np.zeros(problem.dimension)
    lower, upper = problem.bounds()
    options = QPSolverOptions(cfg.qp_absolute_tolerance, cfg.qp_relative_tolerance,
                              cfg.qp_validation_tolerance, cfg.qp_maxiter)
    coordinates, report = joint_solver(MultiObjective(problem), MultiConstraints(problem),
        None, origin, lower, upper, problem.trust(),
        solver=make_joint_contact_qp(options), regularization=cfg.min_hessian,
        tolerance=SQP_STEP_TOLERANCE, feasibility_tolerance=problem.feasibility_tolerance,
        max_iterations=settings.max_iterations, accept=lambda _: False,
        prepare_iteration=problem.prepare_iteration,
        update_trust=ShrinkingTrustRegion(settings.trust_shrink_trigger,
            settings.trust_shrink_factor, settings.minimum_trust_fraction))
    value = problem.evaluate(coordinates)
    report.update({"key_frames": problem.schedule.frames.tolist(),
        "active_columns": [columns.tolist() for columns in problem.schedule.columns],
        "anchor_owner_key": problem.schedule.owners.tolist(), "shared_anchor_count": len(value.anchors),
        "retained_collision_queries_per_key": [len(ids) for ids in problem.collision_working_set],
        "collision_working_set": "retain link contacts discovered at SQP iterates; trial evaluations add no rows",
        "normal_alignment": "same soft contact-normal objective as the full trajectory QP; facing cosine is diagnostic",
        "variables": problem.coordinate_description,
        "final_states": value.diagnostics, "sequential_qp_executed": False})
    report_callback(report)
    if not report["feasible"]:
        print(f"[joint-keys] returning best computed state; constraints not satisfied: {report['max_normalized_constraint_violation']}", flush=True)
    return value


def interpolate_keys(hand, frames, states, frame_count):
    poses = []
    for state in states:
        hand.forward(state)
        poses.append((hand.data.xpos[hand.palm_body_id].copy(),
                      hand.data.xmat[hand.palm_body_id].reshape(XYZ_DIM, XYZ_DIM).copy()))
    if len(frames) == 1:
        return np.repeat(states, frame_count, axis=0)
    positions = position_interpolator(frames, [pose[0] for pose in poses])
    rotations = rotation_interpolator(frames, [pose[1] for pose in poses])
    joints = PchipInterpolator(frames, states[:, hand.base_qpos_dim:])
    output = np.repeat(states[:1], frame_count, axis=0)
    previous = states[0]
    for frame in range(frame_count):
        time_index = np.clip(frame, frames[0], frames[-1])
        state = np.r_[previous[:hand.base_qpos_dim], joints(time_index)]
        output[frame] = _fit_base_pose(hand, state, positions(time_index),
            rotations(time_index), WORLD_FRAME_LENGTH_SCALE_M)
        previous = output[frame]
    output[frames] = states
    return output


def initialize_key_trajectory(inputs, frames, states, ratio, *, palm, confidence,
                              interpolate=interpolate_keys):
    """Start the joint solve from continuous key geometry and demonstrated motion."""
    interpolated = interpolate(inputs.hand, frames, states, len(inputs.mano_joints))
    return palm.initial_states(inputs, frames, states, ratio,
                               interpolated=interpolated, confidence=confidence)

