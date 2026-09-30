"""Grasp initialization, joint key refinement, trajectory SQP and output."""

from copy import copy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from contactaware.contact.guidance import HandContactSequence
from contactaware.contact.runtime_guidance import build_runtime_contact_guidance
from contactaware.initialization.fixed_key_trajectory import KeyPalmObjective
from contactaware.initialization.key_state import ContactKeySeed
from contactaware.initialization.multi_contact_workflow import (
    initialize_key_trajectory,
    solve_keys,
)
from contactaware.inputs import prepare_inputs
from contactaware.metrics import (
    anchor_tracking_metrics,
    build_metrics_report,
    qp_tracking_metrics,
)
from contactaware.models.hand_profiles import supported_hands
from contactaware.models.keyframe_render import render_keyframes
from contactaware.output import save_contactaware_result
from contactaware.keys.dataset_seed import initialize_raw
from contactaware.keys.force import bounded_continuous_temporal_problem
from contactaware.keys.key_sqp import solve_conditional_keys
from contactaware.keys.surface_retraction import solve_surface_keys
from contactaware.solver.capsules import exclude_structural_overlaps
from contactaware.solver.contact_tracking import ContactTrackingObjective
from contactaware.solver.ground import postprocess_world_pose
from contactaware.wrist_mount import mount_trajectory
from contactaware.solver.pose import transform_qpos, transform_qpos_sequence
from contactaware.solver.retarget import (
    build_retarget_runtime,
    interpolation_ratio,
)
from contactaware.solver.temporal import (
    JOINT_ACCELERATION_MULTIPLIER,
    PALM_ACCELERATION_MULTIPLIER,
    physical_motion_operator,
)
from contactaware.solver.trajectory_qp import source_problems
from contactaware.trajectory.branch import unwrap_base_rotation
from contactaware.trajectory.demonstration_warp import warp_free_motion
from contactaware.trajectory.solve import solve as feasible_solve


def normalize_args(args):
    normalized = copy(args)
    normalized.sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    normalized.hand = args.hand.lower()
    if normalized.hand not in supported_hands():
        raise ValueError(f"contactaware supports: {', '.join(supported_hands())}; got {normalized.hand}")
    output = args.output_dir or normalized.sequence_dir / "retarget" / normalized.hand / "contactaware"
    normalized.output_dir = Path(output).expanduser().resolve()
    return normalized


def run(args):
    """Cold-start a complete sequence and write arrays, contact guidance, metrics, keyframes and video."""
    timings = {}
    started = time.perf_counter()
    args = normalize_args(args)
    inputs = prepare_inputs(args)
    if not (args.contact_anchor_weight > 0.0 and np.any(inputs.anchors.mask)):
        raise RuntimeError("ContactAware requires stable MANO contacts and a positive contact weight")
    reports = {"pipeline": dict(timings=timings)}
    reports["pipeline"]["excluded_capsule_pairs"] = exclude_structural_overlaps(inputs.hand)
    timings["preparation"] = time.perf_counter() - started

    stage = time.perf_counter()
    key_initialization, seed_report = initialize_raw(inputs, args)
    timings["seed"] = time.perf_counter() - stage
    stage = time.perf_counter()
    hand = inputs.hand
    acceleration_weights = np.full(hand.qpos_dim, JOINT_ACCELERATION_MULTIPLIER)
    acceleration_weights[: hand.base_qpos_dim] = PALM_ACCELERATION_MULTIPLIER
    motion_operator_factory = partial(
        physical_motion_operator, acceleration_multiplier=acceleration_weights
    )
    origin_rotation = hand.palm_pose_jacobian(key_initialization.initial[0])[1].copy()
    seed_report["trajectory_motion"] = dict(
        coordinates="object",
        acceleration_multiplier=acceleration_weights.tolist(),
        shared_by="joint keys and whole-trajectory QP",
    )
    _, _, _, indices, runtime = build_retarget_runtime(inputs, args)
    joint_problem = bounded_continuous_temporal_problem(
        key_initialization,
        inputs,
        args,
        runtime,
        indices,
        operator_factory=motion_operator_factory,
        origin_rotation=origin_rotation,
    )
    reports["initialization"] = seed_report
    timings["key_preparation"] = time.perf_counter() - stage

    stage = time.perf_counter()
    key_solver = partial(solve_surface_keys, solve=solve_conditional_keys)
    key_solution = solve_keys(
        joint_problem, joint_solver=key_solver, report_callback=partial(reports.__setitem__, "keys")
    )
    timings["keys"] = time.perf_counter() - stage
    stage = time.perf_counter()
    render_keyframes(
        args, hand, key_initialization.schedule.frames, key_solution.states, inputs.obj.xml_path
    )
    timings["keyframe_output"] = time.perf_counter() - stage

    stage = time.perf_counter()
    ratio = interpolation_ratio(args.video_fps, args.internal_fps)
    frames = key_initialization.schedule.frames
    frame = int(frames[0])
    seed = ContactKeySeed(
        transform_qpos(hand, key_solution.states[0], inputs.obj_pose[frame]),
        key_initialization.query_ids,
        np.arange(len(key_initialization.query_ids)),
        key_solution.anchors - key_initialization.references,
        key_solution.normals,
        frame,
        frame * ratio,
    )
    palm = KeyPalmObjective.from_keys(inputs, frames, key_solution.states, ratio)
    confidence = np.max(
        runtime.participation[indices] * runtime.guidance_blend[indices], axis=1, initial=0.0
    )
    initial = initialize_key_trajectory(
        inputs, frames, key_solution.states, ratio, palm=palm, confidence=confidence
    )
    initial, reports["pipeline"]["unwrapped_initial_frames"] = unwrap_base_rotation(hand, initial)
    keys = dict(frames=frames, qpos_obj=initial[frames].copy())
    tracking = ContactTrackingObjective(palm.target, args.trajectory_contact_position_multiplier)
    timings["trajectory_initialization"] = time.perf_counter() - stage
    stage = time.perf_counter()
    qpos, runtime = solve_trajectory(
        inputs,
        args,
        origin=origin_rotation,
        initial_states=initial,
        keys=keys,
        tracking=tracking,
        report_callback=partial(reports.__setitem__, "trajectory"),
        timings=timings,
        runtime_adapter=seed.adapt_runtime,
    )
    active_query_ids = runtime.hand_query_ids[runtime.anchors.mask]
    print(f"  QP-prior mapped robot contact queries: {np.unique(active_query_ids).size} unique")
    qp_seconds = time.perf_counter() - stage
    ratio, qp_tracking_report = qp_tracking_metrics(inputs, qpos, runtime, args=args)

    world_pose_start = time.perf_counter()
    qpos, obj_pose, ground_report = world_pose(args, inputs, qpos=qpos, reports=reports["pipeline"])
    world_pose_seconds = time.perf_counter() - world_pose_start
    guidance, guidance_report = build_runtime_contact_guidance(
        HandContactSequence(inputs.hand, qpos, obj_pose, inputs.obj), runtime, ratio
    )
    mano_anchor_tracking = anchor_tracking_metrics(
        inputs, runtime, qpos=qpos, obj_pose=obj_pose, ratio=ratio
    )
    report = build_metrics_report(
        args,
        inputs,
        mano_anchor_tracking=mano_anchor_tracking,
        guidance_report=guidance_report,
        ground_report=ground_report,
        qp_tracking_report=qp_tracking_report,
        initialization_seconds=sum(timings[name] for name in
                                   ("seed", "key_preparation", "trajectory_initialization")),
        key_seconds=timings["keys"],
        qp_seconds=qp_seconds,
        world_pose_seconds=world_pose_seconds,
    )
    key_qpos = keys["qpos_obj"]
    if args.wrist_mount:
        qpos, (key_qpos,), reports["wrist_mount"] = mount_trajectory(
            args.sequence_dir, args.hand, inputs.hand, qpos, (key_qpos,)
        )
        print("[wrist-mount]", reports["wrist_mount"], flush=True)
    out = save_contactaware_result(
        args,
        inputs,
        object_pose=obj_pose,
        hand_qpos=qpos,
        guidance={**guidance, "keyframe_indices": frames, "keyframe_qpos_obj": key_qpos},
        report={**report, "solver": reports},
    )
    print(f"[done] output={out}")
    timings["trajectory_postprocess_output"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - started
    (args.output_dir / "stage_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    return args.output_dir


def solve_trajectory(
    inputs, args, *, origin, initial_states, keys, tracking, report_callback, timings, runtime_adapter
):
    """Solve between fixed feasible keys, preserving their states and runtime targets."""
    started = time.perf_counter()
    cfg, joints, poses, indices, runtime = build_retarget_runtime(
        inputs, args, adapter=runtime_adapter
    )
    problems = source_problems(inputs, cfg, joints, poses, indices, runtime, None, None)
    trajectory_case = SimpleNamespace(
        hand=inputs.hand,
        inputs=inputs,
        initial=initial_states,
        frames=keys["frames"],
        args=args,
        problems=problems,
        tracking=tracking,
        origin=origin,
    )
    trajectory_case.initial, warped_frames = warp_free_motion(trajectory_case)
    states, report = feasible_solve(trajectory_case)
    report["demonstration_warped_frames"] = warped_frames
    released_frames = set(report["released_keys"])
    kept_key_indices = [
        i for i, frame in enumerate(keys["frames"]) if int(frame) not in released_frames
    ]
    if not np.array_equal(
        states[keys["frames"][kept_key_indices]], keys["qpos_obj"][kept_key_indices]
    ):
        raise RuntimeError("Trajectory solver changed a fixed feasible key state")
    report["seconds"] = time.perf_counter() - started
    timings["trajectory"] = report["seconds"]
    report_callback(report)
    return transform_qpos_sequence(inputs.hand, states, inputs.obj_pose), runtime


def world_pose(args, inputs, *, qpos, reports):
    """Apply the dataset ground policy and keep a continuous base-rotation branch."""
    qpos, obj_pose, report = postprocess_world_pose(args, inputs, qpos=qpos, xml_path=inputs.obj.xml_path)
    qpos, reports["unwrapped_output_frames"] = unwrap_base_rotation(inputs.hand, qpos)
    return qpos.astype(np.float32), obj_pose, report
