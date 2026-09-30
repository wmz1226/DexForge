"""Initial resting correction for contact-aware retargeting."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from contactaware.settings import COMFREE_WARP_ROOT

if str(COMFREE_WARP_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFREE_WARP_ROOT))

import warp as wp
import comfree_warp

from contactaware.solver.trajectory_smoothing import smooth_object_poses, smoothing_report

loaded = Path(comfree_warp.__file__).resolve()
if COMFREE_WARP_ROOT not in loaded.parents:
    raise RuntimeError(f"Expected comfree_warp from {COMFREE_WARP_ROOT}, got {loaded}")

MM = 1000.0
BASE_QPOS_DIM = 6
RESTING_DELTA_DIM = 6
DELTA_POS = slice(0, 3)
DELTA_ROT = slice(3, 6)
DELTA_Z = 2
DELTA_TILT = slice(3, 5)
DELTA_YAW = 5
RESTING_CORRECTION_DOFS = ("world_z", "world_rotvec_x", "world_rotvec_y")
BASE_POSE_POS_SCALE = 1000.0
BASE_POSE_ROT_SCALE = 100.0
BASE_POSE_SOLVE_TOL = 1e-9
BASE_POSE_MAX_NFEV = 80
BASE_POSE_POS_TOL_M = 1e-5
BASE_POSE_ROT_TOL_RAD = 1e-5
FULL_ROTATION_RAD = 2.0 * np.pi


@dataclass(frozen=True)
class RestingIndex:
    base_qadr: np.ndarray
    finger_qadr: np.ndarray
    obj_qadr: int
    obj_vadr: int
    ctrl_base: np.ndarray
    ctrl_finger: np.ndarray
    base_jids: np.ndarray
    base_body_id: int


@dataclass(frozen=True)
class RestingCriteria:
    min_steps: int
    max_steps: int
    stable_steps: int
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    velocity_decay_rate_s: float
    velocity_decay_factor: float
    fail_on_timeout: bool


@dataclass(frozen=True)
class RestingResult:
    pose: np.ndarray
    velocity: np.ndarray
    steps: int
    stable_steps: int


def apply_initial_resting_correction(
    qpos: np.ndarray,
    obj_pose: np.ndarray,
    *,
    args,
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Settle the first frame once, then rigidly correct the full trajectory."""
    mx, model = load_resting_model(xml_path, args)
    idx = build_resting_idx(model)
    full_qpos = build_resting_qpos_batch(qpos, obj_pose, model=model, idx=idx)
    ctrl0 = build_resting_ctrl_batch(model, idx, qpos[:1])[0]
    criteria = make_resting_criteria(args, model.opt.timestep)
    rollout = make_resting_rollout(mx, model=model, idx=idx, criteria=criteria)
    initial_pose = full_qpos[0, idx.obj_qadr:idx.obj_qadr + 7]
    settled = run_resting_rollout(
        rollout,
        full_qpos[0],
        ctrl0,
        criteria=criteria,
        label="Initial resting rollout",
    )
    delta, natural_delta = settled_resting_delta(initial_pose, settled.pose)
    hand_out, obj_out = transform_sequence_with_resting_delta(
        full_qpos,
        delta,
        initial_pose[:3],
        model=model,
        idx=idx,
    )
    hand_out, obj_out, smoothing = smooth_object_and_transport_hand(
        hand_out,
        obj_out,
        model=model,
        idx=idx,
        smooth_weight=float(args.object_trajectory_smooth_weight),
    )
    summary = verify_and_summarize_resting(
        rollout,
        hand_out,
        obj_out,
        delta=delta,
        natural_delta=natural_delta,
        settled=settled,
        criteria=criteria,
        model=model,
        idx=idx,
        args=args,
    )
    summary["trajectory_smoothing"] = smoothing
    return hand_out.astype(np.float32), obj_out.astype(np.float32), summary


def verify_and_summarize_resting(
    rollout,
    hand: np.ndarray,
    obj_pose: np.ndarray,
    *,
    delta: np.ndarray,
    natural_delta: np.ndarray,
    settled: RestingResult,
    criteria: RestingCriteria,
    model: mujoco.MjModel,
    idx: RestingIndex,
    args,
) -> dict:
    verified_drift, verification = verify_resting_correction(
        rollout,
        hand,
        obj_pose,
        model=model,
        idx=idx,
        criteria=criteria,
    )
    summary = resting_summary(
        delta,
        natural_delta=natural_delta,
        settled=settled,
        verification=verification,
        verified_drift_mm=verified_drift,
        criteria=criteria,
        dt=model.opt.timestep,
        args=args,
    )
    return summary


def smooth_object_and_transport_hand(
    hand: np.ndarray,
    obj_pose: np.ndarray,
    *,
    model: mujoco.MjModel,
    idx: RestingIndex,
    smooth_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    target_obj = smooth_object_poses(obj_pose, weight=smooth_weight)
    full_qpos = build_resting_qpos_batch(hand, obj_pose, model=model, idx=idx)
    source = continuous_base_qpos_sequence(full_qpos, model, idx)
    transformed = source.astype(np.float32)
    data = mujoco.MjData(model)
    for frame_id in range(1, source.shape[0]):
        transformed[frame_id] = transform_frame_with_object_pose(
            source[frame_id],
            obj_pose[frame_id],
            target_obj[frame_id],
            model=model,
            data=data,
            idx=idx,
        )
    hand_out, obj_out = split_resting_qpos(idx, transformed)
    return hand_out, obj_out, smoothing_report(
        obj_pose, target_obj, weight=smooth_weight
    )


def transform_frame_with_object_pose(
    source_qpos: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
) -> np.ndarray:
    source_rotation = Rotation.from_quat(source_pose[:4]).as_matrix()
    target_rotation = Rotation.from_quat(target_pose[:4]).as_matrix()
    rotation_delta = target_rotation @ source_rotation.T
    position_delta = target_pose[4:7] - source_pose[4:7]
    transformed = np.asarray(source_qpos, dtype=np.float64).copy()
    transformed[idx.obj_qadr:idx.obj_qadr + 3] = target_pose[4:7]
    transformed[idx.obj_qadr + 3:idx.obj_qadr + 7] = target_pose[[3, 0, 1, 2]]
    transformed[idx.base_qadr] = solve_transformed_base_qpos(
        source_qpos,
        rotation_delta,
        position_delta,
        model=model,
        data=data,
        idx=idx,
        pivot=source_pose[4:7],
    )
    return transformed.astype(np.float32)


def load_resting_model(xml_path: Path, args):
    mx, model = comfree_warp.load_model(str(xml_path))
    model.opt.timestep = float(args.sim_dt)
    mx.opt.timestep.fill_(float(args.sim_dt))
    from comfree_warp.collision_config import CollisionConfig, configure_collision
    configure_collision(mx, CollisionConfig(args.contact_topk, args.contact_distance_offset))
    return mx, model


def build_resting_idx(model: mujoco.MjModel) -> RestingIndex:
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    if obj_jid < 0:
        raise ValueError("Missing joint: obj_joint")
    obj_qadr = int(model.jnt_qposadr[obj_jid])
    obj_vadr = int(model.jnt_dofadr[obj_jid])
    hand_joints = hand_joint_ids_before_object(model, obj_qadr)
    hand_qadr = np.asarray([model.jnt_qposadr[jid] for jid in hand_joints], np.int32)
    ctrl_qadr = resting_ctrl_indices(model, hand_joints)
    return RestingIndex(
        base_qadr=hand_qadr[:BASE_QPOS_DIM],
        finger_qadr=hand_qadr[BASE_QPOS_DIM:],
        obj_qadr=obj_qadr,
        obj_vadr=obj_vadr,
        ctrl_base=ctrl_qadr[:BASE_QPOS_DIM],
        ctrl_finger=ctrl_qadr[BASE_QPOS_DIM:],
        base_jids=np.asarray(hand_joints[:BASE_QPOS_DIM], np.int32),
        base_body_id=int(model.jnt_bodyid[hand_joints[BASE_QPOS_DIM - 1]]),
    )


def hand_joint_ids_before_object(model: mujoco.MjModel, obj_qadr: int) -> list[int]:
    joints = [
        jid for jid in range(model.njnt)
        if int(model.jnt_qposadr[jid]) < int(obj_qadr)
    ]
    joints.sort(key=lambda jid: int(model.jnt_qposadr[jid]))
    if len(joints) <= BASE_QPOS_DIM:
        raise ValueError(
            f"Expected hand joints before obj_joint, got {len(joints)}")
    if len(joints) != int(obj_qadr):
        raise ValueError(
            f"Only scalar hand joints are supported before obj_joint: "
            f"joint_count={len(joints)}, hand_qpos_dim={obj_qadr}")
    return joints


def resting_ctrl_indices(model: mujoco.MjModel, joint_ids: list[int]) -> np.ndarray:
    joint_to_act = resting_joint_actuator_map(model)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for jid in joint_ids
    ]
    missing = [name for name in names if name not in joint_to_act]
    if missing:
        raise ValueError(f"Missing actuators for hand joints: {missing}")
    return np.asarray([joint_to_act[name] for name in names], np.int32)


def resting_joint_actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    out = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise ValueError(f"Actuator {i} does not target a named joint")
        out[name] = i
    return out


def build_resting_qpos_batch(
    hand: np.ndarray,
    obj: np.ndarray,
    *,
    model: mujoco.MjModel,
    idx: RestingIndex,
) -> np.ndarray:
    hand_dim = len(idx.base_qadr) + len(idx.finger_qadr)
    if hand.shape[1] != hand_dim:
        raise ValueError(
            f"Hand qpos dim mismatch: scene expects {hand_dim}, "
            f"got {hand.shape[1]}"
        )
    qpos = np.zeros((hand.shape[0], model.nq), dtype=np.float32)
    qpos[:, idx.base_qadr] = hand[:, :BASE_QPOS_DIM]
    qpos[:, idx.finger_qadr] = hand[:, BASE_QPOS_DIM:]
    qpos[:, idx.obj_qadr:idx.obj_qadr + 3] = obj[:, 4:7]
    qpos[:, idx.obj_qadr + 3:idx.obj_qadr + 7] = obj[:, [3, 0, 1, 2]]
    return qpos


def build_resting_ctrl_batch(
    model: mujoco.MjModel, idx: RestingIndex, hand: np.ndarray,
) -> np.ndarray:
    hand_dim = len(idx.base_qadr) + len(idx.finger_qadr)
    if hand.shape[1] != hand_dim:
        raise ValueError(
            f"Hand qpos dim mismatch: scene expects {hand_dim}, "
            f"got {hand.shape[1]}"
        )
    ctrl = np.zeros((hand.shape[0], model.nu), dtype=np.float32)
    ctrl[:, idx.ctrl_base] = hand[:, :BASE_QPOS_DIM]
    ctrl[:, idx.ctrl_finger] = hand[:, BASE_QPOS_DIM:]
    return ctrl


def make_resting_criteria(args, dt: float) -> RestingCriteria:
    if dt <= 0.0:
        raise ValueError("Resting simulation timestep must be positive")
    durations = (
        float(args.resting_min_seconds),
        float(args.resting_max_seconds),
        float(args.resting_stable_seconds),
    )
    if durations[0] < 0.0 or durations[1] <= 0.0 or durations[2] <= 0.0:
        raise ValueError("Resting durations must be non-negative with positive limits")
    thresholds = (
        float(args.resting_linear_velocity_threshold),
        float(args.resting_angular_velocity_threshold),
    )
    if min(thresholds) <= 0.0:
        raise ValueError("Resting velocity thresholds must be positive")
    decay_rate = float(args.resting_velocity_decay_rate)
    if decay_rate <= 0.0:
        raise ValueError("Resting velocity decay rate must be positive")
    criteria = RestingCriteria(
        min_steps=int(np.ceil(durations[0] / dt)),
        max_steps=max(1, int(np.ceil(durations[1] / dt))),
        stable_steps=max(1, int(np.ceil(durations[2] / dt))),
        linear_velocity_m_s=thresholds[0],
        angular_velocity_rad_s=thresholds[1],
        velocity_decay_rate_s=decay_rate,
        velocity_decay_factor=float(np.exp(-decay_rate * dt)),
        fail_on_timeout=bool(args.resting_fail_on_timeout),
    )
    if criteria.min_steps + criteria.stable_steps > criteria.max_steps + 1:
        raise ValueError("Resting stability window does not fit within maximum time")
    return criteria


@wp.kernel
def _decay_object_velocity(qvel: wp.array2d(dtype=float), first: int, factor: float):
    component = wp.tid()
    qvel[0, first + component] *= factor


def next_stable_count(velocity, step_count, stable_count, criteria):
    stable = (np.linalg.norm(velocity[:3]) <= criteria.linear_velocity_m_s
              and np.linalg.norm(velocity[3:]) <= criteria.angular_velocity_rad_s)
    return stable_count + 1 if step_count >= criteria.min_steps and stable else 0


def collision_capacity_options(mx, model, data):
    """Reserve query-point contacts in addition to the standard MuJoCo budget."""
    graph = getattr(mx, "gaussian_collision", None)
    if graph is None:
        return {}
    batches = graph if isinstance(graph, tuple) else (graph,)
    contacts = sum(batch.contact_count for batch in batches)
    if not contacts:
        return {}
    dimensions = np.concatenate([batch.contact_dim.numpy() for batch in batches])
    rows = dimensions
    if model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        rows = np.where(dimensions > 1, 2 * (dimensions - 1), 1)
    return dict(nconmax=int(data.naconmax + contacts),
                njmax=int(data.njmax + rows.sum()))


def make_resting_rollout(
    mx,
    *,
    model: mujoco.MjModel,
    idx: RestingIndex,
    criteria: RestingCriteria,
):
    """Capture one hard-contact step; check the unchanged stop rule every step."""
    data = comfree_warp.make_data(model, nworld=1, comfree_model=mx)
    capacity = collision_capacity_options(mx, model, data)
    if capacity:
        data = comfree_warp.make_data(model, nworld=1, comfree_model=mx, **capacity)
    graph = None

    def step():
        comfree_warp.step(mx, data)
        wp.launch(_decay_object_velocity, dim=6,
                  inputs=[data.qvel, idx.obj_vadr, criteria.velocity_decay_factor])

    def reset(qpos, ctrl):
        comfree_warp.reset_data(mx, data)
        data.qpos.assign(np.asarray(qpos, np.float32)[None])
        data.ctrl.assign(np.asarray(ctrl, np.float32)[None])
        data.qvel.zero_()
        data.qfrc_applied.zero_()
        data.efc.force.zero_()
        data.efc.pos.zero_()

    def rollout(qpos, ctrl):
        nonlocal graph
        reset(qpos, ctrl)
        if graph is None:
            # Compile kernels before capture, then discard the warm-up step.
            step()
            reset(qpos, ctrl)
            with wp.ScopedCapture() as capture:
                step()
            graph = capture.graph
        stable_count = 0
        for step_count in range(1, criteria.max_steps + 1):
            wp.capture_launch(graph)
            velocity = data.qvel.numpy()[0, idx.obj_vadr:idx.obj_vadr + 6]
            if not np.isfinite(velocity).all():
                raise FloatingPointError("Non-finite velocity in Warp resting simulation")
            stable_count = next_stable_count(velocity, step_count, stable_count, criteria)
            if stable_count >= criteria.stable_steps:
                break
        pose = data.qpos.numpy()[0, idx.obj_qadr:idx.obj_qadr + 7]
        if not np.isfinite(pose).all():
            raise FloatingPointError("Non-finite pose in Warp resting simulation")
        return pose, velocity, step_count, stable_count

    return rollout


def run_resting_rollout(
    rollout,
    qpos: np.ndarray,
    ctrl: np.ndarray,
    *,
    criteria: RestingCriteria,
    label: str,
) -> RestingResult:
    pose, velocity, steps, stable_steps = rollout(
        np.asarray(qpos, np.float32), np.asarray(ctrl, np.float32)
    )
    result = RestingResult(
        pose=np.asarray(pose, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        steps=int(steps),
        stable_steps=int(stable_steps),
    )
    handle_resting_timeout(result, criteria, label=label)
    return result


def resting_result_converged(
    result: RestingResult,
    criteria: RestingCriteria,
) -> bool:
    return result.stable_steps >= criteria.stable_steps


def handle_resting_timeout(
    result: RestingResult,
    criteria: RestingCriteria,
    *,
    label: str,
) -> None:
    if resting_result_converged(result, criteria):
        return
    linear_speed = float(np.linalg.norm(result.velocity[:3]))
    angular_speed = float(np.linalg.norm(result.velocity[3:6]))
    message = (
        f"{label} did not stabilize within {result.steps} steps: "
        f"stable_window={result.stable_steps}/{criteria.stable_steps} steps, "
        f"linear={linear_speed * MM:.3f}mm/s "
        f"(limit={criteria.linear_velocity_m_s * MM:.3f}), "
        f"angular={angular_speed:.6f}rad/s "
        f"(limit={criteria.angular_velocity_rad_s:.6f}); "
        "using the terminal state"
    )
    if criteria.fail_on_timeout:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def settled_resting_delta(
    initial_pose: np.ndarray,
    settled_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initial_rotation = Rotation.from_quat(initial_pose[[4, 5, 6, 3]])
    settled_rotation = Rotation.from_quat(settled_pose[[4, 5, 6, 3]])
    rotation_delta = (settled_rotation * initial_rotation.inv()).as_rotvec()
    natural_delta = np.concatenate([
        settled_pose[:3] - initial_pose[:3],
        rotation_delta,
    ])
    correction = np.zeros(RESTING_DELTA_DIM, dtype=np.float64)
    correction[DELTA_Z] = natural_delta[DELTA_Z]
    correction[DELTA_TILT] = natural_delta[DELTA_TILT]
    return correction, natural_delta


def verify_resting_correction(
    rollout,
    hand: np.ndarray,
    obj_pose: np.ndarray,
    *,
    model: mujoco.MjModel,
    idx: RestingIndex,
    criteria: RestingCriteria,
) -> tuple[float, RestingResult]:
    qpos = build_resting_qpos_batch(
        hand[:1], obj_pose[:1], model=model, idx=idx
    )[0]
    ctrl = build_resting_ctrl_batch(model, idx, hand[:1])[0]
    initial_pose = qpos[idx.obj_qadr:idx.obj_qadr + 7]
    result = run_resting_rollout(
        rollout,
        qpos,
        ctrl,
        criteria=criteria,
        label="Resting correction verification",
    )
    drift_mm = float(np.linalg.norm((result.pose[:3] - initial_pose[:3]) * MM))
    return drift_mm, result


def transform_sequence_with_resting_delta(
    full_qpos: np.ndarray,
    delta: np.ndarray,
    pivot: np.ndarray,
    *,
    model: mujoco.MjModel,
    idx: RestingIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the settled rigid transform using the MuJoCo base parameterization."""
    delta = np.asarray(delta, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    rot_delta, quat_delta = resting_delta_np(delta)
    data = mujoco.MjData(model)
    source_sequence = continuous_base_qpos_sequence(full_qpos, model, idx)
    transformed = source_sequence.astype(np.float32)
    for frame_id, source_qpos in enumerate(source_sequence):
        target = transform_object_qpos(
            source_qpos,
            delta,
            pivot,
            idx=idx,
            rot_delta=rot_delta,
            quat_delta=quat_delta,
        )
        target[idx.base_qadr] = solve_transformed_base_qpos(
            source_qpos,
            rot_delta,
            delta[:3],
            model=model,
            data=data,
            idx=idx,
            pivot=pivot,
        )
        transformed[frame_id] = target.astype(np.float32)
    return split_resting_qpos(idx, transformed)


def continuous_base_qpos_sequence(
    full_qpos: np.ndarray,
    model: mujoco.MjModel,
    idx: RestingIndex,
) -> np.ndarray:
    continuous = np.asarray(full_qpos, dtype=np.float64).copy()
    lower, upper = base_qpos_bounds(model, idx)
    for local_id, (qadr, jid) in enumerate(zip(idx.base_qadr, idx.base_jids)):
        is_hinge = int(model.jnt_type[int(jid)]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        supports_period = upper[local_id] - lower[local_id] >= FULL_ROTATION_RAD
        if not is_hinge or not supports_period:
            continue
        values = np.unwrap(continuous[:, int(qadr)], period=FULL_ROTATION_RAD)
        continuous[:, int(qadr)] = periodic_sequence_in_bounds(
            values,
            lower[local_id],
            upper[local_id],
        )
    return continuous


def periodic_sequence_in_bounds(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    min_turns = int(np.ceil((lower - minimum) / FULL_ROTATION_RAD))
    max_turns = int(np.floor((upper - maximum) / FULL_ROTATION_RAD))
    if min_turns > max_turns:
        # Near gimbal lock an Euler angle can drift by more than a turn without the palm rotating;
        # per frame, take the bounded equivalent angle nearest the previous frame (same FK).
        return nearest_bounded_equivalents(values, lower, upper)
    bounds_center = np.mean([lower, upper])
    values_center = np.mean([minimum, maximum])
    preferred = int(np.rint((bounds_center - values_center) / FULL_ROTATION_RAD))
    turns = int(np.clip(preferred, min_turns, max_turns))
    return values + turns * FULL_ROTATION_RAD


def nearest_bounded_equivalents(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    reference = 0.5 * (lower + upper)
    for t, value in enumerate(result):
        turns = np.arange(np.ceil((lower - value) / FULL_ROTATION_RAD),
                          np.floor((upper - value) / FULL_ROTATION_RAD) + 1)
        candidates = value + turns * FULL_ROTATION_RAD
        result[t] = reference = candidates[np.argmin(np.abs(candidates - reference))]
    return result


def resting_delta_np(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(np.asarray(delta[DELTA_ROT], dtype=np.float64))
    quat_xyzw = rotation.as_quat()
    return rotation.as_matrix(), quat_xyzw[[3, 0, 1, 2]]


def transform_object_qpos(
    qpos: np.ndarray,
    delta: np.ndarray,
    pivot: np.ndarray,
    *,
    idx: RestingIndex,
    rot_delta: np.ndarray,
    quat_delta: np.ndarray,
) -> np.ndarray:
    out = np.asarray(qpos, dtype=np.float64).copy()
    obj_pos = out[idx.obj_qadr:idx.obj_qadr + 3]
    obj_quat = out[idx.obj_qadr + 3:idx.obj_qadr + 7]
    out[idx.obj_qadr:idx.obj_qadr + 3] = (
        pivot + rot_delta @ (obj_pos - pivot) + delta[:3]
    )
    out[idx.obj_qadr + 3:idx.obj_qadr + 7] = normalize_quat_np(
        quat_mul_np(quat_delta, obj_quat))
    return out


def split_resting_qpos(
    idx: RestingIndex,
    transformed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hand_dim = len(idx.base_qadr) + len(idx.finger_qadr)
    hand = np.zeros((transformed.shape[0], hand_dim), dtype=np.float32)
    hand[:, :BASE_QPOS_DIM] = transformed[:, idx.base_qadr]
    hand[:, BASE_QPOS_DIM:] = transformed[:, idx.finger_qadr]

    obj = np.zeros((transformed.shape[0], 7), dtype=np.float32)
    obj[:, 4:7] = transformed[:, idx.obj_qadr:idx.obj_qadr + 3]
    obj[:, 0:4] = transformed[:, [
        idx.obj_qadr + 4, idx.obj_qadr + 5,
        idx.obj_qadr + 6, idx.obj_qadr + 3,
    ]]
    return hand, obj


def solve_transformed_base_qpos(
    source_qpos: np.ndarray,
    rot_delta: np.ndarray,
    delta_pos: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
    pivot: np.ndarray,
) -> np.ndarray:
    source_pos, source_mat = base_body_pose(
        source_qpos, model=model, data=data, idx=idx)
    target_pos = pivot + rot_delta @ (source_pos - pivot) + delta_pos
    target_mat = rot_delta @ source_mat
    return solve_base_pose(
        source_qpos, target_pos, target_mat, model=model, data=data, idx=idx)


def solve_base_pose(
    template_qpos: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
) -> np.ndarray:
    lower, upper = base_qpos_bounds(model, idx)
    guess = np.asarray(template_qpos[idx.base_qadr], dtype=np.float64)
    # Float32 trajectories can round a value a few ulps outside XML limits.
    guess = np.clip(guess, lower, upper)
    result = least_squares(
        lambda values: base_pose_residual(
            values,
            target_pos,
            target_mat,
            model=model,
            data=data,
            idx=idx,
            template_qpos=template_qpos,
        ),
        guess,
        bounds=(lower, upper),
        xtol=BASE_POSE_SOLVE_TOL,
        ftol=BASE_POSE_SOLVE_TOL,
        gtol=BASE_POSE_SOLVE_TOL,
        max_nfev=BASE_POSE_MAX_NFEV,
    )
    validate_base_pose_solution(
        result,
        target_pos,
        target_mat,
        model=model,
        data=data,
        idx=idx,
        template_qpos=template_qpos,
    )
    return np.asarray(result.x, dtype=np.float64)


def base_pose_residual(
    base_values: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
    template_qpos: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[idx.base_qadr] = base_values
    pos, mat = base_body_pose(qpos, model=model, data=data, idx=idx)
    rotvec = Rotation.from_matrix(target_mat @ mat.T).as_rotvec()
    return np.concatenate([
        (pos - target_pos) * BASE_POSE_POS_SCALE,
        rotvec * BASE_POSE_ROT_SCALE,
    ])


def validate_base_pose_solution(
    result,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
    template_qpos: np.ndarray,
) -> None:
    if not result.success:
        raise RuntimeError(f"Resting base pose solve failed: {result.message}")
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[idx.base_qadr] = result.x
    pos, mat = base_body_pose(qpos, model=model, data=data, idx=idx)
    pos_err = float(np.linalg.norm(pos - target_pos))
    rot_err = float(
        np.linalg.norm(Rotation.from_matrix(target_mat @ mat.T).as_rotvec())
    )
    if pos_err > BASE_POSE_POS_TOL_M or rot_err > BASE_POSE_ROT_TOL_RAD:
        raise RuntimeError(
            f"Resting base pose solve residual too large: "
            f"pos={pos_err * MM:.6f}mm rot={rot_err:.6g}rad")


def base_body_pose(
    qpos: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: RestingIndex,
) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return (
        data.xpos[idx.base_body_id].copy(),
        data.xmat[idx.base_body_id].reshape(3, 3).copy(),
    )


def base_qpos_bounds(
    model: mujoco.MjModel,
    idx: RestingIndex,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for jid in idx.base_jids.astype(np.int64):
        if bool(model.jnt_limited[int(jid)]):
            lower.append(float(model.jnt_range[int(jid), 0]))
            upper.append(float(model.jnt_range[int(jid), 1]))
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def normalize_quat_np(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat, dtype=np.float64) / (np.linalg.norm(quat) + 1e-12)


def resting_summary(
    delta: np.ndarray,
    *,
    natural_delta: np.ndarray,
    settled: RestingResult,
    verification: RestingResult,
    verified_drift_mm: float,
    criteria: RestingCriteria,
    dt: float,
    args,
) -> dict:
    settle_converged = resting_result_converged(settled, criteria)
    verification_converged = resting_result_converged(verification, criteria)
    converged = settle_converged and verification_converged
    return {
        "enabled": True,
        "source": (
            "velocity_converged_natural_settle"
            if converged else "best_effort_timeout_settle"
        ),
        "converged": converged,
        "correction_dofs": list(RESTING_CORRECTION_DOFS),
        "verified_terminal_pos_drift_mm": float(verified_drift_mm),
        "comfree_warp_root": str(COMFREE_WARP_ROOT),
        "comfree_warp_file": str(loaded),
        "simulator_name": "comfree_warp",
        "contact_parameter_source": "xml_solref_solimp",
        "sim_dt": float(args.sim_dt),
        "contact_topk": int(args.contact_topk),
        "contact_distance_offset": float(args.contact_distance_offset),
        **resting_transform_summary(delta, natural_delta),
        **resting_convergence_summary(settled, verification, criteria, dt),
        "contact_softness": None,
    }


def resting_transform_summary(
    delta: np.ndarray,
    natural_delta: np.ndarray,
) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    natural_delta = np.asarray(natural_delta, dtype=np.float64)
    return {
        "delta": delta.astype(float).tolist(),
        "delta_pos_mm": (delta[DELTA_POS] * MM).astype(float).tolist(),
        "delta_rot_rad": delta[DELTA_ROT].astype(float).tolist(),
        "natural_delta": natural_delta.astype(float).tolist(),
        "natural_delta_pos_mm": (
            natural_delta[DELTA_POS] * MM
        ).astype(float).tolist(),
        "natural_delta_rot_rad": (
            natural_delta[DELTA_ROT]
        ).astype(float).tolist(),
        "ignored_planar_translation_mm": (
            natural_delta[:2] * MM
        ).astype(float).tolist(),
        "ignored_yaw_rotvec_rad": float(natural_delta[DELTA_YAW]),
    }


def resting_convergence_summary(
    settled: RestingResult,
    verification: RestingResult,
    criteria: RestingCriteria,
    dt: float,
) -> dict:
    velocity = np.asarray(settled.velocity, dtype=np.float64)
    return {
        "settle_converged": resting_result_converged(settled, criteria),
        "verification_converged": resting_result_converged(
            verification, criteria
        ),
        "settled_linear_velocity_mm_s": (
            velocity[:3] * MM
        ).astype(float).tolist(),
        "settled_angular_velocity_rad_s": velocity[3:6].astype(float).tolist(),
        "terminal_linear_speed_mm_s": float(np.linalg.norm(velocity[:3]) * MM),
        "terminal_angular_speed_rad_s": float(np.linalg.norm(velocity[3:6])),
        "settle_seconds": float(settled.steps * dt),
        "settle_steps": int(settled.steps),
        "verification_seconds": float(verification.steps * dt),
        "verification_steps": int(verification.steps),
        "stable_window_seconds": float(criteria.stable_steps * dt),
        "stable_window_steps": int(criteria.stable_steps),
        "min_seconds": float(criteria.min_steps * dt),
        "max_seconds": float(criteria.max_steps * dt),
        "linear_velocity_threshold_mm_s": float(
            criteria.linear_velocity_m_s * MM
        ),
        "angular_velocity_threshold_rad_s": float(
            criteria.angular_velocity_rad_s
        ),
        "velocity_decay_rate_s": float(criteria.velocity_decay_rate_s),
    }
