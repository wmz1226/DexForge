"""Open-loop MuJoCo mesh replay of saved ForceAware controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from forceaware.render import (
    contactaware_camera_pose_path,
    sequence_camera_pose_and_fovy,
)
from tools.render import (
    TrajectoryRenderRequest,
    render_scene_trajectory_video,
    scene_mesh_xml_for,
)

DEFAULT_REPLAY_SIM_DT = 0.002
MUJOCO_CPU = "mujoco_cpu"
METERS_TO_MM = 1000.0
QUAT_EPS = 1e-12


def run_default_mujoco_replay(
    rollout: Path, scene_xml: Path, out_dir: Path | None = None,
    *, sequence_dir: Path | None = None, hand_name: str | None = None,
    camera_pose_path: Path | None = None,
) -> Path:
    rollout = Path(rollout).expanduser().resolve()
    scene_xml = scene_mesh_xml_for(Path(scene_xml).expanduser().resolve())
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    source = load_rollout(rollout, scene_xml, model)
    result = replay_mujoco(source, source.ctrl.shape[0], DEFAULT_REPLAY_SIM_DT)
    if not all(np.isfinite(x).all() for x in (result.qpos, result.qvel, result.ctrl)):
        raise RuntimeError("MuJoCo replay produced nonfinite states or controls")
    output = (Path(out_dir).resolve() if out_dir else rollout.parent / "replay") / MUJOCO_CPU
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "replay.npz", qpos=result.qpos, qvel=result.qvel,
             ctrl=result.ctrl, time=result.time)
    errors = replay_errors(source, result.qpos, object_qadr(model))
    report = {
        "input_rollout": str(rollout), "scene_xml": str(scene_xml),
        "simulator": MUJOCO_CPU, "sim_dt": DEFAULT_REPLAY_SIM_DT,
        "action_dt": source.action_dt, "steps": int(result.ctrl.shape[0]),
        "reference_available": source.object_reference_qpos is not None,
        "summary": {name: stats(values) for name, values in errors.items()},
    }
    (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    sequence_dir = Path(sequence_dir) if sequence_dir else scene_xml.parents[2]
    hand_name = hand_name or scene_xml.parent.name
    camera, fovy = sequence_camera_pose_and_fovy(
        sequence_dir, camera_pose_path=camera_pose_path or contactaware_camera_pose_path(
            sequence_dir, scene_xml, hand_name=hand_name,
        ),
    )
    render_scene_trajectory_video(TrajectoryRenderRequest(
        scene_xml=scene_xml, out_path=output / "replay.mp4", fps=30,
        camera_pose=camera, camera_fovy=fovy,
        qpos=result.qpos, qpos_frames=result.time * 30, height=480, width=640,
    ))
    print(f"[replay_saved] {output}", flush=True)
    return output

@dataclass(frozen=True)
class Rollout:
    path: Path
    method: str
    scene_xml: Path
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    action_dt: float
    frames: np.ndarray
    original_pos_mm: np.ndarray
    original_rot_deg: np.ndarray
    object_reference_qpos: np.ndarray | None = None

@dataclass(frozen=True)
class Replay:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    time: np.ndarray

def load_rollout(path: Path, scene_xml: Path, model: mujoco.MjModel) -> Rollout:
    arrays = np.load(path, allow_pickle=False)
    try:
        qpos, qvel, ctrl = rollout_arrays(arrays)
        return Rollout(
            path=path,
            method=path.parent.name,
            scene_xml=scene_xml,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            action_dt=rollout_action_dt(arrays),
            frames=rollout_frames(arrays, qpos.shape[0]),
            original_pos_mm=original_pos_error(model, arrays, qpos),
            original_rot_deg=original_rot_error(model, arrays, qpos),
            object_reference_qpos=object_reference_qpos(arrays, model, len(qpos)),
        )
    finally:
        arrays.close()


def object_reference_qpos(arrays, model: mujoco.MjModel, count: int) -> np.ndarray | None:
    if "object_reference_qpos" in arrays.files:
        reference = np.asarray(arrays["object_reference_qpos"], np.float32)
    elif "qpos_ref" in arrays.files:
        obj = object_qadr(model)
        reference = reference_qpos(arrays, count)[:, obj:obj + 7]
    else:
        return None
    if reference.shape != (count, 7) or not np.isfinite(reference).all():
        raise ValueError("object reference must contain one finite pose per rollout state")
    if np.any(np.linalg.norm(reference[:, 3:7], axis=1) < QUAT_EPS):
        raise ValueError("object reference contains a zero quaternion")
    return reference


def replay_errors(source: Rollout, qpos: np.ndarray, obj: int) -> dict[str, np.ndarray]:
    actual = qpos[:, obj:obj + 7]
    references = {"replay_delta_": source.qpos[:, obj:obj + 7]}
    if source.object_reference_qpos is not None:
        references[""] = source.object_reference_qpos
    errors = {}
    for prefix, reference in references.items():
        errors[prefix + "position_mm"] = (
            np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1) * METERS_TO_MM
        )
        errors[prefix + "rotation_deg"] = np.asarray([
            quat_angle_deg(q[3:7], ref[3:7]) for q, ref in zip(actual, reference)
        ])
    return errors

def rollout_arrays(arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {"qpos_traj", "qvel_traj", "ctrl_traj"}.issubset(arrays.files):
        qpos, qvel, ctrl = arrays["qpos_traj"], arrays["qvel_traj"], arrays["ctrl_traj"]
    elif {"qpos", "qvel", "ctrl"}.issubset(arrays.files):
        qpos, qvel, ctrl = arrays["qpos"], arrays["qvel"], arrays["ctrl"]
    else:
        raise KeyError(f"rollout must contain qpos/qvel/ctrl arrays, got {sorted(arrays.files)}")
    return validate_rollout_arrays(qpos, qvel, ctrl)

def validate_rollout_arrays(qpos, qvel, ctrl) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, np.float32)
    qvel = np.asarray(qvel, np.float32)
    ctrl = np.asarray(ctrl, np.float32)
    if qpos.ndim != 2 or qvel.ndim != 2 or ctrl.ndim != 2:
        raise ValueError(f"rollout arrays must be rank-2: qpos={qpos.shape}, qvel={qvel.shape}, ctrl={ctrl.shape}")
    if qpos.shape[0] != ctrl.shape[0] + 1 or qvel.shape[0] != qpos.shape[0]:
        raise ValueError(f"rollout length mismatch: qpos={qpos.shape}, qvel={qvel.shape}, ctrl={ctrl.shape}")
    return qpos, qvel, ctrl

def rollout_action_dt(arrays) -> float:
    if "ctrl_dt" in arrays.files:
        return float(np.asarray(arrays["ctrl_dt"]).item())
    if "action_dt" in arrays.files:
        return float(np.asarray(arrays["action_dt"]).item())
    if "time" in arrays.files and arrays["time"].shape[0] >= 2:
        time_values = np.asarray(arrays["time"], np.float64)
        return float(time_values[1] - time_values[0])
    if "sim_dt" in arrays.files:
        return float(np.asarray(arrays["sim_dt"]).item())
    raise KeyError("rollout missing action_dt, sim_dt, or time")

def rollout_frames(arrays, count: int) -> np.ndarray:
    if "qpos_frames" in arrays.files:
        return np.asarray(arrays["qpos_frames"], np.float32)[:count]
    if "time" in arrays.files:
        return np.asarray(arrays["time"], np.float32)[:count]
    return np.arange(count, dtype=np.float32)

def original_pos_error(model: mujoco.MjModel, arrays, qpos: np.ndarray) -> np.ndarray:
    if "drift" in arrays.files:
        return prepend_zero(np.asarray(arrays["drift"], np.float64) * METERS_TO_MM, qpos.shape[0])
    ref = reference_qpos(arrays, qpos.shape[0])
    obj = object_qadr(model)
    return np.linalg.norm(qpos[:, obj:obj + 3] - ref[:, obj:obj + 3], axis=1) * METERS_TO_MM

def original_rot_error(model: mujoco.MjModel, arrays, qpos: np.ndarray) -> np.ndarray:
    if "rot_err_deg" in arrays.files:
        return prepend_zero(np.asarray(arrays["rot_err_deg"], np.float64), qpos.shape[0])
    ref = reference_qpos(arrays, qpos.shape[0])
    obj = object_qadr(model)
    return np.asarray([quat_angle_deg(qpos[i, obj + 3:obj + 7], ref[i, obj + 3:obj + 7]) for i in range(qpos.shape[0])])

def prepend_zero(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[0] < count - 1:
        raise ValueError(f"metric length {values.shape[0]} is shorter than rollout length {count}")
    return np.concatenate([[0.0], values[:count - 1]])

def reference_qpos(arrays, count: int) -> np.ndarray:
    if "qpos_ref" not in arrays.files:
        raise KeyError("rollout missing qpos_ref or forceaware drift/rot_err_deg")
    ref = np.asarray(arrays["qpos_ref"], np.float32)
    if ref.shape[0] < count:
        raise ValueError(f"qpos_ref has {ref.shape[0]} rows, expected at least {count}")
    return ref[:count]

def replay_mujoco(rollout: Rollout, steps: int, sim_dt: float) -> Replay:
    model = mujoco.MjModel.from_xml_path(str(rollout.scene_xml))
    model.opt.timestep = float(sim_dt)
    substeps = replay_substeps(rollout.action_dt, sim_dt)
    validate_model(model, rollout)
    data = seed_mujoco_data(model, rollout)
    qpos_log, qvel_log, ctrl_log = [data.qpos.copy()], [data.qvel.copy()], []
    start = time.perf_counter()
    for action in rollout.ctrl[:steps]:
        ctrl = action_to_ctrl(model, action)
        for _ in range(substeps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        ctrl_log.append(ctrl.copy())
    print_replay_time(rollout, MUJOCO_CPU, steps, sim_dt, start)
    return replay_from_logs(qpos_log, qvel_log, ctrl_log, rollout.action_dt)

def replay_from_logs(qpos_log: list, qvel_log: list, ctrl_log: list, action_dt: float) -> Replay:
    steps = len(ctrl_log)
    return Replay(
        np.asarray(qpos_log, np.float32),
        np.asarray(qvel_log, np.float32),
        np.asarray(ctrl_log, np.float32),
        np.arange(steps + 1, dtype=np.float32) * action_dt,
    )

def print_replay_time(rollout: Rollout, simulator: str, steps: int, sim_dt: float, start: float) -> None:
    elapsed = time.perf_counter() - start
    print(f"[replay] {rollout.method} simulator={simulator} steps={steps} dt={sim_dt:g} time={elapsed:.2f}s")

def replay_substeps(action_dt: float, sim_dt: float) -> int:
    substeps = int(round(action_dt / sim_dt))
    if substeps < 1 or not np.isclose(substeps * sim_dt, action_dt, rtol=0.0, atol=1e-7):
        raise ValueError(f"action_dt={action_dt:g} must be an integer multiple of replay sim_dt={sim_dt:g}")
    return substeps

def validate_model(model: mujoco.MjModel, rollout: Rollout) -> None:
    if rollout.qpos.shape[1] != model.nq or rollout.qvel.shape[1] != model.nv:
        raise ValueError(f"rollout shape does not match model nq/nv: {rollout.scene_xml}")

def seed_mujoco_data(model: mujoco.MjModel, rollout: Rollout) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = rollout.qpos[0]
    data.qvel[:] = rollout.qvel[0]
    data.ctrl[:] = action_to_ctrl(model, rollout.ctrl[0])
    mujoco.mj_forward(model, data)
    return data

def action_to_ctrl(model: mujoco.MjModel, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float64)
    mapping = hand_action_mapping(model)
    if action.shape[0] == mapping.shape[0]:
        ctrl = np.zeros(model.nu, dtype=np.float64)
        ctrl[mapping] = action
        return ctrl
    if action.shape[0] == model.nu:
        return action.copy()
    raise ValueError(f"control width {action.shape[0]} does not match model.nu={model.nu}")

def hand_action_mapping(model: mujoco.MjModel) -> np.ndarray:
    obj = object_qadr(model)
    joints = sorted(
        (jid for jid in range(model.njnt) if int(model.jnt_qposadr[jid]) < obj),
        key=lambda jid: int(model.jnt_qposadr[jid]),
    )
    actuators = actuator_by_joint(model)
    return np.asarray([actuators[joint_name(model, jid)] for jid in joints], np.int32)

def actuator_by_joint(model: mujoco.MjModel) -> dict[str, int]:
    return {joint_name(model, int(model.actuator_trnid[i, 0])): i for i in range(model.nu)}

def joint_name(model: mujoco.MjModel, jid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if name is None:
        raise ValueError(f"joint {jid} has no name")
    return name

def object_qadr(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    if jid < 0:
        raise ValueError("model missing obj_joint")
    return int(model.jnt_qposadr[jid])

def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    return {"mean": float(values.mean()), "max": float(values.max()), "terminal": float(values[-1])}

def quat_angle_deg(q_wxyz: np.ndarray, ref_wxyz: np.ndarray) -> float:
    q = np.asarray(q_wxyz, np.float64)
    ref = np.asarray(ref_wxyz, np.float64)
    q /= np.linalg.norm(q) + QUAT_EPS
    ref /= np.linalg.norm(ref) + QUAT_EPS
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q, ref)), 0.0, 1.0)) * 180.0 / np.pi)
