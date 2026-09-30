"""Recording and output for Warp ForceAware."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comfree_warp.native_adjoint.kinematics_vjp import KINEMATICS_VJP_PROFILE

from forceaware.config import CONFIG_PATH
from forceaware.config import RetargetConfig
from forceaware.config import make_solve_request
from forceaware.contact_schedule import contact_age_ramp
from forceaware.metrics import control_metrics
from forceaware.metrics import dynamic_contact_stats
from forceaware.metric_contract import hand_state_stats
from forceaware.metrics import intended_contact_metrics
from forceaware.metrics import marked_query_metrics
from forceaware.metrics import metric_stats
from forceaware.metrics import save_video
from forceaware.metrics import write_metrics_json
from forceaware.validation import require_finite_tree
from forceaware.contact_mapping import map_guidance_to_gs_outputs
from tools.render import trajectory_time_frame_coordinates
from forceaware.replay import run_default_mujoco_replay
from forceaware.speed_metrics import optimization_fps

from .mpc_loss import GRASP_MIN_PHYSICAL_CONTACTS
from .mpc_loss import GRASP_QUALITY_PROFILE
from .mpc_loss import GRASP_WRENCH_MARGIN
from .mpc_loss import _grasp_output_properties
from .mpc_loss import _object_grasp_geometry
from .mpc_loss import grasp_wrench_volume_numpy
from .mpc_loss import grasp_wrenches_numpy
from .mpc_loss import objective_metadata
from .mpc_loss import sampled_grasp_supports_numpy
from .mpc_loss import stable_grasp_guidance
from comfree_warp.collision_config import CONTACT_FRAME_VJP_PROFILE
from .optimizer import Plan
from .config import config_dict
from .targets import ModelIndex
from .targets import SequenceData
from .targets import WindowTargets
from .targets import hand_qvel_indices
from .targets import _object_qpos


QUATERNION_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CollisionBatch:
    distance: np.ndarray
    position: np.ndarray
    frame: np.ndarray
    active: np.ndarray
    body_position: np.ndarray
    body_matrix: np.ndarray
    threshold: np.ndarray


class RolloutRecorder:
    def __init__(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        start_frame: float,
        warm: np.ndarray,
    ):
        self.qpos = [np.asarray(qpos, np.float32)]
        self.qvel = [np.asarray(qvel, np.float32)]
        self.qpos_frames = [float(start_frame)]
        self.raw_warm = [np.asarray(warm, np.float32)]
        self.ctrl: list[np.ndarray] = []
        self.target_frames: list[float] = []
        self.drift: list[float] = []
        self.rotation: list[float] = []
        self.step_seconds: list[float] = []
        self.step_sizes: list[int] = []
        self.scores: list[np.ndarray] = []
        self.candidate_drifts: list[np.ndarray] = []
        self.has_finite_best: list[np.ndarray] = []
        self.candidate_valid: list[np.ndarray] = []
        self.training_failure_iteration: list[np.ndarray] = []
        self.training_failure_code: list[np.ndarray] = []
        self.rejection_code: list[np.ndarray] = []
        self.selected: list[int] = []

    def replace_initial(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.qpos[0] = np.asarray(qpos, np.float32)
        self.qvel[0] = np.asarray(qvel, np.float32)

    def record(
        self,
        plan: Plan,
        targets: WindowTargets,
        *,
        qpos_sequence: np.ndarray,
        qvel_sequence: np.ndarray,
        index: ModelIndex,
        warm: np.ndarray,
        display_frames: np.ndarray | None = None,
    ) -> None:
        self.step_seconds.append(plan.seconds)
        count = qpos_sequence.shape[0]
        self.step_sizes.append(count)
        frames = targets.frames[:count] if display_frames is None else display_frames
        for step in range(count):
            target = targets.object_qpos[step]
            qpos = np.asarray(qpos_sequence[step], np.float32)
            actual = qpos[index.object_qpos : index.object_qpos + 7]
            self.qpos.append(qpos)
            self.qvel.append(np.asarray(qvel_sequence[step], np.float32))
            self.qpos_frames.append(float(frames[step]))
            self.raw_warm.append(np.asarray(warm, np.float32))
            self.ctrl.append(np.asarray(plan.control_sequence[step], np.float32))
            self.target_frames.append(float(targets.frames[step]))
            self.drift.append(float(np.linalg.norm(actual[:3] - target[:3])))
            self.rotation.append(quaternion_angle_deg(actual[3:7], target[3:7]))
            self.scores.append(np.asarray(plan.diag["scores"], np.float32))
            self.candidate_drifts.append(np.asarray(plan.diag["drifts"], np.float32))
            self.has_finite_best.append(np.asarray(plan.diag["has_finite_best"], bool))
            self.candidate_valid.append(np.asarray(plan.diag["valid"], bool))
            self.training_failure_iteration.append(
                np.asarray(plan.diag["training_failure_iteration"], np.int32)
            )
            self.training_failure_code.append(
                np.asarray(plan.diag["training_failure_code"], np.int32)
            )
            self.rejection_code.append(
                np.asarray(plan.diag["rejection_code"], np.int32)
            )
            self.selected.append(int(plan.diag["selected"]))

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "qpos_traj": np.stack(self.qpos),
            "qvel_traj": np.stack(self.qvel),
            "qpos_frames": np.asarray(self.qpos_frames, np.float32),
            "ctrl_traj": np.stack(self.ctrl),
            "raw_params": np.asarray(self.raw_warm[-1], np.float32),
            "raw_warm_history": np.stack(self.raw_warm),
            "multistart_scores": np.stack(self.scores),
            "multistart_drifts": np.stack(self.candidate_drifts),
            "multistart_has_finite_best": np.stack(self.has_finite_best),
            "multistart_valid": np.stack(self.candidate_valid),
            "multistart_training_failure_iteration": np.stack(
                self.training_failure_iteration
            ),
            "multistart_training_failure_code": np.stack(self.training_failure_code),
            "multistart_rejection_code": np.stack(self.rejection_code),
            "multistart_selected": np.asarray(self.selected, np.int32),
            "multistart_step_sec": np.repeat(
                np.asarray(self.step_seconds, np.float32),
                np.asarray(self.step_sizes, np.int32),
            ),
            "mpc_step_sec": np.asarray(self.step_seconds, np.float32),
            "mpc_execution_steps": np.asarray(self.step_sizes, np.int32),
            "drift": np.asarray(self.drift, np.float32),
            "rot_err_deg": np.asarray(self.rotation, np.float32),
            "target_frames": np.asarray(self.target_frames, np.float32),
        }
        return arrays


def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = left / (float(np.linalg.norm(left)) + QUATERNION_EPSILON)
    right = right / (float(np.linalg.norm(right)) + QUATERNION_EPSILON)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot) * 180.0 / np.pi)


def _validate_recorded_arrays(arrays: dict[str, np.ndarray]) -> None:
    scores = arrays["multistart_scores"]
    valid = arrays["multistart_valid"]
    rejected = arrays["multistart_rejection_code"] != 0
    # Rejected candidates may have +inf; selected states must be finite.
    if not np.array_equal(np.isposinf(scores), ~valid):
        raise FloatingPointError(
            "Multistart +inf scores must match rejected candidates"
        )
    if not np.array_equal(rejected, ~valid):
        raise ValueError("Multistart rejection codes disagree with candidate validity")
    if not valid[np.arange(len(valid)), arrays["multistart_selected"]].all():
        raise ValueError("Recorded rollout selected a rejected multistart candidate")
    require_finite_tree("warp rollout", {**arrays, "multistart_scores": scores[valid]})


def save_outputs(
    cfg: RetargetConfig,
    recorder: RolloutRecorder,
    *,
    compiled,
    cpu_model,
    index: ModelIndex,
    sequence: SequenceData,
    source_config_path: Path = CONFIG_PATH,
) -> Path:
    arrays = recorder.arrays()
    arrays["object_reference_qpos"] = _object_qpos(
        sequence.object_pose_xyzw,
        np.r_[arrays["qpos_frames"][0], arrays["target_frames"]],
    )
    _validate_recorded_arrays(arrays)
    out_dir = cfg.sequence.seq_dir / cfg.sequence.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(config_dict(cfg), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rollout = out_dir / "rollout.npz"
    np.savez(rollout, **arrays, **_metadata(cfg, compiled.collision, source_config_path))
    metrics = build_metrics(
        cfg,
        arrays,
        compiled=compiled,
        cpu_model=cpu_model,
        index=index,
        sequence=sequence,
    )
    metrics.update(_speed_metrics(cfg, arrays))
    metrics_path = out_dir / "metrics.json"
    write_metrics_json(metrics_path, metrics)
    video = out_dir / "rollout.mp4"
    grid = cfg.simulator.time_grid
    video_frames = trajectory_time_frame_coordinates(
        arrays["qpos_traj"].shape[0], grid.mpc_dt, cfg.video.fps
    )
    save_video(
        qpos_traj=arrays["qpos_traj"],
        qpos_frames=video_frames,
        out_path=video,
        seq_dir=cfg.sequence.seq_dir,
        xml_path=cfg.sequence.xml,
        fps=cfg.video.fps,
        hand_name=cfg.sequence.hand,
        camera_pose_path=cfg.sequence.contact_guidance_npz.parent / "camera_pose_7.npy",
    )
    _require_files((config_path, rollout, metrics_path, video))
    replay_dir = run_default_mujoco_replay(
        rollout, cfg.sequence.xml,
        sequence_dir=cfg.sequence.seq_dir, hand_name=cfg.sequence.hand,
        camera_pose_path=cfg.sequence.contact_guidance_npz.parent / "camera_pose_7.npy",
    )
    _require_files((replay_dir / "replay.mp4", replay_dir / "replay.json"))
    return out_dir


def build_metrics(
    cfg: RetargetConfig,
    arrays: dict,
    *,
    compiled,
    cpu_model,
    index: ModelIndex,
    sequence: SequenceData,
) -> dict:
    qpos = np.asarray(arrays["qpos_traj"][1:], np.float32)
    collision = _batch_collision_observation(compiled, qpos)
    phi = collision.distance
    contact_pos = collision.position
    xpos = collision.body_position
    xmat = collision.body_matrix
    threshold = collision.threshold
    guidance = _load_metric_guidance(
        cfg.sequence.contact_guidance_npz,
        compiled.collision,
        cfg.contact.age_ramp_frames,
    )
    guidance["contact_pos_obj"] = sequence.contact.position_object
    frames = np.asarray(arrays["target_frames"], np.float32)
    return {
        "frames": {
            "start": float(frames[0]),
            "end": float(frames[-1]),
            "count": int(frames.shape[0]),
        },
        "pose": {
            "position_mm": metric_stats(arrays["drift"] * 1000.0),
            "rotation_deg": metric_stats(arrays["rot_err_deg"]),
        },
        "penetration": {
            "hand_object": dynamic_contact_stats(phi, threshold),
        },
        "marked_query": marked_query_metrics(
            xpos, xmat, frames, cg=guidance, idx={"obj_bodyid": index.object_body}
        ),
        "intended_contact": intended_contact_metrics(
            phi,
            contact_pos,
            (xpos, xmat),
            target_frames=frames,
            cg=guidance,
            idx={"obj_bodyid": index.object_body},
            contact_threshold=threshold,
        ),
        "grasp_quality": grasp_quality_metrics(
            cfg,
            collision,
            cpu_model=cpu_model,
            index=index,
            target_frames=frames,
            guidance=guidance,
            collision_model=compiled.collision,
        ),
        "control": control_metrics(
            arrays["qpos_traj"][0],
            arrays["ctrl_traj"],
            _metric_index(index),
            mpc_dt=cfg.simulator.time_grid.mpc_dt,
            action_substeps=cfg.simulator.time_grid.action_substeps,
            knot_substeps=cfg.simulator.time_grid.knot_substeps,
        ),
        "hand_state": hand_state_stats(
            arrays["qvel_traj"],
            hand_qvel_indices(cpu_model, index),
            dt=cfg.simulator.time_grid.mpc_dt,
        ),
    }


def _batch_collision_observation(compiled, qpos: np.ndarray) -> CollisionBatch:
    import warp as wp
    from comfree_warp.native_adjoint.observation_adjoint import allocate_workspace
    from comfree_warp.native_adjoint.observation_adjoint import forward

    worlds = qpos.shape[0]
    workspace = allocate_workspace(compiled, worlds, requires_grad=False)
    device_qpos = wp.array(qpos, dtype=float, device=compiled.device)
    result = forward(compiled, device_qpos, workspace)
    wp.synchronize()
    collision = compiled.collision
    rows = collision.output_contact_rows.numpy().astype(np.int32)
    valid = rows >= 0
    thresholds = collision.output_thresholds.numpy().astype(np.float32)
    phi = np.broadcast_to(thresholds, (worlds, rows.size)).copy()
    contact_position = np.zeros((worlds, rows.size, 3), np.float32)
    contact_frame = np.zeros((worlds, rows.size, 3, 3), np.float32)
    output_active = np.zeros((worlds, rows.size), bool)
    if valid.any():
        distance = result.contact_distance.numpy()[:, rows[valid]]
        active = workspace.collision.contacts.active.numpy()[:, rows[valid]] > 0
        phi[:, valid] = np.where(active, distance, thresholds[valid])
        contact_position[:, valid] = result.contact_position.numpy()[:, rows[valid]]
        contact_frame[:, valid] = result.contact_frame.numpy()[:, rows[valid]]
        output_active[:, valid] = active
    return CollisionBatch(
        distance=phi,
        position=contact_position,
        frame=contact_frame,
        active=output_active,
        body_position=result.body_position.numpy(),
        body_matrix=result.body_matrix.numpy(),
        threshold=thresholds,
    )


def grasp_quality_metrics(
    cfg: RetargetConfig,
    batch: CollisionBatch,
    *,
    cpu_model,
    index: ModelIndex,
    target_frames: np.ndarray,
    guidance: dict[str, np.ndarray],
    collision_model,
) -> dict:
    signs, margins, friction = _grasp_output_properties(
        cpu_model, index, collision_model
    )
    object_com, object_radius = _object_grasp_geometry(
        cpu_model, index, collision_model
    )
    frame_ids = np.clip(
        np.rint(target_frames).astype(np.int32),
        0,
        guidance["contact_mask"].shape[0] - 1,
    )
    stable_rows = []
    for row, frame in enumerate(frame_ids):
        weight = guidance["contact_weight"][frame] * guidance["contact_age_ramp"][frame]
        if stable_grasp_guidance(
            guidance["contact_mask"][frame],
            weight,
            guidance["hand_query_body_ids"][frame],
        ):
            stable_rows.append(row)
    if not stable_rows:
        return {
            "profile": GRASP_QUALITY_PROFILE,
            "stable_frame_count": 0,
        }
    support_values = []
    wrench_volume_values = []
    contact_counts = []
    for row in stable_rows:
        physical = batch.active[row] & (batch.distance[row] < margins) & (signs != 0.0)
        wrenches = grasp_wrenches_numpy(
            batch.position[row],
            batch.frame[row],
            physical,
            signs,
            friction,
            batch.body_position[row, index.object_body],
            batch.body_matrix[row, index.object_body],
            object_com,
            object_radius,
        )
        supports = sampled_grasp_supports_numpy(
            batch.position[row],
            batch.frame[row],
            physical,
            signs,
            friction,
            batch.body_position[row, index.object_body],
            batch.body_matrix[row, index.object_body],
            object_com,
            object_radius,
        )
        wrench_volume = grasp_wrench_volume_numpy(wrenches)
        count = int(np.count_nonzero(physical))
        support_values.append(0.0 if supports.size == 0 else float(np.min(supports)))
        wrench_volume_values.append(wrench_volume)
        contact_counts.append(count)
    support = np.asarray(support_values)
    wrench_volume = np.asarray(wrench_volume_values)
    contacts = np.asarray(contact_counts)
    passed = (support >= GRASP_WRENCH_MARGIN) & (
        contacts >= GRASP_MIN_PHYSICAL_CONTACTS
    )
    return {
        "profile": GRASP_QUALITY_PROFILE,
        "stable_frame_count": len(stable_rows),
        "target_support": GRASP_WRENCH_MARGIN,
        "minimum_support": metric_stats(support),
        "wrench_volume": metric_stats(wrench_volume),
        "physical_contact_count": metric_stats(contacts),
        "pass_rate": float(np.mean(passed)),
    }


def _load_metric_guidance(
    path: Path,
    collision,
    ramp_frames: int,
) -> dict[str, np.ndarray]:
    names = (
        "contact_mask",
        "contact_weight",
        "contact_pos_obj",
        "hand_query_body_ids",
        "hand_query_local_pos",
    )
    with np.load(path, allow_pickle=False) as source:
        guidance = {name: np.asarray(source[name]) for name in names}
    guidance["gs_output_indices"] = map_guidance_to_gs_outputs(
        collision.output_body_ids.numpy().astype(np.int32),
        guidance["hand_query_body_ids"],
        guidance["contact_mask"],
    )
    guidance["contact_age_ramp"] = contact_age_ramp(
        guidance["contact_mask"], ramp_frames
    )
    return guidance


def _metric_index(index: ModelIndex) -> dict[str, np.ndarray | int]:
    return {
        "base_qadr": index.hand_qpos[:6],
        "finger_qadr": index.hand_qpos[6:],
    }


def _metadata(
    cfg: RetargetConfig,
    collision,
    source_config_path: Path = CONFIG_PATH,
) -> dict[str, np.ndarray]:
    values = {}
    for section in (
        _time_metadata(cfg),
        _optimizer_metadata(cfg, collision),
        _objective_metadata(cfg),
    ):
        values.update(section)
    values.update(
        {
            "backend": "warp",
            "config_path": str(source_config_path),
            "config_json": json.dumps(config_dict(cfg), default=str, sort_keys=True),
        }
    )
    return {name: np.asarray(value) for name, value in values.items()}


def _time_metadata(cfg: RetargetConfig) -> dict:
    sim, sequence = cfg.simulator, cfg.sequence
    grid = sim.time_grid
    request = make_solve_request(cfg)
    start_frame = sequence.hold_frame if sequence.hold_frame >= 0 else sequence.start
    return {
        "start": sequence.start,
        "start_frame": start_frame,
        "end": sequence.end,
        "horizon": cfg.optimizer.horizon,
        "sim_dt": grid.mpc_dt,
        "action_dt": grid.action_dt,
        "ctrl_dt": grid.mpc_dt,
        "knot_dt": grid.knot_dt,
        "optimizer_action_dt": grid.knot_dt,
        "mpc_dt": grid.mpc_dt,
        "exec_dt": grid.exec_dt,
        "action_substeps": grid.action_substeps,
        "knot_substeps": grid.knot_substeps,
        "exec_substeps": grid.executor_substeps_per_action,
        "action_exec_substeps": grid.executor_substeps_per_action,
        "dense_exec_substeps": grid.executor_substeps_per_mpc_step,
        "ref_dt": grid.ref_dt,
        "frame_step": grid.mpc_dt / grid.ref_dt,
        "action_frame_step": grid.action_dt / grid.ref_dt,
        "knot_frame_step": grid.knot_dt / grid.ref_dt,
        "video_fps": cfg.video.fps,
        "hold_frame": sequence.hold_frame,
        "hold_seconds": sequence.hold_seconds,
        "requested_duration_seconds": request.duration_seconds,
        "executed_duration_seconds": request.executed_duration_seconds,
        "tail_hold_seconds": request.tail_hold_seconds,
        "total_physics_steps": request.total_physics_steps,
        "mpc_window_count": len(request.execution_steps_per_window),
        "execution_steps_per_window": request.execution_steps_per_window,
        "exec_backend": sim.exec_backend,
    }


def _optimizer_metadata(cfg: RetargetConfig, collision) -> dict:
    from comfree_warp.collision_config import stop_normal_gradient

    optimizer = cfg.optimizer
    return {
        "contact_dynamics": optimizer.contact_dynamics,
        "candidate_score_source": "training_loss",
        "n_iter": optimizer.n_iter,
        "lr_max": optimizer.lr_max,
        "lr_min": optimizer.lr_min,
        "adam_beta1": optimizer.adam_beta1,
        "adam_beta2": optimizer.adam_beta2,
        "adam_epsilon": optimizer.adam_epsilon,
        "grad_clip": optimizer.grad_clip,
        "multistart_count": optimizer.multistart_count,
        "multistart_seed": optimizer.multistart_seed,
        "multistart_init_std": optimizer.multistart_init_std,
        "multistart_ctrl_std": optimizer.multistart_ctrl_std,
        "multistart_init_raw_limit": optimizer.multistart_init_raw_limit,
        "warm_shift_scale": optimizer.warm_shift_scale,
        "contact_frame_vjp_profile": CONTACT_FRAME_VJP_PROFILE,
        "contact_frame_vjp_exact_derivative": not any(
            stop_normal_gradient(batch.target.target.kind, batch.model.contact_topk)
            for batch in collision.batches
        ),
        "contact_frame_vjp_forward_unchanged": True,
        "kinematics_vjp_profile": KINEMATICS_VJP_PROFILE,
        "kinematics_vjp_exact_derivative": True,
        **_control_parameterization_metadata(cfg),
    }


def _control_parameterization_metadata(cfg: RetargetConfig) -> dict:
    request = make_solve_request(cfg)
    optimizer = cfg.optimizer
    dense_steps = optimizer.horizon * cfg.simulator.time_grid.knot_substeps
    windows = len(request.execution_steps_per_window)
    training_steps = windows * dense_steps * optimizer.n_iter
    selection_steps = windows * dense_steps
    return {
        "control_parameterization_profile": (
            "linear_knots_fractional_action_replan_v1"
        ),
        "control_smoothness_boundary_profile": "sent_command_history_v1",
        "warm_start_profile": "fractional_additive_knot_resample_v1",
        "dense_steps_per_horizon": dense_steps,
        "nominal_training_physics_steps_per_candidate": training_steps,
        "nominal_selection_physics_steps_per_candidate": selection_steps,
        "nominal_total_physics_steps_per_candidate": (training_steps + selection_steps),
    }


def _objective_metadata(cfg: RetargetConfig) -> dict:
    return {
        "gs_distance_offset": cfg.simulator.gs_distance_offset,
        "contact_anchor_inset": cfg.simulator.gs_distance_offset,
        "gs_mode": "hard" if cfg.simulator.contact_topk == 1 else "soft",
        **objective_metadata(cfg),
        "control_correction_bounded": False,
        "control_rate_bounded": False,
        "contact_guidance_npz": str(cfg.sequence.contact_guidance_npz),
    }


def _speed_metrics(cfg: RetargetConfig, arrays: dict) -> dict[str, float]:
    end_frame = float(arrays["target_frames"][-1])
    start_frame = (
        float(cfg.sequence.hold_frame)
        if cfg.sequence.hold_frame >= 0
        else float(cfg.sequence.start)
    )
    covered_frames = end_frame - start_frame
    if covered_frames < 0.0:
        raise ValueError(
            f"Last target frame {end_frame:g} precedes start frame {start_frame:g}"
        )
    frame_count = max(1, int(np.ceil(covered_frames)))
    seconds = float(np.asarray(arrays["mpc_step_sec"], np.float64).sum())
    return {
        "fps": optimization_fps(frame_count=frame_count, optimization_time_sec=seconds),
        "reference_fps": float(1.0 / cfg.simulator.time_grid.ref_dt),
    }


def _require_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected output file was not created: {path}")
