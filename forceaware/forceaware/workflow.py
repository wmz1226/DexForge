"""Pure-Warp rolling ForceAware workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from forceaware.config import RetargetConfig
from forceaware.config import CONFIG_PATH
from forceaware.config import make_solve_request
from forceaware.control import resample_warm_corrections
from forceaware.time_grid import SolveRequest

from .config import config_dict
from .executor import ActionExecutor
from .executor import ExecutorRuntime
from .executor import build_action_executor
from .servo_prior import ServoPriorOptimizer as WindowOptimizer
from .output import RolloutRecorder
from .output import save_outputs
from .targets import ModelIndex
from .targets import SequenceData
from .targets import build_index
from .targets import initial_state
from .targets import load_sequence
from .targets import make_window
from .wrist_reference import continuous_wrist_reference


@dataclass(frozen=True)
class InitialRun:
    qpos: np.ndarray
    qvel: np.ndarray
    warm: np.ndarray
    previous_action: np.ndarray
    start_frame: float
    request: SolveRequest


@dataclass(frozen=True)
class RunContext:
    cfg: RetargetConfig
    model: mujoco.MjModel
    index: ModelIndex
    sequence: SequenceData
    optimizer: WindowOptimizer
    executor: ActionExecutor
    initial: InitialRun
    recorder: RolloutRecorder


@dataclass(frozen=True)
class RuntimeModels:
    planner_device: object
    planner_cpu: mujoco.MjModel
    executor: ExecutorRuntime


def rolling_mpc_retarget(
    cfg: RetargetConfig,
    *,
    config_path: Path = CONFIG_PATH,
) -> Path:
    import comfree_warp

    print(f"[config_file] {config_path}")
    print(f"[config] {json.dumps(config_dict(cfg), default=str, sort_keys=True)}")
    models = _load_runtime_models(cfg, comfree_warp)
    cpu_model, device_model = models.planner_cpu, models.planner_device
    index = build_index(cpu_model)
    executor_index = _executor_index(models, index)
    output_body_ids, enabled_output_indices = _collision_output_layout(device_model)
    object_surface_spheres = None
    if cfg.simulator.gs_distance_offset != 0.0:
        object_surface_spheres = _object_surface_spheres(
            device_model, index.object_body
        )
    sequence = load_sequence(
        cfg,
        output_body_ids=output_body_ids,
        enabled_output_indices=enabled_output_indices,
        object_surface_spheres=object_surface_spheres,
    )
    _validate_problem(index, sequence)
    continuous_hand = continuous_wrist_reference(cpu_model, index.hand_qpos, sequence.hand)
    corrected_frames = int(np.count_nonzero(np.any(continuous_hand != sequence.hand, axis=1)))
    sequence = replace(sequence, hand=continuous_hand)
    print(f"[wrist_reference] equivalent_branch_corrections={corrected_frames}")
    print(
        f"[hand_reference] {sequence.hand_source} "
        f"samples={sequence.hand.shape} index_per_frame="
        f"{sequence.hand_index_per_frame:g}"
    )
    initial = _initial_run(cfg, cpu_model, index, sequence=sequence)
    executor = build_action_executor(models.executor, executor_index, cfg)
    optimizer = WindowOptimizer(
        cpu_model,
        device_model,
        index,
        guidance_count=sequence.contact.mask.shape[1],
        cfg=cfg,
    )
    recorder = RolloutRecorder(
        initial.qpos, initial.qvel, start_frame=initial.start_frame, warm=initial.warm
    )
    _run_steps(
        RunContext(
            cfg, cpu_model, index, sequence, optimizer, executor, initial, recorder
        )
    )
    output = save_outputs(
        cfg,
        recorder,
        compiled=optimizer.compiled.base,
        cpu_model=cpu_model,
        index=index,
        sequence=sequence,
        source_config_path=config_path,
    )
    print(f"[saved] {output}", flush=True)
    return output


def _load_runtime_models(cfg: RetargetConfig, comfree_warp) -> RuntimeModels:
    planner_device, planner_cpu = comfree_warp.load_model(cfg.sequence.xml)
    _configure_comfree_physics(
        cfg, planner_cpu, planner_device, timestep=cfg.simulator.time_grid.mpc_dt
    )
    executor = _load_executor_runtime(
        cfg,
        comfree_warp,
        planner_device=planner_device,
        planner_cpu=planner_cpu,
    )
    return RuntimeModels(planner_device, planner_cpu, executor)


def _load_executor_runtime(
    cfg: RetargetConfig,
    comfree_warp,
    *,
    planner_device,
    planner_cpu: mujoco.MjModel,
) -> ExecutorRuntime:
    sim = cfg.simulator
    grid = sim.time_grid
    if sim.exec_backend != "comfree":
        raise ValueError(f"Unsupported Warp executor backend: {sim.exec_backend}")
    if grid.exec_dt == grid.mpc_dt:
        return ExecutorRuntime(planner_cpu, planner_device)
    device_model, cpu_model = comfree_warp.load_model(cfg.sequence.xml)
    _configure_comfree_physics(cfg, cpu_model, device_model, timestep=grid.exec_dt)
    return ExecutorRuntime(cpu_model, device_model)


def _configure_comfree_physics(
    cfg: RetargetConfig, cpu_model, device_model, *, timestep: float
) -> None:
    cpu_model.opt.timestep = timestep
    device_model.opt.timestep.fill_(timestep)
    from comfree_warp.collision_config import CollisionConfig, configure_collision
    configure_collision(device_model, CollisionConfig(
        cfg.simulator.contact_topk, cfg.simulator.gs_distance_offset))


def _executor_index(models: RuntimeModels, planner_index: ModelIndex) -> ModelIndex:
    planner = models.planner_cpu
    executor = models.executor.cpu_model
    if (planner.nq, planner.nv, planner.nu) != (executor.nq, executor.nv, executor.nu):
        raise ValueError("Warp planner and executor model shapes do not match")
    executor_index = build_index(executor)
    for name in ("hand_qpos", "hand_ctrl"):
        if not np.array_equal(
            getattr(planner_index, name), getattr(executor_index, name)
        ):
            raise ValueError(f"Warp planner and executor index mismatch for {name}")
    for name in ("object_qpos", "object_body"):
        if getattr(planner_index, name) != getattr(executor_index, name):
            raise ValueError(f"Warp planner and executor index mismatch for {name}")
    return executor_index


def _collision_output_layout(device_model) -> tuple[np.ndarray, np.ndarray]:
    collision = device_model.gaussian_collision
    batches = collision if isinstance(collision, tuple) else (collision,)
    body_rows, enabled_rows, offset = [], [], 0
    for batch in batches:
        source_body_ids = batch.source_body_ids.numpy().astype(np.int32)
        output_starts = batch.output_starts.numpy().astype(np.int32)
        enabled = batch.target_output_ids.numpy().astype(np.int32)
        body_rows.append(source_body_ids[output_starts])
        enabled_rows.append(enabled + offset)
        offset += output_starts.size
    return np.concatenate(body_rows), np.concatenate(enabled_rows)


def _object_surface_spheres(device_model, object_body: int) -> np.ndarray:
    collision = device_model.gaussian_collision
    batches = collision if isinstance(collision, tuple) else (collision,)
    rows = [
        batch.target_spheres.numpy().astype(np.float32)
        for batch in batches
        if int(batch.target_body_id) == object_body
    ]
    if not rows:
        raise ValueError(
            f"Gaussian collision has no target spheres for object body {object_body}"
        )
    return np.concatenate(rows, axis=0)


def _validate_problem(index: ModelIndex, sequence: SequenceData) -> None:
    if sequence.hand.shape[1] != index.action_dim:
        raise ValueError(
            "contactaware hand trajectory does not match scene action dimension: "
            f"trajectory={sequence.hand.shape[1]}, scene={index.action_dim}"
        )


def _initial_run(
    cfg: RetargetConfig,
    model: mujoco.MjModel,
    index: ModelIndex,
    *,
    sequence: SequenceData,
) -> InitialRun:
    start_frame = _start_frame(cfg)
    qpos, qvel = initial_state(
        model, index, sequence, start_frame=start_frame,
        ref_dt=cfg.simulator.time_grid.ref_dt, hold=cfg.sequence.hold_frame >= 0,
    )
    raw_dim = index.action_dim * (1 + cfg.optimizer.horizon)
    warm = np.zeros(raw_dim, np.float32)
    previous = qpos[index.hand_qpos].copy()
    return InitialRun(qpos, qvel, warm, previous, start_frame, make_solve_request(cfg))


def _start_frame(cfg: RetargetConfig) -> float:
    hold = cfg.sequence.hold_frame
    if hold >= 0:
        if cfg.sequence.hold_seconds <= 0.0:
            raise ValueError("hold_frame requires a positive hold_seconds")
        return float(hold)
    return float(cfg.sequence.start)


def _run_steps(ctx: RunContext) -> None:
    cfg, initial = ctx.cfg, ctx.initial
    qpos, qvel = initial.qpos, initial.qvel
    warm, previous = initial.warm, initial.previous_action
    executed_steps = 0
    for window, execution_steps in enumerate(
        initial.request.execution_steps_per_window
    ):
        targets = _step_targets(ctx, window, executed_steps=executed_steps)
        allow_init = window == 0
        plan = ctx.optimizer.solve(
            qpos,
            qvel,
            targets,
            warm=warm,
            previous_action=previous,
            allow_init=allow_init,
            seed=cfg.optimizer.multistart_seed + window,
            execution_steps=execution_steps,
            start_time=ctx.executor.elapsed,
        )
        execution_qpos = plan.initial_qpos if allow_init else qpos
        execution = ctx.executor.execute_action_sequence(
            execution_qpos, qvel, actions=plan.control_sequence
        )
        if allow_init:
            ctx.recorder.replace_initial(plan.initial_qpos, qvel)
        warm = _shift_warm(
            plan.raw,
            ctx.index.action_dim,
            cfg,
            execution_steps=execution_steps,
        )
        ctx.recorder.record(
            plan,
            targets,
            qpos_sequence=execution.qpos_sequence,
            qvel_sequence=execution.qvel_sequence,
            index=ctx.index,
            warm=warm,
            display_frames=_display_frames(
                cfg,
                initial.start_frame,
                executed_steps,
                execution_steps=execution_steps,
            ),
        )
        qpos = execution.next_qpos
        qvel = execution.next_qvel
        previous = plan.action
        executed_steps += execution_steps
        _print_step(
            window,
            len(initial.request.execution_steps_per_window),
            plan,
            recorder=ctx.recorder,
        )
    if executed_steps != initial.request.total_physics_steps:
        raise RuntimeError(
            "executed physics-step count disagrees with the solve request: "
            f"executed={executed_steps}, "
            f"required={initial.request.total_physics_steps}"
        )


def _step_targets(ctx: RunContext, window: int, *, executed_steps: int):
    cfg = ctx.cfg
    targets = make_window(
        ctx.sequence,
        executed_steps=executed_steps,
        horizon=cfg.optimizer.horizon,
        time_grid=cfg.simulator.time_grid,
        start_frame=ctx.initial.start_frame,
        end_frame=cfg.sequence.end,
        hold_frame=cfg.sequence.hold_frame,
    )
    return targets


def _display_frames(
    cfg: RetargetConfig,
    start_frame: float,
    executed_steps: int,
    *,
    execution_steps: int,
) -> np.ndarray | None:
    if cfg.sequence.hold_frame < 0:
        return None
    offsets = executed_steps + np.arange(1, execution_steps + 1, dtype=np.float32)
    grid = cfg.simulator.time_grid
    return start_frame + offsets * grid.mpc_dt / grid.ref_dt


def _shift_warm(
    raw: np.ndarray, action_dim: int, cfg: RetargetConfig, *, execution_steps: int
) -> np.ndarray:
    return resample_warm_corrections(
        raw,
        action_dim=action_dim,
        horizon=cfg.optimizer.horizon,
        executed_dense_steps=execution_steps,
        physics_steps_per_knot=cfg.simulator.time_grid.knot_substeps,
        warm_scale=cfg.optimizer.warm_shift_scale,
    )


def _print_step(step: int, count: int, plan, *, recorder: RolloutRecorder) -> None:
    print(
        f"[warp-forceaware] {step + 1}/{count} loss={plan.loss:.6g} "
        f"pos_mm={recorder.drift[-1] * 1000.0:.3f} "
        f"rot_deg={recorder.rotation[-1]:.3f} grad={plan.grad_norm:.3f} "
        f"valid={int(np.asarray(plan.diag['valid']).sum())}/"
        f"{len(plan.diag['valid'])} sec={plan.seconds:.3f}",
        flush=True,
    )
