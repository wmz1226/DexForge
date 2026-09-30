"""Warp rolling MPC optimizer for ForceAware retargeting."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import mujoco
import numpy as np
import warp as wp
from forceaware.control import action_knot_interpolation_map
from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
from forceaware.numerics import LOSS_GRADIENT_SCALE
from forceaware.optimizer_contract import CANDIDATE_FAILURE_GRADIENT
from forceaware.optimizer_contract import CANDIDATE_FAILURE_LABELS
from forceaware.optimizer_contract import CANDIDATE_FAILURE_LOSS
from forceaware.optimizer_contract import CANDIDATE_FAILURE_NONE
from forceaware.optimizer_contract import CANDIDATE_FAILURE_SIMULATOR
from forceaware.optimizer_contract import CANDIDATE_FAILURE_UPDATE
from forceaware.optimizer_contract import CandidateEvaluation
from forceaware.optimizer_contract import NO_FAILURE_ITERATION
from forceaware.optimizer_contract import finalize_candidate_evaluation
from forceaware.optimizer_contract import finite_best_state
from forceaware.optimizer_contract import multistart_raws
from forceaware.time_grid import validate_execution_steps
from forceaware.validation import require_finite_array
from comfree_warp.native_adjoint.dynamics import DynamicsInput
from comfree_warp.native_adjoint.fast_step import clear_contact_status
from comfree_warp.native_adjoint.fast_step import compile_fast_step
from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall
from comfree_warp.native_adjoint.fast_step_adjoint import StepCotangent
from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace
from comfree_warp.native_adjoint.fast_step_adjoint import backward as step_backward
from comfree_warp.native_adjoint.fast_step_adjoint import forward as step_forward
from comfree_warp.native_adjoint.fast_step_adjoint import record
from comfree_warp.native_adjoint.observation_adjoint import ObservationCotangent
from comfree_warp.native_adjoint.observation_adjoint import (
    allocate_cotangent as allocate_observation_cotangent,
)
from comfree_warp.native_adjoint.observation_adjoint import (
    allocate_workspace as allocate_observation_workspace,
)
from comfree_warp.native_adjoint.observation_adjoint import (
    backward as observation_backward,
)
from comfree_warp.native_adjoint.observation_adjoint import (
    forward as observation_forward,
)
from comfree_warp.native_adjoint.observation_adjoint import record as record_observation
from comfree_warp.native_adjoint.observation_adjoint import (
    zero_cotangent as zero_observation_cotangent,
)
from comfree_warp.native_adjoint.runtime import StepInput
from forceaware.config import RetargetConfig
from .mpc_loss import LossState
from .mpc_loss import MpcObjective
from .targets import ModelIndex
from .targets import WindowTargets

BASE_ACTION_DIM = 6


PI = wp.constant(3.141592653589793)


BEST_LOSS_SENTINEL_VALUE = 1.0e30


BEST_LOSS_SENTINEL = wp.constant(BEST_LOSS_SENTINEL_VALUE)


@dataclass(frozen=True)
class OptimizerSettings:
    horizon: int
    knot_substeps: int
    execution_substeps: int
    worlds: int
    iterations: int
    learning_rate: float
    final_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip: float
    initial_raw_limit: float
    init_noise: float
    control_noise: float

    @classmethod
    def from_config(cls, cfg: RetargetConfig) -> "OptimizerSettings":
        opt = cfg.optimizer
        grid = cfg.simulator.time_grid
        return cls(
            opt.horizon,
            grid.knot_substeps,
            grid.action_substeps,
            opt.multistart_count,
            opt.n_iter,
            opt.lr_max,
            opt.lr_min,
            opt.adam_beta1,
            opt.adam_beta2,
            opt.adam_epsilon,
            opt.grad_clip,
            opt.multistart_init_raw_limit,
            opt.multistart_init_std,
            opt.multistart_ctrl_std,
        )

    @property
    def dense_steps(self) -> int:
        return self.horizon * self.knot_substeps


@dataclass(frozen=True)
class Plan:
    raw: np.ndarray
    initial_qpos: np.ndarray
    next_qpos: np.ndarray
    next_qvel: np.ndarray
    action: np.ndarray
    qpos_sequence: np.ndarray
    qvel_sequence: np.ndarray
    control_sequence: np.ndarray
    loss: float
    drift: float
    grad_norm: float
    seconds: float
    diag: dict

    @property
    def scores(self) -> np.ndarray:
        return self.diag["scores"]


@wp.struct
class ControlModel:
    hand_qpos: wp.array(dtype=int)
    hand_ctrl: wp.array(dtype=int)
    init_bound: wp.array(dtype=float)
    action_dim: int
    qpos_dim: int
    qvel_dim: int


@wp.struct
class InitialStateJob:
    model: ControlModel
    raw: wp.array2d(dtype=float)
    base_qpos: wp.array(dtype=float)
    base_qvel: wp.array(dtype=float)
    hand_reference: wp.array(dtype=float)
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    allow_init: float


@wp.struct
class ControlJob:
    model: ControlModel
    raw: wp.array2d(dtype=float)
    target: wp.array(dtype=float)
    action: wp.array2d(dtype=float)
    step: int


@wp.struct
class DenseControlJob:
    model: ControlModel
    left_action: wp.array2d(dtype=float)
    right_action: wp.array2d(dtype=float)
    control: wp.array2d(dtype=float)
    fraction: float


@wp.struct
class GradientJob:
    model: ControlModel
    control_gradient: wp.array2d(dtype=float)
    raw_gradient: wp.array2d(dtype=float)
    step: int


@wp.struct
class InitialGradientJob:
    model: ControlModel
    raw: wp.array2d(dtype=float)
    qpos_gradient: wp.array2d(dtype=float)
    raw_gradient: wp.array2d(dtype=float)
    allow_init: float


@wp.struct
class BestStateJob:
    evaluation_raw: wp.array2d(dtype=float)
    evaluation_grad_norm: wp.array(dtype=float)
    loss: wp.array(dtype=float)
    valid: wp.array(dtype=int)
    iteration: wp.array(dtype=int)
    best_raw: wp.array2d(dtype=float)
    best_loss: wp.array(dtype=float)
    best_grad_norm: wp.array(dtype=float)
    best_available: wp.array(dtype=int)
    iterations: int


@wp.struct
class RestoreBestJob:
    raw: wp.array2d(dtype=float)
    best_raw: wp.array2d(dtype=float)
    best_available: wp.array(dtype=int)
    last_raw: wp.array2d(dtype=float)


@wp.struct
class CandidateEvaluationJob:
    raw: wp.array2d(dtype=float)
    loss: wp.array(dtype=float)
    gradient: wp.array2d(dtype=float)
    gradient_norm: wp.array(dtype=float)
    valid: wp.array(dtype=int)
    failure_iteration: wp.array(dtype=int)
    failure_code: wp.array(dtype=int)
    last_raw: wp.array2d(dtype=float)
    last_gradient_norm: wp.array(dtype=float)
    iteration: wp.array(dtype=int)
    gradient_scale: float


@wp.struct
class CandidateUpdateJob:
    raw: wp.array2d(dtype=float)
    first_moment: wp.array2d(dtype=float)
    second_moment: wp.array2d(dtype=float)
    valid: wp.array(dtype=int)
    failure_iteration: wp.array(dtype=int)
    failure_code: wp.array(dtype=int)
    last_raw: wp.array2d(dtype=float)
    iteration: wp.array(dtype=int)


@wp.struct
class SimulatorStatusJob:
    status: wp.array(dtype=int)
    valid: wp.array(dtype=int)
    failure_iteration: wp.array(dtype=int)
    failure_code: wp.array(dtype=int)
    iteration: wp.array(dtype=int)


@wp.struct
class AdamState:
    raw: wp.array2d(dtype=float)
    gradient: wp.array2d(dtype=float)
    first_moment: wp.array2d(dtype=float)
    second_moment: wp.array2d(dtype=float)
    gradient_norm: wp.array(dtype=float)
    valid: wp.array(dtype=int)
    iteration: wp.array(dtype=int)
    iterations: int
    learning_rate: float
    final_learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    clip: float


@dataclass(frozen=True)
class StepRuntime:
    inputs: StepInput
    workspace: object
    recorded: object
    cotangent: StepCotangent


@dataclass(frozen=True)
class ObservationRuntime:
    qpos: wp.array
    workspace: object
    recorded: object
    cotangent: ObservationCotangent


@dataclass(frozen=True)
class ControlRuntime:
    model: ControlModel
    raw: wp.array
    best_raw: wp.array
    best_loss: wp.array
    best_grad_norm: wp.array
    best_available: wp.array
    last_evaluated_raw: wp.array
    last_gradient_norm: wp.array
    valid: wp.array
    failure_iteration: wp.array
    failure_code: wp.array
    gradient: wp.array
    first_moment: wp.array
    second_moment: wp.array
    gradient_norm: wp.array
    initial_qpos: wp.array
    initial_qvel: wp.array
    base_qpos: wp.array
    base_qvel: wp.array
    hand_reference: wp.array
    previous_action: wp.array
    target_actions: tuple
    actions: tuple
    controls: tuple
    knot_gradients: tuple


@wp.kernel
def _copy_initial(job: InitialStateJob):
    world, index = wp.tid()
    if index < job.model.qpos_dim:
        job.qpos[world, index] = job.base_qpos[index]
    if index < job.model.qvel_dim:
        job.qvel[world, index] = job.base_qvel[index]


@wp.kernel(enable_backward=False)
def _replace_invalid_raw(raw: wp.array2d(dtype=float), valid: wp.array(dtype=int)):
    world, index = wp.tid()
    if valid[world] == 1:
        return
    source = int(0)
    found = int(0)
    for candidate in range(raw.shape[0]):
        if found == 0 and valid[candidate] == 1:
            source = candidate
            found = 1
    if found == 1:
        raw[world, index] = raw[source, index]


@wp.kernel
def _apply_initial_delta(job: InitialStateJob):
    world, action = wp.tid()
    delta = job.model.init_bound[action] * wp.tanh(job.raw[world, action])
    qpos = job.model.hand_qpos[action]
    job.qpos[world, qpos] = job.hand_reference[action] + job.allow_init * delta


@wp.kernel
def _build_control_kernel(job: ControlJob):
    world, action = wp.tid()
    offset = job.model.action_dim * (1 + job.step) + action
    job.action[world, action] = job.target[action] + job.raw[world, offset]


@wp.kernel(enable_backward=False)
def _build_dense_control_kernel(job: DenseControlJob):
    world, action = wp.tid()
    left = job.left_action[world, action]
    right = job.right_action[world, action]
    job.control[world, job.model.hand_ctrl[action]] = left + job.fraction * (
        right - left
    )


@wp.kernel(enable_backward=False)
def _add_float(source: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
    world, index = wp.tid()
    target[world, index] += source[world, index]


@wp.kernel(enable_backward=False)
def _add_scaled_float(
    source: wp.array2d(dtype=float), scale: float, target: wp.array2d(dtype=float)
):
    world, index = wp.tid()
    target[world, index] += scale * source[world, index]


@wp.kernel(enable_backward=False)
def _control_gradient(job: GradientJob):
    world, action = wp.tid()
    offset = job.model.action_dim * (1 + job.step) + action
    job.raw_gradient[world, offset] = job.control_gradient[
        world, job.model.hand_ctrl[action]
    ]


@wp.kernel(enable_backward=False)
def _initial_gradient(job: InitialGradientJob):
    world, action = wp.tid()
    tangent = wp.tanh(job.raw[world, action])
    direct = job.qpos_gradient[world, job.model.hand_qpos[action]] * job.allow_init
    derivative = job.model.init_bound[action] * (1.0 - tangent * tangent)
    job.raw_gradient[world, action] = direct * derivative


@wp.kernel(enable_backward=False)
def _gradient_norm(gradient: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    world = wp.tid()
    max_abs = float(0.0)
    bad_value = float(0.0)
    for index in range(gradient.shape[1]):
        value = gradient[world, index]
        if wp.isfinite(value):
            max_abs = wp.max(max_abs, wp.abs(value))
        else:
            bad_value = value
    if not wp.isfinite(bad_value):
        output[world] = bad_value
        return
    if max_abs == 0.0:
        output[world] = 0.0
        return
    squared = float(0.0)
    for index in range(gradient.shape[1]):
        normalized = gradient[world, index] / max_abs
        squared += normalized * normalized
    output[world] = max_abs * wp.sqrt(squared)


@wp.kernel(enable_backward=False)
def _adam(state: AdamState):
    world, index = wp.tid()
    if state.valid[world] == 0:
        return
    iteration = state.iteration[0]
    if iteration >= state.iterations:
        return
    fraction = float(iteration) / float(wp.max(state.iterations - 1, 1))
    alpha = state.final_learning_rate / state.learning_rate
    decay = alpha + (1.0 - alpha) * 0.5 * (1.0 + wp.cos(PI * fraction))
    scale = float(1.0)
    if state.clip > 0.0:
        scale = wp.min(1.0, state.clip / state.gradient_norm[world])
    gradient = state.gradient[world, index] * scale
    first = (
        state.beta1 * state.first_moment[world, index] + (1.0 - state.beta1) * gradient
    )
    second = (
        state.beta2 * state.second_moment[world, index]
        + (1.0 - state.beta2) * gradient * gradient
    )
    state.first_moment[world, index] = first
    state.second_moment[world, index] = second
    correction1 = 1.0 - wp.pow(state.beta1, float(iteration + 1))
    correction2 = 1.0 - wp.pow(state.beta2, float(iteration + 1))
    update = (first / correction1) / (wp.sqrt(second / correction2) + state.epsilon)
    state.raw[world, index] -= state.learning_rate * decay * update


@wp.kernel(enable_backward=False)
def _advance_iteration(iteration: wp.array(dtype=int)):
    iteration[0] += 1


@wp.kernel(enable_backward=False)
def _validate_evaluation(job: CandidateEvaluationJob):
    world = wp.tid()
    if job.valid[world] == 0:
        return
    failure = CANDIDATE_FAILURE_NONE
    unscaled_norm = job.gradient_norm[world] / job.gradient_scale
    if not wp.isfinite(job.loss[world]):
        failure = CANDIDATE_FAILURE_LOSS
    elif not wp.isfinite(job.gradient_norm[world]) or not wp.isfinite(unscaled_norm):
        failure = CANDIDATE_FAILURE_GRADIENT
    else:
        for index in range(job.gradient.shape[1]):
            if not wp.isfinite(job.gradient[world, index]):
                failure = CANDIDATE_FAILURE_GRADIENT
    if failure != CANDIDATE_FAILURE_NONE:
        job.valid[world] = 0
        job.failure_iteration[world] = job.iteration[0]
        job.failure_code[world] = failure
        return
    for index in range(job.raw.shape[1]):
        job.last_raw[world, index] = job.raw[world, index]
    job.last_gradient_norm[world] = unscaled_norm


@wp.kernel(enable_backward=False)
def _validate_update(job: CandidateUpdateJob):
    world = wp.tid()
    if job.valid[world] == 0:
        return
    finite = int(1)
    for index in range(job.raw.shape[1]):
        finite = wp.where(wp.isfinite(job.raw[world, index]), finite, 0)
        finite = wp.where(wp.isfinite(job.first_moment[world, index]), finite, 0)
        finite = wp.where(wp.isfinite(job.second_moment[world, index]), finite, 0)
    if finite == 1:
        return
    job.valid[world] = 0
    job.failure_iteration[world] = job.iteration[0]
    job.failure_code[world] = CANDIDATE_FAILURE_UPDATE
    for index in range(job.raw.shape[1]):
        job.raw[world, index] = job.last_raw[world, index]


@wp.kernel(enable_backward=False)
def _validate_simulator_status(job: SimulatorStatusJob):
    world = wp.tid()
    if job.valid[world] == 0 or job.status[world] == 0:
        return
    job.valid[world] = 0
    job.failure_iteration[world] = job.iteration[0]
    job.failure_code[world] = CANDIDATE_FAILURE_SIMULATOR


@wp.kernel(enable_backward=False)
def _update_best(job: BestStateJob):
    world = wp.tid()
    if job.valid[world] == 1 and job.loss[world] < job.best_loss[world]:
        job.best_loss[world] = job.loss[world]
        job.best_grad_norm[world] = job.evaluation_grad_norm[world]
        job.best_available[world] = 1
        for index in range(job.evaluation_raw.shape[1]):
            job.best_raw[world, index] = job.evaluation_raw[world, index]


@wp.kernel(enable_backward=False)
def _restore_best(job: RestoreBestJob):
    world, index = wp.tid()
    if job.best_available[world] == 1:
        job.raw[world, index] = job.best_raw[world, index]
    else:
        job.raw[world, index] = job.last_raw[world, index]


class WindowOptimizer:
    """Stateful Warp MPC optimizer with native forward and adjoint kernels."""

    def __init__(
        self,
        cpu_model: mujoco.MjModel,
        device_model,
        index: ModelIndex,
        *,
        guidance_count: int,
        cfg: RetargetConfig,
    ):
        settings = OptimizerSettings.from_config(cfg)
        if (
            min(
                settings.horizon,
                settings.knot_substeps,
                settings.execution_substeps,
                settings.worlds,
            )
            < 1
        ):
            raise ValueError(
                "horizon, knot/execution substeps, and worlds must be positive"
            )
        if guidance_count < 1:
            raise ValueError("guidance_count must be positive")
        self.cpu_model = cpu_model
        self.index = index
        self.cfg = cfg
        self.settings = settings
        self.guidance_count = guidance_count
        self.dense_knot_map = action_knot_interpolation_map(
            settings.horizon,
            settings.knot_substeps,
        )
        self.compiled = compile_fast_step(cpu_model, device_model)
        self.device = self.compiled.base.device
        self.control = self._allocate_control()
        self.steps = self._allocate_steps()
        self.terminal_observation = self._allocate_terminal_observation()
        self.objective = self._make_objective(
            cpu_model,
            index,
            cfg=cfg,
            guidance_count=guidance_count,
        )
        self.loss = self.objective.training
        self.iteration = wp.zeros(1, dtype=int, device=self.device)
        self.graphs = {}
        self._regularization_previous_action: np.ndarray | None = None

    def _make_objective(
        self,
        cpu_model: mujoco.MjModel,
        index: ModelIndex,
        *,
        cfg: RetargetConfig,
        guidance_count: int,
    ) -> MpcObjective:
        return MpcObjective(
            cpu_model,
            self.compiled,
            index,
            self.steps,
            self.terminal_observation,
            cfg=cfg,
            guidance_count=guidance_count,
            worlds=self.settings.worlds,
        )

    def solve(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        targets: WindowTargets,
        *,
        warm: np.ndarray,
        previous_action: np.ndarray,
        allow_init: bool,
        seed: int,
        execution_steps: int,
        start_time: float,
    ) -> Plan:
        execution_steps = validate_execution_steps(
            execution_steps, self.settings.execution_substeps
        )
        self._clear_contact_history()
        self._prepare_window(
            qpos,
            qvel,
            targets,
            warm=warm,
            previous_action=previous_action,
            seed=seed,
            start_time=start_time,
        )
        rebuilt = self._ensure_graph(allow_init=allow_init)
        if rebuilt:
            self._prepare_window(
                qpos,
                qvel,
                targets,
                warm=warm,
                previous_action=previous_action,
                seed=seed,
                start_time=start_time,
            )
        # Clear warm-up failures before optimizing the first window.
        self._clear_contact_history()
        self.iteration.zero_()
        started = perf_counter()
        # Evaluate the final Adam update without taking another step.
        for _ in range(self.settings.iterations + 1):
            wp.capture_launch(self.graphs[allow_init])
        restore = RestoreBestJob()
        restore.raw = self.control.raw
        restore.best_raw = self.control.best_raw
        restore.best_available = self.control.best_available
        restore.last_raw = self.control.last_evaluated_raw
        wp.launch(_restore_best, dim=self.control.raw.shape, inputs=[restore])
        self._clear_contact_history()
        self._set_hard_contact()
        self._forward(allow_init=allow_init)
        wp.synchronize()
        evaluation = self._evaluate_candidates(targets, execution_steps)
        selected = self._select_candidate(evaluation)
        plan = self._plan(
            selected, started, evaluation=evaluation, execution_steps=execution_steps
        )
        from comfree_warp.native_adjoint.continuous_contact import audit
        audit()
        self._commit_regularization_history()
        return plan

    def _commit_regularization_history(self) -> None:
        previous = self._regularization_previous_action
        if previous is None:
            raise RuntimeError("regularization boundary was not prepared")
        self.objective.commit_window(previous)

    def _select_candidate(self, evaluation: CandidateEvaluation) -> int:
        return int(np.argmin(evaluation.selection_scores))

    def _clear_contact_history(self) -> None:
        for runtime in self.steps:
            clear_contact_status(self.compiled, runtime.workspace.forward)

    def _allocate_control(self) -> ControlRuntime:
        config = self.settings
        model = _control_model(self.cpu_model, self.index, self.device, cfg=self.cfg)
        raw_dim = self.index.action_dim * (1 + config.horizon)
        matrix = lambda width: wp.empty(
            (config.worlds, width),
            dtype=float,
            device=self.device,
            requires_grad=True,
            retain_grad=True,
        )
        actions = tuple(matrix(self.index.action_dim) for _ in range(config.horizon))
        controls = tuple(matrix(self.cpu_model.nu) for _ in range(config.dense_steps))
        targets = tuple(
            wp.empty(self.index.action_dim, dtype=float, device=self.device)
            for _ in range(config.horizon)
        )
        knot_gradients = tuple(matrix(self.cpu_model.nu) for _ in range(config.horizon))
        return ControlRuntime(
            model=model,
            raw=matrix(raw_dim),
            best_raw=matrix(raw_dim),
            best_loss=wp.empty(config.worlds, dtype=float, device=self.device),
            best_grad_norm=wp.empty(config.worlds, dtype=float, device=self.device),
            best_available=wp.empty(config.worlds, dtype=int, device=self.device),
            last_evaluated_raw=matrix(raw_dim),
            last_gradient_norm=wp.empty(config.worlds, dtype=float, device=self.device),
            valid=wp.empty(config.worlds, dtype=int, device=self.device),
            failure_iteration=wp.empty(config.worlds, dtype=int, device=self.device),
            failure_code=wp.empty(config.worlds, dtype=int, device=self.device),
            gradient=matrix(raw_dim),
            first_moment=matrix(raw_dim),
            second_moment=matrix(raw_dim),
            gradient_norm=wp.empty(config.worlds, dtype=float, device=self.device),
            initial_qpos=matrix(self.cpu_model.nq),
            initial_qvel=matrix(self.cpu_model.nv),
            base_qpos=wp.empty(self.cpu_model.nq, dtype=float, device=self.device),
            base_qvel=wp.empty(self.cpu_model.nv, dtype=float, device=self.device),
            hand_reference=wp.empty(
                self.index.action_dim, dtype=float, device=self.device
            ),
            previous_action=matrix(self.index.action_dim),
            target_actions=targets,
            actions=actions,
            controls=controls,
            knot_gradients=knot_gradients,
        )

    def _allocate_steps(self) -> tuple[StepRuntime, ...]:
        steps = []
        qpos = self.control.initial_qpos
        qvel = self.control.initial_qvel
        time = wp.zeros(
            self.settings.worlds,
            dtype=float,
            device=self.device,
            requires_grad=True,
            retain_grad=True,
        )
        for step in range(self.settings.dense_steps):
            runtime = _step_runtime(
                self.compiled,
                qpos=qpos,
                qvel=qvel,
                control=self.control.controls[step],
                time=time,
                worlds=self.settings.worlds,
                freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP,
            )
            steps.append(runtime)
            qpos = runtime.recorded.result.qpos
            qvel = runtime.recorded.result.qvel
            time = runtime.recorded.result.time
        return tuple(steps)

    def _allocate_terminal_observation(self) -> ObservationRuntime:
        qpos = self.steps[-1].recorded.result.qpos
        workspace = allocate_observation_workspace(
            self.compiled.base, self.settings.worlds
        )
        recorded = record_observation(
            self.compiled.base,
            qpos,
            workspace,
            freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP,
        )
        cotangent = allocate_observation_cotangent(
            self.compiled.base, self.settings.worlds
        )
        return ObservationRuntime(qpos, workspace, recorded, cotangent)

    def _prepare_window(
        self, qpos, qvel, targets, *, warm, previous_action, seed, start_time
    ) -> None:
        if not np.isfinite(start_time):
            raise ValueError("Warp planner start_time must be finite")
        control = self.control
        self.steps[0].inputs.time.fill_(float(start_time))
        control.base_qpos.assign(np.asarray(qpos, np.float32))
        control.base_qvel.assign(np.asarray(qvel, np.float32))
        control.hand_reference.assign(np.asarray(qpos)[self.index.hand_qpos])
        regularization_previous = np.asarray(previous_action, np.float32)
        previous = np.broadcast_to(
            regularization_previous, (self.settings.worlds, self.index.action_dim)
        )
        control.previous_action.assign(previous)
        self._regularization_previous_action = regularization_previous.copy()
        for target_buffer, target in zip(control.target_actions, targets.actions):
            target_buffer.assign(target)
        self.objective.prepare_window(
            targets, regularization_previous, previous_previous_action=None
        )
        control.raw.assign(self._multistart(warm, seed))
        control.best_raw.assign(control.raw)
        control.best_loss.fill_(BEST_LOSS_SENTINEL)
        control.best_grad_norm.fill_(float("inf"))
        control.best_available.zero_()
        control.last_evaluated_raw.assign(control.raw)
        control.last_gradient_norm.zero_()
        control.valid.fill_(1)
        control.failure_iteration.fill_(NO_FAILURE_ITERATION)
        control.failure_code.fill_(CANDIDATE_FAILURE_NONE)
        control.first_moment.zero_()
        control.second_moment.zero_()

    def _multistart(self, warm: np.ndarray, seed: int) -> np.ndarray:
        config = self.settings
        return multistart_raws(
            warm,
            candidate_count=config.worlds,
            initial_dimension=self.index.action_dim,
            initial_std=config.init_noise,
            control_std=config.control_noise,
            initial_raw_limit=config.initial_raw_limit,
            seed=seed,
        )

    def _iteration(self, *, allow_init: bool) -> None:
        self._set_hard_contact()
        wp.launch(
            _replace_invalid_raw,
            dim=self.control.raw.shape,
            inputs=[self.control.raw, self.control.valid],
        )
        self._forward(allow_init=allow_init)
        self._validate_simulator_candidates()
        self._evaluate_losses(allow_init=allow_init)
        self._backward(allow_init=allow_init)
        validate_evaluated_candidates(
            self.control, self.loss.total, iteration=self.iteration
        )
        _track_best(
            self.control,
            self.loss.total,
            config=self.settings,
            iteration=self.iteration,
        )
        _adam_update(self.control, self.settings, self.iteration)
        _validate_optimizer_updates(self.control, iteration=self.iteration)
        wp.launch(_advance_iteration, dim=1, inputs=[self.iteration])

    def _ensure_graph(self, *, allow_init: bool) -> bool:
        if allow_init in self.graphs:
            return False
        if not self.graphs:
            self._iteration(allow_init=allow_init)
            wp.synchronize()
        self.iteration.zero_()
        with wp.ScopedCapture(device=self.device) as capture:
            self._iteration(allow_init=allow_init)
        self.graphs[allow_init] = capture.graph
        return True

    def _set_hard_contact(self) -> None:
        for runtime in self.steps:
            runtime.inputs.contact_softness.zero_()

    def _validate_simulator_candidates(self) -> None:
        for runtime in self.steps:
            status = self._contact_failure_history(runtime)
            if status is None:
                continue
            job = SimulatorStatusJob()
            job.status = status
            job.valid = self.control.valid
            job.failure_iteration = self.control.failure_iteration
            job.failure_code = self.control.failure_code
            job.iteration = self.iteration
            wp.launch(
                _validate_simulator_status,
                dim=self.settings.worlds,
                inputs=[job],
            )

    def _contact_failure_history(self, runtime: StepRuntime):
        status = getattr(runtime.workspace.forward.contact, "failure_history", None)
        if self.compiled.sparse_contact and status is None:
            raise RuntimeError("sparse Warp contact solver is missing failure status")
        return status

    def _forward(self, *, allow_init: bool) -> None:
        _build_initial(self.control, allow_init=allow_init)
        for step in range(self.settings.horizon):
            _build_control(self.control, step)
        for dense_step, knot_map in enumerate(self.dense_knot_map):
            _build_dense_control(self.control, dense_step, knot_map)
        for runtime in self.steps:
            step_forward(self.compiled, runtime.inputs, runtime.workspace)
        observation_forward(
            self.compiled.base,
            self.terminal_observation.qpos,
            self.terminal_observation.workspace,
        )

    def _evaluate_losses(self, *, allow_init: bool) -> None:
        self.objective.evaluate(self.control, allow_init=allow_init)


    def _backward(self, *, allow_init: bool) -> None:
        self.objective.backward_tracking()
        for runtime, workspace in zip(self.steps, self.loss.immediate):
            _copy_loss_cotangent(runtime, workspace.job.state)
        self.objective.backward_terminal()
        _add_terminal_state_cotangent(self.steps[-1], self.loss.terminal.job.state)
        observation = self.terminal_observation
        _copy_observation_cotangent(observation.cotangent, self.loss.terminal.job.state)
        terminal_gradient = observation_backward(
            observation.qpos,
            observation.workspace,
            recorded=observation.recorded,
            cotangent=observation.cotangent,
        )
        _add_array(terminal_gradient, self.steps[-1].cotangent.qpos)
        self.objective.add_state_cotangent()
        for gradient in self.control.knot_gradients:
            gradient.zero_()
        for step in reversed(range(self.settings.dense_steps)):
            runtime = self.steps[step]
            gradient = step_backward(
                BackwardCall(
                    self.compiled,
                    runtime.inputs,
                    runtime.workspace,
                    runtime.recorded,
                    runtime.cotangent,
                )
            )
            _accumulate_dense_control_gradient(
                self.control, gradient.ctrl, self.dense_knot_map[step]
            )
            if step > 0:
                _add_array(gradient.qpos, self.steps[step - 1].cotangent.qpos)
                _add_array(gradient.qvel, self.steps[step - 1].cotangent.qvel)
        for step in reversed(range(self.settings.horizon)):
            _control_raw_gradient(
                self.control, self.control.knot_gradients[step], step=step
            )
        _initial_raw_gradient(
            self.control, self.steps[0].inputs.dynamics.qpos.grad, allow_init=allow_init
        )
        self.objective.add_regularization_gradient(self.control, allow_init=allow_init)

    def _evaluate_candidates(
        self,
        targets: WindowTargets,
        execution_steps: int,
    ) -> CandidateEvaluation:
        require_finite_array("optimized Warp raw parameters", self.control.raw.numpy())
        scores = self.control.best_loss.numpy().astype(np.float32)
        drifts = self._candidate_drifts(targets, execution_steps)
        hard_simulator_valid = self._hard_simulator_valid()
        evaluation = finalize_candidate_evaluation(
            scores,
            drifts,
            has_finite_best=self._finite_best_candidates(),
            hard_simulator_valid=hard_simulator_valid,
            training_failure_iteration=self.control.failure_iteration.numpy(),
            training_failure_code=self.control.failure_code.numpy(),
        )
        _report_candidate_failures(evaluation)
        if not np.any(evaluation.valid):
            raise FloatingPointError("all multistart candidates are non-finite")
        return evaluation

    def _finite_best_candidates(self) -> np.ndarray:
        available = self.control.best_available.numpy().astype(bool)
        available &= self.control.best_loss.numpy() < BEST_LOSS_SENTINEL_VALUE
        return finite_best_state(
            self.control.best_loss.numpy(),
            self.control.best_raw.numpy(),
            self.control.best_grad_norm.numpy(),
            available=available,
            array_module=np,
        )

    def _hard_simulator_valid(self) -> np.ndarray:
        valid = np.ones(self.settings.worlds, dtype=bool)
        for runtime in self.steps:
            status = self._contact_failure_history(runtime)
            if status is None:
                continue
            history = status.numpy()
            valid &= history == 0
        return valid

    def _candidate_drifts(
        self,
        targets: WindowTargets,
        execution_steps: int,
    ) -> np.ndarray:
        step = execution_steps - 1
        qpos = self.steps[step].recorded.result.qpos.numpy()
        actual = qpos[:, self.index.object_qpos : self.index.object_qpos + 3]
        target = targets.object_qpos[step, :3]
        return np.linalg.norm(actual - target, axis=1).astype(np.float32)

    def _plan(
        self,
        selected: int,
        started: float,
        *,
        evaluation: CandidateEvaluation,
        execution_steps: int,
    ) -> Plan:
        raw = require_finite_array(
            "selected Warp raw parameters", self.control.raw.numpy()[selected]
        )
        initial_qpos = require_finite_array(
            "selected Warp initial qpos", self.control.initial_qpos.numpy()[selected]
        )
        qpos_sequence, qvel_sequence, control_sequence = self._selected_interval(
            selected, execution_steps
        )
        diag = _candidate_diagnostics(selected, evaluation)
        return Plan(
            raw,
            initial_qpos,
            qpos_sequence[-1],
            qvel_sequence[-1],
            control_sequence[-1],
            qpos_sequence,
            qvel_sequence,
            control_sequence,
            float(evaluation.scores[selected]),
            float(evaluation.drifts[selected]),
            float(self.control.best_grad_norm.numpy()[selected]),
            perf_counter() - started,
            diag,
        )

    def _selected_interval(
        self,
        selected: int,
        execution_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        interval = self.steps[:execution_steps]
        qpos_sequence = require_finite_array(
            "selected Warp rollout qpos",
            np.stack(
                [runtime.recorded.result.qpos.numpy()[selected] for runtime in interval]
            ),
        )
        qvel_sequence = require_finite_array(
            "selected Warp rollout qvel",
            np.stack(
                [runtime.recorded.result.qvel.numpy()[selected] for runtime in interval]
            ),
        )
        control_sequence = require_finite_array(
            "selected Warp controls",
            np.stack(
                [
                    control.numpy()[selected, self.index.hand_ctrl]
                    for control in self.control.controls[:execution_steps]
                ]
            ),
        )
        return qpos_sequence, qvel_sequence, control_sequence


def _candidate_diagnostics(
    selected: int,
    evaluation: CandidateEvaluation,
) -> dict:
    return {
        "selected": selected,
        "scores": evaluation.scores,
        "drifts": evaluation.drifts,
        "has_finite_best": evaluation.has_finite_best,
        "valid": evaluation.valid,
        "training_failure_iteration": evaluation.training_failure_iteration,
        "training_failure_code": evaluation.training_failure_code,
        "rejection_code": evaluation.rejection_code,
    }


def _grouped_action_values(
    action_dim: int,
    values: tuple[float, float, float],
) -> np.ndarray:
    output = np.full(action_dim, values[2], np.float32)
    output[:3] = values[0]
    output[3:BASE_ACTION_DIM] = values[1]
    return output


def _control_model(
    cpu_model, index: ModelIndex, device, *, cfg: RetargetConfig
) -> ControlModel:
    bounds = cfg.bounds
    model = ControlModel()
    model.hand_qpos = wp.array(index.hand_qpos, dtype=int, device=device)
    model.hand_ctrl = wp.array(index.hand_ctrl, dtype=int, device=device)
    model.init_bound = wp.array(
        _grouped_action_values(
            index.action_dim,
            (bounds.init_base_pos, bounds.init_base_rot, bounds.init_finger),
        ),
        dtype=float,
        device=device,
    )
    model.action_dim = index.action_dim
    model.qpos_dim = cpu_model.nq
    model.qvel_dim = cpu_model.nv
    return model


def _step_runtime(
    compiled, *, qpos, qvel, control, time, worlds: int, freeze_frame_vjp: bool,
) -> StepRuntime:
    dynamics = DynamicsInput()
    dynamics.qpos = qpos
    dynamics.qvel = qvel
    dynamics.ctrl = control
    dynamics.qfrc_applied = wp.zeros(
        (worlds, compiled.base.dynamics.dof_count),
        dtype=float,
        device=compiled.base.device,
    )
    softness = wp.zeros(
        worlds,
        dtype=float,
        device=compiled.base.device,
        requires_grad=True,
        retain_grad=True,
    )
    inputs = StepInput(dynamics, time, softness)
    workspace = allocate_workspace(compiled, worlds)
    recorded = record(
        compiled,
        inputs,
        workspace,
        freeze_frame_vjp=freeze_frame_vjp,
    )
    cotangent = _allocate_cotangent(compiled, worlds)
    return StepRuntime(inputs, workspace, recorded, cotangent)


def _allocate_cotangent(compiled, worlds: int) -> StepCotangent:
    base = compiled.base
    zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=base.device)
    return StepCotangent(
        zeros((worlds, base.integration.position_count)),
        zeros((worlds, base.integration.velocity_count)),
        zeros((worlds, base.integration.velocity_count)),
        zeros(worlds),
        zeros((worlds, base.integration.velocity_count)),
        zeros((worlds, base.kinematics.body_count), wp.vec3),
        zeros((worlds, base.kinematics.body_count), wp.mat33),
        zeros((worlds, base.collision.contact_count)),
        zeros((worlds, base.collision.contact_count), wp.vec3),
        zeros((worlds, base.collision.contact_count), wp.mat33),
    )


def _build_initial(control: ControlRuntime, *, allow_init: bool) -> None:
    job = InitialStateJob()
    job.model = control.model
    job.raw = control.raw
    job.base_qpos = control.base_qpos
    job.base_qvel = control.base_qvel
    job.hand_reference = control.hand_reference
    job.qpos = control.initial_qpos
    job.qvel = control.initial_qvel
    job.allow_init = float(allow_init)
    worlds = control.raw.shape[0]
    width = max(control.model.qpos_dim, control.model.qvel_dim)
    wp.launch(_copy_initial, dim=(worlds, width), inputs=[job])
    wp.launch(
        _apply_initial_delta, dim=(worlds, control.model.action_dim), inputs=[job]
    )


def _build_control(control: ControlRuntime, step: int) -> None:
    job = ControlJob()
    job.model = control.model
    job.raw = control.raw
    job.target = control.target_actions[step]
    job.action = control.actions[step]
    job.step = step
    wp.launch(
        _build_control_kernel,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


def _build_dense_control(
    control: ControlRuntime, dense_step: int, knot_map: tuple
) -> None:
    left, right, fraction = knot_map
    control.controls[dense_step].zero_()
    job = DenseControlJob()
    job.model = control.model
    job.left_action = control.previous_action if left < 0 else control.actions[left]
    job.right_action = control.actions[right]
    job.control = control.controls[dense_step]
    job.fraction = fraction
    wp.launch(
        _build_dense_control_kernel,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


def _zero_cotangent(cotangent: StepCotangent) -> None:
    values = (
        cotangent.qpos,
        cotangent.qvel,
        cotangent.qacc,
        cotangent.time,
        cotangent.constraint_force,
        cotangent.body_position,
        cotangent.body_matrix,
        cotangent.contact_distance,
        cotangent.contact_position,
        cotangent.contact_frame,
    )
    for value in values:
        value.zero_()


def _copy_loss_cotangent(runtime: StepRuntime, state: LossState) -> None:
    cotangent = runtime.cotangent
    _zero_cotangent(cotangent)
    wp.copy(cotangent.qpos, state.qpos.grad)
    _copy_context_cotangent(runtime, state)


def _copy_context_cotangent(runtime: StepRuntime, state: LossState) -> None:
    cotangent = runtime.cotangent
    wp.copy(cotangent.body_position, state.body_position.grad)
    wp.copy(cotangent.body_matrix, state.body_matrix.grad)
    wp.copy(cotangent.contact_distance, state.contact_distance.grad)
    wp.copy(cotangent.contact_position, state.contact_position.grad)
    wp.copy(cotangent.contact_frame, state.contact_frame.grad)


def _copy_observation_cotangent(
    cotangent: ObservationCotangent, state: LossState
) -> None:
    zero_observation_cotangent(cotangent)
    wp.copy(cotangent.body_position, state.body_position.grad)
    wp.copy(cotangent.body_matrix, state.body_matrix.grad)
    wp.copy(cotangent.contact_distance, state.contact_distance.grad)
    wp.copy(cotangent.contact_position, state.contact_position.grad)
    wp.copy(cotangent.contact_frame, state.contact_frame.grad)


def _add_terminal_state_cotangent(runtime: StepRuntime, state: LossState) -> None:
    _add_array(state.qpos.grad, runtime.cotangent.qpos)


def _add_array(source, target) -> None:
    wp.launch(_add_float, dim=source.shape, inputs=[source], outputs=[target])


def _add_scaled_array(source, scale: float, target) -> None:
    wp.launch(
        _add_scaled_float, dim=source.shape, inputs=[source, scale], outputs=[target]
    )


def _accumulate_dense_control_gradient(
    control: ControlRuntime,
    source,
    knot_map: tuple,
) -> None:
    left, right, fraction = knot_map
    if left >= 0 and fraction < 1.0:
        _add_scaled_array(source, 1.0 - fraction, control.knot_gradients[left])
    _add_scaled_array(source, fraction, control.knot_gradients[right])


def _control_raw_gradient(control, ctrl_gradient, *, step: int) -> None:
    job = GradientJob()
    job.model = control.model
    job.control_gradient = ctrl_gradient
    job.raw_gradient = control.gradient
    job.step = step
    wp.launch(
        _control_gradient,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


def _initial_raw_gradient(control, qpos_gradient, *, allow_init: bool) -> None:
    job = InitialGradientJob()
    job.model = control.model
    job.raw = control.raw
    job.qpos_gradient = qpos_gradient
    job.raw_gradient = control.gradient
    job.allow_init = float(allow_init)
    wp.launch(
        _initial_gradient,
        dim=(control.raw.shape[0], control.model.action_dim),
        inputs=[job],
    )


def _adam_update(control: ControlRuntime, config: OptimizerSettings, iteration) -> None:
    state = AdamState()
    state.raw = control.raw
    state.gradient = control.gradient
    state.first_moment = control.first_moment
    state.second_moment = control.second_moment
    state.gradient_norm = control.gradient_norm
    state.valid = control.valid
    state.iteration = iteration
    state.iterations = config.iterations
    state.learning_rate = config.learning_rate
    state.final_learning_rate = config.final_learning_rate
    state.beta1 = config.adam_beta1
    state.beta2 = config.adam_beta2
    state.epsilon = config.adam_epsilon * LOSS_GRADIENT_SCALE
    state.clip = config.gradient_clip * LOSS_GRADIENT_SCALE
    wp.launch(_adam, dim=control.raw.shape, inputs=[state])


def validate_evaluated_candidates(control: ControlRuntime, loss, *, iteration) -> None:
    worlds = control.raw.shape[0]
    wp.launch(
        _gradient_norm,
        dim=worlds,
        inputs=[control.gradient],
        outputs=[control.gradient_norm],
    )
    job = CandidateEvaluationJob()
    job.raw = control.raw
    job.loss = loss
    job.gradient = control.gradient
    job.gradient_norm = control.gradient_norm
    job.valid = control.valid
    job.failure_iteration = control.failure_iteration
    job.failure_code = control.failure_code
    job.last_raw = control.last_evaluated_raw
    job.last_gradient_norm = control.last_gradient_norm
    job.iteration = iteration
    job.gradient_scale = LOSS_GRADIENT_SCALE
    wp.launch(_validate_evaluation, dim=worlds, inputs=[job])


def _validate_optimizer_updates(control: ControlRuntime, *, iteration) -> None:
    job = CandidateUpdateJob()
    job.raw = control.raw
    job.first_moment = control.first_moment
    job.second_moment = control.second_moment
    job.valid = control.valid
    job.failure_iteration = control.failure_iteration
    job.failure_code = control.failure_code
    job.last_raw = control.last_evaluated_raw
    job.iteration = iteration
    wp.launch(_validate_update, dim=control.raw.shape[0], inputs=[job])


def _track_best(
    control: ControlRuntime, loss, *, config: OptimizerSettings, iteration
) -> None:
    job = BestStateJob()
    job.evaluation_raw = control.last_evaluated_raw
    job.evaluation_grad_norm = control.last_gradient_norm
    job.loss = loss
    job.valid = control.valid
    job.iteration = iteration
    job.best_raw = control.best_raw
    job.best_loss = control.best_loss
    job.best_grad_norm = control.best_grad_norm
    job.best_available = control.best_available
    job.iterations = config.iterations
    wp.launch(_update_best, dim=config.worlds, inputs=[job])


def _report_candidate_failures(evaluation: CandidateEvaluation) -> None:
    training_failed = np.flatnonzero(
        evaluation.training_failure_code != CANDIDATE_FAILURE_NONE
    )
    rejected = np.flatnonzero(~evaluation.valid)
    if not training_failed.size and not rejected.size:
        return
    training = [
        f"{index + 1}:"
        f"{CANDIDATE_FAILURE_LABELS[evaluation.training_failure_code[index]]}"
        f"@{evaluation.training_failure_iteration[index]}"
        for index in training_failed
    ]
    rejections = [
        f"{index + 1}:{CANDIDATE_FAILURE_LABELS[evaluation.rejection_code[index]]}"
        for index in rejected
    ]
    print(
        f"[multistart] training_failures={training} rejected={rejections} "
        f"valid={int(evaluation.valid.sum())}/{evaluation.valid.size}",
        flush=True,
    )
