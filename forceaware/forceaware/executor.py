"""Hard Warp execution at the configured executor timestep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
import warp as wp

from forceaware.config import RetargetConfig
from forceaware.validation import require_finite_array

from comfree_warp.native_adjoint.dynamics import DynamicsInput
from comfree_warp.native_adjoint.fast_step import allocate_workspace
from comfree_warp.native_adjoint.fast_step import check_contact_status
from comfree_warp.native_adjoint.fast_step import clear_contact_status
from comfree_warp.native_adjoint.fast_step import compile_fast_step
from comfree_warp.native_adjoint.fast_step import step as simulation_step
from comfree_warp.native_adjoint.runtime import StepInput

from .targets import ModelIndex


EXECUTOR_WORLDS = 1
PING_PONG_WORKSPACES = 2


@dataclass(frozen=True)
class ActionExecution:
    next_qpos: np.ndarray
    next_qvel: np.ndarray
    qpos_sequence: np.ndarray
    qvel_sequence: np.ndarray


@dataclass(frozen=True)
class DeviceState:
    qpos: wp.array
    qvel: wp.array
    time: wp.array


@dataclass(frozen=True)
class ExecutorRuntime:
    cpu_model: mujoco.MjModel
    device_model: object


class ActionExecutor(Protocol):
    elapsed: float

    def execute_action_sequence(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        actions: np.ndarray,
    ) -> ActionExecution: ...


def build_action_executor(runtime, index, cfg):
    return WarpActionExecutor(
        runtime.cpu_model, runtime.device_model, index=index, cfg=cfg
    )


class WarpActionExecutor:
    """Executes selected controls with the configured ComFree-Warp step."""

    def __init__(
        self,
        cpu_model: mujoco.MjModel,
        device_model,
        *,
        index: ModelIndex,
        cfg: RetargetConfig,
    ):
        grid = cfg.simulator.time_grid
        self.index = index
        self.compiled = compile_fast_step(cpu_model, device_model)
        self.device = self.compiled.base.device
        self.qpos_dim = cpu_model.nq
        self.qvel_dim = cpu_model.nv
        self.control_dim = cpu_model.nu
        self.substeps = grid.executor_substeps_per_mpc_step
        self.exec_dt = grid.exec_dt
        self.elapsed = 0.0
        self.qpos = wp.empty(
            (EXECUTOR_WORLDS, self.qpos_dim), dtype=float, device=self.device
        )
        self.qvel = wp.empty(
            (EXECUTOR_WORLDS, self.qvel_dim), dtype=float, device=self.device
        )
        self.control = wp.zeros(
            (EXECUTOR_WORLDS, self.control_dim), dtype=float, device=self.device
        )
        self.force = wp.zeros(
            (EXECUTOR_WORLDS, self.qvel_dim), dtype=float, device=self.device
        )
        self.time = wp.zeros(EXECUTOR_WORLDS, dtype=float, device=self.device)
        self.softness = wp.zeros(EXECUTOR_WORLDS, dtype=float, device=self.device)
        self.workspaces = tuple(
            allocate_workspace(self.compiled, EXECUTOR_WORLDS)
            for _ in range(PING_PONG_WORKSPACES)
        )

    def execute_action_sequence(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        actions: np.ndarray,
    ) -> ActionExecution:
        qpos, qvel, actions = self._validated_inputs(qpos, qvel, actions=actions)
        state = self._initial_state(qpos, qvel)
        self._clear_status()
        qpos_log, qvel_log = [], []
        workspace_index = 0
        for action in actions:
            self._assign_action(action)
            state, workspace_index = self._advance(state, workspace_index)
            self._check_status()
            host_qpos, host_qvel = self._host_state(state)
            qpos_log.append(host_qpos)
            qvel_log.append(host_qvel)
        self.elapsed += actions.shape[0] * self.substeps * self.exec_dt
        qpos_sequence = np.stack(qpos_log)
        qvel_sequence = np.stack(qvel_log)
        return ActionExecution(
            qpos_sequence[-1], qvel_sequence[-1], qpos_sequence, qvel_sequence
        )

    def _validated_inputs(self, qpos, qvel, *, actions):
        qpos = require_finite_array("Warp executor qpos", qpos)
        qvel = require_finite_array("Warp executor qvel", qvel)
        actions = require_finite_array("Warp executor actions", actions)
        if qpos.shape != (self.qpos_dim,) or qvel.shape != (self.qvel_dim,):
            raise ValueError(
                "Warp executor state shape mismatch: "
                f"qpos={qpos.shape}, qvel={qvel.shape}"
            )
        expected = (self.index.action_dim,)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1:] != expected:
            raise ValueError(
                f"Warp executor actions must have shape (steps, {expected[0]})"
            )
        return qpos, qvel, actions

    def _initial_state(self, qpos: np.ndarray, qvel: np.ndarray) -> DeviceState:
        self.qpos.assign(qpos[None].astype(np.float32, copy=False))
        self.qvel.assign(qvel[None].astype(np.float32, copy=False))
        self.time.assign(np.asarray([self.elapsed], np.float32))
        self.force.zero_()
        return DeviceState(self.qpos, self.qvel, self.time)

    def _assign_action(self, action: np.ndarray) -> None:
        control = np.zeros((EXECUTOR_WORLDS, self.control_dim), np.float32)
        control[0, self.index.hand_ctrl] = action
        self.control.assign(control)

    def _advance(
        self, state: DeviceState, workspace_index: int
    ) -> tuple[DeviceState, int]:
        for _ in range(self.substeps):
            workspace = self.workspaces[workspace_index]
            state = self._step(state, workspace)
            workspace_index = (workspace_index + 1) % PING_PONG_WORKSPACES
        return state, workspace_index

    def _step(self, state: DeviceState, workspace) -> DeviceState:
        dynamics = DynamicsInput()
        dynamics.qpos = state.qpos
        dynamics.qvel = state.qvel
        dynamics.ctrl = self.control
        dynamics.qfrc_applied = self.force
        inputs = StepInput(dynamics, state.time, self.softness)
        result = simulation_step(self.compiled, inputs, workspace)
        return DeviceState(result.qpos, result.qvel, result.time)

    def _host_state(self, state: DeviceState) -> tuple[np.ndarray, np.ndarray]:
        qpos = require_finite_array("Warp executor result qpos", state.qpos.numpy()[0])
        qvel = require_finite_array("Warp executor result qvel", state.qvel.numpy()[0])
        return qpos.astype(np.float32), qvel.astype(np.float32)

    def _clear_status(self) -> None:
        for workspace in self.workspaces:
            clear_contact_status(self.compiled, workspace)

    def _check_status(self) -> None:
        for workspace in self.workspaces:
            check_contact_status(self.compiled, workspace)
