"""Backend-independent ForceAware time-grid and solve-request contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRID_ATOL_SECONDS = 1.0e-9


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _integer_ratio(outer: float, inner: float, label: str) -> int:
    ratio = outer / inner
    count = int(round(ratio))
    if count < 1 or not np.isclose(
        count * inner, outer, rtol=0.0, atol=GRID_ATOL_SECONDS
    ):
        raise ValueError(
            f"{label} must be an integer grid ratio, got {outer:g}/{inner:g}"
        )
    return count


def _duration_steps(duration: float, physics_dt: float) -> int:
    duration = _positive_finite("duration_seconds", duration)
    ratio = duration / physics_dt
    nearest = int(round(ratio))
    if np.isclose(nearest * physics_dt, duration, rtol=0.0, atol=GRID_ATOL_SECONDS):
        return nearest
    return int(np.ceil(ratio))


@dataclass(frozen=True)
class TimeGrid:
    mpc_dt: float
    action_dt: float
    knot_dt: float
    exec_dt: float
    ref_dt: float
    action_substeps: int
    knot_substeps: int
    executor_substeps_per_mpc_step: int

    @classmethod
    def create(
        cls,
        *,
        mpc_dt: float,
        action_dt: float,
        knot_dt: float,
        exec_dt: float,
        ref_dt: float,
    ) -> "TimeGrid":
        mpc_dt = _positive_finite("mpc_dt", mpc_dt)
        action_dt = _positive_finite("action_dt", action_dt)
        knot_dt = _positive_finite("knot_dt", knot_dt)
        exec_dt = _positive_finite("exec_dt", exec_dt)
        ref_dt = _positive_finite("ref_dt", ref_dt)
        action_substeps = _integer_ratio(action_dt, mpc_dt, "action_dt/mpc_dt")
        knot_substeps = _integer_ratio(knot_dt, mpc_dt, "knot_dt/mpc_dt")
        if action_substeps > knot_substeps:
            raise ValueError(
                f"action_dt must not exceed knot_dt: {action_dt:g} > {knot_dt:g}"
            )
        return cls(
            mpc_dt,
            action_dt,
            knot_dt,
            exec_dt,
            ref_dt,
            action_substeps,
            knot_substeps,
            _integer_ratio(mpc_dt, exec_dt, "mpc_dt/exec_dt"),
        )

    @property
    def executor_substeps_per_knot(self) -> int:
        return self.knot_substeps * self.executor_substeps_per_mpc_step

    @property
    def executor_substeps_per_action(self) -> int:
        return self.action_substeps * self.executor_substeps_per_mpc_step

    def execution_schedule(self, duration_seconds: float) -> tuple[int, ...]:
        total_steps = _duration_steps(duration_seconds, self.mpc_dt)
        return _window_schedule(total_steps, self.action_substeps)


def _window_schedule(total_steps: int, steps_per_window: int) -> tuple[int, ...]:
    windows = (total_steps + steps_per_window - 1) // steps_per_window
    return tuple(
        min(steps_per_window, total_steps - window * steps_per_window)
        for window in range(windows)
    )


def validate_execution_steps(count: int, steps_per_window: int) -> int:
    count = int(count)
    if count < 1 or count > steps_per_window:
        raise ValueError(
            f"execution_steps must be in [1, {steps_per_window}], got {count}"
        )
    return count


@dataclass(frozen=True)
class SolveRequest:
    grid: TimeGrid
    duration_seconds: float
    total_physics_steps: int
    execution_steps_per_window: tuple[int, ...]

    @property
    def executed_duration_seconds(self) -> float:
        return self.total_physics_steps * self.grid.mpc_dt

    @property
    def tail_hold_seconds(self) -> float:
        return self.executed_duration_seconds - self.duration_seconds

    @classmethod
    def create(
        cls,
        grid: TimeGrid,
        *,
        duration_seconds: float,
    ) -> "SolveRequest":
        schedule = grid.execution_schedule(duration_seconds)
        return cls(
            grid,
            float(duration_seconds),
            sum(schedule),
            schedule,
        )
