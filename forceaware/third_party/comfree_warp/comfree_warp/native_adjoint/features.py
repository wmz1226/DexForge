"""Explicit support contract for the native differentiable backend."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


_EULER = int(mujoco.mjtIntegrator.mjINT_EULER)
_FIXED_GAIN = int(mujoco.mjtGain.mjGAIN_FIXED)
_AFFINE_BIAS = int(mujoco.mjtBias.mjBIAS_AFFINE)
_JOINT_TRANSMISSION = int(mujoco.mjtTrn.mjTRN_JOINT)
_NO_DYNAMICS = int(mujoco.mjtDyn.mjDYN_NONE)


@dataclass(frozen=True)
class ModelFeatures:
  """Static dimensions used to compile a differentiable step."""

  bodies: int
  joints: int
  positions: int
  velocities: int
  actuators: int
  max_body_depth: int


def _require_zero(model: mujoco.MjModel, fields: tuple[str, ...]) -> None:
  nonzero = [name for name in fields if int(getattr(model, name)) != 0]
  if nonzero:
    joined = ", ".join(nonzero)
    raise ValueError(f"native differentiable backend does not support: {joined}")


def _require_uniform(name: str, values: np.ndarray, expected: int) -> None:
  invalid = np.flatnonzero(np.asarray(values) != expected)
  if invalid.size:
    raise ValueError(f"unsupported {name} at indices {invalid.tolist()}")


def _body_depths(model: mujoco.MjModel) -> np.ndarray:
  depths = np.zeros(model.nbody, dtype=np.int32)
  for body in range(1, model.nbody):
    depths[body] = depths[int(model.body_parentid[body])] + 1
  return depths


def _validate_actuators(model: mujoco.MjModel) -> None:
  _require_uniform("actuator dynamics", model.actuator_dyntype, _NO_DYNAMICS)
  _require_uniform("actuator gain", model.actuator_gaintype, _FIXED_GAIN)
  _require_uniform("actuator bias", model.actuator_biastype, _AFFINE_BIAS)
  _require_uniform("actuator transmission", model.actuator_trntype,
                   _JOINT_TRANSMISSION)


def validate_model(model: mujoco.MjModel) -> ModelFeatures:
  """Reject semantics not implemented by the native differentiable path."""
  if int(model.opt.integrator) != _EULER:
    raise ValueError("native differentiable backend requires Euler integration")
  _require_zero(model, ("ntendon", "neq", "nflex", "nplugin"))
  if np.any(model.dof_frictionloss):
    raise ValueError("joint friction loss is not implemented")
  if np.any(model.jnt_stiffness):
    raise ValueError("joint stiffness is not implemented")
  _validate_actuators(model)
  depths = _body_depths(model)
  return ModelFeatures(model.nbody, model.njnt, model.nq, model.nv,
                       model.nu, int(depths.max()))
