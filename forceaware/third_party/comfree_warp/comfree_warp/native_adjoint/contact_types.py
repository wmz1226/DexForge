"""Shared data contracts for the native contact solver and adjoint."""

from __future__ import annotations

import warp as wp


@wp.func
def annealed_softplus(value: float, beta: float) -> float:
  scaled = beta * value
  stable = wp.max(scaled, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(scaled)))
  return stable / beta


@wp.func
def stable_sigmoid(value: float) -> float:
  if value >= 0.0:
    return 1.0 / (1.0 + wp.exp(-value))
  exponential = wp.exp(value)
  return exponential / (1.0 + exponential)


@wp.func
def softplus_beta_gradient(value: float, beta: float) -> float:
  """Derivative with respect to positive beta without subtracting large terms."""
  magnitude = wp.abs(beta * value)
  exponential = wp.exp(-magnitude)
  logarithm = wp.log(1.0 + exponential)
  # Warp has no log1p; retain the small tail lost when 1+exp rounds to 1.
  if exponential < 1.0e-3:
    logarithm = exponential * (1.0 + exponential * (-0.5 + exponential / 3.0))
  entropy = logarithm + magnitude * exponential / (1.0 + exponential)
  return -entropy / (beta * beta)


@wp.struct
class ContactParameters:
  timestep: wp.array(dtype=float)
  row_count: int
  dof_count: int
  iterations: int


@wp.struct
class ContactInput:
  mass: wp.array3d(dtype=float)
  jacobian: wp.array3d(dtype=float)
  position: wp.array2d(dtype=float)
  active: wp.array2d(dtype=int)
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  smooth_force: wp.array2d(dtype=float)
  contact_softness: wp.array(dtype=float)


@wp.struct
class ContactOutput:
  active_velocity_weight: wp.array2d(dtype=float)
  projected_solution: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  constraint_force: wp.array2d(dtype=float)
  total_force: wp.array2d(dtype=float)
  activation_velocity: wp.array2d(dtype=float)
