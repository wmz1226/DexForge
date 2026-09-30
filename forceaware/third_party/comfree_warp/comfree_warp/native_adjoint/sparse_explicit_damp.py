"""Compact explicit-damping coefficients for static sparse contact rows."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from comfree_warp.mujoco_warp._src.types import DisableBit
from comfree_warp.mujoco_warp._src.types import vec5
from .contact_rows import SparseConstraintRows
from .contact_rows import StaticSparseLayout


MIN_IMPEDANCE = float(mujoco.mjMINIMP)
MAX_IMPEDANCE = float(mujoco.mjMAXIMP)
MIN_VALUE = float(mujoco.mjMINVAL)


@wp.struct
class SparseExplicitDampParameters:
  layout: StaticSparseLayout
  timestep: wp.array(dtype=float)
  impratio_invsqrt: wp.array(dtype=float)
  body_invweight: wp.array2d(dtype=wp.vec2)
  geom_bodyid: wp.array(dtype=int)
  contact_geom: wp.array(dtype=wp.vec2i)
  contact_friction: wp.array(dtype=vec5)
  friction_scale: wp.array(dtype=float)
  contact_solref: wp.array(dtype=wp.vec2)
  contact_solimp: wp.array(dtype=vec5)
  row_contact: wp.array(dtype=int)
  limit_joint: wp.array(dtype=int)
  jnt_dofadr: wp.array(dtype=int)
  jnt_solref: wp.array2d(dtype=wp.vec2)
  jnt_solimp: wp.array2d(dtype=vec5)
  dof_invweight: wp.array2d(dtype=float)
  refsafe_disabled: bool
  contact_row_count: int
  row_count: int
  row_capacity: int
  dof_count: int


@wp.struct
class SparseCoefficientReductions:
  smooth_acceleration: wp.array2d(dtype=float)
  current_velocity: wp.array2d(dtype=float)


@wp.struct
class SparseExplicitDampState:
  rows: SparseConstraintRows
  qvel: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  weighted_jacobian: wp.array2d(dtype=float)
  reductions: SparseCoefficientReductions
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)


@dataclass(frozen=True)
class SparseExplicitDampInputs:
  rows: SparseConstraintRows
  qvel: wp.array
  qacc_smooth: wp.array


@dataclass(frozen=True)
class CompiledSparseExplicitDamp:
  parameters: SparseExplicitDampParameters
  device: object


@dataclass(frozen=True)
class SparseExplicitDampWorkspace:
  state: SparseExplicitDampState


@wp.func
def _impedance(solimp: vec5, position: float) -> float:
  minimum = wp.clamp(solimp[0], MIN_IMPEDANCE, MAX_IMPEDANCE)
  maximum = wp.clamp(solimp[1], MIN_IMPEDANCE, MAX_IMPEDANCE)
  middle = wp.clamp(solimp[3], MIN_IMPEDANCE, MAX_IMPEDANCE)
  power = wp.max(solimp[4], 1.0)
  width = wp.max(solimp[2], MIN_VALUE)
  x = wp.abs(position) / width
  lower = wp.pow(x, power) / wp.pow(middle, power - 1.0)
  upper = 1.0 - wp.pow(1.0 - x, power) / wp.pow(1.0 - middle, power - 1.0)
  curve = wp.where(x < middle, lower, upper)
  value = wp.clamp(minimum + curve * (maximum - minimum), minimum, maximum)
  return wp.where(x > 1.0, maximum, value)


@wp.kernel
def _weighted_jacobian(params: SparseExplicitDampParameters,
                       state: SparseExplicitDampState):
  world, position = wp.tid()
  state.weighted_jacobian[world, position] = (
      state.rows.jacobian[world, position])


@wp.kernel
def _reduce_rows(params: SparseExplicitDampParameters,
                 state: SparseExplicitDampState):
  world, row = wp.tid()
  if row >= params.row_count:
    state.reductions.smooth_acceleration[world, row] = 0.0
    state.reductions.current_velocity[world, row] = 0.0
    return
  smooth_acceleration = float(0.0)
  current_velocity = float(0.0)
  start = params.layout.row_offsets[row]
  end = start + params.layout.row_nonzero_count[row]
  for position in range(start, end):
    column = params.layout.column_indices[position]
    jacobian = state.rows.jacobian[world, position]
    velocity = state.qvel[world, column]
    current_velocity += jacobian * velocity
    smooth_acceleration += (
        jacobian * state.qacc_smooth[world, column])
  state.reductions.smooth_acceleration[world, row] = smooth_acceleration
  state.reductions.current_velocity[world, row] = current_velocity


@wp.func
def _contact_invweight(params: SparseExplicitDampParameters,
                       world: int, contact: int) -> float:
  geom = params.contact_geom[contact]
  body1 = params.geom_bodyid[geom[0]]
  body2 = params.geom_bodyid[geom[1]]
  model = world % params.body_invweight.shape[0]
  base = (params.body_invweight[model, body1][0]
          + params.body_invweight[model, body2][0])
  friction = params.contact_friction[contact][0]
  friction *= params.friction_scale[world % params.friction_scale.shape[0]]
  ratio = params.impratio_invsqrt[
      world % params.impratio_invsqrt.shape[0]]
  return base * (1.0 + friction * friction) * (
      2.0 * friction * friction * ratio * ratio)


@wp.func
def _row_solref(params: SparseExplicitDampParameters,
                world: int, row: int) -> wp.vec2:
  if row < params.contact_row_count:
    return params.contact_solref[params.row_contact[row]]
  limit = row - params.contact_row_count
  joint = params.limit_joint[limit]
  return params.jnt_solref[world % params.jnt_solref.shape[0], joint]


@wp.func
def _row_solimp(params: SparseExplicitDampParameters,
                world: int, row: int) -> vec5:
  if row < params.contact_row_count:
    return params.contact_solimp[params.row_contact[row]]
  limit = row - params.contact_row_count
  joint = params.limit_joint[limit]
  return params.jnt_solimp[world % params.jnt_solimp.shape[0], joint]


@wp.func
def _row_invweight(params: SparseExplicitDampParameters,
                   world: int, row: int) -> float:
  if row < params.contact_row_count:
    return _contact_invweight(params, world, params.row_contact[row])
  limit = row - params.contact_row_count
  joint = params.limit_joint[limit]
  dof = params.jnt_dofadr[joint]
  return params.dof_invweight[
      world % params.dof_invweight.shape[0], dof]


@wp.func
def _reference_coefficients(params: SparseExplicitDampParameters,
                            world: int, row: int) -> wp.vec2:
  solref = _row_solref(params, world, row)
  solimp = _row_solimp(params, world, row)
  dmax = wp.clamp(solimp[1], MIN_IMPEDANCE, MAX_IMPEDANCE)
  timestep = params.timestep[world % params.timestep.shape[0]]
  timeconst = solref[0]
  if not params.refsafe_disabled:
    timeconst = wp.max(timeconst, 2.0 * timestep)
  k = 1.0 / (dmax * dmax * timeconst * timeconst
             * solref[1] * solref[1])
  b = 2.0 / (dmax * timeconst)
  k = wp.where(solref[0] <= 0.0, -solref[0] / (dmax * dmax), k)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  return wp.vec2(k, b)


@wp.kernel(enable_backward=False)
def _reduce_column_vjp(params: SparseExplicitDampParameters,
                       jacobian: wp.array2d(dtype=float),
                       qvel: wp.array2d(dtype=float),
                       qacc_smooth: wp.array2d(dtype=float),
                       adj_acceleration: wp.array2d(dtype=float),
                       adj_velocity: wp.array2d(dtype=float),
                       jacobian_grad: wp.array2d(dtype=float),
                       qvel_grad: wp.array2d(dtype=float),
                       qacc_grad: wp.array2d(dtype=float)):
  """``d(reductions)/d(jacobian, qvel, qacc_smooth)`` gathered by column.

  Warp's generated adjoint scatters onto each degree of freedom from every row
  that touches it, so its summation order and therefore the gradient are not
  reproducible.  Walking the stored CSC transpose gives every degree of
  freedom a single owning thread and a fixed order; each nonzero belongs to
  exactly one column, so the Jacobian gradient stays disjoint as well.
  """
  world, dof = wp.tid()
  velocity = float(0.0)
  acceleration = float(0.0)
  start = params.layout.column_offsets[dof]
  end = params.layout.column_offsets[dof + 1]
  for slot in range(start, end):
    position = params.layout.csc_positions[slot]
    row = params.layout.csc_rows[slot]
    if row < params.row_count:
      row_velocity = adj_velocity[world, row]
      row_acceleration = adj_acceleration[world, row]
      value = jacobian[world, position]
      velocity += value * row_velocity
      acceleration += value * row_acceleration
      jacobian_grad[world, position] += (
          row_velocity * qvel[world, dof]
          + row_acceleration * qacc_smooth[world, dof])
  qvel_grad[world, dof] += velocity
  qacc_grad[world, dof] += acceleration


@wp.kernel(enable_backward=False)
def _weighted_jacobian_vjp(weighted_grad: wp.array2d(dtype=float),
                           jacobian_grad: wp.array2d(dtype=float)):
  """Reverse of the one-to-one CSR value copy."""
  world, position = wp.tid()
  jacobian_grad[world, position] += weighted_grad[world, position]


@wp.kernel
def _contact_coefficients(params: SparseExplicitDampParameters,
                          state: SparseExplicitDampState):
  world, row = wp.tid()
  if row >= params.row_count or state.rows.active[world, row] == 0:
    state.free_force[world, row] = 0.0
    state.velocity_weight[world, row] = 0.0
    return
  solimp = _row_solimp(params, world, row)
  impedance = _impedance(solimp, state.rows.position[world, row])
  coefficients = _reference_coefficients(params, world, row)
  aref = (-coefficients[0] * impedance * state.rows.position[world, row]
          - coefficients[1] * state.reductions.current_velocity[world, row])
  invweight = _row_invweight(params, world, row)
  D = 1.0 / wp.max(
      invweight * (1.0 - impedance) / impedance, MIN_VALUE)
  z = state.reductions.smooth_acceleration[world, row] - aref
  state.free_force[world, row] = -D * z
  state.velocity_weight[world, row] = D


def compile_sparse_explicit_damp(device_model, collision,
                                 row_layout) -> CompiledSparseExplicitDamp:
  if not row_layout.sparse or row_layout.sparse_layout is None:
    raise ValueError("sparse explicit damping requires a static sparse row layout")
  params = SparseExplicitDampParameters()
  params.layout = row_layout.sparse_layout
  params.timestep = device_model.opt.timestep
  params.impratio_invsqrt = device_model.opt.impratio_invsqrt
  params.body_invweight = device_model.body_invweight0
  params.geom_bodyid = device_model.geom_bodyid
  params.contact_geom = row_layout.parameters.contact_geom
  params.contact_friction = row_layout.parameters.contact_friction
  params.friction_scale = row_layout.parameters.friction_scale
  params.contact_solref = collision.contact_solref
  params.contact_solimp = collision.contact_solimp
  params.row_contact = row_layout.parameters.row_contact
  params.limit_joint = row_layout.parameters.limit_joint
  params.jnt_dofadr = device_model.jnt_dofadr
  params.jnt_solref = device_model.jnt_solref
  params.jnt_solimp = device_model.jnt_solimp
  params.dof_invweight = device_model.dof_invweight0
  params.refsafe_disabled = bool(
      device_model.opt.disableflags & DisableBit.REFSAFE)
  params.contact_row_count = row_layout.contact_row_count
  params.row_count = row_layout.row_count
  params.row_capacity = row_layout.row_capacity
  params.dof_count = row_layout.parameters.dof_count
  return CompiledSparseExplicitDamp(params, device_model.qpos0.device)


def _row_storage(worlds: int, rows: int, device, *, gradient: bool):
  return wp.empty(
      (worlds, rows), dtype=float, device=device,
      requires_grad=gradient, retain_grad=gradient)


def allocate_workspace(compiled: CompiledSparseExplicitDamp, worlds: int,
                       *, gradient: bool) -> SparseExplicitDampWorkspace:
  params = compiled.parameters
  state = SparseExplicitDampState()
  state.weighted_jacobian = wp.empty(
      (worlds, params.layout.nonzero_count), dtype=float,
      device=compiled.device, requires_grad=gradient, retain_grad=gradient)
  reductions = SparseCoefficientReductions()
  for name in ("smooth_acceleration", "current_velocity"):
    setattr(reductions, name, _row_storage(
        worlds, params.row_capacity, compiled.device, gradient=gradient))
  state.reductions = reductions
  state.free_force = _row_storage(
      worlds, params.row_capacity, compiled.device, gradient=gradient)
  state.velocity_weight = _row_storage(
      worlds, params.row_capacity, compiled.device, gradient=gradient)
  return SparseExplicitDampWorkspace(state)


def prepare(compiled: CompiledSparseExplicitDamp,
            inputs: SparseExplicitDampInputs,
            workspace: SparseExplicitDampWorkspace) -> None:
  reduce_rows(compiled, inputs, workspace)
  coefficients(compiled, workspace)


def reduce_rows(compiled: CompiledSparseExplicitDamp,
                inputs: SparseExplicitDampInputs,
                workspace: SparseExplicitDampWorkspace) -> None:
  """Contract the CSR Jacobian against velocity and acceleration.

  Kept off the differentiation tape; ``reduce_rows_vjp`` supplies a reverse
  pass with a fixed accumulation order.
  """
  state = workspace.state
  state.rows = inputs.rows
  state.qvel = inputs.qvel
  state.qacc_smooth = inputs.qacc_smooth
  worlds = inputs.qvel.shape[0]
  wp.launch(
      _weighted_jacobian,
      dim=(worlds, compiled.parameters.layout.nonzero_count),
      inputs=[compiled.parameters, state], device=compiled.device)
  wp.launch(
      _reduce_rows,
      dim=(state.qvel.shape[0], compiled.parameters.row_capacity),
      inputs=[compiled.parameters, state], device=compiled.device)


def reduce_rows_vjp(compiled: CompiledSparseExplicitDamp,
                    workspace: SparseExplicitDampWorkspace) -> None:
  """Deterministic reverse pass of :func:`reduce_rows`, accumulating."""
  state = workspace.state
  params = compiled.parameters
  worlds = state.qvel.shape[0]
  reductions = state.reductions
  wp.launch(
      _reduce_column_vjp, dim=(worlds, params.dof_count),
      inputs=[params, state.rows.jacobian, state.qvel, state.qacc_smooth,
              reductions.smooth_acceleration.grad,
              reductions.current_velocity.grad],
      outputs=[state.rows.jacobian.grad, state.qvel.grad,
               state.qacc_smooth.grad],
      device=compiled.device)
  wp.launch(
      _weighted_jacobian_vjp, dim=(worlds, params.layout.nonzero_count),
      inputs=[state.weighted_jacobian.grad],
      outputs=[state.rows.jacobian.grad], device=compiled.device)


def coefficients(compiled: CompiledSparseExplicitDamp,
                 workspace: SparseExplicitDampWorkspace) -> None:
  """Per-row constraint coefficients; every thread owns one row."""
  state = workspace.state
  wp.launch(
      _contact_coefficients,
      dim=(state.qvel.shape[0], compiled.parameters.row_capacity),
      inputs=[compiled.parameters, state], device=compiled.device)
