"""Hard/soft active-set contact solve for native Warp optimization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import warp as wp

from comfree_warp.comfree_core._src.types import L0_IMPEDANCE

from .contact_types import ContactInput
from .contact_types import ContactOutput
from .contact_types import ContactParameters
from .contact_types import annealed_softplus
from .contact_types import stable_sigmoid


ROW_TILE_SIZE = 32
TILE_BLOCK_SIZE = 64


@wp.struct
class LinearizedRows:
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  slope: wp.array2d(dtype=float)


@wp.struct
class LinearSolveState:
  rows: LinearizedRows
  factor: wp.array3d(dtype=float)
  solution: wp.array2d(dtype=float)
  response: wp.array2d(dtype=float)


@wp.struct
class LinearizeJob:
  params: ContactParameters
  inputs: ContactInput
  response: wp.array2d(dtype=float)
  rows: LinearizedRows


@wp.struct
class SolveJob:
  params: ContactParameters
  inputs: ContactInput
  state: LinearSolveState


@wp.struct
class ProjectionJob:
  params: ContactParameters
  inputs: ContactInput
  state: LinearSolveState
  output: ContactOutput


@wp.struct
class DiagonalJob:
  params: ContactParameters
  inputs: ContactInput
  state: LinearSolveState
  output: ContactOutput


@dataclass(frozen=True)
class AnnealedContactWorkspace:
  output: ContactOutput
  final: LinearSolveState
  states: tuple[LinearSolveState, ...]
  initial_response: wp.array


@wp.kernel(enable_backward=False)
def _no_contact_output(params: ContactParameters, inputs: ContactInput,
                       state: LinearSolveState, output: ContactOutput):
  world, dof = wp.tid()
  timestep = params.timestep[world % params.timestep.shape[0]]
  state.solution[world, dof] = 0.0
  output.constraint_force[world, dof] = 0.0
  output.total_force[world, dof] = inputs.smooth_force[world, dof]
  output.activation_velocity[world, dof] = (
      inputs.qvel[world, dof] + timestep * inputs.qacc_smooth[world, dof])


@cache
def _diagonal_kernel(dofs: int, rows: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: DiagonalJob):
    world = wp.tid()
    size = wp.static(dofs)
    capacity = wp.static(rows)
    projected = wp.tile_zeros(shape=size, dtype=wp.float32)
    for start in range(0, capacity, ROW_TILE_SIZE):
      jacobian = wp.tile_load(
          job.inputs.jacobian[world], shape=(ROW_TILE_SIZE, size),
          offset=(start, 0), bounds_check=False)
      D = wp.tile_load(job.inputs.velocity_weight[world],
                       shape=ROW_TILE_SIZE, offset=start, bounds_check=False)
      free = wp.tile_load(job.inputs.free_force[world],
                          shape=ROW_TILE_SIZE, offset=start, bounds_check=False)
      enabled = wp.tile_load(
          job.inputs.active[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      row_ids = wp.tile_arange(ROW_TILE_SIZE, dtype=int)
      row_bound = wp.tile_ones(
          shape=ROW_TILE_SIZE, dtype=int) * (job.params.row_count - start)
      valid = wp.tile_map(_in_bounds, row_ids, row_bound)
      raw = (free * wp.tile_map(_enabled, enabled) * valid
             * wp.static(L0_IMPEDANCE))
      force = wp.tile_map(wp.max, raw, wp.tile_zeros(
          shape=ROW_TILE_SIZE, dtype=wp.float32))
      active = wp.tile_map(_positive, raw)
      projected += wp.tile_sum(wp.tile_transpose(jacobian) *
          wp.tile_broadcast(force, shape=(size, ROW_TILE_SIZE)), axis=1)
      wp.tile_store(job.state.rows.free_force[world], free, offset=start)
      wp.tile_store(job.state.rows.velocity_weight[world], active * D,
                    offset=start)
      wp.tile_store(job.state.rows.slope[world], active, offset=start)
      wp.tile_store(
          job.state.response[world],
          wp.tile_zeros(shape=ROW_TILE_SIZE, dtype=wp.float32),
          offset=start)
      wp.tile_store(job.output.force[world], force, offset=start)
    smooth = wp.tile_load(job.inputs.smooth_force[world], shape=size)
    qvel = wp.tile_load(job.inputs.qvel[world], shape=size)
    qacc = wp.tile_load(job.inputs.qacc_smooth[world], shape=size)
    timestep = job.params.timestep[world % job.params.timestep.shape[0]]
    wp.tile_store(
        job.state.solution[world],
        wp.tile_zeros(shape=size, dtype=wp.float32))
    wp.tile_store(job.output.constraint_force[world], projected)
    wp.tile_store(job.output.total_force[world], smooth + projected)
    wp.tile_store(job.output.activation_velocity[world],
                  qvel + timestep * qacc)

  return kernel


@wp.func
def _positive(value: float) -> float:
  return wp.where(value > 0.0, 1.0, 0.0)


@wp.func
def _enabled(value: int) -> float:
  return wp.where(value != 0, 1.0, 0.0)


@wp.func
def _in_bounds(index: int, bound: int) -> float:
  return wp.where(index < bound, 1.0, 0.0)


@wp.kernel(enable_backward=False)
def _linearize(job: LinearizeJob):
  world, row = wp.tid()
  if (row >= job.params.row_count
      or job.inputs.active[world, row] == 0):
    job.rows.free_force[world, row] = 0.0
    job.rows.velocity_weight[world, row] = 0.0
    job.rows.slope[world, row] = 0.0
    return
  base_free = job.inputs.free_force[world, row]
  base_weight = job.inputs.velocity_weight[world, row]
  response = job.response[world, row]
  raw = base_free - base_weight * response
  beta = job.inputs.contact_softness[world]
  slope = wp.where(raw > 0.0, 1.0, 0.0)
  linear_free = slope * base_free
  if beta > 0.0:
    slope = stable_sigmoid(beta * raw)
    intercept = annealed_softplus(raw, beta) - slope * raw
    linear_free = slope * base_free + intercept
  job.rows.free_force[world, row] = linear_free
  job.rows.velocity_weight[world, row] = slope * base_weight
  job.rows.slope[world, row] = slope


@cache
def _solve_kernel(dofs: int, rows: int):
  # Reuse tile scratch across row blocks; unrolling duplicates shared memory.
  @wp.kernel(module="unique", enable_backward=False,
             module_options={"max_unroll": 0})
  def kernel(job: SolveJob):
    world = wp.tid()
    size = wp.static(dofs)
    capacity = wp.static(rows)
    matrix = wp.tile_load(
        job.inputs.mass[world], shape=(size, size), bounds_check=False)
    rhs = wp.tile_zeros(shape=size, dtype=wp.float32)
    for start in range(0, capacity, ROW_TILE_SIZE):
      jacobian = wp.tile_load(
          job.inputs.jacobian[world], shape=(ROW_TILE_SIZE, size),
          offset=(start, 0), bounds_check=False)
      weight = wp.tile_load(
          job.state.rows.velocity_weight[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      free = wp.tile_load(
          job.state.rows.free_force[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      weighted = wp.tile_transpose(jacobian) * wp.tile_broadcast(
          weight, shape=(size, ROW_TILE_SIZE))
      matrix += wp.tile_matmul(weighted, jacobian)
      rhs += wp.tile_sum(
          wp.tile_transpose(jacobian)
          * wp.tile_broadcast(free, shape=(size, ROW_TILE_SIZE)), axis=1)
    factor = wp.tile_cholesky(matrix)
    solution = wp.tile_cholesky_solve(factor, rhs)
    wp.tile_store(job.state.factor[world], factor, bounds_check=False)
    wp.tile_store(job.state.solution[world], solution, bounds_check=False)
    for start in range(0, capacity, ROW_TILE_SIZE):
      jacobian = wp.tile_load(
          job.inputs.jacobian[world], shape=(ROW_TILE_SIZE, size),
          offset=(start, 0), bounds_check=False)
      response = wp.tile_sum(
          jacobian * wp.tile_broadcast(
              solution, shape=(ROW_TILE_SIZE, size)), axis=1)
      wp.tile_store(job.state.response[world], response,
                    offset=start, bounds_check=False)

  return kernel


@cache
def _project_kernel(dofs: int, rows: int):
  """Final non-negative projection.

  ``contact_softness == 0`` selects the exact hard branch and reproduces the
  physical model bit for bit.  ``contact_softness == beta > 0`` selects the
  softplus at the final response using the original force coefficients.
  The last Newton linearization is not the nonlinear force law.
  Rows that collision or limit preparation left inactive are
  gated to exactly zero so they never become ``log(2) / beta``.
  """
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: ProjectionJob):
    world = wp.tid()
    size = wp.static(dofs)
    capacity = wp.static(rows)
    timestep = job.params.timestep[world % job.params.timestep.shape[0]]
    beta = job.inputs.contact_softness[world]
    blend = wp.where(beta > 0.0, 1.0, 0.0)
    scale = wp.where(beta > 0.0, beta, 1.0)
    projected = wp.tile_zeros(shape=size, dtype=wp.float32)
    for start in range(0, capacity, ROW_TILE_SIZE):
      jacobian = wp.tile_load(
          job.inputs.jacobian[world],
          shape=(ROW_TILE_SIZE, size), offset=(start, 0), bounds_check=False)
      free = wp.tile_load(
          job.state.rows.free_force[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      weight = wp.tile_load(
          job.state.rows.velocity_weight[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      if beta > 0.0:
        free = wp.tile_load(
            job.inputs.free_force[world], shape=ROW_TILE_SIZE,
            offset=start, bounds_check=False)
        weight = wp.tile_load(
            job.inputs.velocity_weight[world], shape=ROW_TILE_SIZE,
            offset=start, bounds_check=False)
      response = wp.tile_load(
          job.state.response[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      raw = free - weight * response
      hard = wp.tile_map(wp.max, raw,
                         wp.tile_zeros(shape=ROW_TILE_SIZE, dtype=wp.float32))
      enabled = wp.tile_load(
          job.inputs.active[world], shape=ROW_TILE_SIZE,
          offset=start, bounds_check=False)
      row_ids = wp.tile_arange(ROW_TILE_SIZE, dtype=int)
      row_bound = wp.tile_ones(
          shape=ROW_TILE_SIZE, dtype=int) * (job.params.row_count - start)
      gate = (wp.tile_map(_enabled, enabled)
              * wp.tile_map(_in_bounds, row_ids, row_bound))
      soft = wp.tile_map(
          annealed_softplus, raw,
          wp.tile_ones(shape=ROW_TILE_SIZE, dtype=wp.float32) * scale) * gate
      force = hard * (1.0 - blend) + soft * blend
      projected += wp.tile_sum(
          wp.tile_transpose(jacobian)
          * wp.tile_broadcast(force, shape=(size, ROW_TILE_SIZE)), axis=1)
      wp.tile_store(job.output.force[world], force,
                    offset=start, bounds_check=False)
    qacc_smooth = wp.tile_load(
        job.inputs.qacc_smooth[world], shape=size, bounds_check=False)
    smooth_force = wp.tile_load(
        job.inputs.smooth_force[world], shape=size, bounds_check=False)
    qvel = wp.tile_load(job.inputs.qvel[world], shape=size, bounds_check=False)
    correction = wp.tile_load(
        job.state.solution[world], shape=size, bounds_check=False)
    wp.tile_store(job.output.constraint_force[world], projected,
                  bounds_check=False)
    wp.tile_store(job.output.total_force[world], smooth_force + projected,
                  bounds_check=False)
    activation = qvel + timestep * (qacc_smooth + correction)
    wp.tile_store(job.output.activation_velocity[world], activation,
                  bounds_check=False)

  return kernel


def _rows(worlds: int, rows: int, device) -> LinearizedRows:
  result = LinearizedRows()
  shape = (worlds, rows)
  result.free_force = wp.empty(shape, dtype=float, device=device)
  result.velocity_weight = wp.empty(shape, dtype=float, device=device)
  result.slope = wp.empty(shape, dtype=float, device=device)
  return result


def _state(worlds: int, rows: int, dofs: int, *, device) -> LinearSolveState:
  result = LinearSolveState()
  result.rows = _rows(worlds, rows, device)
  result.factor = wp.empty((worlds, dofs, dofs), dtype=float, device=device)
  result.solution = wp.empty((worlds, dofs), dtype=float, device=device)
  result.response = wp.zeros((worlds, rows), dtype=float, device=device)
  return result


def _output(worlds: int, rows: int, dofs: int, *, device,
            final: LinearSolveState) -> ContactOutput:
  output = ContactOutput()
  output.active_velocity_weight = final.rows.velocity_weight
  output.projected_solution = final.solution
  output.force = wp.empty((worlds, rows), dtype=float, device=device)
  output.constraint_force = wp.empty((worlds, dofs), dtype=float, device=device)
  output.total_force = wp.empty((worlds, dofs), dtype=float, device=device)
  output.activation_velocity = wp.empty((worlds, dofs), dtype=float, device=device)
  return output


def _iteration_states(worlds: int, rows: int, dofs: int, *,
                      iterations: int, device) -> tuple:
  if iterations == 0:
    final = _state(worlds, rows, dofs, device=device)
    return final, ()
  states = tuple(
      _state(worlds, rows, dofs, device=device)
      for _ in range(iterations))
  return states[-1], states


def allocate(worlds: int, rows: int, dofs: int, *, iterations: int,
             device) -> AnnealedContactWorkspace:
  if iterations < 0:
    raise ValueError("contact iterations must be non-negative")
  final, states = _iteration_states(
      worlds, rows, dofs, iterations=iterations, device=device)
  return AnnealedContactWorkspace(
      _output(worlds, rows, dofs, device=device, final=final),
      final, states,
      wp.zeros((worlds, rows), dtype=float, device=device))


def _linearize_state(params, inputs, *, response,
                     state: LinearSolveState) -> None:
  job = LinearizeJob()
  job.params = params
  job.inputs = inputs
  job.response = response
  job.rows = state.rows
  wp.launch(_linearize, dim=state.rows.free_force.shape,
            inputs=[job], device=state.rows.free_force.device)


def _solve_state(params, inputs, state: LinearSolveState) -> None:
  job = SolveJob()
  job.params = params
  job.inputs = inputs
  job.state = state
  worlds, rows = state.rows.free_force.shape
  dofs = state.solution.shape[1]
  wp.launch_tiled(_solve_kernel(dofs, rows), dim=worlds,
                  inputs=[job], block_dim=TILE_BLOCK_SIZE,
                  device=state.solution.device)


def _iteration(params, inputs, *, response,
               state: LinearSolveState) -> None:
  _linearize_state(params, inputs, response=response, state=state)
  _solve_state(params, inputs, state)


def _project(params, inputs, *, state, output) -> None:
  job = ProjectionJob()
  job.params = params
  job.inputs = inputs
  job.state = state
  job.output = output
  worlds, rows = state.rows.free_force.shape
  dofs = state.solution.shape[1]
  wp.launch_tiled(_project_kernel(dofs, rows), dim=worlds,
                  inputs=[job], block_dim=TILE_BLOCK_SIZE,
                  device=state.solution.device)


def _solve_diagonal(params, inputs,
                    workspace: AnnealedContactWorkspace) -> wp.array:
  job = DiagonalJob()
  job.params = params
  job.inputs = inputs
  job.state = workspace.final
  job.output = workspace.output
  worlds, rows = inputs.free_force.shape
  dofs = inputs.qvel.shape[1]
  wp.launch_tiled(
      _diagonal_kernel(dofs, rows), dim=worlds, inputs=[job],
      block_dim=TILE_BLOCK_SIZE, device=inputs.qvel.device)
  return workspace.output.constraint_force


def _clear_no_contact_rows(workspace: AnnealedContactWorkspace) -> None:
  workspace.output.force.zero_()
  workspace.final.rows.free_force.zero_()
  workspace.final.rows.velocity_weight.zero_()
  workspace.final.rows.slope.zero_()
  workspace.final.response.zero_()


def solve(params: ContactParameters, inputs: ContactInput,
          workspace: AnnealedContactWorkspace) -> wp.array:
  if len(workspace.states) != params.iterations:
    raise ValueError(
        "contact workspace iteration count does not match parameters: "
        f"{len(workspace.states)} != {params.iterations}")
  if params.row_count == 0:
    _clear_no_contact_rows(workspace)
    wp.launch(
        _no_contact_output,
        dim=(inputs.qvel.shape[0], params.dof_count),
        inputs=[params, inputs, workspace.final, workspace.output],
        device=inputs.qvel.device)
    return workspace.output.constraint_force
  if params.iterations == 0:
    return _solve_diagonal(params, inputs, workspace)
  workspace.initial_response.zero_()
  response = workspace.initial_response
  for state in workspace.states:
    _iteration(params, inputs, response=response, state=state)
    response = state.response
  _project(params, inputs, state=workspace.final, output=workspace.output)
  return workspace.output.constraint_force
