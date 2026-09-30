"""Analytic adjoint of the annealed hard/soft contact solve."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import warp as wp

from comfree_warp.comfree_core._src.types import L0_IMPEDANCE

from .annealed_contact import AnnealedContactWorkspace
from .annealed_contact import LinearSolveState
from .annealed_contact import ROW_TILE_SIZE
from .annealed_contact import TILE_BLOCK_SIZE
from .contact_types import ContactInput
from .contact_types import ContactParameters
from .contact_types import softplus_beta_gradient
from .contact_types import stable_sigmoid


@wp.struct
class LinearSolveGradient:
  projected: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  response: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  adjoint_solution: wp.array2d(dtype=float)
  adjoint_response: wp.array2d(dtype=float)
  jacobian: wp.array3d(dtype=float)
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  mass: wp.array3d(dtype=float)


@wp.struct
class FinalSeedJob:
  params: ContactParameters
  inputs: ContactInput
  jacobian: wp.array3d(dtype=float)
  state: LinearSolveState
  force: wp.array2d(dtype=float)
  constraint_gradient: wp.array2d(dtype=float)
  gradient: LinearSolveGradient
  softness_gradient: wp.array2d(dtype=float)
  base_free_gradient: wp.array2d(dtype=float)
  base_weight_gradient: wp.array2d(dtype=float)


@wp.struct
class SolveBackwardJob:
  params: ContactParameters
  jacobian: wp.array3d(dtype=float)
  state: LinearSolveState
  force: wp.array2d(dtype=float)
  gradient: LinearSolveGradient


@wp.struct
class LinearizeBackwardJob:
  params: ContactParameters
  inputs: ContactInput
  input_response: wp.array2d(dtype=float)
  state: LinearSolveState
  gradient: LinearSolveGradient
  base_free_gradient: wp.array2d(dtype=float)
  base_weight_gradient: wp.array2d(dtype=float)
  response_gradient: wp.array2d(dtype=float)
  softness_gradient: wp.array2d(dtype=float)


@dataclass(frozen=True)
class ContactBackwardInput:
  forward_input: ContactInput
  forward_workspace: AnnealedContactWorkspace
  constraint_gradient: wp.array


@dataclass(frozen=True)
class ContactGradient:
  jacobian: wp.array
  free_force: wp.array
  velocity_weight: wp.array
  mass: wp.array
  contact_softness: wp.array


@dataclass(frozen=True)
class ContactIterationGradient:
  solve: LinearSolveGradient
  base_free: wp.array
  base_weight: wp.array
  softness: wp.array


@dataclass(frozen=True)
class ContactAdjointWorkspace:
  iterations: tuple[ContactIterationGradient, ...]
  initial_response: wp.array
  zero_rows: wp.array
  jacobian: wp.array
  free_force: wp.array
  velocity_weight: wp.array
  mass: wp.array
  softness_rows: wp.array
  contact_softness: wp.array


@wp.kernel(enable_backward=False)
def _projected_seed(job: FinalSeedJob):
  world, dof = wp.tid()
  job.gradient.projected[world, dof] = job.constraint_gradient[world, dof]


@wp.func
def _active(value: float) -> float:
  return wp.where(value > 0.0, 1.0, 0.0)


@cache
def _diagonal_adjoint_kernel(dofs: int, rows: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: FinalSeedJob):
    world = wp.tid()
    size = wp.static(dofs)
    capacity = wp.static(rows)
    cotangent = wp.tile_load(
        job.constraint_gradient[world], shape=size, bounds_check=False)
    for start in range(0, capacity, ROW_TILE_SIZE):
      jacobian = wp.tile_load(
          job.jacobian[world], shape=(ROW_TILE_SIZE, size),
          offset=(start, 0), bounds_check=False)
      force = wp.tile_load(job.force[world], shape=ROW_TILE_SIZE,
                           offset=start, bounds_check=False)
      force_gradient = wp.tile_sum(jacobian * wp.tile_broadcast(
          cotangent, shape=(ROW_TILE_SIZE, size)), axis=1)
      force_gradient *= wp.tile_map(_active, force)
      direct = wp.tile_transpose(wp.tile_broadcast(
          force, shape=(size, ROW_TILE_SIZE))) * wp.tile_broadcast(
              cotangent, shape=(ROW_TILE_SIZE, size))
      wp.tile_store(job.gradient.jacobian[world], direct,
                    offset=(start, 0), bounds_check=False)
      wp.tile_store(
          job.gradient.free_force[world],
          force_gradient * wp.static(L0_IMPEDANCE),
          offset=start, bounds_check=False)
      wp.tile_store(
          job.gradient.velocity_weight[world],
          wp.tile_zeros(shape=ROW_TILE_SIZE, dtype=wp.float32),
          offset=start, bounds_check=False)
    wp.tile_store(
        job.gradient.mass[world],
        wp.tile_zeros(shape=(size, size), dtype=wp.float32),
        bounds_check=False)

  return kernel


@cache
def _final_row_seed(dofs: int):
  """Adjoint of the final projection, matching its forward annealing.

  ``beta == 0`` reproduces the hard Heaviside exactly: ``force > 0`` and
  ``raw > 0`` describe the same set because ``force = max(raw, 0)``.
  """

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: FinalSeedJob):
    world, row = wp.tid()
    projected = float(0.0)
    for dof in range(wp.static(dofs)):
      projected += (job.jacobian[world, row, dof]
                    * job.gradient.projected[world, dof])
    weight = job.state.rows.velocity_weight[world, row]
    free = job.state.rows.free_force[world, row]
    beta = job.inputs.contact_softness[world]
    if beta > 0.0:
      free = job.inputs.free_force[world, row]
      weight = job.inputs.velocity_weight[world, row]
    raw = free - weight * job.state.response[world, row]
    slope = wp.where(raw > 0.0, 1.0, 0.0)
    softness = float(0.0)
    if beta > 0.0:
      gate = wp.where(
          row < job.params.row_count and job.inputs.active[world, row] != 0,
          1.0, 0.0)
      sigmoid = stable_sigmoid(beta * raw)
      slope = gate * sigmoid
      softness = gate * projected * softplus_beta_gradient(raw, beta)
    force_gradient = slope * projected
    job.gradient.force[world, row] = force_gradient
    job.gradient.response[world, row] = -weight * force_gradient
    job.softness_gradient[world, row] = softness
    if beta > 0.0:
      # Final force depends directly on base F,D. Only the response seed
      # traverses the Newton iterations; do not linearize these seeds again.
      job.base_free_gradient[world, row] = force_gradient
      job.base_weight_gradient[world, row] = (
          -job.state.response[world, row] * force_gradient)
      job.gradient.force[world, row] = 0.0

  return kernel


@cache
def _solution_seed(rows: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: SolveBackwardJob):
    world, dof = wp.tid()
    value = float(0.0)
    for row in range(wp.static(rows)):
      value += (job.jacobian[world, row, dof]
                * job.gradient.response[world, row])
    job.gradient.solution[world, dof] = value

  return kernel


@cache
def _solve_adjoint(dofs: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: LinearSolveState, gradient: LinearSolveGradient):
    world = wp.tid()
    size = wp.static(dofs)
    factor = wp.tile_load(
        state.factor[world], shape=(size, size), bounds_check=False)
    source = wp.tile_load(
        gradient.solution[world], shape=size, bounds_check=False)
    solution = wp.tile_cholesky_solve(factor, source)
    wp.tile_store(gradient.adjoint_solution[world], solution,
                  bounds_check=False)

  return kernel


@cache
def _row_gradients(dofs: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: SolveBackwardJob):
    world, row = wp.tid()
    adjoint_response = float(0.0)
    for dof in range(wp.static(dofs)):
      adjoint_response += (job.jacobian[world, row, dof]
                           * job.gradient.adjoint_solution[world, dof])
    response = job.state.response[world, row]
    force_gradient = job.gradient.force[world, row]
    job.gradient.adjoint_response[world, row] = adjoint_response
    job.gradient.free_force[world, row] = force_gradient + adjoint_response
    job.gradient.velocity_weight[world, row] = (
        -response * (force_gradient + adjoint_response))

  return kernel


@wp.kernel(enable_backward=False)
def _jacobian_gradient(job: SolveBackwardJob):
  world, row, dof = wp.tid()
  force = job.force[world, row]
  projected = job.gradient.projected[world, dof]
  response_gradient = job.gradient.response[world, row]
  solution = job.state.solution[world, dof]
  free = job.state.rows.free_force[world, row]
  adjoint_value = job.gradient.adjoint_solution[world, dof]
  weight = job.state.rows.velocity_weight[world, row]
  response = job.state.response[world, row]
  adjoint_response = job.gradient.adjoint_response[world, row]
  job.gradient.jacobian[world, row, dof] = (
      force * projected
      + response_gradient * solution
      + free * adjoint_value
      - weight * (
          response * adjoint_value + adjoint_response * solution))


@wp.kernel(enable_backward=False)
def _mass_gradient(state: LinearSolveState,
                   gradient: LinearSolveGradient):
  world, row, column = wp.tid()
  gradient.mass[world, row, column] = (
      -gradient.adjoint_solution[world, row]
      * state.solution[world, column])


@wp.kernel(enable_backward=False)
def _linearize_backward(job: LinearizeBackwardJob):
  world, row = wp.tid()
  if (row >= job.params.row_count
      or job.inputs.active[world, row] == 0):
    job.base_free_gradient[world, row] = 0.0
    job.base_weight_gradient[world, row] = 0.0
    job.response_gradient[world, row] = 0.0
    job.softness_gradient[world, row] = 0.0
    return
  base_free = job.inputs.free_force[world, row]
  base_weight = job.inputs.velocity_weight[world, row]
  response = job.input_response[world, row]
  slope = job.state.rows.slope[world, row]
  beta = job.inputs.contact_softness[world]
  slope_gradient = float(0.0)
  beta_gradient = float(0.0)
  activation = base_weight * response
  if beta > 0.0:
    slope_gradient = beta * slope * (1.0 - slope)
    raw = base_free - activation
    slope_beta = raw * slope * (1.0 - slope)
    softplus_beta = softplus_beta_gradient(raw, beta)
    beta_gradient = (
        job.gradient.free_force[world, row]
        * (softplus_beta + activation * slope_beta)
        + job.gradient.velocity_weight[world, row]
        * base_weight * slope_beta)
  free_gradient = job.gradient.free_force[world, row]
  weight_gradient = job.gradient.velocity_weight[world, row]
  job.base_free_gradient[world, row] = (
      free_gradient * (slope + slope_gradient * activation)
      + weight_gradient * slope_gradient * base_weight)
  job.base_weight_gradient[world, row] = (
      -free_gradient * slope_gradient * response * activation
      + weight_gradient * (slope - slope_gradient * activation))
  job.response_gradient[world, row] = (
      -slope_gradient * base_weight
      * (free_gradient * activation + weight_gradient * base_weight))
  job.softness_gradient[world, row] = beta_gradient


@wp.kernel(enable_backward=False)
def _accumulate_rows(source: wp.array2d(dtype=float),
                     output: wp.array2d(dtype=float)):
  world, row = wp.tid()
  output[world, row] += source[world, row]


@wp.kernel(enable_backward=False)
def _accumulate_3d(source: wp.array3d(dtype=float),
                   output: wp.array3d(dtype=float)):
  world, row, dof = wp.tid()
  output[world, row, dof] += source[world, row, dof]


@wp.kernel(enable_backward=False)
def _reduce_softness(rows: wp.array2d(dtype=float),
                     output: wp.array(dtype=float), row_count: int):
  world = wp.tid()
  total = float(0.0)
  for row in range(row_count):
    total += rows[world, row]
  output[world] = total


def _gradient(worlds: int, rows: int, dofs: int, *,
              device) -> LinearSolveGradient:
  result = LinearSolveGradient()
  row_shape = (worlds, rows)
  dof_shape = (worlds, dofs)
  result.projected = wp.zeros(dof_shape, dtype=float, device=device)
  result.force = wp.zeros(row_shape, dtype=float, device=device)
  result.response = wp.zeros(row_shape, dtype=float, device=device)
  result.solution = wp.empty(dof_shape, dtype=float, device=device)
  result.adjoint_solution = wp.empty(dof_shape, dtype=float, device=device)
  result.adjoint_response = wp.empty(row_shape, dtype=float, device=device)
  result.jacobian = wp.empty((worlds, rows, dofs), dtype=float, device=device)
  result.free_force = wp.empty(row_shape, dtype=float, device=device)
  result.velocity_weight = wp.empty(row_shape, dtype=float, device=device)
  result.mass = wp.empty((worlds, dofs, dofs), dtype=float, device=device)
  return result


def _iteration_gradient(worlds: int, rows: int, dofs: int, *,
                        device) -> ContactIterationGradient:
  row_shape = (worlds, rows)
  return ContactIterationGradient(
      _gradient(worlds, rows, dofs, device=device),
      wp.empty(row_shape, dtype=float, device=device),
      wp.empty(row_shape, dtype=float, device=device),
      wp.empty(row_shape, dtype=float, device=device))


def allocate(worlds: int, rows: int, dofs: int, *, iterations: int,
             device) -> ContactAdjointWorkspace:
  if iterations < 0:
    raise ValueError("contact adjoint iterations must be non-negative")
  row_shape = (worlds, rows)
  iteration_count = max(iterations, 1)
  gradients = tuple(
      _iteration_gradient(worlds, rows, dofs, device=device)
      for _ in range(iteration_count))
  return ContactAdjointWorkspace(
      gradients,
      wp.empty(row_shape, dtype=float, device=device),
      wp.zeros(row_shape, dtype=float, device=device),
      wp.empty((worlds, rows, dofs), dtype=float, device=device),
      wp.empty(row_shape, dtype=float, device=device),
      wp.empty(row_shape, dtype=float, device=device),
      wp.empty((worlds, dofs, dofs), dtype=float, device=device),
      wp.empty(row_shape, dtype=float, device=device),
      wp.empty(worlds, dtype=float, device=device))


def _projection_force(forward, workspace, *, index: int, last: int):
  """Forward row force that carries the direct ``dL/dJ`` projection term.

  Only the last outer correction feeds the final projection, so every earlier
  correction contributes exactly zero there.  Returning a dedicated zero array
  keeps that intent explicit instead of relying on a cotangent buffer that
  happens to have just been cleared.
  """
  return forward.output.force if index == last else workspace.zero_rows


def _solve_backward(params, jacobian, *, state, force,
                    gradient: LinearSolveGradient) -> None:
  job = SolveBackwardJob()
  job.params = params
  job.jacobian = jacobian
  job.state = state
  job.force = force
  job.gradient = gradient
  worlds, rows, dofs = jacobian.shape
  wp.launch(_solution_seed(rows), dim=(worlds, dofs), inputs=[job])
  wp.launch_tiled(_solve_adjoint(dofs), dim=worlds,
                  inputs=[state, gradient], block_dim=TILE_BLOCK_SIZE)
  wp.launch(_row_gradients(dofs), dim=(worlds, rows), inputs=[job])
  wp.launch(_jacobian_gradient, dim=jacobian.shape, inputs=[job])
  wp.launch(_mass_gradient, dim=(worlds, dofs, dofs),
            inputs=[state, gradient])


def _seed_final(params, inputs, *, forward, constraint_gradient,
                gradient, softness_gradient, base_free_gradient,
                base_weight_gradient) -> None:
  job = FinalSeedJob()
  job.params = params
  job.inputs = inputs
  job.jacobian = inputs.jacobian
  job.state = forward.states[-1]
  job.force = forward.output.force
  job.constraint_gradient = constraint_gradient
  job.gradient = gradient
  job.softness_gradient = softness_gradient
  job.base_free_gradient = base_free_gradient
  job.base_weight_gradient = base_weight_gradient
  worlds, rows, dofs = inputs.jacobian.shape
  wp.launch(_projected_seed, dim=(worlds, dofs), inputs=[job])
  wp.launch(_final_row_seed(dofs), dim=(worlds, rows), inputs=[job])


def _linearize_back(params, inputs, *, input_response, state, gradient,
                    base_free, base_weight, response_gradient,
                    softness_gradient) -> None:
  job = LinearizeBackwardJob()
  job.params = params
  job.inputs = inputs
  job.input_response = input_response
  job.state = state
  job.gradient = gradient
  job.base_free_gradient = base_free
  job.base_weight_gradient = base_weight
  job.response_gradient = response_gradient
  job.softness_gradient = softness_gradient
  wp.launch(_linearize_backward, dim=base_free.shape, inputs=[job])


def _zero_totals(workspace: ContactAdjointWorkspace) -> None:
  workspace.jacobian.zero_()
  workspace.free_force.zero_()
  workspace.velocity_weight.zero_()
  workspace.mass.zero_()
  workspace.softness_rows.zero_()
  workspace.contact_softness.zero_()


def _accumulate_iteration(workspace: ContactAdjointWorkspace,
                          iteration: ContactIterationGradient) -> None:
  wp.launch(
      _accumulate_rows, dim=workspace.free_force.shape,
      inputs=[iteration.base_free, workspace.free_force])
  wp.launch(
      _accumulate_rows, dim=workspace.velocity_weight.shape,
      inputs=[iteration.base_weight, workspace.velocity_weight])
  wp.launch(
      _accumulate_3d, dim=workspace.jacobian.shape,
      inputs=[iteration.solve.jacobian, workspace.jacobian])
  wp.launch(
      _accumulate_3d, dim=workspace.mass.shape,
      inputs=[iteration.solve.mass, workspace.mass])
  wp.launch(
      _accumulate_rows, dim=workspace.softness_rows.shape,
      inputs=[iteration.softness, workspace.softness_rows])


def _reduce_total_softness(workspace: ContactAdjointWorkspace,
                           row_count: int) -> None:
  wp.launch(
      _reduce_softness, dim=workspace.contact_softness.shape,
      inputs=[workspace.softness_rows, workspace.contact_softness, row_count])


def _validate_iterations(params: ContactParameters,
                         forward: AnnealedContactWorkspace,
                         workspace: ContactAdjointWorkspace) -> None:
  if len(forward.states) != params.iterations:
    raise ValueError(
        "forward contact state count does not match parameters: "
        f"{len(forward.states)} != {params.iterations}")
  expected = max(params.iterations, 1)
  if len(workspace.iterations) != expected:
    raise ValueError(
        "contact adjoint state count does not match parameters: "
        f"{len(workspace.iterations)} != {expected}")


def _diagonal_backward(params, inputs: ContactBackwardInput,
                       workspace: ContactAdjointWorkspace) -> ContactGradient:
  source = inputs.forward_input
  forward = inputs.forward_workspace
  gradient = workspace.iterations[0].solve
  job = FinalSeedJob()
  job.params = params
  job.inputs = source
  job.jacobian = source.jacobian
  job.state = forward.final
  job.force = forward.output.force
  job.constraint_gradient = inputs.constraint_gradient
  job.gradient = gradient
  job.softness_gradient = workspace.softness_rows
  worlds, rows, dofs = source.jacobian.shape
  wp.launch_tiled(
      _diagonal_adjoint_kernel(dofs, rows), dim=worlds, inputs=[job],
      block_dim=TILE_BLOCK_SIZE, device=source.jacobian.device)
  workspace.contact_softness.zero_()
  return ContactGradient(
      gradient.jacobian, gradient.free_force,
      gradient.velocity_weight, gradient.mass,
      workspace.contact_softness)


def backward(params: ContactParameters, inputs: ContactBackwardInput,
             workspace: ContactAdjointWorkspace) -> ContactGradient:
  forward = inputs.forward_workspace
  _validate_iterations(params, forward, workspace)
  if params.row_count == 0:
    _zero_totals(workspace)
    return ContactGradient(
        workspace.jacobian, workspace.free_force,
        workspace.velocity_weight, workspace.mass,
        workspace.contact_softness)
  if params.iterations == 0:
    return _diagonal_backward(params, inputs, workspace)
  source = inputs.forward_input
  _zero_totals(workspace)
  workspace.initial_response.zero_()
  for iteration in workspace.iterations:
    iteration.solve.projected.zero_()
    iteration.solve.force.zero_()
    iteration.solve.response.zero_()
  final_iteration = workspace.iterations[-1]
  _seed_final(
      params, source, forward=forward,
      constraint_gradient=inputs.constraint_gradient,
      gradient=final_iteration.solve,
      softness_gradient=workspace.softness_rows,
      base_free_gradient=workspace.free_force,
      base_weight_gradient=workspace.velocity_weight)
  last_index = params.iterations - 1
  for index in range(last_index, -1, -1):
    iteration = workspace.iterations[index]
    state = forward.states[index]
    _solve_backward(
        params, source.jacobian, state=state,
        force=_projection_force(
            forward, workspace, index=index, last=last_index),
        gradient=iteration.solve)
    input_response = (forward.initial_response if index == 0
                      else forward.states[index - 1].response)
    response_gradient = (workspace.initial_response if index == 0
                         else workspace.iterations[index - 1].solve.response)
    _linearize_back(
        params, source, input_response=input_response,
        state=state, gradient=iteration.solve,
        base_free=iteration.base_free,
        base_weight=iteration.base_weight,
        response_gradient=response_gradient,
        softness_gradient=iteration.softness)
    _accumulate_iteration(workspace, iteration)
  _reduce_total_softness(workspace, params.row_count)
  return ContactGradient(
      workspace.jacobian, workspace.free_force, workspace.velocity_weight,
      workspace.mass, workspace.contact_softness)
