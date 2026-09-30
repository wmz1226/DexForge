"""Analytic implicit adjoint of the static sparse hard/soft contact solve."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .contact_types import softplus_beta_gradient
from .contact_types import stable_sigmoid
from .sparse_annealed_contact import CompiledSparseContact
from .sparse_annealed_contact import SparseAnnealedContactWorkspace
from .sparse_annealed_contact import SparseContactInput
from .sparse_annealed_contact import SparseContactParameters
from .sparse_annealed_contact import SparseLinearSolveState
from .sparse_annealed_contact import SparsePcgState
from .sparse_annealed_contact import allocate_pcg
from .sparse_annealed_contact import solve_vector


@wp.struct
class SparseLinearSolveGradient:
  projected: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  response: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  adjoint: SparsePcgState
  adjoint_response: wp.array2d(dtype=float)
  jacobian: wp.array2d(dtype=float)
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  mass: wp.array3d(dtype=float)


@wp.struct
class FinalSeedJob:
  params: SparseContactParameters
  inputs: SparseContactInput
  state: SparseLinearSolveState
  force: wp.array2d(dtype=float)
  constraint_gradient: wp.array2d(dtype=float)
  gradient: SparseLinearSolveGradient
  softness_gradient: wp.array2d(dtype=float)
  base_free_gradient: wp.array2d(dtype=float)
  base_weight_gradient: wp.array2d(dtype=float)


@wp.struct
class SolveBackwardJob:
  params: SparseContactParameters
  values: wp.array2d(dtype=float)
  state: SparseLinearSolveState
  force: wp.array2d(dtype=float)
  gradient: SparseLinearSolveGradient


@wp.struct
class LinearizeBackwardJob:
  params: SparseContactParameters
  inputs: SparseContactInput
  input_response: wp.array2d(dtype=float)
  state: SparseLinearSolveState
  gradient: SparseLinearSolveGradient
  base_free_gradient: wp.array2d(dtype=float)
  base_weight_gradient: wp.array2d(dtype=float)
  response_gradient: wp.array2d(dtype=float)
  softness_gradient: wp.array2d(dtype=float)


@dataclass(frozen=True)
class SparseContactBackwardInput:
  forward_input: SparseContactInput
  forward_workspace: SparseAnnealedContactWorkspace
  constraint_gradient: wp.array


@dataclass(frozen=True)
class SparseContactGradient:
  jacobian: wp.array
  free_force: wp.array
  velocity_weight: wp.array
  mass: wp.array
  contact_softness: wp.array


@dataclass(frozen=True)
class SparseContactIterationGradient:
  solve: SparseLinearSolveGradient
  base_free: wp.array
  base_weight: wp.array
  softness: wp.array


@dataclass(frozen=True)
class SparseContactAdjointWorkspace:
  iterations: tuple[SparseContactIterationGradient, ...]
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


@wp.kernel(enable_backward=False)
def _final_row_seed(job: FinalSeedJob):
  world, row = wp.tid()
  params = job.params
  projected = float(0.0)
  start = params.layout.row_offsets[row]
  end = start + params.layout.row_nonzero_count[row]
  for position in range(start, end):
    column = params.layout.column_indices[position]
    projected += (job.inputs.weighted_jacobian[world, position]
                  * job.gradient.projected[world, column])
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
        row < params.row_count and job.inputs.active[world, row] != 0,
        1.0, 0.0)
    sigmoid = stable_sigmoid(beta * raw)
    slope = gate * sigmoid
    softness = gate * projected * softplus_beta_gradient(raw, beta)
  force_gradient = slope * projected
  job.gradient.force[world, row] = force_gradient
  job.gradient.response[world, row] = -weight * force_gradient
  job.softness_gradient[world, row] = softness
  if beta > 0.0:
    job.base_free_gradient[world, row] = force_gradient
    job.base_weight_gradient[world, row] = (
        -job.state.response[world, row] * force_gradient)
    job.gradient.force[world, row] = 0.0


@wp.kernel(enable_backward=False)
def _solution_seed(job: SolveBackwardJob):
  world, dof = wp.tid()
  total = float(0.0)
  start = job.params.layout.column_offsets[dof]
  end = job.params.layout.column_offsets[dof + 1]
  for slot in range(start, end):
    position = job.params.layout.csc_positions[slot]
    row = job.params.layout.csc_rows[slot]
    total += job.values[world, position] * job.gradient.response[world, row]
  job.gradient.solution[world, dof] = total


@wp.kernel(enable_backward=False)
def _row_gradients(job: SolveBackwardJob):
  world, row = wp.tid()
  adjoint_response = float(0.0)
  start = job.params.layout.row_offsets[row]
  end = start + job.params.layout.row_nonzero_count[row]
  for position in range(start, end):
    column = job.params.layout.column_indices[position]
    adjoint_response += (job.values[world, position]
                         * job.gradient.adjoint.solution[world, column])
  response = job.state.response[world, row]
  force_gradient = job.gradient.force[world, row]
  job.gradient.adjoint_response[world, row] = adjoint_response
  job.gradient.free_force[world, row] = force_gradient + adjoint_response
  job.gradient.velocity_weight[world, row] = (
      -response * (force_gradient + adjoint_response))


@wp.kernel(enable_backward=False)
def _jacobian_gradient(job: SolveBackwardJob):
  world, position = wp.tid()
  row = job.params.layout.entry_rows[position]
  dof = job.params.layout.column_indices[position]
  force = job.force[world, row]
  projected = job.gradient.projected[world, dof]
  response_gradient = job.gradient.response[world, row]
  solution = job.state.pcg.solution[world, dof]
  free = job.state.rows.free_force[world, row]
  adjoint = job.gradient.adjoint.solution[world, dof]
  weight = job.state.rows.velocity_weight[world, row]
  response = job.state.response[world, row]
  adjoint_response = job.gradient.adjoint_response[world, row]
  job.gradient.jacobian[world, position] = (
      force * projected + response_gradient * solution + free * adjoint
      - weight * (
          response * adjoint + adjoint_response * solution))


@wp.kernel(enable_backward=False)
def _mass_gradient(state: SparseLinearSolveState,
                   gradient: SparseLinearSolveGradient):
  world, row, column = wp.tid()
  gradient.mass[world, row, column] = (
      -gradient.adjoint.solution[world, row]
      * state.pcg.solution[world, column])


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
def _accumulate_arrays(source: wp.array2d(dtype=float),
                       output: wp.array2d(dtype=float)):
  first, second = wp.tid()
  output[first, second] += source[first, second]


@wp.kernel(enable_backward=False)
def _accumulate_matrices(source: wp.array3d(dtype=float),
                         output: wp.array3d(dtype=float)):
  world, row, column = wp.tid()
  output[world, row, column] += source[world, row, column]


@wp.kernel(enable_backward=False)
def _reduce_softness(rows: wp.array2d(dtype=float),
                     output: wp.array(dtype=float), row_count: int):
  world = wp.tid()
  total = float(0.0)
  for row in range(row_count):
    total += rows[world, row]
  output[world] = total


def _gradient(compiled: CompiledSparseContact, worlds: int
              ) -> SparseLinearSolveGradient:
  params = compiled.parameters
  rows, dofs = params.layout.row_capacity, params.dof_count
  result = SparseLinearSolveGradient()
  result.projected = wp.zeros((worlds, dofs), dtype=float, device=compiled.device)
  result.force = wp.zeros((worlds, rows), dtype=float, device=compiled.device)
  result.response = wp.zeros((worlds, rows), dtype=float, device=compiled.device)
  result.solution = wp.empty((worlds, dofs), dtype=float, device=compiled.device)
  result.adjoint = allocate_pcg(
      worlds, rows, dofs, device=compiled.device)
  result.adjoint_response = wp.empty(
      (worlds, rows), dtype=float, device=compiled.device)
  result.jacobian = wp.empty(
      (worlds, params.layout.nonzero_count), dtype=float, device=compiled.device)
  result.free_force = wp.empty((worlds, rows), dtype=float, device=compiled.device)
  result.velocity_weight = wp.empty(
      (worlds, rows), dtype=float, device=compiled.device)
  result.mass = wp.empty(
      (worlds, dofs, dofs), dtype=float, device=compiled.device)
  return result


def _iteration_gradient(compiled: CompiledSparseContact, worlds: int
                        ) -> SparseContactIterationGradient:
  params = compiled.parameters
  row_shape = (worlds, params.layout.row_capacity)
  return SparseContactIterationGradient(
      _gradient(compiled, worlds),
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.empty(row_shape, dtype=float, device=compiled.device))


def allocate(compiled: CompiledSparseContact, worlds: int
             ) -> SparseContactAdjointWorkspace:
  params = compiled.parameters
  row_shape = (worlds, params.layout.row_capacity)
  value_shape = (worlds, params.layout.nonzero_count)
  gradients = tuple(
      _iteration_gradient(compiled, worlds)
      for _ in range(params.iterations))
  return SparseContactAdjointWorkspace(
      gradients,
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.zeros(row_shape, dtype=float, device=compiled.device),
      wp.empty(value_shape, dtype=float, device=compiled.device),
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.empty((worlds, params.dof_count, params.dof_count),
               dtype=float, device=compiled.device),
      wp.empty(row_shape, dtype=float, device=compiled.device),
      wp.empty(worlds, dtype=float, device=compiled.device))


def _projection_force(forward, workspace, *, index: int, last: int):
  """Forward row force that carries the direct ``dL/dJ`` projection term.

  Only the last outer correction feeds the final projection, so every earlier
  correction contributes exactly zero there.  Returning a dedicated zero array
  keeps that intent explicit instead of relying on a cotangent buffer that
  happens to have just been cleared.
  """
  return forward.output.force if index == last else workspace.zero_rows


def _solve_job(compiled: CompiledSparseContact, inputs: SparseContactInput, *,
               state: SparseLinearSolveState, force,
               gradient: SparseLinearSolveGradient) -> SolveBackwardJob:
  job = SolveBackwardJob()
  job.params = compiled.parameters
  job.values = inputs.weighted_jacobian
  job.state = state
  job.force = force
  job.gradient = gradient
  return job


def _solve_backward(compiled: CompiledSparseContact,
                    inputs: SparseContactInput, *, state, force, gradient,
                    failure_status, failure_history) -> None:
  job = _solve_job(
      compiled, inputs, state=state, force=force, gradient=gradient)
  worlds = inputs.weighted_jacobian.shape[0]
  params = compiled.parameters
  wp.launch(
      _solution_seed, dim=(worlds, params.dof_count), inputs=[job],
      device=compiled.device)
  solve_vector(
      compiled, inputs.weighted_jacobian, state.rows.velocity_weight,
      mass=inputs.mass, rhs=gradient.solution, pcg=gradient.adjoint,
      failure_status=failure_status, failure_history=failure_history)
  wp.launch(
      _row_gradients, dim=(worlds, params.layout.row_capacity), inputs=[job],
      device=compiled.device)
  wp.launch(
      _jacobian_gradient,
      dim=(worlds, params.layout.nonzero_count), inputs=[job],
      device=compiled.device)
  wp.launch(
      _mass_gradient, dim=(worlds, params.dof_count, params.dof_count),
      inputs=[state, gradient], device=compiled.device)


def _seed_final(compiled: CompiledSparseContact,
                inputs: SparseContactBackwardInput,
                gradient: SparseLinearSolveGradient,
                softness_gradient: wp.array, base_free_gradient: wp.array,
                base_weight_gradient: wp.array) -> None:
  source, forward = inputs.forward_input, inputs.forward_workspace
  job = FinalSeedJob()
  job.params = compiled.parameters
  job.inputs = source
  job.state = forward.states[-1]
  job.force = forward.output.force
  job.constraint_gradient = inputs.constraint_gradient
  job.gradient = gradient
  job.softness_gradient = softness_gradient
  job.base_free_gradient = base_free_gradient
  job.base_weight_gradient = base_weight_gradient
  worlds = source.weighted_jacobian.shape[0]
  wp.launch(
      _projected_seed, dim=(worlds, compiled.parameters.dof_count), inputs=[job],
      device=compiled.device)
  wp.launch(
      _final_row_seed,
      dim=(worlds, compiled.parameters.layout.row_capacity), inputs=[job],
      device=compiled.device)


def _linearize_back(compiled: CompiledSparseContact,
                    inputs: SparseContactInput, *, input_response, state,
                    gradient, base_free, base_weight,
                    response_gradient, softness_gradient) -> None:
  job = LinearizeBackwardJob()
  job.params = compiled.parameters
  job.inputs = inputs
  job.input_response = input_response
  job.state = state
  job.gradient = gradient
  job.base_free_gradient = base_free
  job.base_weight_gradient = base_weight
  job.response_gradient = response_gradient
  job.softness_gradient = softness_gradient
  wp.launch(
      _linearize_backward, dim=base_free.shape, inputs=[job],
      device=compiled.device)


def _zero_totals(workspace: SparseContactAdjointWorkspace) -> None:
  workspace.jacobian.zero_()
  workspace.free_force.zero_()
  workspace.velocity_weight.zero_()
  workspace.mass.zero_()
  workspace.softness_rows.zero_()
  workspace.contact_softness.zero_()


def _accumulate_iteration(
    compiled: CompiledSparseContact,
    workspace: SparseContactAdjointWorkspace,
    iteration: SparseContactIterationGradient) -> None:
  wp.launch(
      _accumulate_arrays, dim=workspace.free_force.shape,
      inputs=[iteration.base_free, workspace.free_force],
      device=compiled.device)
  wp.launch(
      _accumulate_arrays, dim=workspace.velocity_weight.shape,
      inputs=[iteration.base_weight, workspace.velocity_weight],
      device=compiled.device)
  wp.launch(
      _accumulate_arrays, dim=workspace.jacobian.shape,
      inputs=[iteration.solve.jacobian, workspace.jacobian],
      device=compiled.device)
  wp.launch(
      _accumulate_matrices, dim=workspace.mass.shape,
      inputs=[iteration.solve.mass, workspace.mass],
      device=compiled.device)
  wp.launch(
      _accumulate_arrays, dim=workspace.softness_rows.shape,
      inputs=[iteration.softness, workspace.softness_rows],
      device=compiled.device)


def _reduce_total_softness(compiled: CompiledSparseContact,
                           workspace: SparseContactAdjointWorkspace) -> None:
  wp.launch(
      _reduce_softness, dim=workspace.contact_softness.shape,
      inputs=[workspace.softness_rows, workspace.contact_softness,
              compiled.parameters.row_count],
      device=compiled.device)


def _validate_iterations(compiled: CompiledSparseContact,
                         forward: SparseAnnealedContactWorkspace,
                         workspace: SparseContactAdjointWorkspace) -> None:
  expected = compiled.parameters.iterations
  if len(forward.states) != expected:
    raise ValueError(
        "forward sparse contact state count does not match parameters: "
        f"{len(forward.states)} != {expected}")
  if len(workspace.iterations) != expected:
    raise ValueError(
        "sparse contact adjoint state count does not match parameters: "
        f"{len(workspace.iterations)} != {expected}")


def backward(compiled: CompiledSparseContact,
             inputs: SparseContactBackwardInput,
             workspace: SparseContactAdjointWorkspace) -> SparseContactGradient:
  source, forward = inputs.forward_input, inputs.forward_workspace
  _validate_iterations(compiled, forward, workspace)
  if compiled.parameters.row_count == 0:
    _zero_totals(workspace)
    return SparseContactGradient(
        workspace.jacobian, workspace.free_force, workspace.velocity_weight,
        workspace.mass, workspace.contact_softness)
  _zero_totals(workspace)
  workspace.initial_response.zero_()
  for iteration in workspace.iterations:
    iteration.solve.projected.zero_()
    iteration.solve.force.zero_()
    iteration.solve.response.zero_()
  _seed_final(compiled, inputs, workspace.iterations[-1].solve,
              workspace.softness_rows, workspace.free_force,
              workspace.velocity_weight)
  last_index = compiled.parameters.iterations - 1
  for index in range(last_index, -1, -1):
    iteration = workspace.iterations[index]
    state = forward.states[index]
    _solve_backward(
        compiled, source, state=state,
        force=_projection_force(
            forward, workspace, index=index, last=last_index),
        gradient=iteration.solve, failure_status=forward.failure_status,
        failure_history=forward.failure_history)
    input_response = (forward.initial_response if index == 0
                      else forward.states[index - 1].response)
    response_gradient = (workspace.initial_response if index == 0
                         else workspace.iterations[index - 1].solve.response)
    _linearize_back(
        compiled, source, input_response=input_response,
        state=state, gradient=iteration.solve,
        base_free=iteration.base_free, base_weight=iteration.base_weight,
        response_gradient=response_gradient,
        softness_gradient=iteration.softness)
    _accumulate_iteration(compiled, workspace, iteration)
  _reduce_total_softness(compiled, workspace)
  return SparseContactGradient(
      workspace.jacobian, workspace.free_force, workspace.velocity_weight,
      workspace.mass, workspace.contact_softness)
