"""Static CSR/CSC matrix-free contact solve for high-DoF native steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from comfree_warp.comfree_core._src import support as production_support

from .contact_rows import StaticSparseLayout
from .contact_types import ContactOutput
from .contact_types import annealed_softplus
from .contact_types import stable_sigmoid


PCG_ITERATIONS = production_support.FULL_IMPLICIT_PCG_ITERATIONS
PCG_EPSILON = wp.constant(1.0e-20)
DEFAULT_BLOCK_DIM = production_support.FULL_IMPLICIT_SPARSE_BLOCK_DIM
VALID_BLOCK_DIMS = (32, 64, 128, 256)
PCG_STATUS_OK = 0
PCG_STATUS_INVALID_INPUT = 1
PCG_STATUS_BREAKDOWN = 2


@dataclass(frozen=True)
class SparsePcgSettings:
  block_dim: int = DEFAULT_BLOCK_DIM


@wp.struct
class SparseContactParameters:
  layout: StaticSparseLayout
  timestep: wp.array(dtype=float)
  row_count: int
  dof_count: int
  iterations: int


@wp.struct
class SparseContactInput:
  mass: wp.array3d(dtype=float)
  active: wp.array2d(dtype=int)
  weighted_jacobian: wp.array2d(dtype=float)
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  smooth_force: wp.array2d(dtype=float)
  contact_softness: wp.array(dtype=float)


@wp.struct
class SparseLinearizedRows:
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  slope: wp.array2d(dtype=float)


@wp.struct
class SparsePcgState:
  solution: wp.array2d(dtype=float)
  residual: wp.array2d(dtype=float)
  direction: wp.array2d(dtype=float)
  preconditioned: wp.array2d(dtype=float)
  matvec: wp.array2d(dtype=float)
  diagonal: wp.array2d(dtype=float)
  rhs: wp.array2d(dtype=float)
  row_response: wp.array2d(dtype=float)
  status: wp.array(dtype=int)
  residual_squared: wp.array(dtype=float)
  preconditioned_residual_energy: wp.array(dtype=float)
  iterations: wp.array(dtype=int)


@wp.struct
class SparseLinearSolveState:
  rows: SparseLinearizedRows
  pcg: SparsePcgState
  response: wp.array2d(dtype=float)


@wp.struct
class LinearizeJob:
  params: SparseContactParameters
  inputs: SparseContactInput
  response: wp.array2d(dtype=float)
  rows: SparseLinearizedRows


@wp.struct
class SparseSystemJob:
  params: SparseContactParameters
  mass: wp.array3d(dtype=float)
  values: wp.array2d(dtype=float)
  weights: wp.array2d(dtype=float)
  pcg: SparsePcgState
  failure_status: wp.array(dtype=int)
  failure_history: wp.array(dtype=int)


@wp.struct
class BuildRhsJob:
  params: SparseContactParameters
  values: wp.array2d(dtype=float)
  row_rhs: wp.array2d(dtype=float)
  output: wp.array2d(dtype=float)


@wp.struct
class ResponseJob:
  params: SparseContactParameters
  values: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  response: wp.array2d(dtype=float)


@wp.struct
class ProjectionJob:
  params: SparseContactParameters
  inputs: SparseContactInput
  state: SparseLinearSolveState
  output: ContactOutput
  failure_status: wp.array(dtype=int)


@dataclass(frozen=True)
class CompiledSparseContact:
  parameters: SparseContactParameters
  settings: SparsePcgSettings
  device: object


@dataclass(frozen=True)
class SparseAnnealedContactWorkspace:
  output: ContactOutput
  final: SparseLinearSolveState
  states: tuple[SparseLinearSolveState, ...]
  initial_response: wp.array
  failure_status: wp.array
  failure_history: wp.array


@wp.func
def _block_sum(value: float) -> float:
  values = wp.tile(value, preserve_type=True)
  reduced = wp.tile_reduce(wp.add, values)
  return wp.tile_extract(reduced, 0)


@wp.func_native("WP_TILE_SYNC();")
def _block_sync(): ...


@wp.func
def _safe_ratio(numerator: float, denominator: float) -> float:
  near_zero = (wp.abs(numerator) <= PCG_EPSILON
               or wp.abs(denominator) <= PCG_EPSILON)
  if near_zero:
    return 0.0
  return numerator / denominator


@wp.func
def _inputs_valid(job: SparseSystemJob, ids: wp.vec2i) -> bool:
  world = ids[0]
  invalid = float(0.0)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    if not wp.isfinite(job.pcg.rhs[world, dof]):
      invalid = 1.0
    for column in range(job.params.dof_count):
      if not wp.isfinite(job.mass[world, dof, column]):
        invalid = 1.0
  for row in range(ids[1], job.params.row_count, wp.block_dim()):
    weight = job.weights[world, row]
    if not wp.isfinite(weight) or weight < 0.0:
      invalid = 1.0
  count = job.params.layout.nonzero_count
  for position in range(ids[1], count, wp.block_dim()):
    if not wp.isfinite(job.values[world, position]):
      invalid = 1.0
  result = _block_sum(invalid)
  _block_sync()
  return result == 0.0


@wp.func
def _raw_residual_squared(job: SparseSystemJob, ids: wp.vec2i) -> float:
  world = ids[0]
  local = float(0.0)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    residual = job.pcg.residual[world, dof]
    local += residual * residual
  result = _block_sum(local)
  _block_sync()
  return result


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
  raw = base_free - base_weight * job.response[world, row]
  beta = job.inputs.contact_softness[world]
  slope = wp.where(raw > 0.0, 1.0, 0.0)
  linear_free = slope * base_free
  if beta > 0.0:
    slope = stable_sigmoid(beta * raw)
    linear_free = (
        slope * base_free + annealed_softplus(raw, beta) - slope * raw)
  job.rows.free_force[world, row] = linear_free
  job.rows.velocity_weight[world, row] = slope * base_weight
  job.rows.slope[world, row] = slope


@wp.kernel(enable_backward=False)
def _build_rhs(job: BuildRhsJob):
  world, dof = wp.tid()
  total = float(0.0)
  start = job.params.layout.column_offsets[dof]
  end = job.params.layout.column_offsets[dof + 1]
  for slot in range(start, end):
    position = job.params.layout.csc_positions[slot]
    row = job.params.layout.csc_rows[slot]
    total += job.values[world, position] * job.row_rhs[world, row]
  job.output[world, dof] = total


@wp.kernel(enable_backward=False)
def _copy_vector(source: wp.array2d(dtype=float),
                 output: wp.array2d(dtype=float)):
  world, dof = wp.tid()
  output[world, dof] = source[world, dof]


@wp.func
def _prepare_pcg(job: SparseSystemJob, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    diagonal = job.mass[world, dof, dof]
    start = job.params.layout.column_offsets[dof]
    end = job.params.layout.column_offsets[dof + 1]
    for slot in range(start, end):
      position = job.params.layout.csc_positions[slot]
      row = job.params.layout.csc_rows[slot]
      value = job.values[world, position]
      diagonal += job.weights[world, row] * value * value
    rhs = job.pcg.rhs[world, dof]
    job.pcg.diagonal[world, dof] = diagonal
    job.pcg.solution[world, dof] = 0.0
    job.pcg.residual[world, dof] = rhs
    job.pcg.preconditioned[world, dof] = rhs / diagonal
    job.pcg.direction[world, dof] = rhs / diagonal
    job.pcg.matvec[world, dof] = 0.0
  if ids[1] == 0:
    job.pcg.status[world] = PCG_STATUS_OK
    job.pcg.residual_squared[world] = 0.0
    job.pcg.preconditioned_residual_energy[world] = 0.0
    job.pcg.iterations[world] = 0


@wp.func
def _initial_measure(job: SparseSystemJob, ids: wp.vec2i) -> float:
  world = ids[0]
  local = float(0.0)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    local += (job.pcg.residual[world, dof]
              * job.pcg.preconditioned[world, dof])
  result = _block_sum(local)
  _block_sync()
  return result


@wp.func
def _csr_response(job: SparseSystemJob, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], job.params.row_count, wp.block_dim()):
    total = float(0.0)
    start = job.params.layout.row_offsets[row]
    end = start + job.params.layout.row_nonzero_count[row]
    for position in range(start, end):
      column = job.params.layout.column_indices[position]
      total += job.values[world, position] * job.pcg.direction[world, column]
    job.pcg.row_response[world, row] = total


@wp.func
def _csc_matvec(job: SparseSystemJob, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    total = float(0.0)
    for column in range(job.params.dof_count):
      total += (job.mass[world, dof, column]
                * job.pcg.direction[world, column])
    start = job.params.layout.column_offsets[dof]
    end = job.params.layout.column_offsets[dof + 1]
    for slot in range(start, end):
      position = job.params.layout.csc_positions[slot]
      row = job.params.layout.csc_rows[slot]
      value = job.values[world, position]
      total += (value * job.weights[world, row]
                * job.pcg.row_response[world, row])
    job.pcg.matvec[world, dof] = total


@wp.func
def _apply_operator(job: SparseSystemJob, ids: wp.vec2i):
  _csr_response(job, ids)
  _block_sync()
  _csc_matvec(job, ids)
  _block_sync()


@wp.func
def _mark_failure(job: SparseSystemJob, ids: wp.vec2i, code: int):
  world = ids[0]
  if ids[1] == 0:
    job.pcg.status[world] = code
    job.pcg.residual_squared[world] = wp.nan
    job.pcg.preconditioned_residual_energy[world] = wp.nan
    wp.atomic_max(job.failure_status, world, code)
    previous_history = wp.atomic_max(job.failure_history, world, code)
    if previous_history == PCG_STATUS_OK:
      wp.printf("native sparse PCG failure: world=%d code=%d\n", world, code)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    job.pcg.solution[world, dof] = wp.nan


@wp.func
def _pcg_step(job: SparseSystemJob, ids: wp.vec2i,
              measure: float) -> float:
  world = ids[0]
  _apply_operator(job, ids)
  local_denominator = float(0.0)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    local_denominator += (job.pcg.direction[world, dof]
                          * job.pcg.matvec[world, dof])
  denominator = _block_sum(local_denominator)
  _block_sync()
  invalid = (not wp.isfinite(measure) or not wp.isfinite(denominator)
             or measure < -PCG_EPSILON or denominator < -PCG_EPSILON)
  if invalid:
    _mark_failure(job, ids, PCG_STATUS_BREAKDOWN)
    _block_sync()
    return measure
  step = _safe_ratio(measure, denominator)
  local_next = float(0.0)
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    residual = job.pcg.residual[world, dof] - step * job.pcg.matvec[world, dof]
    preconditioned = residual / job.pcg.diagonal[world, dof]
    job.pcg.solution[world, dof] += step * job.pcg.direction[world, dof]
    job.pcg.residual[world, dof] = residual
    job.pcg.preconditioned[world, dof] = preconditioned
    local_next += residual * preconditioned
  result = _block_sum(local_next)
  _block_sync()
  if not wp.isfinite(result) or result < -PCG_EPSILON:
    _mark_failure(job, ids, PCG_STATUS_BREAKDOWN)
    _block_sync()
    return measure
  return result


@wp.func
def _update_direction(job: SparseSystemJob, ids: wp.vec2i,
                      measures: wp.vec2):
  world = ids[0]
  ratio = _safe_ratio(measures[1], measures[0])
  for dof in range(ids[1], job.params.dof_count, wp.block_dim()):
    job.pcg.direction[world, dof] = (
        job.pcg.preconditioned[world, dof]
        + ratio * job.pcg.direction[world, dof])


@wp.func
def _advance_pcg(job: SparseSystemJob, ids: wp.vec2i,
                 measure: float) -> float:
  next_measure = _pcg_step(job, ids, measure)
  if job.pcg.status[ids[0]] != PCG_STATUS_OK:
    return measure
  _update_direction(job, ids, wp.vec2(measure, next_measure))
  if ids[1] == 0:
    job.pcg.iterations[ids[0]] += 1
  return next_measure


@wp.kernel(enable_backward=False)
def _pcg_kernel(job: SparseSystemJob):
  world, lane = wp.tid()
  ids = wp.vec2i(world, lane)
  if lane == 0:
    job.pcg.iterations[world] = 0
  prior_failure = job.failure_status[world]
  if prior_failure != PCG_STATUS_OK:
    _mark_failure(job, ids, prior_failure)
    return
  if not _inputs_valid(job, ids):
    _mark_failure(job, ids, PCG_STATUS_INVALID_INPUT)
    return
  _prepare_pcg(job, ids)
  _block_sync()
  measure = _initial_measure(job, ids)
  for _ in range(PCG_ITERATIONS):
    if (job.pcg.status[world] == PCG_STATUS_OK
        and wp.abs(measure) > PCG_EPSILON):
      measure = _advance_pcg(job, ids, measure)
    _block_sync()
  residual_squared = _raw_residual_squared(job, ids)
  if lane == 0 and job.pcg.status[world] == PCG_STATUS_OK:
    job.pcg.residual_squared[world] = residual_squared
    job.pcg.preconditioned_residual_energy[world] = measure


@wp.kernel(enable_backward=False)
def _solution_response(job: ResponseJob):
  world, row = wp.tid()
  total = float(0.0)
  if row < job.params.row_count:
    start = job.params.layout.row_offsets[row]
    end = start + job.params.layout.row_nonzero_count[row]
    for position in range(start, end):
      column = job.params.layout.column_indices[position]
      total += job.values[world, position] * job.solution[world, column]
  job.response[world, row] = total


@wp.kernel(enable_backward=False)
def _project_rows(job: ProjectionJob):
  """Final non-negative projection; see the dense ``_project_kernel`` contract."""
  world, row = wp.tid()
  if row >= job.params.row_count:
    job.output.force[world, row] = 0.0
    return
  free = job.state.rows.free_force[world, row]
  weight = job.state.rows.velocity_weight[world, row]
  response = job.state.response[world, row]
  beta = job.inputs.contact_softness[world]
  if beta > 0.0:
    free = job.inputs.free_force[world, row]
    weight = job.inputs.velocity_weight[world, row]
  raw = free - weight * response
  if beta > 0.0:
    if job.inputs.active[world, row] == 0:
      job.output.force[world, row] = 0.0
      return
    job.output.force[world, row] = annealed_softplus(raw, beta)
    return
  job.output.force[world, row] = wp.max(raw, 0.0)


@wp.kernel(enable_backward=False)
def _project_dofs(job: ProjectionJob):
  world, dof = wp.tid()
  if job.failure_status[world] != PCG_STATUS_OK:
    job.output.constraint_force[world, dof] = wp.nan
    job.output.total_force[world, dof] = wp.nan
    job.output.activation_velocity[world, dof] = wp.nan
    return
  projected = float(0.0)
  start = job.params.layout.column_offsets[dof]
  end = job.params.layout.column_offsets[dof + 1]
  for slot in range(start, end):
    position = job.params.layout.csc_positions[slot]
    row = job.params.layout.csc_rows[slot]
    projected += (job.inputs.weighted_jacobian[world, position]
                  * job.output.force[world, row])
  timestep = job.params.timestep[world % job.params.timestep.shape[0]]
  job.output.constraint_force[world, dof] = projected
  job.output.total_force[world, dof] = (
      job.inputs.smooth_force[world, dof] + projected)
  job.output.activation_velocity[world, dof] = (
      job.inputs.qvel[world, dof]
      + timestep * (job.inputs.qacc_smooth[world, dof]
                    + job.state.pcg.solution[world, dof]))


@wp.kernel(enable_backward=False)
def _no_contact_output(params: SparseContactParameters,
                       inputs: SparseContactInput,
                       output: ContactOutput):
  world, dof = wp.tid()
  timestep = params.timestep[world % params.timestep.shape[0]]
  output.constraint_force[world, dof] = 0.0
  output.projected_solution[world, dof] = 0.0
  output.total_force[world, dof] = inputs.smooth_force[world, dof]
  output.activation_velocity[world, dof] = (
      inputs.qvel[world, dof] + timestep * inputs.qacc_smooth[world, dof])


def settings_from_environment() -> SparsePcgSettings:
  settings = SparsePcgSettings()
  validate_settings(settings)
  return settings


def validate_settings(settings: SparsePcgSettings) -> None:
  if settings.block_dim not in VALID_BLOCK_DIMS:
    raise ValueError(f"native sparse block_dim must be one of {VALID_BLOCK_DIMS}")


def compile_sparse_contact(source, device, *, iterations: int,
                           settings: SparsePcgSettings | None = None
                           ) -> CompiledSparseContact:
  if iterations <= 0:
    raise ValueError("sparse coupled contact requires a positive L")
  selected = settings_from_environment() if settings is None else settings
  validate_settings(selected)
  params = SparseContactParameters()
  params.layout = source.layout
  params.timestep = source.timestep
  params.row_count = source.row_count
  params.dof_count = source.dof_count
  params.iterations = iterations
  return CompiledSparseContact(params, selected, device)


def _rows(worlds: int, count: int, device) -> SparseLinearizedRows:
  result = SparseLinearizedRows()
  for name in ("free_force", "velocity_weight", "slope"):
    setattr(result, name, wp.empty(
        (worlds, count), dtype=float, device=device))
  return result


def allocate_pcg(worlds: int, rows: int, dofs: int, *, device) -> SparsePcgState:
  result = SparsePcgState()
  for name in (
      "solution", "residual", "direction", "preconditioned", "matvec",
      "diagonal", "rhs"):
    setattr(result, name, wp.empty(
        (worlds, dofs), dtype=float, device=device))
  result.row_response = wp.empty((worlds, rows), dtype=float, device=device)
  result.status = wp.zeros(worlds, dtype=int, device=device)
  result.residual_squared = wp.zeros(worlds, dtype=float, device=device)
  result.preconditioned_residual_energy = wp.zeros(
      worlds, dtype=float, device=device)
  result.iterations = wp.zeros(worlds, dtype=int, device=device)
  return result


def _state(worlds: int, rows: int, dofs: int, *,
           device) -> SparseLinearSolveState:
  result = SparseLinearSolveState()
  result.rows = _rows(worlds, rows, device)
  result.pcg = allocate_pcg(worlds, rows, dofs, device=device)
  result.response = wp.zeros((worlds, rows), dtype=float, device=device)
  return result


def _output(worlds: int, rows: int, dofs: int, *, device,
            final: SparseLinearSolveState) -> ContactOutput:
  output = ContactOutput()
  output.active_velocity_weight = final.rows.velocity_weight
  output.projected_solution = final.pcg.solution
  output.force = wp.empty((worlds, rows), dtype=float, device=device)
  output.constraint_force = wp.empty((worlds, dofs), dtype=float, device=device)
  output.total_force = wp.empty((worlds, dofs), dtype=float, device=device)
  output.activation_velocity = wp.empty((worlds, dofs), dtype=float, device=device)
  return output


def _iteration_states(worlds: int, rows: int, dofs: int, *,
                      iterations: int, device) -> tuple:
  states = tuple(
      _state(worlds, rows, dofs, device=device)
      for _ in range(iterations))
  return states[-1], states


def allocate(compiled: CompiledSparseContact, worlds: int
             ) -> SparseAnnealedContactWorkspace:
  params = compiled.parameters
  if params.iterations <= 0:
    raise ValueError("sparse coupled contact requires a positive L")
  rows, dofs = params.layout.row_capacity, params.dof_count
  final, states = _iteration_states(
      worlds, rows, dofs, iterations=params.iterations,
      device=compiled.device)
  failure = wp.zeros(worlds, dtype=int, device=compiled.device)
  history = wp.zeros(worlds, dtype=int, device=compiled.device)
  return SparseAnnealedContactWorkspace(
      _output(worlds, rows, dofs, device=compiled.device, final=final),
      final, states,
      wp.zeros((worlds, rows), dtype=float, device=compiled.device),
      failure, history)


def _linearize_state(compiled: CompiledSparseContact,
                     inputs: SparseContactInput, *, response,
                     state: SparseLinearSolveState) -> None:
  job = LinearizeJob()
  job.params = compiled.parameters
  job.inputs = inputs
  job.response = response
  job.rows = state.rows
  wp.launch(
      _linearize, dim=state.rows.free_force.shape, inputs=[job],
      device=compiled.device)


def solve_vector(compiled: CompiledSparseContact, values, weights, *, mass, rhs,
                 pcg: SparsePcgState, failure_status,
                 failure_history) -> None:
  if rhs.ptr != pcg.rhs.ptr:
    wp.launch(
        _copy_vector, dim=rhs.shape, inputs=[rhs], outputs=[pcg.rhs],
        device=compiled.device)
  job = SparseSystemJob()
  job.params = compiled.parameters
  job.mass = mass
  job.values = values
  job.weights = weights
  job.pcg = pcg
  job.failure_status = failure_status
  job.failure_history = failure_history
  wp.launch_tiled(
      _pcg_kernel, dim=[values.shape[0]], inputs=[job],
      block_dim=compiled.settings.block_dim,
      device=compiled.device)


def _solve_state(compiled: CompiledSparseContact,
                 inputs: SparseContactInput, state: SparseLinearSolveState,
                 *, failure_status, failure_history) -> None:
  params = compiled.parameters
  rhs_job = BuildRhsJob()
  rhs_job.params = params
  rhs_job.values = inputs.weighted_jacobian
  rhs_job.row_rhs = state.rows.free_force
  rhs_job.output = state.pcg.rhs
  wp.launch(
      _build_rhs, dim=(inputs.free_force.shape[0], params.dof_count),
      inputs=[rhs_job], device=compiled.device)
  solve_vector(
      compiled, inputs.weighted_jacobian, state.rows.velocity_weight,
      mass=inputs.mass, rhs=state.pcg.rhs, pcg=state.pcg,
      failure_status=failure_status,
      failure_history=failure_history)
  response_job = ResponseJob()
  response_job.params = params
  response_job.values = inputs.weighted_jacobian
  response_job.solution = state.pcg.solution
  response_job.response = state.response
  wp.launch(
      _solution_response,
      dim=(inputs.free_force.shape[0], params.layout.row_capacity),
      inputs=[response_job], device=compiled.device)


def _iteration(compiled: CompiledSparseContact, inputs: SparseContactInput, *,
               response, state: SparseLinearSolveState,
               failure_status, failure_history) -> None:
  _linearize_state(compiled, inputs, response=response, state=state)
  _solve_state(
      compiled, inputs, state, failure_status=failure_status,
      failure_history=failure_history)


def _project(compiled: CompiledSparseContact, inputs: SparseContactInput, *,
             state: SparseLinearSolveState,
             workspace: SparseAnnealedContactWorkspace) -> None:
  job = ProjectionJob()
  job.params = compiled.parameters
  job.inputs = inputs
  job.state = state
  job.output = workspace.output
  job.failure_status = workspace.failure_status
  wp.launch(
      _project_rows, dim=state.rows.free_force.shape, inputs=[job],
      device=compiled.device)
  wp.launch(
      _project_dofs,
      dim=(inputs.free_force.shape[0], compiled.parameters.dof_count),
      inputs=[job], device=compiled.device)


def solve(compiled: CompiledSparseContact, inputs: SparseContactInput,
          workspace: SparseAnnealedContactWorkspace) -> wp.array:
  if len(workspace.states) != compiled.parameters.iterations:
    raise ValueError(
        "sparse contact workspace iteration count does not match parameters: "
        f"{len(workspace.states)} != {compiled.parameters.iterations}")
  _clear_current_status(workspace)
  if compiled.parameters.row_count == 0:
    _clear_no_contact_rows(workspace)
    wp.launch(
        _no_contact_output,
        dim=(inputs.qvel.shape[0], compiled.parameters.dof_count),
        inputs=[compiled.parameters, inputs, workspace.output],
        device=compiled.device)
    return workspace.output.constraint_force
  workspace.initial_response.zero_()
  response = workspace.initial_response
  for state in workspace.states:
    _iteration(
        compiled, inputs, response=response, state=state,
        failure_status=workspace.failure_status,
        failure_history=workspace.failure_history)
    response = state.response
  _project(compiled, inputs, state=workspace.final, workspace=workspace)
  return workspace.output.constraint_force


def _clear_no_contact_rows(workspace: SparseAnnealedContactWorkspace) -> None:
  workspace.output.force.zero_()
  workspace.final.rows.free_force.zero_()
  workspace.final.rows.velocity_weight.zero_()
  workspace.final.rows.slope.zero_()
  workspace.final.response.zero_()


def _clear_current_status(workspace: SparseAnnealedContactWorkspace) -> None:
  workspace.failure_status.zero_()
  for state in workspace.states:
    state.pcg.status.zero_()


def clear_status(workspace: SparseAnnealedContactWorkspace) -> None:
  _clear_current_status(workspace)
  workspace.failure_history.zero_()


def check_status(workspace: SparseAnnealedContactWorkspace) -> None:
  status = workspace.failure_history.numpy()
  invalid = np.flatnonzero(status == PCG_STATUS_INVALID_INPUT)
  breakdown = np.flatnonzero(status == PCG_STATUS_BREAKDOWN)
  if invalid.size:
    worlds = ", ".join(str(int(world)) for world in invalid)
    raise RuntimeError(f"native sparse PCG received invalid input in world(s): {worlds}")
  if breakdown.size:
    worlds = ", ".join(str(int(world)) for world in breakdown)
    raise RuntimeError(f"native sparse PCG broke down in world(s): {worlds}")
