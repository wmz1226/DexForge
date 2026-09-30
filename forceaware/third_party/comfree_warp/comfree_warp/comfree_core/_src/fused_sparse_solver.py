# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
"""Single-launch sparse full-implicit contact solver.

The CUDA path assigns one cooperative thread block to each world.  It applies
``A = M + J.T @ diag(D_active) @ J`` directly from CSR rows, so memory is
linear in the number of DoFs and Jacobian nonzeros.  The active-set and fixed
PCG iteration counts intentionally match the existing blocked solver.
"""

from __future__ import annotations

import dataclasses
import functools

import warp as wp

from .types import DUAL_CORRECTION


DEFAULT_COUPLED_ITERATIONS = 5
DEFAULT_BLOCK_DIM = 128
DEFAULT_PCG_ITERATIONS = 32
MAX_BLOCK_DIM = 1024
PCG_EPSILON = wp.constant(1.0e-20)
SQRT_INVWEIGHT_EPSILON = wp.constant(1.0e-12)

wp.set_module_options({"enable_backward": False})


@dataclasses.dataclass(frozen=True, kw_only=True)
class FusedSparseSolverConfig:
  """Compile-time iteration counts and CUDA launch configuration."""

  coupled_iterations: int = DEFAULT_COUPLED_ITERATIONS
  pcg_iterations: int = DEFAULT_PCG_ITERATIONS
  block_dim: int = DEFAULT_BLOCK_DIM
  early_stop: bool = True


@wp.struct
class FusedSparseSolverState:
  timestep: wp.array(dtype=float)
  qM: wp.array3d(dtype=float)
  qM_rowadr: wp.array(dtype=int)
  qM_col: wp.array(dtype=int)
  qM_madr: wp.array(dtype=int)
  dof_Madr: wp.array(dtype=int)
  qLD: wp.array3d(dtype=float)
  qLDiagInv: wp.array2d(dtype=float)
  M_rownnz: wp.array(dtype=int)
  M_rowadr: wp.array(dtype=int)
  M_colind: wp.array(dtype=int)
  rownnz: wp.array2d(dtype=int)
  rowadr: wp.array2d(dtype=int)
  colind: wp.array3d(dtype=int)
  jacobian: wp.array3d(dtype=float)
  weighted_jacobian: wp.array3d(dtype=float)
  nnz_count: wp.array(dtype=int)
  overflow: wp.array(dtype=int)
  free_force: wp.array2d(dtype=float)
  D: wp.array2d(dtype=float)
  imp: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  nefc: wp.array(dtype=int)
  qvel: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  qfrc_smooth: wp.array2d(dtype=float)
  qfrc_constraint: wp.array2d(dtype=float)
  qfrc_total: wp.array2d(dtype=float)
  qvel_predicted: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  diagonal: wp.array2d(dtype=float)
  rhs: wp.array2d(dtype=float)
  residual: wp.array2d(dtype=float)
  preconditioned: wp.array2d(dtype=float)
  direction: wp.array2d(dtype=float)
  matvec: wp.array2d(dtype=float)
  coupled_iterations_used: wp.array(dtype=float)
  pcg_iterations_used: wp.array(dtype=float)
  solver_niter: wp.array(dtype=int)
  dual_x: wp.array2d(dtype=float)
  dual_y: wp.array2d(dtype=float)
  dual_g: wp.array2d(dtype=float)
  nv: int
  vector_capacity: int
  row_limit: int
  row_capacity: int
  nnz_capacity: int


@wp.func_native("WP_TILE_SYNC();")
def _block_sync(): ...


@wp.func
def _block_sum(value: float) -> float:
  values = wp.tile(value, preserve_type=True)
  reduced = wp.tile_reduce(wp.add, values)
  return wp.tile_extract(reduced, 0)


@wp.func
def _safe_ratio(numerator: float, denominator: float) -> float:
  if wp.abs(numerator) <= PCG_EPSILON or wp.abs(denominator) <= PCG_EPSILON:
    return 0.0
  return numerator / denominator


@wp.func
def _clear_outputs(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  lane = ids[1]
  block_size = wp.block_dim()
  timestep = state.timestep[world % state.timestep.shape[0]]
  if lane == 0:
    state.coupled_iterations_used[world] = 0.0
    state.pcg_iterations_used[world] = 0.0
    state.solver_niter[world] = 0
  for row in range(lane, state.row_capacity, block_size):
    state.velocity_weight[world, row] = 0.0
    state.force[world, row] = 0.0
  for dof in range(lane, state.vector_capacity, block_size):
    state.solution[world, dof] = 0.0
  for dof in range(lane, state.nv, block_size):
    state.qfrc_constraint[world, dof] = 0.0
    state.qfrc_total[world, dof] = state.qfrc_smooth[world, dof]
    acceleration = state.qacc_smooth[world, dof]
    state.qvel_predicted[world, dof] = state.qvel[world, dof] + timestep * acceleration


@wp.func
def _reset_iteration_counters(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  lane = ids[1]
  if lane == 0:
    state.coupled_iterations_used[world] = 0.0
    state.pcg_iterations_used[world] = 0.0
    state.solver_niter[world] = 0
  timestep = state.timestep[world % state.timestep.shape[0]]
  for dof in range(lane, state.nv, wp.block_dim()):
    state.qfrc_constraint[world, dof] = 0.0
    state.qfrc_total[world, dof] = state.qfrc_smooth[world, dof]
    acceleration = state.qacc_smooth[world, dof]
    state.qvel_predicted[world, dof] = (
        state.qvel[world, dof] + timestep * acceleration)


@wp.func
def _metadata_is_invalid(state: FusedSparseSolverState, world: int) -> bool:
  row_count = state.nefc[world]
  nnz_count = state.nnz_count[world]
  invalid_rows = row_count < 0 or row_count > state.row_limit or row_count > state.row_capacity
  invalid_nnz = nnz_count < 0 or nnz_count > state.nnz_capacity
  return invalid_rows or invalid_nnz


@wp.func
def _mark_invalid_metadata(state: FusedSparseSolverState, ids: wp.vec2i):
  if ids[1] != 0:
    return
  if _metadata_is_invalid(state, ids[0]):
    state.overflow[ids[0]] = 1


@wp.func
def _row_range_is_invalid(state: FusedSparseSolverState, world: int, row: int) -> bool:
  address = state.rowadr[world, row]
  count = state.rownnz[world, row]
  end = address + count
  outside_nnz = end > state.nnz_count[world] or end > state.nnz_capacity
  return address < 0 or count < 0 or outside_nnz


@wp.func
def _validate_columns(state: FusedSparseSolverState, world: int, row: int):
  address = state.rowadr[world, row]
  count = state.rownnz[world, row]
  for position in range(address, address + count):
    column = state.colind[world, 0, position]
    if column < 0 or column >= state.nv:
      wp.atomic_max(state.overflow, world, 1)


@wp.func
def _validate_row(state: FusedSparseSolverState, world: int, row: int):
  if _row_range_is_invalid(state, world, row):
    wp.atomic_max(state.overflow, world, 1)
    return
  _validate_columns(state, world, row)


@wp.func
def _validate_rows(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  lane = ids[1]
  _mark_invalid_metadata(state, ids)
  row_count = state.nefc[world]
  safe_rows = wp.max(0, wp.min(row_count, state.row_capacity))
  for row in range(lane, safe_rows, wp.block_dim()):
    _validate_row(state, world, row)


@wp.func
def _validate_dense_rows(state: FusedSparseSolverState, ids: wp.vec2i):
  if ids[1] != 0:
    return
  row_count = state.nefc[ids[0]]
  if row_count < 0 or row_count > state.row_limit:
    state.overflow[ids[0]] = 1


@wp.func
def _write_invalid(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  lane = ids[1]
  block_size = wp.block_dim()
  for row in range(lane, state.row_capacity, block_size):
    state.velocity_weight[world, row] = wp.nan
    state.force[world, row] = wp.nan
  for dof in range(lane, state.vector_capacity, block_size):
    state.solution[world, dof] = wp.nan
  for dof in range(lane, state.nv, block_size):
    state.qfrc_constraint[world, dof] = wp.nan
    state.qfrc_total[world, dof] = wp.nan
    state.qvel_predicted[world, dof] = wp.nan


@wp.func
def _row_dot(state: FusedSparseSolverState, world: int, row: int) -> float:
  response = float(0.0)
  address = state.rowadr[world, row]
  count = state.rownnz[world, row]
  for position in range(address, address + count):
    column = state.colind[world, 0, position]
    response += state.jacobian[world, 0, position] * state.solution[world, column]
  return response


@wp.func
def _row_dot_dense(
    state: FusedSparseSolverState, world: int, row: int) -> float:
  response = float(0.0)
  for dof in range(state.nv):
    response += state.jacobian[world, row, dof] * state.solution[world, dof]
  return response


@wp.func
def _update_active_rows(state: FusedSparseSolverState, ids: wp.vec2i) -> float:
  world = ids[0]
  local_changes = float(0.0)
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    response = _row_dot(state, world, row)
    full_weight = state.D[world, row]
    candidate = state.free_force[world, row] - full_weight * response
    active = float(0.0)
    if candidate > 0.0:
      active = 1.0
    updated_weight = active * full_weight
    if updated_weight != state.velocity_weight[world, row]:
      local_changes += 1.0
    state.velocity_weight[world, row] = updated_weight
  return _block_sum(local_changes)


@wp.func
def _update_active_rows_dense(
    state: FusedSparseSolverState, ids: wp.vec2i) -> float:
  world = ids[0]
  local_changes = float(0.0)
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    response = _row_dot_dense(state, world, row)
    full_weight = state.D[world, row]
    candidate = state.free_force[world, row] - full_weight * response
    updated_weight = wp.where(candidate > 0.0, full_weight, 0.0)
    if updated_weight != state.velocity_weight[world, row]:
      local_changes += 1.0
    state.velocity_weight[world, row] = updated_weight
  return _block_sum(local_changes)


@wp.func
def _zero_system(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    state.diagonal[world, dof] = state.qM[world, 0, state.dof_Madr[dof]]
    state.rhs[world, dof] = 0.0
    state.solution[world, dof] = 0.0
    state.residual[world, dof] = 0.0
    state.preconditioned[world, dof] = 0.0
    state.direction[world, dof] = 0.0
    state.matvec[world, dof] = 0.0


@wp.func
def _zero_system_dense(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    state.diagonal[world, dof] = state.qM[world, dof, dof]
    state.rhs[world, dof] = 0.0
    state.solution[world, dof] = 0.0
    state.residual[world, dof] = 0.0
    state.preconditioned[world, dof] = 0.0
    state.direction[world, dof] = 0.0
    state.matvec[world, dof] = 0.0


@wp.func
def _accumulate_system(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    row_weight = state.velocity_weight[world, row]
    active_free = float(0.0)
    if row_weight > 0.0:
      active_free = state.free_force[world, row]
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      value = state.jacobian[world, 0, position]
      wp.atomic_add(state.diagonal, world, column, row_weight * value * value)
      wp.atomic_add(state.rhs, world, column, active_free * value)


@wp.func
def _accumulate_system_dense(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    row_weight = state.velocity_weight[world, row]
    active_free = wp.where(
        row_weight > 0.0, state.free_force[world, row], 0.0)
    for dof in range(state.nv):
      value = state.jacobian[world, row, dof]
      wp.atomic_add(
          state.diagonal, world, dof, row_weight * value * value)
      wp.atomic_add(state.rhs, world, dof, active_free * value)


@wp.func
def _initialize_pcg(state: FusedSparseSolverState, ids: wp.vec2i) -> float:
  world = ids[0]
  local_rz = float(0.0)
  for dof in range(ids[1], state.nv, wp.block_dim()):
    residual = state.rhs[world, dof]
    preconditioned = residual / state.diagonal[world, dof]
    state.residual[world, dof] = residual
    state.preconditioned[world, dof] = preconditioned
    state.direction[world, dof] = preconditioned
    local_rz += residual * preconditioned
  total_rz = _block_sum(local_rz)
  _block_sync()
  return total_rz


@wp.func
def _apply_operator(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    total = float(0.0)
    for entry in range(state.qM_rowadr[dof], state.qM_rowadr[dof + 1]):
      column = state.qM_col[entry]
      address = state.qM_madr[entry]
      total += state.qM[world, 0, address] * state.direction[world, column]
    state.matvec[world, dof] = total
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    response = float(0.0)
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      value = state.jacobian[world, 0, position]
      response += value * state.direction[world, column]
    weighted = state.velocity_weight[world, row] * response
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      value = state.jacobian[world, 0, position]
      wp.atomic_add(state.matvec, world, column, value * weighted)
  _block_sync()


@wp.func
def _apply_operator_dense(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    total = float(0.0)
    for column in range(state.nv):
      total += state.qM[world, dof, column] * state.direction[world, column]
    state.matvec[world, dof] = total
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    response = float(0.0)
    for dof in range(state.nv):
      response += (
          state.jacobian[world, row, dof]
          * state.direction[world, dof])
    weighted = state.velocity_weight[world, row] * response
    for dof in range(state.nv):
      wp.atomic_add(
          state.matvec, world, dof,
          state.jacobian[world, row, dof] * weighted)
  _block_sync()


@wp.func
def _pcg_update(state: FusedSparseSolverState, ids: wp.vec2i, rz_old: float) -> float:
  world = ids[0]
  _apply_operator(state, ids)
  local_pap = float(0.0)
  for dof in range(ids[1], state.nv, wp.block_dim()):
    local_pap += state.direction[world, dof] * state.matvec[world, dof]
  alpha = _safe_ratio(rz_old, _block_sum(local_pap))
  local_rz = float(0.0)
  for dof in range(ids[1], state.nv, wp.block_dim()):
    solution = state.solution[world, dof] + alpha * state.direction[world, dof]
    residual = state.residual[world, dof] - alpha * state.matvec[world, dof]
    preconditioned = residual / state.diagonal[world, dof]
    state.solution[world, dof] = solution
    state.residual[world, dof] = residual
    state.preconditioned[world, dof] = preconditioned
    local_rz += residual * preconditioned
  rz_new = _block_sum(local_rz)
  _block_sync()
  return rz_new


@wp.func
def _pcg_update_dense(
    state: FusedSparseSolverState,
    ids: wp.vec2i,
    rz_old: float,
) -> float:
  world = ids[0]
  _apply_operator_dense(state, ids)
  local_pap = float(0.0)
  for dof in range(ids[1], state.nv, wp.block_dim()):
    local_pap += state.direction[world, dof] * state.matvec[world, dof]
  alpha = _safe_ratio(rz_old, _block_sum(local_pap))
  local_rz = float(0.0)
  for dof in range(ids[1], state.nv, wp.block_dim()):
    solution = state.solution[world, dof] + alpha * state.direction[world, dof]
    residual = state.residual[world, dof] - alpha * state.matvec[world, dof]
    preconditioned = residual / state.diagonal[world, dof]
    state.solution[world, dof] = solution
    state.residual[world, dof] = residual
    state.preconditioned[world, dof] = preconditioned
    local_rz += residual * preconditioned
  rz_new = _block_sum(local_rz)
  _block_sync()
  return rz_new


@wp.func
def _update_direction(state: FusedSparseSolverState, ids: wp.vec2i, rhos: wp.vec2):
  world = ids[0]
  beta = _safe_ratio(rhos[1], rhos[0])
  for dof in range(ids[1], state.nv, wp.block_dim()):
    previous = state.direction[world, dof]
    state.direction[world, dof] = state.preconditioned[world, dof] + beta * previous
  _block_sync()


@wp.func
def _project_row_forces(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    response = _row_dot(state, world, row)
    row_weight = state.D[world, row]
    force = state.free_force[world, row] - row_weight * response
    if state.velocity_weight[world, row] > 0.0:
      state.force[world, row] = wp.max(force, 0.0)
    else:
      state.force[world, row] = 0.0


@wp.func
def _project_row_forces_dense(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    response = _row_dot_dense(state, world, row)
    force = state.free_force[world, row] - state.D[world, row] * response
    active = state.velocity_weight[world, row] > 0.0
    state.force[world, row] = wp.where(active, wp.max(force, 0.0), 0.0)


@wp.func
def _scatter_projected_solution(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.vector_capacity, wp.block_dim()):
    state.solution[world, dof] = 0.0
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    force = state.force[world, row]
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      value = state.jacobian[world, 0, position]
      wp.atomic_add(state.solution, world, column, value * force)
  _block_sync()


@wp.func
def _scatter_projected_solution_dense(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.vector_capacity, wp.block_dim()):
    state.solution[world, dof] = 0.0
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    force = state.force[world, row]
    for dof in range(state.nv):
      wp.atomic_add(
          state.solution, world, dof,
          state.jacobian[world, row, dof] * force)
  _block_sync()


@wp.func
def _finalize_dofs(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    constraint_force = state.solution[world, dof]
    state.qfrc_constraint[world, dof] = constraint_force
    state.qfrc_total[world, dof] = state.qfrc_smooth[world, dof] + constraint_force


#   e = J^T λ − M x ;  y = M^-1 e ;  w = J y ;  g = D·w
#   δλ = −[ g − D·J·(M + J^T W J)^-1 J^T g ]


@wp.func
def _save_solution(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    state.dual_x[world, dof] = state.solution[world, dof]


@wp.func
def _dual_residual(state: FusedSparseSolverState, ids: wp.vec2i):
  """Write J^T lambda - M x to the residual buffer."""
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    total = float(0.0)
    for entry in range(state.qM_rowadr[dof], state.qM_rowadr[dof + 1]):
      total -= (state.qM[world, 0, state.qM_madr[entry]]
                * state.dual_x[world, state.qM_col[entry]])
    state.dual_y[world, dof] = total
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    force = state.force[world, row]
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      value = state.jacobian[world, 0, position]
      wp.atomic_add(state.dual_y, world, column, value * force)


@wp.func
def _sparse_mass_solve(state: FusedSparseSolverState, ids: wp.vec2i):
  """Apply the inverse mass matrix using the sparse LDL factors."""
  if ids[1] != 0:
    return
  world = ids[0]
  for offset in range(state.nv):
    dof = state.nv - offset - 1
    address = state.M_rowadr[dof]
    count = state.M_rownnz[dof]
    value = state.dual_y[world, dof]
    for factor in range(address, address + count - 1):
      ancestor = state.M_colind[factor]
      state.dual_y[world, ancestor] = (state.dual_y[world, ancestor]
                                       - state.qLD[world, 0, factor] * value)
  for dof in range(state.nv):
    state.dual_y[world, dof] = (state.dual_y[world, dof]
                                * state.qLDiagInv[world, dof])
  for dof in range(state.nv):
    address = state.M_rowadr[dof]
    count = state.M_rownnz[dof]
    value = state.dual_y[world, dof]
    for factor in range(address, address + count - 1):
      ancestor = state.M_colind[factor]
      value -= state.qLD[world, 0, factor] * state.dual_y[world, ancestor]
    state.dual_y[world, dof] = value


@wp.func
def _dual_rows(state: FusedSparseSolverState, ids: wp.vec2i):
  """Compute D (J y) and the right-hand side of the correction solve."""
  world = ids[0]
  for dof in range(ids[1], state.vector_capacity, wp.block_dim()):
    state.rhs[world, dof] = 0.0
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    total = float(0.0)
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      total += (state.jacobian[world, 0, position]
                * state.dual_y[world, column])
    weight = float(0.0)
    if state.velocity_weight[world, row] > 0.0:
      weight = state.D[world, row] * total
    state.dual_g[world, row] = weight
  _block_sync()
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    weight = state.dual_g[world, row]
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      wp.atomic_add(state.rhs, world, column,
                    state.jacobian[world, 0, position] * weight)


@wp.func
def _dual_apply(state: FusedSparseSolverState, ids: wp.vec2i):
  """Apply the dual contact-force correction in place."""
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    if state.velocity_weight[world, row] <= 0.0:
      continue
    total = float(0.0)
    address = state.rowadr[world, row]
    count = state.rownnz[world, row]
    for position in range(address, address + count):
      column = state.colind[world, 0, position]
      total += (state.jacobian[world, 0, position]
                * state.solution[world, column])
    phi = state.dual_g[world, row] - state.D[world, row] * total
    state.force[world, row] = wp.max(state.force[world, row] - phi, 0.0)


@wp.func
def _dual_residual_dense(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.nv, wp.block_dim()):
    total = float(0.0)
    for row in range(state.nefc[world]):
      total += state.jacobian[world, row, dof] * state.force[world, row]
    for column in range(state.nv):
      total -= state.qM[world, dof, column] * state.dual_x[world, column]
    state.dual_y[world, dof] = total


@wp.func
def _dense_mass_solve(state: FusedSparseSolverState, ids: wp.vec2i):
  if ids[1] != 0:
    return
  world = ids[0]
  for row in range(state.nv):
    value = state.dual_y[world, row]
    for column in range(row):
      value -= (
          state.qLD[world, row, column]
          * state.dual_y[world, column])
    state.dual_y[world, row] = value / state.qLD[world, row, row]
  for offset in range(state.nv):
    row = state.nv - offset - 1
    value = state.dual_y[world, row]
    for column in range(row + 1, state.nv):
      value -= (
          state.qLD[world, column, row]
          * state.dual_y[world, column])
    state.dual_y[world, row] = value / state.qLD[world, row, row]


@wp.func
def _dual_rows_dense(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    total = float(0.0)
    for dof in range(state.nv):
      total += state.jacobian[world, row, dof] * state.dual_y[world, dof]
    active = state.velocity_weight[world, row] > 0.0
    state.dual_g[world, row] = wp.where(
        active, state.D[world, row] * total, 0.0)
  _block_sync()
  for dof in range(ids[1], state.nv, wp.block_dim()):
    total = float(0.0)
    for row in range(state.nefc[world]):
      total += state.jacobian[world, row, dof] * state.dual_g[world, row]
    state.rhs[world, dof] = total


@wp.func
def _dual_apply_dense(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    if state.velocity_weight[world, row] <= 0.0:
      continue
    response = _row_dot_dense(state, world, row)
    phi = state.dual_g[world, row] - state.D[world, row] * response
    state.force[world, row] = wp.max(state.force[world, row] - phi, 0.0)


@functools.cache
def _create_solver_body(
  coupled_iterations: int,
  pcg_iterations: int,
  early_stop: bool,
  is_sparse: bool = True,
):
  validate_rows = _validate_rows if is_sparse else _validate_dense_rows
  zero_system = _zero_system if is_sparse else _zero_system_dense
  accumulate_system = (
      _accumulate_system if is_sparse else _accumulate_system_dense)
  pcg_update = _pcg_update if is_sparse else _pcg_update_dense
  update_active = (
      _update_active_rows if is_sparse else _update_active_rows_dense)
  project_forces = (
      _project_row_forces if is_sparse else _project_row_forces_dense)
  dual_residual = _dual_residual if is_sparse else _dual_residual_dense
  mass_solve = _sparse_mass_solve if is_sparse else _dense_mass_solve
  dual_rows = _dual_rows if is_sparse else _dual_rows_dense
  dual_apply = _dual_apply if is_sparse else _dual_apply_dense
  scatter_solution = (
      _scatter_projected_solution
      if is_sparse else _scatter_projected_solution_dense)

  @wp.func
  def _solve_world(state: FusedSparseSolverState, ids: wp.vec2i):
    world = ids[0]
    lane = ids[1]
    _reset_iteration_counters(state, ids)
    _block_sync()
    wp.static(validate_rows)(state, ids)
    _block_sync()
    if state.overflow[world] != 0:
      _write_invalid(state, ids)
      return
    if state.nefc[world] == 0:
      return
    outer_count = int(0)
    pcg_count = int(0)
    for outer in range(wp.static(coupled_iterations)):
      outer_count += 1
      wp.static(zero_system)(state, ids)
      _block_sync()
      wp.static(accumulate_system)(state, ids)
      _block_sync()
      rz = _initialize_pcg(state, ids)
      for iteration in range(wp.static(pcg_iterations)):
        if wp.abs(rz) <= PCG_EPSILON:
          break
        rz_new = wp.static(pcg_update)(state, ids, rz)
        pcg_count += 1
        if iteration + 1 < wp.static(pcg_iterations):
          _update_direction(state, ids, wp.vec2(rz, rz_new))
        rz = rz_new
      if outer + 1 < wp.static(coupled_iterations):
        changed = wp.static(update_active)(state, ids)
        _block_sync()
        if wp.static(early_stop) and changed == 0.0:
          break
    wp.static(project_forces)(state, ids)
    _block_sync()
    if wp.static(DUAL_CORRECTION):
      _save_solution(state, ids)
      _block_sync()
      for dof in range(lane, state.vector_capacity, wp.block_dim()):
        state.dual_y[world, dof] = 0.0
      _block_sync()
      wp.static(dual_residual)(state, ids)
      _block_sync()
      wp.static(mass_solve)(state, ids)
      _block_sync()
      wp.static(dual_rows)(state, ids)
      _block_sync()
      for dof in range(lane, state.vector_capacity, wp.block_dim()):
        state.solution[world, dof] = 0.0
      _block_sync()
      rz2 = _initialize_pcg(state, ids)
      for iteration in range(wp.static(pcg_iterations)):
        if wp.abs(rz2) <= PCG_EPSILON:
          break
        rz2_new = wp.static(pcg_update)(state, ids, rz2)
        pcg_count += 1
        if iteration + 1 < wp.static(pcg_iterations):
          _update_direction(state, ids, wp.vec2(rz2, rz2_new))
        rz2 = rz2_new
      wp.static(dual_apply)(state, ids)
      _block_sync()
    if lane == 0:
      state.coupled_iterations_used[world] = float(outer_count)
      state.pcg_iterations_used[world] = float(pcg_count)
      state.solver_niter[world] = outer_count
    wp.static(scatter_solution)(state, ids)
    _finalize_dofs(state, ids)

  return _solve_world


@functools.cache
def _create_solver_kernel(
  coupled_iterations: int,
  pcg_iterations: int,
  early_stop: bool,
  is_sparse: bool = True,
):
  solve_world = _create_solver_body(
    coupled_iterations,
    pcg_iterations,
    early_stop,
    is_sparse,
  )

  @wp.kernel(enable_backward=False, module="unique")
  def _solve(state: FusedSparseSolverState):
    world, lane = wp.tid()
    wp.static(solve_world)(state, wp.vec2i(world, lane))

  return _solve


def _validate_config(config: FusedSparseSolverConfig) -> None:
  if config.coupled_iterations <= 0:
    raise ValueError("coupled_iterations must be positive")
  if config.pcg_iterations <= 0:
    raise ValueError("pcg_iterations must be positive")
  if config.block_dim <= 0 or config.block_dim > MAX_BLOCK_DIM:
    raise ValueError(f"block_dim must be in [1, {MAX_BLOCK_DIM}]")


def _validate_sparse_storage(data) -> None:
  if data.efc.J.shape[1] != 1 or data.efc.weighted_J.shape[1] != 1:
    raise ValueError("constraint Jacobian must use sparse (world, 1, nnz) layout")
  sparse_fields = ("J", "J_colind", "weighted_J")
  for name in sparse_fields:
    if getattr(data.efc, name).shape[2] < data.njmax_nnz:
      raise ValueError(f"{name} storage is smaller than njmax_nnz")


def _validate_vector_storage(model, data) -> None:
  vector_fields = (
    "contact_solution", "contact_diag", "contact_rhs", "contact_pcg_r",
    "contact_pcg_z", "contact_pcg_p", "contact_pcg_Ap",
  )
  for name in vector_fields:
    if getattr(data.efc, name).shape[1] < model.nv:
      raise ValueError(f"{name} storage is smaller than model.nv")


def _validate_scalar_storage(data) -> None:
  scalar_fields = ("contact_pcg_rz", "contact_pcg_rz_next")
  for name in scalar_fields:
    if getattr(data.efc, name).shape[0] < data.nworld:
      raise ValueError(f"{name} storage is smaller than data.nworld")


def _validate_row_storage(data) -> None:
  row_fields = (
    "J_rownnz", "J_rowadr",
    "contact_free_force", "contact_active_D", "D", "force",
  )
  for name in row_fields:
    if getattr(data.efc, name).shape[1] < data.njmax:
      raise ValueError(f"{name} storage is smaller than data.njmax")


def _validate_layout(model, data, rows) -> None:
  if not model.is_sparse:
    raise ValueError("fused sparse solver requires model.is_sparse=True")
  invalid_rows = rows.limit < 0 or rows.budget < 0 or rows.limit > rows.budget
  if invalid_rows or rows.limit > data.njmax:
    raise ValueError("constraint row limit and budget are inconsistent")
  _validate_sparse_storage(data)
  _validate_vector_storage(model, data)
  _validate_scalar_storage(data)
  _validate_row_storage(data)


def _validate_dense_layout(model, data, rows) -> None:
  if model.is_sparse:
    raise ValueError("dense PCG requires model.is_sparse=False")
  invalid_rows = rows.limit < 0 or rows.budget < 0 or rows.limit > rows.budget
  if invalid_rows or rows.limit > data.njmax:
    raise ValueError("constraint row limit and budget are inconsistent")
  if data.efc.J.shape[1] < rows.budget or data.efc.J.shape[2] < model.nv:
    raise ValueError("dense constraint Jacobian storage is too small")
  _validate_vector_storage(model, data)
  _validate_scalar_storage(data)
  for name in ("contact_free_force", "contact_active_D", "D", "force"):
    if getattr(data.efc, name).shape[1] < data.njmax:
      raise ValueError(f"{name} storage is smaller than data.njmax")


def _required_dual_array(data, name: str):
  if not hasattr(data.efc, name):
    raise RuntimeError(
        f"missing preallocated constraint workspace d.efc.{name}")
  return getattr(data.efc, name)


def _bind_mass_state(state, model, data) -> None:
  state.timestep = model.opt.timestep
  state.qM = data.qM
  state.qM_rowadr = model.qM_mulm_rowadr
  state.qM_col = model.qM_mulm_col
  state.qM_madr = model.qM_mulm_madr
  state.dof_Madr = model.dof_Madr
  state.qLD = data.qLD
  state.qLDiagInv = data.qLDiagInv
  state.M_rownnz = model.M_rownnz
  state.M_rowadr = model.M_rowadr
  state.M_colind = model.M_colind


def _bind_contact_state(state, data) -> None:
  state.rownnz = data.efc.J_rownnz
  state.rowadr = data.efc.J_rowadr
  state.colind = data.efc.J_colind
  state.jacobian = data.efc.J
  state.weighted_jacobian = data.efc.weighted_J
  state.nnz_count = data.efc.J_nnz
  state.overflow = data.efc.J_overflow
  state.free_force = data.efc.contact_free_force
  state.D = data.efc.D
  state.imp = data.efc.efc_imp
  state.velocity_weight = data.efc.contact_active_D
  state.force = data.efc.force
  state.nefc = data.nefc


def _bind_runtime_state(state, data) -> None:
  state.qvel = data.qvel
  state.qacc_smooth = data.qacc_smooth
  state.qfrc_smooth = data.qfrc_smooth
  state.qfrc_constraint = data.qfrc_constraint
  state.qfrc_total = data.qfrc_total
  state.qvel_predicted = data.qvel_smooth_pred
  state.solution = data.efc.contact_solution
  state.diagonal = data.efc.contact_diag
  state.rhs = data.efc.contact_rhs
  state.residual = data.efc.contact_pcg_r
  state.preconditioned = data.efc.contact_pcg_z
  state.direction = data.efc.contact_pcg_p
  state.matvec = data.efc.contact_pcg_Ap
  # The fused kernel keeps rho in registers, leaving these scalar workspaces
  # available for exact iteration diagnostics without another allocation.
  state.coupled_iterations_used = data.efc.contact_pcg_rz
  state.pcg_iterations_used = data.efc.contact_pcg_rz_next
  state.solver_niter = data.solver_niter
  state.dual_x = _required_dual_array(data, "dual_x")
  state.dual_y = _required_dual_array(data, "dual_y")
  state.dual_g = _required_dual_array(data, "dual_g")


def _make_state(model, data, rows) -> FusedSparseSolverState:
  state = FusedSparseSolverState()
  _bind_mass_state(state, model, data)
  _bind_contact_state(state, data)
  _bind_runtime_state(state, data)
  state.nv = model.nv
  state.vector_capacity = data.efc.contact_solution.shape[1]
  state.row_limit = rows.limit
  state.row_capacity = data.njmax
  state.nnz_capacity = data.njmax_nnz
  return state


def solve_sparse_contacts(model, data, rows, *, config: FusedSparseSolverConfig) -> None:
  """Launch the fused CSR solver once for all worlds."""

  _validate_config(config)
  _validate_layout(model, data, rows)
  state = _make_state(model, data, rows)
  kernel = _create_solver_kernel(
    config.coupled_iterations,
    config.pcg_iterations,
    config.early_stop,
  )
  wp.launch_tiled(
    kernel,
    dim=[data.nworld],
    inputs=[state],
    block_dim=config.block_dim,
    device=data.qvel.device,
  )


def solve_dense_contacts(
    model,
    data,
    rows,
    *,
    config: FusedSparseSolverConfig,
) -> None:
  """Run the same matrix-free PCG algorithm with dense M and J access."""
  _validate_config(config)
  _validate_dense_layout(model, data, rows)
  state = _make_state(model, data, rows)
  kernel = _create_solver_kernel(
      config.coupled_iterations,
      config.pcg_iterations,
      config.early_stop,
      False,
  )
  wp.launch_tiled(
      kernel,
      dim=[data.nworld],
      inputs=[state],
      block_dim=config.block_dim,
      device=data.qvel.device,
  )
