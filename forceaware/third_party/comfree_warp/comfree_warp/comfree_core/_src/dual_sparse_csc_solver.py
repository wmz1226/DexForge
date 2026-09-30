# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
"""Dual CSR/CSC full-implicit contact solver.

CSR remains the source of truth for row operations.  A compact CSC view is
built once per solve so all transpose products gather by DoF without global
floating-point atomics.  The workspace is caller-owned and capture-safe.
"""

from __future__ import annotations

import dataclasses
import functools

import warp as wp

from .fused_sparse_solver import FusedSparseSolverConfig
from .fused_sparse_solver import FusedSparseSolverState
from .fused_sparse_solver import PCG_EPSILON
from .fused_sparse_solver import _block_sum
from .fused_sparse_solver import _block_sync
from .fused_sparse_solver import _finalize_dofs
from .fused_sparse_solver import _make_state
from .types import DUAL_CORRECTION
from .fused_sparse_solver import _dual_apply
from .fused_sparse_solver import _project_row_forces
from .fused_sparse_solver import _save_solution
from .fused_sparse_solver import _sparse_mass_solve
from .fused_sparse_solver import _reset_iteration_counters
from .fused_sparse_solver import _safe_ratio
from .fused_sparse_solver import _update_active_rows
from .fused_sparse_solver import _update_direction
from .fused_sparse_solver import _validate_config
from .fused_sparse_solver import _validate_layout
from .fused_sparse_solver import _validate_rows
from .fused_sparse_solver import _write_invalid


wp.set_module_options({"enable_backward": False})


SPARSE_SOLVER_AUTO = "auto"
SPARSE_SOLVER_GLOBAL = "global"
SPARSE_SOLVER_DUAL = "dual"
VALID_SPARSE_SOLVERS = (
  SPARSE_SOLVER_AUTO,
  SPARSE_SOLVER_GLOBAL,
  SPARSE_SOLVER_DUAL,
)
AUTO_DUAL_MIN_NWORLD = 256
AUTO_DUAL_MAX_NV = 128
DUAL_ENTRY_POSITION_FIELD = "contact_csc_entry_position"
DUAL_ENTRY_ROW_FIELD = "contact_csc_entry_row"
DUAL_ROW_RESPONSE_FIELD = "contact_csc_row_response"


def validate_sparse_solver(value) -> str:
  """Return a validated public sparse-solver request."""

  if not isinstance(value, str) or value not in VALID_SPARSE_SOLVERS:
    choices = ", ".join(VALID_SPARSE_SOLVERS)
    raise ValueError(f"comfree_sparse_solver must be one of: {choices}; got {value!r}")
  return value


def resolve_sparse_solver(requested: str, *, is_sparse: bool, nworld: int, nv: int) -> str:
  """Freeze the requested policy for one allocated data layout."""

  validate_sparse_solver(requested)
  if requested == SPARSE_SOLVER_DUAL and not is_sparse:
    raise ValueError("comfree_sparse_solver='dual' requires a sparse constraint Jacobian")
  if not is_sparse or requested == SPARSE_SOLVER_GLOBAL:
    return SPARSE_SOLVER_GLOBAL
  if requested == SPARSE_SOLVER_DUAL:
    return SPARSE_SOLVER_DUAL
  # Real Fig.7 crossover measurements show a durable gain from 256 worlds;
  # the measured low-DoF regime is bounded conservatively to nv <= 128.
  if nworld >= AUTO_DUAL_MIN_NWORLD and nv <= AUTO_DUAL_MAX_NV:
    return SPARSE_SOLVER_DUAL
  return SPARSE_SOLVER_GLOBAL


@dataclasses.dataclass(frozen=True, kw_only=True)
class DualSparseCscWorkspace:
  """Preallocated O(nnz + nv + rows) transpose-product storage."""

  column_count: object
  column_address: object
  column_cursor: object
  entry_position: object
  entry_row: object
  row_response: object
  entry_count: object


@wp.struct
class DualSparseCscState:
  base: FusedSparseSolverState
  column_count: wp.array2d(dtype=int)
  column_address: wp.array2d(dtype=int)
  column_cursor: wp.array2d(dtype=int)
  entry_position: wp.array2d(dtype=int)
  entry_row: wp.array2d(dtype=int)
  row_response: wp.array2d(dtype=float)
  entry_count: wp.array(dtype=int)


@wp.func
def _clear_csc(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  lane = ids[1]
  if lane == 0:
    state.entry_count[world] = 0
  for dof in range(lane, state.base.vector_capacity, wp.block_dim()):
    state.column_count[world, dof] = 0
    state.column_address[world, dof] = 0
    state.column_cursor[world, dof] = 0


@wp.func
def _count_columns(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.base.nefc[world], wp.block_dim()):
    address = state.base.rowadr[world, row]
    count = state.base.rownnz[world, row]
    for position in range(address, address + count):
      column = state.base.colind[world, 0, position]
      wp.atomic_add(state.column_count, world, column, 1)


@wp.func
def _prefix_columns(state: DualSparseCscState, ids: wp.vec2i):
  if ids[1] != 0:
    return
  world = ids[0]
  cursor = int(0)
  for dof in range(state.base.nv):
    state.column_address[world, dof] = cursor
    state.column_cursor[world, dof] = cursor
    cursor += state.column_count[world, dof]
  state.entry_count[world] = cursor
  if cursor > state.base.nnz_capacity:
    state.base.overflow[world] = 1


@wp.func
def _fill_csc(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.base.nefc[world], wp.block_dim()):
    address = state.base.rowadr[world, row]
    count = state.base.rownnz[world, row]
    for position in range(address, address + count):
      column = state.base.colind[world, 0, position]
      slot = wp.atomic_add(state.column_cursor, world, column, 1)
      state.entry_position[world, slot] = position
      state.entry_row[world, slot] = row


@wp.func
def _verify_csc(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    expected = state.column_address[world, dof] + state.column_count[world, dof]
    if state.column_cursor[world, dof] != expected:
      wp.atomic_max(state.base.overflow, world, 1)


@wp.func
def _build_csc(state: DualSparseCscState, ids: wp.vec2i):
  _clear_csc(state, ids)
  _block_sync()
  _validate_rows(state.base, ids)
  _block_sync()
  if state.base.overflow[ids[0]] != 0:
    return
  _count_columns(state, ids)
  _block_sync()
  _prefix_columns(state, ids)
  _block_sync()
  if state.base.overflow[ids[0]] != 0:
    return
  _fill_csc(state, ids)
  _block_sync()
  _verify_csc(state, ids)


@wp.kernel(enable_backward=False, module="unique")
def _build_csc_kernel(state: DualSparseCscState):
  world, lane = wp.tid()
  _build_csc(state, wp.vec2i(world, lane))


@wp.func
def _gather_system_column(state: DualSparseCscState, world: int, dof: int) -> wp.vec2:
  diagonal = state.base.qM[world, 0, state.base.dof_Madr[dof]]
  rhs = float(0.0)
  address = state.column_address[world, dof]
  count = state.column_count[world, dof]
  for slot in range(address, address + count):
    row = state.entry_row[world, slot]
    position = state.entry_position[world, slot]
    value = state.base.jacobian[world, 0, position]
    row_weight = state.base.velocity_weight[world, row]
    diagonal += row_weight * value * value
    if row_weight > 0.0:
      rhs += state.base.free_force[world, row] * value
  return wp.vec2(diagonal, rhs)


@wp.func
def _prepare_system(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.base.vector_capacity, wp.block_dim()):
    diagonal_rhs = wp.vec2(0.0, 0.0)
    if dof < state.base.nv:
      diagonal_rhs = _gather_system_column(state, world, dof)
    state.base.diagonal[world, dof] = diagonal_rhs[0]
    state.base.rhs[world, dof] = diagonal_rhs[1]
    state.base.solution[world, dof] = 0.0
    state.base.residual[world, dof] = 0.0
    state.base.preconditioned[world, dof] = 0.0
    state.base.direction[world, dof] = 0.0
    state.base.matvec[world, dof] = 0.0


@wp.func
def _initialize_pcg(state: DualSparseCscState, ids: wp.vec2i) -> float:
  world = ids[0]
  local_rz = float(0.0)
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    residual = state.base.rhs[world, dof]
    preconditioned = residual / state.base.diagonal[world, dof]
    state.base.residual[world, dof] = residual
    state.base.preconditioned[world, dof] = preconditioned
    state.base.direction[world, dof] = preconditioned
    local_rz += residual * preconditioned
  total_rz = _block_sum(local_rz)
  _block_sync()
  return total_rz


@wp.func
def _compute_row_responses(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.base.nefc[world], wp.block_dim()):
    total = float(0.0)
    address = state.base.rowadr[world, row]
    count = state.base.rownnz[world, row]
    for position in range(address, address + count):
      column = state.base.colind[world, 0, position]
      value = state.base.jacobian[world, 0, position]
      total += value * state.base.direction[world, column]
    state.row_response[world, row] = total


@wp.func
def _gather_operator_columns(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    total = float(0.0)
    for entry in range(
        state.base.qM_rowadr[dof], state.base.qM_rowadr[dof + 1]):
      column = state.base.qM_col[entry]
      address = state.base.qM_madr[entry]
      total += (state.base.qM[world, 0, address]
                * state.base.direction[world, column])
    address = state.column_address[world, dof]
    count = state.column_count[world, dof]
    for slot in range(address, address + count):
      row = state.entry_row[world, slot]
      position = state.entry_position[world, slot]
      value = state.base.jacobian[world, 0, position]
      row_weight = state.base.velocity_weight[world, row]
      total += value * row_weight * state.row_response[world, row]
    state.base.matvec[world, dof] = total


@wp.func
def _apply_operator(state: DualSparseCscState, ids: wp.vec2i):
  _compute_row_responses(state, ids)
  _block_sync()
  _gather_operator_columns(state, ids)
  _block_sync()


@wp.func
def _pcg_update(state: DualSparseCscState, ids: wp.vec2i, rz_old: float) -> float:
  world = ids[0]
  _apply_operator(state, ids)
  local_pap = float(0.0)
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    local_pap += state.base.direction[world, dof] * state.base.matvec[world, dof]
  alpha = _safe_ratio(rz_old, _block_sum(local_pap))
  local_rz = float(0.0)
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    residual = state.base.residual[world, dof] - alpha * state.base.matvec[world, dof]
    preconditioned = residual / state.base.diagonal[world, dof]
    state.base.solution[world, dof] += alpha * state.base.direction[world, dof]
    state.base.residual[world, dof] = residual
    state.base.preconditioned[world, dof] = preconditioned
    local_rz += residual * preconditioned
  rz_new = _block_sum(local_rz)
  _block_sync()
  return rz_new


@wp.func
def _gather_projected_solution(state: DualSparseCscState, ids: wp.vec2i):
  world = ids[0]
  for dof in range(ids[1], state.base.vector_capacity, wp.block_dim()):
    total = float(0.0)
    if dof < state.base.nv:
      address = state.column_address[world, dof]
      count = state.column_count[world, dof]
      for slot in range(address, address + count):
        row = state.entry_row[world, slot]
        position = state.entry_position[world, slot]
        value = state.base.jacobian[world, 0, position]
        total += value * state.base.force[world, row]
    state.base.solution[world, dof] = total


@wp.func
def _dual_residual_csc(state: DualSparseCscState, ids: wp.vec2i):
  """Compute J^T lambda - M x using deterministic CSC gathers."""
  world = ids[0]
  for dof in range(ids[1], state.base.nv, wp.block_dim()):
    total = float(0.0)
    for entry in range(state.base.qM_rowadr[dof], state.base.qM_rowadr[dof + 1]):
      total -= (state.base.qM[world, 0, state.base.qM_madr[entry]]
                * state.base.dual_x[world, state.base.qM_col[entry]])
    address = state.column_address[world, dof]
    count = state.column_count[world, dof]
    for slot in range(address, address + count):
      row = state.entry_row[world, slot]
      position = state.entry_position[world, slot]
      total += (state.base.jacobian[world, 0, position]
                * state.base.force[world, row])
    state.base.dual_y[world, dof] = total


@wp.func
def _dual_rows_csc(state: DualSparseCscState, ids: wp.vec2i):
  """Compute D (J y) with CSR rows and gather J^T g with CSC columns."""
  world = ids[0]
  for row in range(ids[1], state.base.nefc[world], wp.block_dim()):
    total = float(0.0)
    address = state.base.rowadr[world, row]
    count = state.base.rownnz[world, row]
    for position in range(address, address + count):
      column = state.base.colind[world, 0, position]
      total += (state.base.jacobian[world, 0, position]
                * state.base.dual_y[world, column])
    weight = float(0.0)
    if state.base.velocity_weight[world, row] > 0.0:
      weight = state.base.D[world, row] * total
    state.base.dual_g[world, row] = weight
  _block_sync()
  for dof in range(ids[1], state.base.vector_capacity, wp.block_dim()):
    total = float(0.0)
    if dof < state.base.nv:
      address = state.column_address[world, dof]
      count = state.column_count[world, dof]
      for slot in range(address, address + count):
        row = state.entry_row[world, slot]
        position = state.entry_position[world, slot]
        total += (state.base.jacobian[world, 0, position]
                  * state.base.dual_g[world, row])
    state.base.rhs[world, dof] = total


@functools.cache
def _create_solver_body(
  coupled_iterations: int,
  pcg_iterations: int,
  early_stop: bool,
):
  @wp.func
  def _solve_world(state: DualSparseCscState, ids: wp.vec2i):
    world = ids[0]
    lane = ids[1]
    _reset_iteration_counters(state.base, ids)
    _block_sync()
    if state.base.overflow[world] != 0:
      _write_invalid(state.base, ids)
      return
    if state.base.nefc[world] == 0:
      return
    outer_count = int(0)
    pcg_count = int(0)
    for outer in range(wp.static(coupled_iterations)):
      outer_count += 1
      _prepare_system(state, ids)
      _block_sync()
      rz = _initialize_pcg(state, ids)
      for iteration in range(wp.static(pcg_iterations)):
        if wp.abs(rz) <= PCG_EPSILON:
          break
        rz_new = _pcg_update(state, ids, rz)
        pcg_count += 1
        if iteration + 1 < wp.static(pcg_iterations):
          _update_direction(state.base, ids, wp.vec2(rz, rz_new))
        rz = rz_new
      if outer + 1 < wp.static(coupled_iterations):
        changed = _update_active_rows(state.base, ids)
        _block_sync()
        if wp.static(early_stop) and changed == 0.0:
          break
    if lane == 0:
      state.base.coupled_iterations_used[world] = float(outer_count)
      state.base.pcg_iterations_used[world] = float(pcg_count)
      state.base.solver_niter[world] = outer_count
    _project_row_forces(state.base, ids)
    _block_sync()
    if wp.static(DUAL_CORRECTION):
      _save_solution(state.base, ids)
      _block_sync()
      _dual_residual_csc(state, ids)
      _block_sync()
      _sparse_mass_solve(state.base, ids)
      _block_sync()
      _dual_rows_csc(state, ids)
      _block_sync()
      for dof in range(lane, state.base.vector_capacity, wp.block_dim()):
        state.base.solution[world, dof] = 0.0
      _block_sync()
      rz2 = _initialize_pcg(state, ids)
      for iteration in range(wp.static(pcg_iterations)):
        if wp.abs(rz2) <= PCG_EPSILON:
          break
        rz2_new = _pcg_update(state, ids, rz2)
        if iteration + 1 < wp.static(pcg_iterations):
          _update_direction(state.base, ids, wp.vec2(rz2, rz2_new))
        rz2 = rz2_new
      _dual_apply(state.base, ids)
      _block_sync()
    _gather_projected_solution(state, ids)
    _finalize_dofs(state.base, ids)

  return _solve_world


@functools.cache
def _create_solver_kernel(
  coupled_iterations: int,
  pcg_iterations: int,
  early_stop: bool,
):
  solve_world = _create_solver_body(
    coupled_iterations,
    pcg_iterations,
    early_stop,
  )

  @wp.kernel(enable_backward=False, module="unique")
  def _solve(state: DualSparseCscState):
    world, lane = wp.tid()
    wp.static(solve_world)(state, wp.vec2i(world, lane))

  return _solve


def make_dual_sparse_csc_workspace(model, data) -> DualSparseCscWorkspace:
  """Allocate workspace before graph capture for one model/data layout."""

  vector_capacity = data.efc.contact_solution.shape[1]
  device = data.qvel.device
  vector_shape = (data.nworld, vector_capacity)
  entry_shape = (data.nworld, data.njmax_nnz)
  row_shape = (data.nworld, data.njmax)
  return DualSparseCscWorkspace(
    column_count=wp.zeros(vector_shape, dtype=int, device=device),
    column_address=wp.zeros(vector_shape, dtype=int, device=device),
    column_cursor=wp.zeros(vector_shape, dtype=int, device=device),
    entry_position=wp.zeros(entry_shape, dtype=int, device=device),
    entry_row=wp.zeros(entry_shape, dtype=int, device=device),
    row_response=wp.zeros(row_shape, dtype=float, device=device),
    entry_count=wp.zeros(data.nworld, dtype=int, device=device),
  )


def _data_workspace_field(data, name: str):
  if not hasattr(data.efc, name):
    raise ValueError(f"dual CSC workspace field is missing: efc.{name}")
  return getattr(data.efc, name)


def workspace_from_data(data) -> DualSparseCscWorkspace:
  """Create a zero-allocation view over a sparse data object's CSC workspace."""

  return DualSparseCscWorkspace(
    column_count=_data_workspace_field(data, "contact_dof_mask"),
    column_address=_data_workspace_field(data, "contact_dof_prefix"),
    column_cursor=_data_workspace_field(data, "contact_dof_ids"),
    entry_position=_data_workspace_field(data, DUAL_ENTRY_POSITION_FIELD),
    entry_row=_data_workspace_field(data, DUAL_ENTRY_ROW_FIELD),
    row_response=_data_workspace_field(data, DUAL_ROW_RESPONSE_FIELD),
    entry_count=_data_workspace_field(data, "contact_dof_count"),
  )


def _validate_workspace_array(array, shape, *, dtype, device, name: str) -> None:
  if array.shape != shape:
    raise ValueError(f"workspace {name} has shape {array.shape}, expected {shape}")
  if array.dtype != dtype:
    raise TypeError(f"workspace {name} has dtype {array.dtype}, expected {dtype}")
  if array.device != device:
    raise ValueError(f"workspace {name} and runtime data must share a device")


def _validate_workspace(model, data, workspace: DualSparseCscWorkspace) -> None:
  vector_shape = (data.nworld, data.efc.contact_solution.shape[1])
  entry_shape = (data.nworld, data.njmax_nnz)
  row_shape = (data.nworld, data.njmax)
  device = data.qvel.device
  for name in ("column_count", "column_address", "column_cursor"):
    _validate_workspace_array(
      getattr(workspace, name), vector_shape, dtype=wp.int32, device=device, name=name
    )
  for name in ("entry_position", "entry_row"):
    _validate_workspace_array(
      getattr(workspace, name), entry_shape, dtype=wp.int32, device=device, name=name
    )
  _validate_workspace_array(
    workspace.row_response, row_shape, dtype=wp.float32, device=device, name="row_response"
  )
  _validate_workspace_array(
    workspace.entry_count, (data.nworld,), dtype=wp.int32, device=device, name="entry_count"
  )
  if vector_shape[1] < model.nv:
    raise ValueError("CSC workspace vector capacity is smaller than model.nv")


def _make_dual_state(model, data, rows, *, workspace) -> DualSparseCscState:
  state = DualSparseCscState()
  state.base = _make_state(model, data, rows)
  state.column_count = workspace.column_count
  state.column_address = workspace.column_address
  state.column_cursor = workspace.column_cursor
  state.entry_position = workspace.entry_position
  state.entry_row = workspace.entry_row
  state.row_response = workspace.row_response
  state.entry_count = workspace.entry_count
  return state


def solve_dual_sparse_contacts(
  model,
  data,
  rows,
  *,
  workspace: DualSparseCscWorkspace,
  config: FusedSparseSolverConfig,
) -> None:
  """Build CSC from runtime CSR, then solve with transpose gathers."""

  _validate_config(config)
  _validate_layout(model, data, rows)
  _validate_workspace(model, data, workspace)
  state = _make_dual_state(model, data, rows, workspace=workspace)
  wp.launch_tiled(
    _build_csc_kernel,
    dim=[data.nworld],
    inputs=[state],
    block_dim=config.block_dim,
    device=data.qvel.device,
  )
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
