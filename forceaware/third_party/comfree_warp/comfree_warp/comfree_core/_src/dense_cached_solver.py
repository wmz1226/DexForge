# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
# Modified for this distribution: environment-variable prefix anonymized; numerical defaults unchanged.
# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
"""MJWarp-style cached Cholesky active-set solver."""

from dataclasses import dataclass, replace

import warp as wp

from comfree_warp.mujoco_warp._src.block_cholesky import (
    create_blocked_cholesky_factorize_solve_func,
)
from comfree_warp.mujoco_warp._src.block_cholesky import (
    create_blocked_upper_cholesky_solve_func,
)
from comfree_warp.mujoco_warp._src import smooth
from comfree_warp.mujoco_warp._src.warp_util import cache_kernel

from . import diagonal_contact
from .config import env_int
from .types import ConstraintRows, Data, DUAL_CORRECTION, Model

wp.set_module_options({"enable_backward": False})

DENSE_BLOCK_SIZE = 16
DENSE_BLOCK_DIM = 32
ROW_BLOCK_DIM = 64
DIRECT_NV_LIMIT = 32


BLOCKED_L0_ROW_TILE_SIZE = env_int(
    "COMFREE_BLOCKED_L0_ROW_TILE", 16, minimum=1)
AUTO_BLOCK_SIZE = 0
BLOCK_SIZE_CANDIDATES = (DENSE_BLOCK_SIZE, 8, 4, 2, 1)


@dataclass(frozen=True)
class DenseCachedConfig:
  """Fixed active-set budget and blocked-Cholesky launch settings."""

  iterations: int
  early_stop: bool = False
  finalize_acceleration: bool = True
  block_size: int = AUTO_BLOCK_SIZE
  block_dim: int = DENSE_BLOCK_DIM


@dataclass(frozen=True)
class DenseCachedWorkspace:
  """Device storage owned by one isolated cached solver."""

  matrix: wp.array
  factor: wp.array
  rhs: wp.array
  solution: wp.array
  changed_rows: wp.array
  changed_count: wp.array
  done: wp.array
  delta_weight: wp.array
  delta_free: wp.array
  row_response: wp.array


@wp.struct
class DenseCachedState:
  nv: int
  qM: wp.array3d(dtype=float)
  qM_rowadr: wp.array(dtype=int)
  qM_col: wp.array(dtype=int)
  qM_madr: wp.array(dtype=int)
  qLD: wp.array3d(dtype=float)
  J: wp.array3d(dtype=float)
  J_rownnz: wp.array2d(dtype=int)
  J_rowadr: wp.array2d(dtype=int)
  J_colind: wp.array3d(dtype=int)
  free: wp.array2d(dtype=float)
  D: wp.array2d(dtype=float)
  active_D: wp.array2d(dtype=float)
  nefc: wp.array(dtype=int)
  row_limit: int
  matrix: wp.array3d(dtype=float)
  factor: wp.array3d(dtype=float)
  rhs: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  changed_rows: wp.array2d(dtype=int)
  changed_count: wp.array(dtype=int)
  done: wp.array(dtype=bool)
  delta_weight: wp.array2d(dtype=float)
  delta_free: wp.array2d(dtype=float)
  row_response: wp.array2d(dtype=float)
  solver_niter: wp.array(dtype=int)
  qfrc_smooth: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  qfrc_constraint: wp.array2d(dtype=float)
  qfrc_total: wp.array2d(dtype=float)
  dual_x: wp.array2d(dtype=float)
  dual_y: wp.array2d(dtype=float)
  dual_g: wp.array2d(dtype=float)


@dataclass(frozen=True)
class DenseCachedSolver:
  """Bound solver whose workspace is allocated before CUDA graph capture."""

  config: DenseCachedConfig
  workspace: DenseCachedWorkspace | None

  def solve(self, m: Model, d: Data, *, rows: ConstraintRows) -> None:
    _validate_inputs(m, d, rows=rows, config=self.config)
    if m.is_sparse:
      _solve_sparse(self, m, d, rows=rows)
      return
    if m.nv_pad <= DIRECT_NV_LIMIT:
      _solve_direct(self, m, d, rows=rows)
      return
    if self.config.iterations == 0:
      diagonal_contact.solve_fused_dense_contacts(
          m, d, rows, row_tile_size=BLOCKED_L0_ROW_TILE_SIZE)
      return
    diagonal_contact.seed_fused_dense_contacts(
        m, d, rows, row_tile_size=BLOCKED_L0_ROW_TILE_SIZE)
    _build_and_factor(self, m, d, rows=rows)
    for _ in range(1, self.config.iterations):
      _update_and_solve(self, m, d, rows=rows)
    _write_solution(self, m, d, rows=rows)
    if self.config.finalize_acceleration:
      smooth.solve_m(m, d, d.qacc, d.qfrc_total)


@wp.func_native("WP_TILE_SYNC();")
def _block_sync():
  ...


@wp.func
def _upper_triangle_indices(element: int) -> wp.vec2i:
  row = int((wp.sqrt(float(8 * element + 1)) - 1.0) * 0.5)
  column = element - row * (row + 1) // 2
  return wp.vec2i(column, row)


@cache_kernel
def _build_system(nv: int, nv_pad: int, row_tile: int, row_budget: int):
  """Assemble M + J^T D A J and J^T f0 using row-block tile products."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, lane = wp.tid()
    NV_PAD = wp.static(nv_pad)
    ROW_TILE = wp.static(row_tile)
    ROW_BUDGET = wp.static(row_budget)
    active_rows = min(state.nefc[world], state.row_limit)


    dof_id = wp.tile_arange(NV_PAD, dtype=int)
    nv_bound = wp.tile_ones(shape=NV_PAD, dtype=int) * wp.static(nv)
    padding = wp.tile_ones(shape=NV_PAD, dtype=float) - wp.tile_map(
        diagonal_contact._in_bounds, dof_id, nv_bound)
    matrix = wp.tile_load(
        state.qM[world], shape=(NV_PAD, NV_PAD), bounds_check=True)
    matrix = wp.tile_diag_add(matrix, padding)
    rhs = wp.tile_zeros(shape=NV_PAD, dtype=float)

    for row_start in range(0, ROW_BUDGET, ROW_TILE):
      if row_start >= active_rows:
        break
      x_rows = wp.tile_load(
          state.J[world], shape=(ROW_TILE, NV_PAD),
          offset=(row_start, 0), bounds_check=True)
      weight = wp.tile_load(
          state.active_D[world], shape=ROW_TILE,
          offset=row_start, bounds_check=True)
      free = wp.tile_load(
          state.free[world], shape=ROW_TILE,
          offset=row_start, bounds_check=True)

      row_id = wp.tile_arange(ROW_TILE, dtype=int)
      bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (active_rows - row_start)
      mask = wp.tile_map(diagonal_contact._in_bounds, row_id, bound)
      weight = weight * mask
      active = wp.tile_map(diagonal_contact._active, weight)
      matrix += wp.tile_matmul(
          wp.tile_map(
              wp.mul, wp.tile_transpose(x_rows),
              wp.tile_broadcast(weight, shape=(NV_PAD, ROW_TILE))),
          x_rows)
      rhs += wp.tile_sum(
          wp.tile_map(
              wp.mul, wp.tile_transpose(x_rows),
              wp.tile_broadcast(active * free, shape=(NV_PAD, ROW_TILE))),
          axis=1)


    wp.tile_store(state.matrix[world], matrix)
    wp.tile_store(state.rhs[world], rhs)

  return kernel


@wp.kernel(enable_backward=False)
def _initialize_sparse_system(state: DenseCachedState):
  world, element = wp.tid()
  if element < state.rhs.shape[1]:
    state.rhs[world, element] = 0.0
  indices = _upper_triangle_indices(element)
  row = indices[0]
  column = indices[1]
  value = float(0.0)
  if row == column:
    value = 1.0
  if column < state.nv:
    mass_value = float(0.0)
    for entry in range(
        state.qM_rowadr[row], state.qM_rowadr[row + 1]):
      if state.qM_col[entry] == column:
        mass_value += state.qM[world, 0, state.qM_madr[entry]]
    value = mass_value
  state.matrix[world, row, column] = value


@wp.func
def _scatter_sparse_row(
    state: DenseCachedState,
    ids: wp.vec2i,
    update: wp.vec2,
):
  world = ids[0]
  contact = ids[1]
  address = state.J_rowadr[world, contact]
  count = state.J_rownnz[world, contact]
  for first in range(address, address + count):
    row = state.J_colind[world, 0, first]
    row_value = state.J[world, 0, first]
    wp.atomic_add(state.rhs, world, row, row_value * update[1])
    for second in range(first, address + count):
      column = state.J_colind[world, 0, second]
      column_value = state.J[world, 0, second]
      wp.atomic_add(
          state.matrix, world, wp.min(row, column), wp.max(row, column),
          update[0] * row_value * column_value)


@wp.kernel(enable_backward=False)
def _accumulate_sparse_system(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  weight = state.active_D[world, contact]
  if weight != 0.0:
    _scatter_sparse_row(
        state, wp.vec2i(world, contact),
        wp.vec2(weight, state.free[world, contact]))


@cache_kernel
def _factorize_solve(block_size: int, nv_pad: int):
  factorize_solve = create_blocked_cholesky_factorize_solve_func(
      block_size, nv_pad)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, lane = wp.tid()
    factorize_solve(
        state.matrix[world], state.rhs[world], wp.static(nv_pad),
        state.factor[world], state.solution[world])
    if lane == 0:
      state.done[world] = False
      state.solver_niter[world] = 1

  return kernel


@wp.func
def _update_contact(
    state: DenseCachedState,
    ids: wp.vec2i,
    nv: int,
):
  world = ids[0]
  contact = ids[1]
  response = state.row_response[world, contact]
  old_weight = state.active_D[world, contact]
  candidate = state.free[world, contact] - state.D[world, contact] * response
  new_weight = wp.where(candidate > 0.0, state.D[world, contact], 0.0)
  if new_weight == old_weight:
    return
  slot = wp.atomic_add(state.changed_count, world, 1)
  state.changed_rows[world, slot] = contact
  state.delta_weight[world, contact] = new_weight - old_weight
  state.delta_free[world, contact] = (
      wp.where(new_weight != 0.0, 1.0, 0.0)
      - wp.where(old_weight != 0.0, 1.0, 0.0)
  ) * state.free[world, contact]
  state.active_D[world, contact] = new_weight


@wp.func
def _update_rhs(
    state: DenseCachedState,
    ids: wp.vec2i,
):
  world = ids[0]
  dof = ids[1]
  value = state.rhs[world, dof]
  for slot in range(state.changed_count[world]):
    contact = state.changed_rows[world, slot]
    value += state.J[world, contact, dof] * state.delta_free[world, contact]
  state.rhs[world, dof] = value


@wp.func
def _update_matrix(
    state: DenseCachedState,
    ids: wp.vec2i,
    nv: int,
):
  indices = _upper_triangle_indices(ids[1])
  row = indices[0]
  column = indices[1]
  if column >= nv:
    return
  value = state.matrix[ids[0], row, column]
  for slot in range(state.changed_count[ids[0]]):
    contact = state.changed_rows[ids[0], slot]
    value += (state.delta_weight[ids[0], contact]
              * state.J[ids[0], contact, row]
              * state.J[ids[0], contact, column])
  state.matrix[ids[0], row, column] = value


@cache_kernel
def _update_active(
    nv: int, nv_pad: int, row_budget: int, row_tile: int, early_stop: bool):
  """Update the active set using tiled J x products and compact changed rows."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, lane = wp.tid()
    if state.done[world]:
      return
    if lane == 0:
      state.changed_count[world] = 0
    NV_PAD = wp.static(nv_pad)
    ROW_TILE = wp.static(row_tile)
    active_rows = min(state.nefc[world], state.row_limit)
    solution = wp.tile_load(
        state.solution[world], shape=NV_PAD, bounds_check=True)
    for start in range(0, wp.static(row_budget), ROW_TILE):
      if start >= active_rows:
        break
      x_rows = wp.tile_load(
          state.J[world], shape=(ROW_TILE, NV_PAD),
          offset=(start, 0), bounds_check=True)
      response = wp.tile_sum(
          x_rows * wp.tile_broadcast(solution, shape=(ROW_TILE, NV_PAD)),
          axis=1)
      wp.tile_store(
          state.row_response[world], response, offset=start, bounds_check=True)
    _block_sync()
    for contact in range(
        lane, wp.static(row_budget), wp.block_dim()):
      if contact < active_rows:
        _update_contact(state, wp.vec2i(world, contact), wp.static(nv))
    _block_sync()
    if lane == 0 and wp.static(early_stop):
      state.done[world] = state.changed_count[world] == 0

  return kernel


@wp.kernel(enable_backward=False)
def _clear_sparse_changes(state: DenseCachedState):
  world = wp.tid()
  if not state.done[world]:
    state.changed_count[world] = 0


@wp.func
def _sparse_row_response(
    state: DenseCachedState,
    ids: wp.vec2i,
) -> float:
  world = ids[0]
  contact = ids[1]
  response = float(0.0)
  address = state.J_rowadr[world, contact]
  count = state.J_rownnz[world, contact]
  for position in range(address, address + count):
    dof = state.J_colind[world, 0, position]
    response += state.J[world, 0, position] * state.solution[world, dof]
  return response


@wp.kernel(enable_backward=False)
def _update_sparse_active(state: DenseCachedState):
  world, contact = wp.tid()
  if state.done[world]:
    return
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  ids = wp.vec2i(world, contact)
  response = _sparse_row_response(state, ids)
  old_weight = state.active_D[world, contact]
  candidate = state.free[world, contact] - state.D[world, contact] * response
  new_weight = wp.where(candidate > 0.0, state.D[world, contact], 0.0)
  if new_weight == old_weight:
    return
  slot = wp.atomic_add(state.changed_count, world, 1)
  state.changed_rows[world, slot] = contact
  state.delta_weight[world, contact] = new_weight - old_weight
  state.delta_free[world, contact] = (
      wp.where(new_weight != 0.0, 1.0, 0.0)
      - wp.where(old_weight != 0.0, 1.0, 0.0)
  ) * state.free[world, contact]
  state.active_D[world, contact] = new_weight


@wp.kernel(enable_backward=False)
def _finish_sparse_active(state: DenseCachedState, early_stop: bool):
  world = wp.tid()
  if not state.done[world] and early_stop:
    state.done[world] = state.changed_count[world] == 0


@wp.kernel(enable_backward=False)
def _accumulate_sparse_updates(state: DenseCachedState):
  world, slot = wp.tid()
  if state.done[world] or slot >= state.changed_count[world]:
    return
  contact = state.changed_rows[world, slot]
  _scatter_sparse_row(
      state, wp.vec2i(world, contact),
      wp.vec2(
          state.delta_weight[world, contact],
          state.delta_free[world, contact]))


@cache_kernel
def _update_system(nv: int, nv_pad: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, element = wp.tid()
    if state.done[world] or state.changed_count[world] == 0:
      return
    if element < wp.static(nv):
      _update_rhs(state, wp.vec2i(world, element))
    _update_matrix(state, wp.vec2i(world, element), wp.static(nv))

  return kernel


@cache_kernel
def _refactor_or_solve(block_size: int, nv_pad: int):
  factorize_solve = create_blocked_cholesky_factorize_solve_func(
      block_size, nv_pad)
  solve = create_blocked_upper_cholesky_solve_func(
      block_size, nv_pad)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, lane = wp.tid()
    if state.done[world]:
      return
    if state.changed_count[world] == 0:
      solve(
          state.factor[world], state.rhs[world], wp.static(nv_pad),
          state.solution[world])
    else:
      factorize_solve(
          state.matrix[world], state.rhs[world], wp.static(nv_pad),
          state.factor[world], state.solution[world])
    if lane == 0:
      state.solver_niter[world] += 1

  return kernel


@wp.struct
class DualCorrectionScratch:
  residual: wp.array2d(dtype=float)
  mass_response: wp.array2d(dtype=float)
  row_scratch: wp.array2d(dtype=float)
  dof_scratch: wp.array2d(dtype=float)
  correction: wp.array2d(dtype=float)


def _correction_scratch(d: Data) -> DualCorrectionScratch:
  """Bind correction scratch allocated before CUDA graph capture."""
  scratch = DualCorrectionScratch()
  for field, name in (
      ("residual", "dual_residual"),
      ("mass_response", "dual_mass_response"),
      ("row_scratch", "dual_row_scratch"),
      ("dof_scratch", "dual_dof_scratch"),
      ("correction", "dual_correction")):
    setattr(scratch, field, _required_efc_array(d, name))
  return scratch


def _required_efc_array(d: Data, name: str) -> wp.array:
  if not hasattr(d.efc, name):
    raise RuntimeError(
        f"missing preallocated constraint workspace d.efc.{name}")
  return getattr(d.efc, name)


def _dual_state_arrays(
    m: Model,
    d: Data,
    workspace: DenseCachedWorkspace,
) -> tuple[wp.array, wp.array, wp.array]:
  if m.is_sparse:
    return (
        _required_efc_array(d, "dual_x"),
        _required_efc_array(d, "dual_y"),
        _required_efc_array(d, "dual_g"),
    )
  return workspace.solution, workspace.rhs, d.efc.force


@cache_kernel
def _finalize(nv: int, row_budget: int, nv_pad: int, block_size: int,
              correct: bool):
  """Write contact and generalized forces, optionally reusing factors for residual correction."""
  solve_upper = create_blocked_upper_cholesky_solve_func(block_size, nv_pad)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState, scratch: DualCorrectionScratch):
    world, lane = wp.tid()
    active_rows = min(state.nefc[world], state.row_limit)

    # P1: λ = max(f0 − D·(J x), 0) · G
    for contact in range(lane, wp.static(row_budget), wp.block_dim()):
      value = float(0.0)
      if contact < active_rows:
        response = float(0.0)
        for dof in range(wp.static(nv)):
          response += (state.J[world, contact, dof]
                       * state.solution[world, dof])
        candidate = (state.free[world, contact]
                     - state.D[world, contact] * response)
        value = wp.max(candidate, 0.0)
        value *= wp.where(
            state.active_D[world, contact] != 0.0, 1.0, 0.0)
      state.force[world, contact] = value
    _block_sync()

    if wp.static(not correct):
      for dof in range(lane, wp.static(nv), wp.block_dim()):
        value = float(0.0)
        for contact in range(active_rows):
          value += (state.J[world, contact, dof]
                    * state.force[world, contact])
        state.qfrc_constraint[world, dof] = value
        state.qfrc_total[world, dof] = state.qfrc_smooth[world, dof] + value
      return


    for dof in range(lane, wp.static(nv_pad), wp.block_dim()):
      value = float(0.0)
      if dof < wp.static(nv):
        for contact in range(active_rows):
          value += (state.J[world, contact, dof]
                    * state.force[world, contact])
        for column in range(wp.static(nv)):
          value -= state.qM[world, dof, column] * state.solution[world, column]
      scratch.residual[world, dof] = value
    _block_sync()

    wp.tile_store(
        scratch.mass_response[world],
        wp.tile_cholesky_solve(
            wp.tile_load(state.qLD[world],
                         shape=(wp.static(nv), wp.static(nv))),
            wp.tile_load(scratch.residual[world], shape=wp.static(nv))))
    _block_sync()


    for contact in range(lane, active_rows, wp.block_dim()):
      value = float(0.0)
      if state.active_D[world, contact] != 0.0:
        for dof in range(wp.static(nv)):
          value += (state.J[world, contact, dof]
                    * scratch.mass_response[world, dof])
        value *= state.D[world, contact]
      scratch.row_scratch[world, contact] = value
    _block_sync()

    # P4: J^T g
    for dof in range(lane, wp.static(nv_pad), wp.block_dim()):
      value = float(0.0)
      if dof < wp.static(nv):
        for contact in range(active_rows):
          value += (state.J[world, contact, dof]
                    * scratch.row_scratch[world, contact])
      scratch.dof_scratch[world, dof] = value
    _block_sync()

    solve_upper(
        state.factor[world], scratch.dof_scratch[world], wp.static(nv_pad),
        scratch.correction[world])
    _block_sync()


    for contact in range(lane, active_rows, wp.block_dim()):
      response = float(0.0)
      for dof in range(wp.static(nv)):
        response += (state.J[world, contact, dof]
                     * scratch.correction[world, dof])
      phi = (scratch.row_scratch[world, contact]
             - state.D[world, contact] * response)
      updated = wp.max(state.force[world, contact] - phi, 0.0)
      state.force[world, contact] = updated * wp.where(
          state.active_D[world, contact] != 0.0, 1.0, 0.0)
    _block_sync()

    for dof in range(lane, wp.static(nv), wp.block_dim()):
      value = float(0.0)
      for contact in range(active_rows):
        value += (state.J[world, contact, dof]
                  * state.force[world, contact])
      state.qfrc_constraint[world, dof] = value
      state.qfrc_total[world, dof] = state.qfrc_smooth[world, dof] + value

  return kernel


@wp.kernel(enable_backward=False)
def _project_sparse_forces(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  candidate = (
      state.free[world, contact]
      - state.D[world, contact]
      * _sparse_row_response(state, wp.vec2i(world, contact)))
  active = wp.where(
      state.active_D[world, contact] != 0.0, 1.0, 0.0)
  state.force[world, contact] = wp.max(candidate, 0.0) * active


@wp.kernel(enable_backward=False)
def _clear_sparse_generalized(state: DenseCachedState):
  world, dof = wp.tid()
  state.qfrc_constraint[world, dof] = 0.0


@wp.kernel(enable_backward=False)
def _scatter_sparse_forces(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  force = state.force[world, contact]
  address = state.J_rowadr[world, contact]
  count = state.J_rownnz[world, contact]
  for position in range(address, address + count):
    dof = state.J_colind[world, 0, position]
    wp.atomic_add(
        state.qfrc_constraint, world, dof,
        state.J[world, 0, position] * force)


@wp.kernel(enable_backward=False)
def _finish_sparse_generalized(state: DenseCachedState):
  world, dof = wp.tid()
  state.qfrc_total[world, dof] = (
      state.qfrc_smooth[world, dof]
      + state.qfrc_constraint[world, dof])


@wp.kernel(enable_backward=False)
def _sparse_dual_residual(state: DenseCachedState):
  world, dof = wp.tid()
  mass_response = float(0.0)
  for entry in range(
      state.qM_rowadr[dof], state.qM_rowadr[dof + 1]):
    column = state.qM_col[entry]
    mass_response += (
        state.qM[world, 0, state.qM_madr[entry]]
        * state.solution[world, column])
  state.dual_x[world, dof] = (
      state.qfrc_constraint[world, dof] - mass_response)


@wp.kernel(enable_backward=False)
def _sparse_dual_rows(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  value = float(0.0)
  if state.active_D[world, contact] != 0.0:
    address = state.J_rowadr[world, contact]
    count = state.J_rownnz[world, contact]
    for position in range(address, address + count):
      dof = state.J_colind[world, 0, position]
      value += state.J[world, 0, position] * state.dual_y[world, dof]
    value *= state.D[world, contact]
  state.dual_g[world, contact] = value


@wp.kernel(enable_backward=False)
def _clear_sparse_dual_rhs(state: DenseCachedState):
  world, dof = wp.tid()
  state.dual_x[world, dof] = 0.0


@wp.kernel(enable_backward=False)
def _scatter_sparse_dual_rhs(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  address = state.J_rowadr[world, contact]
  count = state.J_rownnz[world, contact]
  for position in range(address, address + count):
    dof = state.J_colind[world, 0, position]
    wp.atomic_add(
        state.dual_x, world, dof,
        state.J[world, 0, position] * state.dual_g[world, contact])


@cache_kernel
def _solve_sparse_dual(block_size: int, nv_pad: int):
  solve = create_blocked_upper_cholesky_solve_func(block_size, nv_pad)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world, lane = wp.tid()
    solve(
        state.factor[world], state.dual_x[world], wp.static(nv_pad),
        state.dual_y[world])

  return kernel


@cache_kernel
def _solve_sparse_dual_fused(nv_pad: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(state: DenseCachedState):
    world = wp.tid()
    factor = wp.tile_load(
        state.factor[world], shape=(wp.static(nv_pad), wp.static(nv_pad)))
    rhs = wp.tile_load(state.dual_x[world], shape=wp.static(nv_pad))
    wp.tile_store(
        state.dual_y[world], wp.tile_cholesky_solve(factor, rhs))

  return kernel


@wp.kernel(enable_backward=False)
def _apply_sparse_dual(state: DenseCachedState):
  world, contact = wp.tid()
  if contact >= state.nefc[world] or contact >= state.row_limit:
    return
  if state.active_D[world, contact] == 0.0:
    state.force[world, contact] = 0.0
    return
  response = float(0.0)
  address = state.J_rowadr[world, contact]
  count = state.J_rownnz[world, contact]
  for position in range(address, address + count):
    dof = state.J_colind[world, 0, position]
    response += state.J[world, 0, position] * state.dual_y[world, dof]
  phi = state.dual_g[world, contact] - state.D[world, contact] * response
  state.force[world, contact] = wp.max(
      state.force[world, contact] - phi, 0.0)


def workspace_from_data(d: Data) -> DenseCachedWorkspace:
  """Binds the production workspace allocated by ``make_data``."""
  return DenseCachedWorkspace(
      matrix=d.efc.contact_matrix,
      factor=d.efc.contact_factor,
      rhs=d.efc.contact_rhs,
      solution=d.efc.contact_solution,
      changed_rows=d.efc.contact_changed_rows,
      changed_count=d.efc.contact_active_changed,
      done=d.efc.contact_active_done,
      delta_weight=d.efc.contact_delta_weight,
      delta_free=d.efc.contact_delta_free,
      row_response=d.efc.contact_row_response,
  )


def solve_dense_cached_contacts(
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
    config: DenseCachedConfig,
) -> None:
  """Runs the cached blocked dense path with production-owned storage."""
  resolved = _resolve_config(config, m.nv_pad)
  solver = DenseCachedSolver(
      config=resolved,
      workspace=None if resolved.iterations == 0 else workspace_from_data(d),
  )
  solver.solve(m, d, rows=rows)


def _resolve_config(
    config: DenseCachedConfig,
    nv_pad: int,
) -> DenseCachedConfig:
  if config.block_size != AUTO_BLOCK_SIZE:
    return config
  block_size = next(
      size for size in BLOCK_SIZE_CANDIDATES if nv_pad % size == 0)
  return replace(config, block_size=block_size)


def _validate_inputs(
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
    config: DenseCachedConfig,
) -> None:
  if m.is_sparse != d.is_sparse:
    raise ValueError("model and data Jacobian layouts must match")
  if config.iterations < 0:
    raise ValueError("iterations must be non-negative")
  if config.block_size <= 0 or config.block_dim <= 0:
    raise ValueError("block_size and block_dim must be positive")
  if m.nv_pad % config.block_size:
    raise ValueError(
        f"nv_pad={m.nv_pad} must be divisible by block_size={config.block_size}")


def _triangle_size(nv_pad: int) -> int:
  return nv_pad * (nv_pad + 1) // 2


def _solve_direct(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  if solver.config.iterations == 0:
    diagonal_contact.solve_fused_dense_contacts(m, d, rows)
    return
  from . import support
  diagonal_contact.seed_fused_dense_contacts(m, d, rows)
  wp.launch_tiled(
      support.solve_contact_forces(
          m.nv, m.nv_pad, rows.tile_size, rows.budget,
          solver.config.iterations, solver.config.early_stop,
          support.DUAL_CORRECTION),
      dim=d.nworld,
      inputs=[
          d.qM, d.qLD, d.efc.J, d.efc.contact_free_force, d.efc.D,
          d.efc.aref, d.qacc_smooth, d.qfrc_smooth, d.nefc, rows.limit,
      ],
      outputs=[
          d.efc.contact_active_D, d.efc.contact_solution, d.efc.force,
          d.qfrc_constraint, d.qfrc_total, d.qacc, d.solver_niter,
      ],
      block_dim=ROW_BLOCK_DIM,
      device=d.qvel.device)


def _solver_state(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> DenseCachedState:
  workspace = solver.workspace
  if workspace is None:
    raise RuntimeError("blocked dense workspace is not allocated")
  state = DenseCachedState()
  state.nv = m.nv
  state.qM = d.qM
  state.qM_rowadr = m.qM_mulm_rowadr
  state.qM_col = m.qM_mulm_col
  state.qM_madr = m.qM_mulm_madr
  state.qLD = d.qLD
  state.J = d.efc.J
  state.J_rownnz = d.efc.J_rownnz
  state.J_rowadr = d.efc.J_rowadr
  state.J_colind = d.efc.J_colind
  state.free = d.efc.contact_free_force
  state.D = d.efc.D
  state.active_D = d.efc.contact_active_D
  state.nefc = d.nefc
  state.row_limit = rows.limit
  state.matrix = workspace.matrix
  state.factor = workspace.factor
  state.rhs = workspace.rhs
  state.solution = workspace.solution
  state.changed_rows = workspace.changed_rows
  state.changed_count = workspace.changed_count
  state.done = workspace.done
  state.delta_weight = workspace.delta_weight
  state.delta_free = workspace.delta_free
  state.row_response = workspace.row_response
  state.solver_niter = d.solver_niter
  state.qfrc_smooth = d.qfrc_smooth
  state.force = d.efc.force
  state.qfrc_constraint = d.qfrc_constraint
  state.qfrc_total = d.qfrc_total
  # Dense kernels do not access these sparse-only struct members.
  state.dual_x, state.dual_y, state.dual_g = _dual_state_arrays(
      m, d, workspace)
  return state


def _build_and_factor(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch_tiled(
      _build_system(m.nv, m.nv_pad, rows.tile_size, rows.budget),
      dim=d.nworld,
      inputs=[state],
      block_dim=solver.config.block_dim,
      device=d.qvel.device)
  wp.launch_tiled(
      _factorize_solve(solver.config.block_size, m.nv_pad),
      dim=d.nworld,
      inputs=[state],
      block_dim=solver.config.block_dim,
      device=d.qvel.device)


def _update_and_solve(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch_tiled(
      _update_active(
          m.nv, m.nv_pad, rows.budget, rows.tile_size,
          solver.config.early_stop),
      dim=d.nworld,
      inputs=[state],
      block_dim=ROW_BLOCK_DIM,
      device=d.qvel.device)
  wp.launch(
      _update_system(m.nv, m.nv_pad),
      dim=(d.nworld, _triangle_size(m.nv_pad)),
      inputs=[state],
      device=d.qvel.device)
  wp.launch_tiled(
      _refactor_or_solve(solver.config.block_size, m.nv_pad),
      dim=d.nworld,
      inputs=[state],
      block_dim=solver.config.block_dim,
      device=d.qvel.device)


def _write_solution(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch_tiled(
      _finalize(m.nv, rows.budget, m.nv_pad, solver.config.block_size,
                DUAL_CORRECTION),
      dim=d.nworld,
      inputs=[state, _correction_scratch(d)],
      block_dim=ROW_BLOCK_DIM,
      device=d.qvel.device)


def _solve_sparse(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  diagonal_contact.solve_sparse_diagonal_contacts(
      m, d, rows, block_dim=ROW_BLOCK_DIM)
  if solver.config.iterations == 0:
    return
  _build_sparse_and_factor(solver, m, d, rows=rows)
  for _ in range(1, solver.config.iterations):
    _update_sparse_and_solve(solver, m, d, rows=rows)
  _write_sparse_solution(solver, m, d, rows=rows)
  if solver.config.finalize_acceleration:
    smooth.solve_m(m, d, d.qacc, d.qfrc_total)


def _build_sparse_and_factor(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch(
      _initialize_sparse_system,
      dim=(d.nworld, _triangle_size(m.nv_pad)),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _accumulate_sparse_system,
      dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  wp.launch_tiled(
      _factorize_solve(solver.config.block_size, m.nv_pad),
      dim=d.nworld, inputs=[state],
      block_dim=solver.config.block_dim, device=d.qvel.device)


def _update_sparse_and_solve(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch(
      _clear_sparse_changes, dim=d.nworld,
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _update_sparse_active, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _finish_sparse_active, dim=d.nworld,
      inputs=[state, solver.config.early_stop], device=d.qvel.device)
  wp.launch(
      _accumulate_sparse_updates, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  wp.launch_tiled(
      _refactor_or_solve(solver.config.block_size, m.nv_pad),
      dim=d.nworld, inputs=[state],
      block_dim=solver.config.block_dim, device=d.qvel.device)


def _scatter_sparse_generalized(
    state: DenseCachedState,
    d: Data,
    rows: ConstraintRows,
) -> None:
  wp.launch(
      _clear_sparse_generalized, dim=(d.nworld, state.nv),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _scatter_sparse_forces, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _finish_sparse_generalized, dim=(d.nworld, state.nv),
      inputs=[state], device=d.qvel.device)


def _correct_sparse_solution(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch(
      _sparse_dual_residual, dim=(d.nworld, m.nv),
      inputs=[state], device=d.qvel.device)
  smooth.solve_m(m, d, state.dual_y, state.dual_x)
  wp.launch(
      _sparse_dual_rows, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _clear_sparse_dual_rhs, dim=(d.nworld, m.nv_pad),
      inputs=[state], device=d.qvel.device)
  wp.launch(
      _scatter_sparse_dual_rhs, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)

  if m.nv_pad <= DIRECT_NV_LIMIT:
    wp.launch_tiled(
        _solve_sparse_dual_fused(m.nv_pad),
        dim=d.nworld, inputs=[state],
        block_dim=solver.config.block_dim, device=d.qvel.device)
  else:
    wp.launch_tiled(
        _solve_sparse_dual(solver.config.block_size, m.nv_pad),
        dim=d.nworld, inputs=[state],
        block_dim=solver.config.block_dim, device=d.qvel.device)
  wp.launch(
      _apply_sparse_dual, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  _scatter_sparse_generalized(state, d, rows)


def _write_sparse_solution(
    solver: DenseCachedSolver,
    m: Model,
    d: Data,
    *,
    rows: ConstraintRows,
) -> None:
  state = _solver_state(solver, m, d, rows=rows)
  wp.launch(
      _project_sparse_forces, dim=(d.nworld, rows.budget),
      inputs=[state], device=d.qvel.device)
  _scatter_sparse_generalized(state, d, rows)
  if DUAL_CORRECTION:
    _correct_sparse_solution(solver, m, d, rows=rows)
