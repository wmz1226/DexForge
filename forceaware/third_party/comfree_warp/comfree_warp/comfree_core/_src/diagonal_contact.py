# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
"""Direct Delassus-diagonal solve for zero iterations."""

import warp as wp

from comfree_warp.mujoco_warp._src.warp_util import cache_kernel

from .fused_sparse_solver import FusedSparseSolverState
from .fused_sparse_solver import MAX_BLOCK_DIM
from .fused_sparse_solver import _block_sync
from .fused_sparse_solver import _clear_outputs
from .fused_sparse_solver import _finalize_dofs
from .fused_sparse_solver import _make_state
from .fused_sparse_solver import _scatter_projected_solution
from .fused_sparse_solver import _validate_layout
from .fused_sparse_solver import _write_invalid
from .types import ConstraintRows
from .types import L0_IMPEDANCE
from .types import Data
from .types import Model

wp.set_module_options({"enable_backward": False})

DENSE_ROW_TILE_SIZE = 32
DENSE_BLOCK_DIM = 64
FUSED_DENSE_NV_LIMIT = 32


@wp.struct
class FusedDenseDiagonalJob:
  qLD: wp.array3d(dtype=float)
  jacobian: wp.array3d(dtype=float)
  D: wp.array2d(dtype=float)
  aref: wp.array2d(dtype=float)
  l0_aref: wp.array2d(dtype=float)
  timestep: wp.array(dtype=float)
  qvel: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  qfrc_smooth: wp.array2d(dtype=float)
  nefc: wp.array(dtype=int)
  row_limit: int
  free_force: wp.array2d(dtype=float)
  velocity_weight: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  solution: wp.array2d(dtype=float)
  qfrc_constraint: wp.array2d(dtype=float)
  qfrc_total: wp.array2d(dtype=float)
  qvel_smooth_pred: wp.array2d(dtype=float)
  qacc: wp.array2d(dtype=float)
  solver_niter: wp.array(dtype=int)


@wp.func
def _active(force: float) -> float:
  return wp.where(force > 0.0, 1.0, 0.0)


@cache_kernel
def _solve_fused_dense(
    nv: int,
    nv_pad: int,
    row_budget: int,
    row_tile_size: int,
    finalize_state: bool,
):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: FusedDenseDiagonalJob):
    world = wp.tid()
    NV_PAD = wp.static(nv_pad)
    ROW_TILE = wp.static(row_tile_size)
    active_rows = min(job.nefc[world], job.row_limit)
    factor = wp.tile_load(job.qLD[world], shape=(NV_PAD, NV_PAD), bounds_check=True)
    dof_ids = wp.tile_arange(NV_PAD, dtype=int)
    dof_bound = wp.tile_ones(shape=NV_PAD, dtype=int) * wp.static(nv)
    padding = (wp.tile_ones(shape=NV_PAD, dtype=wp.float32)
               - wp.tile_map(_in_bounds, dof_ids, dof_bound))
    factor = wp.tile_diag_add(factor, padding)
    qacc = wp.tile_load(job.qacc_smooth[world], shape=NV_PAD, bounds_check=True)
    qvel = wp.tile_load(job.qvel[world], shape=NV_PAD, bounds_check=True)
    timestep = job.timestep[world % job.timestep.shape[0]]
    wp.tile_store(
        job.qvel_smooth_pred[world], qvel + timestep * qacc,
        bounds_check=True)
    projected = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
    for start in range(0, wp.static(row_budget), ROW_TILE):
      if start >= active_rows:
        break
      jacobian_rows = wp.tile_load(
          job.jacobian[world], shape=(ROW_TILE, NV_PAD),
          offset=(start, 0), bounds_check=True)
      row_ids = wp.tile_arange(ROW_TILE, dtype=int)
      row_bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (active_rows - start)
      mask = wp.tile_map(_in_bounds, row_ids, row_bound)
      masked_jacobian = wp.tile_transpose(
          wp.tile_transpose(jacobian_rows)
          * wp.tile_broadcast(mask, shape=(NV_PAD, ROW_TILE)))
      D = wp.tile_load(job.D[world], shape=ROW_TILE, offset=start,
                       bounds_check=True) * mask
      if wp.static(finalize_state):
        aref = wp.tile_load(
            job.l0_aref[world], shape=ROW_TILE, offset=start,
            bounds_check=True) * mask
      else:
        aref = wp.tile_load(
            job.aref[world], shape=ROW_TILE, offset=start,
            bounds_check=True) * mask
      z = wp.tile_sum(masked_jacobian * wp.tile_broadcast(
          qacc, shape=(ROW_TILE, NV_PAD)), axis=1) - aref
      free = -D * z
      wp.tile_store(job.free_force[world], free, offset=start,
                    bounds_check=True)
      if wp.static(not finalize_state):
        wp.tile_store(
            job.velocity_weight[world], wp.tile_map(_active, free) * D,
            offset=start, bounds_check=True)
      if wp.static(finalize_state):

        raw = free * wp.static(L0_IMPEDANCE)
        wp.tile_store(
            job.velocity_weight[world], wp.tile_map(_active, raw) * D,
            offset=start, bounds_check=True)
        force = wp.tile_map(
            wp.max, raw, wp.tile_zeros(
                shape=ROW_TILE, dtype=wp.float32)) * mask
        projected += wp.tile_sum(
            wp.tile_transpose(masked_jacobian)
            * wp.tile_broadcast(force, shape=(NV_PAD, ROW_TILE)), axis=1)
        wp.tile_store(
            job.force[world], force, offset=start, bounds_check=True)
    if wp.static(not finalize_state):
      return
    if job.nefc[world] > job.row_limit:
      projected *= wp.nan
    acceleration = wp.tile_cholesky_solve(factor, projected)
    smooth = wp.tile_load(
        job.qfrc_smooth[world], shape=NV_PAD, bounds_check=True)
    total = smooth + projected
    wp.tile_store(job.solution[world], projected, bounds_check=True)
    wp.tile_store(job.qfrc_constraint[world], projected, bounds_check=True)
    wp.tile_store(job.qfrc_total[world], total, bounds_check=True)
    wp.tile_store(job.qacc[world], qacc + acceleration, bounds_check=True)
    job.solver_niter[world] = 0

  return kernel


@wp.func
def _in_bounds(index: int, count: int) -> float:
  return wp.where(index < count, 1.0, 0.0)


@cache_kernel
def _seed_fused_dense(
    nv_pad: int,
    row_budget: int,
    row_tile_size: int,
):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(job: FusedDenseDiagonalJob):
    world = wp.tid()
    NV_PAD = wp.static(nv_pad)
    ROW_TILE = wp.static(row_tile_size)
    active_rows = min(job.nefc[world], job.row_limit)
    timestep = job.timestep[world % job.timestep.shape[0]]
    qacc = wp.tile_load(
        job.qacc_smooth[world], shape=NV_PAD, bounds_check=True)
    qvel = wp.tile_load(job.qvel[world], shape=NV_PAD, bounds_check=True)
    wp.tile_store(
        job.qvel_smooth_pred[world], qvel + timestep * qacc,
        bounds_check=True)
    for start in range(0, wp.static(row_budget), ROW_TILE):
      if start >= active_rows:
        break
      jacobian = wp.tile_load(
          job.jacobian[world], shape=(ROW_TILE, NV_PAD),
          offset=(start, 0), bounds_check=True)
      row_ids = wp.tile_arange(ROW_TILE, dtype=int)
      row_bound = wp.tile_ones(
          shape=ROW_TILE, dtype=int) * (active_rows - start)
      mask = wp.tile_map(_in_bounds, row_ids, row_bound)
      D = wp.tile_load(
          job.D[world], shape=ROW_TILE, offset=start,
          bounds_check=True) * mask
      aref = wp.tile_load(
          job.aref[world], shape=ROW_TILE, offset=start,
          bounds_check=True) * mask
      response = wp.tile_sum(
          jacobian * wp.tile_broadcast(
              qacc, shape=(ROW_TILE, NV_PAD)), axis=1)
      free = -D * (response - aref)
      wp.tile_store(
          job.free_force[world], free, offset=start, bounds_check=True)
      wp.tile_store(
          job.velocity_weight[world], wp.tile_map(_active, free) * D,
          offset=start, bounds_check=True)

  return kernel


def solve_fused_dense_contacts(
    m: Model,
    d: Data,
    rows: ConstraintRows,
    *,
    row_tile_size: int = DENSE_ROW_TILE_SIZE,
) -> None:
  """Runs dense L0 and writes forces, total force, velocity, and acceleration."""
  job = _fused_dense_job(m, d, rows)
  wp.launch_tiled(
      _solve_fused_dense(
          m.nv, m.nv_pad, rows.budget, row_tile_size, True),
      dim=d.nworld,
      inputs=[job],
      block_dim=DENSE_BLOCK_DIM,
      device=d.qvel.device)


def seed_fused_dense_contacts(
    m: Model,
    d: Data,
    rows: ConstraintRows,
    *,
    row_tile_size: int = DENSE_ROW_TILE_SIZE,
) -> None:
  """Computes only the L0 free force and active diagonal for refinement."""
  job = _fused_dense_job(m, d, rows)
  wp.launch_tiled(
      _seed_fused_dense(m.nv_pad, rows.budget, row_tile_size),
      dim=d.nworld,
      inputs=[job],
      block_dim=DENSE_BLOCK_DIM,
      device=d.qvel.device)


def _fused_dense_job(
    m: Model, d: Data, rows: ConstraintRows) -> FusedDenseDiagonalJob:
  job = FusedDenseDiagonalJob()
  job.qLD = d.qLD
  job.jacobian = d.efc.J
  job.D = d.efc.D
  job.aref = d.efc.aref
  job.l0_aref = d.efc.efc_mass
  job.timestep = m.opt.timestep
  job.qvel = d.qvel
  job.qacc_smooth = d.qacc_smooth
  job.qfrc_smooth = d.qfrc_smooth
  job.nefc = d.nefc
  job.row_limit = rows.limit
  job.free_force = d.efc.contact_free_force
  job.velocity_weight = d.efc.contact_active_D
  job.force = d.efc.force
  job.solution = d.efc.contact_solution
  job.qfrc_constraint = d.qfrc_constraint
  job.qfrc_total = d.qfrc_total
  job.qvel_smooth_pred = d.qvel_smooth_pred
  job.qacc = d.qacc
  job.solver_niter = d.solver_niter
  return job


@wp.func
def _project_sparse_row(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  row = ids[1]
  D = state.D[world, row]
  raw_force = state.free_force[world, row] * wp.static(L0_IMPEDANCE)
  state.velocity_weight[world, row] = _active(raw_force) * D
  state.force[world, row] = wp.max(raw_force, 0.0)


@wp.func
def _project_sparse_rows(
    state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  for row in range(ids[1], state.nefc[world], wp.block_dim()):
    _project_sparse_row(state, wp.vec2i(world, row))


@wp.func
def _finalize_sparse_world(state: FusedSparseSolverState, ids: wp.vec2i):
  world = ids[0]
  if state.overflow[world] != 0:
    _write_invalid(state, ids)
    return
  if state.nefc[world] == 0:
    return
  _project_sparse_rows(state, ids)
  _block_sync()
  _scatter_projected_solution(state, ids)
  _finalize_dofs(state, ids)


@wp.kernel(enable_backward=False, module="unique")
def _prepare_sparse_kernel(
    state: FusedSparseSolverState,
):
  world, lane = wp.tid()
  ids = wp.vec2i(world, lane)
  _clear_outputs(state, ids)
  _block_sync()
  if state.overflow[world] != 0:
    _write_invalid(state, ids)
    return


@wp.kernel(enable_backward=False, module="unique")
def _finalize_sparse_kernel(state: FusedSparseSolverState):
  world, lane = wp.tid()
  _finalize_sparse_world(state, wp.vec2i(world, lane))


def solve_sparse_diagonal_contacts(
    m: Model, d: Data, rows: ConstraintRows, *, block_dim: int) -> None:
  """Runs direct-diagonal CSR L0."""
  if not m.is_sparse:
    raise ValueError("sparse diagonal solve requires model.is_sparse=True")
  if block_dim <= 0 or block_dim > MAX_BLOCK_DIM:
    raise ValueError(f"block_dim must be in [1, {MAX_BLOCK_DIM}]")
  _validate_layout(m, d, rows)
  state = _make_state(m, d, rows)
  wp.launch_tiled(
      _prepare_sparse_kernel,
      dim=[d.nworld],
      inputs=[state],
      block_dim=block_dim,
      device=d.qvel.device)
  wp.launch_tiled(
      _finalize_sparse_kernel,
      dim=[d.nworld],
      inputs=[state],
      block_dim=block_dim,
      device=d.qvel.device)
