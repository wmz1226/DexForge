# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
# Modified for this distribution: environment-variable prefix anonymized; numerical defaults unchanged.
# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
"""Support kernels for the full-implicit ComFree contact solve."""

import warp as wp

from comfree_warp.mujoco_warp._src.support import *  # noqa: F401,F403
from comfree_warp.mujoco_warp._src.warp_util import cache_kernel

from . import diagonal_contact
from . import dense_cached_solver
from . import dual_sparse_csc_solver
from . import fused_sparse_solver
from .config import env_bool
from .config import env_int
from .types import ConstraintRows
from .types import Data
from .types import DUAL_CORRECTION
from .types import Model
from .types import SolverType

wp.set_module_options({"enable_backward": False})

L_ENV = "COMFREE_FULL_IMPLICIT_ITERATIONS"
EARLY_STOP_ENV = "COMFREE_FULL_IMPLICIT_EARLY_STOP"
DEFAULT_L = 4
DEFAULT_FULL_IMPLICIT_PCG_ITERATIONS = 32


# L counts coupled corrections after the shared diagonal L0 initialization.
L = env_int(L_ENV, DEFAULT_L, minimum=0)
FULL_IMPLICIT_EARLY_STOP = env_bool(EARLY_STOP_ENV, True)


FULL_IMPLICIT_TILE_SIZE = env_int(
    "COMFREE_FULL_IMPLICIT_TILE_SIZE", 8, minimum=1)
FULL_IMPLICIT_DIRECT_NV_LIMIT = env_int(
    "COMFREE_FULL_IMPLICIT_DIRECT_NV_LIMIT", 32, minimum=1)
FULL_IMPLICIT_PCG_ITERATIONS = env_int(
    "COMFREE_FULL_IMPLICIT_PCG_ITERATIONS",
    DEFAULT_FULL_IMPLICIT_PCG_ITERATIONS,
    minimum=1)
FULL_IMPLICIT_SPARSE_BLOCK_DIM = env_int(
    "COMFREE_FULL_IMPLICIT_SPARSE_BLOCK_DIM", 128, minimum=1)
TILED_BLOCK_DIM = 64


def _constraint_tile_size(row_budget: int) -> int:
  for tile_size in (FULL_IMPLICIT_TILE_SIZE, 24, 16, 8, 4, 2, 1):
    if row_budget % tile_size == 0:
      return tile_size
  return 1


def _constraint_rows(d: Data) -> ConstraintRows:
  budget = d.comfree_constraint_budget
  tile_size = _constraint_tile_size(budget)
  return ConstraintRows(
    limit=d.comfree_constraint_limit,
    budget=budget,
    tile_size=tile_size,
    blocks=(budget + tile_size - 1) // tile_size,
  )


@wp.func
def _in_bounds(index: int, count: int) -> float:
  if index < count:
    return 1.0
  return 0.0


@wp.func
def _linear_force(free_force: float, row_weight: float, response: float) -> float:
  return free_force - row_weight * response


@wp.func
def _project_contact_force(force: float) -> float:
  return wp.max(force, 0.0)


@wp.func
def _active_contact(force: float) -> float:
  if force > 0.0:
    return 1.0
  return 0.0


@wp.func
def _changed_weight(current: float, previous: float) -> float:
  if current != previous:
    return 1.0
  return 0.0


@cache_kernel
def solve_contact_forces(
    nv: int,
    nv_pad: int,
    row_tile_size: int,
    row_budget: int,
    iterations: int,
    early_stop: bool,
    correct: bool,
):
  @wp.kernel(module="unique", enable_backward=False)
  def _solve_contact_forces(
    qM: wp.array3d(dtype=float),
    qLD: wp.array3d(dtype=float),
    J: wp.array3d(dtype=float),
    free_force: wp.array2d(dtype=float),
    D: wp.array2d(dtype=float),
    aref: wp.array2d(dtype=float),
    qacc_smooth: wp.array2d(dtype=float),
    qfrc_smooth: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int),
    row_limit: int,
    velocity_weight_out: wp.array2d(dtype=float),
    solution_cache: wp.array2d(dtype=float),
    efc_force: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
    qfrc_total: wp.array2d(dtype=float),
    qacc_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
  ):
    worldid, lane = wp.tid()
    NV_PAD = wp.static(nv_pad)
    ROW_TILE = wp.static(row_tile_size)
    ROW_BUDGET = wp.static(row_budget)
    if lane == 0:
      solver_niter_out[worldid] = 0

    if nefc[worldid] > row_limit:
      nan_vec = wp.tile_ones(shape=NV_PAD, dtype=wp.float32) * wp.nan
      wp.tile_store(qfrc_constraint[worldid], nan_vec, bounds_check=True)
      wp.tile_store(qfrc_total[worldid], nan_vec, bounds_check=True)
      wp.tile_store(qacc_out[worldid], nan_vec, bounds_check=True)
      return

    active_rows = min(nefc[worldid], row_limit)
    qfrc_smooth_tile = wp.tile_load(
        qfrc_smooth[worldid], shape=NV_PAD, bounds_check=True)
    qacc_smooth_tile = wp.tile_load(
        qacc_smooth[worldid], shape=NV_PAD, bounds_check=True)

    if active_rows == 0:
      zero_vec = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
      wp.tile_store(solution_cache[worldid], zero_vec, bounds_check=False)
      wp.tile_store(qfrc_constraint[worldid], zero_vec, bounds_check=True)
      wp.tile_store(qfrc_total[worldid], qfrc_smooth_tile, bounds_check=True)
      wp.tile_store(qacc_out[worldid], qacc_smooth_tile, bounds_check=True)
      return

    solution = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
    dof_ids = wp.tile_arange(NV_PAD, dtype=int)
    dof_bound = wp.tile_ones(shape=NV_PAD, dtype=int) * wp.static(nv)
    padding = (
        wp.tile_ones(shape=NV_PAD, dtype=wp.float32)
        - wp.tile_map(_in_bounds, dof_ids, dof_bound))
    matrix = wp.tile_load(
        qM[worldid], shape=(NV_PAD, NV_PAD), bounds_check=True)
    matrix = wp.tile_diag_add(matrix, padding)
    rhs = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)

    for row_start in range(0, ROW_BUDGET, ROW_TILE):
      if row_start >= active_rows:
        break
      x_rows = wp.tile_load(
          J[worldid], shape=(ROW_TILE, NV_PAD),
          offset=(row_start, 0), bounds_check=False)
      row_id = wp.tile_arange(ROW_TILE, dtype=int)
      bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (
          active_rows - row_start)
      mask = wp.tile_map(_in_bounds, row_id, bound)
      row_D = wp.tile_load(
          D[worldid], shape=ROW_TILE,
          offset=row_start, bounds_check=True) * mask
      row_aref = wp.tile_load(
          aref[worldid], shape=ROW_TILE,
          offset=row_start, bounds_check=True) * mask
      smooth_response = wp.tile_sum(
          wp.tile_map(
              wp.mul, x_rows,
              wp.tile_broadcast(
                  qacc_smooth_tile, shape=(ROW_TILE, NV_PAD))),
          axis=1)
      free_base = -row_D * (smooth_response - row_aref)
      active = wp.tile_map(_active_contact, free_base)
      velocity_weight = active * row_D
      wp.tile_store(
          free_force[worldid], free_base,
          offset=row_start, bounds_check=True)
      wp.tile_store(
          velocity_weight_out[worldid], velocity_weight,
          offset=row_start, bounds_check=True)
      matrix += wp.tile_matmul(
          wp.tile_map(
              wp.mul,
              wp.tile_transpose(x_rows),
              wp.tile_broadcast(velocity_weight, shape=(NV_PAD, ROW_TILE))),
          x_rows)
      rhs += wp.tile_sum(
          wp.tile_map(
              wp.mul,
              wp.tile_transpose(x_rows),
              wp.tile_broadcast(active * free_base, shape=(NV_PAD, ROW_TILE))),
          axis=1)

    # L0 provides the initial active set. Each positive L value adds one
    # coupled solve, followed by an active-set update only when another solve
    # remains in the budget.
    for iteration in range(iterations):
      factor = wp.tile_cholesky(matrix, fill_mode="upper")
      solution = wp.tile_cholesky_solve(
          factor, rhs, fill_mode="upper")
      if lane == 0:
        solver_niter_out[worldid] += 1
      if iteration + 1 >= iterations:
        break
      changed_count = float(0.0)
      for row_start in range(0, ROW_BUDGET, ROW_TILE):
        if row_start >= active_rows:
          break
        x_rows = wp.tile_load(
            J[worldid], shape=(ROW_TILE, NV_PAD),
            offset=(row_start, 0), bounds_check=False)
        free_base = wp.tile_load(free_force[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
        full_weight = wp.tile_load(D[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
        response = wp.tile_sum(wp.tile_map(wp.mul, x_rows, wp.tile_broadcast(solution, shape=(ROW_TILE, NV_PAD))), axis=1)
        candidate_force = wp.tile_map(_linear_force, free_base, full_weight, response)
        active = wp.tile_map(_active_contact, candidate_force)
        velocity_weight = active * full_weight
        row_id = wp.tile_arange(ROW_TILE, dtype=int)
        bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (active_rows - row_start)
        mask = wp.tile_map(_in_bounds, row_id, bound)
        velocity_weight *= mask
        previous_weight = wp.tile_load(velocity_weight_out[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
        changed = wp.tile_map(_changed_weight, velocity_weight, previous_weight) * mask
        tile_changed = wp.tile_sum(changed)[0]
        changed_count += tile_changed
        wp.tile_store(velocity_weight_out[worldid], velocity_weight, offset=row_start, bounds_check=True)

        if tile_changed != 0.0:
          previous_active = wp.tile_map(_active_contact, previous_weight) * mask
          delta_free = (active - previous_active) * free_base * mask
          delta_weight = (velocity_weight - previous_weight) * mask
          matrix += wp.tile_matmul(
            wp.tile_map(wp.mul, wp.tile_transpose(x_rows), wp.tile_broadcast(delta_weight, shape=(NV_PAD, ROW_TILE))),
            x_rows,
          )
          rhs += wp.tile_sum(
            wp.tile_map(wp.mul, wp.tile_transpose(x_rows), wp.tile_broadcast(delta_free, shape=(NV_PAD, ROW_TILE))),
            axis=1,
          )

      if wp.static(early_stop) and changed_count == 0.0:
        break

    projected_solution = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
    final_changed_count = float(0.0)
    for row_start in range(0, ROW_BUDGET, ROW_TILE):
      if row_start >= active_rows:
        break
      x_rows = wp.tile_load(
          J[worldid], shape=(ROW_TILE, NV_PAD),
          offset=(row_start, 0), bounds_check=False)
      row_weight = wp.tile_load(D[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
      active_weight = wp.tile_load(
          velocity_weight_out[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
      free = wp.tile_load(free_force[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
      row_id = wp.tile_arange(ROW_TILE, dtype=int)
      bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (active_rows - row_start)
      mask = wp.tile_map(_in_bounds, row_id, bound)
      free *= mask
      response = wp.tile_sum(wp.tile_map(wp.mul, x_rows, wp.tile_broadcast(solution, shape=(ROW_TILE, NV_PAD))), axis=1)
      active = wp.tile_map(_active_contact, active_weight)
      candidate_force = wp.tile_map(
          _linear_force, free, row_weight, response)
      candidate_active = wp.tile_map(_active_contact, candidate_force) * mask
      final_changed_count += wp.tile_sum(
          wp.tile_map(_changed_weight, candidate_active, active) * mask)[0]
      force = wp.tile_map(
          _project_contact_force, candidate_force) * active * mask
      projected_solution += wp.tile_sum(
        wp.tile_map(wp.mul, wp.tile_transpose(x_rows), wp.tile_broadcast(force, shape=(NV_PAD, ROW_TILE))),
        axis=1,
      )
      wp.tile_store(efc_force[worldid], force, offset=row_start, bounds_check=True)

    if wp.static(correct):


      mass = wp.tile_load(
          qM[worldid], shape=(NV_PAD, NV_PAD), bounds_check=True)
      mass_x = wp.tile_sum(wp.tile_map(
          wp.mul, mass,
          wp.tile_broadcast(solution, shape=(NV_PAD, NV_PAD))), axis=1)
      mass_chol = wp.tile_load(
          qLD[worldid], shape=(NV_PAD, NV_PAD), bounds_check=True)
      mass_chol = wp.tile_diag_add(mass_chol, padding)
      dual_y = wp.tile_cholesky_solve(
          mass_chol, projected_solution - mass_x)

      dual_jtg = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
      for row_start in range(0, ROW_BUDGET, ROW_TILE):
        if row_start >= active_rows:
          break
        x_rows = wp.tile_load(
            J[worldid], shape=(ROW_TILE, NV_PAD),
            offset=(row_start, 0), bounds_check=False)
        row_weight = wp.tile_load(
            D[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
        active_weight = wp.tile_load(
            velocity_weight_out[worldid], shape=ROW_TILE,
            offset=row_start, bounds_check=True)
        row_id = wp.tile_arange(ROW_TILE, dtype=int)
        bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (
            active_rows - row_start)
        keep = wp.tile_map(_active_contact, active_weight) * wp.tile_map(
            _in_bounds, row_id, bound)
        dual_g = row_weight * wp.tile_sum(wp.tile_map(
            wp.mul, x_rows,
            wp.tile_broadcast(dual_y, shape=(ROW_TILE, NV_PAD))), axis=1) * keep
        dual_jtg += wp.tile_sum(wp.tile_map(
            wp.mul, wp.tile_transpose(x_rows),
            wp.tile_broadcast(dual_g, shape=(NV_PAD, ROW_TILE))), axis=1)

      dual_u = wp.tile_cholesky_solve(
          factor, dual_jtg, fill_mode="upper")

      projected_solution = wp.tile_zeros(shape=NV_PAD, dtype=wp.float32)
      for row_start in range(0, ROW_BUDGET, ROW_TILE):
        if row_start >= active_rows:
          break
        x_rows = wp.tile_load(
            J[worldid], shape=(ROW_TILE, NV_PAD),
            offset=(row_start, 0), bounds_check=False)
        row_weight = wp.tile_load(
            D[worldid], shape=ROW_TILE, offset=row_start, bounds_check=True)
        active_weight = wp.tile_load(
            velocity_weight_out[worldid], shape=ROW_TILE,
            offset=row_start, bounds_check=True)
        row_id = wp.tile_arange(ROW_TILE, dtype=int)
        bound = wp.tile_ones(shape=ROW_TILE, dtype=int) * (
            active_rows - row_start)
        keep = wp.tile_map(_active_contact, active_weight) * wp.tile_map(
            _in_bounds, row_id, bound)
        dual_g = row_weight * wp.tile_sum(wp.tile_map(
            wp.mul, x_rows,
            wp.tile_broadcast(dual_y, shape=(ROW_TILE, NV_PAD))), axis=1) * keep
        dual_ju = row_weight * wp.tile_sum(wp.tile_map(
            wp.mul, x_rows,
            wp.tile_broadcast(dual_u, shape=(ROW_TILE, NV_PAD))), axis=1)
        force = wp.tile_load(
            efc_force[worldid], shape=ROW_TILE,
            offset=row_start, bounds_check=True) - (dual_g - dual_ju * keep)
        force = wp.tile_map(_project_contact_force, force) * keep
        wp.tile_store(
            efc_force[worldid], force, offset=row_start, bounds_check=True)
        projected_solution += wp.tile_sum(wp.tile_map(
            wp.mul, wp.tile_transpose(x_rows),
            wp.tile_broadcast(force, shape=(NV_PAD, ROW_TILE))), axis=1)

    qfrc_constraint_tile = projected_solution
    qfrc_total_tile = qfrc_smooth_tile + qfrc_constraint_tile
    # For an unchanged active set, the coupled solution is exactly M^-1 J^T f.
    # A final sign change requires the exact mass-factor backsubstitution.
    qacc_tile = qacc_smooth_tile + solution
    if wp.static(correct) or final_changed_count != 0.0:
      mass_factor = wp.tile_load(
          qLD[worldid], shape=(NV_PAD, NV_PAD), bounds_check=True)
      mass_factor = wp.tile_diag_add(mass_factor, padding)
      qacc_tile = wp.tile_cholesky_solve(mass_factor, qfrc_total_tile)
    wp.tile_store(solution_cache[worldid], solution, bounds_check=False)
    wp.tile_store(
        qfrc_constraint[worldid], qfrc_constraint_tile,
        bounds_check=True)
    wp.tile_store(qfrc_total[worldid], qfrc_total_tile, bounds_check=True)
    wp.tile_store(qacc_out[worldid], qacc_tile, bounds_check=True)

  return _solve_contact_forces


def _launch_force_solve_direct(m: Model, d: Data, rows: ConstraintRows):
  wp.launch_tiled(
      solve_contact_forces(
          m.nv, m.nv_pad, rows.tile_size, rows.budget, L,
          FULL_IMPLICIT_EARLY_STOP, DUAL_CORRECTION),
      dim=d.nworld,
      inputs=[
          d.qM, d.qLD, d.efc.J, d.efc.contact_free_force, d.efc.D,
          d.efc.aref, d.qacc_smooth, d.qfrc_smooth, d.nefc, rows.limit,
      ],
      outputs=[
          d.efc.contact_active_D, d.efc.contact_solution, d.efc.force,
          d.qfrc_constraint, d.qfrc_total, d.qacc, d.solver_niter,
      ],
      block_dim=TILED_BLOCK_DIM)


def _sparse_solver_config() -> fused_sparse_solver.FusedSparseSolverConfig:
  return fused_sparse_solver.FusedSparseSolverConfig(
    coupled_iterations=L,
    pcg_iterations=FULL_IMPLICIT_PCG_ITERATIONS,
    block_dim=FULL_IMPLICIT_SPARSE_BLOCK_DIM,
    early_stop=FULL_IMPLICIT_EARLY_STOP,
  )


def _dense_solver_config() -> dense_cached_solver.DenseCachedConfig:
  return dense_cached_solver.DenseCachedConfig(
    iterations=L,
    early_stop=FULL_IMPLICIT_EARLY_STOP,
    finalize_acceleration=False,
    block_dim=TILED_BLOCK_DIM,
  )


def _launch_sparse_force_solve(m: Model, d: Data, rows: ConstraintRows):
  config = _sparse_solver_config()
  strategy = d.comfree_sparse_solver
  if strategy == dual_sparse_csc_solver.SPARSE_SOLVER_GLOBAL:
    fused_sparse_solver.solve_sparse_contacts(m, d, rows, config=config)
    return
  if strategy == dual_sparse_csc_solver.SPARSE_SOLVER_DUAL:
    workspace = dual_sparse_csc_solver.workspace_from_data(d)
    dual_sparse_csc_solver.solve_dual_sparse_contacts(
      m, d, rows, workspace=workspace, config=config
    )
    return
  raise ValueError(f"unsupported fixed ComFree sparse solver: {strategy!r}")


def _launch_dense_pcg_solve(m: Model, d: Data, rows: ConstraintRows):
  if L == 0:
    dense_cached_solver.solve_dense_cached_contacts(
        m, d, rows=rows, config=_dense_solver_config())
    return
  diagonal_contact.seed_fused_dense_contacts(m, d, rows)
  fused_sparse_solver.solve_dense_contacts(
      m, d, rows, config=_sparse_solver_config())


#


LINEAR_SOLVER_CHOLESKY = "cholesky"
LINEAR_SOLVER_PCG = "pcg"            # matrix-free
VALID_LINEAR_SOLVERS = (LINEAR_SOLVER_CHOLESKY, LINEAR_SOLVER_PCG)
LINEAR_SOLVER_ENV = "COMFREE_COMFREE_LINEAR_SOLVER"


ROUTE_CHOLESKY_DENSE_FUSED = "cholesky_dense_fused"
ROUTE_CHOLESKY_DENSE_BLOCKED = "cholesky_dense_blocked"
ROUTE_CHOLESKY_SPARSE_FUSED = "cholesky_sparse_fused"
ROUTE_CHOLESKY_SPARSE_BLOCKED = "cholesky_sparse_blocked"
ROUTE_PCG_DENSE = "pcg_dense"
ROUTE_PCG_SPARSE = "pcg_sparse"

CHOLESKY_ROUTES = (
    ROUTE_CHOLESKY_DENSE_FUSED, ROUTE_CHOLESKY_DENSE_BLOCKED,
    ROUTE_CHOLESKY_SPARSE_FUSED, ROUTE_CHOLESKY_SPARSE_BLOCKED)
PCG_ROUTES = (ROUTE_PCG_DENSE, ROUTE_PCG_SPARSE)


def validate_linear_solver(value) -> str:
  if not isinstance(value, str) or value not in VALID_LINEAR_SOLVERS:
    choices = ", ".join(VALID_LINEAR_SOLVERS)
    raise ValueError(
        f"comfree_linear_solver must be one of: {choices}; got {value!r}")
  return value


#


DEFAULT_LINEAR_SOLVER = LINEAR_SOLVER_CHOLESKY


def solver_route(*, linear_solver: str, is_sparse: bool, nv_pad=None) -> str:
  """Resolve a linear algebra strategy and Jacobian layout to one solver route."""
  validate_linear_solver(linear_solver)
  if linear_solver == LINEAR_SOLVER_PCG:
    return ROUTE_PCG_SPARSE if is_sparse else ROUTE_PCG_DENSE
  if nv_pad is None:
    raise ValueError("nv_pad is required to choose a factorization route")
  direct_limit = min(
      FULL_IMPLICIT_DIRECT_NV_LIMIT, diagonal_contact.FUSED_DENSE_NV_LIMIT)
  fused = nv_pad <= direct_limit
  if is_sparse:
    return (ROUTE_CHOLESKY_SPARSE_FUSED if fused
            else ROUTE_CHOLESKY_SPARSE_BLOCKED)
  return ROUTE_CHOLESKY_DENSE_FUSED if fused else ROUTE_CHOLESKY_DENSE_BLOCKED


def route_family(route: str) -> str:
  """Return the linear algebra family for a resolved solver route."""
  if route in CHOLESKY_ROUTES:
    return LINEAR_SOLVER_CHOLESKY
  if route in PCG_ROUTES:
    return LINEAR_SOLVER_PCG
  raise ValueError(f"unknown ComFree solver route: {route!r}")


def model_route(m: Model, linear_solver: str = None) -> str:
  if linear_solver is None:
    linear_solver = DEFAULT_LINEAR_SOLVER
  return solver_route(
      linear_solver=linear_solver, is_sparse=bool(m.is_sparse),
      nv_pad=getattr(m, "nv_pad", None))


def _launch_force_solve(m: Model, d: Data, rows: ConstraintRows):
  route = model_route(m, getattr(d, "comfree_linear_solver", None))
  if route == ROUTE_CHOLESKY_DENSE_FUSED:
    if L == 0:
      diagonal_contact.solve_fused_dense_contacts(m, d, rows)
      return
    _launch_force_solve_direct(m, d, rows)
    return
  if route in CHOLESKY_ROUTES:


    dense_cached_solver.solve_dense_cached_contacts(
        m, d, rows=rows, config=_dense_solver_config())
    return
  if route == ROUTE_PCG_SPARSE:
    diagonal_contact.solve_sparse_diagonal_contacts(
      m, d, rows, block_dim=FULL_IMPLICIT_SPARSE_BLOCK_DIM)
    if L != 0:
      _launch_sparse_force_solve(m, d, rows)
    return
  if route == ROUTE_PCG_DENSE:
    _launch_dense_pcg_solve(m, d, rows)
    return
  raise ValueError(f"unhandled ComFree solver route: {route!r}")


def constraint_rows(d: Data) -> ConstraintRows:
  return _constraint_rows(d)


def uses_dense_direct_solver(m: Model, d: Data = None) -> bool:
  """Determine whether the fused solver writes forces and accelerations directly."""
  choice = getattr(d, "comfree_linear_solver", None)
  route = model_route(m, choice)
  if route == ROUTE_CHOLESKY_DENSE_FUSED:
    return True
  # Dense L0 always dispatches to solve_fused_dense_contacts(), including
  # nv_pad > 32 and PCG selections, so row preparation and solve_m are redundant.
  return L == 0 and route in (
      ROUTE_CHOLESKY_DENSE_BLOCKED, ROUTE_PCG_DENSE)


def requires_contact_row_preparation(m: Model, d: Data = None) -> bool:
  """Whether the selected solver consumes the prepared row workspace."""
  return not uses_dense_direct_solver(m, d)


def solve_fullimplicit_contacts(m: Model, d: Data, rows: ConstraintRows):
  _launch_force_solve(m, d, rows)
