# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
# Modified for this distribution: environment-variable prefix anonymized; numerical defaults unchanged.
"""Comfree engine API built on top of comfree_core patches."""

import os

import numpy as np
import warp as wp

from .comfree_core._src import dual_sparse_csc_solver
from .comfree_core._src import support as _support
from .comfree_core._src.forward import forward_comfree as forward
from .comfree_core._src.forward import step_comfree as step
from .comfree_core._src.forward import FULL_IMPLICIT_TILE_SIZE as CONSTRAINT_TILE_SIZE
from .comfree_core._src.types import Data as Data
from .comfree_core._src.types import Model as Model

from . import mujoco_warp as _mjwarp


SPARSE_SOLVER_ENV = "COMFREE_COMFREE_SPARSE_SOLVER"


def _round_up(value, multiple):
  return ((value + multiple - 1) // multiple) * multiple


def _constraint_budget(d, requested):
  capacity = int(d.njmax)
  padded_rows = int(d.njmax_pad)
  if requested is None:
    limit = capacity
  else:
    limit = int(requested)
    if limit <= 0:
      raise ValueError("comfree_constraint_budget must be positive")
    if limit > capacity:
      raise ValueError(f"comfree_constraint_budget={limit} exceeds allocated constraint rows={capacity}")

  budget = min(padded_rows, _round_up(limit, CONSTRAINT_TILE_SIZE))
  if budget < limit:
    raise ValueError(f"constraint budget={budget} is smaller than requested limit={limit}")
  return limit, budget


def _ensure_array(owner, name, shape, *, dtype=wp.float32, device):
  if not hasattr(owner, name):
    setattr(owner, name, wp.zeros(shape, dtype=dtype, device=device))
    return
  array = getattr(owner, name)
  if array.shape != shape:
    raise ValueError(
        f"{name} has shape {array.shape}, expected {shape}")
  if array.dtype != dtype:
    raise TypeError(
        f"{name} has dtype {array.dtype}, expected {dtype}")
  if array.device != device:
    raise ValueError(f"{name} and runtime data must share a device")


def _ensure_constraint_row_fields(d):
  shape = d.efc.pos.shape
  device = d.qvel.device
  names = (
    "efc_dist",
    "efc_mass",
    "efc_imp",
    "contact_free_force",
    "contact_stiffness",
    "contact_damping",
    "contact_active_D",
  )
  for name in names:
    _ensure_array(d.efc, name, shape, device=device)


def _ensure_jacobian_fields(d):
  _ensure_array(
      d.efc, "weighted_J", d.efc.J.shape, device=d.qvel.device)


def _ensure_sparse_dof_fields(d):
  vector_shape = (d.nworld, d.nv_pad)
  device = d.qvel.device
  for name in ("contact_dof_mask", "contact_dof_prefix", "contact_dof_ids"):
    _ensure_array(
        d.efc, name, vector_shape, dtype=wp.int32, device=device)
  _ensure_array(
      d.efc, "contact_dof_count", (d.nworld,),
      dtype=wp.int32, device=device)


def _ensure_correction_fields(d, *, dense: bool):
  """Allocate dual-residual correction scratch before CUDA graph capture."""
  dof_shape = (d.nworld, d.nv_pad)
  row_shape = d.efc.force.shape
  names = (
    (("dual_residual", dof_shape), ("dual_mass_response", dof_shape),
     ("dual_row_scratch", row_shape), ("dual_dof_scratch", dof_shape),
     ("dual_correction", dof_shape))
    if dense else
    (("dual_x", dof_shape), ("dual_y", dof_shape), ("dual_g", row_shape))
  )
  device = d.qvel.device
  for name, shape in names:
    _ensure_array(d.efc, name, shape, device=device)


def _ensure_csc_workspace_fields(d):
  entry_shape = (d.nworld, d.njmax_nnz)
  row_shape = (d.nworld, d.njmax)
  device = d.qvel.device
  _ensure_array(
      d.efc, dual_sparse_csc_solver.DUAL_ENTRY_POSITION_FIELD,
      entry_shape, dtype=wp.int32, device=device)
  _ensure_array(
      d.efc, dual_sparse_csc_solver.DUAL_ENTRY_ROW_FIELD,
      entry_shape, dtype=wp.int32, device=device)
  _ensure_array(
      d.efc, dual_sparse_csc_solver.DUAL_ROW_RESPONSE_FIELD,
      row_shape, device=device)


def _ensure_cholesky_fields(d):
  vector_shape = (d.nworld, d.nv_pad)
  row_shape = d.efc.pos.shape
  matrix_shape = (d.nworld, d.nv_pad, d.nv_pad)
  device = d.qvel.device
  for name in ("contact_matrix", "contact_factor"):
    _ensure_array(d.efc, name, matrix_shape, device=device)
  for name in ("contact_rhs", "contact_solution"):
    _ensure_array(d.efc, name, vector_shape, device=device)
  _ensure_array(
      d.efc, "contact_changed_rows", row_shape,
      dtype=wp.int32, device=device)
  for name in (
      "contact_delta_weight", "contact_delta_free", "contact_row_response"):
    _ensure_array(d.efc, name, row_shape, device=device)


def _ensure_pcg_fields(d):
  vector_shape = (d.nworld, d.nv_pad)
  device = d.qvel.device
  for name in (
    "contact_diag",
    "contact_rhs",
    "contact_solution",
    "contact_pcg_r",
    "contact_pcg_z",
    "contact_pcg_p",
    "contact_pcg_Ap",
  ):
    _ensure_array(d.efc, name, vector_shape, device=device)
  for name in ("contact_pcg_rz", "contact_pcg_rz_next"):
    _ensure_array(d.efc, name, (d.nworld,), device=device)


def _ensure_solver_fields(d, route, resolved_solver):
  """Allocate workspaces using the same solver route as execution."""
  device = d.qvel.device
  _ensure_array(
      d.efc, "contact_active_changed", (d.nworld,),
      dtype=wp.int32, device=device)
  _ensure_array(
      d.efc, "contact_active_done", (d.nworld,),
      dtype=wp.bool, device=device)

  factorizes = route in _support.CHOLESKY_ROUTES
  if factorizes:
    _ensure_cholesky_fields(d)


  if not factorizes or d.is_sparse:
    _ensure_pcg_fields(d)

  _ensure_correction_fields(d, dense=factorizes and not d.is_sparse)


  if (
      route == _support.ROUTE_PCG_SPARSE
      and resolved_solver == dual_sparse_csc_solver.SPARSE_SOLVER_DUAL
  ):
    _ensure_sparse_dof_fields(d)
    _ensure_csc_workspace_fields(d)


def _ensure_data_fields(d):
  device = d.qvel.device
  _ensure_array(d, "qvel_smooth_pred", d.qvel.shape, device=device)
  _ensure_array(d, "qfrc_total", d.qvel.shape, device=device)


def _sparse_solver_request(requested) -> str:
  value = os.environ.get(SPARSE_SOLVER_ENV, dual_sparse_csc_solver.SPARSE_SOLVER_AUTO)
  if requested is not None:
    value = requested
  return dual_sparse_csc_solver.validate_sparse_solver(value)


def _data_sparse_solver(d, requested: str) -> str:
  return dual_sparse_csc_solver.resolve_sparse_solver(
    requested,
    is_sparse=d.is_sparse,
    nworld=d.nworld,
    nv=d.qvel.shape[1],
  )


LINEAR_SOLVER_ENV = _support.LINEAR_SOLVER_ENV


def _linear_solver_request(requested):
  """Return an explicit linear-solver request, or None for the default."""
  value = requested if requested is not None else os.environ.get(
      LINEAR_SOLVER_ENV)
  return None if value is None else _support.validate_linear_solver(value)


def _data_route(d) -> str:
  return _support.solver_route(
    linear_solver=d.comfree_linear_solver,
    is_sparse=bool(d.is_sparse),
    nv_pad=int(d.nv_pad),
  )


def _ensure_comfree_fields(
    d,
    constraint_budget=None,
    sparse_solver=None,
    *,
    linear_solver=None,
):
  limit, budget = _constraint_budget(d, constraint_budget)


  requested_linear = _linear_solver_request(linear_solver)
  d.comfree_linear_solver = (
      requested_linear if requested_linear is not None
      else _support.DEFAULT_LINEAR_SOLVER)
  route = _data_route(d)
  d.comfree_linear_solver = _support.route_family(route)


  requested = _sparse_solver_request(sparse_solver)


  resolved = _data_sparse_solver(d, requested)
  resolved_solver = (
    resolved if route == _support.ROUTE_PCG_SPARSE
    else dual_sparse_csc_solver.SPARSE_SOLVER_GLOBAL
  )
  _ensure_constraint_row_fields(d)
  _ensure_jacobian_fields(d)
  _ensure_solver_fields(d, route, resolved_solver)
  _ensure_data_fields(d)

  d.comfree_constraint_limit = limit
  d.comfree_constraint_budget = budget
  d.comfree_sparse_solver = resolved_solver
  d.comfree_route = route
  return d


def put_model(*args, **kwargs):
  m = _mjwarp.put_model(*args, **kwargs)
  sqrt_invweight = np.sqrt(m.dof_invweight0.numpy())
  m.sqrt_dof_invweight0 = wp.array(
      sqrt_invweight, dtype=wp.float32, device=m.dof_invweight0.device)
  return m


def load_model(*args, **kwargs):
  """Loads standard MuJoCo XML plus optional gs/querypoint collision geoms."""
  from .gaussian_loader import load_model as load_gaussian_model
  return load_gaussian_model(*args, **kwargs)


def _gaussian_data_options(args, options, comfree_model, *, host_data):
  reserve = int(getattr(comfree_model, "gaussian_sparse_nnz_reserve", 0))
  if reserve == 0 or options.get("njmax_nnz") is not None:
    return options

  expected_args = 2 if host_data else 1
  if len(args) != expected_args:
    raise TypeError(
        "Gaussian-aware data allocation requires optional arguments by keyword")

  from .mujoco_warp._src import io
  mj_model = args[0]
  if not io.is_sparse(mj_model):
    return options
  mj_data = args[1] if host_data else None
  njmax = options.get("njmax")
  if njmax is None:
    njmax = io._default_njmax(mj_model, mj_data)
  standard = io._default_njmax_nnz(mj_model, int(njmax))
  capacity = min(int(njmax) * int(mj_model.nv), standard + reserve)
  return {**options, "njmax": int(njmax), "njmax_nnz": capacity}


def put_data(*args, **kwargs):
  options = dict(kwargs)
  constraint_budget = options.pop("comfree_constraint_budget", None)
  sparse_solver = _sparse_solver_request(options.pop("comfree_sparse_solver", None))
  linear_solver = options.pop("comfree_linear_solver", None)
  comfree_model = options.pop("comfree_model", None)
  options = _gaussian_data_options(
      args, options, comfree_model, host_data=True)
  d = _mjwarp.put_data(*args, **options)
  return _ensure_comfree_fields(
      d,
      constraint_budget,
      sparse_solver,
      linear_solver=linear_solver,
  )


def make_data(*args, **kwargs):
  options = dict(kwargs)
  constraint_budget = options.pop("comfree_constraint_budget", None)
  sparse_solver = _sparse_solver_request(options.pop("comfree_sparse_solver", None))
  linear_solver = options.pop("comfree_linear_solver", None)
  comfree_model = options.pop("comfree_model", None)
  options = _gaussian_data_options(
      args, options, comfree_model, host_data=False)
  d = _mjwarp.make_data(*args, **options)
  return _ensure_comfree_fields(
      d,
      constraint_budget,
      sparse_solver,
      linear_solver=linear_solver,
  )


def get_data_into(*args, **kwargs):
  return _mjwarp.get_data_into(*args, **kwargs)


def reset_data(*args, **kwargs):
  return _mjwarp.reset_data(*args, **kwargs)

__all__ = [
  "step",
  "forward",
  "load_model",
  "Model",
  "Data",
  "get_data_into",
  "make_data",
  "put_data",
  "put_model",
  "reset_data",
]
