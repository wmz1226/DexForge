"""Sparse constraint-row preparation for the ComFree contact solve."""

from __future__ import annotations

import warp as wp


@wp.struct
class SparsePrepareModel:
  nv: int


@wp.struct
class SparsePrepareState:
  rownnz: wp.array2d(dtype=int)
  rowadr: wp.array2d(dtype=int)
  colind: wp.array3d(dtype=int)
  jacobian: wp.array3d(dtype=float)
  D: wp.array2d(dtype=float)
  aref: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  nefc: wp.array(dtype=int)
  row_limit: int
  nnz_capacity: int


@wp.struct
class SparsePrepareOutput:
  free_force: wp.array2d(dtype=float)
  weighted_jacobian: wp.array3d(dtype=float)
  active_D: wp.array2d(dtype=float)
  force: wp.array2d(dtype=float)
  overflow: wp.array(dtype=int)


@wp.kernel(enable_backward=False)
def _prepare_sparse_rows(model: SparsePrepareModel, state: SparsePrepareState,
                         output: SparsePrepareOutput):
  world, row = wp.tid()
  if row >= state.nefc[world] or row >= state.row_limit:
    return
  address = state.rowadr[world, row]
  nonzeros = state.rownnz[world, row]
  if address < 0 or nonzeros < 0 or address + nonzeros > state.nnz_capacity:
    output.overflow[world] = 1
    output.free_force[world, row] = wp.nan
    return
  z = -state.aref[world, row]
  for offset in range(nonzeros):
    sparse_index = address + offset
    dof = state.colind[world, 0, sparse_index]
    if dof < 0 or dof >= model.nv:
      output.overflow[world] = 1
      output.free_force[world, row] = wp.nan
      return
    jacobian = state.jacobian[world, 0, sparse_index]
    output.weighted_jacobian[world, 0, sparse_index] = jacobian
    z += jacobian * state.qacc_smooth[world, dof]
  output.free_force[world, row] = -state.D[world, row] * z
  output.active_D[world, row] = 0.0
  output.force[world, row] = 0.0


def prepare_sparse_rows(model, data, rows, *, aref) -> None:
  prepare_model = SparsePrepareModel()
  prepare_model.nv = model.nv
  state = _prepare_state(data, rows, aref)
  output = _prepare_output(data)
  wp.launch(
    _prepare_sparse_rows,
    dim=(data.nworld, rows.budget),
    inputs=[prepare_model, state, output],
  )


def _prepare_state(data, rows, aref) -> SparsePrepareState:
  state = SparsePrepareState()
  state.rownnz = data.efc.J_rownnz
  state.rowadr = data.efc.J_rowadr
  state.colind = data.efc.J_colind
  state.jacobian = data.efc.J
  state.D = data.efc.D
  state.aref = aref
  state.qacc_smooth = data.qacc_smooth
  state.nefc = data.nefc
  state.row_limit = rows.limit
  state.nnz_capacity = data.njmax_nnz
  return state


def _prepare_output(data) -> SparsePrepareOutput:
  output = SparsePrepareOutput()
  output.free_force = data.efc.contact_free_force
  output.weighted_jacobian = data.efc.weighted_J
  output.active_D = data.efc.contact_active_D
  output.force = data.efc.force
  output.overflow = data.efc.J_overflow
  return output
