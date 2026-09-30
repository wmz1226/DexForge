"""Deterministic reverse pass for static-CSR contact rows.

The sparse row builder launches one thread per CSR entry.  Warp's generated
adjoint therefore scatters gradients from many entries into shared contact,
joint, and body inputs.  This module groups CSR entries by their gradient
owner at compile time and gathers each owner's contributions in a fixed order.
The sparse forward layout and values are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from .contact_rows import ContactMotion
from .contact_rows import ContactParameters
from .contact_rows import FixedContacts
from .contact_rows import _angular_velocity
from .contact_rows import _basis
from .contact_rows import _penetrating
from .contact_rows import _point_velocity
from .contact_rows_vjp import _angular_axis_seed
from .contact_rows_vjp import _angular_seed
from .contact_rows_vjp import _contact_bodies
from .contact_rows_vjp import _edge_scale
from .contact_rows_vjp import _linear_seed
from .contact_rows_vjp import _point_axis_seed
from .contact_rows_vjp import _point_lever_seed
from .contact_rows_vjp import _query
from .contact_rows_vjp import _uses_body_matrix
from .contact_rows_vjp import _uses_joint_axis


FREE = int(mujoco.mjtJoint.mjJNT_FREE)
EDGES_PER_FRICTION_AXIS = 2
FIRST_TANGENT_AXIS = 1
SECOND_TANGENT_AXIS = 2
TORSIONAL_AXIS = 3
FIRST_ROLLING_AXIS = 4
FREE_ROTATION_OFFSET = 3
ROTATION_AXIS_COUNT = 3


@wp.struct
class SparseRowVjpLayout:
  contact_row_start: wp.array(dtype=int)
  contact_row_count: wp.array(dtype=int)
  row_offsets: wp.array(dtype=int)
  column_indices: wp.array(dtype=int)
  entry_rows: wp.array(dtype=int)
  joint_entry_start: wp.array(dtype=int)
  joint_entry_count: wp.array(dtype=int)
  joint_entries: wp.array(dtype=int)
  body_entry_start: wp.array(dtype=int)
  body_entry_count: wp.array(dtype=int)
  body_entries: wp.array(dtype=int)
  contact_count: int
  joint_count: int
  body_count: int


@wp.struct
class SparseRowVjpJob:
  params: ContactParameters
  layout: SparseRowVjpLayout
  contacts: FixedContacts
  motion: ContactMotion
  jacobian_grad: wp.array2d(dtype=float)
  position_grad: wp.array2d(dtype=float)
  frame_grad: wp.array2d(dtype=wp.mat33)
  point_grad: wp.array2d(dtype=wp.vec3)
  distance_grad: wp.array2d(dtype=float)
  joint_axis_grad: wp.array2d(dtype=wp.vec3)
  joint_anchor_grad: wp.array2d(dtype=wp.vec3)
  body_matrix_grad: wp.array2d(dtype=wp.mat33)
  qpos: wp.array2d(dtype=float)
  qpos_grad: wp.array2d(dtype=float)


@dataclass(frozen=True)
class CompiledSparseContactRowsVjp:
  layout: SparseRowVjpLayout
  device: object


@wp.func
def _frame_point_seed(job: SparseRowVjpJob, world: int, contact: int,
                      edge: int, dof: int, gradient: float) -> wp.mat44:
  """Frame rows and contact-point seed packed into a four-row matrix."""
  params = job.params
  bodies = _contact_bodies(params, contact)
  point = job.contacts.position[world, contact]
  query1 = _query(point, bodies[0], dof, world)
  query2 = _query(point, bodies[1], dof, world)
  linear = (_point_velocity(params, job.motion, query2)
            - _point_velocity(params, job.motion, query1))
  row0 = gradient * linear
  row1 = wp.vec3(0.0)
  row2 = wp.vec3(0.0)
  axis = edge // EDGES_PER_FRICTION_AXIS + FIRST_TANGENT_AXIS
  scale = _edge_scale(params, world, contact, edge)
  if axis == FIRST_TANGENT_AXIS:
    row1 = (gradient * scale) * linear
  elif axis == SECOND_TANGENT_AXIS:
    row2 = (gradient * scale) * linear
  else:
    angular = (_angular_velocity(params, job.motion, query2)
               - _angular_velocity(params, job.motion, query1))
    weighted = (gradient * scale) * angular
    if axis == TORSIONAL_AXIS:
      row0 += weighted
    elif axis == FIRST_ROLLING_AXIS:
      row1 += weighted
    else:
      row2 += weighted
  linear_seed = _linear_seed(
      job.contacts, params, world, contact, edge, gradient)
  point_seed = (
      _point_lever_seed(params, job.motion, query2, linear_seed)
      + _point_lever_seed(params, job.motion, query1, -linear_seed))
  return wp.mat44(
      row0[0], row0[1], row0[2], 0.0,
      row1[0], row1[1], row1[2], 0.0,
      row2[0], row2[1], row2[2], 0.0,
      point_seed[0], point_seed[1], point_seed[2], 0.0)


@wp.kernel(enable_backward=False)
def _frame_position_vjp(job: SparseRowVjpJob):
  """Gather one contact's frame, point, and signed-distance gradient."""
  world, contact = wp.tid()
  row_start = job.layout.contact_row_start[contact]
  row_count = job.layout.contact_row_count[contact]
  distance = float(0.0)
  for row_offset in range(row_count):
    distance += job.position_grad[world, row_start + row_offset]
  job.distance_grad[world, contact] += distance
  if not _penetrating(job.contacts, job.params, world, contact):
    return
  total = wp.mat44(0.0)
  for row_offset in range(row_count):
    row = row_start + row_offset
    edge = job.params.row_edge[row]
    entry_start = job.layout.row_offsets[row]
    entry_end = job.layout.row_offsets[row + 1]
    for entry in range(entry_start, entry_end):
      gradient = job.jacobian_grad[world, entry]
      if gradient == 0.0:
        continue
      dof = job.layout.column_indices[entry]
      total += _frame_point_seed(
          job, world, contact, edge, dof, gradient)
  job.frame_grad[world, contact] += wp.mat33(
      total[0, 0], total[0, 1], total[0, 2],
      total[1, 0], total[1, 1], total[1, 2],
      total[2, 0], total[2, 1], total[2, 2])
  job.point_grad[world, contact] += wp.vec3(
      total[3, 0], total[3, 1], total[3, 2])


@wp.func
def _joint_entry_seed(job: SparseRowVjpJob, world: int,
                      entry: int) -> wp.mat33:
  """Joint-axis and anchor seeds packed into the first two matrix rows."""
  row = job.layout.entry_rows[entry]
  contact = job.params.row_contact[row]
  if not _penetrating(job.contacts, job.params, world, contact):
    return wp.mat33(0.0)
  gradient = job.jacobian_grad[world, entry]
  if gradient == 0.0:
    return wp.mat33(0.0)
  dof = job.layout.column_indices[entry]
  edge = job.params.row_edge[row]
  bodies = _contact_bodies(job.params, contact)
  point = job.contacts.position[world, contact]
  query1 = _query(point, bodies[0], dof, world)
  query2 = _query(point, bodies[1], dof, world)
  linear = _linear_seed(
      job.contacts, job.params, world, contact, edge, gradient)
  angular = _angular_seed(
      job.contacts, job.params, world, contact, edge, gradient)
  anchor = -(
      _point_lever_seed(job.params, job.motion, query2, linear)
      + _point_lever_seed(job.params, job.motion, query1, -linear))
  axis = wp.vec3(0.0)
  if _uses_joint_axis(job.params, query2):
    axis += _point_axis_seed(job.params, job.motion, query2, linear)
    axis += _angular_axis_seed(job.params, query2, angular)
  if _uses_joint_axis(job.params, query1):
    axis += _point_axis_seed(job.params, job.motion, query1, -linear)
    axis += _angular_axis_seed(job.params, query1, -angular)
  return wp.mat33(
      axis[0], axis[1], axis[2],
      anchor[0], anchor[1], anchor[2],
      0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _joint_motion_vjp(job: SparseRowVjpJob):
  """Gather all CSR entries driven by one joint."""
  world, joint = wp.tid()
  start = job.layout.joint_entry_start[joint]
  count = job.layout.joint_entry_count[joint]
  total = wp.mat33(0.0)
  for offset in range(count):
    entry = job.layout.joint_entries[start + offset]
    total += _joint_entry_seed(job, world, entry)
  job.joint_axis_grad[world, joint] += wp.vec3(
      total[0, 0], total[0, 1], total[0, 2])
  job.joint_anchor_grad[world, joint] += wp.vec3(
      total[1, 0], total[1, 1], total[1, 2])


@wp.func
def _body_entry_seed(job: SparseRowVjpJob, world: int,
                     entry: int) -> wp.mat33:
  """Gradient matrix contributed by one CSR entry to its joint body."""
  row = job.layout.entry_rows[entry]
  contact = job.params.row_contact[row]
  if not _penetrating(job.contacts, job.params, world, contact):
    return wp.mat33(0.0)
  gradient = job.jacobian_grad[world, entry]
  if gradient == 0.0:
    return wp.mat33(0.0)
  dof = job.layout.column_indices[entry]
  joint = job.params.dof_jntid[dof]
  joint_type = job.params.jnt_type[joint]
  offset = dof - job.params.jnt_dofadr[joint]
  column = offset - wp.where(
      joint_type == FREE, FREE_ROTATION_OFFSET, 0)
  if column < 0 or column >= ROTATION_AXIS_COUNT:
    return wp.mat33(0.0)
  edge = job.params.row_edge[row]
  bodies = _contact_bodies(job.params, contact)
  point = job.contacts.position[world, contact]
  query1 = _query(point, bodies[0], dof, world)
  query2 = _query(point, bodies[1], dof, world)
  linear = _linear_seed(
      job.contacts, job.params, world, contact, edge, gradient)
  angular = _angular_seed(
      job.contacts, job.params, world, contact, edge, gradient)
  seed = wp.vec3(0.0)
  if _uses_body_matrix(job.params, query2):
    seed += _point_axis_seed(job.params, job.motion, query2, linear)
    seed += _angular_axis_seed(job.params, query2, angular)
  if _uses_body_matrix(job.params, query1):
    seed += _point_axis_seed(job.params, job.motion, query1, -linear)
    seed += _angular_axis_seed(job.params, query1, -angular)
  return wp.outer(seed, _basis(column))


@wp.kernel(enable_backward=False)
def _body_matrix_vjp(job: SparseRowVjpJob):
  """Gather all CSR entries whose rotational axis belongs to one body."""
  world, body = wp.tid()
  start = job.layout.body_entry_start[body]
  count = job.layout.body_entry_count[body]
  total = wp.mat33(0.0)
  for offset in range(count):
    entry = job.layout.body_entries[start + offset]
    total += _body_entry_seed(job, world, entry)
  job.body_matrix_grad[world, body] += total


@wp.kernel(enable_backward=False)
def _limit_position_vjp(job: SparseRowVjpJob):
  """Gather a scalar joint-limit position derivative into qpos."""
  world, limit = wp.tid()
  joint = job.params.limit_joint[limit]
  address = job.params.jnt_qposadr[joint]
  interval = job.params.jnt_range[
      world % job.params.jnt_range.shape[0], joint]
  value = job.qpos[world, address]
  row = job.params.contact_row_count + limit
  lower = value - interval[0]
  upper = interval[1] - value
  slope = wp.where(lower < upper, 1.0, -1.0)
  job.qpos_grad[world, address] += slope * job.position_grad[world, row]


def _starts(counts: np.ndarray) -> np.ndarray:
  result = np.zeros(counts.size, np.int32)
  if counts.size > 1:
    result[1:] = np.cumsum(counts[:-1], dtype=np.int32)
  return result


def _contact_ranges(rows) -> tuple[np.ndarray, np.ndarray]:
  row_contact = np.asarray(rows.parameters.row_contact.numpy(), np.int32)
  contact_count = int(rows.contact_count)
  counts = np.bincount(
      row_contact, minlength=contact_count).astype(np.int32)
  expected = np.repeat(np.arange(contact_count, dtype=np.int32), counts)
  if not np.array_equal(expected, row_contact):
    raise ValueError(
        "contact rows must be contiguous per contact for the gathered VJP")
  return _starts(counts), counts


def _group_entries(entries: np.ndarray, owners: np.ndarray,
                   owner_count: int) -> tuple[np.ndarray, ...]:
  if entries.shape != owners.shape:
    raise ValueError("CSR entries and owners must have the same shape")
  counts = np.bincount(owners, minlength=owner_count).astype(np.int32)
  order = np.argsort(owners, kind="stable")
  grouped = entries[order].astype(np.int32, copy=False)
  return _starts(counts), counts, grouped


def compile_sparse_contact_rows_vjp(
    cpu_model: mujoco.MjModel, rows) -> CompiledSparseContactRowsVjp:
  """Compile stable CSR-entry groups for contact, joint, and body gathers."""
  if not rows.sparse or rows.sparse_layout is None:
    raise ValueError("sparse contact-row VJP requires a static sparse layout")
  sparse = rows.sparse_layout
  entry_rows = np.asarray(sparse.entry_rows.numpy(), np.int32)
  columns = np.asarray(sparse.column_indices.numpy(), np.int32)
  if entry_rows.shape != columns.shape:
    raise ValueError("sparse row and column index arrays must have equal size")
  contact_entries = np.flatnonzero(
      entry_rows < rows.contact_row_count).astype(np.int32)
  contact_columns = columns[contact_entries]
  dof_joint = np.asarray(cpu_model.dof_jntid, np.int32)
  joint_body = np.asarray(cpu_model.jnt_bodyid, np.int32)
  entry_joints = dof_joint[contact_columns]
  joint_groups = _group_entries(
      contact_entries, entry_joints, int(cpu_model.njnt))
  body_groups = _group_entries(
      contact_entries, joint_body[entry_joints], int(cpu_model.nbody))
  contact_starts, contact_counts = _contact_ranges(rows)
  device = rows.device
  layout = SparseRowVjpLayout()
  layout.contact_row_start = wp.array(
      contact_starts, dtype=int, device=device)
  layout.contact_row_count = wp.array(
      contact_counts, dtype=int, device=device)
  layout.row_offsets = sparse.row_offsets
  layout.column_indices = sparse.column_indices
  layout.entry_rows = sparse.entry_rows
  names = ("entry_start", "entry_count", "entries")
  for prefix, group in (("joint", joint_groups), ("body", body_groups)):
    for suffix, values in zip(names, group, strict=True):
      setattr(layout, f"{prefix}_{suffix}", wp.array(
          values, dtype=int, device=device))
  layout.contact_count = int(rows.contact_count)
  layout.joint_count = int(cpu_model.njnt)
  layout.body_count = int(cpu_model.nbody)
  return CompiledSparseContactRowsVjp(layout, device)


def backward(compiled: CompiledSparseContactRowsVjp,
             params: ContactParameters, contacts: FixedContacts,
             motion: ContactMotion, rows, qpos) -> None:
  """Accumulate the static-CSR row VJP into geometry and motion inputs."""
  job = SparseRowVjpJob()
  job.params = params
  job.layout = compiled.layout
  job.contacts = contacts
  job.motion = motion
  job.jacobian_grad = rows.jacobian.grad
  job.position_grad = rows.position.grad
  job.frame_grad = contacts.frame.grad
  job.point_grad = contacts.position.grad
  job.distance_grad = contacts.distance.grad
  job.joint_axis_grad = motion.joint_axis.grad
  job.joint_anchor_grad = motion.joint_anchor.grad
  job.body_matrix_grad = motion.body_matrix.grad
  job.qpos = qpos
  job.qpos_grad = qpos.grad
  worlds = rows.jacobian.shape[0]
  wp.launch(_frame_position_vjp,
            dim=(worlds, compiled.layout.contact_count),
            inputs=[job], device=compiled.device)
  wp.launch(_joint_motion_vjp,
            dim=(worlds, compiled.layout.joint_count),
            inputs=[job], device=compiled.device)
  wp.launch(_body_matrix_vjp,
            dim=(worlds, compiled.layout.body_count),
            inputs=[job], device=compiled.device)
  if params.limit_count:
    wp.launch(_limit_position_vjp, dim=(worlds, params.limit_count),
              inputs=[job], device=compiled.device)
