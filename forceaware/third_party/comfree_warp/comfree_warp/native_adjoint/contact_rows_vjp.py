"""Deterministic reverse pass for the contact-row Jacobian.

``contact_rows._build_jacobian`` is launched over ``(world, row, dof)`` but
reads data indexed by contact, joint and body.  Warp's generated adjoint
therefore returns those gradients with ``atomic_add``, and float addition is
not associative, so the summation order and with it the gradient change from
run to run.  Over a long backpropagation the difference is amplified by the
same factor as any other perturbation.

This module inverts the same expression by gathering instead: every kernel
owns one input element and loops over the ``(row, dof)`` pairs that read it,
in a fixed order.  The forward program is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from .contact_rows import ContactMotion
from .contact_rows import ContactParameters
from .contact_rows import FixedContacts
from .contact_rows import PointDofQuery
from .contact_rows import _angular_velocity
from .contact_rows import _frame_row
from .contact_rows import _matrix_axis
from .contact_rows import _penetrating
from .contact_rows import _point_velocity


FREE = int(mujoco.mjtJoint.mjJNT_FREE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


@wp.struct
class RowVjpLayout:
  contact_row_start: wp.array(dtype=int)
  contact_row_count: wp.array(dtype=int)
  joint_dof_start: wp.array(dtype=int)
  joint_dof_count: wp.array(dtype=int)
  body_joint_start: wp.array(dtype=int)
  body_joint_count: wp.array(dtype=int)
  body_joint_index: wp.array(dtype=int)
  contact_count: int
  joint_count: int
  body_count: int
  dof_count: int


@wp.struct
class RowVjpJob:
  params: ContactParameters
  layout: RowVjpLayout
  contacts: FixedContacts
  motion: ContactMotion
  jacobian_grad: wp.array3d(dtype=float)
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
class CompiledContactRowsVjp:
  layout: RowVjpLayout
  device: object


@wp.func
def _edge_scale(params: ContactParameters, world: int, contact: int,
                edge: int) -> float:
  """Signed friction weight of a pyramidal cone edge."""
  axis = edge // 2 + 1
  sign = wp.where(edge % 2 == 0, 1.0, -1.0)
  friction = params.contact_friction[contact][axis - 1]
  friction *= params.friction_scale[world % params.friction_scale.shape[0]]
  return sign * friction


@wp.func
def _linear_seed(contacts: FixedContacts, params: ContactParameters,
                 world: int, contact: int, edge: int,
                 gradient: float) -> wp.vec3:
  """``d(row jacobian)/d(relative linear velocity)`` times ``gradient``."""
  frame = contacts.frame[world, contact]
  axis = edge // 2 + 1
  seed = _frame_row(frame, 0)
  if axis < 3:
    seed += _edge_scale(params, world, contact, edge) * _frame_row(frame, axis)
  return gradient * seed


@wp.func
def _angular_seed(contacts: FixedContacts, params: ContactParameters,
                  world: int, contact: int, edge: int,
                  gradient: float) -> wp.vec3:
  """``d(row jacobian)/d(relative angular velocity)`` times ``gradient``."""
  axis = edge // 2 + 1
  if axis < 3:
    return wp.vec3(0.0)
  frame = contacts.frame[world, contact]
  scale = _edge_scale(params, world, contact, edge)
  return (gradient * scale) * _frame_row(frame, axis - 3)


@wp.func
def _query(point: wp.vec3, body: int, dof: int, world: int) -> PointDofQuery:
  query = PointDofQuery()
  query.point = point
  query.body = body
  query.dof = dof
  query.world = world
  return query


@wp.func
def _lever(params: ContactParameters, motion: ContactMotion,
           query: PointDofQuery) -> wp.vec3:
  """``point - anchor`` of the joint driving ``query.dof``."""
  joint = params.dof_jntid[query.dof]
  return query.point - motion.joint_anchor[query.world, joint]


@wp.func
def _point_axis_seed(params: ContactParameters, motion: ContactMotion,
                     query: PointDofQuery, seed: wp.vec3) -> wp.vec3:
  """``d(point velocity)/d(joint axis)`` contracted with ``seed``."""
  if params.body_dof_mask[query.body, query.dof] == 0:
    return wp.vec3(0.0)
  joint = params.dof_jntid[query.dof]
  joint_type = params.jnt_type[joint]
  offset = query.dof - params.jnt_dofadr[joint]
  if joint_type == SLIDE:
    return seed
  if joint_type == FREE and offset < 3:
    return wp.vec3(0.0)
  return wp.cross(_lever(params, motion, query), seed)


@wp.func
def _point_lever_seed(params: ContactParameters, motion: ContactMotion,
                      query: PointDofQuery, seed: wp.vec3) -> wp.vec3:
  """``d(point velocity)/d(point - anchor)`` contracted with ``seed``."""
  if params.body_dof_mask[query.body, query.dof] == 0:
    return wp.vec3(0.0)
  joint = params.dof_jntid[query.dof]
  joint_type = params.jnt_type[joint]
  offset = query.dof - params.jnt_dofadr[joint]
  if joint_type == SLIDE:
    return wp.vec3(0.0)
  if joint_type == FREE and offset < 3:
    return wp.vec3(0.0)
  if joint_type == HINGE:
    axis = motion.joint_axis[query.world, joint]
  else:
    joint_body = params.jnt_bodyid[joint]
    axis = _matrix_axis(
        motion.body_matrix[query.world, joint_body],
        offset - wp.where(joint_type == FREE, 3, 0))
  return wp.cross(seed, axis)


@wp.func
def _angular_axis_seed(params: ContactParameters, query: PointDofQuery,
                       seed: wp.vec3) -> wp.vec3:
  """``d(angular velocity)/d(joint axis)`` contracted with ``seed``."""
  if params.body_dof_mask[query.body, query.dof] == 0:
    return wp.vec3(0.0)
  joint = params.dof_jntid[query.dof]
  joint_type = params.jnt_type[joint]
  offset = query.dof - params.jnt_dofadr[joint]
  if joint_type == SLIDE or (joint_type == FREE and offset < 3):
    return wp.vec3(0.0)
  return seed


@wp.func
def _uses_body_matrix(params: ContactParameters, query: PointDofQuery) -> bool:
  if params.body_dof_mask[query.body, query.dof] == 0:
    return False
  joint = params.dof_jntid[query.dof]
  joint_type = params.jnt_type[joint]
  offset = query.dof - params.jnt_dofadr[joint]
  if joint_type == SLIDE or joint_type == HINGE:
    return False
  return not (joint_type == FREE and offset < 3)


@wp.func
def _uses_joint_axis(params: ContactParameters, query: PointDofQuery) -> bool:
  if params.body_dof_mask[query.body, query.dof] == 0:
    return False
  joint = params.dof_jntid[query.dof]
  joint_type = params.jnt_type[joint]
  return joint_type == SLIDE or joint_type == HINGE


@wp.func
def _contact_bodies(params: ContactParameters, contact: int) -> wp.vec2i:
  geoms = params.contact_geom[contact]
  return wp.vec2i(params.geom_bodyid[geoms[0]], params.geom_bodyid[geoms[1]])


@wp.kernel(enable_backward=False)
def _frame_position_vjp(job: RowVjpJob):
  """Gradient of the contact frame, contact point and signed distance.

  One thread owns one contact and walks its own rows and every degree of
  freedom, so nothing is scattered.
  """
  world, contact = wp.tid()
  params = job.params
  start = job.layout.contact_row_start[contact]
  count = job.layout.contact_row_count[contact]
  distance = float(0.0)
  for index in range(count):
    distance += job.position_grad[world, start + index]
  job.distance_grad[world, contact] += distance
  if not _penetrating(job.contacts, params, world, contact):
    return
  bodies = _contact_bodies(params, contact)
  point = job.contacts.position[world, contact]
  row0 = wp.vec3(0.0)
  row1 = wp.vec3(0.0)
  row2 = wp.vec3(0.0)
  lever = wp.vec3(0.0)
  for index in range(count):
    row = start + index
    edge = params.row_edge[row]
    axis = edge // 2 + 1
    scale = _edge_scale(params, world, contact, edge)
    for dof in range(job.layout.dof_count):
      gradient = job.jacobian_grad[world, row, dof]
      if gradient == 0.0:
        continue
      query1 = _query(point, bodies[0], dof, world)
      query2 = _query(point, bodies[1], dof, world)
      linear = (_point_velocity(params, job.motion, query2)
                - _point_velocity(params, job.motion, query1))
      row0 += gradient * linear
      if axis == 1:
        row1 += (gradient * scale) * linear
      elif axis == 2:
        row2 += (gradient * scale) * linear
      else:
        angular = (_angular_velocity(params, job.motion, query2)
                   - _angular_velocity(params, job.motion, query1))
        weighted = (gradient * scale) * angular
        if axis == 3:
          row0 += weighted
        elif axis == 4:
          row1 += weighted
        else:
          row2 += weighted
      seed = _linear_seed(job.contacts, params, world, contact, edge, gradient)
      lever += (_point_lever_seed(params, job.motion, query2, seed)
                + _point_lever_seed(params, job.motion, query1, -seed))
  job.frame_grad[world, contact] += wp.mat33(
      row0[0], row0[1], row0[2],
      row1[0], row1[1], row1[2],
      row2[0], row2[1], row2[2])
  job.point_grad[world, contact] += lever


@wp.kernel(enable_backward=False)
def _joint_motion_vjp(job: RowVjpJob):
  """Gradient of the joint axis and joint anchor.

  One thread owns one joint; every degree of freedom maps to exactly one
  joint, so the loop below visits each ``(row, dof)`` contribution once.
  """
  world, joint = wp.tid()
  params = job.params
  dof_start = job.layout.joint_dof_start[joint]
  dof_count = job.layout.joint_dof_count[joint]
  axis_gradient = wp.vec3(0.0)
  anchor_gradient = wp.vec3(0.0)
  for contact in range(job.layout.contact_count):
    if not _penetrating(job.contacts, params, world, contact):
      continue
    bodies = _contact_bodies(params, contact)
    point = job.contacts.position[world, contact]
    start = job.layout.contact_row_start[contact]
    count = job.layout.contact_row_count[contact]
    for index in range(count):
      row = start + index
      edge = params.row_edge[row]
      for offset in range(dof_count):
        dof = dof_start + offset
        gradient = job.jacobian_grad[world, row, dof]
        if gradient == 0.0:
          continue
        query1 = _query(point, bodies[0], dof, world)
        query2 = _query(point, bodies[1], dof, world)
        linear = _linear_seed(
            job.contacts, params, world, contact, edge, gradient)
        angular = _angular_seed(
            job.contacts, params, world, contact, edge, gradient)
        anchor_gradient -= (
            _point_lever_seed(params, job.motion, query2, linear)
            + _point_lever_seed(params, job.motion, query1, -linear))
        if _uses_joint_axis(params, query2):
          axis_gradient += _point_axis_seed(params, job.motion, query2, linear)
          axis_gradient += _angular_axis_seed(params, query2, angular)
        if _uses_joint_axis(params, query1):
          axis_gradient += _point_axis_seed(params, job.motion, query1, -linear)
          axis_gradient += _angular_axis_seed(params, query1, -angular)
  job.joint_axis_grad[world, joint] += axis_gradient
  job.joint_anchor_grad[world, joint] += anchor_gradient


@wp.kernel(enable_backward=False)
def _body_matrix_vjp(job: RowVjpJob):
  """Gradient of the body orientation used as a ball or free rotation axis."""
  world, body = wp.tid()
  params = job.params
  joint_start = job.layout.body_joint_start[body]
  joint_count = job.layout.body_joint_count[body]
  column0 = wp.vec3(0.0)
  column1 = wp.vec3(0.0)
  column2 = wp.vec3(0.0)
  for slot in range(joint_count):
    joint = job.layout.body_joint_index[joint_start + slot]
    joint_type = params.jnt_type[joint]
    if joint_type == SLIDE or joint_type == HINGE:
      continue
    dof_start = job.layout.joint_dof_start[joint]
    dof_count = job.layout.joint_dof_count[joint]
    for offset in range(dof_count):
      dof = dof_start + offset
      column = offset - wp.where(joint_type == FREE, 3, 0)
      if column < 0:
        continue
      for contact in range(job.layout.contact_count):
        if not _penetrating(job.contacts, params, world, contact):
          continue
        bodies = _contact_bodies(params, contact)
        point = job.contacts.position[world, contact]
        start = job.layout.contact_row_start[contact]
        count = job.layout.contact_row_count[contact]
        for index in range(count):
          row = start + index
          edge = params.row_edge[row]
          gradient = job.jacobian_grad[world, row, dof]
          if gradient == 0.0:
            continue
          query1 = _query(point, bodies[0], dof, world)
          query2 = _query(point, bodies[1], dof, world)
          linear = _linear_seed(
              job.contacts, params, world, contact, edge, gradient)
          angular = _angular_seed(
              job.contacts, params, world, contact, edge, gradient)
          seed = wp.vec3(0.0)
          if _uses_body_matrix(params, query2):
            seed += _point_axis_seed(params, job.motion, query2, linear)
            seed += _angular_axis_seed(params, query2, angular)
          if _uses_body_matrix(params, query1):
            seed += _point_axis_seed(params, job.motion, query1, -linear)
            seed += _angular_axis_seed(params, query1, -angular)
          if column == 0:
            column0 += seed
          elif column == 1:
            column1 += seed
          else:
            column2 += seed
  job.body_matrix_grad[world, body] += wp.mat33(
      column0[0], column1[0], column2[0],
      column0[1], column1[1], column2[1],
      column0[2], column1[2], column2[2])


@wp.kernel(enable_backward=False)
def _limit_position_vjp(job: RowVjpJob):
  """Gradient of a joint-limit row position with respect to ``qpos``.

  Each limited joint owns one row and one position address, so the write is
  disjoint by construction.
  """
  world, limit = wp.tid()
  params = job.params
  joint = params.limit_joint[limit]
  address = params.jnt_qposadr[joint]
  interval = params.jnt_range[world % params.jnt_range.shape[0], joint]
  value = job.qpos[world, address]
  row = params.contact_row_count + limit
  lower = value - interval[0]
  upper = interval[1] - value
  slope = wp.where(lower < upper, 1.0, -1.0)
  job.qpos_grad[world, address] += slope * job.position_grad[world, row]


def _joint_dof_counts(cpu_model: mujoco.MjModel) -> np.ndarray:
  sizes = {FREE: 6, BALL: 3, SLIDE: 1, HINGE: 1}
  return np.asarray(
      [sizes[int(value)] for value in cpu_model.jnt_type], np.int32)


def _body_joint_index(cpu_model: mujoco.MjModel):
  order = np.argsort(np.asarray(cpu_model.jnt_bodyid, np.int32),
                     kind="stable").astype(np.int32)
  counts = np.bincount(
      np.asarray(cpu_model.jnt_bodyid, np.int32),
      minlength=cpu_model.nbody).astype(np.int32)
  starts = np.concatenate(
      [np.zeros(1, np.int32), np.cumsum(counts)[:-1].astype(np.int32)])
  return starts, counts, order


def compile_contact_rows_vjp(cpu_model: mujoco.MjModel, rows
                             ) -> CompiledContactRowsVjp:
  """Row, joint and body index tables that let the reverse pass gather."""
  device = rows.device
  row_contact = rows.parameters.row_contact.numpy()
  contact_count = int(rows.contact_count)
  counts = np.bincount(row_contact, minlength=contact_count).astype(np.int32)
  starts = np.concatenate(
      [np.zeros(1, np.int32), np.cumsum(counts)[:-1].astype(np.int32)])
  expected = np.repeat(np.arange(contact_count, dtype=np.int32), counts)
  if not np.array_equal(expected, row_contact):
    raise ValueError(
        "contact rows must be contiguous per contact for the gathered VJP")
  body_starts, body_counts, body_order = _body_joint_index(cpu_model)
  layout = RowVjpLayout()
  layout.contact_row_start = wp.array(starts, dtype=int, device=device)
  layout.contact_row_count = wp.array(counts, dtype=int, device=device)
  layout.joint_dof_start = wp.array(
      np.asarray(cpu_model.jnt_dofadr, np.int32), dtype=int, device=device)
  layout.joint_dof_count = wp.array(
      _joint_dof_counts(cpu_model), dtype=int, device=device)
  layout.body_joint_start = wp.array(body_starts, dtype=int, device=device)
  layout.body_joint_count = wp.array(body_counts, dtype=int, device=device)
  layout.body_joint_index = wp.array(body_order, dtype=int, device=device)
  layout.contact_count = contact_count
  layout.joint_count = int(cpu_model.njnt)
  layout.body_count = int(cpu_model.nbody)
  layout.dof_count = int(cpu_model.nv)
  return CompiledContactRowsVjp(layout, device)


def backward(compiled: CompiledContactRowsVjp, params: ContactParameters,
             contacts: FixedContacts, motion: ContactMotion, rows,
             qpos) -> None:
  """Accumulate the contact-row Jacobian gradient into its inputs.

  Reads ``rows.jacobian.grad`` and ``rows.position.grad``; adds into the
  gradients of the contact frame, contact point, signed distance, joint axis,
  joint anchor and body orientation.
  """
  job = RowVjpJob()
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
  wp.launch(_frame_position_vjp, dim=(worlds, compiled.layout.contact_count),
            inputs=[job], device=compiled.device)
  wp.launch(_joint_motion_vjp, dim=(worlds, compiled.layout.joint_count),
            inputs=[job], device=compiled.device)
  wp.launch(_body_matrix_vjp, dim=(worlds, compiled.layout.body_count),
            inputs=[job], device=compiled.device)
  if params.limit_count:
    wp.launch(_limit_position_vjp, dim=(worlds, params.limit_count),
              inputs=[job], device=compiled.device)
