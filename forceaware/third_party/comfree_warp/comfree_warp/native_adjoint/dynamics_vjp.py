"""Gathered adjoints for dynamics reductions with data-dependent loops."""

from __future__ import annotations

import warp as wp

from comfree_warp.mujoco_warp._src import math
from comfree_warp.mujoco_warp._src.types import vec10
from . import dynamics as dyn
from .geometry_vjp import GeometryTape


# A float64 scalar return avoids Warp 1.15's wp.grad float cast codegen bug.
@wp.func
def _inertia_cost(offset: wp.vec3, rotation: wp.mat33, diagonal: wp.vec3,
                  mass: float, seed: vec10) -> wp.float64:
  inertia = rotation @ wp.diag(diagonal) @ wp.transpose(rotation)
  return wp.float64(wp.dot(dyn._pack_inertia(
      dyn._parallel_axis(inertia, offset, mass), offset, mass), seed))


@wp.func
def _inert_vec_cost(inertia: vec10, motion: wp.spatial_vector,
                    seed: wp.spatial_vector) -> wp.float64:
  return wp.float64(wp.dot(math.inert_vec(inertia, motion), seed))


@wp.kernel(enable_backward=False)
def _inertia_center(job: dyn.BodyInertiaJob, adj: dyn.BodyInertiaJob):
  world, root = wp.tid()
  if job.model.body_rootid[root] != root:
    return
  row = world % job.model.body_mass.shape[0]
  value = wp.vec3(0.0)
  for body in range(job.model.body_count):
    if job.model.body_rootid[body] == root:
      offset = job.inertial_position[world, body] - job.subtree_com[world, root]
      go, gr, gi, gm, gs = wp.grad(_inertia_cost)(
          offset, job.inertial_matrix[world, body], job.model.body_inertia[row, body],
          job.model.body_mass[row, body], adj.body_inertia[world, body])
      value -= go
  adj.subtree_com[world, root] += value


@wp.kernel(enable_backward=False)
def _crb_inertia(job: dyn.CrbDofJob, adj: dyn.CrbDofJob):
  world, body = wp.tid()
  value = vec10(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for dof in range(job.model.dof_count):
    if job.model.dof_bodyid[dof] == body:
      gi, gv, gs = wp.grad(_inert_vec_cost)(job.composite_inertia[world, body],
          job.cdof[world, dof], adj.crb_cdof[world, dof])
      value += gi
  adj.composite_inertia[world, body] += value


@wp.kernel(enable_backward=False)
def _mass_motion(job: dyn.MassMatrixJob, adj: dyn.MassMatrixJob):
  world, dof = wp.tid()
  gc = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  gv = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for other in range(job.model.dof_count):
    first = job.model.mass_mask[dof, other]
    second = job.model.mass_mask[other, dof]
    if first == 1:
      gc += job.cdof[world, other] * adj.mass[world, dof, other]
    if second == 2:
      gc += job.cdof[world, other] * adj.mass[world, other, dof]
    if second == 1:
      gv += job.crb_cdof[world, other] * adj.mass[world, other, dof]
    if first == 2:
      gv += job.crb_cdof[world, other] * adj.mass[world, dof, other]
  if adj.crb_cdof:
    adj.crb_cdof[world, dof] += gc
  if adj.cdof:
    adj.cdof[world, dof] += gv


@wp.kernel(enable_backward=False)
def _mass_armature(job: dyn.MassMatrixJob, adj: dyn.MassMatrixJob):
  row, dof = wp.tid()
  value = float(0.0)
  for world in range(job.mass.shape[0]):
    if world % job.model.dof_armature.shape[0] == row:
      value += adj.mass[world, dof, dof]
  adj.model.dof_armature[row, dof] += value


@wp.func
def _motion_seed(job: dyn.MotionDofJob, adj: dyn.MotionDofJob,
                  world: int, dof: int):
  joint = job.model.dof_jntid[dof]
  body = job.model.dof_bodyid[dof]
  kind = job.model.jnt_type[joint]
  offset = dof - job.model.jnt_dofadr[joint]
  angular = wp.vec3(adj.cdof_scalar[world, dof, 0],
                    adj.cdof_scalar[world, dof, 1], adj.cdof_scalar[world, dof, 2])
  linear = wp.vec3(adj.cdof_scalar[world, dof, 3],
                   adj.cdof_scalar[world, dof, 4], adj.cdof_scalar[world, dof, 5])
  ga = wp.vec3(0.0)
  go = wp.vec3(0.0)
  if kind == dyn._SLIDE:
    ga = linear
  elif kind != dyn._FREE or offset >= 3:
    axis = job.joint_axis[world, joint]
    if kind == dyn._FREE:
      axis = dyn._matrix_axis(job.body_matrix[world, body], offset - 3)
    elif kind == dyn._BALL:
      axis = dyn._matrix_axis(job.body_matrix[world, body], offset)
    lever = (job.subtree_com[world, job.model.body_rootid[body]]
             - job.joint_anchor[world, joint])
    ga = angular + wp.cross(lever, linear)
    go = wp.cross(linear, axis)
  return ga, go


@wp.kernel(enable_backward=False)
def _motion_joint(job: dyn.MotionDofJob, adj: dyn.MotionDofJob):
  world, joint = wp.tid()
  anchor = wp.vec3(0.0)
  axis = wp.vec3(0.0)
  kind = job.model.jnt_type[joint]
  for dof in range(job.model.dof_count):
    if job.model.dof_jntid[dof] == joint:
      ga, go = _motion_seed(job, adj, world, dof)
      anchor -= go
      if kind == dyn._HINGE or kind == dyn._SLIDE:
        axis += ga
  if adj.joint_anchor:
    adj.joint_anchor[world, joint] += anchor
  if adj.joint_axis:
    adj.joint_axis[world, joint] += axis


@wp.kernel(enable_backward=False)
def _motion_body(job: dyn.MotionDofJob, adj: dyn.MotionDofJob):
  world, body = wp.tid()
  matrix = wp.mat33(0.0)
  center = wp.vec3(0.0)
  for dof in range(job.model.dof_count):
    owner = job.model.dof_bodyid[dof]
    joint = job.model.dof_jntid[dof]
    kind = job.model.jnt_type[joint]
    offset = dof - job.model.jnt_dofadr[joint]
    if owner == body or job.model.body_rootid[owner] == body:
      ga, go = _motion_seed(job, adj, world, dof)
      if job.model.body_rootid[owner] == body:
        center += go
      if owner == body:
        if kind == dyn._BALL:
          matrix += wp.outer(ga, dyn._basis(offset))
        elif kind == dyn._FREE and offset >= 3:
          matrix += wp.outer(ga, dyn._basis(offset - 3))
  if adj.body_matrix:
    adj.body_matrix[world, body] += matrix
  if adj.subtree_com:
    adj.subtree_com[world, body] += center


@wp.kernel(enable_backward=False)
def _bias_body(job: dyn.BiasForceJob, adj: dyn.BiasForceJob):
  world, body = wp.tid()
  value = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for dof in range(job.model.dof_count):
    if job.model.dof_bodyid[dof] == body:
      value += job.cdof[world, dof] * adj.bias_force[world, dof]
  adj.composite_force[world, body] += value


@wp.kernel(enable_backward=False)
def _applied_body(job: dyn.AppliedForceJob, adj: dyn.AppliedForceJob):
  world, body = wp.tid()
  value = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for dof in range(job.model.dof_count):
    if job.model.dof_bodyid[dof] == body:
      seed = adj.passive_force[world, dof] + adj.smooth_force[world, dof]
      value += job.cdof[world, dof] * seed
  adj.gravcomp_force[world, body] += value


@wp.kernel(enable_backward=False)
def _gravcomp_center(job: dyn.GravcompForceJob, adj: dyn.GravcompForceJob):
  world, root = wp.tid()
  if job.model.body_rootid[root] != root:
    return
  row = world % job.model.body_mass.shape[0]
  gravity = job.model.gravity[world % job.model.gravity.shape[0]]
  gravity *= float(job.model.gravity_enabled)
  value = wp.vec3(0.0)
  for body in range(job.model.body_count):
    if job.model.body_rootid[body] == root:
      linear = -gravity * (job.model.body_mass[row, body]
                           * job.model.body_gravcomp[row, body])
      seed = adj.local_force[world, body]
      value -= wp.cross(linear, wp.vec3(seed[0], seed[1], seed[2]))
  adj.subtree_com[world, root] += value


@wp.kernel(enable_backward=False)
def _force(model: dyn.DynamicsModel,
           seed: wp.array2d(dtype=wp.spatial_vector),
           gradient: wp.array2d(dtype=wp.spatial_vector)):
  world, child = wp.tid()
  value = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for body in range(model.body_count):
    if model.body_descendant[body, child] == 1:
      value += seed[world, body]
  gradient[world, child] += value


@wp.kernel(enable_backward=False)
def _inertia(model: dyn.DynamicsModel, seed: wp.array2d(dtype=vec10),
             gradient: wp.array2d(dtype=vec10)):
  world, child = wp.tid()
  value = vec10(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for body in range(model.body_count):
    if model.body_descendant[body, child] == 1:
      value += seed[world, body]
  gradient[world, child] += value


@wp.kernel(enable_backward=False)
def _velocity(job: dyn.MaskedVelocityJob, adj: dyn.MaskedVelocityJob):
  world, dof = wp.tid()
  seed = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for row in range(job.mask.shape[0]):
    if job.mask[row, dof] == 1:
      seed += adj.velocity[world, row]
  if adj.qvel:
    adj.qvel[world, dof] += wp.dot(seed, job.cdof[world, dof])
  if adj.cdof:
    adj.cdof[world, dof] += seed * job.qvel[world, dof]


@wp.kernel(enable_backward=False)
def _inverse_mass(model: dyn.DynamicsModel, result: wp.array2d(dtype=float)):
  row, body = wp.tid()
  mass = float(0.0)
  for child in range(model.body_count):
    if model.body_descendant[body, child] == 1:
      mass += model.body_mass[row, child]
  inverse = float(0.0)
  if mass > dyn.MINIMUM_MASS:
    inverse = 1.0 / mass
  result[row, body] = inverse


@wp.func
def _center_seed(adj: dyn.SubtreeJob, world: int, body: int) -> wp.vec3:
  return wp.vec3(adj.subtree_com_scalar[world, body, 0],
                 adj.subtree_com_scalar[world, body, 1],
                 adj.subtree_com_scalar[world, body, 2])


@wp.kernel(enable_backward=False)
def _center_position(job: dyn.SubtreeJob, adj: dyn.SubtreeJob,
                     inverse: wp.array2d(dtype=float)):
  world, child = wp.tid()
  row = world % job.model.body_mass.shape[0]
  mass = job.model.body_mass[row, child]
  value = wp.vec3(0.0)
  for body in range(job.model.body_count):
    inv = inverse[row, body]
    if inv > 0.0 and job.model.body_descendant[body, child] == 1:
      value += _center_seed(adj, world, body) * (mass * inv)
    elif inv == 0.0 and body == child:
      value += _center_seed(adj, world, body)
  adj.inertial_position[world, child] += value


@wp.kernel(enable_backward=False)
def _center_mass(job: dyn.SubtreeJob, adj: dyn.SubtreeJob,
                 inverse: wp.array2d(dtype=float)):
  row, child = wp.tid()
  value = float(0.0)
  for world in range(job.inertial_position.shape[0]):
    if world % job.model.body_mass.shape[0] == row:
      for body in range(job.model.body_count):
        inv = inverse[row, body]
        if inv > 0.0 and job.model.body_descendant[body, child] == 1:
          center = wp.vec3(job.subtree_com_scalar[world, body, 0],
                           job.subtree_com_scalar[world, body, 1],
                           job.subtree_com_scalar[world, body, 2])
          offset = job.inertial_position[world, child] - center
          value += wp.dot(_center_seed(adj, world, body), offset) * inv
  adj.model.body_mass[row, child] += value


class DynamicsTape(GeometryTape):
  """Use gathered VJPs for reductions and Warp adjoints for local kernels."""

  def record_launch(self, kernel, dim, max_blocks, inputs, outputs, device,
                    block_dim=0, metadata=None):
    split = {
        dyn._body_spatial_inertia: ('subtree_com', _inertia_center),
        dyn._crb_dofs: ('composite_inertia', _crb_inertia),
        dyn._bias_force: ('composite_force', _bias_body),
        dyn._applied_forces: ('gravcomp_force', _applied_body),
        dyn._gravcomp_body_force: ('subtree_com', _gravcomp_center),
    }
    if kernel in split:
      job = inputs[0]
      adj = self.get_adjoint(job)
      local = self.get_adjoint(job)
      field, gather = split[kernel]
      gradient = getattr(adj, field)
      setattr(local, field, None)
      source = getattr(job, field)
      detached = job._cls()
      for name in job._cls.vars:
        setattr(detached, name, getattr(job, name))
      # Clearing the adjoint alone leaves Warp's primal .grad fallback active.
      setattr(detached, field, wp.array(
          ptr=source.ptr, dtype=source.dtype, shape=source.shape,
          strides=source.strides, device=source.device, requires_grad=False))

      def backward_split():
        wp.launch(kernel, dim=dim, inputs=[detached], outputs=outputs,
                  adj_inputs=[local], adj_outputs=[], adjoint=True,
                  device=device, max_blocks=max_blocks, block_dim=block_dim)
        if gradient is not None:
          wp.launch(gather, dim=gradient.shape, inputs=[job, adj], device=device)

      self.launches.append(backward_split)
      return
    if kernel is dyn._motion_dof_components:
      job = inputs[0]
      adj = self.get_adjoint(job)

      def backward_motion():
        wp.launch(_motion_joint, dim=job.joint_axis.shape,
                  inputs=[job, adj], device=device)
        wp.launch(_motion_body, dim=job.body_matrix.shape,
                  inputs=[job, adj], device=device)

      self.launches.append(backward_motion)
      return
    if kernel is dyn._mass_matrix:
      job = inputs[0]
      adj = self.get_adjoint(job)

      def backward_mass():
        wp.launch(_mass_motion, dim=job.cdof.shape, inputs=[job, adj], device=device)
        if adj.model.dof_armature is not None:
          wp.launch(_mass_armature, dim=job.model.dof_armature.shape,
                    inputs=[job, adj], device=device)

      self.launches.append(backward_mass)
      return
    if kernel in (dyn._composite_force, dyn._composite_inertia):
      model, source = inputs
      target = outputs[0]
      self.get_adjoint(source)
      self.get_adjoint(target)
      reverse = _force if kernel is dyn._composite_force else _inertia
      if source.grad is not None:
        self.record_func(lambda: wp.launch(
            reverse, dim=source.shape, inputs=[model, target.grad, source.grad],
            device=device), [source, target])
      return
    if kernel is dyn._masked_velocity:
      job = inputs[0]
      adj = self.get_adjoint(job)
      self.record_func(lambda: wp.launch(
          _velocity, dim=job.qvel.shape, inputs=[job, adj], device=device),
          [x for x in (job.qvel, job.cdof, job.velocity) if x.grad is not None])
      return
    if kernel is dyn._subtree_com:
      job = inputs[0]
      adj = self.get_adjoint(job)
      inverse = wp.empty(job.model.body_mass.shape, dtype=float, device=device)

      def backward_center():
        wp.launch(_inverse_mass, dim=inverse.shape,
                  inputs=[job.model, inverse], device=device)
        if adj.inertial_position is not None:
          wp.launch(_center_position, dim=job.inertial_position.shape,
                    inputs=[job, adj, inverse], device=device)
        if adj.model.body_mass is not None:
          wp.launch(_center_mass, dim=job.model.body_mass.shape,
                    inputs=[job, adj, inverse], device=device)

      self.record_func(backward_center, [x for x in (
          job.model.body_mass, job.inertial_position, job.subtree_com_scalar)
          if x.grad is not None])
      return
    super().record_launch(kernel, dim, max_blocks, inputs, outputs, device,
                          block_dim=block_dim, metadata=metadata)
