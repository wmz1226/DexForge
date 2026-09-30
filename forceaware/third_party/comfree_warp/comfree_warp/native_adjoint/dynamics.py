"""Alias-free smooth rigid-body dynamics for native Warp VJPs."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from comfree_warp.mujoco_warp._src import math
from comfree_warp.mujoco_warp._src.types import vec10

from .features import validate_model
from .kinematics import KinematicOutput


_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
MINIMUM_MASS = wp.constant(1.0e-12)


@wp.struct
class DynamicsModel:
  body_ipos: wp.array2d(dtype=wp.vec3)
  body_iquat: wp.array2d(dtype=wp.quat)
  body_mass: wp.array2d(dtype=float)
  body_gravcomp: wp.array2d(dtype=float)
  body_inertia: wp.array2d(dtype=wp.vec3)
  body_rootid: wp.array(dtype=int)
  body_descendant: wp.array2d(dtype=int)
  body_dof: wp.array2d(dtype=int)
  dof_predecessor: wp.array2d(dtype=int)
  mass_mask: wp.array2d(dtype=int)
  dof_bodyid: wp.array(dtype=int)
  dof_jntid: wp.array(dtype=int)
  dof_armature: wp.array2d(dtype=float)
  dof_damping: wp.array2d(dtype=float)
  jnt_type: wp.array(dtype=int)
  jnt_dofadr: wp.array(dtype=int)
  actuator_dofid: wp.array(dtype=int)
  actuator_qposadr: wp.array(dtype=int)
  actuator_ctrllimited: wp.array(dtype=bool)
  actuator_forcelimited: wp.array(dtype=bool)
  actuator_gainprm: wp.array2d(dtype=vec10)
  actuator_biasprm: wp.array2d(dtype=vec10)
  actuator_ctrlrange: wp.array2d(dtype=wp.vec2)
  actuator_forcerange: wp.array2d(dtype=wp.vec2)
  actuator_gear: wp.array2d(dtype=wp.spatial_vector)
  gravity: wp.array(dtype=wp.vec3)
  gravity_enabled: int
  damper_enabled: int
  actuation_enabled: int
  clamp_control: int
  body_count: int
  dof_count: int
  actuator_count: int


@wp.struct
class DynamicsInput:
  qpos: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)
  ctrl: wp.array2d(dtype=float)
  qfrc_applied: wp.array2d(dtype=float)


@wp.struct
class DynamicsOutput:
  inertial_position: wp.array2d(dtype=wp.vec3)
  inertial_matrix: wp.array2d(dtype=wp.mat33)
  subtree_com_scalar: wp.array3d(dtype=float)
  subtree_com: wp.array2d(dtype=wp.vec3)
  body_inertia: wp.array2d(dtype=vec10)
  composite_inertia: wp.array2d(dtype=vec10)
  cdof_scalar: wp.array3d(dtype=float)
  cdof: wp.array2d(dtype=wp.spatial_vector)
  crb_cdof: wp.array2d(dtype=wp.spatial_vector)
  mass: wp.array3d(dtype=float)
  cvel: wp.array2d(dtype=wp.spatial_vector)
  predecessor_velocity: wp.array2d(dtype=wp.spatial_vector)
  cdof_dot: wp.array2d(dtype=wp.spatial_vector)
  bias_acceleration: wp.array2d(dtype=wp.spatial_vector)
  cacc: wp.array2d(dtype=wp.spatial_vector)
  local_force: wp.array2d(dtype=wp.spatial_vector)
  composite_force: wp.array2d(dtype=wp.spatial_vector)
  gravcomp_local_force: wp.array2d(dtype=wp.spatial_vector)
  gravcomp_composite_force: wp.array2d(dtype=wp.spatial_vector)
  bias_force: wp.array2d(dtype=float)
  passive_force: wp.array2d(dtype=float)
  actuator_force: wp.array2d(dtype=float)
  smooth_force: wp.array2d(dtype=float)


@wp.struct
class InertialFrameJob:
  model: DynamicsModel
  body_position: wp.array2d(dtype=wp.vec3)
  body_matrix: wp.array2d(dtype=wp.mat33)
  inertial_position: wp.array2d(dtype=wp.vec3)
  inertial_matrix: wp.array2d(dtype=wp.mat33)


@wp.struct
class SubtreeJob:
  model: DynamicsModel
  inertial_position: wp.array2d(dtype=wp.vec3)
  subtree_com_scalar: wp.array3d(dtype=float)


@wp.struct
class BodyInertiaJob:
  model: DynamicsModel
  inertial_position: wp.array2d(dtype=wp.vec3)
  inertial_matrix: wp.array2d(dtype=wp.mat33)
  subtree_com: wp.array2d(dtype=wp.vec3)
  body_inertia: wp.array2d(dtype=vec10)


@wp.struct
class MotionDofJob:
  model: DynamicsModel
  body_matrix: wp.array2d(dtype=wp.mat33)
  joint_anchor: wp.array2d(dtype=wp.vec3)
  joint_axis: wp.array2d(dtype=wp.vec3)
  subtree_com: wp.array2d(dtype=wp.vec3)
  cdof_scalar: wp.array3d(dtype=float)


@wp.struct
class CrbDofJob:
  model: DynamicsModel
  composite_inertia: wp.array2d(dtype=vec10)
  cdof: wp.array2d(dtype=wp.spatial_vector)
  crb_cdof: wp.array2d(dtype=wp.spatial_vector)


@wp.struct
class MassMatrixJob:
  model: DynamicsModel
  crb_cdof: wp.array2d(dtype=wp.spatial_vector)
  cdof: wp.array2d(dtype=wp.spatial_vector)
  mass: wp.array3d(dtype=float)


@wp.struct
class MaskedVelocityJob:
  mask: wp.array2d(dtype=int)
  qvel: wp.array2d(dtype=float)
  cdof: wp.array2d(dtype=wp.spatial_vector)
  velocity: wp.array2d(dtype=wp.spatial_vector)
  dof_count: int


@wp.struct
class BodyForceJob:
  model: DynamicsModel
  body_inertia: wp.array2d(dtype=vec10)
  cvel: wp.array2d(dtype=wp.spatial_vector)
  bias_acceleration: wp.array2d(dtype=wp.spatial_vector)
  cacc: wp.array2d(dtype=wp.spatial_vector)
  local_force: wp.array2d(dtype=wp.spatial_vector)


@wp.struct
class BiasForceJob:
  model: DynamicsModel
  cdof: wp.array2d(dtype=wp.spatial_vector)
  composite_force: wp.array2d(dtype=wp.spatial_vector)
  bias_force: wp.array2d(dtype=float)


@wp.struct
class GravcompForceJob:
  model: DynamicsModel
  inertial_position: wp.array2d(dtype=wp.vec3)
  subtree_com: wp.array2d(dtype=wp.vec3)
  local_force: wp.array2d(dtype=wp.spatial_vector)


@wp.struct
class AppliedForceJob:
  model: DynamicsModel
  inputs: DynamicsInput
  cdof: wp.array2d(dtype=wp.spatial_vector)
  gravcomp_force: wp.array2d(dtype=wp.spatial_vector)
  bias_force: wp.array2d(dtype=float)
  passive_force: wp.array2d(dtype=float)
  actuator_force: wp.array2d(dtype=float)
  smooth_force: wp.array2d(dtype=float)


@dataclass(frozen=True)
class CompiledDynamics:
  model: DynamicsModel
  body_count: int
  dof_count: int
  actuator_count: int
  device: object


@dataclass(frozen=True)
class _DynamicsExecution:
  compiled: CompiledDynamics
  inputs: DynamicsInput
  pose: KinematicOutput
  output: DynamicsOutput


@wp.func
def _row(world: int, count: int) -> int:
  return world % count


@wp.func
def _basis(axis: int) -> wp.vec3:
  return wp.vec3(float(axis == 0), float(axis == 1), float(axis == 2))


@wp.func
def _spatial(angular: wp.vec3, linear: wp.vec3) -> wp.spatial_vector:
  return wp.spatial_vector(angular[0], angular[1], angular[2],
                           linear[0], linear[1], linear[2])


@wp.func
def _spatial_basis(component: int) -> wp.spatial_vector:
  return wp.spatial_vector(float(component == 0), float(component == 1),
                           float(component == 2), float(component == 3),
                           float(component == 4), float(component == 5))


@wp.func
def _matrix_axis(matrix: wp.mat33, axis: int) -> wp.vec3:
  return wp.vec3(matrix[0, axis], matrix[1, axis], matrix[2, axis])


@wp.func
def _angular_offset(angular: wp.vec3, offset: wp.vec3) -> wp.vec3:
  return wp.vec3(angular[1] * offset[2] - angular[2] * offset[1],
                 angular[2] * offset[0] - angular[0] * offset[2],
                 angular[0] * offset[1] - angular[1] * offset[0])


@wp.func
def _parallel_axis(inertia: wp.mat33, offset: wp.vec3,
                   mass: float) -> wp.mat33:
  squared = wp.dot(offset, offset)
  identity = wp.identity(n=3, dtype=float)
  return inertia + mass * (squared * identity - wp.outer(offset, offset))


@wp.func
def _pack_inertia(inertia: wp.mat33, offset: wp.vec3,
                  mass: float) -> vec10:
  weighted = offset * mass
  return vec10(inertia[0, 0], inertia[1, 1], inertia[2, 2],
               inertia[0, 1], inertia[0, 2], inertia[1, 2],
               weighted[0], weighted[1], weighted[2], mass)


@wp.kernel
def _inertial_frames(job: InertialFrameJob):
  world, body = wp.tid()
  model = job.model
  row = _row(world, model.body_ipos.shape[0])
  rotation = job.body_matrix[world, body]
  local_matrix = math.quat_to_mat(model.body_iquat[row, body])
  job.inertial_position[world, body] = (
      job.body_position[world, body] + rotation @ model.body_ipos[row, body])
  job.inertial_matrix[world, body] = rotation @ local_matrix


@wp.func
def _subtree_component(job: SubtreeJob, index: wp.vec3i) -> float:
  model = job.model
  world = index[0]
  body = index[1]
  component = index[2]
  row = _row(world, model.body_mass.shape[0])
  weighted = float(0.0)
  mass = float(0.0)
  for candidate in range(model.body_count):
    if model.body_descendant[body, candidate] == 1:
      candidate_mass = model.body_mass[row, candidate]
      weighted += job.inertial_position[world, candidate][component] * candidate_mass
      mass += candidate_mass
  if mass > MINIMUM_MASS:
    return weighted / mass
  return job.inertial_position[world, body][component]


@wp.func_grad(_subtree_component)
def _adj_subtree_component(job: SubtreeJob, index: wp.vec3i,
                           adj_ret: float):
  model = job.model
  world = index[0]
  body = index[1]
  component = index[2]
  row = _row(world, model.body_mass.shape[0])
  weighted = float(0.0)
  mass = float(0.0)
  for candidate in range(model.body_count):
    if model.body_descendant[body, candidate] == 1:
      candidate_mass = model.body_mass[row, candidate]
      weighted += job.inertial_position[world, candidate][component] * candidate_mass
      mass += candidate_mass
  if mass <= MINIMUM_MASS:
    if wp.adjoint[job].inertial_position:
      wp.atomic_add(wp.adjoint[job].inertial_position, world, body,
                    _basis(component) * adj_ret)
    return
  center = weighted / mass
  for candidate in range(model.body_count):
    if model.body_descendant[body, candidate] == 1:
      candidate_mass = model.body_mass[row, candidate]
      position = job.inertial_position[world, candidate][component]
      if wp.adjoint[job].inertial_position:
        scale = adj_ret * candidate_mass / mass
        wp.atomic_add(
            wp.adjoint[job].inertial_position, world, candidate,
            _basis(component) * scale)
      if wp.adjoint[job].model.body_mass:
        wp.atomic_add(wp.adjoint[job].model.body_mass, row, candidate,
                      adj_ret * (position - center) / mass)


@wp.kernel
def _subtree_com(job: SubtreeJob):
  world, body, component = wp.tid()
  job.subtree_com_scalar[world, body, component] = _subtree_component(
      job, wp.vec3i(world, body, component))


@wp.kernel
def _pack_subtree_com(subtree_scalar: wp.array3d(dtype=float),
                      subtree_com: wp.array2d(dtype=wp.vec3)):
  world, body = wp.tid()
  subtree_com[world, body] = wp.vec3(
      subtree_scalar[world, body, 0], subtree_scalar[world, body, 1],
      subtree_scalar[world, body, 2])


@wp.kernel
def _body_spatial_inertia(job: BodyInertiaJob):
  world, body = wp.tid()
  model = job.model
  row = _row(world, model.body_mass.shape[0])
  root = model.body_rootid[body]
  offset = job.inertial_position[world, body] - job.subtree_com[world, root]
  rotation = job.inertial_matrix[world, body]
  diagonal = wp.diag(model.body_inertia[row, body])
  inertia = rotation @ diagonal @ wp.transpose(rotation)
  mass = model.body_mass[row, body]
  job.body_inertia[world, body] = _pack_inertia(
      _parallel_axis(inertia, offset, mass), offset, mass)


@wp.kernel
def _motion_dof_components(job: MotionDofJob):
  world, dof, component = wp.tid()
  model = job.model
  joint = model.dof_jntid[dof]
  body = model.dof_bodyid[dof]
  joint_type = model.jnt_type[joint]
  dof_offset = dof - model.jnt_dofadr[joint]
  angular = wp.vec3(0.0)
  linear = wp.vec3(0.0)
  if joint_type == _FREE and dof_offset < 3:
    linear = _basis(dof_offset)
  if joint_type == _FREE and dof_offset >= 3:
    angular = _matrix_axis(job.body_matrix[world, body], dof_offset - 3)
    offset = (job.subtree_com[world, model.body_rootid[body]]
              - job.joint_anchor[world, joint])
    linear = _angular_offset(angular, offset)
  if joint_type == _BALL:
    angular = _matrix_axis(job.body_matrix[world, body], dof_offset)
    offset = (job.subtree_com[world, model.body_rootid[body]]
              - job.joint_anchor[world, joint])
    linear = _angular_offset(angular, offset)
  if joint_type == _HINGE:
    angular = job.joint_axis[world, joint]
    offset = (job.subtree_com[world, model.body_rootid[body]]
              - job.joint_anchor[world, joint])
    linear = _angular_offset(angular, offset)
  if joint_type == _SLIDE:
    linear = job.joint_axis[world, joint]
  if component < 3:
    job.cdof_scalar[world, dof, component] = angular[component]
  else:
    job.cdof_scalar[world, dof, component] = linear[component - 3]


@wp.kernel
def _pack_motion_dofs(cdof_scalar: wp.array3d(dtype=float),
                      cdof: wp.array2d(dtype=wp.spatial_vector)):
  world, dof = wp.tid()
  cdof[world, dof] = wp.spatial_vector(
      cdof_scalar[world, dof, 0], cdof_scalar[world, dof, 1],
      cdof_scalar[world, dof, 2], cdof_scalar[world, dof, 3],
      cdof_scalar[world, dof, 4], cdof_scalar[world, dof, 5])


@wp.kernel
def _composite_inertia(model: DynamicsModel,
                       body_inertia: wp.array2d(dtype=vec10),
                       composite_inertia: wp.array2d(dtype=vec10)):
  world, body = wp.tid()
  inertia = vec10(0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0)
  for candidate in range(model.body_count):
    if model.body_descendant[body, candidate] == 1:
      inertia += body_inertia[world, candidate]
  composite_inertia[world, body] = inertia


@wp.kernel
def _crb_dofs(job: CrbDofJob):
  world, dof = wp.tid()
  inertia = job.composite_inertia[world, job.model.dof_bodyid[dof]]
  job.crb_cdof[world, dof] = math.inert_vec(inertia, job.cdof[world, dof])


@wp.kernel
def _mass_matrix(job: MassMatrixJob):
  world, row_dof, column_dof = wp.tid()
  model = job.model
  value = float(0.0)
  relation = model.mass_mask[row_dof, column_dof]
  if relation == 1:
    value = wp.dot(job.crb_cdof[world, row_dof],
                   job.cdof[world, column_dof])
  elif relation == 2:
    value = wp.dot(job.crb_cdof[world, column_dof],
                   job.cdof[world, row_dof])
  if row_dof == column_dof:
    row = _row(world, model.dof_armature.shape[0])
    value += model.dof_armature[row, row_dof]
  job.mass[world, row_dof, column_dof] = value


@wp.func
def _masked_velocity_component(job: MaskedVelocityJob,
                               index: wp.vec3i) -> float:
  world = index[0]
  row = index[1]
  component = index[2]
  value = float(0.0)
  for dof in range(job.dof_count):
    if job.mask[row, dof] == 1:
      value += job.cdof[world, dof][component] * job.qvel[world, dof]
  return value


@wp.func_grad(_masked_velocity_component)
def _adj_masked_velocity_component(job: MaskedVelocityJob,
                                   index: wp.vec3i,
                                   adj_ret: float):
  world = index[0]
  row = index[1]
  component = index[2]
  for dof in range(job.dof_count):
    if job.mask[row, dof] == 1:
      motion = job.cdof[world, dof][component]
      speed = job.qvel[world, dof]
      if wp.adjoint[job].qvel:
        wp.atomic_add(wp.adjoint[job].qvel, world, dof, adj_ret * motion)
      if wp.adjoint[job].cdof:
        wp.atomic_add(wp.adjoint[job].cdof, world, dof,
                      _spatial_basis(component) * adj_ret * speed)


@wp.kernel
def _masked_velocity(job: MaskedVelocityJob):
  world, row = wp.tid()
  job.velocity[world, row] = wp.spatial_vector(
      _masked_velocity_component(job, wp.vec3i(world, row, 0)),
      _masked_velocity_component(job, wp.vec3i(world, row, 1)),
      _masked_velocity_component(job, wp.vec3i(world, row, 2)),
      _masked_velocity_component(job, wp.vec3i(world, row, 3)),
      _masked_velocity_component(job, wp.vec3i(world, row, 4)),
      _masked_velocity_component(job, wp.vec3i(world, row, 5)))


@wp.kernel
def _motion_derivative(predecessor_velocity: wp.array2d(dtype=wp.spatial_vector),
                       cdof: wp.array2d(dtype=wp.spatial_vector),
                       cdof_dot: wp.array2d(dtype=wp.spatial_vector)):
  world, dof = wp.tid()
  cdof_dot[world, dof] = math.motion_cross(
      predecessor_velocity[world, dof], cdof[world, dof])


@wp.kernel
def _body_force(job: BodyForceJob):
  world, body = wp.tid()
  gravity = job.model.gravity[_row(world, job.model.gravity.shape[0])]
  gravity *= float(job.model.gravity_enabled)
  bias = job.bias_acceleration[world, body]
  acceleration = bias + _spatial(wp.vec3(0.0), -gravity)
  inertia = job.body_inertia[world, body]
  velocity = job.cvel[world, body]
  inertial_velocity = math.inert_vec(inertia, velocity)
  job.cacc[world, body] = acceleration
  job.local_force[world, body] = (
      math.inert_vec(inertia, acceleration)
      + math.motion_cross_force(velocity, inertial_velocity))


@wp.kernel
def _composite_force(model: DynamicsModel,
                     local_force: wp.array2d(dtype=wp.spatial_vector),
                     composite_force: wp.array2d(dtype=wp.spatial_vector)):
  world, body = wp.tid()
  force = _spatial(wp.vec3(0.0), wp.vec3(0.0))
  for candidate in range(model.body_count):
    if model.body_descendant[body, candidate] == 1:
      force += local_force[world, candidate]
  composite_force[world, body] = force


@wp.kernel
def _bias_force(job: BiasForceJob):
  world, dof = wp.tid()
  body = job.model.dof_bodyid[dof]
  job.bias_force[world, dof] = wp.dot(
      job.cdof[world, dof], job.composite_force[world, body])


@wp.kernel
def _gravcomp_body_force(job: GravcompForceJob):
  world, body = wp.tid()
  model = job.model
  root = model.body_rootid[body]
  row = _row(world, model.body_mass.shape[0])
  gravity = model.gravity[_row(world, model.gravity.shape[0])]
  gravity *= float(model.gravity_enabled)
  scale = model.body_mass[row, body] * model.body_gravcomp[row, body]
  linear = -gravity * scale
  offset = job.inertial_position[world, body] - job.subtree_com[world, root]
  job.local_force[world, body] = _spatial(wp.cross(offset, linear), linear)


@wp.func
def _actuator_generalized_force(model: DynamicsModel, inputs: DynamicsInput,
                                world: int, dof: int) -> float:
  if model.actuation_enabled == 0:
    return 0.0
  result = float(0.0)
  for actuator in range(model.actuator_count):
    if model.actuator_dofid[actuator] == dof:
      row = _row(world, model.actuator_gainprm.shape[0])
      gear = model.actuator_gear[row, actuator][0]
      ctrl = inputs.ctrl[world, actuator]
      if model.clamp_control != 0 and model.actuator_ctrllimited[actuator]:
        limits = model.actuator_ctrlrange[row, actuator]
        ctrl = wp.clamp(ctrl, limits[0], limits[1])
      length = inputs.qpos[world, model.actuator_qposadr[actuator]] * gear
      velocity = inputs.qvel[world, dof] * gear
      gain = model.actuator_gainprm[row, actuator][0]
      bias = model.actuator_biasprm[row, actuator]
      force = gain * ctrl + bias[0] + bias[1] * length + bias[2] * velocity
      if model.actuator_forcelimited[actuator]:
        limits = model.actuator_forcerange[row, actuator]
        force = wp.clamp(force, limits[0], limits[1])
      result += gear * force
  return result


@wp.kernel
def _applied_forces(job: AppliedForceJob):
  world, dof = wp.tid()
  model, inputs = job.model, job.inputs
  actuator_force = _actuator_generalized_force(model, inputs, world, dof)
  row = _row(world, model.dof_damping.shape[0])
  body = model.dof_bodyid[dof]
  gravcomp = wp.dot(job.cdof[world, dof], job.gravcomp_force[world, body])
  damping = (-float(model.damper_enabled) * model.dof_damping[row, dof]
             * inputs.qvel[world, dof])
  passive = damping + gravcomp
  job.passive_force[world, dof] = passive
  job.actuator_force[world, dof] = actuator_force
  job.smooth_force[world, dof] = (
      passive - job.bias_force[world, dof] + actuator_force
      + inputs.qfrc_applied[world, dof])


def _body_descendants(model: mujoco.MjModel) -> np.ndarray:
  mask = np.zeros((model.nbody, model.nbody), dtype=np.int32)
  for descendant in range(model.nbody):
    body = descendant
    while True:
      mask[body, descendant] = 1
      if body == 0:
        break
      body = int(model.body_parentid[body])
  return mask


def _body_dofs(model: mujoco.MjModel) -> np.ndarray:
  mask = np.zeros((model.nbody, model.nv), dtype=np.int32)
  for body in range(model.nbody):
    ancestor = body
    while ancestor > 0:
      start = int(model.body_dofadr[ancestor])
      count = int(model.body_dofnum[ancestor])
      mask[body, start:start + count] = 1
      ancestor = int(model.body_parentid[ancestor])
  return mask


def _dof_predecessors(model: mujoco.MjModel) -> np.ndarray:
  body_dof = _body_dofs(model)
  mask = np.zeros((model.nv, model.nv), dtype=np.int32)
  for dof in range(model.nv):
    joint = int(model.dof_jntid[dof])
    body = int(model.dof_bodyid[dof])
    parent = int(model.body_parentid[body])
    if parent >= 0:
      mask[dof] = body_dof[parent]
    body_start = int(model.body_dofadr[body])
    joint_start = int(model.jnt_dofadr[joint])
    mask[dof, body_start:joint_start] = 1
    offset = dof - int(model.jnt_dofadr[joint])
    if int(model.jnt_type[joint]) == _FREE and offset >= 3:
      start = int(model.jnt_dofadr[joint])
      mask[dof, start:start + 3] = 1
  return mask


def _mass_mask(model: mujoco.MjModel) -> np.ndarray:
  mask = np.zeros((model.nv, model.nv), dtype=np.int32)
  for dof in range(model.nv):
    ancestor = dof
    while ancestor >= 0:
      mask[dof, ancestor] = 1
      if ancestor != dof:
        mask[ancestor, dof] = 2
      ancestor = int(model.dof_parentid[ancestor])
  return mask


def _actuator_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  joints = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
  dofs = np.asarray(model.jnt_dofadr[joints], dtype=np.int32)
  qpos = np.asarray(model.jnt_qposadr[joints], dtype=np.int32)
  return dofs, qpos


def _assign_device_fields(target: DynamicsModel, source) -> None:
  names = ("body_ipos", "body_iquat", "body_mass", "body_gravcomp",
           "body_inertia",
           "body_rootid", "dof_bodyid", "dof_jntid", "dof_armature",
           "dof_damping", "jnt_type", "jnt_dofadr", "actuator_ctrllimited",
           "actuator_forcelimited", "actuator_gainprm", "actuator_biasprm",
           "actuator_ctrlrange", "actuator_forcerange", "actuator_gear")
  for name in names:
    setattr(target, name, getattr(source, name))
  target.gravity = source.opt.gravity


def _feature_enabled(model: mujoco.MjModel,
                     feature: mujoco.mjtDisableBit) -> int:
  return int(not bool(int(model.opt.disableflags) & int(feature)))


def compile_dynamics(cpu_model: mujoco.MjModel, device_model) -> CompiledDynamics:
  features = validate_model(cpu_model)
  device = device_model.qpos0.device
  model = DynamicsModel()
  _assign_device_fields(model, device_model)
  model.gravity_enabled = _feature_enabled(
      cpu_model, mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  model.damper_enabled = _feature_enabled(
      cpu_model, mujoco.mjtDisableBit.mjDSBL_DAMPER)
  model.actuation_enabled = _feature_enabled(
      cpu_model, mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  model.clamp_control = _feature_enabled(
      cpu_model, mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  model.body_descendant = wp.array(
      _body_descendants(cpu_model), dtype=int, device=device)
  model.body_dof = wp.array(_body_dofs(cpu_model), dtype=int, device=device)
  model.dof_predecessor = wp.array(
      _dof_predecessors(cpu_model), dtype=int, device=device)
  model.mass_mask = wp.array(_mass_mask(cpu_model), dtype=int, device=device)
  dofs, qpos = _actuator_addresses(cpu_model)
  model.actuator_dofid = wp.array(dofs, dtype=int, device=device)
  model.actuator_qposadr = wp.array(qpos, dtype=int, device=device)
  model.body_count = features.bodies
  model.dof_count = features.velocities
  model.actuator_count = features.actuators
  return CompiledDynamics(model, features.bodies, features.velocities,
                          features.actuators, device)


def _gradient_array(shape, dtype, device, *, requires_grad: bool):
  return wp.empty(shape, dtype=dtype, device=device,
                  requires_grad=requires_grad, retain_grad=requires_grad)


def allocate_output(compiled: CompiledDynamics, worlds: int,
                    *, requires_grad: bool) -> DynamicsOutput:
  array = lambda shape, dtype: _gradient_array(
      shape, dtype, compiled.device, requires_grad=requires_grad)
  bodies = (worlds, compiled.body_count)
  dofs = (worlds, compiled.dof_count)
  output = DynamicsOutput()
  output.inertial_position = array(bodies, wp.vec3)
  output.inertial_matrix = array(bodies, wp.mat33)
  output.subtree_com_scalar = array((worlds, compiled.body_count, 3), float)
  output.subtree_com = array(bodies, wp.vec3)
  output.body_inertia = array(bodies, vec10)
  output.composite_inertia = array(bodies, vec10)
  output.cdof_scalar = array((worlds, compiled.dof_count, 6), float)
  output.cdof = array(dofs, wp.spatial_vector)
  output.crb_cdof = array(dofs, wp.spatial_vector)
  output.mass = array((worlds, compiled.dof_count, compiled.dof_count), float)
  output.cvel = array(bodies, wp.spatial_vector)
  output.predecessor_velocity = array(dofs, wp.spatial_vector)
  output.cdof_dot = array(dofs, wp.spatial_vector)
  output.bias_acceleration = array(bodies, wp.spatial_vector)
  output.cacc = array(bodies, wp.spatial_vector)
  output.local_force = array(bodies, wp.spatial_vector)
  output.composite_force = array(bodies, wp.spatial_vector)
  output.gravcomp_local_force = array(bodies, wp.spatial_vector)
  output.gravcomp_composite_force = array(bodies, wp.spatial_vector)
  output.bias_force = array(dofs, float)
  output.passive_force = array(dofs, float)
  output.actuator_force = array(dofs, float)
  output.smooth_force = array(dofs, float)
  return output


def _job(struct, **fields):
  job = struct()
  for name, value in fields.items():
    setattr(job, name, value)
  return job


def _body_inertia_stages(execution: _DynamicsExecution) -> None:
  compiled = execution.compiled
  pose = execution.pose
  output = execution.output
  worlds = execution.inputs.qpos.shape[0]
  model = compiled.model
  frame_job = _job(
      InertialFrameJob, model=model, body_position=pose.body_position,
      body_matrix=pose.body_matrix, inertial_position=output.inertial_position,
      inertial_matrix=output.inertial_matrix)
  wp.launch(_inertial_frames, dim=(worlds, compiled.body_count),
            inputs=[frame_job])
  subtree_job = _job(
      SubtreeJob, model=model, inertial_position=output.inertial_position,
      subtree_com_scalar=output.subtree_com_scalar)
  wp.launch(_subtree_com, dim=(worlds, compiled.body_count, 3),
            inputs=[subtree_job])
  wp.launch(_pack_subtree_com, dim=(worlds, compiled.body_count),
            inputs=[output.subtree_com_scalar], outputs=[output.subtree_com])
  inertia_job = _job(
      BodyInertiaJob, model=model,
      inertial_position=output.inertial_position,
      inertial_matrix=output.inertial_matrix, subtree_com=output.subtree_com,
      body_inertia=output.body_inertia)
  wp.launch(_body_spatial_inertia, dim=(worlds, compiled.body_count),
            inputs=[inertia_job])


def _mass_stages(execution: _DynamicsExecution) -> None:
  compiled = execution.compiled
  pose = execution.pose
  output = execution.output
  worlds = execution.inputs.qpos.shape[0]
  model = compiled.model
  motion_job = _job(
      MotionDofJob, model=model, body_matrix=pose.body_matrix,
      joint_anchor=pose.joint_anchor, joint_axis=pose.joint_axis,
      subtree_com=output.subtree_com, cdof_scalar=output.cdof_scalar)
  wp.launch(_motion_dof_components,
            dim=(worlds, compiled.dof_count, 6),
            inputs=[motion_job])
  wp.launch(_pack_motion_dofs, dim=(worlds, compiled.dof_count),
            inputs=[output.cdof_scalar], outputs=[output.cdof])
  wp.launch(_composite_inertia, dim=(worlds, compiled.body_count),
            inputs=[model, output.body_inertia],
            outputs=[output.composite_inertia])
  crb_job = _job(
      CrbDofJob, model=model, composite_inertia=output.composite_inertia,
      cdof=output.cdof, crb_cdof=output.crb_cdof)
  wp.launch(_crb_dofs, dim=(worlds, compiled.dof_count),
            inputs=[crb_job])
  mass_job = _job(
      MassMatrixJob, model=model, crb_cdof=output.crb_cdof,
      cdof=output.cdof, mass=output.mass)
  wp.launch(_mass_matrix, dim=(worlds, compiled.dof_count, compiled.dof_count),
            inputs=[mass_job])


def _inertia_stages(execution: _DynamicsExecution) -> None:
  _body_inertia_stages(execution)
  _mass_stages(execution)


def _launch_masked_velocity(execution: _DynamicsExecution, mask,
                            velocity, *, cdof) -> None:
  job = MaskedVelocityJob()
  job.mask = mask
  job.qvel = execution.inputs.qvel
  job.cdof = cdof
  job.velocity = velocity
  job.dof_count = execution.compiled.dof_count
  wp.launch(_masked_velocity, dim=velocity.shape, inputs=[job])


def _velocity_stages(execution: _DynamicsExecution) -> None:
  compiled = execution.compiled
  inputs = execution.inputs
  output = execution.output
  worlds = inputs.qpos.shape[0]
  model = compiled.model
  _launch_masked_velocity(
      execution, model.body_dof, output.cvel, cdof=output.cdof)
  _launch_masked_velocity(
      execution, model.dof_predecessor, output.predecessor_velocity,
      cdof=output.cdof)
  wp.launch(_motion_derivative, dim=(worlds, compiled.dof_count),
            inputs=[output.predecessor_velocity, output.cdof],
            outputs=[output.cdof_dot])
  _launch_masked_velocity(
      execution, model.body_dof, output.bias_acceleration,
      cdof=output.cdof_dot)


def _force_stages(execution: _DynamicsExecution) -> None:
  compiled = execution.compiled
  inputs = execution.inputs
  output = execution.output
  worlds = inputs.qpos.shape[0]
  model = compiled.model
  body_job = _job(
      BodyForceJob, model=model, body_inertia=output.body_inertia,
      cvel=output.cvel, bias_acceleration=output.bias_acceleration,
      cacc=output.cacc, local_force=output.local_force)
  wp.launch(_body_force, dim=(worlds, compiled.body_count),
            inputs=[body_job])
  wp.launch(_composite_force, dim=(worlds, compiled.body_count),
            inputs=[model, output.local_force], outputs=[output.composite_force])
  bias_job = _job(
      BiasForceJob, model=model, cdof=output.cdof,
      composite_force=output.composite_force, bias_force=output.bias_force)
  wp.launch(_bias_force, dim=(worlds, compiled.dof_count),
            inputs=[bias_job])
  gravcomp_job = _job(
      GravcompForceJob, model=model,
      inertial_position=output.inertial_position,
      subtree_com=output.subtree_com,
      local_force=output.gravcomp_local_force)
  wp.launch(_gravcomp_body_force, dim=(worlds, compiled.body_count),
            inputs=[gravcomp_job])
  wp.launch(_composite_force, dim=(worlds, compiled.body_count),
            inputs=[model, output.gravcomp_local_force],
            outputs=[output.gravcomp_composite_force])
  applied_job = _job(
      AppliedForceJob, model=model, inputs=inputs,
      cdof=output.cdof,
      gravcomp_force=output.gravcomp_composite_force,
      bias_force=output.bias_force, passive_force=output.passive_force,
      actuator_force=output.actuator_force, smooth_force=output.smooth_force)
  wp.launch(_applied_forces, dim=(worlds, compiled.dof_count),
            inputs=[applied_job])


def dynamics(compiled: CompiledDynamics, inputs: DynamicsInput,
             pose: KinematicOutput, *, output: DynamicsOutput) -> None:
  execution = _DynamicsExecution(compiled, inputs, pose, output)
  _inertia_stages(execution)
  _velocity_stages(execution)
  _force_stages(execution)
