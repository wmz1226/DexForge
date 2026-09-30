"""Alias-free MuJoCo tree kinematics with native Warp gradients."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from comfree_warp.mujoco_warp._src import math

from .features import validate_model


_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


@wp.struct
class KinematicModel:
  body_depth: wp.array(dtype=int)
  body_parentid: wp.array(dtype=int)
  body_jntnum: wp.array(dtype=int)
  body_jntadr: wp.array(dtype=int)
  body_pos: wp.array2d(dtype=wp.vec3)
  body_quat: wp.array2d(dtype=wp.quat)
  jnt_type: wp.array(dtype=int)
  jnt_bodyid: wp.array(dtype=int)
  jnt_qposadr: wp.array(dtype=int)
  jnt_pos: wp.array2d(dtype=wp.vec3)
  jnt_axis: wp.array2d(dtype=wp.vec3)
  qpos0: wp.array2d(dtype=float)


@wp.struct
class JointPoseQuery:
  model: KinematicModel
  qpos: wp.array2d(dtype=float)
  world: int
  joint: int
  position: wp.vec3
  quaternion: wp.quat


@wp.struct
class KinematicLevelInput:
  model: KinematicModel
  qpos: wp.array2d(dtype=float)
  depth: int
  position: wp.array2d(dtype=wp.vec3)
  quaternion: wp.array2d(dtype=wp.quat)


@wp.struct
class PoseOutput:
  position: wp.array2d(dtype=wp.vec3)
  quaternion: wp.array2d(dtype=wp.quat)


@wp.struct
class JointFrameInput:
  model: KinematicModel
  qpos: wp.array2d(dtype=float)
  body_position: wp.array2d(dtype=wp.vec3)
  body_quaternion: wp.array2d(dtype=wp.quat)


@wp.struct
class JointFrameOutput:
  anchor: wp.array2d(dtype=wp.vec3)
  axis: wp.array2d(dtype=wp.vec3)
  quaternion: wp.array2d(dtype=wp.quat)


@dataclass(frozen=True)
class BodyPose:
  position: wp.array
  quaternion: wp.array


@dataclass(frozen=True)
class KinematicOutput:
  stages: tuple[BodyPose, ...]
  body_position: wp.array
  body_quaternion: wp.array
  body_matrix: wp.array
  joint_anchor: wp.array
  joint_axis: wp.array
  joint_quaternion: wp.array


@dataclass(frozen=True)
class CompiledKinematics:
  model: KinematicModel
  body_count: int
  joint_count: int
  max_depth: int
  device: object


@wp.func
def _model_row(world: int, rows: int) -> int:
  return world % rows


@wp.func
def _free_pose(qpos: wp.array2d(dtype=float), world: int, address: int):
  position = wp.vec3(qpos[world, address], qpos[world, address + 1],
                     qpos[world, address + 2])
  quaternion = wp.quat(qpos[world, address + 3], qpos[world, address + 4],
                       qpos[world, address + 5], qpos[world, address + 6])
  return position, wp.normalize(quaternion)


@wp.func
def _apply_joint(query: JointPoseQuery):
  model = query.model
  qpos = query.qpos
  world = query.world
  joint = query.joint
  address = model.jnt_qposadr[joint]
  joint_type = model.jnt_type[joint]
  if joint_type == _FREE:
    free_position, free_quaternion = _free_pose(qpos, world, address)
    row = _model_row(world, model.jnt_axis.shape[0])
    return free_position, free_quaternion, free_position, model.jnt_axis[row, joint]
  row = _model_row(world, model.jnt_pos.shape[0])
  local_anchor = model.jnt_pos[row, joint]
  local_axis = model.jnt_axis[row, joint]
  anchor = query.position + math.rot_vec_quat(local_anchor, query.quaternion)
  axis = math.rot_vec_quat(local_axis, query.quaternion)
  if joint_type == _SLIDE:
    delta = qpos[world, address] - model.qpos0[
        _model_row(world, model.qpos0.shape[0]), address]
    return query.position + axis * delta, query.quaternion, anchor, axis
  if joint_type == _HINGE:
    delta = qpos[world, address] - model.qpos0[
        _model_row(world, model.qpos0.shape[0]), address]
    rotation = math.axis_angle_to_quat(local_axis, delta)
  else:
    rotation = wp.quat(qpos[world, address], qpos[world, address + 1],
                       qpos[world, address + 2], qpos[world, address + 3])
    rotation = wp.normalize(rotation)
  result = math.mul_quat(query.quaternion, rotation)
  return anchor - math.rot_vec_quat(local_anchor, result), result, anchor, axis


@wp.kernel
def _kinematic_level(inputs: KinematicLevelInput, output: PoseOutput):
  world, body = wp.tid()
  model = inputs.model
  position = inputs.position[world, body]
  quaternion = inputs.quaternion[world, body]
  if model.body_depth[body] == inputs.depth:
    row = _model_row(world, model.body_pos.shape[0])
    position = model.body_pos[row, body]
    quaternion = model.body_quat[row, body]
    if body > 0:
      parent = model.body_parentid[body]
      parent_position = inputs.position[world, parent]
      parent_quaternion = inputs.quaternion[world, parent]
      position = parent_position + math.rot_vec_quat(position, parent_quaternion)
      quaternion = math.mul_quat(parent_quaternion, quaternion)
    if model.body_jntnum[body] > 0:
      query = JointPoseQuery()
      query.model = model
      query.qpos = inputs.qpos
      query.world = world
      first_joint = model.body_jntadr[body]
      for offset in range(model.body_jntnum[body]):
        query.joint = first_joint + offset
        query.position = position
        query.quaternion = quaternion
        position, quaternion, _, _ = _apply_joint(query)
  output.position[world, body] = position
  output.quaternion[world, body] = wp.normalize(quaternion)


@wp.kernel
def _body_matrices(quaternion: wp.array2d(dtype=wp.quat),
                   matrix: wp.array2d(dtype=wp.mat33)):
  world, body = wp.tid()
  matrix[world, body] = math.quat_to_mat(quaternion[world, body])


@wp.kernel
def _joint_frames(inputs: JointFrameInput, output: JointFrameOutput):
  world, body = wp.tid()
  model = inputs.model
  if body > 0 and model.body_jntnum[body] > 0:
    parent = model.body_parentid[body]
    position = inputs.body_position[world, parent]
    quaternion = inputs.body_quaternion[world, parent]
    body_row = _model_row(world, model.body_pos.shape[0])
    position += math.rot_vec_quat(model.body_pos[body_row, body], quaternion)
    quaternion = math.mul_quat(quaternion, model.body_quat[body_row, body])
    query = JointPoseQuery()
    query.model = model
    query.qpos = inputs.qpos
    query.world = world
    first_joint = model.body_jntadr[body]
    for offset in range(model.body_jntnum[body]):
      joint = first_joint + offset
      query.joint = joint
      query.position = position
      query.quaternion = quaternion
      output.quaternion[world, joint] = quaternion
      position, quaternion, anchor, axis = _apply_joint(query)
      output.anchor[world, joint] = anchor
      output.axis[world, joint] = axis


def _body_depths(model: mujoco.MjModel) -> np.ndarray:
  depths = np.zeros(model.nbody, dtype=np.int32)
  for body in range(1, model.nbody):
    depths[body] = depths[int(model.body_parentid[body])] + 1
  return depths


def compile_kinematics(cpu_model: mujoco.MjModel,
                       device_model) -> CompiledKinematics:
  features = validate_model(cpu_model)
  depths = _body_depths(cpu_model)
  device = device_model.qpos0.device
  model = KinematicModel()
  model.body_depth = wp.array(depths, dtype=int, device=device)
  fields = ("body_parentid", "body_jntnum", "body_jntadr", "body_pos",
            "body_quat", "jnt_type", "jnt_bodyid", "jnt_qposadr",
            "jnt_pos", "jnt_axis", "qpos0")
  for name in fields:
    setattr(model, name, getattr(device_model, name))
  return CompiledKinematics(model, features.bodies, features.joints,
                            features.max_body_depth, device)


def allocate_output(compiled: CompiledKinematics, worlds: int,
                    *, requires_grad: bool) -> KinematicOutput:
  kwargs = {"device": compiled.device, "requires_grad": requires_grad,
            "retain_grad": requires_grad}
  stages = []
  for _ in range(compiled.max_depth + 2):
    position = wp.empty((worlds, compiled.body_count), dtype=wp.vec3, **kwargs)
    quaternion = wp.empty((worlds, compiled.body_count), dtype=wp.quat, **kwargs)
    stages.append(BodyPose(position, quaternion))
  stages = tuple(stages)
  stages[0].position.zero_()
  identity = np.zeros((worlds, compiled.body_count, 4), dtype=np.float32)
  identity[..., 0] = 1.0
  stages[0].quaternion.assign(identity)
  final = stages[-1]
  matrix = wp.empty((worlds, compiled.body_count), dtype=wp.mat33, **kwargs)
  anchor = wp.empty((worlds, compiled.joint_count), dtype=wp.vec3, **kwargs)
  axis = wp.empty((worlds, compiled.joint_count), dtype=wp.vec3, **kwargs)
  joint_quaternion = wp.empty(
      (worlds, compiled.joint_count), dtype=wp.quat,
      device=compiled.device)
  return KinematicOutput(stages, final.position, final.quaternion,
                         matrix, anchor, axis, joint_quaternion)


def kinematics(compiled: CompiledKinematics, qpos: wp.array,
               output: KinematicOutput) -> None:
  for depth in range(compiled.max_depth + 1):
    source = output.stages[depth]
    target = output.stages[depth + 1]
    level = KinematicLevelInput()
    level.model = compiled.model
    level.qpos = qpos
    level.depth = depth
    level.position = source.position
    level.quaternion = source.quaternion
    target_view = PoseOutput()
    target_view.position = target.position
    target_view.quaternion = target.quaternion
    wp.launch(_kinematic_level, dim=(qpos.shape[0], compiled.body_count),
              inputs=[level], outputs=[target_view],
              device=compiled.device)
  wp.launch(_body_matrices, dim=(qpos.shape[0], compiled.body_count),
            inputs=[output.body_quaternion], outputs=[output.body_matrix],
            device=compiled.device)
  joint_input = JointFrameInput()
  joint_input.model = compiled.model
  joint_input.qpos = qpos
  joint_input.body_position = output.body_position
  joint_input.body_quaternion = output.body_quaternion
  joint_output = JointFrameOutput()
  joint_output.anchor = output.joint_anchor
  joint_output.axis = output.joint_axis
  joint_output.quaternion = output.joint_quaternion
  wp.launch(_joint_frames, dim=(qpos.shape[0], compiled.body_count),
            inputs=[joint_input], outputs=[joint_output],
            device=compiled.device)
