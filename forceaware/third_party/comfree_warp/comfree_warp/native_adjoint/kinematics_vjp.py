"""Analytic tree-kinematics VJP for the native differentiable step.

The authoritative simulator forward is unchanged.  The VJP consumes the body
and joint frames produced by that forward and contracts their cotangents with
closed-form spatial Jacobians.  It avoids reverse differentiation through the
depth-staged kinematics implementation, whose copied quaternion stages do not
produce a reliable long-chain adjoint in Warp 1.15.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from comfree_warp.mujoco_warp._src import math

from .kinematics import KinematicOutput


FREE = int(mujoco.mjtJoint.mjJNT_FREE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
KINEMATICS_VJP_PROFILE = "analytic_tree_spatial_jacobian"


@wp.struct
class KinematicsVjpModel:
  coordinate_joint: wp.array(dtype=int)
  coordinate_component: wp.array(dtype=int)
  joint_type: wp.array(dtype=int)
  joint_qpos_address: wp.array(dtype=int)
  joint_body: wp.array(dtype=int)
  body_affected: wp.array2d(dtype=int)
  joint_affected: wp.array2d(dtype=int)
  body_count: int
  joint_count: int


@wp.struct
class KinematicsVjpSeed:
  body_position: wp.array2d(dtype=wp.vec3)
  body_quaternion: wp.array2d(dtype=wp.quat)
  body_matrix: wp.array2d(dtype=wp.mat33)
  joint_anchor: wp.array2d(dtype=wp.vec3)
  joint_axis: wp.array2d(dtype=wp.vec3)


@wp.struct
class KinematicsVjpPose:
  body_position: wp.array2d(dtype=wp.vec3)
  body_quaternion: wp.array2d(dtype=wp.quat)
  body_matrix: wp.array2d(dtype=wp.mat33)
  joint_anchor: wp.array2d(dtype=wp.vec3)
  joint_axis: wp.array2d(dtype=wp.vec3)
  joint_quaternion: wp.array2d(dtype=wp.quat)


@dataclass(frozen=True)
class CompiledKinematicsVjp:
  model: KinematicsVjpModel
  qpos_count: int
  device: object


@wp.func
def _basis(index: int) -> wp.vec3:
  if index == 0:
    return wp.vec3(1.0, 0.0, 0.0)
  if index == 1:
    return wp.vec3(0.0, 1.0, 0.0)
  return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _quat_dot(left: wp.quat, right: wp.quat) -> float:
  return (left[0] * right[0] + left[1] * right[1]
          + left[2] * right[2] + left[3] * right[3])


@wp.func
def _matrix_column(matrix: wp.mat33, column: int) -> wp.vec3:
  return wp.vec3(
      matrix[0, column], matrix[1, column], matrix[2, column])


@wp.func
def _matrix_cotangent(seed: wp.mat33, matrix: wp.mat33,
                      angular: wp.vec3) -> float:
  value = float(0.0)
  for column in range(3):
    derivative = wp.cross(angular, _matrix_column(matrix, column))
    value += seed[0, column] * derivative[0]
    value += seed[1, column] * derivative[1]
    value += seed[2, column] * derivative[2]
  return value


@wp.func
def _quaternion_cotangent(seed: wp.quat, quaternion: wp.quat,
                          angular: wp.vec3) -> float:
  tangent = math.mul_quat(
      wp.quat(0.0, angular[0], angular[1], angular[2]), quaternion)
  return 0.5 * _quat_dot(seed, tangent)


@wp.func
def _body_cotangent(pose: KinematicsVjpPose, seed: KinematicsVjpSeed,
                    world: int, body: int, linear: wp.vec3,
                    angular: wp.vec3) -> float:
  value = wp.dot(seed.body_position[world, body], linear)
  value += _quaternion_cotangent(
      seed.body_quaternion[world, body],
      pose.body_quaternion[world, body], angular)
  value += _matrix_cotangent(
      seed.body_matrix[world, body], pose.body_matrix[world, body], angular)
  return value


@wp.func
def _joint_cotangent(pose: KinematicsVjpPose, seed: KinematicsVjpSeed,
                     world: int, joint: int, linear: wp.vec3,
                     angular: wp.vec3) -> float:
  value = wp.dot(seed.joint_anchor[world, joint], linear)
  axis_derivative = wp.cross(angular, pose.joint_axis[world, joint])
  value += wp.dot(seed.joint_axis[world, joint], axis_derivative)
  return value


@wp.func
def _quaternion_angular(qpos: wp.array2d(dtype=float), world: int,
                        address: int, component: int) -> wp.vec3:
  raw = wp.quat(
      qpos[world, address], qpos[world, address + 1],
      qpos[world, address + 2], qpos[world, address + 3])
  norm = wp.sqrt(_quat_dot(raw, raw))
  quaternion = raw / norm
  selected = quaternion[component]
  derivative = wp.quat(
      -quaternion[0] * selected,
      -quaternion[1] * selected,
      -quaternion[2] * selected,
      -quaternion[3] * selected) / norm
  derivative[component] += 1.0 / norm
  vector = (-derivative[0] * wp.vec3(
      quaternion[1], quaternion[2], quaternion[3]))
  vector += quaternion[0] * wp.vec3(
      derivative[1], derivative[2], derivative[3])
  vector += wp.cross(
      wp.vec3(quaternion[1], quaternion[2], quaternion[3]),
      wp.vec3(derivative[1], derivative[2], derivative[3]))
  return 2.0 * vector


@wp.func
def _scalar_coordinate(model: KinematicsVjpModel,
                       pose: KinematicsVjpPose,
                       seed: KinematicsVjpSeed,
                       world: int, coordinate: int, joint: int,
                       joint_type: int) -> float:
  axis = pose.joint_axis[world, joint]
  anchor = pose.joint_anchor[world, joint]
  angular = wp.vec3(0.0)
  translation = axis
  if joint_type == HINGE:
    angular = axis
    translation = wp.vec3(0.0)
  value = float(0.0)
  for body in range(model.body_count):
    if model.body_affected[coordinate, body] != 0:
      linear = translation
      if joint_type == HINGE:
        offset = pose.body_position[world, body] - anchor
        linear = wp.cross(angular, offset)
      value += _body_cotangent(
          pose, seed, world, body, linear, angular)
  for target in range(model.joint_count):
    if model.joint_affected[coordinate, target] != 0:
      linear = translation
      if joint_type == HINGE:
        offset = pose.joint_anchor[world, target] - anchor
        linear = wp.cross(angular, offset)
      value += _joint_cotangent(
          pose, seed, world, target, linear, angular)
  return value


@wp.func
def _rotation_coordinate(model: KinematicsVjpModel,
                         pose: KinematicsVjpPose,
                         seed: KinematicsVjpSeed,
                         world: int, coordinate: int,
                         angular: wp.vec3,
                         pivot: wp.vec3) -> float:
  value = float(0.0)
  for body in range(model.body_count):
    if model.body_affected[coordinate, body] != 0:
      offset = pose.body_position[world, body] - pivot
      linear = wp.cross(angular, offset)
      value += _body_cotangent(
          pose, seed, world, body, linear, angular)
  for target in range(model.joint_count):
    if model.joint_affected[coordinate, target] != 0:
      offset = pose.joint_anchor[world, target] - pivot
      linear = wp.cross(angular, offset)
      value += _joint_cotangent(
          pose, seed, world, target, linear, angular)
  return value


@wp.func
def _free_coordinate(model: KinematicsVjpModel,
                     pose: KinematicsVjpPose,
                     seed: KinematicsVjpSeed,
                     qpos: wp.array2d(dtype=float),
                     world: int, coordinate: int, joint: int,
                     component: int) -> float:
  body_root = model.joint_body[joint]
  if component >= 3:
    angular = _quaternion_angular(
        qpos, world, model.joint_qpos_address[joint] + 3,
        component - 3)
    return _rotation_coordinate(
        model, pose, seed, world, coordinate, angular,
        pose.body_position[world, body_root])
  translation = _basis(component)
  value = wp.dot(seed.joint_anchor[world, joint], translation)
  for body in range(model.body_count):
    if model.body_affected[coordinate, body] != 0:
      value += _body_cotangent(
          pose, seed, world, body, translation, wp.vec3(0.0))
  for target in range(model.joint_count):
    if model.joint_affected[coordinate, target] != 0:
      value += _joint_cotangent(
          pose, seed, world, target, translation, wp.vec3(0.0))
  return value


@wp.func
def _ball_coordinate(model: KinematicsVjpModel,
                     pose: KinematicsVjpPose,
                     seed: KinematicsVjpSeed,
                     qpos: wp.array2d(dtype=float),
                     world: int, coordinate: int, joint: int,
                     component: int) -> float:
  local_angular = _quaternion_angular(
      qpos, world, model.joint_qpos_address[joint], component)
  angular = math.rot_vec_quat(
      local_angular, pose.joint_quaternion[world, joint])
  return _rotation_coordinate(
      model, pose, seed, world, coordinate, angular,
      pose.joint_anchor[world, joint])


@wp.kernel(enable_backward=False)
def _kinematics_vjp(model: KinematicsVjpModel,
                    pose: KinematicsVjpPose,
                    seed: KinematicsVjpSeed,
                    qpos: wp.array2d(dtype=float),
                    output: wp.array2d(dtype=float)):
  world, coordinate = wp.tid()
  joint = model.coordinate_joint[coordinate]
  joint_type = model.joint_type[joint]
  component = model.coordinate_component[coordinate]
  value = float(0.0)
  if joint_type == SLIDE or joint_type == HINGE:
    value = _scalar_coordinate(
        model, pose, seed, world, coordinate, joint, joint_type)
  elif joint_type == FREE:
    value = _free_coordinate(
        model, pose, seed, qpos, world, coordinate, joint, component)
  elif joint_type == BALL:
    value = _ball_coordinate(
        model, pose, seed, qpos, world, coordinate, joint, component)
  output[world, coordinate] += value


def _is_ancestor(model: mujoco.MjModel, ancestor: int, body: int) -> bool:
  current = body
  while True:
    if current == ancestor:
      return True
    parent = int(model.body_parentid[current])
    if parent == current:
      return False
    current = parent


def _joint_order(model: mujoco.MjModel, joint: int) -> int:
  body = int(model.jnt_bodyid[joint])
  return joint - int(model.body_jntadr[body])


def _joint_precedes(model: mujoco.MjModel, source: int, target: int) -> bool:
  source_body = int(model.jnt_bodyid[source])
  target_body = int(model.jnt_bodyid[target])
  if source_body == target_body:
    return _joint_order(model, source) < _joint_order(model, target)
  return _is_ancestor(model, source_body, target_body)


def _coordinate_layout(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  joint_by_coordinate = np.full(model.nq, -1, dtype=np.int32)
  component = np.full(model.nq, -1, dtype=np.int32)
  widths = {FREE: 7, BALL: 4, SLIDE: 1, HINGE: 1}
  for joint in range(model.njnt):
    joint_type = int(model.jnt_type[joint])
    width = widths[joint_type]
    address = int(model.jnt_qposadr[joint])
    joint_by_coordinate[address:address + width] = joint
    component[address:address + width] = np.arange(width, dtype=np.int32)
  if np.any(joint_by_coordinate < 0):
    raise ValueError("every qpos coordinate must belong to a joint")
  return joint_by_coordinate, component


def compile_kinematics_vjp(
    cpu_model: mujoco.MjModel,
    device,
) -> CompiledKinematicsVjp:
  joint_by_coordinate, component = _coordinate_layout(cpu_model)
  joint_types = np.asarray(cpu_model.jnt_type, dtype=np.int32)
  body_affected = np.zeros((cpu_model.nq, cpu_model.nbody), np.int32)
  joint_affected = np.zeros((cpu_model.nq, cpu_model.njnt), np.int32)
  for coordinate, joint in enumerate(joint_by_coordinate):
    source_body = int(cpu_model.jnt_bodyid[joint])
    for body in range(cpu_model.nbody):
      body_affected[coordinate, body] = _is_ancestor(
          cpu_model, source_body, body)
    for target in range(cpu_model.njnt):
      joint_affected[coordinate, target] = _joint_precedes(
          cpu_model, int(joint), target)
  model = KinematicsVjpModel()
  model.coordinate_joint = wp.array(
      joint_by_coordinate, dtype=int, device=device)
  model.coordinate_component = wp.array(component, dtype=int, device=device)
  model.joint_type = wp.array(joint_types, dtype=int, device=device)
  model.joint_qpos_address = wp.array(
      np.asarray(cpu_model.jnt_qposadr, np.int32), dtype=int, device=device)
  model.joint_body = wp.array(
      np.asarray(cpu_model.jnt_bodyid, np.int32), dtype=int, device=device)
  model.body_affected = wp.array(body_affected, dtype=int, device=device)
  model.joint_affected = wp.array(joint_affected, dtype=int, device=device)
  model.body_count = cpu_model.nbody
  model.joint_count = cpu_model.njnt
  return CompiledKinematicsVjp(model, cpu_model.nq, device)


def add_kinematics_vjp(
    compiled: CompiledKinematicsVjp,
    pose: KinematicOutput,
    qpos: wp.array,
    *,
    output: wp.array,
) -> None:
  pose_view = KinematicsVjpPose()
  pose_view.body_position = pose.body_position
  pose_view.body_quaternion = pose.body_quaternion
  pose_view.body_matrix = pose.body_matrix
  pose_view.joint_anchor = pose.joint_anchor
  pose_view.joint_axis = pose.joint_axis
  pose_view.joint_quaternion = pose.joint_quaternion
  seed = KinematicsVjpSeed()
  seed.body_position = pose.body_position.grad
  seed.body_quaternion = pose.body_quaternion.grad
  seed.body_matrix = pose.body_matrix.grad
  seed.joint_anchor = pose.joint_anchor.grad
  seed.joint_axis = pose.joint_axis.grad
  wp.launch(
      _kinematics_vjp,
      dim=(qpos.shape[0], compiled.qpos_count),
      inputs=[compiled.model, pose_view, seed, qpos],
      outputs=[output],
      device=compiled.device,
  )
