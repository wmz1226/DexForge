"""Alias-free semi-implicit Euler integration for MuJoCo joint states."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from comfree_warp.mujoco_warp._src import math
from comfree_warp.mujoco_warp._src.types import DisableBit


_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)


@wp.struct
class IntegrationModel:
  timestep: wp.array(dtype=float)
  dof_damping: wp.array2d(dtype=float)
  joint_type: wp.array(dtype=int)
  joint_qpos_address: wp.array(dtype=int)
  joint_dof_address: wp.array(dtype=int)


@wp.struct
class IntegrationInput:
  qpos: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)
  qacc: wp.array2d(dtype=float)
  time: wp.array(dtype=float)


@wp.struct
class PositionInput:
  qpos: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)


@dataclass(frozen=True)
class CompiledIntegration:
  model: IntegrationModel
  position_count: int
  velocity_count: int
  joint_count: int
  implicit_damping: bool
  device: object


@dataclass(frozen=True)
class IntegratedState:
  qpos: wp.array
  qvel: wp.array
  time: wp.array


@wp.func
def _quaternion(array: wp.array2d(dtype=float), world: int,
                address: int) -> wp.quat:
  return wp.quat(array[world, address], array[world, address + 1],
                 array[world, address + 2], array[world, address + 3])


@wp.func
def _angular_velocity(array: wp.array2d(dtype=float), world: int,
                      address: int) -> wp.vec3:
  return wp.vec3(array[world, address], array[world, address + 1],
                 array[world, address + 2])


@wp.kernel
def _advance_velocity(model: IntegrationModel, inputs: IntegrationInput,
                      qvel: wp.array2d(dtype=float)):
  world, dof = wp.tid()
  timestep = model.timestep[world % model.timestep.shape[0]]
  qvel[world, dof] = (
      inputs.qvel[world, dof] + timestep * inputs.qacc[world, dof])


@wp.kernel(enable_backward=False)
def _euler_mass(model: IntegrationModel,
                mass: wp.array3d(dtype=float),
                output: wp.array3d(dtype=float)):
  world, row, column = wp.tid()
  value = mass[world, row, column]
  if row == column:
    model_row = world % model.dof_damping.shape[0]
    timestep = model.timestep[world % model.timestep.shape[0]]
    value += timestep * model.dof_damping[model_row, row]
  output[world, row, column] = value


@wp.kernel
def _advance_position(model: IntegrationModel, inputs: PositionInput,
                      qpos_out: wp.array2d(dtype=float)):
  world, joint = wp.tid()
  timestep = model.timestep[world % model.timestep.shape[0]]
  qpos = model.joint_qpos_address[joint]
  dof = model.joint_dof_address[joint]
  joint_type = model.joint_type[joint]
  if joint_type == _FREE:
    for component in range(3):
      qpos_out[world, qpos + component] = (
          inputs.qpos[world, qpos + component]
          + timestep * inputs.qvel[world, dof + component])
    quaternion = math.quat_integrate(
        _quaternion(inputs.qpos, world, qpos + 3),
        _angular_velocity(inputs.qvel, world, dof + 3), timestep)
    qpos_out[world, qpos + 3] = quaternion[0]
    qpos_out[world, qpos + 4] = quaternion[1]
    qpos_out[world, qpos + 5] = quaternion[2]
    qpos_out[world, qpos + 6] = quaternion[3]
  elif joint_type == _BALL:
    quaternion = math.quat_integrate(
        _quaternion(inputs.qpos, world, qpos),
        _angular_velocity(inputs.qvel, world, dof), timestep)
    qpos_out[world, qpos] = quaternion[0]
    qpos_out[world, qpos + 1] = quaternion[1]
    qpos_out[world, qpos + 2] = quaternion[2]
    qpos_out[world, qpos + 3] = quaternion[3]
  else:
    qpos_out[world, qpos] = (
        inputs.qpos[world, qpos] + timestep * inputs.qvel[world, dof])


@wp.kernel
def _advance_time(model: IntegrationModel, time_in: wp.array(dtype=float),
                  time_out: wp.array(dtype=float)):
  world = wp.tid()
  timestep = model.timestep[world % model.timestep.shape[0]]
  time_out[world] = time_in[world] + timestep


def compile_integration(cpu_model: mujoco.MjModel,
                        device_model) -> CompiledIntegration:
  model = IntegrationModel()
  model.timestep = device_model.opt.timestep
  model.dof_damping = device_model.dof_damping
  model.joint_type = device_model.jnt_type
  model.joint_qpos_address = device_model.jnt_qposadr
  model.joint_dof_address = device_model.jnt_dofadr
  disabled = int(DisableBit.EULERDAMP | DisableBit.DAMPER)
  implicit_damping = not bool(int(cpu_model.opt.disableflags) & disabled)
  return CompiledIntegration(
      model, cpu_model.nq, cpu_model.nv, cpu_model.njnt,
      implicit_damping, device_model.qpos0.device)


def allocate_output(compiled: CompiledIntegration, worlds: int,
                    *, requires_grad: bool) -> IntegratedState:
  kwargs = {"device": compiled.device, "requires_grad": requires_grad,
            "retain_grad": requires_grad}
  qpos = wp.empty((worlds, compiled.position_count), dtype=float, **kwargs)
  qvel = wp.empty((worlds, compiled.velocity_count), dtype=float, **kwargs)
  time = wp.empty(worlds, dtype=float, **kwargs)
  return IntegratedState(qpos, qvel, time)


def build_euler_mass(compiled: CompiledIntegration, mass: wp.array,
                     output: wp.array) -> None:
  wp.launch(
      _euler_mass, dim=mass.shape,
      inputs=[compiled.model, mass], outputs=[output],
      device=compiled.device)


def integrate(compiled: CompiledIntegration, inputs: IntegrationInput,
              output: IntegratedState) -> None:
  worlds = inputs.qpos.shape[0]
  wp.launch(_advance_velocity, dim=(worlds, compiled.velocity_count),
            inputs=[compiled.model, inputs], outputs=[output.qvel])
  position = PositionInput()
  position.qpos = inputs.qpos
  position.qvel = output.qvel
  wp.launch(_advance_position, dim=(worlds, compiled.joint_count),
            inputs=[compiled.model, position], outputs=[output.qpos])
  wp.launch(_advance_time, dim=worlds,
            inputs=[compiled.model, inputs.time], outputs=[output.time])
