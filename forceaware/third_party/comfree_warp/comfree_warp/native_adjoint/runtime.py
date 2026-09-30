"""Compiled model and workspace assembly for the native Warp step."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from .contact_rows import ContactMotion
from .contact_rows import ContactRowBuildInput
from .contact_rows import allocate_workspace as allocate_rows
from .contact_rows import build_rows
from .contact_rows import compile_contact_rows
from .explicit_damp import allocate_workspace as allocate_solver
from .explicit_damp import compile_explicit_damp
from .sparse_explicit_damp import allocate_workspace as allocate_sparse_solver
from .sparse_explicit_damp import compile_sparse_explicit_damp
from .gaussian_collision import CollisionAllocation
from .gaussian_collision import CollisionFrames
from .gaussian_collision import allocate_workspace as allocate_collision
from .gaussian_collision import compile_collision
from .gaussian_collision import narrowphase
from .local_frames import BodyPose as LocalFramePose
from .local_frames import allocate_local_frames
from .local_frames import compile_geom_frames
from .local_frames import local_frames

from .dynamics import DynamicsInput
from .dynamics import allocate_output as allocate_dynamics
from .dynamics import compile_dynamics
from .integration import allocate_output as allocate_integration
from .integration import compile_integration
from .kinematics import allocate_output as allocate_kinematics
from .kinematics import compile_kinematics
from .kinematics_vjp import compile_kinematics_vjp
from .contact_rows_vjp import compile_contact_rows_vjp
from .sparse_contact_rows_vjp import compile_sparse_contact_rows_vjp


@dataclass(frozen=True)
class CompiledStep:
  kinematics: object
  kinematics_vjp: object
  dynamics: object
  geom_frames: object
  collision: object
  contact_rows: object
  contact_rows_vjp: object
  contact_solver: object
  integration: object
  sparse_contact: bool
  device: object


@dataclass(frozen=True)
class StepInput:
  dynamics: DynamicsInput
  time: wp.array
  contact_softness: wp.array


@dataclass(frozen=True)
class StepWorkspace:
  pose: object
  dynamics: object
  geom_frames: object
  collision: object
  contact_rows: object
  contact_solver: object
  integrated: object


@dataclass(frozen=True)
class StepResult:
  qpos: wp.array
  qvel: wp.array
  qacc: wp.array
  time: wp.array
  constraint_force: wp.array
  body_position: wp.array
  body_matrix: wp.array
  contact_distance: wp.array
  contact_position: wp.array
  contact_frame: wp.array


def compile_step(cpu_model: mujoco.MjModel, device_model) -> CompiledStep:
  kinematics = compile_kinematics(cpu_model, device_model)
  kinematics_vjp = compile_kinematics_vjp(cpu_model, device_model.qpos0.device)
  dynamics = compile_dynamics(cpu_model, device_model)
  geom_frames = compile_geom_frames(cpu_model, device_model)
  collision = compile_collision(device_model)
  rows = compile_contact_rows(
      cpu_model, device_model, collision.contact_layout)
  solver = _compile_contact_solver(
      device_model, collision.contact_layout, rows)
  integration = compile_integration(cpu_model, device_model)
  compile_rows_vjp = (
      compile_sparse_contact_rows_vjp if rows.sparse
      else compile_contact_rows_vjp)
  rows_vjp = compile_rows_vjp(cpu_model, rows)
  return CompiledStep(
      kinematics, kinematics_vjp, dynamics, geom_frames, collision, rows,
      rows_vjp, solver, integration, rows.sparse, device_model.qpos0.device)


def _compile_contact_solver(device_model, collision, rows):
  if rows.sparse:
    return compile_sparse_explicit_damp(device_model, collision, rows)
  return compile_explicit_damp(device_model, collision, rows)


def allocate_workspace(compiled: CompiledStep, worlds: int, *,
                       requires_grad: bool) -> StepWorkspace:
  collision_allocation = CollisionAllocation(
      worlds, compiled.device, requires_grad)
  return StepWorkspace(
      allocate_kinematics(
          compiled.kinematics, worlds, requires_grad=requires_grad),
      allocate_dynamics(
          compiled.dynamics, worlds, requires_grad=requires_grad),
      allocate_local_frames(
          compiled.geom_frames, worlds, gradient=requires_grad),
      allocate_collision(compiled.collision, collision_allocation),
      allocate_rows(compiled.contact_rows, worlds, gradient=requires_grad),
      _allocate_contact_solver(compiled, worlds, gradient=requires_grad),
      allocate_integration(
          compiled.integration, worlds, requires_grad=requires_grad),
  )


def _allocate_contact_solver(compiled: CompiledStep, worlds: int, *, gradient: bool):
  allocate = allocate_sparse_solver if compiled.sparse_contact else allocate_solver
  return allocate(compiled.contact_solver, worlds, gradient=gradient)


def collision_frames(compiled: CompiledStep,
                     workspace: StepWorkspace) -> CollisionFrames:
  pose = LocalFramePose(
      workspace.pose.body_position, workspace.pose.body_quaternion)
  local_frames(compiled.geom_frames, pose, workspace.geom_frames)
  return CollisionFrames(
      workspace.pose.body_position, workspace.pose.body_matrix,
      workspace.geom_frames.position, workspace.geom_frames.matrix)


def collision_contacts(compiled: CompiledStep, workspace: StepWorkspace, *,
                       freeze_frame_vjp: bool = False):
  return narrowphase(
      compiled.collision,
      workspace.collision,
      freeze_frame_vjp=freeze_frame_vjp,
  )


def contact_motion(workspace: StepWorkspace) -> ContactMotion:
  motion = ContactMotion()
  motion.joint_anchor = workspace.pose.joint_anchor
  motion.joint_axis = workspace.pose.joint_axis
  motion.body_matrix = workspace.pose.body_matrix
  return motion


def build_contact_rows(compiled: CompiledStep, workspace: StepWorkspace,
                       contacts, motion, qpos):
  inputs = ContactRowBuildInput(contacts, motion, qpos)
  return build_rows(compiled.contact_rows, inputs, workspace.contact_rows)


def contact_rows(compiled: CompiledStep, workspace: StepWorkspace, qpos, *,
                 freeze_frame_vjp: bool = False):
  contacts = collision_contacts(
      compiled, workspace, freeze_frame_vjp=freeze_frame_vjp)
  return build_contact_rows(
      compiled, workspace, contacts, contact_motion(workspace), qpos)
