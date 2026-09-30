"""Differentiable kinematics and collision observation without a physics step."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .geometry_vjp import GeometryTape
from .gaussian_collision import CollisionAllocation
from .gaussian_collision import CollisionFrames
from .gaussian_collision import allocate_workspace as allocate_collision
from .gaussian_collision import broadphase
from .gaussian_collision import narrowphase
from .local_frames import BodyPose
from .local_frames import allocate_local_frames
from .local_frames import local_frames

from .kinematics import allocate_output as allocate_kinematics
from .kinematics import kinematics
from .kinematics_vjp import add_kinematics_vjp
from .runtime import CompiledStep


@dataclass(frozen=True)
class ObservationResult:
  body_position: wp.array
  body_matrix: wp.array
  contact_distance: wp.array
  contact_position: wp.array
  contact_frame: wp.array


@dataclass(frozen=True)
class ObservationWorkspace:
  pose: object
  geom_frames: object
  collision: object


@dataclass(frozen=True)
class RecordedObservation:
  pose_tape: wp.Tape
  collision_tape: wp.Tape
  result: ObservationResult
  kinematics_vjp: object


@dataclass(frozen=True)
class ObservationCotangent:
  body_position: wp.array
  body_matrix: wp.array
  contact_distance: wp.array
  contact_position: wp.array
  contact_frame: wp.array


@wp.kernel(enable_backward=False)
def _add_vec3(source: wp.array2d(dtype=wp.vec3),
              target: wp.array2d(dtype=wp.vec3)):
  world, index = wp.tid()
  target[world, index] += source[world, index]


@wp.kernel(enable_backward=False)
def _add_mat33(source: wp.array2d(dtype=wp.mat33),
               target: wp.array2d(dtype=wp.mat33)):
  world, index = wp.tid()
  target[world, index] += source[world, index]


def allocate_workspace(compiled: CompiledStep,
                       worlds: int, *,
                       requires_grad: bool = True) -> ObservationWorkspace:
  pose = allocate_kinematics(
      compiled.kinematics, worlds, requires_grad=requires_grad)
  frames = allocate_local_frames(
      compiled.geom_frames, worlds, gradient=requires_grad)
  collision = allocate_collision(
      compiled.collision,
      CollisionAllocation(worlds, compiled.device, requires_grad))
  return ObservationWorkspace(pose, frames, collision)


def _pose_downstream(compiled: CompiledStep,
                     workspace: ObservationWorkspace) -> CollisionFrames:
  pose = BodyPose(
      workspace.pose.body_position, workspace.pose.body_quaternion)
  local_frames(compiled.geom_frames, pose, workspace.geom_frames)
  return CollisionFrames(
      workspace.pose.body_position, workspace.pose.body_matrix,
      workspace.geom_frames.position, workspace.geom_frames.matrix)


def _pose_forward(compiled: CompiledStep, qpos,
                  workspace: ObservationWorkspace) -> CollisionFrames:
  kinematics(compiled.kinematics, qpos, workspace.pose)
  return _pose_downstream(compiled, workspace)


def _result(workspace: ObservationWorkspace) -> ObservationResult:
  contacts = workspace.collision.contacts
  return ObservationResult(
      workspace.pose.body_position, workspace.pose.body_matrix,
      contacts.distance, contacts.position, contacts.frame)


def record(compiled: CompiledStep, qpos,
           workspace: ObservationWorkspace, *,
           freeze_frame_vjp: bool = False) -> RecordedObservation:
  kinematics(compiled.kinematics, qpos, workspace.pose)
  pose_tape = GeometryTape()
  with pose_tape:
    frames = _pose_downstream(compiled, workspace)
  broadphase(compiled.collision, frames, workspace.collision)
  collision_tape = GeometryTape()
  with collision_tape:
    narrowphase(
        compiled.collision,
        workspace.collision,
        freeze_frame_vjp=freeze_frame_vjp,
    )
  return RecordedObservation(
      pose_tape, collision_tape, _result(workspace), compiled.kinematics_vjp)


def forward(compiled: CompiledStep, qpos,
            workspace: ObservationWorkspace) -> ObservationResult:
  frames = _pose_forward(compiled, qpos, workspace)
  broadphase(compiled.collision, frames, workspace.collision)
  narrowphase(compiled.collision, workspace.collision)
  return _result(workspace)


def allocate_cotangent(compiled: CompiledStep,
                       worlds: int) -> ObservationCotangent:
  zeros = lambda shape, dtype: wp.zeros(
      shape, dtype=dtype, device=compiled.device)
  bodies = (worlds, compiled.kinematics.body_count)
  contacts = (worlds, compiled.collision.contact_count)
  return ObservationCotangent(
      zeros(bodies, wp.vec3), zeros(bodies, wp.mat33),
      zeros(contacts, float), zeros(contacts, wp.vec3),
      zeros(contacts, wp.mat33))


def zero_cotangent(cotangent: ObservationCotangent) -> None:
  for value in (
      cotangent.body_position, cotangent.body_matrix,
      cotangent.contact_distance, cotangent.contact_position,
      cotangent.contact_frame):
    value.zero_()


def _zero_gradients(qpos, recorded: RecordedObservation) -> None:
  recorded.pose_tape.zero()
  recorded.collision_tape.zero()
  if qpos.grad is not None:
    qpos.grad.zero_()
  result = recorded.result
  for value in (
      result.body_position, result.body_matrix, result.contact_distance,
      result.contact_position, result.contact_frame):
    if value.grad is not None:
      value.grad.zero_()


def backward(qpos, workspace: ObservationWorkspace, *,
             recorded: RecordedObservation,
             cotangent: ObservationCotangent) -> wp.array:
  _zero_gradients(qpos, recorded)
  contacts = workspace.collision.contacts
  recorded.collision_tape.backward(grads={
      contacts.distance: cotangent.contact_distance,
      contacts.position: cotangent.contact_position,
      contacts.frame: cotangent.contact_frame,
  })
  pose = workspace.pose
  wp.launch(_add_vec3, dim=pose.body_position.shape,
            inputs=[cotangent.body_position], outputs=[pose.body_position.grad])
  wp.launch(_add_mat33, dim=pose.body_matrix.shape,
            inputs=[cotangent.body_matrix], outputs=[pose.body_matrix.grad])
  recorded.pose_tape.backward(grads={
      pose.body_position: pose.body_position.grad,
      pose.body_matrix: pose.body_matrix.grad,
      workspace.geom_frames.position: workspace.geom_frames.position.grad,
      workspace.geom_frames.matrix: workspace.geom_frames.matrix.grad,
  })
  add_kinematics_vjp(
      recorded.kinematics_vjp,
      pose,
      qpos,
      output=qpos.grad,
  )
  return qpos.grad
