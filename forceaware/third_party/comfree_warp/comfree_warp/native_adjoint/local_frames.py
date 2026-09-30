"""Differentiable body-local geometry frames for the native Warp step."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from comfree_warp.mujoco_warp._src import math


@wp.struct
class PoseView:
  position: wp.array2d(dtype=wp.vec3)
  quaternion: wp.array2d(dtype=wp.quat)


@wp.struct
class LocalFrameParameters:
  body_id: wp.array(dtype=int)
  position: wp.array2d(dtype=wp.vec3)
  quaternion: wp.array2d(dtype=wp.quat)


@wp.struct
class LocalFrameJob:
  body_pose: PoseView
  position: wp.array2d(dtype=wp.vec3)
  matrix: wp.array2d(dtype=wp.mat33)


@dataclass(frozen=True)
class BodyPose:
  position: wp.array
  quaternion: wp.array


@dataclass(frozen=True)
class CompiledLocalFrames:
  parameters: LocalFrameParameters
  count: int
  device: object


@dataclass(frozen=True)
class LocalFrames:
  position: wp.array
  matrix: wp.array


@wp.kernel
def _local_frames(params: LocalFrameParameters, job: LocalFrameJob):
  world, frame = wp.tid()
  row = world % params.position.shape[0]
  body = params.body_id[frame]
  body_position = job.body_pose.position[world, body]
  body_quaternion = job.body_pose.quaternion[world, body]
  job.position[world, frame] = body_position + math.rot_vec_quat(
      params.position[row, frame], body_quaternion)
  quaternion = math.mul_quat(
      body_quaternion, params.quaternion[row, frame])
  job.matrix[world, frame] = math.quat_to_mat(quaternion)


def compile_geom_frames(cpu_model: mujoco.MjModel,
                        device_model) -> CompiledLocalFrames:
  params = LocalFrameParameters()
  params.body_id = device_model.geom_bodyid
  params.position = device_model.geom_pos
  params.quaternion = device_model.geom_quat
  return CompiledLocalFrames(
      params, cpu_model.ngeom, device_model.qpos0.device)


def allocate_local_frames(compiled: CompiledLocalFrames, worlds: int, *,
                          gradient: bool) -> LocalFrames:
  options = {
      "device": compiled.device,
      "requires_grad": gradient,
      "retain_grad": gradient,
  }
  position = wp.empty(
      (worlds, compiled.count), dtype=wp.vec3, **options)
  matrix = wp.empty(
      (worlds, compiled.count), dtype=wp.mat33, **options)
  return LocalFrames(position, matrix)


def local_frames(compiled: CompiledLocalFrames, pose: BodyPose,
                 output: LocalFrames) -> None:
  view = PoseView()
  view.position = pose.position
  view.quaternion = pose.quaternion
  job = LocalFrameJob()
  job.body_pose = view
  job.position = output.position
  job.matrix = output.matrix
  wp.launch(
      _local_frames, dim=(pose.position.shape[0], compiled.count),
      inputs=[compiled.parameters, job], device=compiled.device)
