"""Segmented native-Warp adjoint for the fast ComFree simulation step."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .gaussian_collision import broadphase

from .annealed_contact_adjoint import ContactBackwardInput
from .annealed_contact_adjoint import allocate as allocate_contact_adjoint
from .annealed_contact_adjoint import backward as backward_contact
from .dynamics import dynamics
from .dynamics_vjp import DynamicsTape
from .geometry_vjp import GeometryTape
from .fast_mass import backward as backward_mass
from .fast_mass import solve as solve_mass
from .fast_step import CompiledFastStep
from .fast_step import FastStepWorkspace
from .fast_step import contact_input
from .fast_step import allocate_workspace as allocate_forward
from .fast_step import contact_coefficients
from .fast_step import integration_acceleration
from .fast_step import prepare_contact_system
from .fast_step import reduce_contact_rows
from .fast_step import reduce_contact_rows_vjp
from .fast_step import solve_contact_system
from .integration import IntegrationInput
from .integration import integrate
from .kinematics import kinematics
from .kinematics_vjp import add_kinematics_vjp
from .runtime import StepInput
from .runtime import StepResult
from .runtime import build_contact_rows
from .runtime import collision_contacts
from .runtime import collision_frames
from .runtime import contact_motion
from .runtime import contact_rows
from . import contact_rows_vjp
from . import sparse_contact_rows_vjp
from .sparse_annealed_contact_adjoint import SparseContactBackwardInput
from .sparse_annealed_contact_adjoint import allocate as allocate_sparse_contact_adjoint
from .sparse_annealed_contact_adjoint import backward as backward_sparse_contact


@dataclass(frozen=True)
class StepCotangent:
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


@dataclass(frozen=True)
class StepGradient:
  qpos: wp.array
  qvel: wp.array
  ctrl: wp.array
  qfrc_applied: wp.array
  time: wp.array
  contact_softness: wp.array


@dataclass(frozen=True)
class AdjointBuffers:
  qacc: wp.array
  constraint_force: wp.array
  total_rhs: wp.array
  total_matrix: wp.array
  mass: wp.array
  smooth_force: wp.array


@dataclass(frozen=True)
class MassGradient:
  matrix_gradient: wp.array
  rhs_gradient: wp.array


@dataclass(frozen=True)
class FastStepAdjointWorkspace:
  forward: FastStepWorkspace
  contact: object
  buffers: AdjointBuffers


@dataclass(frozen=True)
class RecordedStep:
  pose_tape: wp.Tape
  contact_tape: wp.Tape
  coefficient_tape: wp.Tape
  integration_tape: wp.Tape
  result: StepResult


@dataclass(frozen=True)
class BackwardCall:
  compiled: CompiledFastStep
  inputs: StepInput
  workspace: FastStepAdjointWorkspace
  recorded: RecordedStep
  cotangent: StepCotangent


@wp.kernel(enable_backward=False)
def _sum_vector(left: wp.array2d(dtype=float),
                right: wp.array2d(dtype=float),
                output: wp.array2d(dtype=float)):
  world, index = wp.tid()
  output[world, index] = left[world, index] + right[world, index]


@wp.kernel(enable_backward=False)
def _sum_matrix(left: wp.array3d(dtype=float),
                right: wp.array3d(dtype=float),
                output: wp.array3d(dtype=float)):
  world, row, column = wp.tid()
  output[world, row, column] = (
      left[world, row, column] + right[world, row, column])


@wp.kernel(enable_backward=False)
def _add_matrix(source: wp.array3d(dtype=float),
                target: wp.array3d(dtype=float)):
  world, row, column = wp.tid()
  target[world, row, column] += source[world, row, column]


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


@wp.kernel(enable_backward=False)
def _copy_scalar(source: wp.array(dtype=float),
                 target: wp.array(dtype=float)):
  index = wp.tid()
  target[index] = source[index]


def allocate_workspace(compiled: CompiledFastStep,
                       worlds: int) -> FastStepAdjointWorkspace:
  forward = allocate_forward(compiled, worlds, requires_grad=True)
  rows = compiled.base.contact_rows.row_capacity
  dofs = compiled.base.dynamics.dof_count
  device = compiled.base.device
  contact = _allocate_contact_adjoint(
      compiled, worlds, rows=rows, dofs=dofs, device=device)
  buffers = AdjointBuffers(
      wp.empty((worlds, dofs), dtype=float, device=device),
      wp.empty((worlds, dofs), dtype=float, device=device),
      wp.empty((worlds, dofs), dtype=float, device=device),
      wp.empty((worlds, dofs, dofs), dtype=float, device=device),
      wp.empty((worlds, dofs, dofs), dtype=float, device=device),
      wp.empty((worlds, dofs), dtype=float, device=device),
  )
  return FastStepAdjointWorkspace(forward, contact, buffers)


def _allocate_contact_adjoint(compiled: CompiledFastStep, worlds: int, *,
                              rows: int, dofs: int, device):
  if compiled.sparse_contact:
    return allocate_sparse_contact_adjoint(compiled.contact, worlds)
  return allocate_contact_adjoint(
      worlds, rows, dofs, iterations=compiled.contact.iterations,
      device=device)


def _pose_downstream(compiled: CompiledFastStep, inputs: StepInput,
                     workspace: FastStepAdjointWorkspace):
  base = compiled.base
  state = workspace.forward.base
  dynamics(base.dynamics, inputs.dynamics, state.pose, output=state.dynamics)
  return collision_frames(base, state)


def _pose_forward(compiled: CompiledFastStep, inputs: StepInput,
                  workspace: FastStepAdjointWorkspace):
  state = workspace.forward.base
  kinematics(
      compiled.base.kinematics, inputs.dynamics.qpos, state.pose)
  return _pose_downstream(compiled, inputs, workspace)


def _contact_prepare(compiled: CompiledFastStep, inputs: StepInput,
                     workspace: FastStepAdjointWorkspace, *,
                     qacc_smooth, freeze_frame_vjp: bool = False) -> None:
  base = compiled.base
  state = workspace.forward.base
  rows = contact_rows(
      base,
      state,
      inputs.dynamics.qpos,
      freeze_frame_vjp=freeze_frame_vjp,
  )
  prepare_contact_system(
      compiled, rows, inputs.dynamics.qvel, qacc_smooth=qacc_smooth,
      workspace=state.contact_solver)


def _integration_forward(compiled: CompiledFastStep, inputs: StepInput,
                         workspace: FastStepAdjointWorkspace, *, qacc) -> None:
  integration = IntegrationInput()
  integration.qpos = inputs.dynamics.qpos
  integration.qvel = inputs.dynamics.qvel
  integration.qacc = qacc
  integration.time = inputs.time
  integrate(compiled.base.integration, integration,
            workspace.forward.base.integrated)


def _result(workspace: FastStepAdjointWorkspace, qacc,
            constraint) -> StepResult:
  state = workspace.forward.base
  integrated = state.integrated
  contacts = state.collision.contacts
  return StepResult(
      integrated.qpos, integrated.qvel, qacc, integrated.time, constraint,
      state.pose.body_position, state.pose.body_matrix,
      contacts.distance, contacts.position, contacts.frame)


def record(compiled: CompiledFastStep, inputs: StepInput,
           workspace: FastStepAdjointWorkspace, *,
           freeze_frame_vjp: bool = False) -> RecordedStep:
  state = workspace.forward.base
  kinematics(
      compiled.base.kinematics, inputs.dynamics.qpos, state.pose)
  pose_tape = DynamicsTape()
  with pose_tape:
    frames = _pose_downstream(compiled, inputs, workspace)
  broadphase(compiled.base.collision, frames,
             workspace.forward.base.collision)
  qacc_smooth = solve_mass(
      workspace.forward.base.dynamics.mass,
      workspace.forward.base.dynamics.smooth_force,
      workspace.forward.smooth_mass)
  # Contact-row construction and row reduction both have gathered VJPs, so
  # only the collision narrow phase stays on this tape.
  state = workspace.forward.base
  contact_tape = GeometryTape()
  with contact_tape:
    contacts = collision_contacts(
        compiled.base, state, freeze_frame_vjp=freeze_frame_vjp)
  rows = build_contact_rows(
      compiled.base, state, contacts, contact_motion(state),
      inputs.dynamics.qpos)
  # The row reduction contracts every row onto the same degrees of freedom.
  # Warp would reverse it with atomic scatter, so it is kept off the tape and
  # reversed by `reduce_contact_rows_vjp` in a fixed order instead.
  reduce_contact_rows(
      compiled, rows, inputs.dynamics.qvel, qacc_smooth=qacc_smooth,
      workspace=state.contact_solver)
  coefficient_tape = wp.Tape()
  with coefficient_tape:
    contact_coefficients(compiled, state.contact_solver)
  contact = contact_input(
      inputs, workspace.forward, qacc_smooth,
      sparse=compiled.sparse_contact)
  constraint = solve_contact_system(
      compiled, contact, workspace.forward.contact)
  qacc = solve_mass(
      workspace.forward.base.dynamics.mass,
      workspace.forward.contact.output.total_force,
      workspace.forward.total_mass)
  qacc_integration = integration_acceleration(
      compiled, workspace.forward,
      mass=workspace.forward.base.dynamics.mass,
      rhs=workspace.forward.contact.output.total_force, qacc=qacc)
  integration_tape = wp.Tape()
  with integration_tape:
    _integration_forward(
        compiled, inputs, workspace, qacc=qacc_integration)
  result = _result(workspace, qacc, constraint)
  return RecordedStep(pose_tape, contact_tape, coefficient_tape,
                      integration_tape, result)


def forward(compiled: CompiledFastStep, inputs: StepInput,
            workspace: FastStepAdjointWorkspace) -> StepResult:
  frames = _pose_forward(compiled, inputs, workspace)
  broadphase(compiled.base.collision, frames,
             workspace.forward.base.collision)
  qacc_smooth = solve_mass(
      workspace.forward.base.dynamics.mass,
      workspace.forward.base.dynamics.smooth_force,
      workspace.forward.smooth_mass)
  _contact_prepare(
      compiled, inputs, workspace, qacc_smooth=qacc_smooth)
  contact = contact_input(
      inputs, workspace.forward, qacc_smooth,
      sparse=compiled.sparse_contact)
  constraint = solve_contact_system(
      compiled, contact, workspace.forward.contact)
  qacc = solve_mass(
      workspace.forward.base.dynamics.mass,
      workspace.forward.contact.output.total_force,
      workspace.forward.total_mass)
  qacc_integration = integration_acceleration(
      compiled, workspace.forward,
      mass=workspace.forward.base.dynamics.mass,
      rhs=workspace.forward.contact.output.total_force, qacc=qacc)
  _integration_forward(
      compiled, inputs, workspace, qacc=qacc_integration)
  return _result(workspace, qacc, constraint)


def _zero_input_gradients(inputs: StepInput) -> None:
  differentiable_inputs = (
      inputs.dynamics.qpos,
      inputs.dynamics.qvel,
      inputs.dynamics.ctrl,
      inputs.dynamics.qfrc_applied,
      inputs.time,
      inputs.contact_softness,
  )
  for value in differentiable_inputs:
    if value.grad is not None:
      value.grad.zero_()


def _zero_result_gradients(result: StepResult) -> None:
  differentiable_results = (
      result.qpos,
      result.qvel,
      result.qacc,
      result.time,
      result.constraint_force,
      result.body_position,
      result.body_matrix,
      result.contact_distance,
      result.contact_position,
      result.contact_frame,
  )
  for value in differentiable_results:
    if value.grad is not None:
      value.grad.zero_()


def _zero(compiled: CompiledFastStep, recorded: RecordedStep, inputs: StepInput,
          workspace: FastStepAdjointWorkspace) -> None:
  recorded.pose_tape.zero()
  recorded.contact_tape.zero()
  recorded.coefficient_tape.zero()
  recorded.integration_tape.zero()
  # Row construction and reduction are off-tape, so no tape clears their row
  # accumulators or the motion gradients they write into.
  if compiled.sparse_contact:
    workspace.forward.base.contact_rows.rows.jacobian.grad.zero_()
  workspace.forward.smooth_mass.forward.solution.grad.zero_()
  pose = workspace.forward.base.pose
  pose.joint_anchor.grad.zero_()
  pose.joint_axis.grad.zero_()
  integration_mass = workspace.forward.integration_mass
  if integration_mass is not None:
    integration_mass.forward.solution.grad.zero_()
  _zero_input_gradients(inputs)
  _zero_result_gradients(recorded.result)


def _backward_integration(compiled: CompiledFastStep,
                          recorded: RecordedStep,
                          cotangent: StepCotangent,
                          *,
                          workspace: FastStepAdjointWorkspace) -> None:
  recorded.integration_tape.backward(grads={
      recorded.result.qpos: cotangent.qpos,
      recorded.result.qvel: cotangent.qvel,
      recorded.result.time: cotangent.time,
  })
  if compiled.base.integration.implicit_damping:
    return
  qacc_gradient = workspace.forward.total_mass.forward.solution.grad
  wp.launch(_sum_vector, dim=qacc_gradient.shape,
            inputs=[qacc_gradient, cotangent.qacc],
            outputs=[workspace.buffers.qacc])


def _backward_total_mass(compiled: CompiledFastStep,
                         cotangent: StepCotangent,
                         workspace: FastStepAdjointWorkspace):
  if not compiled.base.integration.implicit_damping:
    return backward_mass(
        workspace.buffers.qacc, workspace.forward.total_mass)
  integration = backward_mass(
      workspace.forward.integration_mass.forward.solution.grad,
      workspace.forward.integration_mass)
  total = backward_mass(cotangent.qacc, workspace.forward.total_mass)
  wp.launch(
      _sum_vector, dim=total.rhs_gradient.shape,
      inputs=[integration.rhs_gradient, total.rhs_gradient],
      outputs=[workspace.buffers.total_rhs])
  wp.launch(
      _sum_matrix, dim=total.matrix_gradient.shape,
      inputs=[integration.matrix_gradient, total.matrix_gradient],
      outputs=[workspace.buffers.total_matrix])
  return MassGradient(
      workspace.buffers.total_matrix, workspace.buffers.total_rhs)


def _backward_contact(compiled: CompiledFastStep, inputs: StepInput,
                      cotangent: StepCotangent,
                      *, workspace: FastStepAdjointWorkspace):
  total = _backward_total_mass(compiled, cotangent, workspace)
  wp.launch(_sum_vector, dim=total.rhs_gradient.shape,
            inputs=[total.rhs_gradient, cotangent.constraint_force],
            outputs=[workspace.buffers.constraint_force])
  qacc_smooth = workspace.forward.smooth_mass.forward.solution
  forward_input = contact_input(
      inputs, workspace.forward, qacc_smooth,
      sparse=compiled.sparse_contact)
  contact = _backward_contact_system(
      compiled, forward_input, workspace.buffers.constraint_force,
      forward=workspace.forward.contact, adjoint=workspace.contact)
  return total, contact


def _backward_contact_system(compiled: CompiledFastStep, forward_input,
                             constraint_gradient, *, forward, adjoint):
  if compiled.sparse_contact:
    backward_input = SparseContactBackwardInput(
        forward_input, forward, constraint_gradient)
    return backward_sparse_contact(
        compiled.contact, backward_input, adjoint)
  backward_input = ContactBackwardInput(
      forward_input, forward, constraint_gradient)
  return backward_contact(compiled.contact, backward_input, adjoint)


def _jacobian_seed_target(compiled: CompiledFastStep,
                          workspace: FastStepAdjointWorkspace):
  """Array whose cotangent the contact solve reports as its Jacobian gradient.

  The sparse solve consumes its own CSR value array, so its gradient belongs
  there.  The dense solve reads the constraint rows directly.
  """
  state = workspace.forward.base
  if compiled.sparse_contact:
    return state.contact_solver.state.weighted_jacobian
  return state.contact_rows.rows.jacobian


def _backward_prepare(compiled: CompiledFastStep, recorded: RecordedStep,
                      inputs: StepInput, total, contact, *,
                      cotangent: StepCotangent,
                      workspace: FastStepAdjointWorkspace):
  solver = workspace.forward.base.contact_solver.state
  contacts = workspace.forward.base.collision.contacts
  recorded.coefficient_tape.backward(grads={
      solver.free_force: contact.free_force,
      solver.velocity_weight: contact.velocity_weight,
  })
  wp.copy(_jacobian_seed_target(compiled, workspace).grad, contact.jacobian)
  reduce_contact_rows_vjp(compiled, workspace.forward.base.contact_solver)
  wp.copy(contacts.distance.grad, cotangent.contact_distance)
  wp.copy(contacts.position.grad, cotangent.contact_position)
  wp.copy(contacts.frame.grad, cotangent.contact_frame)
  state = workspace.forward.base
  if compiled.sparse_contact:
    sparse_contact_rows_vjp.backward(
        compiled.base.contact_rows_vjp,
        compiled.base.contact_rows.parameters,
        contacts, contact_motion(state), state.contact_rows.rows,
        inputs.dynamics.qpos)
  else:
    contact_rows_vjp.backward(
        compiled.base.contact_rows_vjp,
        compiled.base.contact_rows.parameters,
        contacts, contact_motion(state), state.contact_rows.rows,
        inputs.dynamics.qpos)
  recorded.contact_tape.backward()
  smooth = backward_mass(
      workspace.forward.smooth_mass.forward.solution.grad,
      workspace.forward.smooth_mass)
  wp.launch(_sum_matrix, dim=total.matrix_gradient.shape,
            inputs=[total.matrix_gradient, smooth.matrix_gradient],
            outputs=[workspace.buffers.mass])
  wp.launch(_add_matrix, dim=contact.mass.shape,
            inputs=[contact.mass], outputs=[workspace.buffers.mass])
  wp.launch(_sum_vector, dim=total.rhs_gradient.shape,
            inputs=[total.rhs_gradient, smooth.rhs_gradient],
            outputs=[workspace.buffers.smooth_force])
  return smooth


def _copy_softness_gradient(inputs: StepInput, contact) -> None:
  wp.launch(
      _copy_scalar, dim=contact.contact_softness.shape,
      inputs=[contact.contact_softness], outputs=[inputs.contact_softness.grad],
      device=inputs.contact_softness.device)


def _add_pose_cotangent(cotangent: StepCotangent,
                        workspace: FastStepAdjointWorkspace) -> None:
  pose = workspace.forward.base.pose
  wp.launch(_add_vec3, dim=pose.body_position.shape,
            inputs=[cotangent.body_position], outputs=[pose.body_position.grad])
  wp.launch(_add_mat33, dim=pose.body_matrix.shape,
            inputs=[cotangent.body_matrix], outputs=[pose.body_matrix.grad])


def _pose_seeds(workspace: FastStepAdjointWorkspace) -> dict:
  state = workspace.forward.base
  return {
      state.dynamics.mass: workspace.buffers.mass,
      state.dynamics.smooth_force: workspace.buffers.smooth_force,
      state.pose.body_position: state.pose.body_position.grad,
      state.pose.body_matrix: state.pose.body_matrix.grad,
      state.pose.joint_anchor: state.pose.joint_anchor.grad,
      state.pose.joint_axis: state.pose.joint_axis.grad,
      state.geom_frames.position: state.geom_frames.position.grad,
      state.geom_frames.matrix: state.geom_frames.matrix.grad,
  }


def backward(call: BackwardCall) -> StepGradient:
  compiled = call.compiled
  inputs = call.inputs
  workspace = call.workspace
  recorded = call.recorded
  cotangent = call.cotangent
  _zero(compiled, recorded, inputs, workspace)
  _backward_integration(
      compiled, recorded, cotangent, workspace=workspace)
  total, contact = _backward_contact(
      compiled, inputs, cotangent, workspace=workspace)
  _backward_prepare(
      compiled, recorded, inputs, total, contact, cotangent=cotangent,
      workspace=workspace)
  _copy_softness_gradient(inputs, contact)
  _add_pose_cotangent(cotangent, workspace)
  recorded.pose_tape.backward(grads=_pose_seeds(workspace))
  add_kinematics_vjp(
      compiled.base.kinematics_vjp,
      workspace.forward.base.pose,
      inputs.dynamics.qpos,
      output=inputs.dynamics.qpos.grad,
  )
  return StepGradient(
      inputs.dynamics.qpos.grad, inputs.dynamics.qvel.grad,
      inputs.dynamics.ctrl.grad, inputs.dynamics.qfrc_applied.grad,
      inputs.time.grad,
      inputs.contact_softness.grad)
