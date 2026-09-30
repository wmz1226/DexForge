"""Fast native-Warp forward step built from production tile kernels."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from comfree_warp.comfree_core._src import support as production_support

from .explicit_damp import ExplicitDampInputs
from .explicit_damp import coefficients as coefficients_dense
from .explicit_damp import prepare as prepare_dense
from .explicit_damp import reduce_rows as reduce_dense
from .explicit_damp import reduce_rows_vjp as reduce_dense_vjp
from .gaussian_collision import broadphase

from .dynamics import dynamics
from .annealed_contact import allocate as allocate_contact
from .annealed_contact import solve as solve_contact
from .contact_types import ContactInput
from .contact_types import ContactParameters
from .fast_mass import allocate as allocate_mass
from .fast_mass import solve as solve_mass
from .integration import IntegrationInput
from .integration import build_euler_mass
from .integration import integrate
from .kinematics import kinematics
from .runtime import CompiledStep
from .runtime import StepInput
from .runtime import StepResult
from .runtime import allocate_workspace as allocate_base
from .runtime import collision_frames
from .runtime import compile_step
from .runtime import contact_rows
from .sparse_annealed_contact import SparseContactInput
from .sparse_annealed_contact import allocate as allocate_sparse_contact
from .sparse_annealed_contact import check_status as check_sparse_status
from .sparse_annealed_contact import clear_status as clear_sparse_status
from .sparse_annealed_contact import compile_sparse_contact
from .sparse_annealed_contact import solve as solve_sparse_contact
from .sparse_explicit_damp import SparseExplicitDampInputs
from .sparse_explicit_damp import coefficients as coefficients_sparse
from .sparse_explicit_damp import prepare as prepare_sparse
from .sparse_explicit_damp import reduce_rows as reduce_sparse
from .sparse_explicit_damp import reduce_rows_vjp as reduce_sparse_vjp


# The differentiable route intentionally omits the production-only dual
# residual correction.  Keeping this compile-time constant explicit prevents
# an environment variable from silently changing the differentiated program.
NATIVE_DUAL_CORRECTION = False


@dataclass(frozen=True)
class CompiledFastStep:
  base: CompiledStep
  contact: object
  sparse_contact: bool
  dual_correction: bool


@dataclass(frozen=True)
class FastStepWorkspace:
  base: object
  smooth_mass: object
  total_mass: object
  integration_mass: object
  integration_matrix: object
  contact: object


def compile_fast_step(cpu_model: mujoco.MjModel,
                      device_model) -> CompiledFastStep:
  base = compile_step(cpu_model, device_model)
  source = base.contact_solver.parameters
  if base.sparse_contact:
    contact = compile_sparse_contact(
        source, base.device, iterations=production_support.L)
    return CompiledFastStep(
        base, contact, True, NATIVE_DUAL_CORRECTION)
  contact = ContactParameters()
  contact.timestep = source.timestep
  contact.row_count = source.row_count
  contact.dof_count = source.dof_count
  contact.iterations = production_support.L
  return CompiledFastStep(
      base, contact, False, NATIVE_DUAL_CORRECTION)


def allocate_workspace(compiled: CompiledFastStep,
                       worlds: int, *, requires_grad: bool = False
                       ) -> FastStepWorkspace:
  base = compiled.base
  dofs = base.dynamics.dof_count
  rows = base.contact_rows.row_capacity
  contact = _allocate_contact_workspace(
      compiled, worlds, rows=rows, dofs=dofs)
  integration_mass, integration_matrix = _allocate_integration_workspace(
      compiled, worlds, dofs=dofs, requires_grad=requires_grad)
  return FastStepWorkspace(
      allocate_base(base, worlds, requires_grad=requires_grad),
      allocate_mass(worlds, dofs, base.device,
                    requires_grad=requires_grad),
      allocate_mass(worlds, dofs, base.device,
                    requires_grad=requires_grad),
      integration_mass,
      integration_matrix,
      contact,
  )


def _allocate_integration_workspace(
    compiled: CompiledFastStep, worlds: int, *, dofs: int,
    requires_grad: bool):
  if not compiled.base.integration.implicit_damping:
    return None, None
  mass = allocate_mass(
      worlds, dofs, compiled.base.device, requires_grad=requires_grad)
  matrix = wp.empty(
      (worlds, dofs, dofs), dtype=float, device=compiled.base.device)
  return mass, matrix


def _allocate_contact_workspace(compiled: CompiledFastStep, worlds: int,
                                *, rows: int, dofs: int):
  if compiled.sparse_contact:
    return allocate_sparse_contact(compiled.contact, worlds)
  return allocate_contact(
      worlds, rows, dofs, iterations=compiled.contact.iterations,
      device=compiled.base.device)


def contact_input(inputs: StepInput, workspace: FastStepWorkspace,
                  qacc_smooth, *, sparse: bool = False):
  base = workspace.base
  solver = base.contact_solver.state
  if sparse:
    result = SparseContactInput()
    result.mass = base.dynamics.mass
    result.active = base.contact_rows.rows.active
    result.weighted_jacobian = solver.weighted_jacobian
    result.free_force = solver.free_force
    result.velocity_weight = solver.velocity_weight
    result.qvel = inputs.dynamics.qvel
    result.qacc_smooth = qacc_smooth
    result.smooth_force = base.dynamics.smooth_force
    result.contact_softness = inputs.contact_softness
    return result
  result = ContactInput()
  result.mass = base.dynamics.mass
  result.jacobian = base.contact_rows.rows.jacobian
  result.position = base.contact_rows.rows.position
  result.active = base.contact_rows.rows.active
  result.free_force = solver.free_force
  result.velocity_weight = solver.velocity_weight
  result.qvel = inputs.dynamics.qvel
  result.qacc_smooth = qacc_smooth
  result.smooth_force = base.dynamics.smooth_force
  result.contact_softness = inputs.contact_softness
  return result


def _prepare(compiled: CompiledFastStep, inputs: StepInput,
             workspace: FastStepWorkspace, *, qacc_smooth) -> None:
  base = compiled.base
  frames = collision_frames(base, workspace.base)
  broadphase(base.collision, frames, workspace.base.collision)
  rows = contact_rows(base, workspace.base, inputs.dynamics.qpos)
  prepare_contact_system(
      compiled, rows, inputs.dynamics.qvel, qacc_smooth=qacc_smooth,
      workspace=workspace.base.contact_solver)


def _solver_inputs(compiled: CompiledFastStep, rows, qvel, qacc_smooth):
  if compiled.sparse_contact:
    return SparseExplicitDampInputs(rows, qvel, qacc_smooth)
  return ExplicitDampInputs(rows, qvel, qacc_smooth)


def prepare_contact_system(compiled: CompiledFastStep, rows, qvel, *,
                           qacc_smooth, workspace) -> None:
  inputs = _solver_inputs(compiled, rows, qvel, qacc_smooth)
  prepare = prepare_sparse if compiled.sparse_contact else prepare_dense
  prepare(compiled.base.contact_solver, inputs, workspace)


def reduce_contact_rows(compiled: CompiledFastStep, rows, qvel, *,
                        qacc_smooth, workspace) -> None:
  """First half of :func:`prepare_contact_system`.

  The differentiable step runs the two halves separately so the contended
  reduction can stay off the tape and use its own deterministic reverse pass.
  """
  inputs = _solver_inputs(compiled, rows, qvel, qacc_smooth)
  reduce = reduce_sparse if compiled.sparse_contact else reduce_dense
  reduce(compiled.base.contact_solver, inputs, workspace)


def reduce_contact_rows_vjp(compiled: CompiledFastStep, workspace) -> None:
  reduce = reduce_sparse_vjp if compiled.sparse_contact else reduce_dense_vjp
  reduce(compiled.base.contact_solver, workspace)


def contact_coefficients(compiled: CompiledFastStep, workspace) -> None:
  """Second half of :func:`prepare_contact_system`."""
  coefficients = (coefficients_sparse if compiled.sparse_contact
                  else coefficients_dense)
  coefficients(compiled.base.contact_solver, workspace)


def solve_contact_system(compiled: CompiledFastStep, contact, workspace):
  if compiled.sparse_contact:
    return solve_sparse_contact(compiled.contact, contact, workspace)
  return solve_contact(compiled.contact, contact, workspace)


def clear_contact_status(compiled: CompiledFastStep,
                         workspace: FastStepWorkspace) -> None:
  if compiled.sparse_contact:
    clear_sparse_status(workspace.contact)


def check_contact_status(compiled: CompiledFastStep,
                         workspace: FastStepWorkspace) -> None:
  if compiled.sparse_contact:
    check_sparse_status(workspace.contact)


def _integrate(compiled: CompiledFastStep, inputs: StepInput,
               workspace: FastStepWorkspace, *, qacc) -> None:
  integration = IntegrationInput()
  integration.qpos = inputs.dynamics.qpos
  integration.qvel = inputs.dynamics.qvel
  integration.qacc = qacc
  integration.time = inputs.time
  integrate(compiled.base.integration, integration,
            workspace.base.integrated)


def integration_acceleration(
    compiled: CompiledFastStep, workspace: FastStepWorkspace, *,
    mass, rhs, qacc):
  integration = compiled.base.integration
  if not integration.implicit_damping:
    return qacc
  build_euler_mass(integration, mass, workspace.integration_matrix)
  return solve_mass(
      workspace.integration_matrix, rhs, workspace.integration_mass)


def step(compiled: CompiledFastStep, inputs: StepInput,
         workspace: FastStepWorkspace) -> StepResult:
  base = compiled.base
  state = workspace.base
  kinematics(base.kinematics, inputs.dynamics.qpos, state.pose)
  dynamics(base.dynamics, inputs.dynamics, state.pose, output=state.dynamics)
  qacc_smooth = solve_mass(
      state.dynamics.mass, state.dynamics.smooth_force,
      workspace.smooth_mass)
  _prepare(compiled, inputs, workspace, qacc_smooth=qacc_smooth)
  contact = contact_input(
      inputs, workspace, qacc_smooth, sparse=compiled.sparse_contact)
  constraint = solve_contact_system(compiled, contact, workspace.contact)
  qacc = solve_mass(
      state.dynamics.mass, workspace.contact.output.total_force,
      workspace.total_mass)
  qacc_integration = integration_acceleration(
      compiled, workspace, mass=state.dynamics.mass,
      rhs=workspace.contact.output.total_force, qacc=qacc)
  _integrate(compiled, inputs, workspace, qacc=qacc_integration)
  collision = state.collision.contacts
  return StepResult(
      state.integrated.qpos, state.integrated.qvel, qacc,
      state.integrated.time, constraint,
      state.pose.body_position, state.pose.body_matrix,
      collision.distance, collision.position, collision.frame)
