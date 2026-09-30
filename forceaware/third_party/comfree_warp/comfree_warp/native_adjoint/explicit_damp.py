"""Differentiable coefficient preparation for explicit-damping contact."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import warp as wp

from comfree_warp.mujoco_warp._src.types import DisableBit
from comfree_warp.mujoco_warp._src.types import vec5
from comfree_warp.comfree_core._src import support as production_support
from comfree_warp.comfree_core._src.types import L0_REF_GAIN
from .contact_rows import ConstraintRows


DOF_TILE_SIZE = 32
MIN_IMPEDANCE = float(mujoco.mjMINIMP)
MAX_IMPEDANCE = float(mujoco.mjMAXIMP)
MIN_VALUE = float(mujoco.mjMINVAL)


@wp.struct
class ExplicitDampParameters:
    timestep: wp.array(dtype=float)
    impratio_invsqrt: wp.array(dtype=float)
    body_invweight: wp.array2d(dtype=wp.vec2)
    geom_bodyid: wp.array(dtype=int)
    contact_geom: wp.array(dtype=wp.vec2i)
    contact_friction: wp.array(dtype=vec5)
    friction_scale: wp.array(dtype=float)
    contact_solref: wp.array(dtype=wp.vec2)
    contact_solimp: wp.array(dtype=vec5)
    row_contact: wp.array(dtype=int)
    limit_joint: wp.array(dtype=int)
    jnt_dofadr: wp.array(dtype=int)
    jnt_solref: wp.array2d(dtype=wp.vec2)
    jnt_solimp: wp.array2d(dtype=vec5)
    dof_invweight: wp.array2d(dtype=float)
    refsafe_disabled: bool
    l0_mode: bool
    l0_ref_gain: float
    contact_row_count: int
    row_count: int
    row_capacity: int
    dof_count: int


@wp.struct
class ReductionInputs:
    jacobian: wp.array3d(dtype=float)
    qvel: wp.array2d(dtype=float)
    qacc_smooth: wp.array2d(dtype=float)


@wp.struct
class CoefficientReductions:
    smooth_acceleration: wp.array2d(dtype=float)
    current_velocity: wp.array2d(dtype=float)


@wp.struct
class CoefficientInputs:
    position: wp.array2d(dtype=float)
    active: wp.array2d(dtype=int)
    reductions: CoefficientReductions


@wp.struct
class CoefficientOutputs:
    free_force: wp.array2d(dtype=float)
    velocity_weight: wp.array2d(dtype=float)


@wp.struct
class ExplicitDampState:
    rows: ConstraintRows
    qvel: wp.array2d(dtype=float)
    qacc_smooth: wp.array2d(dtype=float)
    reductions: CoefficientReductions
    free_force: wp.array2d(dtype=float)
    velocity_weight: wp.array2d(dtype=float)


@dataclass(frozen=True)
class ExplicitDampInputs:
    rows: ConstraintRows
    qvel: wp.array
    qacc_smooth: wp.array


@dataclass(frozen=True)
class CompiledExplicitDamp:
    parameters: ExplicitDampParameters
    reduction_kernel: object
    coefficient_kernel: object
    device: object


@dataclass(frozen=True)
class ExplicitDampWorkspace:
    state: ExplicitDampState


@wp.func
def _impedance(solimp: vec5, position: float) -> float:
    minimum = wp.clamp(solimp[0], MIN_IMPEDANCE, MAX_IMPEDANCE)
    maximum = wp.clamp(solimp[1], MIN_IMPEDANCE, MAX_IMPEDANCE)
    middle = wp.clamp(solimp[3], MIN_IMPEDANCE, MAX_IMPEDANCE)
    power = wp.max(solimp[4], 1.0)
    width = wp.max(solimp[2], MIN_VALUE)
    x = wp.abs(position) / width
    lower = wp.pow(x, power) / wp.pow(middle, power - 1.0)
    upper = 1.0 - wp.pow(1.0 - x, power) / wp.pow(1.0 - middle, power - 1.0)
    curve = wp.where(x < middle, lower, upper)
    value = wp.clamp(minimum + curve * (maximum - minimum), minimum, maximum)
    return wp.where(x > 1.0, maximum, value)


def _make_reduction_kernel(dof_count: int):
    @wp.kernel(module="unique")
    def _reduce(params: ExplicitDampParameters, inputs: ReductionInputs,
                outputs: CoefficientReductions):
        world, row = wp.tid()
        jacobian = wp.tile_load(
            inputs.jacobian[world, row], shape=dof_count, bounds_check=True)
        qvel = wp.tile_load(
            inputs.qvel[world], shape=dof_count, bounds_check=True)
        qacc = wp.tile_load(
            inputs.qacc_smooth[world], shape=dof_count, bounds_check=True)
        smooth_acceleration = wp.tile_sum(jacobian * qacc)
        current_velocity = wp.tile_sum(jacobian * qvel)
        wp.tile_store(outputs.smooth_acceleration[world], smooth_acceleration,
                      offset=row, bounds_check=True)
        wp.tile_store(outputs.current_velocity[world], current_velocity,
                      offset=row, bounds_check=True)

    return _reduce


@wp.func
def _contact_invweight(params: ExplicitDampParameters,
                       world: int, contact: int) -> float:
    geom = params.contact_geom[contact]
    body1 = params.geom_bodyid[geom[0]]
    body2 = params.geom_bodyid[geom[1]]
    model = world % params.body_invweight.shape[0]
    base = (params.body_invweight[model, body1][0]
            + params.body_invweight[model, body2][0])
    friction = params.contact_friction[contact][0]
    friction *= params.friction_scale[world % params.friction_scale.shape[0]]
    ratio = params.impratio_invsqrt[
        world % params.impratio_invsqrt.shape[0]]
    return base * (1.0 + friction * friction) * (
        2.0 * friction * friction * ratio * ratio)


@wp.func
def _row_solref(params: ExplicitDampParameters,
                world: int, row: int) -> wp.vec2:
    if row < params.contact_row_count:
        return params.contact_solref[params.row_contact[row]]
    limit = row - params.contact_row_count
    joint = params.limit_joint[limit]
    model = world % params.jnt_solref.shape[0]
    return params.jnt_solref[model, joint]


@wp.func
def _row_solimp(params: ExplicitDampParameters,
                world: int, row: int) -> vec5:
    if row < params.contact_row_count:
        return params.contact_solimp[params.row_contact[row]]
    limit = row - params.contact_row_count
    joint = params.limit_joint[limit]
    model = world % params.jnt_solimp.shape[0]
    return params.jnt_solimp[model, joint]


@wp.func
def _row_invweight(params: ExplicitDampParameters,
                   world: int, row: int) -> float:
    if row < params.contact_row_count:
        return _contact_invweight(params, world, params.row_contact[row])
    limit = row - params.contact_row_count
    joint = params.limit_joint[limit]
    dof = params.jnt_dofadr[joint]
    model = world % params.dof_invweight.shape[0]
    return params.dof_invweight[model, dof]


@wp.func
def _reference_coefficients(params: ExplicitDampParameters,
                            world: int, row: int) -> wp.vec2:
    solref = _row_solref(params, world, row)
    solimp = _row_solimp(params, world, row)
    dmax = wp.clamp(solimp[1], MIN_IMPEDANCE, MAX_IMPEDANCE)
    timestep = params.timestep[world % params.timestep.shape[0]]
    timeconst = solref[0]
    if params.l0_mode:
        timeconst = timeconst / params.l0_ref_gain
    elif not params.refsafe_disabled:
        timeconst = wp.max(timeconst, 2.0 * timestep)
    k = 1.0 / (dmax * dmax * timeconst * timeconst
               * solref[1] * solref[1])
    b = 2.0 / (dmax * timeconst)
    k = wp.where(solref[0] <= 0.0, -solref[0] / (dmax * dmax), k)
    b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
    return wp.vec2(k, b)


@wp.kernel
def _contact_coefficients(params: ExplicitDampParameters,
                          inputs: CoefficientInputs,
                          outputs: CoefficientOutputs):
    world, row = wp.tid()
    if row >= params.row_count or inputs.active[world, row] == 0:
        outputs.free_force[world, row] = 0.0
        outputs.velocity_weight[world, row] = 0.0
        return
    solimp = _row_solimp(params, world, row)
    impedance = _impedance(solimp, inputs.position[world, row])
    coefficients = _reference_coefficients(params, world, row)
    aref = (-coefficients[0] * impedance * inputs.position[world, row]
            - coefficients[1] * inputs.reductions.current_velocity[world, row])
    invweight = _row_invweight(params, world, row)
    D = 1.0 / wp.max(
        invweight * (1.0 - impedance) / impedance, MIN_VALUE)
    z = inputs.reductions.smooth_acceleration[world, row] - aref
    outputs.free_force[world, row] = -D * z
    outputs.velocity_weight[world, row] = D


def compile_explicit_damp(device_model, collision,
                          row_layout) -> CompiledExplicitDamp:
    params = ExplicitDampParameters()
    params.timestep = device_model.opt.timestep
    params.impratio_invsqrt = device_model.opt.impratio_invsqrt
    params.body_invweight = device_model.body_invweight0
    params.geom_bodyid = device_model.geom_bodyid
    params.contact_geom = row_layout.parameters.contact_geom
    params.contact_friction = row_layout.parameters.contact_friction
    params.friction_scale = row_layout.parameters.friction_scale
    params.contact_solref = collision.contact_solref
    params.contact_solimp = collision.contact_solimp
    params.row_contact = row_layout.parameters.row_contact
    params.limit_joint = row_layout.parameters.limit_joint
    params.jnt_dofadr = device_model.jnt_dofadr
    params.jnt_solref = device_model.jnt_solref
    params.jnt_solimp = device_model.jnt_solimp
    params.dof_invweight = device_model.dof_invweight0
    params.refsafe_disabled = bool(
        device_model.opt.disableflags & DisableBit.REFSAFE)
    params.l0_mode = production_support.L == 0
    params.l0_ref_gain = L0_REF_GAIN
    params.contact_row_count = row_layout.contact_row_count
    params.row_count = row_layout.row_count
    params.row_capacity = row_layout.row_capacity
    params.dof_count = row_layout.parameters.dof_count
    reduction = _make_reduction_kernel(params.dof_count)
    return CompiledExplicitDamp(
        params, reduction, _contact_coefficients,
        device_model.qpos0.device)


def allocate_workspace(compiled: CompiledExplicitDamp, worlds: int,
                       *, gradient: bool) -> ExplicitDampWorkspace:
    params = compiled.parameters
    state = ExplicitDampState()
    row_shape = (worlds, params.row_capacity)
    reductions = CoefficientReductions()
    for name in ("smooth_acceleration", "current_velocity"):
        setattr(reductions, name, wp.empty(
            row_shape, dtype=float, device=compiled.device,
            requires_grad=gradient, retain_grad=gradient))
    state.reductions = reductions
    for name in ("free_force", "velocity_weight"):
        setattr(state, name, wp.empty(
            row_shape, dtype=float, device=compiled.device,
            requires_grad=gradient, retain_grad=gradient))
    return ExplicitDampWorkspace(state)


@wp.kernel(enable_backward=False)
def _reduce_jacobian_vjp(adj_acceleration: wp.array2d(dtype=float),
                         adj_velocity: wp.array2d(dtype=float),
                         qacc_smooth: wp.array2d(dtype=float),
                         qvel: wp.array2d(dtype=float),
                         jacobian_grad: wp.array3d(dtype=float)):
    """``d(reductions)/d(jacobian)``; one thread owns one Jacobian entry."""
    world, row, dof = wp.tid()
    jacobian_grad[world, row, dof] += (
        adj_acceleration[world, row] * qacc_smooth[world, dof]
        + adj_velocity[world, row] * qvel[world, dof])


@wp.kernel(enable_backward=False)
def _reduce_velocity_vjp(jacobian: wp.array3d(dtype=float),
                         adj_acceleration: wp.array2d(dtype=float),
                         adj_velocity: wp.array2d(dtype=float),
                         row_capacity: int,
                         qvel_grad: wp.array2d(dtype=float),
                         qacc_grad: wp.array2d(dtype=float)):
    """``d(reductions)/d(qvel, qacc_smooth)`` gathered over rows.

    Warp's generated adjoint scatters this with ``atomic_add`` from every row
    onto the same degree of freedom, so its summation order and therefore the
    gradient are not reproducible.  One thread per degree of freedom
    accumulates in a fixed order instead.
    """
    world, dof = wp.tid()
    velocity = float(0.0)
    acceleration = float(0.0)
    for row in range(row_capacity):
        value = jacobian[world, row, dof]
        velocity += value * adj_velocity[world, row]
        acceleration += value * adj_acceleration[world, row]
    qvel_grad[world, dof] += velocity
    qacc_grad[world, dof] += acceleration


def reduce_rows(compiled: CompiledExplicitDamp, inputs: ExplicitDampInputs,
                workspace: ExplicitDampWorkspace) -> None:
    """Contract the constraint Jacobian against velocity and acceleration.

    Kept off the differentiation tape: its reverse pass is supplied by
    ``reduce_rows_vjp`` so the accumulation order is fixed.
    """
    state = workspace.state
    state.rows = inputs.rows
    state.qvel = inputs.qvel
    state.qacc_smooth = inputs.qacc_smooth
    reduction_inputs = ReductionInputs()
    reduction_inputs.jacobian = state.rows.jacobian
    reduction_inputs.qvel = state.qvel
    reduction_inputs.qacc_smooth = state.qacc_smooth
    wp.launch_tiled(
        compiled.reduction_kernel,
        dim=(inputs.qvel.shape[0], compiled.parameters.row_capacity),
        inputs=[compiled.parameters, reduction_inputs],
        outputs=[state.reductions], block_dim=DOF_TILE_SIZE,
        device=compiled.device)


def reduce_rows_vjp(compiled: CompiledExplicitDamp,
                    workspace: ExplicitDampWorkspace) -> None:
    """Deterministic reverse pass of :func:`reduce_rows`, accumulating."""
    state = workspace.state
    params = compiled.parameters
    worlds = state.qvel.shape[0]
    reductions = state.reductions
    wp.launch(
        _reduce_jacobian_vjp,
        dim=(worlds, params.row_capacity, params.dof_count),
        inputs=[reductions.smooth_acceleration.grad,
                reductions.current_velocity.grad,
                state.qacc_smooth, state.qvel],
        outputs=[state.rows.jacobian.grad], device=compiled.device)
    wp.launch(
        _reduce_velocity_vjp, dim=(worlds, params.dof_count),
        inputs=[state.rows.jacobian,
                reductions.smooth_acceleration.grad,
                reductions.current_velocity.grad,
                params.row_capacity],
        outputs=[state.qvel.grad, state.qacc_smooth.grad],
        device=compiled.device)


def coefficients(compiled: CompiledExplicitDamp,
                 workspace: ExplicitDampWorkspace) -> None:
    """Per-row constraint coefficients; every thread owns one row."""
    state = workspace.state
    coefficient_inputs = CoefficientInputs()
    coefficient_inputs.position = state.rows.position
    coefficient_inputs.active = state.rows.active
    coefficient_inputs.reductions = state.reductions
    coefficient_outputs = CoefficientOutputs()
    coefficient_outputs.free_force = state.free_force
    coefficient_outputs.velocity_weight = state.velocity_weight
    wp.launch(compiled.coefficient_kernel,
              dim=(state.qvel.shape[0], compiled.parameters.row_capacity),
              inputs=[compiled.parameters, coefficient_inputs],
              outputs=[coefficient_outputs], device=compiled.device)


def prepare(compiled: CompiledExplicitDamp, inputs: ExplicitDampInputs,
            workspace: ExplicitDampWorkspace) -> None:
    reduce_rows(compiled, inputs, workspace)
    coefficients(compiled, workspace)
