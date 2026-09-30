# Modified for this distribution: documentation streamlined; numerical implementation unchanged.
# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
import warp as wp

from comfree_warp.mujoco_warp._src import collision_driver
from comfree_warp.mujoco_warp._src import sensor
from comfree_warp.mujoco_warp._src import smooth
from comfree_warp.mujoco_warp._src.forward import euler
from comfree_warp.mujoco_warp._src.forward import fwd_acceleration
from comfree_warp.mujoco_warp._src.forward import fwd_actuation
from comfree_warp.mujoco_warp._src.forward import fwd_velocity
from comfree_warp.mujoco_warp._src.forward import implicit
from comfree_warp.mujoco_warp._src.warp_util import cache_kernel
from comfree_warp.mujoco_warp._src.warp_util import event_scope

from . import constraint
from . import collision_gaussian_bvh
from . import sparse_contact
from . import support
from .types import Data
from .types import DisableBit
from .types import EnableBit
from .types import IntegratorType
from .types import Model

wp.set_module_options({"enable_backward": False})

FULL_IMPLICIT_TILE_SIZE = support.FULL_IMPLICIT_TILE_SIZE


@cache_kernel
def _advance_vel(clear_constraint: bool):
  @wp.kernel(module="unique")
  def _kernel(
    opt_timestep: wp.array(dtype=float),
    qvel_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    qvel_out: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
  ):
    worldid, dofid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_smooth_in[worldid, dofid] * timestep
    if wp.static(clear_constraint):
      qfrc_constraint[worldid, dofid] = 0.0

  return _kernel


def _prepare_dense_rows(m: Model, d: Data, rows) -> None:
  aref = d.efc.efc_mass if support.L == 0 else d.efc.aref
  wp.launch(
    _prepare_canonical_qp_rows(m.nv),
    dim=(d.nworld, rows.budget),
    inputs=[
      d.efc.J,
      d.efc.D,
      aref,
      d.qacc_smooth,
      d.nefc,
      rows.limit,
    ],
    outputs=[
      d.efc.contact_free_force,
      d.efc.contact_active_D,
      d.efc.force,
    ],
  )


def _prepare_contact_rows(m: Model, d: Data, rows) -> None:
  if not support.requires_contact_row_preparation(m, d):
    return
  if m.is_sparse:
    aref = d.efc.efc_mass if support.L == 0 else d.efc.aref
    sparse_contact.prepare_sparse_rows(m, d, rows, aref=aref)
    return
  _prepare_dense_rows(m, d, rows)


@cache_kernel
def _prepare_canonical_qp_rows(nv: int):
  @wp.kernel(module="unique")
  def _kernel(
    J: wp.array3d(dtype=float),
    D: wp.array2d(dtype=float),
    aref: wp.array2d(dtype=float),
    qacc_smooth: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int),
    row_limit: int,
    free_force: wp.array2d(dtype=float),
    active_D: wp.array2d(dtype=float),
    efc_force: wp.array2d(dtype=float),
  ):
    worldid, efcid = wp.tid()
    if efcid >= nefc[worldid] or efcid >= row_limit:
      return


    z = -aref[worldid, efcid]
    for dofid in range(wp.static(nv)):
      z += J[worldid, efcid, dofid] * qacc_smooth[worldid, dofid]

    free_force[worldid, efcid] = -D[worldid, efcid] * z
    active_D[worldid, efcid] = 0.0
    efc_force[worldid, efcid] = 0.0

  return _kernel


def _initialize_smooth_state(m: Model, d: Data) -> None:
  clear_constraint = not support.uses_dense_direct_solver(m, d)
  wp.launch(
    _advance_vel(clear_constraint),
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      d.qvel,
      d.qacc_smooth,
    ],
    outputs=[
      d.qvel_smooth_pred,
      d.qfrc_constraint,
    ],
  )


@event_scope
def compute_qfrc_total(m: Model, d: Data):
  _initialize_smooth_state(m, d)
  rows = support.constraint_rows(d)
  _prepare_contact_rows(m, d, rows)
  support.solve_fullimplicit_contacts(m, d, rows)


def resolve_constraints(m: Model, d: Data) -> None:
  """Resolves contact forces and writes the constrained acceleration."""
  compute_qfrc_total(m, d)
  if support.uses_dense_direct_solver(m, d):
    return
  smooth.solve_m(m, d, d.qacc, d.qfrc_total)


@event_scope
def forward_comfree(m: Model, d: Data, factorize: bool = True):
  """Forward dynamics with complementarity-free model."""

  # forward position
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
    collision_gaussian_bvh.collision(m, d)
  constraint.make_constraint(m, d)
  smooth.transmission(m, d)

  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  energy = m.opt.enableflags & EnableBit.ENERGY
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  # forward velocity
  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  # forward actuation and smooth acceleration
  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=False)

  # call comfree model to resolve constraints
  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
  else:
    resolve_constraints(m, d)

  sensor.sensor_acc(m, d)


@event_scope
def step_comfree(m: Model, d: Data):
  """Advance simulation with complementarity-free model."""
  forward_comfree(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    wp.copy(d.efc.Ma, d.qfrc_total)
    euler(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    wp.copy(d.efc.Ma, d.qfrc_total)
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
