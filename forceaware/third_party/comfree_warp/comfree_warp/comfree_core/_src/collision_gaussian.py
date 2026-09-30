"""Exact Warp implementation of JAX-GS hard sphere-cloud collision."""

from __future__ import annotations
from typing import Any

import numpy as np
import warp as wp

from comfree_warp.mujoco_warp._src.types import vec5
from comfree_warp.mujoco_warp._src.warp_util import event_scope


CONTACT_CONSTRAINT = wp.constant(1)
NORMAL_EPS = wp.constant(1.0e-9)
GS_BROADPHASE_EPS = wp.constant(1.0e-6)
PLANE_CONTACT_COUNT = 4
REFERENCE_PLANE_CONTACT_START = 0
PLANE_BLOCK_DIM = 256
INVALID_SUPPORT_ID_KEY = wp.constant(2147483647)

wp.set_module_options({"enable_backward": False})


@wp.struct
class GaussianCollisionModel:
  target_spheres: wp.array(dtype=wp.vec4)
  source_centers: wp.array(dtype=wp.vec3)
  source_radii: wp.array(dtype=float)
  source_body_ids: wp.array(dtype=int)
  output_starts: wp.array(dtype=int)
  output_counts: wp.array(dtype=int)
  target_output_ids: wp.array(dtype=int)
  bvh_center_lower: wp.array(dtype=wp.vec3)
  bvh_center_upper: wp.array(dtype=wp.vec3)
  bvh_max_radius: wp.array(dtype=float)
  bvh_ranges: wp.array(dtype=wp.vec2i)
  bvh_escape: wp.array(dtype=int)
  bvh_sphere_ids: wp.array(dtype=int)
  contact_geom: wp.array(dtype=wp.vec2i)
  contact_normal_sign: wp.array(dtype=float)
  contact_dim: wp.array(dtype=int)
  contact_includemargin: wp.array(dtype=float)
  contact_friction: wp.array(dtype=vec5)
  contact_solref: wp.array(dtype=wp.vec2)
  contact_solreffriction: wp.array(dtype=wp.vec2)
  contact_solimp: wp.array(dtype=vec5)
  distance_offset: wp.array(dtype=float)
  target_body_id: int
  plane_geom_id: int
  target_contact_offset: int
  contact_id_offset: int
  contact_count: int
  target_plane_enabled: int
  target_count: int
  threshold: float
  # Candidates fused per pair slot (1: hard minimum).
  contact_topk: int


@wp.struct
class GaussianCollisionState:
  xpos: wp.array2d(dtype=wp.vec3)
  xmat: wp.array2d(dtype=wp.mat33)
  geom_xpos: wp.array2d(dtype=wp.vec3)
  geom_xmat: wp.array2d(dtype=wp.mat33)
  nworld: int


@wp.struct
class GaussianContactOutput:
  dist: wp.array(dtype=float)
  pos: wp.array(dtype=wp.vec3)
  frame: wp.array(dtype=wp.mat33)
  includemargin: wp.array(dtype=float)
  friction: wp.array(dtype=vec5)
  solref: wp.array(dtype=wp.vec2)
  solreffriction: wp.array(dtype=wp.vec2)
  solimp: wp.array(dtype=vec5)
  dim: wp.array(dtype=int)
  geom: wp.array(dtype=wp.vec2i)
  efc_address: wp.array2d(dtype=int)
  worldid: wp.array(dtype=int)
  type: wp.array(dtype=int)
  geomcollisionid: wp.array(dtype=int)
  nacon: wp.array(dtype=int)
  capacity: int
  efc_width: int


@wp.struct
class _ContactCandidate:
  row: int
  world: int
  distance: float
  position: wp.vec3
  normal: wp.vec3


@wp.func
def _normalized(value: wp.vec3):
  return value / wp.sqrt(wp.dot(value, value) + NORMAL_EPS * NORMAL_EPS)


@wp.func
def _sphere_center(sphere: wp.vec4):
  return wp.vec3(sphere[0], sphere[1], sphere[2])


@wp.func
def _contact_frame(normal: wp.vec3):
  normal = _normalized(normal)
  tangent_x = _normalized(wp.cross(normal, wp.vec3(1.0, 1.0, 1.0)))
  tangent_y = _normalized(wp.cross(normal, tangent_x))
  return wp.mat33(
      normal[0], normal[1], normal[2],
      tangent_x[0], tangent_x[1], tangent_x[2],
      tangent_y[0], tangent_y[1], tangent_y[2])


@wp.func
def _append_contact(
    model: GaussianCollisionModel,
    candidate: _ContactCandidate,
    output: GaussianContactOutput,
):
  contact_id = wp.atomic_add(output.nacon, 0, 1)
  if contact_id >= output.capacity:
    return
  row = candidate.row
  output.dist[contact_id] = candidate.distance
  output.pos[contact_id] = candidate.position
  output.frame[contact_id] = _contact_frame(
      candidate.normal * model.contact_normal_sign[row])
  output.includemargin[contact_id] = model.contact_includemargin[row]
  output.friction[contact_id] = model.contact_friction[row]
  output.solref[contact_id] = model.contact_solref[row]
  output.solreffriction[contact_id] = model.contact_solreffriction[row]
  output.solimp[contact_id] = model.contact_solimp[row]
  output.dim[contact_id] = model.contact_dim[row]
  output.geom[contact_id] = model.contact_geom[row]
  output.worldid[contact_id] = candidate.world
  output.type[contact_id] = CONTACT_CONSTRAINT
  output.geomcollisionid[contact_id] = model.contact_id_offset + row
  for index in range(output.efc_width):
    output.efc_address[contact_id, index] = -1


@wp.struct
class _SupportMinimum:
  distance: float
  sphere_id: int


@wp.struct
class _PlaneReductionContext:
  normal: wp.vec3
  active_limit: float
  selected0: int
  selected1: int
  selected2: int
  selected_count: int


@wp.struct
class _PlaneSupportSet:
  s0: _SupportMinimum
  s1: _SupportMinimum
  s2: _SupportMinimum
  s3: _SupportMinimum


@wp.struct
class _TargetPlaneContact:
  target_pos: wp.vec3
  target_mat: wp.mat33
  plane_pos: wp.vec3
  normal: wp.vec3
  support: _SupportMinimum
  world: int
  row: int


@wp.func
def _minimum_support(left: _SupportMinimum, right: _SupportMinimum):
  improve = right.distance < left.distance
  tied = right.distance == left.distance and right.sphere_id < left.sphere_id
  return wp.where(improve or tied, right, left)


@wp.func
def _empty_support():
  support = _SupportMinimum()
  support.distance = float(1.0e30)
  support.sphere_id = -1
  return support


@wp.func
def _update_support(current: _SupportMinimum, distance: float, sphere_id: int):
  candidate = _SupportMinimum()
  candidate.distance = distance
  candidate.sphere_id = sphere_id
  return _minimum_support(current, candidate)


@wp.func
def _sphere_plane_distance(sphere: wp.vec4, normal: wp.vec3):
  return wp.dot(normal, _sphere_center(sphere)) - sphere[3]


@wp.func
def _sphere_plane_point(sphere: wp.vec4, normal: wp.vec3):
  return _sphere_center(sphere) - normal * sphere[3]


@wp.func
def _distance_offset(model: GaussianCollisionModel, world: int) -> float:
  return model.distance_offset[world % model.distance_offset.shape[0]]


@wp.func
def _inset_contact_position(
    position: Any, inward_normal: Any, distance_offset: Any,
):
  if distance_offset == 0.0:
    return position
  return position + inward_normal * distance_offset


@wp.func
def _plane_reduction_context(
    model: GaussianCollisionModel,
    contact: _TargetPlaneContact,
):
  context = _PlaneReductionContext()
  context.normal = wp.transpose(contact.target_mat) @ contact.normal
  base_distance = wp.dot(
      contact.target_pos - contact.plane_pos, contact.normal)
  base_distance += _distance_offset(model, contact.world)
  context.active_limit = (
      model.contact_includemargin[REFERENCE_PLANE_CONTACT_START] - base_distance)
  context.selected0 = -1
  context.selected1 = -1
  context.selected2 = -1
  context.selected_count = 0
  return context


@wp.func
def _scan_deepest_plane_support(
    model: GaussianCollisionModel,
    context: _PlaneReductionContext,
    lane: int,
):
  support = _empty_support()
  sphere_id = lane
  while sphere_id < model.target_count:
    sphere = model.target_spheres[sphere_id]
    distance = _sphere_plane_distance(sphere, context.normal)
    if distance < context.active_limit:
      support = _update_support(support, distance, sphere_id)
    sphere_id += wp.block_dim()
  return support


@wp.func
def _is_selected(context: _PlaneReductionContext, sphere_id: int):
  selected = context.selected_count > 0 and sphere_id == context.selected0
  selected = selected or (
      context.selected_count > 1 and sphere_id == context.selected1)
  selected = selected or (
      context.selected_count > 2 and sphere_id == context.selected2)
  return selected


@wp.func
def _tangent_distance_squared(
    first: wp.vec3,
    second: wp.vec3,
    normal: wp.vec3,
):
  delta = first - second
  normal_distance = wp.dot(delta, normal)
  return wp.max(wp.dot(delta, delta) - normal_distance * normal_distance, 0.0)


@wp.func
def _selected_separation_squared(
    model: GaussianCollisionModel,
    context: _PlaneReductionContext,
    point: wp.vec3,
):
  first = _sphere_plane_point(
      model.target_spheres[context.selected0], context.normal)
  separation = _tangent_distance_squared(point, first, context.normal)
  if context.selected_count > 1:
    second = _sphere_plane_point(
        model.target_spheres[context.selected1], context.normal)
    separation = wp.min(
        separation, _tangent_distance_squared(point, second, context.normal))
  if context.selected_count > 2:
    third = _sphere_plane_point(
        model.target_spheres[context.selected2], context.normal)
    separation = wp.min(
        separation, _tangent_distance_squared(point, third, context.normal))
  return separation


@wp.func
def _scan_farthest_plane_support(
    model: GaussianCollisionModel,
    context: _PlaneReductionContext,
    lane: int,
):
  support = _empty_support()
  sphere_id = lane
  while sphere_id < model.target_count:
    sphere = model.target_spheres[sphere_id]
    distance = _sphere_plane_distance(sphere, context.normal)
    eligible = distance < context.active_limit and not _is_selected(
        context, sphere_id)
    if eligible:
      point = _sphere_plane_point(sphere, context.normal)
      separation = _selected_separation_squared(model, context, point)
      support = _update_support(support, -separation, sphere_id)
    sphere_id += wp.block_dim()
  return support


@wp.func
def _support_id_key(support: _SupportMinimum):
  return wp.where(
      support.sphere_id >= 0, support.sphere_id, INVALID_SUPPORT_ID_KEY)


@wp.func
def _sort_plane_supports(supports: _PlaneSupportSet):
  if _support_id_key(supports.s1) < _support_id_key(supports.s0):
    temporary = supports.s0
    supports.s0 = supports.s1
    supports.s1 = temporary
  if _support_id_key(supports.s3) < _support_id_key(supports.s2):
    temporary = supports.s2
    supports.s2 = supports.s3
    supports.s3 = temporary
  if _support_id_key(supports.s2) < _support_id_key(supports.s0):
    temporary = supports.s0
    supports.s0 = supports.s2
    supports.s2 = temporary
  if _support_id_key(supports.s3) < _support_id_key(supports.s1):
    temporary = supports.s1
    supports.s1 = supports.s3
    supports.s3 = temporary
  if _support_id_key(supports.s2) < _support_id_key(supports.s1):
    temporary = supports.s1
    supports.s1 = supports.s2
    supports.s2 = temporary
  return supports


@wp.func
def _append_target_plane(
    model: GaussianCollisionModel,
    contact: _TargetPlaneContact,
    output: GaussianContactOutput,
):
  sphere_id = contact.support.sphere_id
  if sphere_id < 0:
    return
  sphere = model.target_spheres[sphere_id]
  center = contact.target_pos
  center += contact.target_mat @ _sphere_center(sphere)
  radius = sphere[3]
  candidate = _ContactCandidate()
  candidate.row = contact.row
  candidate.world = contact.world
  candidate.distance = wp.dot(
      center - contact.plane_pos, contact.normal) - radius
  distance_offset = _distance_offset(model, contact.world)
  candidate.distance += distance_offset
  candidate.position = _inset_contact_position(
      center - contact.normal * radius, contact.normal, distance_offset)
  candidate.normal = contact.normal
  if candidate.distance < model.contact_includemargin[contact.row]:
    _append_contact(model, candidate, output)


@wp.func
def _target_plane_context(
    model: GaussianCollisionModel,
    state: GaussianCollisionState,
    world: int,
):
  context = _TargetPlaneContact()
  context.world = world
  context.target_pos = state.xpos[world, model.target_body_id]
  context.target_mat = state.xmat[world, model.target_body_id]
  context.plane_pos = state.geom_xpos[world, model.plane_geom_id]
  plane_mat = state.geom_xmat[world, model.plane_geom_id]
  context.normal = wp.vec3(plane_mat[0, 2], plane_mat[1, 2], plane_mat[2, 2])
  return context


@wp.kernel
def _write_target_plane(
    model: GaussianCollisionModel,
    state: GaussianCollisionState,
    output: GaussianContactOutput,
):
  world, lane = wp.tid()
  context = _target_plane_context(model, state, world)
  selection = _plane_reduction_context(model, context)

  support0 = _scan_deepest_plane_support(model, selection, lane)
  reduced0 = wp.tile_reduce(
      _minimum_support, wp.tile(support0, preserve_type=True))
  selected0 = wp.tile_extract(reduced0, 0)
  selection.selected0 = selected0.sphere_id
  selection.selected_count = wp.where(selected0.sphere_id >= 0, 1, 0)

  support1 = _scan_farthest_plane_support(model, selection, lane)
  reduced1 = wp.tile_reduce(
      _minimum_support, wp.tile(support1, preserve_type=True))
  selected1 = wp.tile_extract(reduced1, 0)
  selection.selected1 = selected1.sphere_id
  selection.selected_count = selection.selected_count + wp.where(
      selected1.sphere_id >= 0, 1, 0)

  support2 = _scan_farthest_plane_support(model, selection, lane)
  reduced2 = wp.tile_reduce(
      _minimum_support, wp.tile(support2, preserve_type=True))
  selected2 = wp.tile_extract(reduced2, 0)
  selection.selected2 = selected2.sphere_id
  selection.selected_count = selection.selected_count + wp.where(
      selected2.sphere_id >= 0, 1, 0)

  support3 = _scan_farthest_plane_support(model, selection, lane)
  reduced3 = wp.tile_reduce(
      _minimum_support, wp.tile(support3, preserve_type=True))
  supports = _PlaneSupportSet()
  supports.s0 = selected0
  supports.s1 = selected1
  supports.s2 = selected2
  supports.s3 = wp.tile_extract(reduced3, 0)
  supports = _sort_plane_supports(supports)
  if lane == 0:
    context.row = 0
    context.support = supports.s0
    _append_target_plane(model, context, output)
    context.row = 1
    context.support = supports.s1
    _append_target_plane(model, context, output)
    context.row = 2
    context.support = supports.s2
    _append_target_plane(model, context, output)
    context.row = 3
    context.support = supports.s3
    _append_target_plane(model, context, output)


def create_collision_models(graph, device):
  models = tuple(create_collision_model(host, device) for host in graph.pairs)
  if not models:
    return None
  offset = 0
  for model in models:
    model.contact_id_offset = offset
    offset += model.contact_count
  return models[0] if len(models) == 1 else models


def create_collision_model(host, device):
  model = GaussianCollisionModel()
  _copy_geometry(model, host, device)
  _copy_bvh(model, host, device)
  _copy_layout(model, host, device)
  _copy_contact_parameters(model, host, device)
  model.distance_offset = wp.zeros(1, dtype=float, device=device)
  model.target_body_id = host.reference_body_id
  model.plane_geom_id = host.plane_geom_id
  model.target_plane_enabled = int(host.reference_plane_enabled)
  model.target_contact_offset = (
      PLANE_CONTACT_COUNT if host.reference_plane_enabled else 0)
  model.contact_id_offset = 0
  model.contact_count = host.contact_geom.shape[0]
  model.target_count = host.reference_centers.shape[0]
  model.threshold = float(host.contact_threshold)
  model.contact_topk = 1
  return model


def _copy_geometry(model, host, device):
  target_spheres = np.concatenate(
      (host.reference_centers, host.reference_radii[:, None]), axis=1)
  model.target_spheres = wp.array(target_spheres, dtype=wp.vec4, device=device)
  model.source_centers = wp.array(host.query_centers, dtype=wp.vec3, device=device)
  model.source_radii = wp.array(host.query_radii, dtype=float, device=device)
  model.source_body_ids = wp.array(host.query_body_ids, dtype=int, device=device)


def _copy_bvh(model, host, device):
  from comfree_warp.gaussian_bvh import cached_sphere_bvh
  from comfree_warp.gaussian_bvh import TARGET_BVH_LEAF_SIZE
  bvh = cached_sphere_bvh(
      host.reference_centers, host.reference_radii, TARGET_BVH_LEAF_SIZE)
  model.bvh_center_lower = wp.array(
      bvh.center_lower, dtype=wp.vec3, device=device)
  model.bvh_center_upper = wp.array(
      bvh.center_upper, dtype=wp.vec3, device=device)
  model.bvh_max_radius = wp.array(
      bvh.max_radius, dtype=float, device=device)
  model.bvh_ranges = wp.array(bvh.ranges, dtype=wp.vec2i, device=device)
  model.bvh_escape = wp.array(bvh.escape, dtype=int, device=device)
  model.bvh_sphere_ids = wp.array(
      bvh.sphere_ids, dtype=int, device=device)


def _copy_layout(model, host, device):
  model.output_starts = wp.array(host.output_starts, dtype=int, device=device)
  model.output_counts = wp.array(host.output_counts, dtype=int, device=device)
  model.target_output_ids = wp.array(
      host.query_output_ids, dtype=int, device=device)


def _copy_contact_parameters(model, host, device):
  model.contact_geom = wp.array(host.contact_geom, dtype=wp.vec2i, device=device)
  model.contact_normal_sign = wp.array(
      host.contact_normal_sign, dtype=float, device=device)
  model.contact_dim = wp.array(host.contact_dim, dtype=int, device=device)
  model.contact_includemargin = wp.array(
      host.contact_includemargin, dtype=float, device=device)
  model.contact_friction = wp.array(host.contact_friction, dtype=vec5, device=device)
  model.contact_solref = wp.array(host.contact_solref, dtype=wp.vec2, device=device)
  model.contact_solreffriction = wp.array(
      host.contact_solreffriction, dtype=wp.vec2, device=device)
  model.contact_solimp = wp.array(host.contact_solimp, dtype=vec5, device=device)


def _workspace(model, data, cache_index=None):
  if cache_index is None:
    workspace = getattr(data, "gaussian_collision_state", None)
  else:
    workspace = _indexed_cache(data, "gaussian_collision_states", cache_index)
  if workspace is not None:
    return workspace
  state = _new_workspace(model, data)
  if cache_index is None:
    data.gaussian_collision_state = state
  else:
    _store_indexed_cache(
        data, "gaussian_collision_states", cache_index, value=state)
  return state


def _new_workspace(model, data):
  state = GaussianCollisionState()
  state.xpos = data.xpos
  state.xmat = data.xmat
  state.geom_xpos = data.geom_xpos
  state.geom_xmat = data.geom_xmat
  state.nworld = data.nworld
  return state


def _indexed_cache(data, name, index):
  cache = getattr(data, name, None)
  if cache is None or index >= len(cache):
    return None
  return cache[index]


def _store_indexed_cache(data, name, index, *, value):
  cache = list(getattr(data, name, ()))
  cache.extend([None] * (index + 1 - len(cache)))
  cache[index] = value
  setattr(data, name, cache)


def _contact_output(data):
  cached = getattr(data, "gaussian_contact_output", None)
  if cached is not None:
    return cached
  output = GaussianContactOutput()
  contact = data.contact
  output.dist = contact.dist
  output.pos = contact.pos
  output.frame = contact.frame
  output.includemargin = contact.includemargin
  output.friction = contact.friction
  output.solref = contact.solref
  output.solreffriction = contact.solreffriction
  output.solimp = contact.solimp
  output.dim = contact.dim
  output.geom = contact.geom
  output.efc_address = contact.efc_address
  output.worldid = contact.worldid
  output.type = contact.type
  output.geomcollisionid = contact.geomcollisionid
  output.nacon = data.nacon
  output.capacity = data.naconmax
  output.efc_width = contact.efc_address.shape[1]
  data.gaussian_contact_output = output
  return output


def _validate_capacity(contact_count, data):
  required = contact_count * data.nworld
  if data.naconmax >= required:
    return
  per_world = (required + data.nworld - 1) // data.nworld
  raise ValueError(
      f"Gaussian collision needs nconmax >= {per_world} per world; "
      f"allocated total {data.naconmax} for {data.nworld} worlds")


@event_scope
def collision(model, data):
  """Runs the canonical grouped-BVH Gaussian collision pipeline."""
  from . import collision_gaussian_bvh
  collision_gaussian_bvh.collision(model, data)
