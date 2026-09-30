"""Contact selection and output for query-point collision (Gaussian or triangle-mesh targets).

Pair slots are selected and fused by ``comfree_warp.native_adjoint.gaussian_collision``; this
module selects the four plane supports of a target's static spheres and appends fused contacts
to MuJoCo Warp's contact buffers for the generic step.
"""

from __future__ import annotations

import warp as wp

from .collision_gaussian import PLANE_BLOCK_DIM
from .collision_gaussian import GaussianCollisionModel
from .collision_gaussian import GaussianCollisionState
from .collision_gaussian import GaussianContactOutput
from .collision_gaussian import _ContactCandidate
from .collision_gaussian import _PlaneSupportSet
from .collision_gaussian import _append_contact
from .collision_gaussian import _contact_output
from .collision_gaussian import _minimum_support
from .collision_gaussian import _plane_reduction_context
from .collision_gaussian import _scan_deepest_plane_support
from .collision_gaussian import _scan_farthest_plane_support
from .collision_gaussian import _sort_plane_supports
from .collision_gaussian import _target_plane_context
from .collision_gaussian import _validate_capacity
from .collision_gaussian import _workspace
from .collision_gaussian import _write_target_plane


wp.set_module_options({"enable_backward": False})


@wp.struct
class PairSelection:
  # Plane rows: selected support sphere per row (source ids unused, -1).
  source_id: wp.array2d(dtype=int)
  target_id: wp.array2d(dtype=int)
  # Ranked candidates per (world, pair slot, rank): source id, target component id and the
  # component's feature in the target body frame (collision_targets: anchor, axis).
  candidate_source: wp.array3d(dtype=int)
  candidate_target: wp.array3d(dtype=int)
  candidate_anchor: wp.array3d(dtype=wp.vec4)
  candidate_axis: wp.array3d(dtype=wp.vec4)


@wp.kernel(enable_backward=False)
def select_plane(model: GaussianCollisionModel,
                 state: GaussianCollisionState,
                 selection: PairSelection):
  world, lane = wp.tid()
  context = _target_plane_context(model, state, world)
  reduction = _plane_reduction_context(model, context)

  support0 = _scan_deepest_plane_support(model, reduction, lane)
  reduced0 = wp.tile_reduce(
      _minimum_support, wp.tile(support0, preserve_type=True))
  selected0 = wp.tile_extract(reduced0, 0)
  reduction.selected0 = selected0.sphere_id
  reduction.selected_count = wp.where(selected0.sphere_id >= 0, 1, 0)

  support1 = _scan_farthest_plane_support(model, reduction, lane)
  reduced1 = wp.tile_reduce(
      _minimum_support, wp.tile(support1, preserve_type=True))
  selected1 = wp.tile_extract(reduced1, 0)
  reduction.selected1 = selected1.sphere_id
  reduction.selected_count = reduction.selected_count + wp.where(
      selected1.sphere_id >= 0, 1, 0)

  support2 = _scan_farthest_plane_support(model, reduction, lane)
  reduced2 = wp.tile_reduce(
      _minimum_support, wp.tile(support2, preserve_type=True))
  selected2 = wp.tile_extract(reduced2, 0)
  reduction.selected2 = selected2.sphere_id
  reduction.selected_count = reduction.selected_count + wp.where(
      selected2.sphere_id >= 0, 1, 0)

  support3 = _scan_farthest_plane_support(model, reduction, lane)
  reduced3 = wp.tile_reduce(
      _minimum_support, wp.tile(support3, preserve_type=True))
  supports = _PlaneSupportSet()
  supports.s0 = selected0
  supports.s1 = selected1
  supports.s2 = selected2
  supports.s3 = wp.tile_extract(reduced3, 0)
  supports = _sort_plane_supports(supports)
  if lane == 0:
    selection.source_id[world, 0] = -1
    selection.source_id[world, 1] = -1
    selection.source_id[world, 2] = -1
    selection.source_id[world, 3] = -1
    selection.target_id[world, 0] = supports.s0.sphere_id
    selection.target_id[world, 1] = supports.s1.sphere_id
    selection.target_id[world, 2] = supports.s2.sphere_id
    selection.target_id[world, 3] = supports.s3.sphere_id


@wp.kernel
def _append_fused_pairs(model: GaussianCollisionModel, distance: wp.array2d(dtype=float),
                        position: wp.array2d(dtype=wp.vec3), frame: wp.array2d(dtype=wp.mat33),
                        active: wp.array2d(dtype=int), contact_offset: int,
                        output: GaussianContactOutput):
  enabled, world = wp.tid()
  row = model.target_contact_offset + enabled
  column = contact_offset + row
  if active[world, column] == 0:
    return
  candidate = _ContactCandidate()
  candidate.row = row
  candidate.world = world
  candidate.distance = distance[world, column]
  candidate.position = position[world, column]
  # The fused frame already carries the row's normal sign; _append_contact applies it again.
  axis = frame[world, column]
  candidate.normal = wp.vec3(axis[0, 0], axis[0, 1], axis[0, 2]) * model.contact_normal_sign[row]
  _append_contact(model, candidate, output)


def _fused_collision(model, data):
  """Unified pair collision (Gaussian or mesh target, ranked fusion), cached per data."""
  from comfree_warp.native_adjoint import gaussian_collision as unified
  cached = getattr(data, "fused_collision", None)
  if cached is None or cached[0] is not model:
    compiled = unified.compile_collision(model)
    allocation = unified.CollisionAllocation(data.nworld, data.qpos.device, False)
    cached = (model, compiled, unified.allocate_workspace(compiled, allocation))
    data.fused_collision = cached
  _, compiled, workspace = cached
  frames = unified.CollisionFrames(data.xpos, data.xmat, data.geom_xpos, data.geom_xmat)
  unified.broadphase(compiled, frames, workspace)
  return compiled, unified.narrowphase(compiled, workspace)


def collision(model, data) -> None:
  """Appends contacts from every compiled collision batch."""
  graph = getattr(model, "gaussian_collision", None)
  if graph is None:
    return
  batches = graph if isinstance(graph, tuple) else (graph,)
  contact_count = sum(batch.contact_count for batch in batches)
  if contact_count == 0:
    return
  _validate_capacity(contact_count, data)
  output = _contact_output(data)
  compiled, contacts = _fused_collision(model, data)
  for gaussian, batch in zip(batches, compiled.batches):
    if batch.source_target_count:
      wp.launch(_append_fused_pairs, dim=(batch.source_target_count, data.nworld),
                inputs=[gaussian, contacts.distance, contacts.position, contacts.frame,
                        contacts.active, batch.contact_offset, output])
    if gaussian.target_plane_enabled:
      state = _workspace(gaussian, data, None if len(batches) == 1 else compiled.batches.index(batch))
      wp.launch_tiled(_write_target_plane, dim=data.nworld, inputs=[gaussian, state, output],
                      block_dim=PLANE_BLOCK_DIM)
