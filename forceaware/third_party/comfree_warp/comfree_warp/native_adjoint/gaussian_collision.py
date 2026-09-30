"""Fixed-slot differentiable collision between query-point sources and a Gaussian or mesh target.

Every hand collision group owns one slot. Hard and mesh use ranked nearest-pair
selection and `contact_fusion`. GS soft dispatches to complete-support continuous
fusion, reusing the same BVH and nearest-pair seeds. The target kind comes from
scene geoms (`gs` or `sdfmesh`). Plane rows retain the deepest-plus-farthest
supports of the target's static spheres (Gaussians or mesh hull vertices).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from comfree_warp import collision_targets as targets
from comfree_warp import contact_fusion as fusion
from comfree_warp.collision_config import stop_normal_gradient
from comfree_warp.comfree_core._src import collision_gaussian as search
from comfree_warp.comfree_core._src import collision_gaussian_bvh as bvh


PLANE_CONTACTS = 4
GS_BROADPHASE_EPS = search.GS_BROADPHASE_EPS
WEIGHT_EPSILON = fusion.WEIGHT_EPSILON


ContactSelection = bvh.PairSelection


@wp.struct
class FixedContacts:
    distance: wp.array2d(dtype=float)
    position: wp.array2d(dtype=wp.vec3)
    frame: wp.array2d(dtype=wp.mat33)
    active: wp.array2d(dtype=int)


@wp.struct
class CollisionJob:
    state: search.GaussianCollisionState
    selected: ContactSelection
    contacts: FixedContacts
    contact_offset: int
    freeze_frame_vjp: int
    stop_frame_vjp: int


@dataclass(frozen=True)
class CollisionFrames:
    body_position: wp.array
    body_matrix: wp.array
    geom_position: wp.array
    geom_matrix: wp.array


@dataclass(frozen=True)
class CollisionAllocation:
    worlds: int
    device: object
    gradient: bool


@dataclass(frozen=True)
class CompiledGaussianBatch:
    model: object
    contact_offset: int
    source_target_count: int
    has_target_plane: bool
    target: targets.TargetResources
    source_slot: wp.array


@dataclass(frozen=True)
class CompiledContactLayout:
    contact_count: int
    contact_geom: object
    contact_normal_sign: object
    contact_dim: object
    contact_includemargin: object
    contact_friction: object
    contact_solref: object
    contact_solreffriction: object
    contact_solimp: object


@dataclass(frozen=True)
class CompiledGaussianCollision:
    batches: tuple[CompiledGaussianBatch, ...]
    contact_layout: CompiledContactLayout
    contact_count: int
    output_count: int
    output_contact_rows: object
    output_thresholds: object
    output_body_ids: object


@dataclass(frozen=True)
class SlotBuffers:
    bounds: wp.array
    distance: wp.array
    count: wp.array
    items: wp.array


@dataclass(frozen=True)
class GaussianBatchWorkspace:
    state: object
    selection: ContactSelection
    slots: SlotBuffers


@dataclass(frozen=True)
class GaussianCollisionWorkspace:
    batches: tuple[GaussianBatchWorkspace, ...]
    contacts: FixedContacts


_normalized = fusion.unit
_selected_frame = fusion.selected_frame


@wp.func
def _clear_contact(contacts: FixedContacts, world: int, row: int):
    contacts.distance[world, row] = 0.0
    contacts.position[world, row] = wp.vec3(0.0)
    contacts.frame[world, row] = wp.identity(n=3, dtype=float)
    contacts.active[world, row] = 0


# -- broadphase -----------------------------------------------------------------------------------
@wp.func
def _source_local(model: search.GaussianCollisionModel, state: search.GaussianCollisionState,
                  world: int, source: int) -> wp.vec3:
    """Source centre in the target body frame, through the source-to-target body transform.

    The operation order is the original Gaussian broadphase's, so near-ties resolve alike.
    """
    body = model.source_body_ids[source]
    inverse = wp.transpose(state.xmat[world, model.target_body_id])
    position = inverse @ (state.xpos[world, body] - state.xpos[world, model.target_body_id])
    rotation = inverse @ state.xmat[world, body]
    return position + rotation @ model.source_centers[source]


@wp.kernel(enable_backward=False)
def _slot_bounds(model: search.GaussianCollisionModel, state: search.GaussianCollisionState,
                 target: targets.CollisionTarget, candidate_source: wp.array3d(dtype=int),
                 candidate_target: wp.array3d(dtype=int), bounds: wp.array2d(dtype=float),
                 enabled_slots: wp.array2d(dtype=int)):
    """Safe upper bound on each slot's boundary distance, from the previous candidates re-evaluated.

    The previous pairs are distinct pairs of the current configuration, so the largest of their
    current distances bounds the (k+1)-th smallest pair distance. It only tightens pruning. One
    lane per rank; a missing or out-of-reach candidate gives no bound.
    """
    enabled, world, lane = wp.tid()
    if enabled_slots and enabled_slots[world, enabled] == 0:
        return
    value = float(-1.0e30)
    if lane < candidate_source.shape[2]:
        value = 1.0e30
        source = candidate_source[world, enabled, lane]
        if source >= 0:
            offset = search._distance_offset(model, world)
            radius = model.source_radii[source]
            local = _source_local(model, state, world, source)
            if target.kind == targets.MESH_TARGET:
                # One component per source: the exact signed distance of its nearest triangle.
                values, ids = targets.nearest_components(target, local, model.threshold + radius - offset, 1)
                if ids[0] >= 0:
                    value = values[0] - radius + offset
            else:
                sphere = model.target_spheres[candidate_target[world, enabled, lane]]
                value = wp.length(local - search._sphere_center(sphere)) - sphere[3] - radius + offset
    bound = wp.tile_extract(wp.tile_max(wp.tile(value)), 0)
    if lane == 0:
        bounds[world, enabled] = wp.min(bound, model.threshold)


@wp.kernel(enable_backward=False)
def _source_nearest(model: search.GaussianCollisionModel, state: search.GaussianCollisionState,
                    target: targets.CollisionTarget, source_slot: wp.array(dtype=int),
                    bounds: wp.array2d(dtype=float),
                    distance: wp.array2d(dtype=float), slot_count: wp.array2d(dtype=int),
                    slot_items: wp.array3d(dtype=int), enabled_slots: wp.array2d(dtype=int)):
    """Nearest-pair distance of every source; below-threshold sources join their slot's list.

    Bounds pruning never drops a selectable source: its pair distance would exceed the threshold.
    """
    source, world = wp.tid()
    slot = source_slot[source]
    if slot < 0:
        return
    if enabled_slots and enabled_slots[world, slot] == 0:
        return
    distance[world, source] = model.threshold
    offset = search._distance_offset(model, world)
    radius = model.source_radii[source]
    cutoff = bounds[world, slot]
    reach = cutoff + radius - offset + GS_BROADPHASE_EPS
    local = _source_local(model, state, world, source)
    if targets.outside_bounds(target, local, reach):
        return
    nearest, ids = targets.nearest_components(target, local, reach, 1)
    if ids[0] < 0:
        return
    value = nearest[0] - radius + offset
    # A loose prefilter; ``_select_slots`` applies the exact contact test.
    if value > cutoff + GS_BROADPHASE_EPS:
        return
    distance[world, source] = value
    item = wp.atomic_add(slot_count, world, slot, 1)
    slot_items[world, slot, item] = source


SELECT_BLOCK_DIM = 16


@wp.struct
class RankedPairs:
    """Up to ``MAX_RANKS`` pairs sorted by (distance, source, component); missing pairs have source -1."""
    distance: targets.vec_ranks
    source: targets.ivec_ranks
    component: targets.ivec_ranks


@wp.func
def _empty_pairs(value: float):
    pairs = RankedPairs()
    for rank in range(targets.MAX_RANKS):
        pairs.distance[rank] = value
        pairs.source[rank] = -1
        pairs.component[rank] = -1
    return pairs


@wp.func
def _before(distance: float, source: int, component: int,
            other_distance: float, other_source: int, other_component: int) -> bool:
    # Total order (distance, component, source), the original Gaussian broadphase's tie order; the
    # selection is independent of scheduling and reduction order.
    if source < 0:
        return False
    if other_source < 0 or distance < other_distance:
        return True
    if distance > other_distance:
        return False
    if component != other_component:
        return component < other_component
    return source < other_source


@wp.func
def _insert(pairs: RankedPairs, distance: float, source: int, component: int):
    result = pairs
    last = targets.MAX_RANKS - 1
    if _before(distance, source, component, result.distance[last], result.source[last], result.component[last]):
        slot = int(last)
        while slot > 0 and _before(distance, source, component, result.distance[slot - 1],
                                   result.source[slot - 1], result.component[slot - 1]):
            result.distance[slot] = result.distance[slot - 1]
            result.source[slot] = result.source[slot - 1]
            result.component[slot] = result.component[slot - 1]
            slot -= 1
        result.distance[slot] = distance
        result.source[slot] = source
        result.component[slot] = component
    return result


@wp.func
def _merge(left: RankedPairs, right: RankedPairs):
    """Merge of two ranked lists; associative and commutative under the total order."""
    result = left
    for rank in range(targets.MAX_RANKS):
        result = _insert(result, right.distance[rank], right.source[rank], right.component[rank])
    return result


@wp.kernel(enable_backward=False)
def _select_slots(model: search.GaussianCollisionModel, state: search.GaussianCollisionState,
                  target: targets.CollisionTarget, ranks: int,
                  distance: wp.array2d(dtype=float), slot_count: wp.array2d(dtype=int),
                  slot_items: wp.array3d(dtype=int), bounds: wp.array2d(dtype=float),
                  selection: ContactSelection, enabled_slots: wp.array2d(dtype=int)):
    """The ``ranks`` nearest (source, component) pairs of a slot, with their features.

    A block cooperates per slot. A source in the top pairs is among the ``ranks`` sources nearest
    by their own nearest pair, so only those sources are expanded and the selection is exact.
    """
    enabled, world, lane = wp.tid()
    if enabled_slots and enabled_slots[world, enabled] == 0:
        return
    # An empty slot needs no ranked-list reduction or second BVH traversal.
    # Clear the complete record, including features left by a previous contact.
    if slot_count[world, enabled] == 0:
        if lane < ranks:
            selection.candidate_source[world, enabled, lane] = -1
            selection.candidate_target[world, enabled, lane] = -1
            selection.candidate_anchor[world, enabled, lane] = wp.vec4(0.0)
            selection.candidate_axis[world, enabled, lane] = wp.vec4(0.0)
        return
    near = _empty_pairs(model.threshold)
    k = lane
    while k < slot_count[world, enabled]:
        source = slot_items[world, enabled, k]
        near = _insert(near, distance[world, source], source, 0)
        k += SELECT_BLOCK_DIM
    near = wp.tile_extract(wp.tile_reduce(_merge, wp.tile(near, preserve_type=True)), 0)
    best = _empty_pairs(model.threshold)
    offset = search._distance_offset(model, world)
    if lane < ranks:
        source = near.source[lane]
        if source >= 0:
            radius = model.source_radii[source]
            local = _source_local(model, state, world, source)
            reach = bounds[world, enabled] + radius - offset + GS_BROADPHASE_EPS
            values, ids = targets.nearest_components(target, local, reach, ranks)
            for rank in range(targets.MAX_RANKS):
                if rank < ranks and ids[rank] >= 0:
                    # Contact test: pair distance below the threshold; for spheres exactly as the
                    # original broadphase, |c - s| - (r_s + r_t).
                    value = values[rank] - radius
                    if target.kind == targets.SPHERE_TARGET:
                        sphere = target.spheres[ids[rank]]
                        value = wp.length(search._sphere_center(sphere) - local) - (radius + sphere[3])
                    if value < model.threshold - offset:
                        best = _insert(best, value, source, ids[rank])
    best = wp.tile_extract(wp.tile_reduce(_merge, wp.tile(best, preserve_type=True)), 0)
    if lane < ranks:
        source = best.source[lane]
        anchor = wp.vec4(0.0)
        axis = wp.vec4(0.0)
        if source >= 0:
            anchor, axis = targets.component_feature(
                target, _source_local(model, state, world, source), best.component[lane])
        selection.candidate_source[world, enabled, lane] = source
        selection.candidate_target[world, enabled, lane] = best.component[lane]
        selection.candidate_anchor[world, enabled, lane] = anchor
        selection.candidate_axis[world, enabled, lane] = axis


# -- narrowphase ----------------------------------------------------------------------------------
@wp.func
def slot_candidates(model: search.GaussianCollisionModel, job: CollisionJob, world: int, enabled: int):
    """World source centres and target features of a slot's ranked candidates."""
    sources = fusion.mat_ranks3()
    anchors = fusion.mat_ranks3()
    axes = fusion.mat_ranks3()
    source_radii = targets.vec_ranks()
    radii = targets.vec_ranks()
    codes = targets.ivec_ranks()
    ranks = job.selected.candidate_source.shape[2]
    body = model.target_body_id
    for rank in range(targets.MAX_RANKS):
        if rank < ranks:
            source = job.selected.candidate_source[world, enabled, rank]
            if source >= 0:
                source_body = model.source_body_ids[source]
                sources[rank] = (job.state.xpos[world, source_body]
                                 + job.state.xmat[world, source_body] @ model.source_centers[source])
                anchor = job.selected.candidate_anchor[world, enabled, rank]
                axis = job.selected.candidate_axis[world, enabled, rank]
                anchors[rank] = job.state.xpos[world, body] + job.state.xmat[world, body] @ search._sphere_center(anchor)
                axes[rank] = job.state.xmat[world, body] @ wp.vec3(axis[0], axis[1], axis[2])
                source_radii[rank] = model.source_radii[source]
                radii[rank] = anchor[3]
                codes[rank] = int(axis[3])
    c = fusion.Candidates()
    c.source = sources
    c.anchor = anchors
    c.axis = axes
    c.source_radius = source_radii
    c.radius = radii
    c.code = codes
    return c


@wp.kernel
def _narrow_pairs(model: search.GaussianCollisionModel, job: CollisionJob):
    enabled, world = wp.tid()
    local_row = model.target_contact_offset + enabled
    global_row = job.contact_offset + local_row
    candidates = slot_candidates(model, job, world, enabled)
    count, distance, position, normal = fusion.fuse(
        candidates, model.contact_topk, model.threshold, search._distance_offset(model, world))
    if count == 0:
        _clear_contact(job.contacts, world, global_row)
        return
    job.contacts.distance[world, global_row] = distance
    job.contacts.position[world, global_row] = position
    job.contacts.frame[world, global_row] = fusion.configured_frame(
        normal * model.contact_normal_sign[local_row], job.freeze_frame_vjp, job.stop_frame_vjp)
    job.contacts.active[world, global_row] = wp.where(distance < model.threshold, 1, 0)


@wp.kernel
def _narrow_plane(model: search.GaussianCollisionModel, job: CollisionJob):
    local_row, world = wp.tid()
    global_row = job.contact_offset + local_row
    target_id = job.selected.target_id[world, local_row]
    if target_id < 0:
        _clear_contact(job.contacts, world, global_row)
        return
    sphere = model.target_spheres[target_id]
    center = job.state.xpos[world, model.target_body_id]
    center += (
        job.state.xmat[world, model.target_body_id]
        @ search._sphere_center(sphere))
    plane_matrix = job.state.geom_xmat[world, model.plane_geom_id]
    normal = wp.vec3(plane_matrix[0, 2], plane_matrix[1, 2],
                     plane_matrix[2, 2])
    plane = job.state.geom_xpos[world, model.plane_geom_id]
    distance = wp.dot(center - plane, normal) - sphere[3]
    distance_offset = search._distance_offset(model, world)
    distance += distance_offset
    position = search._inset_contact_position(
        center - normal * sphere[3], normal, distance_offset)
    job.contacts.distance[world, global_row] = distance
    job.contacts.position[world, global_row] = position
    contact_normal = normal * model.contact_normal_sign[local_row]
    job.contacts.frame[world, global_row] = fusion.configured_frame(
        contact_normal, job.freeze_frame_vjp, job.stop_frame_vjp)
    job.contacts.active[world, global_row] = wp.where(
        distance < model.contact_includemargin[local_row], 1, 0)


# -- compilation and workspaces -------------------------------------------------------------------
def candidate_ranks(model) -> int:
    """Stored candidates per slot: the top k plus the boundary (k+1)-th."""
    topk = int(model.contact_topk)
    if not 1 <= topk <= targets.MAX_TOPK:
        raise ValueError(f"contact_topk must be in [1, {targets.MAX_TOPK}], got {topk}")
    return topk + 1


def _models(device_model) -> tuple[object, ...]:
    graph = getattr(device_model, "gaussian_collision", None)
    if graph is None:
        raise ValueError("model does not contain query-point collision data")
    models = graph if isinstance(graph, tuple) else (graph,)
    if not models:
        raise ValueError("collision graph contains no batches")
    return models


def _source_slots(model) -> np.ndarray:
    """Pair slot of every source (its group's enabled row), -1 for groups without a slot."""
    starts, counts = model.output_starts.numpy(), model.output_counts.numpy()
    slots = np.full(model.source_centers.shape[0], -1, np.int32)
    for enabled, output in enumerate(model.target_output_ids.numpy()):
        slots[starts[output]:starts[output] + counts[output]] = enabled
    return slots


def _compiled_batch(model, contact_offset: int, meshes: dict) -> CompiledGaussianBatch:
    if int(model.contact_id_offset) != contact_offset:
        raise ValueError(
            "collision contact_id_offset must be contiguous: "
            f"expected {contact_offset}, got {int(model.contact_id_offset)}")
    pair_count = int(model.target_output_ids.shape[0])
    plane_count = PLANE_CONTACTS if model.target_plane_enabled else 0
    if int(model.target_contact_offset) != plane_count:
        raise ValueError(
            "target_contact_offset does not match plane layout: "
            f"expected {plane_count}, got {int(model.target_contact_offset)}")
    expected = int(model.target_contact_offset) + pair_count
    if int(model.contact_count) != expected:
        raise ValueError(
            f"collision batch has {int(model.contact_count)} contacts; "
            f"expected {expected} from its plane and pair outputs")
    candidate_ranks(model)
    device = model.target_output_ids.device
    resources = targets.TargetResources(model, meshes.get(int(model.target_body_id)), device)
    return CompiledGaussianBatch(
        model, contact_offset, pair_count, bool(model.target_plane_enabled), resources,
        wp.array(_source_slots(model), dtype=int, device=device))


def _concat(models, name: str, dtype, device):
    values = [np.asarray(getattr(model, name).numpy()) for model in models]
    return wp.array(np.concatenate(values), dtype=dtype, device=device)


def _contact_layout(models, contact_count: int, device) -> CompiledContactLayout:
    layout = CompiledContactLayout(
        contact_count=contact_count,
        contact_geom=_concat(models, "contact_geom", wp.vec2i, device),
        contact_normal_sign=_concat(models, "contact_normal_sign", float, device),
        contact_dim=_concat(models, "contact_dim", int, device),
        contact_includemargin=_concat(models, "contact_includemargin", float, device),
        contact_friction=_concat(models, "contact_friction", search.vec5, device),
        contact_solref=_concat(models, "contact_solref", wp.vec2, device),
        contact_solreffriction=_concat(models, "contact_solreffriction", wp.vec2, device),
        contact_solimp=_concat(models, "contact_solimp", search.vec5, device),
    )
    names = (
        "contact_geom", "contact_normal_sign", "contact_dim",
        "contact_includemargin", "contact_friction", "contact_solref",
        "contact_solreffriction", "contact_solimp")
    invalid = [name for name in names if getattr(layout, name).shape[0] != contact_count]
    if invalid:
        raise ValueError("contact metadata does not match contact_count: " + ", ".join(invalid))
    return layout


def _output_metadata(batch: CompiledGaussianBatch):
    model = batch.model
    target_outputs = model.target_output_ids.numpy().astype(np.int32)
    output_starts = model.output_starts.numpy().astype(np.int32)
    if np.unique(target_outputs).size != target_outputs.size:
        raise ValueError("collision batch maps multiple contacts to one output")
    if np.any((target_outputs < 0) | (target_outputs >= output_starts.size)):
        raise ValueError("target output index is outside output layout")
    output_rows = np.full(output_starts.size, -1, dtype=np.int32)
    local_rows = model.target_contact_offset + np.arange(target_outputs.size)
    output_rows[target_outputs] = batch.contact_offset + local_rows
    source_bodies = model.source_body_ids.numpy().astype(np.int32)
    output_bodies = source_bodies[output_starts]
    thresholds = np.full(output_starts.size, model.threshold, np.float32)
    return output_rows, output_bodies, thresholds


def compile_collision(device_model) -> CompiledGaussianCollision:
    models = _models(device_model)
    meshes = getattr(device_model, "mesh_targets", {}) or {}
    batches, offset = [], 0
    for model in models:
        batches.append(_compiled_batch(model, offset, meshes))
        offset += int(model.contact_count)
    metadata = [_output_metadata(batch) for batch in batches]
    output_rows = np.concatenate([item[0] for item in metadata])
    output_bodies = np.concatenate([item[1] for item in metadata])
    output_thresholds = np.concatenate([item[2] for item in metadata])
    device = models[0].target_output_ids.device
    return CompiledGaussianCollision(
        tuple(batches), _contact_layout(models, offset, device), offset,
        int(output_rows.size), wp.array(output_rows, dtype=int, device=device),
        wp.array(output_thresholds, dtype=float, device=device),
        wp.array(output_bodies, dtype=int, device=device))


def _selection(batch: CompiledGaussianBatch, allocation: CollisionAllocation) -> ContactSelection:
    shape = (allocation.worlds, int(batch.model.contact_count))
    selected = ContactSelection()
    selected.source_id = wp.full(shape, -1, dtype=int, device=allocation.device)
    selected.target_id = wp.full(shape, -1, dtype=int, device=allocation.device)
    candidates = (allocation.worlds, max(batch.source_target_count, 1), candidate_ranks(batch.model))
    selected.candidate_source = wp.full(candidates, -1, dtype=int, device=allocation.device)
    selected.candidate_target = wp.full(candidates, -1, dtype=int, device=allocation.device)
    selected.candidate_anchor = wp.zeros(candidates, dtype=wp.vec4, device=allocation.device)
    selected.candidate_axis = wp.zeros(candidates, dtype=wp.vec4, device=allocation.device)
    return selected


def _slots(batch: CompiledGaussianBatch, allocation: CollisionAllocation) -> SlotBuffers:
    model = batch.model
    counts = model.output_counts.numpy()
    capacity = max(int(counts.max()) if counts.size else 1, 1)
    pairs = max(batch.source_target_count, 1)
    return SlotBuffers(
        wp.empty((allocation.worlds, pairs), dtype=float, device=allocation.device),
        wp.empty((allocation.worlds, int(model.source_centers.shape[0])), dtype=float,
                 device=allocation.device),
        wp.zeros((allocation.worlds, pairs), dtype=int, device=allocation.device),
        wp.empty((allocation.worlds, pairs, capacity), dtype=int, device=allocation.device))


def _state(batch: CompiledGaussianBatch, allocation: CollisionAllocation):
    state = search.GaussianCollisionState()
    state.nworld = allocation.worlds
    return state


def _contacts(compiled: CompiledGaussianCollision, allocation: CollisionAllocation) -> FixedContacts:
    shape = (allocation.worlds, compiled.contact_count)
    contacts = FixedContacts()
    gradient = dict(requires_grad=allocation.gradient, retain_grad=allocation.gradient)
    contacts.distance = wp.empty(shape, dtype=float, device=allocation.device, **gradient)
    contacts.position = wp.empty(shape, dtype=wp.vec3, device=allocation.device, **gradient)
    contacts.frame = wp.empty(shape, dtype=wp.mat33, device=allocation.device, **gradient)
    contacts.active = wp.empty(shape, dtype=int, device=allocation.device)
    return contacts


def allocate_workspace(compiled: CompiledGaussianCollision,
                       allocation: CollisionAllocation) -> GaussianCollisionWorkspace:
    batches = tuple(
        GaussianBatchWorkspace(_state(batch, allocation), _selection(batch, allocation),
                               _slots(batch, allocation))
        for batch in compiled.batches)
    return GaussianCollisionWorkspace(batches, _contacts(compiled, allocation))


def _bind_frames(workspace: GaussianBatchWorkspace, frames: CollisionFrames) -> None:
    workspace.state.xpos = frames.body_position
    workspace.state.xmat = frames.body_matrix
    workspace.state.geom_xpos = frames.geom_position
    workspace.state.geom_xmat = frames.geom_matrix


def _broadphase_discrete(batch: CompiledGaussianBatch, frames: CollisionFrames,
                      workspace: GaussianBatchWorkspace) -> None:
    model, state, slots = batch.model, workspace.state, workspace.slots
    device = frames.body_position.device
    _bind_frames(workspace, frames)
    target = batch.target.target
    if batch.source_target_count:
        slots.count.zero_()
        wp.launch_tiled(_slot_bounds, dim=(batch.source_target_count, state.nworld),
                        inputs=[model, state, target, workspace.selection.candidate_source,
                                workspace.selection.candidate_target, slots.bounds, None],
                        block_dim=SELECT_BLOCK_DIM, device=device)
        wp.launch(_source_nearest, dim=(model.source_centers.shape[0], state.nworld),
                  inputs=[model, state, target, batch.source_slot, slots.bounds,
                          slots.distance, slots.count, slots.items, None], device=device)
        wp.launch_tiled(_select_slots, dim=(batch.source_target_count, state.nworld),
                        inputs=[model, state, target, candidate_ranks(model), slots.distance,
                                slots.count, slots.items, slots.bounds, workspace.selection, None],
                        block_dim=SELECT_BLOCK_DIM, device=device)
    if batch.has_target_plane:
        wp.launch_tiled(bvh.select_plane, dim=state.nworld,
                        inputs=[model, state, workspace.selection],
                        block_dim=search.PLANE_BLOCK_DIM, device=device)


def broadphase(compiled: CompiledGaussianCollision, frames: CollisionFrames,
               workspace: GaussianCollisionWorkspace) -> None:
    for batch, batch_workspace in zip(compiled.batches, workspace.batches, strict=True):
        _broadphase_batch(batch, frames, batch_workspace)


def _narrowphase_discrete(batch: CompiledGaussianBatch, workspace: GaussianBatchWorkspace,
                       contacts: FixedContacts, *, freeze_frame_vjp: bool) -> None:
    model = batch.model
    state = workspace.state
    job = CollisionJob()
    job.state = state
    job.selected = workspace.selection
    job.contacts = contacts
    job.contact_offset = batch.contact_offset
    job.freeze_frame_vjp = int(freeze_frame_vjp)
    job.stop_frame_vjp = int(stop_normal_gradient(batch.target.target.kind, model.contact_topk))
    if batch.has_target_plane:
        wp.launch(_narrow_plane, dim=(PLANE_CONTACTS, state.nworld),
                  inputs=[model, job], device=state.xpos.device)
    if batch.source_target_count:
        wp.launch(_narrow_pairs, dim=(batch.source_target_count, state.nworld),
                  inputs=[model, job], device=state.xpos.device)


def narrowphase(compiled: CompiledGaussianCollision, workspace: GaussianCollisionWorkspace, *,
                freeze_frame_vjp: bool = False) -> FixedContacts:
    for batch, batch_workspace in zip(compiled.batches, workspace.batches, strict=True):
        _narrowphase_batch(batch, batch_workspace, workspace.contacts,
                           freeze_frame_vjp=freeze_frame_vjp)
    return workspace.contacts


def _continuous(batch):
    return batch.model.contact_topk > 1 and batch.target.target.kind == targets.SPHERE_TARGET


def _broadphase_batch(batch, frames, workspace):
    if _continuous(batch):
        from .continuous_query import broad
        return broad(batch, frames, workspace)
    return _broadphase_discrete(batch, frames, workspace)


def _narrowphase_batch(batch, workspace, contacts, *, freeze_frame_vjp):
    if _continuous(batch):
        from .continuous_query import narrow
        return narrow(batch, workspace, contacts, freeze_frame_vjp=freeze_frame_vjp)
    return _narrowphase_discrete(batch, workspace, contacts, freeze_frame_vjp=freeze_frame_vjp)
