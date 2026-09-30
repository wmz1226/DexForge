"""Warm complete-support query with masked native nearest-pair fallback."""

import warp as wp
from . import continuous_contact as R

C = R.C
Support = R.Support
from .workspace_cache import WorkspaceCache

COLD = WorkspaceCache()
QUERY_MODELS = WorkspaceCache()


@wp.kernel(enable_backward=False)
def warm_seed(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    b: Support,
    cold: wp.array2d(dtype=int),
):
    s, w = wp.tid()
    n = wp.min(b.count[w, s], model.contact_topk)
    cold[w, s] = 1
    b.expand[w, s] = 0
    if n > 0:
        min_distance = float(1.0e30)
        for j in range(n):
            d, p, nn, src, anc, r = R.pair(
                model, job, b.source[w, s, j], b.target[w, s, j], w
            )
            b.distance[w, s, j] = d
            min_distance = wp.min(min_distance, d)
        # A subset root bounds the complete root from above. Re-solving the
        # previous active subset is much tighter than max motion across all
        # old pairs when different hand bodies move in different directions.
        tau = R.threshold(b, w, s, n)
        if min_distance < model.threshold:
            b.upper[w, s] = float(tau) + 2.0e-7
            b.seed_count[w, s] = wp.min(n, model.contact_topk)
            b.count[w, s] = b.seed_count[w, s]
            b.expand[w, s] = 1
            cold[w, s] = 0
            wp.atomic_add(b.stats, 2, 1)
    if cold[w, s] == 1:
        b.count[w, s] = 0
        b.seed_count[w, s] = 0


@wp.kernel(enable_backward=False)
def mark_cold(b: Support, cold: wp.array2d(dtype=int)):
    s, w = wp.tid()
    if b.count[w, s] > b.source.shape[2]:
        cold[w, s] = 1
    b.expand[w, s] = 0


@wp.kernel(enable_backward=False)
def cold_seed(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    b: Support,
    cold: wp.array2d(dtype=int),
):
    s, w = wp.tid()
    if cold[w, s] == 0:
        return
    count = int(0)
    for j in range(model.contact_topk):
        sid = job.selected.candidate_source[w, s, j]
        tid = job.selected.candidate_target[w, s, j]
        if sid >= 0 and tid >= 0:
            d, p, n, src, anc, r = R.pair(model, job, sid, tid, w)
            b.source[w, s, count] = sid
            b.target[w, s, count] = tid
            b.distance[w, s, count] = d
            count += 1
    b.count[w, s] = count
    b.seed_count[w, s] = count
    b.expand[w, s] = 0
    if count > 0:
        wp.atomic_add(b.stats, 2, 1)
        tau = R.threshold(b, w, s, count)
        b.upper[w, s] = float(tau) + 2.0e-7
        guard = model.threshold
        sid = job.selected.candidate_source[w, s, model.contact_topk]
        tid = job.selected.candidate_target[w, s, model.contact_topk]
        if sid >= 0 and tid >= 0:
            guard, p, n, src, anc, r = R.pair(model, job, sid, tid, w)
        if float(tau) + 2.0e-7 >= guard:
            b.expand[w, s] = 1
            wp.atomic_add(b.stats, 1, 1)


def cold_flags(ws, b):
    return COLD.get_or_create(
        ws, lambda: wp.zeros(b.count.shape, dtype=int, device=ws.state.xpos.device)
    )


def make_job(batch, ws, contacts=None, freeze=False):
    job = C.CollisionJob()
    job.state = ws.state
    job.selected = ws.selection
    job.contact_offset = batch.contact_offset
    if contacts is not None:
        job.contacts = contacts
    job.freeze_frame_vjp = int(freeze)
    job.stop_frame_vjp = 0
    return job


def query_model(batch):
    """Share geometry arrays while giving broadphase a conservative soft cutoff.

    For D=min_w <w,d> + 2h/3*(sum(w**1.5)-1), D >= min(d)-2h/3.
    Thus a slot can be active only if some pair is below threshold+2h/3.
    This cutoff only prunes queries; fuse always uses the physical threshold.
    """

    def create():
        result = C.search.GaussianCollisionModel()
        for name in C.search.GaussianCollisionModel.vars:
            setattr(result, name, getattr(batch.model, name))
        return result

    result = QUERY_MODELS.get_or_create(batch, create)
    result.contact_topk = batch.model.contact_topk
    result.threshold = float(batch.model.threshold) + 2.0 * float(R.H) / 3.0 + 2.0e-7
    return result


def broad(batch, frames, ws):
    """GS soft broadphase; gaussian_collision owns mode dispatch."""
    C._bind_frames(ws, frames)
    model = query_model(batch)
    state = ws.state
    slots = ws.slots
    target = batch.target.target
    if batch.source_target_count:
        b = R.buffers(ws, batch.source_target_count, batch)
        cold = cold_flags(ws, b)
        job = make_job(batch, ws)
        wp.launch(
            warm_seed,
            dim=b.count.shape[::-1],
            inputs=[model, job, b, cold],
            record_tape=False,
        )
        wp.launch(
            R.gather,
            dim=(model.source_centers.shape[0], state.nworld),
            inputs=[model, job, target, batch.source_slot, b],
            record_tape=False,
        )
        wp.launch(
            mark_cold, dim=b.count.shape[::-1], inputs=[b, cold], record_tape=False
        )
        slots.count.zero_()
        wp.launch_tiled(
            C._slot_bounds,
            dim=(batch.source_target_count, state.nworld),
            inputs=[
                model,
                state,
                target,
                ws.selection.candidate_source,
                ws.selection.candidate_target,
                slots.bounds,
                cold,
            ],
            block_dim=C.SELECT_BLOCK_DIM,
            record_tape=False,
        )
        wp.launch(
            C._source_nearest,
            dim=(model.source_centers.shape[0], state.nworld),
            inputs=[
                model,
                state,
                target,
                batch.source_slot,
                slots.bounds,
                slots.distance,
                slots.count,
                slots.items,
                cold,
            ],
            record_tape=False,
        )
        wp.launch_tiled(
            C._select_slots,
            dim=(batch.source_target_count, state.nworld),
            inputs=[
                model,
                state,
                target,
                C.candidate_ranks(model),
                slots.distance,
                slots.count,
                slots.items,
                slots.bounds,
                ws.selection,
                cold,
            ],
            block_dim=C.SELECT_BLOCK_DIM,
            record_tape=False,
        )
        wp.launch(
            cold_seed,
            dim=b.count.shape[::-1],
            inputs=[model, job, b, cold],
            record_tape=False,
        )
        wp.launch(
            R.gather,
            dim=(model.source_centers.shape[0], state.nworld),
            inputs=[model, job, target, batch.source_slot, b],
            record_tape=False,
        )
        for _ in range(3):
            wp.launch(
                R.refine_overflow,
                dim=b.count.shape[::-1],
                inputs=[b],
                record_tape=False,
            )
            wp.launch(
                R.gather,
                dim=(model.source_centers.shape[0], state.nworld),
                inputs=[model, job, target, batch.source_slot, b],
                record_tape=False,
            )
        wp.launch_tiled(
            R.sort_small,
            dim=b.count.shape,
            inputs=[b, model.target_spheres.shape[0]],
            block_dim=64,
            record_tape=False,
        )
        wp.launch_tiled(
            R.sort_support,
            dim=b.count.shape,
            inputs=[b, model.target_spheres.shape[0]],
            block_dim=R.CAP,
            record_tape=False,
        )
    if batch.has_target_plane:
        wp.launch_tiled(
            C.bvh.select_plane,
            dim=state.nworld,
            inputs=[model, state, ws.selection],
            block_dim=C.search.PLANE_BLOCK_DIM,
            record_tape=False,
        )


def narrow(batch, ws, contacts, *, freeze_frame_vjp):
    """GS soft narrowphase; gaussian_collision owns mode dispatch."""
    job = make_job(batch, ws, contacts, freeze_frame_vjp)
    if batch.has_target_plane:
        wp.launch(
            C._narrow_plane,
            dim=(C.PLANE_CONTACTS, ws.state.nworld),
            inputs=[batch.model, job],
        )
    if batch.source_target_count:
        b = R.buffers(ws, batch.source_target_count, batch)
        wp.launch(R.fuse, dim=b.count.shape[::-1], inputs=[batch.model, job, b])

