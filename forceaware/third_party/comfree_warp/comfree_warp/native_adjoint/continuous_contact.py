"""Complete sparse GS smooth-min support and consistent contact derivatives.

Reuses the native nearest-pair BVH. Extra range queries run only on slots whose
positive support is not covered by those pairs. Hard and mesh use native code.
The smooth minimum is min_w <w,d> + 2h/3 (sum(w**1.5)-1), w>=0, sum(w)=1.
"""

import math
from .workspace_cache import WorkspaceCache
import warp as wp
from comfree_warp import contact_fusion as F, collision_targets as T
from comfree_warp.native_adjoint import (
    gaussian_collision as C,
)

from comfree_warp.smooth_contact import WIDTH as H, CAPACITY as CAP
from comfree_warp import smooth_contact as smooth

CACHE = WorkspaceCache()
BODY_LAYOUT = WorkspaceCache()


@wp.struct
class Support:
    source: wp.array3d(dtype=int)
    target: wp.array3d(dtype=int)
    distance: wp.array3d(dtype=float)
    count: wp.array2d(dtype=int)
    expand: wp.array2d(dtype=int)
    seed_count: wp.array2d(dtype=int)
    upper: wp.array2d(dtype=float)
    tau: wp.array2d(dtype=wp.float64)
    point: wp.array2d(dtype=wp.vec3)
    normal: wp.array2d(dtype=wp.vec3)
    gs: wp.array3d(dtype=wp.vec3)
    ga: wp.array3d(dtype=wp.vec3)
    # overflow, full-support expansions, queried slots, maximum support count
    stats: wp.array(dtype=int)
    residual: wp.array2d(dtype=float)
    body_slot_starts: wp.array(dtype=int)
    body_slot_ids: wp.array(dtype=int)


@wp.func
def pair(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    sid: int,
    tid: int,
    w: int,
):
    tb = model.target_body_id
    sb = model.source_body_ids[sid]
    sphere = model.target_spheres[tid]
    source = job.state.xpos[w, sb] + job.state.xmat[w, sb] @ model.source_centers[sid]
    anchor = job.state.xpos[w, tb] + job.state.xmat[w, tb] @ wp.vec3(
        sphere[0], sphere[1], sphere[2]
    )
    off = C.search._distance_offset(model, w)
    d, p, n = F.sphere_contact(source, anchor, model.source_radii[sid], sphere[3], off)
    return d, p, n, source, anchor, sphere[3]


@wp.func
def threshold(buf: Support, w: int, s: int, count: int):
    return smooth.float32.threshold(buf.distance, w, s, count)


@wp.kernel(enable_backward=False)
def gather(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    target: T.CollisionTarget,
    source_slot: wp.array(dtype=int),
    buf: Support,
):
    sid, w = wp.tid()
    s = source_slot[sid]
    if s < 0:
        return
    if buf.expand[w, s] == 0:
        return
    local = C._source_local(model, job.state, w, sid)
    reach = (
        buf.upper[w, s]
        + model.source_radii[sid]
        - C.search._distance_offset(model, w)
        + 2.0e-7
    )
    if T.outside_bounds(target, local, reach):
        return
    node = int(0)
    if buf.count[w, s] > buf.source.shape[2]:
        return
    full = bool(False)
    while node < target.bvh_ranges.shape[0]:
        if T._sphere_bound(target, local, node) > reach:
            node = target.bvh_escape[node]
        elif target.bvh_ranges[node][1] > 0:
            span = target.bvh_ranges[node]
            for off in range(span[1]):
                tid = target.bvh_ids[span[0] + off]
                sphere = target.spheres[tid]
                if (
                    wp.length(local - wp.vec3(sphere[0], sphere[1], sphere[2]))
                    - sphere[3]
                    < reach
                ):
                    d, p, n, src, anc, r = pair(model, job, sid, tid, w)
                    duplicate = bool(False)
                    for j0 in range(buf.seed_count[w, s]):
                        if buf.source[w, s, j0] == sid and buf.target[w, s, j0] == tid:
                            duplicate = True
                    if d < buf.upper[w, s] and not duplicate:
                        j = wp.atomic_add(buf.count, w, s, 1)
                        if j < buf.source.shape[2]:
                            buf.source[w, s, j] = sid
                            buf.target[w, s, j] = tid
                            buf.distance[w, s, j] = d
                        else:
                            wp.atomic_add(buf.stats, 5, 1)
                            full = True
                            break
            if full:
                break
            node = target.bvh_escape[node]
        else:
            node += 1


@wp.kernel(enable_backward=False)
def refine_overflow(buf: Support):
    s, w = wp.tid()
    if buf.count[w, s] > buf.source.shape[2]:
        t = threshold(buf, w, s, buf.source.shape[2])
        buf.upper[w, s] = wp.min(buf.upper[w, s], float(t) + 2.0e-7)
        buf.count[w, s] = buf.seed_count[w, s]
        buf.expand[w, s] = 1
        wp.atomic_add(buf.stats, 6, 1)
    else:
        buf.expand[w, s] = 0


@wp.func_native("""
union { float f; uint32_t u; } bits;
bits.f = d == 0.0f ? 0.0f : d;
uint32_t ordered = bits.u ^ ((bits.u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
return (uint64_t(ordered) << 32) | uint32_t(identity);
""")
def order_key(d: float, identity: int) -> wp.uint64: ...


@wp.kernel(enable_backward=False)
def sort_small(buf: Support, target_count: int):
    w, s, lane = wp.tid()
    n = wp.min(buf.count[w, s], buf.source.shape[2])
    if n <= 1 or n > 64:
        return
    key = wp.uint64(0xFFFFFFFFFFFFFFFF)
    d = float(0.0)
    if lane < n:
        d = buf.distance[w, s, lane]
        key = order_key(
            d, buf.source[w, s, lane] * target_count + buf.target[w, s, lane]
        )
    keys = wp.tile(key)
    values = wp.tile(d)
    wp.tile_sort(keys, values)
    sorted_key = wp.tile_extract(keys, lane)
    sorted_d = wp.tile_extract(values, lane)
    if lane < n:
        identity = int(sorted_key & wp.uint64(0xFFFFFFFF))
        buf.source[w, s, lane] = identity // target_count
        buf.target[w, s, lane] = identity % target_count
        buf.distance[w, s, lane] = sorted_d


@wp.kernel(enable_backward=False)
def sort_support(buf: Support, target_count: int):
    w, s, lane = wp.tid()
    n = wp.min(buf.count[w, s], buf.source.shape[2])
    if lane == 0:
        wp.atomic_max(buf.stats, 3, buf.count[w, s])
        if buf.count[w, s] > buf.source.shape[2]:
            wp.atomic_add(buf.stats, 0, buf.count[w, s] - buf.source.shape[2])
    if n <= 64:
        return
    key = wp.uint64(0xFFFFFFFFFFFFFFFF)
    d = float(0.0)
    if lane < n:
        d = buf.distance[w, s, lane]
        key = order_key(
            d, buf.source[w, s, lane] * target_count + buf.target[w, s, lane]
        )
    keys = wp.tile(key)
    values = wp.tile(d)
    wp.tile_sort(keys, values)
    sorted_key = wp.tile_extract(keys, lane)
    sorted_d = wp.tile_extract(values, lane)
    if lane < n:
        identity = int(sorted_key & wp.uint64(0xFFFFFFFF))
        buf.source[w, s, lane] = identity // target_count
        buf.target[w, s, lane] = identity % target_count
        buf.distance[w, s, lane] = sorted_d


@wp.func
def sorted_threshold(buf: Support, w: int, s: int, count: int):
    return smooth.float32.sorted_threshold(buf.distance, w, s, count)


@wp.func
def invalidate_contact(job: C.CollisionJob, buf: Support, w: int, s: int, row: int):
    # A truncated support is not the requested field. Keep the error visible to
    # the rollout validity check; never present it as a successful contact.
    job.contacts.distance[w, row] = wp.nan
    job.contacts.position[w, row] = wp.vec3(wp.nan)
    job.contacts.frame[w, row] = wp.mat33(wp.nan)
    job.contacts.active[w, row] = 1
    buf.count[w, s] = 0
    buf.tau[w, s] = wp.float64(0.0)
    buf.point[w, s] = wp.vec3(0.0)
    buf.normal[w, s] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def fuse(model: C.search.GaussianCollisionModel, job: C.CollisionJob, buf: Support):
    s, w = wp.tid()
    n = wp.min(buf.count[w, s], buf.source.shape[2])
    row = model.target_contact_offset + s + job.contact_offset
    if buf.count[w, s] > buf.source.shape[2]:
        buf.residual[w, s] = wp.nan
        invalidate_contact(job, buf, w, s, row)
        return
    if n == 0:
        C._clear_contact(job.contacts, w, row)
        buf.tau[w, s] = wp.float64(0.0)
        buf.point[w, s] = wp.vec3(0.0)
        buf.normal[w, s] = wp.vec3(0.0)
        buf.residual[w, s] = 0.0
        return
    t = sorted_threshold(buf, w, s, n)
    buf.tau[w, s] = t
    p = wp.vec3(0.0)
    normal = wp.vec3(0.0)
    sum_a3 = wp.float64(0.0)
    sum_a2 = wp.float64(0.0)
    active_count = int(0)
    for j in range(n):
        if wp.float64(buf.distance[w, s, j]) >= t:
            break
        d, pp, nn, src, anc, r = pair(
            model, job, buf.source[w, s, j], buf.target[w, s, j], w
        )
        a = wp.max((t - wp.float64(d)) / wp.float64(H), wp.float64(0.0))
        weight = a * a
        if a > wp.float64(0.0):
            buf.source[w, s, active_count] = buf.source[w, s, j]
            buf.target[w, s, active_count] = buf.target[w, s, j]
            buf.distance[w, s, active_count] = d
            active_count += 1
        p += float(weight) * pp
        normal += float(weight) * nn
        sum_a3 += a * weight
        sum_a2 += weight
    buf.count[w, s] = active_count
    distance = float(smooth.fused_distance(t, sum_a3))
    buf.residual[w, s] = float(wp.abs(sum_a2 - wp.float64(1.0)))
    if not wp.isfinite(buf.residual[w, s]) or buf.residual[w, s] > 1.0e-6:
        wp.atomic_add(buf.stats, 4, 1)
        invalidate_contact(job, buf, w, s, row)
        return
    buf.point[w, s] = p
    buf.normal[w, s] = normal
    job.contacts.distance[w, row] = distance
    job.contacts.position[w, row] = p
    job.contacts.frame[w, row] = F.configured_frame(
        F.unit(normal) * model.contact_normal_sign[model.target_contact_offset + s],
        job.freeze_frame_vjp,
        job.stop_frame_vjp,
    )
    job.contacts.active[w, row] = wp.where(distance < model.threshold, 1, 0)


@wp.func
def unit_vjp(v: wp.vec3, g: wp.vec3):
    inv = 1.0 / wp.sqrt(wp.dot(v, v) + 1.0e-18)
    n = v * inv
    return (g - n * wp.dot(n, g)) * inv


@wp.func
def frame_vjp(v: wp.vec3, g: wp.mat33, freeze: int):
    n = F.unit(v)
    cross1 = wp.cross(n, wp.vec3(1.0))
    tx = F.unit(cross1)
    cross2 = wp.cross(n, tx)
    ty = F.unit(cross2)
    gn = wp.vec3(g[0, 0], g[0, 1], g[0, 2])
    gx = wp.vec3(g[1, 0], g[1, 1], g[1, 2])
    gy = wp.vec3(g[2, 0], g[2, 1], g[2, 2])
    if freeze == 1:
        rotation = wp.cross(tx, gx) + wp.cross(ty, gy)
        return unit_vjp(v, gn + wp.cross(rotation, n))
    gcross2 = unit_vjp(cross2, gy)
    gn += wp.cross(tx, gcross2)
    gx += wp.cross(gcross2, n)
    gcross1 = unit_vjp(cross1, gx)
    gn += wp.cross(wp.vec3(1.0), gcross1)
    return unit_vjp(v, gn)


@wp.kernel(enable_backward=False)
def pair_vjp(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    adj: C.CollisionJob,
    buf: Support,
):
    s, w = wp.tid()
    n = wp.min(buf.count[w, s], buf.source.shape[2])
    row = model.target_contact_offset + s + job.contact_offset
    gp = adj.contacts.position[w, row]
    gd = adj.contacts.distance[w, row]
    gf = adj.contacts.frame[w, row]
    sign = model.contact_normal_sign[model.target_contact_offset + s]
    if job.stop_frame_vjp:
        gf = wp.mat33(0.0)
    blended = buf.normal[w, s]
    normal = F.unit(blended)
    gn = sign * frame_vjp(normal * sign, gf, job.freeze_frame_vjp)
    gb = unit_vjp(blended, gn)
    t = buf.tau[w, s]
    sum_a = wp.float64(0.0)
    sum_spread = wp.float64(0.0)
    for j in range(n):
        d, p, nn, src, anc, r = pair(
            model, job, buf.source[w, s, j], buf.target[w, s, j], w
        )
        a = wp.max((t - wp.float64(d)) / wp.float64(H), wp.float64(0.0))
        spread = wp.dot(gp, p - buf.point[w, s]) + wp.dot(gb, nn - blended)
        sum_a += a
        sum_spread += a * wp.float64(spread)
    mean = wp.float64(0.0)
    if sum_a > wp.float64(0.0):
        mean = sum_spread / sum_a
    for j in range(n):
        d, p, nn, src, anc, r = pair(
            model, job, buf.source[w, s, j], buf.target[w, s, j], w
        )
        a = wp.max((t - wp.float64(d)) / wp.float64(H), wp.float64(0.0))
        weight = float(a * a)
        spread = wp.dot(gp, p - buf.point[w, s]) + wp.dot(gb, nn - blended)
        dd = gd * weight + float(
            smooth.weight_seed(a, mean, wp.float64(spread))
        )
        pp = gp * weight
        ng = gb * weight + (C.search._distance_offset(model, w) - r) * pp
        rel = anc - src
        length = wp.length(rel)
        dg = wp.vec3(0.0)
        if length > 0.0:
            dg = dd * rel / length
        dg += unit_vjp(rel, ng)
        buf.gs[w, s, j] = -dg
        buf.ga[w, s, j] = pp + dg


@wp.kernel(enable_backward=False)
def body_vjp(
    model: C.search.GaussianCollisionModel,
    job: C.CollisionJob,
    adj: C.CollisionJob,
    buf: Support,
):
    w, body = wp.tid()
    position = wp.vec3(0.0)
    matrix = wp.mat33(0.0)
    for index in range(buf.body_slot_starts[body], buf.body_slot_starts[body + 1]):
        s = buf.body_slot_ids[index]
        for j in range(wp.min(buf.count[w, s], buf.source.shape[2])):
            sid = buf.source[w, s, j]
            tid = buf.target[w, s, j]
            if model.source_body_ids[sid] == body:
                g = buf.gs[w, s, j]
                position += g
                matrix += wp.outer(g, model.source_centers[sid])
            if model.target_body_id == body:
                g = buf.ga[w, s, j]
                sphere = model.target_spheres[tid]
                position += g
                matrix += wp.outer(g, wp.vec3(sphere[0], sphere[1], sphere[2]))
    if adj.state.xpos:
        adj.state.xpos[w, body] += position
    if adj.state.xmat:
        adj.state.xmat[w, body] += matrix


def record_backward(tape, inputs, device):
    model, job, buf = inputs
    for name in (
        "source_centers",
        "source_radii",
        "target_spheres",
        "distance_offset",
        "contact_normal_sign",
    ):
        array = getattr(model, name)
        if array is not None and array.requires_grad:
            raise NotImplementedError(
                f"continuous GS contact currently differentiates body poses, not model parameter {name}"
            )
    adj = tape.get_adjoint(job)

    # Register state gradients so Tape.zero() clears them between backward calls.
    def backward():
        wp.launch(
            pair_vjp,
            dim=buf.count.shape[::-1],
            inputs=[model, job, adj, buf],
            device=device,
        )
        wp.launch(
            body_vjp,
            dim=job.state.xpos.shape,
            inputs=[model, job, adj, buf],
            device=device,
        )

    tape.launches.append(backward)



def body_layout(batch, bodies, device):
    def create():
        import numpy as np

        owners = batch.model.source_body_ids.numpy()
        slots = batch.source_slot.numpy()
        rows = [set() for _ in range(bodies)]
        for body, slot in zip(owners, slots):
            if slot >= 0:
                rows[int(body)].add(int(slot))
        rows[int(batch.model.target_body_id)].update(range(batch.source_target_count))
        offsets = [0]
        indices = []
        for row in rows:
            indices.extend(sorted(row))
            offsets.append(len(indices))
        return (
            wp.array(np.asarray(offsets, dtype=np.int32), device=device),
            wp.array(np.asarray(indices, dtype=np.int32), device=device),
        )

    return BODY_LAYOUT.get_or_create(batch, create)


def buffers(ws, slots, batch):
    if (
        int(batch.model.source_centers.shape[0])
        * int(batch.model.target_spheres.shape[0])
        > 0x7FFFFFFF
    ):
        raise ValueError("pair identity exceeds the deterministic sort key capacity")
    if ws.selection.candidate_source.shape[2] != C.candidate_ranks(batch.model):
        raise ValueError(
            "contact_topk changed after workspace allocation; recompile and reallocate"
        )

    def create():
        worlds = ws.state.nworld
        dev = ws.state.xpos.device
        shape = (worlds, slots, CAP)
        b = Support()
        for name, dtype in [
            ("source", int),
            ("target", int),
            ("distance", float),
            ("gs", wp.vec3),
            ("ga", wp.vec3),
        ]:
            setattr(b, name, wp.empty(shape, dtype=dtype, device=dev))
        for name, dtype in [
            ("count", int),
            ("expand", int),
            ("seed_count", int),
            ("upper", float),
            ("tau", wp.float64),
            ("point", wp.vec3),
            ("normal", wp.vec3),
            ("residual", float),
        ]:
            setattr(b, name, wp.zeros((worlds, slots), dtype=dtype, device=dev))
        b.stats = wp.zeros(7, dtype=int, device=dev)
        b.body_slot_starts, b.body_slot_ids = body_layout(
            batch, ws.state.xpos.shape[1], dev
        )
        return b

    b = CACHE.get_or_create(ws, create)
    if (
        b.count.shape != (ws.state.nworld, slots)
        or b.count.device != ws.state.xpos.device
    ):
        raise ValueError(
            "a collision workspace cannot change its world count, slots or device"
        )
    return b


def audit():
    import numpy as np

    entries = CACHE.values()
    stats = (
        np.array([x.stats.numpy() for x in entries])
        if entries
        else np.zeros((1, 7), dtype=int)
    )
    res = (
        float(np.max([float(x.residual.numpy().max()) for x in entries]))
        if entries
        else 0.0
    )
    result = {
        "overflow": int(stats[:, 0].sum()),
        "expanded_slots": int(stats[:, 1].sum()),
        "queried_slots": int(stats[:, 2].sum()),
        "max_pair_count": int(stats[:, 3].max()),
        "max_root_residual": res,
        "bad_root_queries": int(stats[:, 4].sum()),
        "initial_spills": int(stats[:, 5].sum()),
        "capacity_refinements": int(stats[:, 6].sum()),
        "width_m": float(H),
        "capacity": CAP,
    }
    if result["overflow"]:
        raise RuntimeError(result)
    if not math.isfinite(res) or res > 1.0e-6 or result["bad_root_queries"]:
        raise RuntimeError(result)
    from .condition_audit import check
    result["frame_condition"] = check(entries)
    return result


def clear_audit():
    for b in CACHE.values():
        b.stats.zero_()
