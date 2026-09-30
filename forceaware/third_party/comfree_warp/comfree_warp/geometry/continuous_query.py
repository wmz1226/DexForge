"""Float64 point/sphere queries of the simulator's complete-support GS field.

The native BVH supplies an initial subset. Its root is an upper bound on the
complete root; range traversal then includes every positive-weight sphere.
Only query storage and precision differ from the rigid-body contact path.
"""
import warp as wp

from comfree_warp import collision_targets as T, contact_fusion as F, smooth_contact as S


@wp.struct
class PointSupport:
    ids: wp.array3d(dtype=int)
    distance: wp.array3d(dtype=wp.float64)
    count: wp.array(dtype=int)
    tau: wp.array(dtype=wp.float64)
    normal: wp.array(dtype=wp.vec3d)


@wp.func
def pair(target: T.CollisionTarget, point: wp.vec3d, radius: wp.float64,
         offset: wp.float64, sid: int):
    sphere = target.spheres[sid]
    center = wp.vec3d(wp.float64(sphere[0]), wp.float64(sphere[1]), wp.float64(sphere[2]))
    d, p, n = F.precise.sphere_contact(point, center, radius, wp.float64(sphere[3]), offset)
    return d, p, n


@wp.kernel(enable_backward=False)
def gather(target: T.CollisionTarget, points: wp.array(dtype=wp.vec3d),
           radii: wp.array(dtype=wp.float64), seeds: wp.array2d(dtype=int),
           topk: int, offset: wp.float64, buf: PointSupport):
    i = wp.tid()
    p = points[i]
    seed_count = int(0)
    for rank in range(topk):
        sid = seeds[i, rank]
        if sid >= 0:
            d, unused_p, unused_n = pair(target, p, radii[i], offset, sid)
            buf.distance[0, i, seed_count] = d
            seed_count += 1
    buf.count[i] = -1
    if seed_count == 0:
        return
    upper = S.float64.threshold(buf.distance, 0, i, seed_count)
    # BVH arithmetic is float32, while the field and derivatives stay float64.
    query = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
    padding = 2.0e-7 + 2.0e-6 * wp.length(query)
    for attempt in range(4):
        count = int(0)
        reach = float(upper + radii[i] - offset) + padding
        node = int(0)
        while node < target.bvh_ranges.shape[0]:
            if T._sphere_bound(target, query, node) > reach:
                node = target.bvh_escape[node]
            elif target.bvh_ranges[node][1] > 0:
                span = target.bvh_ranges[node]
                for j in range(span[1]):
                    sid = target.bvh_ids[span[0] + j]
                    d, unused_p, unused_n = pair(target, p, radii[i], offset, sid)
                    if d <= upper + wp.float64(1.0e-12):
                        if count < S.CAPACITY:
                            buf.ids[0, i, count] = sid
                            buf.distance[0, i, count] = d
                        count += 1
                node = target.bvh_escape[node]
            else:
                node += 1
        if count > 0 and count <= S.CAPACITY:
            buf.count[i] = count
            buf.tau[i] = S.float64.threshold(buf.distance, 0, i, count)
            break
        if count > S.CAPACITY:
            upper = S.float64.threshold(buf.distance, 0, i, S.CAPACITY)


@wp.kernel(enable_backward=False)
def values(target: T.CollisionTarget, points: wp.array(dtype=wp.vec3d),
           radii: wp.array(dtype=wp.float64), offset: wp.float64, buf: PointSupport,
           out: wp.array2d(dtype=wp.float64), distances: wp.array2d(dtype=wp.float64),
           bounds: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    n = buf.count[i]
    tau = wp.float64(0.0)
    if n > 0:
        tau = buf.tau[i]
    pos = wp.vec3d(0.0)
    normal = wp.vec3d(0.0)
    gradient = wp.vec3d(0.0)
    cubic = wp.float64(0.0)
    mass = wp.float64(0.0)
    for j in range(n):
        sid = buf.ids[0, i, j]
        d, p, inward = pair(target, points[i], radii[i], offset, sid)
        a = S.activation(tau, d)
        weight = a*a
        pos += weight*p
        normal += weight*inward
        sphere = target.spheres[sid]
        delta = points[i] - wp.vec3d(wp.float64(sphere[0]), wp.float64(sphere[1]), wp.float64(sphere[2]))
        length = wp.length(delta)
        if length > wp.float64(0.0):
            gradient += weight*delta/length
        mass += weight
        cubic += weight*a
    distance = S.fused_distance(tau, cubic)
    outward = -F.precise.unit(normal)
    if n <= 0 or not wp.isfinite(mass) or wp.abs(mass-wp.float64(1.0)) > wp.float64(1.0e-9) or wp.length(normal) < wp.float64(1.0e-6):
        buf.count[i] = -1
        distance = wp.float64(wp.nan)
        pos = wp.vec3d(wp.float64(wp.nan))
        outward = wp.vec3d(wp.float64(wp.nan))
        gradient = wp.vec3d(wp.float64(wp.nan))
    buf.normal[i] = normal
    out[i, 0] = distance
    distances[i, 0] = distance
    # This is a scheduling lower bound, never hard-geometry acceptance.
    distances[i, 4] = distance
    bounds[i, 0] = distance
    bounds[i, 1] = distance
    for c in range(3):
        out[i, 1+c] = pos[c]
        out[i, 4+c] = outward[c]
        distances[i, 1+c] = gradient[c]


@wp.func
def unit_vjp(value: wp.vec3d, seed: wp.vec3d):
    inverse = wp.float64(1.0)/wp.sqrt(wp.dot(value, value)+wp.float64(1.0e-18))
    normal = value*inverse
    return (seed-normal*wp.dot(normal, seed))*inverse


@wp.kernel(enable_backward=False)
def jacobians(target: T.CollisionTarget, points: wp.array(dtype=wp.vec3d),
              radii: wp.array(dtype=wp.float64), offset: wp.float64, buf: PointSupport,
              components: wp.array(dtype=int), output: wp.array2d(dtype=wp.vec3d)):
    i, column = wp.tid()
    component = components[column]
    n = buf.count[i]
    gp = wp.vec3d(0.0)
    gn = wp.vec3d(0.0)
    gd = wp.float64(0.0)
    if component == 0:
        gd = wp.float64(1.0)
    elif component < 4:
        gp[component-1] = wp.float64(1.0)
    else:
        gn[component-4] = wp.float64(-1.0)
    gb = unit_vjp(buf.normal[i], gn)
    total = wp.float64(0.0)
    spread_sum = wp.float64(0.0)
    for j in range(n):
        d, p, inward = pair(target, points[i], radii[i], offset, buf.ids[0, i, j])
        a = S.activation(buf.tau[i], d)
        total += a
        spread_sum += a*(wp.dot(gp, p)+wp.dot(gb, inward))
    result = wp.vec3d(0.0)
    if total > wp.float64(0.0):
        mean = spread_sum/total
        for j in range(n):
            sid = buf.ids[0, i, j]
            d, p, inward = pair(target, points[i], radii[i], offset, sid)
            a = S.activation(buf.tau[i], d)
            dd = S.distance_seed(a, gd, mean, wp.dot(gp, p)+wp.dot(gb, inward))
            sphere = target.spheres[sid]
            delta = wp.vec3d(wp.float64(sphere[0]), wp.float64(sphere[1]), wp.float64(sphere[2])) - points[i]
            length = wp.length(delta)
            if length > wp.float64(0.0):
                result -= dd*delta/length
            result -= unit_vjp(delta, a*a*(gb+(offset-wp.float64(sphere[3]))*gp))
    if n <= 0:
        result = wp.vec3d(wp.float64(wp.nan))
    output[i, column] = result


def allocate(batch, target, config, count, device):
    buf = PointSupport()
    buf.ids = wp.empty((1, count, S.CAPACITY), dtype=int, device=device)
    buf.distance = wp.empty((1, count, S.CAPACITY), dtype=wp.float64, device=device)
    buf.count = wp.empty(count, dtype=int, device=device)
    buf.tau = wp.empty(count, dtype=wp.float64, device=device)
    buf.normal = wp.empty(count, dtype=wp.vec3d, device=device)
    arguments = [target, batch['points'], batch['radii'], wp.float64(config.distance_offset), buf]
    batch['smooth_support'] = buf
    batch['smooth_gather'] = wp.launch(gather, count, inputs=[target, batch['points'], batch['radii'],
        batch['ids'], config.contact_topk, wp.float64(config.distance_offset), buf], device=device, record_cmd=True)
    batch['smooth_values'] = wp.launch(values, count, inputs=arguments,
        outputs=[batch['values'], batch['distances'], batch['bounds']], device=device, record_cmd=True)
    batch['smooth_arguments'] = arguments
