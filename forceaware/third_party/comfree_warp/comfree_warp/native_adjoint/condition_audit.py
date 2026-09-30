"""GPU reductions for soft-frame conditioning, read only outside CUDA graphs."""
import math
import warp as wp
from . import continuous_contact as R


@wp.kernel(enable_backward=False)
def _reduce(buffers: wp.array(dtype=R.Support), minima: wp.array(dtype=float),
            counts: wp.array(dtype=int)):
    entry, world, slot = wp.tid()
    buf = buffers[entry]
    if world >= buf.count.shape[0] or slot >= buf.count.shape[1]:
        return
    if buf.count[world, slot] <= 0:
        return
    wp.atomic_add(counts, 0, 1)
    v = buf.normal[world, slot]
    length = wp.length(v)
    tangent = wp.length(wp.cross(R.F.unit(v), wp.vec3(1.0)))
    point_length = wp.length(buf.point[world, slot])
    if not wp.isfinite(length) or not wp.isfinite(tangent) or not wp.isfinite(point_length):
        wp.atomic_add(counts, 1, 1)
    else:
        wp.atomic_min(minima, 0, length)
        wp.atomic_min(minima, 1, tangent)
        if length < 1.0e-6 or tangent < 1.0e-6:
            wp.atomic_add(counts, 1, 1)


def check(entries):
    """Reject invalid frames; report small valid norms without changing gradients.

    The temporary descriptor array is not cached: caching it would keep the
    workspace arrays alive after their owning native workspace was released.
    """
    if not entries:
        return {'checked_slots': 0, 'degenerate_slots': 0,
                'minimum_blended_normal_norm': None, 'minimum_tangent_norm': None}
    groups = {}
    for entry in entries:
        groups.setdefault(str(entry.normal.device), []).append(entry)
    if len(groups) > 1:
        reports = [check(group) for group in groups.values()]
        result = {key: sum(r[key] for r in reports)
                  for key in ('checked_slots', 'degenerate_slots')}
        for key in ('minimum_blended_normal_norm', 'minimum_tangent_norm'):
            values = [r[key] for r in reports if r[key] is not None]
            result[key] = min(values) if values else None
        return result
    device = entries[0].normal.device
    if device.is_capturing:
        raise RuntimeError('Condition audit must run outside CUDA graph capture')
    worlds = max(entry.count.shape[0] for entry in entries)
    slots = max(entry.count.shape[1] for entry in entries)
    descriptors = wp.array(entries, dtype=R.Support, device=device)
    minima = wp.full(2, float('inf'), dtype=float, device=device)
    counts = wp.zeros(2, dtype=int, device=device)
    wp.launch(_reduce, dim=(len(entries), worlds, slots), inputs=[descriptors, minima, counts], device=device)
    minimum, count = minima.numpy(), counts.numpy()
    result = {'checked_slots': int(count[0]), 'degenerate_slots': int(count[1]),
              'minimum_blended_normal_norm': float(minimum[0]) if count[0] else None,
              'minimum_tangent_norm': float(minimum[1]) if count[0] else None}
    if count[1] or (count[0] and not all(math.isfinite(float(x)) for x in minimum)):
        raise RuntimeError({'invalid_soft_contact_frame': result})
    return result
