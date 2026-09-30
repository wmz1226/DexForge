"""Gaussian-source adapter to the simulator's BVH, contact fusion and VJP.

This module assembles batches and selects output seeds. It contains no independent
collision-distance, normal, fusion-weight, or derivative formula.
"""
from collections import OrderedDict
from dataclasses import dataclass
from comfree_warp.collision_targets import SPHERE_TARGET
from . import continuous_query
import numpy as np
import warp as wp
from comfree_warp import object_query, contact_fusion as fusion
from comfree_warp.collision_config import stop_normal_gradient

F = fusion.precise
MAX_QUERY_POINTS = 262144
MAX_CACHED_BATCHES = 8
MAX_CACHED_BYTES = 512 * 1024 * 1024
MAX_WARM_START_BYTES = 64 * 1024 * 1024


def _input_chunks(points, fixed=None, source_radii=None, warm_ids=None, max_points=MAX_QUERY_POINTS):
    radii = None if source_radii is None else np.broadcast_to(source_radii, (len(points),))
    for start in range(0, len(points), max_points):
        selected = slice(start, start + max_points)
        yield (points[selected], None if fixed is None else tuple(a[selected] for a in fixed),
               None if radii is None else radii[selected],
               None if warm_ids is None else warm_ids[selected])


def _buffer_bytes(batch):
    arrays = [a for a in batch.values() if hasattr(a, 'capacity')]
    arrays += [a for values in batch['jacobians'].values() for a in values if hasattr(a, 'capacity')]
    support = batch.get('smooth_support')
    if support is not None:
        arrays += [getattr(support, name) for name in ('ids', 'distance', 'count', 'tau', 'normal')]
    return sum(a.capacity for a in arrays)


@wp.func
def _candidates(point: wp.vec3d, source_radius: wp.float64, anchors: wp.array2d(dtype=wp.vec4),
                axes: wp.array2d(dtype=wp.vec4), index: int, threshold: wp.float64, offset: wp.float64):
    source = F.mat_ranks3()
    anchor = F.mat_ranks3()
    axis = F.mat_ranks3()
    radius = F.vec_ranks()
    source_radii = F.vec_ranks()
    codes = fusion.ivec_ranks()
    for rank in range(fusion.MAX_RANKS):
        if rank < anchors.shape[1]:
            a = anchors[index, rank]
            x = axes[index, rank]
            source[rank] = point
            anchor[rank] = wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))
            axis[rank] = wp.vec3d(wp.float64(x[0]), wp.float64(x[1]), wp.float64(x[2]))
            radius[rank] = wp.float64(a[3])
            source_radii[rank] = source_radius
            codes[rank] = int(x[3])
    c = F.Candidates()
    c.source = source
    c.anchor = anchor
    c.axis = axis
    c.radius = radius
    c.code = codes
    c.source_radius = source_radii
    # Match simulation's active-candidate cutoff and missing-boundary fallback.
    # Keep the nearest feature outside the active range for geometric queries.
    for rank in range(1, fusion.MAX_RANKS):
        if codes[rank] != 0:
            distance, unused_p, unused_n = F.candidate_contact(c, rank, offset)
            if distance >= threshold:
                codes[rank] = 0
    c.code = codes
    return c


@wp.kernel(enable_backward=False)
def _values(points: wp.array(dtype=wp.vec3d), source_radii: wp.array(dtype=wp.float64), anchors: wp.array2d(dtype=wp.vec4),
            axes: wp.array2d(dtype=wp.vec4), topk: int, fallback: wp.float64,
            offset: wp.float64, output: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    c = _candidates(points[i], source_radii[i], anchors, axes, i, fallback, offset)
    count, distance, position, inward = F.fuse(c, topk, fallback, offset)
    output[i, 0] = distance
    for j in range(3):
        output[i, j+1] = position[j]
        output[i, j+4] = -inward[j]


@wp.kernel(enable_backward=False)
def _distances(points: wp.array(dtype=wp.vec3d), source_radii: wp.array(dtype=wp.float64), anchors: wp.array2d(dtype=wp.vec4),
               axes: wp.array2d(dtype=wp.vec4), topk: int, fallback: wp.float64,
               offset: wp.float64, output: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    c = _candidates(points[i], source_radii[i], anchors, axes, i, fallback, offset)
    count, distance, position, inward = F.fuse(c, topk, fallback, offset)
    sources, unused_a, unused_x = F.fuse_vjp(c, topk, fallback, offset, wp.float64(-1.),
        0, wp.float64(1.), wp.vec3d(0.), wp.mat33d(0.))
    gradient = wp.vec3d(0.)
    for rank in range(fusion.MAX_RANKS):
        gradient += sources[rank]
    lower, p0, n0 = F.candidate_contact(c, 0, offset)
    output[i, 0] = distance
    for j in range(3):
        output[i, j+1] = gradient[j]
    output[i, 4] = lower


@wp.kernel(enable_backward=False)
def _bounds(points: wp.array(dtype=wp.vec3d), source_radii: wp.array(dtype=wp.float64), anchors: wp.array2d(dtype=wp.vec4),
            axes: wp.array2d(dtype=wp.vec4), topk: int, fallback: wp.float64,
            offset: wp.float64, output: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    c = _candidates(points[i], source_radii[i], anchors, axes, i, fallback, offset)
    count, distance, position, inward = F.fuse(c, topk, fallback, offset)
    lower, p0, n0 = F.candidate_contact(c, 0, offset)
    output[i, 0] = distance
    output[i, 1] = lower


@wp.kernel(enable_backward=False)
def _jacobians(points: wp.array(dtype=wp.vec3d), source_radii: wp.array(dtype=wp.float64), anchors: wp.array2d(dtype=wp.vec4),
               axes: wp.array2d(dtype=wp.vec4), topk: int, fallback: wp.float64,
               offset: wp.float64, components: wp.array(dtype=int), stop_normal: int,
               output: wp.array2d(dtype=wp.vec3d)):
    i, column = wp.tid()
    component = components[column]
    result = wp.vec3d(0.)
    if component < 4 or stop_normal == 0:
        c = _candidates(points[i], source_radii[i], anchors, axes, i, fallback, offset)
        gd = wp.float64(0.)
        gp = wp.vec3d(0.)
        gf = wp.mat33d(0.)
        if component == 0:
            gd = wp.float64(1.)
        elif component < 4:
            gp[component-1] = wp.float64(1.)
        else:
            gf[0, component-4] = wp.float64(1.)
        sources, unused_a, unused_x = F.fuse_vjp(c, topk, fallback, offset,
            wp.float64(-1.), 0, gd, gp, gf)
        for rank in range(fusion.MAX_RANKS):
            result += sources[rank]
    output[i, column] = result


@dataclass(frozen=True)
class DenseRows:
    rows: np.ndarray

    def split(self, boundaries):
        return tuple(type(self)(a) for a in np.split(self.rows, boundaries))

    def __getitem__(self, indices):
        return self.rows[indices]


@dataclass(frozen=True)
class QueryRows:
    query: object
    points: np.ndarray
    radii: object = None

    def split(self, boundaries):
        points = np.split(self.points, boundaries)
        radii = [None]*len(points) if self.radii is None else np.split(self.radii, boundaries)
        return tuple(type(self)(self.query, p, r) for p, r in zip(points, radii))

    def __getitem__(self, indices):
        points, components = indices if isinstance(indices, tuple) else (indices, slice(None))
        selected = np.asarray(self.points[points])
        scalar_point = selected.ndim == 1
        columns = np.asarray(np.arange(7)[components])
        result = self.query.jacobians(np.atleast_2d(selected), np.atleast_1d(columns),
            source_radii=None if self.radii is None else np.atleast_1d(self.radii[points]))
        if columns.ndim == 0:
            result = result[:, 0]
        return result[0] if scalar_point else result


class SourceQuery:
    """Cached source centres/radii against the target compiled for simulation.

    Zero-radius sources use exactly the same kernels and contact functions.
    """
    def __init__(self, target, config, threshold, device='cuda'):
        self.target, self.config, self.threshold, self.device = target, config, float(threshold), device
        self.continuous = target.kind == SPHERE_TARGET and config.contact_topk > 1
        self.max_points = min(8192, MAX_QUERY_POINTS) if self.continuous else MAX_QUERY_POINTS
        self.topk = config.contact_topk
        self.ranks = 1 if self.topk == 1 else self.topk+1
        self.stop_normal = int(stop_normal_gradient(target.kind, self.topk))
        self._batches = OrderedDict()
        self._warm_starts = OrderedDict()

    def _warm_ids(self, count):
        """Keep compact search seeds by full batch size, independently of device buffers."""
        if count in self._warm_starts:
            self._warm_starts.move_to_end(count)
            return self._warm_starts[count]
        size = count * self.ranks * np.dtype(np.int32).itemsize
        if size > MAX_WARM_START_BYTES:
            return None
        while self._warm_starts and (
                len(self._warm_starts) >= MAX_CACHED_BATCHES or
                sum(a.nbytes for a in self._warm_starts.values()) + size > MAX_WARM_START_BYTES):
            self._warm_starts.popitem(last=False)
        ids = np.full((count, self.ranks), -1, dtype=np.int32)
        self._warm_starts[count] = ids
        return ids

    def _trim_batches(self):
        # Always retain the active batch, even if a caller requests an unusually
        # large set of Jacobian columns. Inactive allocations obey the budget.
        while len(self._batches) > 1 and sum(map(_buffer_bytes, self._batches.values())) > MAX_CACHED_BYTES:
            self._batches.popitem(last=False)

    def _batch(self, count, ranks, topk):
        key = count, ranks, topk
        if key in self._batches:
            self._batches.move_to_end(key)
            return self._batches[key]
        # Previous query results have been synchronized and copied to NumPy.
        # Lazy derivative handles own their points, so eviction is safe.
        while len(self._batches) >= MAX_CACHED_BATCHES:
            self._batches.popitem(last=False)
        b = {}
        for name, shape, dtype in [('points', count, wp.vec3d), ('search_points', count, wp.vec3),
                ('radii', count, wp.float64),
                ('anchors', (count, ranks), wp.vec4), ('axes', (count, ranks), wp.vec4),
                ('ids', (count, ranks), int), ('values', (count, 7), wp.float64),
                ('distances', (count, 5), wp.float64), ('bounds', (count, 2), wp.float64)]:
            b[name] = wp.empty(shape, dtype=dtype, device=self.device)
            b['host_'+name] = wp.empty(shape, dtype=dtype, device='cpu', pinned=True)
        b['ids'].fill_(-1)
        b['search'] = wp.launch(object_query.query_points, count,
            inputs=[self.target, b['search_points'], ranks],
            outputs=[b['anchors'], b['axes'], b['ids']], device=self.device, record_cmd=True)
        arguments = [b['points'], b['radii'], b['anchors'], b['axes'], topk,
                     wp.float64(self.threshold), wp.float64(self.config.distance_offset)]
        b['arguments'], b['jacobians'] = arguments, {}
        if self.continuous and topk > 1:
            continuous_query.allocate(b, self.target, self.config, count, self.device)
        else:
            for name, kernel in [('values', _values), ('distances', _distances), ('bounds', _bounds)]:
                b[name+'_launch'] = wp.launch(kernel, count, inputs=arguments,
                    outputs=[b[name]], device=self.device, record_cmd=True)
        self._batches[key] = b
        self._trim_batches()
        return b

    def _upload(self, points, fixed=None, source_radii=None, warm_ids=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError('Collision query points must be finite')
        ranks, topk = (self.ranks, self.topk) if fixed is None else (fixed[0].shape[1], 1)
        b = self._batch(len(points), ranks, topk)
        b['host_points'].numpy()[:] = points
        wp.copy(b['points'], b['host_points'])
        radii = np.broadcast_to(0. if source_radii is None else source_radii, (len(points),))
        if not np.isfinite(radii).all() or np.any(radii < 0.):
            raise ValueError('Source radii must be finite and nonnegative')
        b['host_radii'].numpy()[:] = radii
        wp.copy(b['radii'], b['host_radii'])
        if fixed is None:
            if warm_ids is not None:
                b['host_ids'].numpy()[:] = warm_ids
                wp.copy(b['ids'], b['host_ids'])
            b['host_search_points'].numpy()[:] = points
            wp.copy(b['search_points'], b['host_search_points'])
            b['search'].launch()
        else:
            for name, value in zip(['anchors', 'axes'], fixed):
                b['host_'+name].numpy()[:] = value
                wp.copy(b[name], b['host_'+name])
        if self.continuous and fixed is None:
            b['smooth_gather'].launch()
            b['smooth_values'].launch()
        return b

    def _synchronize(self, b, warm_ids=None):
        if warm_ids is not None:
            wp.copy(b['host_ids'], b['ids'])
        wp.synchronize_device(self.device)
        if warm_ids is not None:
            np.copyto(warm_ids, b['host_ids'].numpy())

    def _result(self, b, name, warm_ids=None):
        wp.copy(b['host_'+name], b[name])
        self._synchronize(b, warm_ids)
        result = b['host_'+name].numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Invalid collision query: incomplete soft support or degenerate normal')
        return result

    def evaluate(self, points, *, distance=False, bounds=False, fixed=None, source_radii=None,
                 _warm_ids=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) > self.max_points:
            seeds = self._warm_ids(len(points)) if fixed is None else None
            return np.concatenate([self.evaluate(p, distance=distance, bounds=bounds,
                fixed=f, source_radii=r, _warm_ids=ids)
                for p, f, r, ids in _input_chunks(points, fixed, source_radii, seeds, self.max_points)])
        name = 'distances' if distance else ('bounds' if bounds else 'values')
        if not len(points):
            return np.empty((0, {'distances': 5, 'bounds': 2, 'values': 7}[name]))
        b = self._upload(points, fixed, source_radii, _warm_ids)
        if not self.continuous or fixed is not None:
            b[name+'_launch'].launch()
        return self._result(b, name, _warm_ids)

    def candidates(self, points, *, _warm_ids=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points):
            return (np.empty((0, self.ranks, 4), np.float32),
                    np.empty((0, self.ranks, 4), np.float32),
                    np.empty((0, self.ranks), np.int32))
        if len(points) > self.max_points:
            chunks = [self.candidates(p, _warm_ids=ids)
                for p, _, _, ids in _input_chunks(points, warm_ids=self._warm_ids(len(points)), max_points=self.max_points)]
            return tuple(np.concatenate([c[i] for c in chunks]) for i in range(3))
        b = self._upload(points, warm_ids=_warm_ids)
        for name in ['anchors', 'axes', 'ids']:
            wp.copy(b['host_'+name], b[name])
        self._synchronize(b)
        if _warm_ids is not None:
            np.copyto(_warm_ids, b['host_ids'].numpy())
        return tuple(b['host_'+name].numpy().copy() for name in ['anchors', 'axes', 'ids'])

    def jacobians(self, points, components, source_radii=None, *, _warm_ids=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        components = tuple(int(c) for c in components)
        if any(c < 0 or c >= 7 for c in components):
            raise ValueError('Collision derivative components must be in [0, 6]')
        if not len(points) or not components:
            return np.empty((len(points), len(components), 3))
        if len(points) > self.max_points:
            return np.concatenate([self.jacobians(p, components, source_radii=r, _warm_ids=ids)
                for p, _, r, ids in _input_chunks(points, source_radii=source_radii,
                                                 warm_ids=self._warm_ids(len(points)), max_points=self.max_points)])
        b = self._upload(points, source_radii=source_radii, warm_ids=_warm_ids)
        if components not in b['jacobians']:
            ids = wp.array(components, dtype=int, device=self.device)
            out = wp.empty((len(points), len(components)), dtype=wp.vec3d, device=self.device)
            host = wp.empty(out.shape, dtype=out.dtype, device='cpu', pinned=True)
            kernel = continuous_query.jacobians if self.continuous else _jacobians
            arguments = ([*b['smooth_arguments'], ids] if self.continuous else
                         [*b['arguments'], ids, self.stop_normal])
            launch = wp.launch(kernel, out.shape, inputs=arguments, outputs=[out],
                               device=self.device, record_cmd=True)
            b['jacobians'][components] = ids, out, host, launch
            self._trim_batches()
        ids, out, host, launch = b['jacobians'][components]
        launch.launch()
        wp.copy(host, out)
        self._synchronize(b, _warm_ids)
        result = host.numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite collision query derivative')
        return result
