"""Shared fixed-width GS geometry for point queries and simulation contacts.

Minimize <w,d> + 2h/3 (sum(w**1.5)-1), with w>=0 and sum(w)=1.
The positive weights are ((tau-d)/h)**2. Candidate counts only seed the
BVH search; every pair with positive weight belongs to the final support.
"""
from types import SimpleNamespace

import warp as wp

WIDTH = wp.constant(0.001)
CAPACITY = 512


@wp.func
def root_from_moments(lo: wp.float64, total: wp.float64,
                      squared: wp.float64, count: wp.float64):
    disc = wp.max(total * total - count * (squared - wp.float64(1.0)), wp.float64(0.0))
    return lo + wp.float64(WIDTH) * (total + wp.sqrt(disc)) / count


@wp.func
def activation(tau: wp.float64, distance: wp.float64):
    return wp.max((tau - distance) / wp.float64(WIDTH), wp.float64(0.0))


@wp.func
def fused_distance(tau: wp.float64, cubic_sum: wp.float64):
    return tau - wp.float64(WIDTH) * (cubic_sum + wp.float64(2.0)) / wp.float64(3.0)


@wp.func
def weight_seed(a: wp.float64, mean: wp.float64, spread: wp.float64):
    return wp.float64(2.0) * a / wp.float64(WIDTH) * (mean - spread)


@wp.func
def distance_seed(a: wp.float64, direct: wp.float64,
                  mean: wp.float64, spread: wp.float64):
    return direct * a * a + weight_seed(a, mean, spread)


def _make_roots(scalar):
    array = wp.array3d(dtype=scalar)

    @wp.func
    def threshold(distances: array, w: int, s: int, count: int):
        lo = wp.float64(1.0e30)
        for j in range(count):
            lo = wp.min(lo, wp.float64(distances[w, s, j]))
        tau = lo + wp.float64(WIDTH)
        for it in range(14):
            mass = wp.float64(0.0)
            total = wp.float64(0.0)
            for j in range(count):
                a = activation(tau, wp.float64(distances[w, s, j]))
                total += a
                mass += a * a
            if wp.abs(mass - wp.float64(1.0)) < wp.float64(1.0e-12):
                break
            if total > wp.float64(0.0):
                tau -= wp.float64(WIDTH) * (mass - wp.float64(1.0)) / (wp.float64(2.0) * total)
        total = wp.float64(0.0)
        squared = wp.float64(0.0)
        active = int(0)
        for j in range(count):
            d = wp.float64(distances[w, s, j])
            if d < tau:
                x = (d - lo) / wp.float64(WIDTH)
                total += x
                squared += x * x
                active += 1
        if active > 0:
            tau = root_from_moments(lo, total, squared, wp.float64(active))
        return tau

    @wp.func
    def sorted_threshold(distances: array, w: int, s: int, count: int):
        lo = wp.float64(distances[w, s, 0])
        total = wp.float64(0.0)
        squared = wp.float64(0.0)
        count_active = wp.float64(1.0)
        for j in range(1, count):
            x = (wp.float64(distances[w, s, j]) - lo) / wp.float64(WIDTH)
            if count_active*x*x - wp.float64(2.0)*x*total + squared >= wp.float64(1.0):
                break
            total += x
            squared += x*x
            count_active += wp.float64(1.0)
        return root_from_moments(lo, total, squared, count_active)

    return SimpleNamespace(threshold=threshold, sorted_threshold=sorted_threshold)


float32 = _make_roots(wp.float32)
float64 = _make_roots(wp.float64)
