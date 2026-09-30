"""Native exact nearest-sphere search, used to retract anchors onto the sphere-union surface."""

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import weakref

import numpy as np


XYZ_DIM = 3
COMPILE_FLAGS = ("-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp")
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
INT_POINTER = ctypes.POINTER(ctypes.c_int)


def _library():
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes() + repr(COMPILE_FLAGS).encode()).hexdigest()
    user_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = Path(os.environ.get("DEXFORGE_CACHE_DIR", user_cache / "dexforge")) / "geometry"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"sphere_bvh_{digest}.so"
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            target = Path(directory) / "sphere_bvh.so"
            subprocess.run(["c++", *COMPILE_FLAGS, str(source), "-o", str(target)], check=True)
            target.replace(library)
    api = ctypes.CDLL(str(library))
    api.sphere_bvh_create.argtypes = [DOUBLE_POINTER, DOUBLE_POINTER, ctypes.c_int]
    api.sphere_bvh_create.restype = ctypes.c_void_p
    api.sphere_bvh_destroy.argtypes = [ctypes.c_void_p]
    api.sphere_bvh_destroy.restype = None
    api.sphere_bvh_query.argtypes = [ctypes.c_void_p, DOUBLE_POINTER, ctypes.c_int,
                                   INT_POINTER, ctypes.c_int]
    api.sphere_bvh_query.restype = None
    api.sphere_bvh_retreat.argtypes = [ctypes.c_void_p, DOUBLE_POINTER, ctypes.c_int,
                                      DOUBLE_POINTER, ctypes.c_double, ctypes.c_double]
    api.sphere_bvh_retreat.restype = ctypes.c_double
    return api


class SphereBVH:
    def __init__(self, centers, radii, *, api, threads=1):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("Sphere query thread count must be a positive integer")
        self.threads = threads
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        if centers.shape != (len(radii), XYZ_DIM) or not len(radii):
            raise ValueError("Sphere BVH requires matching nonempty centers and radii")
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("Sphere BVH requires finite centers and positive finite radii")
        self.api = api
        self.handle = api.sphere_bvh_create(centers.ctypes.data_as(DOUBLE_POINTER),
                                           radii.ctypes.data_as(DOUBLE_POINTER), len(radii))
        if not self.handle:
            raise RuntimeError("Native sphere BVH allocation/build failed")
        self._cleanup = weakref.finalize(self, api.sphere_bvh_destroy, self.handle)

    def __call__(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != XYZ_DIM or not np.isfinite(points).all():
            raise ValueError("Sphere BVH query requires finite (N, 3) points")
        ids = np.empty(len(points), dtype=np.int32)
        self.api.sphere_bvh_query(self.handle, points.ctypes.data_as(DOUBLE_POINTER),
                                  len(points), ids.ctypes.data_as(INT_POINTER), self.threads)
        return ids

    def clearance_retreat(self, points, direction, padding, limit):
        points = np.ascontiguousarray(points, dtype=np.float64)
        direction = np.ascontiguousarray(direction, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != XYZ_DIM or not np.isfinite(points).all():
            raise ValueError("Retreat requires finite (N, 3) query points")
        if direction.shape != (XYZ_DIM,) or not np.isfinite(direction).all():
            raise ValueError("Retreat direction must have three finite components")
        if not np.isclose(np.linalg.norm(direction), 1.) or not np.isfinite([padding, limit]).all():
            raise ValueError("Retreat requires a unit direction and finite padding/limit")
        result = self.api.sphere_bvh_retreat(
            self.handle, points.ctypes.data_as(DOUBLE_POINTER), len(points),
            direction.ctypes.data_as(DOUBLE_POINTER), padding, limit,
        )
        if not np.isfinite(result):
            raise RuntimeError("Native sphere ray-interval search failed")
        return result


def make_bvh_sphere_selector(centers, radii, *, threads=1):
    return SphereBVH(centers, radii, api=_library(), threads=threads)
