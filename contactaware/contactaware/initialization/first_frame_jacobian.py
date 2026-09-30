"""Native batched point Jacobians with the same arithmetic order as the NumPy implementation."""

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from contactaware.cache import CACHE_ROOT

XYZ_DIM = 3
COMPILER_OPTIONS = ("-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off")


@lru_cache(maxsize=1)
def load_point_jacobian_kernel():
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes() + repr(COMPILER_OPTIONS).encode()).hexdigest()
    directory = CACHE_ROOT / "first_frame_jacobian" / digest
    library = directory / "point_jacobians.so"
    if not library.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            compiled = Path(temporary) / library.name
            subprocess.run(["c++", *COMPILER_OPTIONS, str(source), "-o", str(compiled)], check=True)
            compiled.replace(library)
    api = ctypes.CDLL(str(library))
    function = api.point_jacobians
    floats = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    indices = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    function.argtypes = [floats, indices, floats, ctypes.c_int64, ctypes.c_int64, floats]
    function.restype = None
    return function


def point_jacobians_from_bodies(body_jac, origins, body_ids, points, *, kernel):
    """Reuse rigid-body Jacobians for surface, contact and capsule witnesses."""
    offsets = np.ascontiguousarray(points - origins[body_ids], dtype=np.float64)
    result = np.empty((len(points), XYZ_DIM, body_jac.shape[-1]), dtype=np.float64)
    kernel(offsets, np.asarray(body_ids, dtype=np.int64), body_jac,
           len(points), body_jac.shape[-1], result)
    return result
