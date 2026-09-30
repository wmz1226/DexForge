#!/usr/bin/env python3
"""Contact-aware MANO-to-robot retargeting."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# Set BLAS threading before importing NumPy.
for name, value in (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_WAIT_POLICY", "PASSIVE"),
    ("MUJOCO_GL", "egl"),
):
    os.environ.setdefault(name, value)
sys.path.insert(0, ROOT)

from contactaware.settings import parse_args  # noqa: E402

if __name__ == "__main__":
    args = parse_args()
    from contactaware.pipeline import run

    run(args)
