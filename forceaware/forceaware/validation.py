"""Explicit finite-value validation for ForceAware outputs."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np


def require_finite_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    invalid_mask = ~np.isfinite(array)
    if not np.any(invalid_mask):
        return array
    if array.ndim == 0:
        raise FloatingPointError(
            f"{name} contains 1 non-finite value; first at (): {array.item()}"
        )

    invalid = np.argwhere(invalid_mask)
    first = tuple(int(index) for index in invalid[0])
    value = array[first]
    raise FloatingPointError(
        f"{name} contains {invalid.shape[0]} non-finite values; "
        f"first at {first}: {value}"
    )


def require_finite_tree(name: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_tree(f"{name}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite_tree(f"{name}[{index}]", child)
        return
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        require_finite_array(name, value)
        return
    if isinstance(value, Number) and not np.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite: {value}")
