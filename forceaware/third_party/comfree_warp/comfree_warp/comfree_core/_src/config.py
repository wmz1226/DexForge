"""Validated environment configuration for the ComFree solver."""

from __future__ import annotations

import math
import os


def env_bool(name: str, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  normalized = raw.strip().lower()
  if normalized in {"1", "true"}:
    return True
  if normalized in {"0", "false"}:
    return False
  raise ValueError(
      f"{name} must be one of 0, 1, false, true; got {raw!r}")


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
  raw = os.environ.get(name, str(default))
  try:
    value = int(raw)
  except ValueError as error:
    raise ValueError(f"{name} must be an integer; got {raw!r}") from error
  if minimum is not None and value < minimum:
    raise ValueError(f"{name} must be at least {minimum}; got {value}")
  return value


def env_positive_float(name: str, default: float) -> float:
  raw = os.environ.get(name, str(default))
  try:
    value = float(raw)
  except ValueError as error:
    raise ValueError(f"{name} must be a number; got {raw!r}") from error
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive; got {raw!r}")
  return value
