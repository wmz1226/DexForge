"""Host-side spatial index for immutable Gaussian sphere clouds."""

from __future__ import annotations

import dataclasses
from operator import attrgetter

import numpy as np


GS_BROADPHASE_EPS = 1.0e-6
HASH_EMPTY_KEY = -1
HASH_MULTIPLIER = 2_654_435_761
HASH_LOAD_FACTOR = 0.5
GRID_CELLS_PER_SUPPORT_RADIUS = np.float32(2.0)
TABLE_MODE_HASH = 0
TABLE_MODE_DENSE = 1
UINT32_MASK = 0xFFFFFFFF


@dataclasses.dataclass(frozen=True)
class RadiusGrid:
  """Exact multi-level center grid partitioned by sphere radius."""

  table: np.ndarray
  cell_sizes: np.ndarray
  max_radii: np.ndarray
  origins: np.ndarray
  cell_min: np.ndarray
  cell_dim: np.ndarray
  table_offsets: np.ndarray
  table_masks: np.ndarray
  level_shifts: np.ndarray
  table_modes: np.ndarray
  max_probe: int


@dataclasses.dataclass(frozen=True)
class RadiusGridBuild:
  order: np.ndarray
  grid: RadiusGrid


@dataclasses.dataclass(frozen=True)
class _GridLevel:
  order: np.ndarray
  keys: np.ndarray
  starts: np.ndarray
  counts: np.ndarray
  cell_min: np.ndarray
  cell_dim: np.ndarray


def build_radius_grid(centers, radii, threshold):
  """Builds the exact radius hierarchy used by grouped GPU narrow phase."""
  centers, radii = _validated_spheres(centers, radii)
  support = radii + np.float32(threshold + GS_BROADPHASE_EPS)
  level_ids, cell_sizes = _radius_levels(support)
  finest_cells = np.floor(centers / cell_sizes[0]).astype(np.int32)
  orders, tables, metadata = [], [], []
  sphere_start, table_start = 0, 0
  for level_id, cell_size in zip(np.unique(level_ids), cell_sizes):
    sphere_ids = np.flatnonzero(level_ids == level_id)
    level = _grid_level(
        finest_cells, sphere_ids, int(level_id), sphere_start=sphere_start)
    table, mask, probe, mode = _cell_table(level)
    orders.append(level.order)
    tables.append(table)
    metadata.append((level, cell_size, table_start, mask, probe,
                     float(np.max(radii[sphere_ids])), int(level_id), mode))
    sphere_start += sphere_ids.size
    table_start += table.shape[0]
  grid = _radius_grid(np.concatenate(tables), metadata)
  return RadiusGridBuild(np.concatenate(orders).astype(np.int32), grid)


def _radius_levels(support):
  base = np.float32(np.min(support))
  logarithm = np.log2(support.astype(np.float64) / float(base))
  level_ids = np.maximum(0, np.ceil(logarithm).astype(np.int32))
  represented = np.float32(base * np.exp2(level_ids.astype(np.float64)))
  level_ids += (represented < support).astype(np.int32)
  levels = np.unique(level_ids)
  sizes = np.asarray(
      [base * np.exp2(int(level)) / GRID_CELLS_PER_SUPPORT_RADIUS
       for level in levels],
      dtype=np.float32,
  )
  return level_ids, sizes


def _grid_level(finest_cells, sphere_ids, shift, *, sphere_start):
  cells = np.right_shift(finest_cells[sphere_ids].astype(np.int64), shift)
  cell_min = cells.min(axis=0)
  cell_dim = cells.max(axis=0) - cell_min + 1
  local = cells - cell_min
  keys = (local[:, 0] * cell_dim[1] + local[:, 1]) * cell_dim[2] + local[:, 2]
  order = np.argsort(keys, kind="stable")
  sorted_keys = keys[order].astype(np.int32)
  unique, first, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
  starts = first.astype(np.int32) + np.int32(sphere_start)
  return _GridLevel(
      sphere_ids[order].astype(np.int32), unique.astype(np.int32), starts,
      counts.astype(np.int32), cell_min.astype(np.int32), cell_dim.astype(np.int32))


def _cell_table(level):
  volume = int(np.prod(level.cell_dim, dtype=np.int64))
  if volume <= _table_size(level.keys.size):
    table = np.full((volume, 3), (HASH_EMPTY_KEY, 0, 0), dtype=np.int32)
    table[level.keys] = np.stack((level.keys, level.starts, level.counts), axis=1)
    return table, 0, 1, TABLE_MODE_DENSE
  return (*_hash_table(level.keys, level.starts, level.counts), TABLE_MODE_HASH)


def _hash_table(keys, starts, counts):
  size = _table_size(keys.size)
  table = np.full((size, 3), (HASH_EMPTY_KEY, 0, 0), dtype=np.int32)
  mask, max_probe = size - 1, 1
  for key, start, count in zip(keys, starts, counts):
    slot, probe = _host_hash(int(key), mask), 1
    while table[slot, 0] != HASH_EMPTY_KEY:
      slot, probe = (slot + 1) & mask, probe + 1
    table[slot] = (key, start, count)
    max_probe = max(max_probe, probe)
  return table, mask, max_probe


def _table_size(cell_count):
  required = max(2, int(np.ceil(cell_count / HASH_LOAD_FACTOR)))
  return 1 << (required - 1).bit_length()


def _host_hash(key, mask):
  return ((key * HASH_MULTIPLIER) & UINT32_MASK) & mask


def _radius_grid(table, metadata):
  columns = tuple(zip(*metadata))
  levels = columns[0]
  cell_sizes = np.asarray(columns[1], dtype=np.float32)
  cell_min = np.stack(tuple(map(attrgetter("cell_min"), levels))).astype(np.int32)
  return RadiusGrid(
      table=np.asarray(table, dtype=np.int32),
      cell_sizes=cell_sizes,
      max_radii=np.asarray(columns[5], dtype=np.float32),
      origins=(cell_min * cell_sizes[:, None]).astype(np.float32),
      cell_min=cell_min,
      cell_dim=np.stack(tuple(map(attrgetter("cell_dim"), levels))).astype(np.int32),
      table_offsets=np.asarray(columns[2], dtype=np.int32),
      table_masks=np.asarray(columns[3], dtype=np.int32),
      level_shifts=np.asarray(columns[6], dtype=np.int32),
      table_modes=np.asarray(columns[7], dtype=np.int32),
      max_probe=max(columns[4]),
  )


def _validated_spheres(centers, radii):
  centers = np.asarray(centers, dtype=np.float32)
  radii = np.asarray(radii, dtype=np.float32)
  if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
    raise ValueError(f"sphere centers must have non-empty shape (N, 3), got {centers.shape}")
  if radii.shape != (centers.shape[0],):
    raise ValueError(f"sphere radii must have shape ({centers.shape[0]},), got {radii.shape}")
  if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
    raise ValueError("sphere centers and radii must be finite")
  if np.any(radii < 0.0):
    raise ValueError("sphere radii must be non-negative")
  return centers, radii
