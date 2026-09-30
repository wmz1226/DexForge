"""Static exact BVH construction for sphere clouds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np


DEFAULT_LEAF_SIZE = 8
GROUP_BVH_BLOCK_DIM = 128
GROUP_BVH_LEAF_SIZE = 8
TARGET_BVH_LEAF_SIZE = 4
DUAL_BVH_MIN_WORK_SPHERES = 2 * GROUP_BVH_LEAF_SIZE
DUAL_BVH_MAX_WORK_SPHERES = 8 * GROUP_BVH_LEAF_SIZE
DUAL_BVH_TARGET_WORK_ROOTS = 12
DUAL_BVH_BLOCK_DIM = 32
SAH_BIN_COUNT = 16
EXTENT_EPSILON = 1.0e-9
CACHE_SCHEMA = "sah16_dual_v1"


@dataclass(frozen=True)
class SphereBvh:
  geometry_lower: np.ndarray
  geometry_upper: np.ndarray
  center_lower: np.ndarray
  center_upper: np.ndarray
  max_radius: np.ndarray
  ranges: np.ndarray
  escape: np.ndarray
  sphere_ids: np.ndarray
  subtree_starts: np.ndarray
  subtree_counts: np.ndarray
  subtree_depths: np.ndarray


@dataclass(frozen=True)
class GroupedSphereBvh:
  geometry_lower: np.ndarray
  geometry_upper: np.ndarray
  center_lower: np.ndarray
  center_upper: np.ndarray
  max_radius: np.ndarray
  ranges: np.ndarray
  escape: np.ndarray
  sphere_ids: np.ndarray
  subtree_starts: np.ndarray
  subtree_counts: np.ndarray
  subtree_depths: np.ndarray
  group_roots: np.ndarray
  group_work_starts: np.ndarray
  group_work_counts: np.ndarray
  work_roots: np.ndarray


@dataclass
class _BuildState:
  geometry_lower: list[np.ndarray]
  geometry_upper: list[np.ndarray]
  center_lower: list[np.ndarray]
  center_upper: list[np.ndarray]
  max_radius: list[float]
  ranges: list[tuple[int, int]]
  escape: list[int]
  sphere_ids: list[int]


def _bounds(centers: np.ndarray, radii: np.ndarray,
            ids: np.ndarray) -> tuple[np.ndarray, ...]:
  selected = centers[ids]
  selected_radii = radii[ids, None]
  return (np.min(selected - selected_radii, axis=0),
          np.max(selected + selected_radii, axis=0),
          np.min(selected, axis=0), np.max(selected, axis=0),
          np.max(selected_radii))


def _append_node(state: _BuildState, bounds: tuple[np.ndarray, ...]) -> int:
  node = len(state.ranges)
  state.geometry_lower.append(bounds[0])
  state.geometry_upper.append(bounds[1])
  state.center_lower.append(bounds[2])
  state.center_upper.append(bounds[3])
  state.max_radius.append(float(bounds[4]))
  state.ranges.append((0, 0))
  state.escape.append(0)
  return node


def _median_split(centers: np.ndarray, ids: np.ndarray,
                  center_bounds: tuple[np.ndarray, np.ndarray]):
  axis = int(np.argmax(center_bounds[1] - center_bounds[0]))
  order = np.argsort(centers[ids, axis], kind="stable")
  ordered = ids[order]
  middle = ordered.size // 2
  return ordered[:middle], ordered[middle:]


def _surface_area(lower: np.ndarray, upper: np.ndarray) -> float:
  size = np.maximum(upper - lower, 0.0)
  return float(2.0 * (size[0] * size[1] + size[1] * size[2]
                      + size[2] * size[0]))


def _bin_ids(values: np.ndarray, lower: float, extent: float) -> np.ndarray:
  scaled = (values - lower) * (SAH_BIN_COUNT / extent)
  return np.minimum(scaled.astype(np.int32), SAH_BIN_COUNT - 1)


def _axis_sah(centers: np.ndarray, radii: np.ndarray, ids: np.ndarray, *,
              axis: int, lower: float, upper: float):
  extent = upper - lower
  if extent <= EXTENT_EPSILON:
    return None
  bins = _bin_ids(centers[ids, axis], lower, extent)
  counts = np.bincount(bins, minlength=SAH_BIN_COUNT)
  geometry_lower = centers[ids] - radii[ids, None]
  geometry_upper = centers[ids] + radii[ids, None]
  best = None
  for split in range(SAH_BIN_COUNT - 1):
    left = bins <= split
    right = ~left
    if not np.any(left) or not np.any(right):
      continue
    left_area = _surface_area(
        np.min(geometry_lower[left], axis=0), np.max(geometry_upper[left], axis=0))
    right_area = _surface_area(
        np.min(geometry_lower[right], axis=0), np.max(geometry_upper[right], axis=0))
    cost = left_area * counts[:split + 1].sum()
    cost += right_area * counts[split + 1:].sum()
    if best is None or cost < best[0]:
      best = (cost, bins, split)
  return best


def _split(centers: np.ndarray, radii: np.ndarray, ids: np.ndarray, *,
           center_bounds: tuple[np.ndarray, np.ndarray]):
  candidates = []
  for axis in range(3):
    candidate = _axis_sah(
        centers, radii, ids, axis=axis,
        lower=float(center_bounds[0][axis]),
        upper=float(center_bounds[1][axis]))
    if candidate is not None:
      candidates.append(candidate)
  if not candidates:
    return _median_split(centers, ids, center_bounds)
  _, bins, split = min(candidates, key=lambda value: value[0])
  return ids[bins <= split], ids[bins > split]


def _build_node(state: _BuildState, centers: np.ndarray, radii: np.ndarray, *,
                ids: np.ndarray, leaf_size: int) -> None:
  bounds = _bounds(centers, radii, ids)
  node = _append_node(state, bounds)
  if ids.size <= leaf_size:
    start = len(state.sphere_ids)
    state.sphere_ids.extend(map(int, ids))
    state.ranges[node] = (start, int(ids.size))
  else:
    left, right = _split(
        centers, radii, ids, center_bounds=(bounds[2], bounds[3]))
    _build_node(state, centers, radii, ids=left, leaf_size=leaf_size)
    _build_node(state, centers, radii, ids=right, leaf_size=leaf_size)
  state.escape[node] = len(state.ranges)


def _empty_state() -> _BuildState:
  return _BuildState([], [], [], [], [], [], [], [])


def _subtree_metadata(ranges: np.ndarray, escape: np.ndarray):
  starts = np.zeros(ranges.shape[0], dtype=np.int32)
  counts = np.zeros(ranges.shape[0], dtype=np.int32)
  depths = np.zeros(ranges.shape[0], dtype=np.int32)

  def visit(node: int) -> tuple[int, int, int]:
    start, count = map(int, ranges[node])
    if count > 0:
      starts[node], counts[node], depths[node] = start, count, 1
      return start, count, 1
    left = node + 1
    right = int(escape[left])
    left_start, left_count, left_depth = visit(left)
    _, right_count, right_depth = visit(right)
    starts[node] = left_start
    counts[node] = left_count + right_count
    depths[node] = max(left_depth, right_depth) + 1
    return int(starts[node]), int(counts[node]), int(depths[node])

  visit(0)
  return starts, counts, depths


def build_sphere_bvh(centers: np.ndarray, radii: np.ndarray,
                     leaf_size: int = DEFAULT_LEAF_SIZE) -> SphereBvh:
  """Builds a deterministic depth-first BVH with stackless escape links."""
  centers = np.asarray(centers, dtype=np.float32)
  radii = np.asarray(radii, dtype=np.float32)
  if centers.ndim != 2 or centers.shape[1] != 3:
    raise ValueError("sphere centers must have shape (count, 3)")
  if radii.shape != (centers.shape[0],):
    raise ValueError("sphere radii must have shape (count,)")
  if centers.shape[0] == 0 or leaf_size <= 0:
    raise ValueError("BVH needs spheres and a positive leaf size")
  state = _empty_state()
  ids = np.arange(centers.shape[0], dtype=np.int32)
  _build_node(state, centers, radii, ids=ids, leaf_size=leaf_size)
  ranges = np.asarray(state.ranges, np.int32)
  escape = np.asarray(state.escape, np.int32)
  subtree = _subtree_metadata(ranges, escape)
  return SphereBvh(
      np.asarray(state.geometry_lower, np.float32),
      np.asarray(state.geometry_upper, np.float32),
      np.asarray(state.center_lower, np.float32),
      np.asarray(state.center_upper, np.float32),
      np.asarray(state.max_radius, np.float32),
      ranges, escape, np.asarray(state.sphere_ids, np.int32), *subtree)


def _cache_root() -> Path:
  configured = os.environ.get("COMFREE_GS_BVH_CACHE")
  if configured:
    return Path(configured).expanduser()
  return Path.home() / ".cache/comfree_warp/gaussian_bvh"


def _geometry_digest(centers: np.ndarray, radii: np.ndarray,
                     leaf_size: int) -> str:
  digest = hashlib.sha256()
  digest.update(f"{CACHE_SCHEMA}:leaf={leaf_size}".encode("ascii"))
  for value in (centers, radii):
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
  return digest.hexdigest()


def _load_bvh(path: Path) -> SphereBvh:
  with np.load(path, allow_pickle=False) as data:
    return SphereBvh(
        data["geometry_lower"], data["geometry_upper"],
        data["center_lower"], data["center_upper"], data["max_radius"],
        data["ranges"], data["escape"], data["sphere_ids"],
        data["subtree_starts"], data["subtree_counts"],
        data["subtree_depths"])


def _save_bvh(path: Path, bvh: SphereBvh) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  with temporary.open("wb") as stream:
    np.savez(
        stream, geometry_lower=bvh.geometry_lower,
        geometry_upper=bvh.geometry_upper, center_lower=bvh.center_lower,
        center_upper=bvh.center_upper, max_radius=bvh.max_radius,
        ranges=bvh.ranges, escape=bvh.escape, sphere_ids=bvh.sphere_ids,
        subtree_starts=bvh.subtree_starts,
        subtree_counts=bvh.subtree_counts,
        subtree_depths=bvh.subtree_depths)
  os.replace(temporary, path)


def cached_sphere_bvh(centers: np.ndarray, radii: np.ndarray,
                      leaf_size: int = DEFAULT_LEAF_SIZE) -> SphereBvh:
  """Loads or atomically caches an exact static sphere-cloud BVH."""
  centers = np.asarray(centers, dtype=np.float32)
  radii = np.asarray(radii, dtype=np.float32)
  key = _geometry_digest(centers, radii, leaf_size)
  path = _cache_root() / f"{key}.npz"
  if path.exists():
    return _load_bvh(path)
  bvh = build_sphere_bvh(centers, radii, leaf_size)
  _save_bvh(path, bvh)
  return bvh


def _validate_groups(centers: np.ndarray, starts: np.ndarray,
                     counts: np.ndarray) -> None:
  if starts.ndim != 1 or counts.shape != starts.shape:
    raise ValueError("group starts and counts must be matching vectors")
  if np.any(counts <= 0):
    raise ValueError("grouped BVH requires non-empty groups")
  expected = np.cumsum(np.concatenate(([0], counts[:-1]))).astype(np.int32)
  if not np.array_equal(starts, expected) or int(counts.sum()) != centers.shape[0]:
    raise ValueError("group starts and counts must densely partition the spheres")


def _work_roots(bvh: SphereBvh, *, width: int) -> list[int]:
  roots = []
  pending = [0]
  while pending:
    node = pending.pop()
    count = int(bvh.subtree_counts[node])
    if count > width and bvh.ranges[node, 1] == 0:
      left = node + 1
      pending.extend((int(bvh.escape[left]), left))
      continue
    roots.append(node)
  return roots


def _adaptive_work_width(counts: np.ndarray) -> int:
  mean_count = (int(counts.sum()) + counts.size - 1) // counts.size
  desired = (
      mean_count + DUAL_BVH_TARGET_WORK_ROOTS - 1
  ) // DUAL_BVH_TARGET_WORK_ROOTS
  power_of_two = 1 << (max(desired, 1) - 1).bit_length()
  return min(
      max(power_of_two, DUAL_BVH_MIN_WORK_SPHERES),
      DUAL_BVH_MAX_WORK_SPHERES)


def _offset_group(bvh: SphereBvh, *, node_offset: int, sphere_offset: int,
                  source_offset: int) -> tuple[np.ndarray, ...]:
  ranges = bvh.ranges.copy()
  leaves = ranges[:, 1] > 0
  ranges[leaves, 0] += sphere_offset
  return (bvh.geometry_lower, bvh.geometry_upper, bvh.center_lower,
          bvh.center_upper, bvh.max_radius, ranges,
          bvh.escape + node_offset, bvh.sphere_ids + source_offset,
          bvh.subtree_starts + sphere_offset, bvh.subtree_counts,
          bvh.subtree_depths)


def build_grouped_sphere_bvh(
    centers: np.ndarray, radii: np.ndarray, *, starts: np.ndarray,
    counts: np.ndarray, leaf_size: int = DEFAULT_LEAF_SIZE,
    work_width: int | None = None) -> GroupedSphereBvh:
  """Builds a stackless BVH forest and bounded GPU source work tiles."""
  centers = np.asarray(centers, dtype=np.float32)
  radii = np.asarray(radii, dtype=np.float32)
  starts = np.asarray(starts, dtype=np.int32)
  counts = np.asarray(counts, dtype=np.int32)
  _validate_groups(centers, starts, counts)
  if work_width is None:
    work_width = _adaptive_work_width(counts)
  if work_width <= 0:
    raise ValueError("work width must be positive")
  columns: list[list[np.ndarray]] = [[] for _ in range(11)]
  group_roots, work_roots = [], []
  group_work_starts, group_work_counts = [], []
  node_offset = sphere_offset = 0
  for start, count in zip(starts, counts):
    local = build_sphere_bvh(
        centers[start:start + count], radii[start:start + count], leaf_size)
    group_roots.append(node_offset)
    for column, values in zip(
        columns, _offset_group(local, node_offset=node_offset,
                               sphere_offset=sphere_offset,
                               source_offset=int(start))):
      column.append(values)
    local_work_roots = _work_roots(local, width=work_width)
    group_work_starts.append(len(work_roots))
    group_work_counts.append(len(local_work_roots))
    work_roots.extend(node_offset + int(value) for value in local_work_roots)
    node_offset += local.ranges.shape[0]
    sphere_offset += local.sphere_ids.size
  merged = [np.concatenate(column) for column in columns]
  return GroupedSphereBvh(
      *merged, np.asarray(group_roots, np.int32),
      np.asarray(group_work_starts, np.int32),
      np.asarray(group_work_counts, np.int32),
      np.asarray(work_roots, np.int32))
