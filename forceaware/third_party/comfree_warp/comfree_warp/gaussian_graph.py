"""Compile unified Gaussian geoms into MuJoCo-filtered Warp batches."""

from __future__ import annotations

import dataclasses
from typing import Optional
from itertools import combinations

import mujoco
import numpy as np

from .contact_parameters import resolve_contact_parameters
from .gaussian_spatial import RadiusGrid
from .gaussian_spatial import build_radius_grid


CONTACT_THRESHOLD = 0.01
BROADPHASE_EPS = 1.0e-6
PLANE_CONTACT_COUNT = 4


@dataclasses.dataclass(frozen=True)
class LoadedGaussianGeom:
  name: str
  geom_id: int
  body_id: int
  centers: np.ndarray
  radii: np.ndarray
  group_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class GaussianPairHost:
  reference_centers: np.ndarray
  reference_radii: np.ndarray
  reference_grid: RadiusGrid
  query_centers: np.ndarray
  query_radii: np.ndarray
  query_body_ids: np.ndarray
  output_starts: np.ndarray
  output_counts: np.ndarray
  query_output_ids: np.ndarray
  contact_geom: np.ndarray
  contact_normal_sign: np.ndarray
  contact_dim: np.ndarray
  contact_includemargin: np.ndarray
  contact_friction: np.ndarray
  contact_solref: np.ndarray
  contact_solreffriction: np.ndarray
  contact_solimp: np.ndarray
  reference_body_id: int
  reference_geom_id: int
  plane_geom_id: int
  reference_plane_enabled: bool
  contact_threshold: float


@dataclasses.dataclass(frozen=True)
class GaussianHostGraph:
  pairs: tuple[GaussianPairHost, ...]
  cloud_geom_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class _BatchSpec:
  reference: LoadedGaussianGeom
  group_id: Optional[int]
  queries: tuple[LoadedGaussianGeom, ...]
  plane_geom_id: int = -1


@dataclasses.dataclass(frozen=True)
class _QueryCloud:
  centers: np.ndarray
  radii: np.ndarray
  body_ids: np.ndarray
  output_starts: np.ndarray
  output_counts: np.ndarray
  output_geom_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class _PairHostInput:
  model: mujoco.MjModel
  spec: _BatchSpec


def compile_gaussian_graph(model, clouds):
  specs = _cloud_specs(model, clouds)
  specs = _attach_plane_specs(model, clouds, specs)
  hosts = tuple(_build_pair_host(_PairHostInput(model, spec))
                for spec in specs)
  geom_ids = np.asarray([cloud.geom_id for cloud in clouds], np.int32)
  return GaussianHostGraph(hosts, geom_ids)


def _cloud_specs(model, clouds):
  edges = tuple(pair for pair in combinations(clouds, 2)
                if pair[0].centers.shape[0] and pair[1].centers.shape[0]
                if _geom_pair_enabled(model, pair[0].geom_id, pair[1].geom_id))
  oriented = _orient_edges(edges, clouds)
  buckets = {}
  for query, reference in oriented:
    bucket = buckets.setdefault(reference.geom_id, [reference, []])
    bucket[1].append(query)
  specs = []
  for reference, queries in buckets.values():
    for group_id in np.unique(reference.group_ids):
      specs.append(_BatchSpec(reference, int(group_id), tuple(queries)))
  return tuple(specs)


def _orient_edges(edges, clouds):
  degree = {cloud.geom_id: 0 for cloud in clouds}
  for first, second in edges:
    degree[first.geom_id] += 1
    degree[second.geom_id] += 1
  return tuple(_orient_edge(edge, degree) for edge in edges)


def _orient_edge(edge, degree):
  first, second = edge
  if _reference_score(first, degree) >= _reference_score(second, degree):
    return second, first
  return first, second


def _reference_score(cloud, degree):
  sphere_count = int(cloud.centers.shape[0])
  solid_count = int(np.count_nonzero(cloud.radii > 0.0))
  return degree[cloud.geom_id], sphere_count, solid_count, -cloud.geom_id


def _attach_plane_specs(model, clouds, specs):
  result = list(specs)
  plane_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
  for plane_id in plane_ids:
    for cloud in clouds:
      if cloud.centers.shape[0] == 0:
        continue
      if not _geom_pair_enabled(model, int(plane_id), cloud.geom_id):
        continue
      _attach_plane_spec(result, cloud, int(plane_id))
  return tuple(result)


def _attach_plane_spec(specs, cloud, plane_id):
  groups = np.unique(cloud.group_ids)
  merge_group = int(groups[0]) if groups.size == 1 else None
  for index, spec in enumerate(specs):
    same_reference = spec.reference.geom_id == cloud.geom_id
    if same_reference and spec.group_id == merge_group and spec.plane_geom_id < 0:
      specs[index] = dataclasses.replace(spec, plane_geom_id=plane_id)
      return
  specs.append(_BatchSpec(cloud, None, (), plane_id))


def _reference_mask(spec):
  if spec.group_id is None:
    return np.ones(spec.reference.group_ids.shape, dtype=bool)
  return spec.reference.group_ids == spec.group_id


def _build_pair_host(build):
  model, spec = build.model, build.spec
  query = _load_query_cloud(spec.queries)
  contact_geom = _contact_rows(model, spec, query.output_geom_ids)
  parameters = resolve_contact_parameters(model, contact_geom)
  thresholds = _contact_thresholds(parameters.includemargin)
  threshold = float(np.max(thresholds))
  centers, radii, grid = _reference_geometry(spec, threshold)
  signs = _normal_signs(model, spec, query.output_geom_ids)
  return GaussianPairHost(
      centers, radii, grid, query.centers, query.radii, query.body_ids,
      query.output_starts, query.output_counts,
      np.arange(query.output_starts.size, dtype=np.int32),
      contact_geom, signs, parameters.dim, parameters.includemargin,
      parameters.friction, parameters.solref, parameters.solreffriction,
      parameters.solimp, spec.reference.body_id, spec.reference.geom_id,
      spec.plane_geom_id, spec.plane_geom_id >= 0, threshold)


def _load_query_cloud(clouds):
  if not clouds:
    empty = np.zeros(0, np.float32)
    return _QueryCloud(
        np.zeros((0, 3), np.float32), empty, np.zeros(0, np.int32),
        np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32))
  rows = []
  for cloud in clouds:
    rows.extend(_query_group_rows(cloud))
  return _pack_query_rows(rows)


def _query_group_rows(cloud):
  rows = []
  for group_id in np.unique(cloud.group_ids):
    mask = cloud.group_ids == group_id
    rows.append((cloud.centers[mask], cloud.radii[mask],
                 cloud.body_id, cloud.geom_id))
  return rows


def _pack_query_rows(rows):
  centers = np.concatenate([row[0] for row in rows])
  radii = np.concatenate([row[1] for row in rows])
  counts = np.asarray([row[0].shape[0] for row in rows], np.int32)
  starts = np.cumsum(np.concatenate(([0], counts[:-1]))).astype(np.int32)
  body_ids = np.concatenate([
      np.full(count, row[2], np.int32) for count, row in zip(counts, rows)])
  geom_ids = np.asarray([row[3] for row in rows], np.int32)
  return _QueryCloud(centers, radii, body_ids, starts, counts, geom_ids)


def _reference_geometry(spec, threshold):
  mask = _reference_mask(spec)
  centers = spec.reference.centers[mask]
  radii = spec.reference.radii[mask]
  grid = build_radius_grid(centers, radii, threshold)
  return centers[grid.order], radii[grid.order], grid.grid


def _contact_rows(model, spec, query_geom_ids):
  rows = []
  if spec.plane_geom_id >= 0:
    pair = _canonical_pair(model, spec.plane_geom_id, spec.reference.geom_id)
    rows.extend([pair] * PLANE_CONTACT_COUNT)
  rows.extend(_canonical_pair(model, int(geom_id), spec.reference.geom_id)
              for geom_id in query_geom_ids)
  return np.asarray(rows, np.int32).reshape((-1, 2))


def _normal_signs(model, spec, query_geom_ids):
  signs = []
  if spec.plane_geom_id >= 0:
    pair = _canonical_pair(model, spec.plane_geom_id, spec.reference.geom_id)
    sign = 1.0 if pair[0] == spec.plane_geom_id else -1.0
    signs.extend([sign] * PLANE_CONTACT_COUNT)
  for geom_id in query_geom_ids:
    pair = _canonical_pair(model, int(geom_id), spec.reference.geom_id)
    signs.append(1.0 if pair[0] == geom_id else -1.0)
  return np.asarray(signs, np.float32)


def _canonical_pair(model, first, second):
  pair_index = _explicit_pair_index(model, first, second)
  if pair_index >= 0:
    return int(model.pair_geom1[pair_index]), int(model.pair_geom2[pair_index])
  first_key = int(model.geom_type[first]), first
  second_key = int(model.geom_type[second]), second
  return (first, second) if first_key <= second_key else (second, first)


def _contact_thresholds(includemargin):
  margin = np.asarray(includemargin, np.float32)
  return np.maximum(CONTACT_THRESHOLD, margin + BROADPHASE_EPS)


def _geom_pair_enabled(model, geom1, geom2):
  if _explicit_pair_index(model, geom1, geom2) >= 0:
    return True
  body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
  if _excluded_body_pair(model, body1, body2):
    return False
  weld1, weld2 = int(model.body_weldid[body1]), int(model.body_weldid[body2])
  if weld1 == weld2 or _filtered_parent_child(model, weld1, weld2):
    return False
  mask = model.geom_contype[geom1] & model.geom_conaffinity[geom2]
  mask |= model.geom_contype[geom2] & model.geom_conaffinity[geom1]
  return bool(mask)


def _explicit_pair_index(model, geom1, geom2):
  matches = np.flatnonzero(
      ((model.pair_geom1 == geom1) & (model.pair_geom2 == geom2)) |
      ((model.pair_geom1 == geom2) & (model.pair_geom2 == geom1)))
  return int(matches[0]) if matches.size else -1


def _excluded_body_pair(model, body1, body2):
  low, high = sorted((body1, body2))
  signature = (low << 16) + high
  return signature in set(np.asarray(model.exclude_signature).tolist())


def _filtered_parent_child(model, weld1, weld2):
  disabled = model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  if disabled or weld1 == 0 or weld2 == 0:
    return False
  parent1 = model.body_weldid[model.body_parentid[weld1]]
  parent2 = model.body_weldid[model.body_parentid[weld2]]
  return weld1 == parent2 or weld2 == parent1
