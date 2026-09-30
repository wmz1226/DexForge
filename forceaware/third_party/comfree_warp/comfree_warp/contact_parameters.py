"""Resolve static contact parameters from a compiled MuJoCo model."""

from typing import NamedTuple

import mujoco
import numpy as np


CONTACT_DIMS = frozenset((1, 3, 4, 6))
FRICTION_EXPANSION = np.asarray((0, 0, 1, 2, 2), dtype=np.int32)
ADDITIVE_GEOM_MARGIN_VERSION = (3, 5, 0)


class ContactParameters(NamedTuple):
  dim: np.ndarray
  includemargin: np.ndarray
  friction: np.ndarray
  solref: np.ndarray
  solreffriction: np.ndarray
  solimp: np.ndarray


class _GeomPairContext(NamedTuple):
  model: mujoco.MjModel
  geom1: np.ndarray
  geom2: np.ndarray
  selected: np.ndarray
  has_priority: np.ndarray
  higher1: np.ndarray
  higher2: np.ndarray
  solmix: np.ndarray


def resolve_contact_parameters(
    model: mujoco.MjModel,
    geom_pairs: np.ndarray,
) -> ContactParameters:
  """Applies MuJoCo's explicit-pair and dynamic-pair parameter rules."""
  pairs = _validated_geom_pairs(model, geom_pairs)
  geom_parameters = _geom_parameters(model, pairs)
  pair_ids = _explicit_pair_ids(model, pairs)
  if not np.any(pair_ids >= 0):
    return geom_parameters
  pair_parameters = _pair_parameters(model, pair_ids)
  return _select_explicit_pairs(geom_parameters, pair_parameters, pair_ids)


def _validated_geom_pairs(model, geom_pairs):
  pairs = np.asarray(geom_pairs, dtype=np.int32)
  if pairs.ndim != 2 or pairs.shape[1:] != (2,):
    raise ValueError(f"geom_pairs must have shape (N, 2), got {pairs.shape}")
  if pairs.size and (np.min(pairs) < 0 or np.max(pairs) >= model.ngeom):
    raise ValueError("geom_pairs contains an invalid MuJoCo geom id")
  return pairs


def _geom_parameters(model, pairs):
  geom1, geom2 = pairs.T
  priority1 = np.asarray(model.geom_priority)[geom1]
  priority2 = np.asarray(model.geom_priority)[geom2]
  higher1, higher2 = priority1 > priority2, priority2 > priority1
  selected = np.where(higher1, geom1, geom2)
  has_priority = higher1 | higher2
  solmix = _solmix_weight(model.geom_solmix[geom1], model.geom_solmix[geom2])
  context = _GeomPairContext(
      model, geom1, geom2, selected, has_priority, higher1, higher2, solmix)
  dim = _geom_dim(context)
  friction = _geom_friction(context)
  solref = _geom_solref(context)
  solimp = _mixed_or_priority(context, model.geom_solimp)
  includemargin = _geom_includemargin(model, geom1, geom2)
  zeros = np.zeros((pairs.shape[0], mujoco.mjNREF), dtype=np.float32)
  return ContactParameters(
      _validated_dim(dim), _float32(includemargin), friction, solref,
      zeros, _float32(solimp))


def _geom_includemargin(model, geom1, geom2):
  margin1, margin2 = model.geom_margin[geom1], model.geom_margin[geom2]
  gap1, gap2 = model.geom_gap[geom1], model.geom_gap[geom2]
  if _mujoco_version() >= ADDITIVE_GEOM_MARGIN_VERSION:
    return margin1 + margin2 - gap1 - gap2
  return np.maximum(margin1, margin2) - np.maximum(gap1, gap2)


def _mujoco_version():
  fields = mujoco.mj_versionString().split(".")
  if len(fields) < 3:
    raise ValueError(f"invalid MuJoCo version: {mujoco.mj_versionString()!r}")
  return tuple(int(field.split("-")[0]) for field in fields[:3])


def _geom_dim(context):
  dim1 = np.asarray(context.model.geom_condim)[context.geom1]
  dim2 = np.asarray(context.model.geom_condim)[context.geom2]
  return np.where(
      context.higher1, dim1,
      np.where(context.higher2, dim2, np.maximum(dim1, dim2)))


def _geom_friction(context):
  model = context.model
  friction = np.maximum(
      model.geom_friction[context.geom1], model.geom_friction[context.geom2])
  friction = np.where(
      context.has_priority[:, None], model.geom_friction[context.selected], friction)
  return _float32(friction[:, FRICTION_EXPANSION])


def _geom_solref(context):
  model = context.model
  solref1 = model.geom_solref[context.geom1]
  solref2 = model.geom_solref[context.geom2]
  standard = context.solmix[:, None] * solref1
  standard += (1.0 - context.solmix[:, None]) * solref2
  direct = np.minimum(solref1, solref2)
  both_standard = (solref1[:, 0] > 0.0) & (solref2[:, 0] > 0.0)
  mixed = np.where(both_standard[:, None], standard, direct)
  return _float32(np.where(
      context.has_priority[:, None], model.geom_solref[context.selected], mixed))


def _mixed_or_priority(context, values):
  mixed = context.solmix[:, None] * values[context.geom1]
  mixed += (1.0 - context.solmix[:, None]) * values[context.geom2]
  return np.where(
      context.has_priority[:, None], values[context.selected], mixed)


def _solmix_weight(solmix1, solmix2):
  denominator = solmix1 + solmix2
  weight = np.divide(
      solmix1, denominator, out=np.full_like(solmix1, 0.5),
      where=denominator >= mujoco.mjMINVAL)
  weight = np.where(
      (solmix1 < mujoco.mjMINVAL) & (solmix2 >= mujoco.mjMINVAL), 0.0, weight)
  return np.where(
      (solmix1 >= mujoco.mjMINVAL) & (solmix2 < mujoco.mjMINVAL), 1.0, weight)


def _explicit_pair_ids(model, pairs):
  if model.npair == 0:
    return np.full(pairs.shape[0], -1, dtype=np.int32)
  pair1 = np.asarray(model.pair_geom1, dtype=np.int32)
  pair2 = np.asarray(model.pair_geom2, dtype=np.int32)
  direct = (pairs[:, :1] == pair1) & (pairs[:, 1:] == pair2)
  reverse = (pairs[:, :1] == pair2) & (pairs[:, 1:] == pair1)
  matches = direct | reverse
  has_match = np.any(matches, axis=1)
  pair_ids = np.argmax(matches, axis=1).astype(np.int32)
  return np.where(has_match, pair_ids, -1)


def _pair_parameters(model, pair_ids):
  safe_ids = np.maximum(pair_ids, 0)
  friction = np.maximum(model.pair_friction[safe_ids], mujoco.mjMINVAL)
  return ContactParameters(
      _validated_dim(model.pair_dim[safe_ids]),
      _float32(model.pair_margin[safe_ids] - model.pair_gap[safe_ids]),
      _float32(friction), _float32(model.pair_solref[safe_ids]),
      _float32(model.pair_solreffriction[safe_ids]),
      _float32(model.pair_solimp[safe_ids]))


def _select_explicit_pairs(geom, pair, pair_ids):
  selected = pair_ids >= 0
  return ContactParameters(
      _select_values(selected, pair.dim, geom.dim),
      _select_values(selected, pair.includemargin, geom.includemargin),
      _select_values(selected, pair.friction, geom.friction),
      _select_values(selected, pair.solref, geom.solref),
      _select_values(selected, pair.solreffriction, geom.solreffriction),
      _select_values(selected, pair.solimp, geom.solimp))


def _select_values(selected, pair_value, geom_value):
  shape = (-1,) + (1,) * (pair_value.ndim - 1)
  return np.where(selected.reshape(shape), pair_value, geom_value)


def _validated_dim(dim):
  values = np.asarray(dim, dtype=np.int32)
  invalid = sorted(set(values.tolist()) - CONTACT_DIMS)
  if invalid:
    raise ValueError(f"MuJoCo contact dimensions must be 1, 3, 4, or 6: {invalid}")
  return values


def _float32(value):
  return np.asarray(value, dtype=np.float32)
