"""Build validated robot-hand profiles from declarative asset specifications."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from itertools import product

import mujoco
import numpy as np
from scipy.optimize import minimize_scalar

from contactaware.types import Capsule, HandProfile, QueryBodySpec
from contactaware.models.hand_specs import ASSET_HANDS, AssetHandSpec

BASE_QPOS_DIM = 6
MANO_FINGERS = {"if": 0, "mf": 1, "rf": 2, "th": 3, "pk": 5}
TRACK_ORDER = ("th", "if", "mf", "rf", "pk")
FINGER_LABEL = {0: "index", 1: "middle", 2: "ring", 3: "thumb", 5: "pinky"}
FINGER_NAME_RULES = (
    (0, ("index", "_ff"), ("if_",)),
    (1, ("middle", "_mf", "mid"), ("mf_",)),
    (2, ("ring", "_rf"), ("rf_",)),
    (3, ("thumb", "_th"), ("th_",)),
    (5, ("pinky",), ()),
)
CAPSULE_AXIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
CHAIN_POINT_EPS = 1e-8


def build_hand_profile(spec: AssetHandSpec) -> HandProfile:
    model = mujoco.MjModel.from_xml_path(str(spec.mesh_xml))
    default_qpos = default_qpos_for(model, spec.default_qpos)
    rows = query_specs(spec)
    track_keys = track_finger_keys(model, spec, rows)
    tracked_tips = {key: spec.tip_sites[key] for key in track_keys}
    automatic_chains = chain_points(
        model,
        palm_body=spec.palm_body,
        tip_sites=tracked_tips,
        default_qpos=default_qpos,
    )
    shape_points = configured_chains(
        model, default_qpos, spec.shape_points, fallback=automatic_chains
    )
    merged_terminal_ids = merged_terminal_finger_ids(spec, track_keys)
    return HandProfile(
        name=spec.name,
        mesh_xml=spec.mesh_xml,
        qp_dir=spec.qp_dir,
        base_qpos_dim=BASE_QPOS_DIM,
        palm_body=spec.palm_body,
        palm_anchor_offset=palm_offset(model, spec),
        track_sites=tuple(tracked_tips[key] for key in track_keys),
        track_finger_ids=tuple(MANO_FINGERS[key] for key in track_keys),
        query_specs=rows,
        capsules=capsules_for(model),
        chain_points=shape_points,
        tip_references=tip_references(model, tracked_tips),
        merged_terminal_finger_ids=merged_terminal_ids,
        default_qpos=default_qpos,
        terminal_flexion_signs={
            MANO_FINGERS[key]: spec.terminal_flexion_signs.get(key, 1.0)
            for key in track_keys
        },
    )


def merged_terminal_finger_ids(
    spec: AssetHandSpec,
    tracked_keys: tuple[str, ...],
) -> tuple[int, ...]:
    configured = set(spec.merged_terminal_fingers)
    unknown = configured - set(tracked_keys)
    if unknown:
        raise ValueError(
            f"{spec.name} merged terminal fingers are not tracked: {sorted(unknown)}"
        )
    return tuple(MANO_FINGERS[key] for key in tracked_keys if key in configured)


def track_finger_keys(model: mujoco.MjModel, spec: AssetHandSpec,
                      rows: tuple[QueryBodySpec, ...]) -> tuple[str, ...]:
    tip_query_fingers = {row.finger_id for row in rows if row.is_tip}
    keys = []
    for key in TRACK_ORDER:
        site = spec.tip_sites.get(key)
        finger_id = MANO_FINGERS[key]
        if site is None or not has_site(model, site):
            continue
        if finger_id not in tip_query_fingers:
            raise ValueError(f"{spec.name} has tip site '{site}' but no tip query points for {key}")
        keys.append(key)
    if not keys:
        raise ValueError(f"{spec.name} has no trackable fingertip sites")
    return tuple(keys)


def default_qpos_for(model: mujoco.MjModel,
                     configured: tuple[float, ...] | None) -> np.ndarray:
    if configured is not None:
        values = np.asarray(configured, dtype=np.float64)
        if values.shape[0] != model.nq:
            raise ValueError(f"default_qpos dim {values.shape[0]} does not match nq={model.nq}")
        return values
    qpos = np.zeros(model.nq, dtype=np.float64)
    ranges = np.asarray(model.jnt_range, dtype=np.float64)
    lower, upper = ranges[:, 0], ranges[:, 1]
    finite = np.isfinite(lower) & np.isfinite(upper)
    qpos[BASE_QPOS_DIM:] = lower[BASE_QPOS_DIM:] + 0.35 * (upper - lower)[BASE_QPOS_DIM:]
    qpos[~finite] = 0.0
    return np.clip(qpos, lower, upper)


def query_specs(spec: AssetHandSpec) -> tuple[QueryBodySpec, ...]:
    rows = [query_row(path) for path in sorted(spec.qp_dir.glob("*.npz"))]
    explicit_tip = {row.finger_id for row in rows if "tip" in row.file_stem}
    max_link = max_link_by_finger(rows)
    specs = []
    for row in rows:
        is_tip = row.finger_id in MANO_FINGERS.values()
        is_tip = (
            is_tip
            and row.finger_id not in explicit_tip
            and row.link_id == max_link[row.finger_id]
        )
        specs.append(row if ("tip" not in row.file_stem and not is_tip)
                     else QueryBodySpec(row.file_stem, row.body, row.finger_id, row.link_id, True))
    return tuple(specs)


def query_row(path: Path) -> QueryBodySpec:
    stem = path.stem
    with np.load(path) as npz:
        finger = first_or_default(npz, "finger_idx", infer_finger_id(stem))
        link = first_or_default(npz, "link_id", infer_link_id(stem))
    return QueryBodySpec(stem, stem, finger, link, "tip" in stem)


def first_or_default(npz, key: str, default: int) -> int:
    if key not in npz.files or npz[key].size == 0:
        return int(default)
    return int(np.asarray(npz[key]).reshape(-1)[0])


def infer_finger_id(name: str) -> int:
    normalized = name.lower()
    for finger_id, substrings, prefixes in FINGER_NAME_RULES:
        if any(token in normalized for token in substrings) or normalized.startswith(prefixes):
            return finger_id
    return 4


def infer_link_id(name: str) -> int:
    low = name.lower()
    if any(token in low for token in ("base", "_bs", "palm")):
        return 0
    if any(token in low for token in ("proximal", "_px", "l1")):
        return 1
    if any(token in low for token in ("medial", "intermediate", "_md")):
        return 2
    return 3 if any(token in low for token in ("distal", "_ds", "tip", "l2")) else 0


def max_link_by_finger(rows: list[QueryBodySpec]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in rows:
        out[row.finger_id] = max(out.get(row.finger_id, row.link_id), row.link_id)
    return out


def chain_points(
    model: mujoco.MjModel,
    *,
    palm_body: str,
    tip_sites: dict[str, str],
    default_qpos: np.ndarray,
) -> dict[str, tuple[tuple[str, str], ...]]:
    data = mujoco.MjData(model)
    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)
    return {
        finger: make_chain(
            model,
            data,
            palm_body=palm_body,
            tip_site=site,
        )
        for finger, site in tip_sites.items()
    }


def make_chain(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    palm_body: str,
    tip_site: str,
) -> tuple[tuple[str, str], ...]:
    path = body_path(model, palm_body, site_body(model, tip_site))
    body_names = select_chain_bodies(model, data, path)
    specs = tuple(("body", name) for name in body_names) + (("site", tip_site),)
    validate_chain_segments(model, data, specs)
    return specs


def configured_chains(
    model: mujoco.MjModel,
    default_qpos: np.ndarray,
    configured: dict[str, tuple[tuple[str, str], ...]],
    *,
    fallback: dict[str, tuple[tuple[str, str], ...]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    if not configured:
        return fallback
    if set(configured) != set(fallback):
        raise ValueError(
            f"Configured retarget fingers {sorted(configured)} do not match "
            f"tracked fingers {sorted(fallback)}"
        )
    data = mujoco.MjData(model)
    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)
    for specs in configured.values():
        validate_chain_segments(model, data, specs)
    return configured


def validate_chain_segments(model: mujoco.MjModel, data: mujoco.MjData,
                            specs: tuple[tuple[str, str], ...]) -> None:
    positions = np.asarray([chain_point_position(model, data, spec) for spec in specs])
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    bad = np.flatnonzero(lengths <= CHAIN_POINT_EPS)
    if bad.size == 0:
        return
    left = specs[int(bad[0])]
    right = specs[int(bad[0]) + 1]
    raise ValueError(f"Degenerate hand chain segment: {left} -> {right}")


def chain_point_position(model: mujoco.MjModel, data: mujoco.MjData,
                         spec: tuple[str, str]) -> np.ndarray:
    kind, name = spec
    if kind == "body":
        return np.asarray(data.xpos[body_id(model, name)], dtype=np.float64)
    if kind == "site":
        return np.asarray(data.site_xpos[site_id(model, name)], dtype=np.float64)
    raise ValueError(f"Unknown chain point kind: {kind}")


def select_chain_bodies(model: mujoco.MjModel, data: mujoco.MjData,
                        path: list[str]) -> list[str]:
    candidates = path[1:] if len(path) >= 4 else path
    candidates = remove_coincident_bodies(model, data, candidates)
    if len(candidates) >= 3:
        return candidates[:3]
    return ([path[0]] + candidates)[-3:]


def remove_coincident_bodies(model: mujoco.MjModel, data: mujoco.MjData,
                             names: list[str]) -> list[str]:
    kept, previous = [], None
    for name in names:
        pos = np.asarray(data.xpos[body_id(model, name)], dtype=np.float64)
        if previous is None or np.linalg.norm(pos - previous) > CHAIN_POINT_EPS:
            kept.append(name)
            previous = pos
    return kept


def capsules_for(model: mujoco.MjModel) -> tuple[Capsule, ...]:
    capsules: list[Capsule] = []
    for geom_id in range(model.ngeom):
        capsules.extend(collision_geom_capsules(model, geom_id))
    if not capsules:
        raise ValueError("No finger collision geoms found for self-collision capsules")
    return tuple(capsules)


def collision_geom_capsules(model: mujoco.MjModel, geom_id: int) -> tuple[Capsule, ...]:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if "collision" not in name:
        return ()
    body = body_name(model, int(model.geom_bodyid[geom_id]))
    finger_id = infer_finger_id(f"{name}_{body}")
    if finger_id not in FINGER_LABEL:
        return ()
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        parts = box_capsules(model.geom_size[geom_id])
        rotation, center = geom_rotation(model, geom_id), model.geom_pos[geom_id]
        parts = [(center + rotation @ start, center + rotation @ end, radius)
                 for start, end, radius in parts]
    else:
        parts = [capsule_from_geom(model, geom_id)]
    return tuple(Capsule(name, body, FINGER_LABEL[finger_id], tuple(start), tuple(end), radius)
                 for start, end, radius in parts)


def box_capsules(half_sizes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Enclose four transverse box cells with capsules."""
    half_sizes = np.asarray(half_sizes, dtype=np.float64)
    axis = int(np.argmax(half_sizes))
    transverse = [i for i in range(3) if i != axis]
    corners = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
    candidates = []
    for counts in ((1, 4), (2, 2), (4, 1)):
        cell = half_sizes.copy()
        cell[transverse] /= counts
        start, end, radius = enclosing_capsule(corners * cell)
        excess = max(radius - np.min(cell - np.abs(start)),
                     radius - np.min(cell - np.abs(end)))
        candidates.append((excess, counts, cell, start, end, radius))
    _, counts, cell, start, end, radius = min(candidates, key=lambda item: item[0])
    parts = []
    for first, second in product(range(counts[0]), range(counts[1])):
        offset = np.zeros(3)
        offset[transverse] = (-half_sizes[transverse]
                              + (2 * np.asarray([first, second]) + 1) * cell[transverse])
        parts.append((start + offset, end + offset, radius))
    return parts


def capsule_from_geom(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return sphere_capsule(model, geom_id)
    if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return axial_capsule(model, geom_id, 0)
    if geom_type in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH):
        return box_like_capsule(model, geom_id)
    raise ValueError(
        f"Unsupported collision geom type={geom_type} for "
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)}")


def sphere_capsule(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
    return center, center.copy(), float(model.geom_size[geom_id, 0])


def axial_capsule(model: mujoco.MjModel, geom_id: int,
                  axis_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
    radius = float(model.geom_size[geom_id, 0])
    half_length = float(model.geom_size[geom_id, axis_id + 1])
    axis = geom_rotation(model, geom_id) @ CAPSULE_AXIS
    return center - half_length * axis, center + half_length * axis, radius


def box_like_capsule(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        points = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh_id])]
    else:
        points = np.asarray(tuple(product((-1.0, 1.0), repeat=3))) * model.geom_size[geom_id]
    start, end, radius = enclosing_capsule(points)
    rotation = geom_rotation(model, geom_id)
    center = model.geom_pos[geom_id]
    return center + rotation @ start, center + rotation @ end, radius


def enclosing_capsule(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimum-volume centered capsule enclosing every geometry vertex."""
    points = np.asarray(points, dtype=np.float64)
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    scale = float(np.max(np.abs(points - center)))
    local = (points - center) / scale
    best = None
    for axis in range(3):
        axial = np.abs(local[:, axis])
        radial_sq = np.sum(np.delete(local, axis, axis=1) ** 2, axis=1)

        def radius(length):
            return float(np.sqrt(np.max(radial_sq + np.maximum(axial - length, 0.0) ** 2)))

        def volume(length):
            r = radius(length)
            return 2.0 * length * r * r + (4.0 / 3.0) * r ** 3

        extent = float(axial.max())
        fit = minimize_scalar(volume, bounds=(0.0, extent), method="bounded",
                              options={"xatol": 1e-8})
        length = min((0.0, extent, float(fit.x)), key=volume)
        candidate = (volume(length), axis, length, radius(length))
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, axis, length, r = best
    offset = np.eye(3)[axis] * length * scale
    return center - offset, center + offset, float(np.nextafter(r * scale, np.inf))


def geom_rotation(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(model.geom_quat[geom_id], dtype=np.float64))
    return mat.reshape(3, 3)


def tip_references(model: mujoco.MjModel,
                   tip_sites: dict[str, str]) -> dict[int, tuple[str, tuple[float, float, float]]]:
    out = {}
    for finger, site in tip_sites.items():
        sid = site_id(model, site)
        out[MANO_FINGERS[finger]] = (
            body_name(model, int(model.site_bodyid[sid])),
            tuple(np.asarray(model.site_pos[sid], dtype=float)),
        )
    return out


def palm_offset(model: mujoco.MjModel, spec: AssetHandSpec) -> tuple[float, float, float]:
    if not spec.palm_site or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.palm_site) < 0:
        return (0.0, 0.0, 0.0)
    sid = site_id(model, spec.palm_site)
    if int(model.site_bodyid[sid]) != body_id(model, spec.palm_body):
        return (0.0, 0.0, 0.0)
    return tuple(np.asarray(model.site_pos[sid], dtype=float))


def body_path(model: mujoco.MjModel, root_name: str, leaf_name: str) -> list[str]:
    ids = body_path_ids(model, body_id(model, root_name), body_id(model, leaf_name))
    return [body_name(model, item) for item in ids]


def body_path_ids(model: mujoco.MjModel, root: int, leaf: int) -> list[int]:
    path = [leaf]
    while path[-1] != root:
        parent = int(model.body_parentid[path[-1]])
        if parent == path[-1] or parent < 0:
            raise ValueError(f"Body {body_name(model, leaf)} is not under {body_name(model, root)}")
        path.append(parent)
    return list(reversed(path))


def site_body(model: mujoco.MjModel, name: str) -> str:
    return body_name(model, site_body_id(model, name))


def site_body_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.site_bodyid[site_id(model, name)])


def has_site(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0


def site_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if idx < 0:
        raise ValueError(f"Missing MuJoCo site: {name}")
    return int(idx)


def body_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise ValueError(f"Missing MuJoCo body: {name}")
    return int(idx)


def body_name(model: mujoco.MjModel, idx: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
    if not name:
        raise ValueError(f"Body id {idx} is unnamed")
    return name


HAND_SPECS = {spec.name: spec for spec in ASSET_HANDS}
_PROFILE_CACHE: dict[tuple[str, str], HandProfile] = {}


def supported_hands() -> tuple[str, ...]:
    return tuple(sorted(HAND_SPECS))


def get_hand_profile(name: str, mesh_xml: Path | None = None) -> HandProfile:
    """Build one hand profile on demand and reuse it for the whole process; mesh_xml replaces the asset robot."""
    key = name.lower()
    if key not in HAND_SPECS:
        supported = ", ".join(supported_hands())
        raise ValueError(f"Unsupported hand '{name}'. Supported hands: {supported}")
    spec = HAND_SPECS[key] if mesh_xml is None else replace(HAND_SPECS[key], mesh_xml=Path(mesh_xml))
    cache_key = (key, str(spec.mesh_xml))
    if cache_key not in _PROFILE_CACHE:
        _PROFILE_CACHE[cache_key] = build_hand_profile(spec)
    return _PROFILE_CACHE[cache_key]
