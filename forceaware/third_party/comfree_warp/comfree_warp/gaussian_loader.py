"""MuJoCo XML adapter for Warp Gaussian sphere-cloud collision."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from .gaussian_graph import compile_gaussian_graph
from .gaussian_graph import LoadedGaussianGeom
from . import collision_targets


GS_RADIUS_MULTIPLIER = 3.0
_SPHERE_FIELDS = frozenset({"local_pos", "local_normal", "radius"})
_PLY_SCALARS = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclasses.dataclass(frozen=True)
class _CustomGeom:
  name: str
  file: str
  xml_dir: str


@dataclasses.dataclass(frozen=True)
class _PreprocessedXML:
  model_xml: str
  collision_xml: str
  geoms: tuple[_CustomGeom, ...]
  assets: dict[str, bytes]


@dataclasses.dataclass(frozen=True)
class _CloudAsset:
  centers: np.ndarray
  radii: np.ndarray
  groups: np.ndarray
  mesh: tuple | None = None


@dataclasses.dataclass(frozen=True)
class _PlyHeader:
  encoding: str
  vertex_count: int
  properties: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class _PlyHeaderState:
  encoding: str = ""
  vertex_count: int = -1
  current_element: str = ""
  properties: tuple[tuple[str, str], ...] = ()


def load_model(xml_path: str, *, device=None):
  """Loads a MuJoCo model and attaches exact Warp GS collision metadata."""
  from .api import put_model
  from .comfree_core._src.collision_gaussian import create_collision_models

  source_path = os.path.abspath(os.fspath(xml_path))
  prepared = _preprocess_xml(source_path)
  if not prepared.geoms:
    mj_model = mujoco.MjModel.from_xml_path(source_path)
    return _put_model(put_model, mj_model, device), mj_model

  mj_model = mujoco.MjModel.from_xml_string(prepared.model_xml, assets=prepared.assets)
  collision_model = mujoco.MjModel.from_xml_string(
      prepared.collision_xml, assets=prepared.assets)
  host, meshes = _build_host_graph(mj_model, collision_model, prepared)
  model = _put_model(put_model, mj_model, device)
  model.gaussian_collision = create_collision_models(
      host, model.geom_type.device)
  # Triangle-mesh targets by body id; their hull vertices form the target cloud above.
  model.mesh_targets = meshes
  model.gaussian_sparse_nnz_reserve = _gaussian_sparse_nnz_reserve(
      mj_model, host)
  return model, mj_model


def _gaussian_sparse_nnz_reserve(mj_model, graph) -> int:
  from .mujoco_warp._src.io import _body_pair_nnz

  reserve = 0
  for batch in graph.pairs:
    for geom_pair, dimension in zip(
        batch.contact_geom, batch.contact_dim, strict=True):
      body_ids = mj_model.geom_bodyid[np.asarray(geom_pair, dtype=np.int32)]
      support = _body_pair_nnz(mj_model, int(body_ids[0]), int(body_ids[1]))
      rows = 1 if int(dimension) == 1 else 2 * (int(dimension) - 1)
      reserve += rows * support
  return reserve


def _put_model(put_model, mj_model, device):
  if device is None:
    return put_model(mj_model)
  with wp.ScopedDevice(device):
    return put_model(mj_model)


def _preprocess_xml(xml_path):
  root = ET.parse(xml_path).getroot()
  source_dirs = {}
  xml_dir = os.path.dirname(xml_path)
  _record_sources(root, xml_dir, source_dirs)
  _inline_includes(
      root, xml_dir, source_dirs, active_paths=frozenset((xml_path,)))
  geoms, elements = _collect_custom_geoms(root, source_dirs)
  collision_xml = ET.tostring(root, encoding="unicode")
  for geom in elements:
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
  _remove_custom_pairs(root, {descriptor.name for descriptor in geoms})
  return _PreprocessedXML(
      ET.tostring(root, encoding="unicode"), collision_xml, tuple(geoms),
      _load_assets(xml_path))


def _remove_custom_pairs(root, custom_names):
  for contact in root.iter("contact"):
    for pair in list(contact.findall("pair")):
      geom_names = {pair.get("geom1"), pair.get("geom2")}
      if geom_names & custom_names:
        contact.remove(pair)


def _inline_includes(root, xml_dir, source_dirs, *, active_paths):
  replacements = []
  for parent in root.iter():
    for index, child in enumerate(list(parent)):
      if child.tag != "include":
        continue
      included = _read_include(
          child, xml_dir, source_dirs, active_paths=active_paths)
      replacements.append((parent, index, child, included))
  for parent, index, child, included in replacements:
    parent.remove(child)
    for offset, element in enumerate(included):
      parent.insert(index + offset, element)


def _read_include(element, xml_dir, source_dirs, *, active_paths):
  filename = element.get("file", "")
  if not filename:
    raise ValueError("MuJoCo <include> requires a non-empty file attribute")
  path = os.path.abspath(os.path.join(xml_dir, filename))
  if path in active_paths:
    raise ValueError(f"cyclic MuJoCo XML include: {path}")
  if not os.path.exists(path):
    raise FileNotFoundError(f"included MuJoCo XML does not exist: {path}")
  root = ET.parse(path).getroot()
  source_dir = os.path.dirname(path)
  _inline_includes(
      root, source_dir, source_dirs,
      active_paths=active_paths | frozenset((path,)))
  _record_sources(root, source_dir, source_dirs)
  return list(root)


def _record_sources(root, xml_dir, source_dirs):
  for element in root.iter():
    source_dirs.setdefault(id(element), xml_dir)


def _collect_custom_geoms(root, source_dirs):
  geoms, elements = [], []
  for index, geom in enumerate(root.iter("geom")):
    geom_type = geom.get("type", "")
    if geom_type not in ("gs", "querypoint", "sdfmesh"):
      continue
    descriptor = _geom_descriptor(geom, index, source_dirs)
    geoms.append(descriptor)
    elements.append(geom)
    _replace_cloud_geom(geom)
  return geoms, elements


def _geom_descriptor(geom, index, source_dirs):
  name = geom.get("name") or f"__gaussian_cloud_{index}"
  geom.set("name", name)
  return _CustomGeom(name, geom.get("file", ""), source_dirs[id(geom)])


def _replace_cloud_geom(geom):
  geom.set("type", "sphere")
  geom.set("size", geom.get("size", "0.0001").split()[0])
  geom.set("rgba", "0 0 0 0")
  geom.set("group", "4")
  geom.attrib.pop("file", None)
  geom.attrib.pop("normal_file", None)


def _load_assets(xml_path):
  assets = {}
  _collect_assets(xml_path, assets, frozenset())
  return assets


def _collect_assets(xml_path, assets, visited):
  xml_path = os.path.abspath(xml_path)
  if xml_path in visited:
    return
  root = ET.parse(xml_path).getroot()
  xml_dir = os.path.dirname(xml_path)
  compiler = next(root.iter("compiler"), None)
  meshdir = compiler.get("meshdir", "") if compiler is not None else ""
  texturedir = compiler.get("texturedir", "") if compiler is not None else ""
  _collect_declared_assets(root.iter("mesh"), os.path.join(xml_dir, meshdir), assets)
  _collect_declared_assets(root.iter("texture"), os.path.join(xml_dir, texturedir), assets)
  for include in root.findall("include"):
    filename = include.get("file", "")
    if not filename:
      raise ValueError("MuJoCo <include> requires a non-empty file attribute")
    path = os.path.join(xml_dir, filename)
    assets[filename] = _read_required(path)
    _collect_assets(path, assets, visited | frozenset((xml_path,)))


def _collect_declared_assets(elements, base_dir, assets):
  for element in elements:
    filename = element.get("file", "")
    if filename:
      assets[filename] = _read_required(os.path.join(base_dir, filename))


def _read_required(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"MuJoCo asset does not exist: {path}")
  return Path(path).read_bytes()


def _build_host_graph(model, collision_model, prepared):
  loaded = tuple(_load_compiled_cloud(model, geom) for geom in prepared.geoms)
  clouds = tuple(cloud for cloud, _ in loaded)
  meshes = {cloud.body_id: mesh for cloud, mesh in loaded if mesh is not None}
  return compile_gaussian_graph(collision_model, clouds), meshes


def _load_compiled_cloud(model, descriptor):
  geom_id = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, descriptor.name)
  asset = _load_cloud_asset(descriptor)
  transform = _geom_transform(model, geom_id)
  centers = _transform_points(asset.centers, transform)
  mesh = None
  if asset.mesh is not None:
    vertices, faces = asset.mesh
    mesh = (_transform_points(vertices, transform).astype(np.float32), faces)
  return LoadedGaussianGeom(
      descriptor.name, geom_id, int(model.geom_bodyid[geom_id]),
      centers, asset.radii, asset.groups), mesh


@dataclasses.dataclass(frozen=True)
class _RigidTransform:
  position: np.ndarray
  rotation: np.ndarray


def _geom_transform(model, geom_id):
  rotation = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
  return _RigidTransform(
      np.asarray(model.geom_pos[geom_id], dtype=np.float32),
      rotation.reshape((3, 3)).astype(np.float32))


def _transform_points(points, transform):
  return points @ transform.rotation.T + transform.position[None]


def _asset_path(geom):
  if not geom.file:
    raise ValueError(f"custom collision geom '{geom.name}' is missing file")
  return geom.file if os.path.isabs(geom.file) else os.path.join(geom.xml_dir, geom.file)


def _load_cloud_asset(descriptor):
  path = _asset_path(descriptor)
  suffix = Path(path).suffix.lower()
  if suffix == ".obj":
    vertices, faces, hull = collision_targets.load_mesh_asset(path)
    return _CloudAsset(
        hull.astype(np.float32),
        np.full(len(hull), collision_targets.HULL_VERTEX_RADIUS, np.float32),
        np.zeros(len(hull), np.int32), (vertices, faces))
  if suffix == ".npz":
    centers, radii, groups = _load_sphere_npz(path)
    return _CloudAsset(centers, radii, groups)
  if suffix == ".ply":
    centers, radii = _load_gaussian_ply(path)
    groups = np.zeros(centers.shape[0], dtype=np.int32)
    return _CloudAsset(
        centers, radii * np.float32(GS_RADIUS_MULTIPLIER), groups)
  raise ValueError(
      f"Gaussian cloud '{descriptor.name}' uses unsupported asset: {path}")


def _load_sphere_npz(path):
  with np.load(path) as data:
    centers = np.asarray(data["local_pos"], dtype=np.float32)
    _validate_centers(centers, path)
    radii = _source_radii(data, centers.shape[0])
    groups = _source_groups(data, centers.shape[0])
  return centers, radii, groups


def _validate_centers(centers, source):
  if centers.ndim != 2 or centers.shape[1] != 3:
    raise ValueError(f"{source}: local_pos must have shape (N, 3)")
  if not np.all(np.isfinite(centers)):
    raise ValueError(f"{source}: local_pos must be finite")


def _source_radii(data, count):
  radii = np.zeros(count, dtype=np.float32)
  if "radius" in data:
    radii = np.asarray(data["radius"], dtype=np.float32)
  if radii.shape != (count,) or not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
    raise ValueError(f"sphere radius must have finite non-negative shape ({count},)")
  return radii


def _source_groups(data, count):
  if "group_id" in data:
    values = np.asarray(data["group_id"], dtype=np.int32)
    if values.shape != (count,):
      raise ValueError(f"group_id must have shape ({count},)")
    return values
  columns = []
  for name in sorted(set(data.files) - _SPHERE_FIELDS):
    values = np.asarray(data[name])
    if values.shape == (count,) and np.issubdtype(values.dtype, np.integer):
      columns.append(values)
  if not columns:
    return np.zeros(count, dtype=np.int32)
  return np.unique(np.stack(columns, axis=1), axis=0, return_inverse=True)[1].astype(np.int32)


def _load_gaussian_ply(path):
  with open(path, "rb") as stream:
    header = _read_ply_header(stream, path)
    rows = _read_ply_vertices(stream, header, path)
  names = rows.dtype.names or ()
  if not {"x", "y", "z"}.issubset(names):
    raise ValueError(f"PLY is missing x/y/z vertex fields: {path}")
  scales = sorted((name for name in names if name.startswith("scale_")), key=_scale_index)
  if not scales:
    raise ValueError(f"PLY is missing scale_* vertex fields: {path}")
  centers = np.stack([rows[name] for name in ("x", "y", "z")], axis=1)
  radii = np.exp(np.asarray(rows[scales[0]], dtype=np.float32))
  return centers.astype(np.float32), radii.astype(np.float32)


def _read_ply_header(stream, path):
  first = stream.readline().decode("ascii", errors="strict").strip()
  if first != "ply":
    raise ValueError(f"not a PLY file: {path}")
  state = _PlyHeaderState()
  while True:
    line = stream.readline()
    if not line:
      raise ValueError(f"unterminated PLY header: {path}")
    fields = line.decode("ascii", errors="strict").strip().split()
    state, complete = _parse_ply_header_line(state, fields, path)
    if complete:
      break
  if state.vertex_count < 0 or not state.encoding:
    raise ValueError(f"PLY header is missing format or vertex element: {path}")
  return _PlyHeader(state.encoding, state.vertex_count, state.properties)


def _parse_ply_header_line(state, fields, path):
  if not fields:
    return state, False
  keyword = fields[0]
  if keyword == "format":
    return dataclasses.replace(state, encoding=fields[1]), False
  if keyword == "element":
    count = int(fields[2]) if fields[1] == "vertex" else state.vertex_count
    return dataclasses.replace(
        state, current_element=fields[1], vertex_count=count), False
  if keyword == "property" and state.current_element == "vertex":
    if len(fields) != 3:
      raise ValueError(f"list-valued vertex properties are unsupported: {path}")
    properties = state.properties + ((fields[2], fields[1]),)
    return dataclasses.replace(state, properties=properties), False
  return state, keyword == "end_header"


def _read_ply_vertices(stream, header, path):
  endian = "<" if header.encoding == "binary_little_endian" else ">"
  if header.encoding == "ascii":
    matrix = np.loadtxt(stream, max_rows=header.vertex_count, ndmin=2)
    dtype = _ply_dtype(header.properties, "=")
    rows = np.empty(header.vertex_count, dtype=dtype)
    for column, (name, _) in enumerate(header.properties):
      rows[name] = matrix[:, column]
    return rows
  if header.encoding not in ("binary_little_endian", "binary_big_endian"):
    raise ValueError(f"unsupported PLY encoding '{header.encoding}': {path}")
  rows = np.fromfile(stream, dtype=_ply_dtype(header.properties, endian), count=header.vertex_count)
  if rows.shape[0] != header.vertex_count:
    raise ValueError(f"truncated PLY vertex data: {path}")
  return rows


def _ply_dtype(properties, endian):
  fields = []
  for name, scalar_type in properties:
    if scalar_type not in _PLY_SCALARS:
      raise ValueError(f"unsupported PLY scalar type '{scalar_type}'")
    fields.append((name, endian + _PLY_SCALARS[scalar_type]))
  return np.dtype(fields)


def _scale_index(name):
  try:
    return int(name.rsplit("_", 1)[1])
  except ValueError as error:
    raise ValueError(f"invalid Gaussian scale field '{name}'") from error


def _object_id(model, object_type, name):
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise ValueError(f"MuJoCo object '{name}' was not found")
  return int(object_id)
