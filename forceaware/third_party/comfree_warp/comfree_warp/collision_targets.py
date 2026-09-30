"""Collision targets of the fixed-slot contact layout: Gaussian sphere clouds and triangle meshes.

A target is a set of components. A candidate contact with a component is described by a
*feature* in the target body frame, ``anchor = (x, y, z, radius)`` and ``axis = (x, y, z, code)``:

* ``POINT_FEATURE``: a point with a radius. A Gaussian sphere, or a mesh vertex (radius 0).
* ``EDGE_FEATURE``: the line through ``anchor`` with unit direction ``axis`` (a mesh edge).
* ``FACE_FEATURE``: the plane through ``anchor`` with outward unit normal ``axis`` (a mesh face).

``code = sign * kind``, where ``sign`` is the side of the surface the query point is on (-1 inside
a mesh, from the closest feature's angle-weighted pseudo-normal; always +1 for spheres). The
feature is the exact local geometry of the nearest-point map, so the distance it gives is the
exact signed distance and its derivatives are the exact derivatives (``contact_fusion``).

``nearest_components`` returns the ``ranks`` nearest components of a point within ``reach``: a
stackless BVH k-nearest search for spheres. A mesh has one component per point, its nearest
triangle, whose value is the exact signed distance: for a point inside a closed mesh the minimum of
per-triangle signed distances is not the signed distance, so triangles are never ranked against
each other. Contact smoothing over a mesh therefore happens across source points.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import warp as wp

SPHERE_TARGET = 0
MESH_TARGET = 1
MAX_TOPK = 8
MAX_RANKS = MAX_TOPK + 1
POINT_FEATURE = 1
EDGE_FEATURE = 2
FACE_FEATURE = 3
HULL_VERTEX_RADIUS = 1.0e-6
WELD_TOLERANCE = 1.0e-5
UNBOUNDED = 1.0e30
CACHE_DIR = Path(os.environ.get("COMFREE_MESH_CACHE", Path.home() / ".cache" / "comfree_mesh"))

vec_ranks = wp.types.vector(length=MAX_RANKS, dtype=float)
ivec_ranks = wp.types.vector(length=MAX_RANKS, dtype=int)


# -- host mesh preparation ------------------------------------------------------------------------
def read_obj(path) -> tuple[np.ndarray, np.ndarray]:
  """Vertices and fan-triangulated faces of a Wavefront OBJ file."""
  vertices, faces = [], []
  with open(path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
      if line.startswith("v "):
        vertices.append(line.split()[1:4])
      elif line.startswith("f "):
        ids = [int(token.split("/")[0]) for token in line.split()[1:]]
        ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
        faces.extend([ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1))
  if not vertices or not faces:
    raise ValueError(f"OBJ file has no triangles: {path}")
  return np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def weld_triangles(vertices, faces, tolerance=WELD_TOLERANCE):
  """Merge coincident vertices (texture seams), drop degenerate faces, require a closed mesh."""
  vertices = np.asarray(vertices, np.float64)
  keys = np.round(vertices / tolerance).astype(np.int64)
  _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
  faces = inverse.reshape(-1)[np.asarray(faces, np.int64)]
  faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                & (faces[:, 2] != faces[:, 0])]
  used, faces = np.unique(faces, return_inverse=True)
  faces = faces.reshape(-1, 3)
  vertices = vertices[first][used]
  directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
  _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
  if np.any(counts != 2):
    raise ValueError(f"mesh target is not closed after welding: "
                     f"{int(np.sum(counts != 2))} edges lack exactly two faces")
  if len(np.unique(directed, axis=0)) != len(directed):
    raise ValueError("mesh target faces are not consistently oriented")
  return vertices, faces.astype(np.int32)


def hull_vertices(vertices) -> np.ndarray:
  from scipy.spatial import ConvexHull
  return np.asarray(vertices)[ConvexHull(vertices).vertices]


def load_mesh_asset(path):
  """Welded mesh and its hull vertices, the plane-support cloud of the target."""
  vertices, faces = weld_triangles(*read_obj(path))
  return vertices, faces, hull_vertices(vertices)


def mesh_tables(vertices, faces) -> dict:
  """Face, edge and vertex pseudo-normals, cached by mesh content."""
  vertices = np.ascontiguousarray(vertices, np.float64)
  faces = np.ascontiguousarray(faces, np.int32)
  digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()
                          + b"pseudo-normals-v1").hexdigest()[:24]
  path = CACHE_DIR / f"mesh_{digest}.npz"
  if path.exists():
    with np.load(path, allow_pickle=False) as cached:
      return {key: cached[key] for key in cached.files}
  corners = vertices[faces]
  face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
  face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
  directed = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
  edges, face_edges = np.unique(np.sort(directed.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
  face_edges = face_edges.reshape(-1, 3)
  edge_normals = np.zeros((len(edges), 3))
  np.add.at(edge_normals, face_edges.ravel(), np.repeat(face_normals, 3, axis=0))
  edge_normals /= np.linalg.norm(edge_normals, axis=1, keepdims=True)
  vertex_normals = np.zeros_like(vertices)
  for k in range(3):
    u = corners[:, (k + 1) % 3] - corners[:, k]
    v = corners[:, (k + 2) % 3] - corners[:, k]
    cosine = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    np.add.at(vertex_normals, faces[:, k], np.arccos(np.clip(cosine, -1, 1))[:, None] * face_normals)
  vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)
  tables = dict(face_normals=face_normals, face_edges=face_edges.astype(np.int32),
                edge_normals=edge_normals, vertex_normals=vertex_normals,
)
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
  np.savez(temporary, **tables)
  temporary.replace(path)
  return tables


# -- device target --------------------------------------------------------------------------------
@wp.struct
class CollisionTarget:
  kind: int
  body_id: int
  lower: wp.vec3
  upper: wp.vec3
  spheres: wp.array(dtype=wp.vec4)
  bvh_lower: wp.array(dtype=wp.vec3)
  bvh_upper: wp.array(dtype=wp.vec3)
  bvh_radius: wp.array(dtype=float)
  bvh_ranges: wp.array(dtype=wp.vec2i)
  bvh_escape: wp.array(dtype=int)
  bvh_ids: wp.array(dtype=int)
  mesh: wp.uint64
  vertices: wp.array(dtype=wp.vec3)
  faces: wp.array(dtype=wp.vec3i)
  face_normals: wp.array(dtype=wp.vec3)
  face_edges: wp.array(dtype=wp.vec3i)
  edge_normals: wp.array(dtype=wp.vec3)
  vertex_normals: wp.array(dtype=wp.vec3)


class TargetResources:
  """Owns the device arrays referenced by a ``CollisionTarget``."""

  def __init__(self, model, mesh, device):
    self.target = target = CollisionTarget()
    target.body_id = int(model.target_body_id)
    target.spheres = model.target_spheres
    target.bvh_lower, target.bvh_upper = model.bvh_center_lower, model.bvh_center_upper
    target.bvh_radius, target.bvh_ranges = model.bvh_max_radius, model.bvh_ranges
    target.bvh_escape, target.bvh_ids = model.bvh_escape, model.bvh_sphere_ids
    spheres = model.target_spheres.numpy()
    if mesh is None:
      target.kind = SPHERE_TARGET
      lower = (spheres[:, :3] - spheres[:, 3:]).min(axis=0)
      upper = (spheres[:, :3] + spheres[:, 3:]).max(axis=0)
      self.arrays = ()
    else:
      vertices, faces = mesh
      tables = mesh_tables(vertices, faces)
      target.kind = MESH_TARGET
      lower, upper = vertices.min(axis=0), vertices.max(axis=0)
      floats = lambda values: wp.array(np.asarray(values, np.float32), dtype=wp.vec3, device=device)
      indices = lambda values: wp.array(np.asarray(values, np.int32), dtype=wp.vec3i, device=device)
      self.mesh = wp.Mesh(points=wp.array(np.asarray(vertices, np.float32), dtype=wp.vec3, device=device),
                          indices=wp.array(np.asarray(faces, np.int32).ravel(), dtype=int, device=device))
      target.mesh = self.mesh.id
      target.vertices = self.mesh.points
      target.faces = indices(faces)
      target.face_normals = floats(tables["face_normals"])
      target.face_edges = indices(tables["face_edges"])
      target.edge_normals = floats(tables["edge_normals"])
      target.vertex_normals = floats(tables["vertex_normals"])
      self.arrays = (target.faces, target.face_normals, target.face_edges, target.edge_normals,
                     target.vertex_normals)
    target.lower = wp.vec3(*np.asarray(lower, np.float32))
    target.upper = wp.vec3(*np.asarray(upper, np.float32))


# -- component geometry ---------------------------------------------------------------------------
@wp.func
def closest_on_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
  """Ericson's closest point; region 0 face, 1..3 edges ab, bc, ca, 4..6 vertices a, b, c."""
  ab = b - a
  ac = c - a
  ap = p - a
  d1 = wp.dot(ab, ap)
  d2 = wp.dot(ac, ap)
  if d1 <= 0.0 and d2 <= 0.0:
    return a, 4
  bp = p - b
  d3 = wp.dot(ab, bp)
  d4 = wp.dot(ac, bp)
  if d3 >= 0.0 and d4 <= d3:
    return b, 5
  vc = d1 * d4 - d3 * d2
  if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
    return a + (d1 / (d1 - d3)) * ab, 1
  cp = p - c
  d5 = wp.dot(ab, cp)
  d6 = wp.dot(ac, cp)
  if d6 >= 0.0 and d5 <= d6:
    return c, 6
  vb = d5 * d2 - d1 * d6
  if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
    return a + (d2 / (d2 - d6)) * ac, 3
  va = d3 * d6 - d5 * d4
  if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
    return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b), 2
  denom = 1.0 / (va + vb + vc)
  return a + ab * (vb * denom) + ac * (vc * denom), 0


@wp.func
def triangle_feature(target: CollisionTarget, p: wp.vec3, face: int):
  """Signed distance of ``p`` to a triangle and the feature of its closest point."""
  ids = target.faces[face]
  point, region = closest_on_triangle(
      p, target.vertices[ids[0]], target.vertices[ids[1]], target.vertices[ids[2]])
  pseudo = target.face_normals[face]
  if region >= 1 and region <= 3:
    pseudo = target.edge_normals[target.face_edges[face][region - 1]]
  elif region >= 4:
    pseudo = target.vertex_normals[ids[region - 4]]
  delta = p - point
  sign = 1.0
  code = 1
  if wp.dot(delta, pseudo) < 0.0:
    sign = -1.0
    code = -1
  anchor = point
  axis = wp.vec3(0.0)
  if region == 0:
    anchor = target.vertices[ids[0]]
    axis = target.face_normals[face]
    code = FACE_FEATURE
  elif region <= 3:
    start = ids[region - 1]
    stop = ids[region % 3]
    anchor = target.vertices[start]
    axis = wp.normalize(target.vertices[stop] - anchor)
    code *= EDGE_FEATURE
  else:
    code *= POINT_FEATURE
  return (sign * wp.length(delta), wp.vec4(anchor[0], anchor[1], anchor[2], 0.0),
          wp.vec4(axis[0], axis[1], axis[2], float(code)))


@wp.func
def component_feature(target: CollisionTarget, p: wp.vec3, component: int):
  """Feature ``(anchor, axis)`` of a component for the query point ``p``."""
  if target.kind == MESH_TARGET:
    distance, anchor, axis = triangle_feature(target, p, component)
    return anchor, axis
  return target.spheres[component], wp.vec4(0.0, 0.0, 0.0, float(POINT_FEATURE))


@wp.func
def insert_best(distances: vec_ranks, ids: ivec_ranks, ranks: int, distance: float, component: int):
  """Keep the ``ranks`` smallest (distance, id) pairs sorted."""
  last = ranks - 1
  if distance < distances[last] or (distance == distances[last] and component < ids[last]):
    slot = last
    while slot > 0 and (distance < distances[slot - 1]
                        or (distance == distances[slot - 1] and component < ids[slot - 1])):
      distances[slot] = distances[slot - 1]
      ids[slot] = ids[slot - 1]
      slot -= 1
    distances[slot] = distance
    ids[slot] = component
  return distances, ids


@wp.func
def empty_ranks(reach: float):
  distances = vec_ranks()
  ids = ivec_ranks()
  for rank in range(MAX_RANKS):
    distances[rank] = reach
    ids[rank] = -1
  return distances, ids


@wp.func
def _sphere_bound(target: CollisionTarget, p: wp.vec3, node: int) -> float:
  gap = wp.max(wp.max(target.bvh_lower[node] - p, p - target.bvh_upper[node]), wp.vec3(0.0))
  return wp.length(gap) - target.bvh_radius[node]


@wp.func
def nearest_spheres(target: CollisionTarget, p: wp.vec3, reach: float, ranks: int):
  distances, ids = empty_ranks(reach)
  node = int(0)
  while node < target.bvh_ranges.shape[0]:
    if _sphere_bound(target, p, node) > distances[ranks - 1]:
      node = target.bvh_escape[node]
    elif target.bvh_ranges[node][1] > 0:
      span = target.bvh_ranges[node]
      for offset in range(span[1]):
        sphere_id = target.bvh_ids[span[0] + offset]
        sphere = target.spheres[sphere_id]
        distance = wp.length(p - wp.vec3(sphere[0], sphere[1], sphere[2])) - sphere[3]
        distances, ids = insert_best(distances, ids, ranks, distance, sphere_id)
      node = target.bvh_escape[node]
    else:
      node += 1
  return distances, ids


@wp.func
def nearest_triangles(target: CollisionTarget, p: wp.vec3, reach: float, ranks: int):
  distances, ids = empty_ranks(reach)
  hit = wp.mesh_query_point_sign_normal(target.mesh, p, wp.min(wp.max(reach, 1.0e-6), UNBOUNDED))
  if not hit.result:
    # No surface within reach: far outside, or deep inside; the full query resolves the sign.
    hit = wp.mesh_query_point_sign_normal(target.mesh, p, UNBOUNDED)
    if not hit.result or hit.sign > 0.0:
      return distances, ids
  distance, anchor, axis = triangle_feature(target, p, hit.face)
  if distance < reach:
    distances[0] = distance
    ids[0] = hit.face
  return distances, ids


@wp.func
def nearest_components(target: CollisionTarget, p: wp.vec3, reach: float, ranks: int):
  """The ``ranks`` nearest components within ``reach`` (sorted; missing ranks have id -1)."""
  if target.kind == MESH_TARGET:
    return nearest_triangles(target, p, reach, ranks)
  return nearest_spheres(target, p, reach, ranks)


@wp.func
def outside_bounds(target: CollisionTarget, p: wp.vec3, reach: float) -> bool:
  """True when no component can be within ``reach`` (which is negative under penetration).

  The box only bounds the distance from outside: inside it the distance may be negative, so a
  point is pruned only when its gap to the box exceeds both zero and the reach.
  """
  gap = wp.length(wp.max(wp.max(target.lower - p, p - target.upper), wp.vec3(0.0)))
  return gap > wp.max(reach, 0.0)
