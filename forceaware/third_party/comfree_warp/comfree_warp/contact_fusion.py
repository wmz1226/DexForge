"""Contact of a source sphere with a target feature, top-k fusion of ranked candidates, and VJPs.

A candidate is a (source sphere, target feature) pair (``collision_targets``: a point with a
radius, a line or a plane, carried with the query's side of the surface; a Gaussian is a point
feature with its radius). Its distance, contact
position on the target and inward normal are closed-form in the source centre and the feature,
so the pair derivatives are exact.

Candidates are sorted by distance; rank ``topk`` is the boundary (``fallback`` when missing).
Linear, scale-free weights ``a_i = max(d_b - d_i, 0)`` blend the distance, the contact position and
the inward normal, which is then normalized. A candidate enters or leaves the top k with zero
weight, so the fused contact is continuous, and the fused distance is a weighted mean: it is never
deeper than the candidates. When at most one candidate carries weight (always for ``topk = 1``,
the hard minimum) or the blended normal degenerates, the nearest pair is returned exactly; the
switch is continuous because the other weights vanish there.

Seeds on the fused distance, position and normal are distributed to every candidate and to the
boundary analytically, then through each pair with ``wp.grad``. Loops have static bounds.
"""

import warp as wp
from types import SimpleNamespace

from comfree_warp.collision_targets import (EDGE_FEATURE, FACE_FEATURE, MAX_RANKS, MAX_TOPK,
                                            POINT_FEATURE, ivec_ranks)
from comfree_warp.comfree_core._src import collision_gaussian as search

NORMAL_EPSILON = 1.0e-9
WEIGHT_EPSILON = 1.0e-12
DEGENERATE_NORMAL = 1.0e-6

def _make_fusion(scalar):
  suffix = scalar.__name__
  vector = wp.types.vector(length=3, dtype=scalar)
  matrix = wp.types.matrix(shape=(3, 3), dtype=scalar)
  vec_ranks = wp.types.vector(length=MAX_RANKS, dtype=scalar)
  mat_ranks3 = wp.types.matrix(shape=(MAX_RANKS, 3), dtype=scalar)


  @wp.func(name="unit_" + suffix)
  def unit(vector: vector) -> vector:
    return vector / wp.sqrt(wp.dot(vector, vector) + scalar(NORMAL_EPSILON) * scalar(NORMAL_EPSILON))


  # Contact frame: normal plus a fixed tangent basis; the frozen variant's VJP drops the basis spin.
  @wp.func(name="contact_frame_" + suffix)
  def contact_frame(normal: vector) -> matrix:
    normal = unit(normal)
    tangent_x = unit(wp.cross(normal, vector(scalar(1.0), scalar(1.0), scalar(1.0))))
    tangent_y = unit(wp.cross(normal, tangent_x))
    return matrix(normal[0], normal[1], normal[2],
                    tangent_x[0], tangent_x[1], tangent_x[2],
                    tangent_y[0], tangent_y[1], tangent_y[2])


  @wp.func(name="_frozen_frame_" + suffix)
  def _frozen_frame(normal: vector) -> matrix:
    return contact_frame(normal)


  @wp.func_grad(_frozen_frame)
  def _adj_frozen_frame(normal: vector, adj_ret: matrix):
    # Preserve the physical normal derivative while parallel-transporting the
    # tangent basis, removing only its arbitrary spin about the normal.
    squared_norm = wp.dot(normal, normal) + scalar(NORMAL_EPSILON) * scalar(NORMAL_EPSILON)
    inverse_norm = scalar(1.0) / wp.sqrt(squared_norm)
    unit_normal = normal * inverse_norm
    frame = contact_frame(normal)
    tangent_x = vector(frame[1, 0], frame[1, 1], frame[1, 2])
    tangent_y = vector(frame[2, 0], frame[2, 1], frame[2, 2])
    normal_seed = vector(adj_ret[0, 0], adj_ret[0, 1], adj_ret[0, 2])
    tangent_x_seed = vector(adj_ret[1, 0], adj_ret[1, 1], adj_ret[1, 2])
    tangent_y_seed = vector(adj_ret[2, 0], adj_ret[2, 1], adj_ret[2, 2])
    rotation_seed = (
      wp.cross(tangent_x, tangent_x_seed)
      + wp.cross(tangent_y, tangent_y_seed))
    seed = normal_seed + wp.cross(rotation_seed, unit_normal)
    projected = seed * inverse_norm
    projected -= normal * (
      wp.dot(normal, seed) * inverse_norm / squared_norm)
    wp.adjoint[normal] += projected


  @wp.func(name="selected_frame_" + suffix)
  def selected_frame(normal: vector, freeze_vjp: int) -> matrix:
    if freeze_vjp == 1:
      return _frozen_frame(normal)
    return contact_frame(normal)



  @wp.func(name="_stopped_frame_" + suffix)
  def _stopped_frame(normal: vector) -> matrix:
    return contact_frame(normal)


  @wp.func_grad(_stopped_frame)
  def _stopped_frame_grad(normal: vector, adj_ret: matrix):
    wp.adjoint[normal] += vector(scalar(0.0))


  @wp.func(name="configured_frame_" + suffix)
  def configured_frame(normal: vector, freeze_vjp: int, stop_vjp: int) -> matrix:
    """Detach frame outputs only; the distance and position paths remain intact."""
    if stop_vjp == 1:
      return _stopped_frame(normal)
    return selected_frame(normal, freeze_vjp)


  @wp.func(name="sphere_contact_" + suffix)
  def sphere_contact(source: vector, center: vector, source_radius: scalar, radius: scalar,
                     offset: scalar):
    """Sphere pair (the original Gaussian expressions): distance, position on the target, inward normal."""
    normal = unit(center - source)
    distance = wp.length(center - source) - source_radius - radius + offset
    position = search._inset_contact_position(center - normal * radius, normal, offset)
    return distance, position, normal


  @wp.func(name="feature_contact_" + suffix)
  def feature_contact(source: vector, anchor: vector, axis: vector, source_radius: scalar,
                      radius: scalar, code: int, offset: scalar):
    """Source sphere and a mesh feature: exact signed distance, closest point and inward normal.

    The closest point is ``c = a + P (s - a)`` (``P`` = 0 for a point, ``e e^T`` for a line); a
    plane uses its normal directly, so its distance stays linear through the surface.
    """
    kind = wp.abs(code)
    side = wp.where(code < 0, -scalar(1.0), scalar(1.0))
    closest = anchor
    if kind == EDGE_FEATURE:
      closest = anchor + axis * wp.dot(axis, source - anchor)
    normal = unit(closest - source) * side
    distance = side * wp.length(closest - source)
    if kind == FACE_FEATURE:
      height = wp.dot(axis, source - anchor)
      closest = source - axis * height
      normal = -axis
      distance = height
    distance = distance - source_radius - radius + offset
    position = search._inset_contact_position(closest - normal * radius, normal, offset)
    return distance, position, normal


  @wp.func(name="pair_contact_" + suffix)
  def pair_contact(source: vector, anchor: vector, axis: vector, source_radius: scalar,
                   radius: scalar, code: int, offset: scalar):
    """Contact of a source sphere and a feature; a Gaussian (``code = POINT_FEATURE``) is a sphere.

    Forward only: gradients go through ``sphere_cost``, ``feature_cost`` and ``normal_cost``, which
    avoid reassigning a returned tuple inside a branch (its Warp adjoint is wrong on the CPU).
    """
    distance, position, normal = feature_contact(source, anchor, axis, source_radius, radius, code, offset)
    if code == POINT_FEATURE:
      distance, position, normal = sphere_contact(source, anchor, source_radius, radius, offset)
    return distance, position, normal


  @wp.func(name="matrix_dot_" + suffix)
  def matrix_dot(a: matrix, b: matrix) -> scalar:
    value = scalar(0.0)
    for i in range(3):
      for j in range(3):
        value += a[i, j] * b[i, j]
    return value


  @wp.func(name="sphere_cost_" + suffix)
  def sphere_cost(source: vector, target: vector, radius: scalar, offset: scalar,
                  sign: scalar, frozen: int, gd: scalar, gp: vector,
                  gf: matrix) -> wp.float64:
    """Seeded sphere-pair outputs, written as the original sphere-pair VJP cost."""
    difference = target - source
    normal = unit(difference)
    position = search._inset_contact_position(
        target - normal * radius, normal, offset)
    frame = selected_frame(normal * sign, frozen)
    return wp.float64(gd * wp.length(difference) + wp.dot(gp, position)
                      + matrix_dot(gf, frame))


  @wp.func(name="feature_cost_" + suffix)
  def feature_cost(source: vector, anchor: vector, axis: vector, radius: scalar, code: int,
                   offset: scalar, sign: scalar, frozen: int, gd: scalar, gp: vector,
                   gf: matrix) -> wp.float64:
    """Seeded mesh-feature pair outputs: distance, contact position and the frame of ``sign * n``."""
    distance, position, normal = feature_contact(source, anchor, axis, scalar(0.0), radius, code, offset)
    frame = selected_frame(normal * sign, frozen)
    return wp.float64(gd * distance + wp.dot(gp, position) + matrix_dot(gf, frame))


  @wp.func(name="pair_vjp_" + suffix)
  def pair_vjp(source: vector, anchor: vector, axis: vector, radius: scalar, code: int,
               offset: scalar, sign: scalar, frozen: int, gd: scalar, gp: vector, gf: matrix):
    """Source, anchor and axis gradients of the seeded pair outputs, per feature kind."""
    gs = vector(scalar(0.0))
    ga = vector(scalar(0.0))
    gx = vector(scalar(0.0))
    if code == POINT_FEATURE:
      gs, ga, gr, go, gsg, gfz, ggd, ggp, ggf = wp.grad(sphere_cost)(
          source, anchor, radius, offset, sign, frozen, gd, gp, gf)
    else:
      gs, ga, gx, gr, gc, go, gsg, gfz, ggd, ggp, ggf = wp.grad(feature_cost)(
          source, anchor, axis, radius, code, offset, sign, frozen, gd, gp, gf)
    return gs, ga, gx


  @wp.func(name="normal_cost_" + suffix)
  def normal_cost(source: vector, anchor: vector, axis: vector, code: int, gn: vector) -> wp.float64:
    """Seeded inward pair normal, for blended contacts."""
    distance, position, normal = feature_contact(source, anchor, axis, scalar(0.0), scalar(0.0), code, scalar(0.0))
    return wp.float64(wp.dot(gn, normal))


  @wp.struct
  class Candidates:
    """A slot's ranked candidates in world coordinates; ``code`` 0 marks a missing rank."""
    source: mat_ranks3
    anchor: mat_ranks3
    axis: mat_ranks3
    source_radius: vec_ranks
    radius: vec_ranks
    code: ivec_ranks


  @wp.func(name="candidate_contact_" + suffix)
  def candidate_contact(c: Candidates, rank: int, offset: scalar):
    return pair_contact(c.source[rank], c.anchor[rank], c.axis[rank], c.source_radius[rank],
                        c.radius[rank], c.code[rank], offset)


  @wp.func(name="boundary_distance_" + suffix)
  def boundary_distance(c: Candidates, topk: int, fallback: scalar, offset: scalar):
    distance = fallback
    present = int(0)
    for rank in range(MAX_RANKS):
      if rank == topk and c.code[rank] != 0:
        distance, position, normal = candidate_contact(c, rank, offset)
        present = 1
    return present, distance


  @wp.func(name="_blend_" + suffix)
  def _blend(c: Candidates, topk: int, boundary: scalar, offset: scalar):
    """Weight sum, weighted sums, candidate count, weighted-candidate count and nearest pair."""
    count = int(0)
    weighted = int(0)
    total = scalar(0.0)
    distance_sum = scalar(0.0)
    position_sum = vector(scalar(0.0))
    normal_sum = vector(scalar(0.0))
    nearest_distance = scalar(0.0)
    nearest_position = vector(scalar(0.0))
    nearest_normal = vector(scalar(0.0))
    for rank in range(MAX_TOPK):
      if rank < topk and c.code[rank] != 0:
        distance, position, normal = candidate_contact(c, rank, offset)
        if count == 0:
          nearest_distance = distance
          nearest_position = position
          nearest_normal = normal
        weight = wp.max(boundary - distance, scalar(0.0))
        if weight > scalar(0.0):
          weighted += 1
        total += weight
        distance_sum += weight * distance
        position_sum += weight * position
        normal_sum += weight * normal
        count += 1
    return (count, weighted, total, distance_sum, position_sum, normal_sum,
            nearest_distance, nearest_position, nearest_normal)


  @wp.func(name="_smooth_" + suffix)
  def _smooth(weighted: int, total: scalar, normal_sum: vector) -> bool:
    return weighted > 1 and total > scalar(WEIGHT_EPSILON) and wp.length(normal_sum) > scalar(DEGENERATE_NORMAL) * total


  @wp.func(name="fuse_" + suffix)
  def fuse(c: Candidates, topk: int, fallback: scalar, offset: scalar):
    """Fused (count, distance, position, unit inward normal) of the top-k candidates."""
    present, boundary = boundary_distance(c, topk, fallback, offset)
    count, weighted, total, distance_sum, position_sum, normal_sum, d0, p0, n0 = _blend(
        c, topk, boundary, offset)
    if _smooth(weighted, total, normal_sum):
      return count, distance_sum / total, position_sum / total, unit(normal_sum)
    return count, d0, p0, n0


  @wp.func(name="_frame_cost_" + suffix)
  def _frame_cost(normal: vector, sign: scalar, frozen: int, gf: matrix) -> wp.float64:
    return wp.float64(matrix_dot(gf, selected_frame(normal * sign, frozen)))


  @wp.func(name="fuse_vjp_" + suffix)
  def fuse_vjp(c: Candidates, topk: int, fallback: scalar, offset: scalar, sign: scalar, frozen: int,
               gd: scalar, gp: vector, gf: matrix):
    """Gradients of ``gd.D + gp.P + gf.F(sign N)`` for every candidate's source, anchor and axis.

    When the nearest pair is returned exactly, its seeds pass straight through ``pair_vjp``.
    """
    present, boundary = boundary_distance(c, topk, fallback, offset)
    count, weighted, total, distance_sum, position_sum, normal_sum, d0, p0, n0 = _blend(
        c, topk, boundary, offset)
    smooth = _smooth(weighted, total, normal_sum)
    fused = scalar(0.0)
    position = vector(scalar(0.0))
    blended = vector(scalar(0.0))
    gb = vector(scalar(0.0))
    if smooth:
      fused = distance_sum / total
      position = position_sum / total
      blended = normal_sum / total
      norm = wp.length(blended)
      normal = blended / norm
      seed, gsign, gfrozen, ggf = wp.grad(_frame_cost)(normal, sign, frozen, gf)
      # Seed on the unnormalized blend through N = unit(blended).
      gb = (seed - normal * wp.dot(normal, seed)) / norm
    source_gradient = mat_ranks3()
    anchor_gradient = mat_ranks3()
    axis_gradient = mat_ranks3()
    boundary_seed = scalar(0.0)
    seen = int(0)
    zero_frame = matrix(scalar(0.0))
    for rank in range(MAX_RANKS):
      seed_distance = scalar(0.0)
      seed_position = vector(scalar(0.0))
      seed_normal = vector(scalar(0.0))
      seed_frame = zero_frame
      active = False
      if rank < topk and c.code[rank] != 0 and count > 0:
        distance, point, direction = candidate_contact(c, rank, offset)
        weight = wp.max(boundary - distance, scalar(0.0))
        if smooth:
          if weight > scalar(0.0):
            active = True
            spread = gd * (distance - fused) + wp.dot(gp, point - position) + wp.dot(gb, direction - blended)
            seed_distance = (gd * weight - spread) / total
            seed_position = gp * (weight / total)
            seed_normal = gb * (weight / total)
            boundary_seed += spread / total
        elif seen == 0:
          active = True
          seed_distance = gd
          seed_position = gp
          seed_frame = gf
        seen += 1
      elif rank == topk and present == 1 and smooth:
        active = True
        seed_distance = boundary_seed
      if active:
        gs, ga, gx = pair_vjp(c.source[rank], c.anchor[rank], c.axis[rank], c.radius[rank],
                              c.code[rank], offset, sign, frozen, seed_distance, seed_position,
                              seed_frame)
        if smooth:
          ns, na, nx, nc, ngn = wp.grad(normal_cost)(
              c.source[rank], c.anchor[rank], c.axis[rank], c.code[rank], seed_normal)
          gs += ns
          ga += na
          gx += nx
        source_gradient[rank] = gs
        anchor_gradient[rank] = ga
        axis_gradient[rank] = gx
    return source_gradient, anchor_gradient, axis_gradient

  return SimpleNamespace(mat_ranks3=mat_ranks3, vec_ranks=vec_ranks, Candidates=Candidates, unit=unit, contact_frame=contact_frame, _frozen_frame=_frozen_frame, _adj_frozen_frame=_adj_frozen_frame, selected_frame=selected_frame, _stopped_frame=_stopped_frame, _stopped_frame_grad=_stopped_frame_grad, configured_frame=configured_frame, sphere_contact=sphere_contact, feature_contact=feature_contact, pair_contact=pair_contact, matrix_dot=matrix_dot, sphere_cost=sphere_cost, feature_cost=feature_cost, pair_vjp=pair_vjp, normal_cost=normal_cost, candidate_contact=candidate_contact, boundary_distance=boundary_distance, _blend=_blend, _smooth=_smooth, fuse=fuse, _frame_cost=_frame_cost, fuse_vjp=fuse_vjp)

# Geometry solvers need double precision, while simulation retains its original
# float32 arithmetic. Both specializations are built from these same functions.
standard = _make_fusion(wp.float32)
precise = _make_fusion(wp.float64)
for _name, _value in vars(standard).items():
  globals()[_name] = _value
