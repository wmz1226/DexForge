"""Fixed-order body gradients for local frames and selected contacts."""

from __future__ import annotations

import warp as wp

from comfree_warp.mujoco_warp._src import math
from comfree_warp import contact_fusion as fusion
from . import gaussian_collision as collision
from . import local_frames as frames


@wp.func
def _matrix_dot(a: wp.mat33, b: wp.mat33) -> float:
  value = float(0.0)
  for i in range(3):
    for j in range(3):
      value += a[i, j] * b[i, j]
  return value


# Scalar returns use float64 for Warp 1.15 wp.grad codegen compatibility.
@wp.func
def _frame_cost(q: wp.quat, p: wp.vec3, r: wp.quat,
                 gp: wp.vec3, gm: wp.mat33) -> wp.float64:
  matrix = math.quat_to_mat(math.mul_quat(q, r))
  return wp.float64(wp.dot(math.rot_vec_quat(p, q), gp) + _matrix_dot(matrix, gm))


@wp.kernel(enable_backward=False)
def _frame_body(params: frames.LocalFrameParameters, job: frames.LocalFrameJob,
                 adj: frames.LocalFrameJob):
  world, body = wp.tid()
  row = world % params.position.shape[0]
  position = wp.vec3(0.0)
  quaternion = wp.quat(0.0, 0.0, 0.0, 0.0)
  for frame in range(params.body_id.shape[0]):
    if params.body_id[frame] == body:
      gp, gm = adj.position[world, frame], adj.matrix[world, frame]
      gq, glp, glq, ggp, ggm = wp.grad(_frame_cost)(
          job.body_pose.quaternion[world, body], params.position[row, frame],
          params.quaternion[row, frame], gp, gm)
      position += gp
      quaternion += gq
  if adj.body_pose.position:
    adj.body_pose.position[world, body] += position
  if adj.body_pose.quaternion:
    adj.body_pose.quaternion[world, body] += quaternion


# -- pair contacts --------------------------------------------------------------------------------
@wp.struct
class PairGradients:
  """Per (world, slot * ranks + rank) candidate gradients: source centre, feature anchor and axis."""
  source: wp.array2d(dtype=wp.vec3)
  anchor: wp.array2d(dtype=wp.vec3)
  axis: wp.array2d(dtype=wp.vec3)


@wp.kernel(enable_backward=False)
def _pair_gradients(model: collision.search.GaussianCollisionModel,
                    job: collision.CollisionJob, adj: collision.CollisionJob,
                    gradients: PairGradients):
  """Candidate gradients of one slot's contact (fused, or the nearest pair for k = 1)."""
  world, enabled = wp.tid()
  row = model.target_contact_offset + enabled
  output = job.contact_offset + row
  ranks = job.selected.candidate_source.shape[2]
  c = collision.slot_candidates(model, job, world, enabled)
  offset = collision.search._distance_offset(model, world)
  source = fusion.mat_ranks3()
  anchor = fusion.mat_ranks3()
  axis = fusion.mat_ranks3()
  if model.contact_topk == 1:
    # The hard contact is the nearest pair: its seeds go straight through the pair cost. The
    # sphere cost is differentiated here, in the kernel, as the original Gaussian VJP was, so
    # distance/position paths retain the original hard Gaussian derivatives.
    sign = model.contact_normal_sign[row]
    gd = adj.contacts.distance[world, output]
    gp = adj.contacts.position[world, output]
    gf = adj.contacts.frame[world, output]
    if job.stop_frame_vjp == 1:
      gf = wp.mat33(0.0)
    if c.code[0] == fusion.POINT_FEATURE:
      gs, ga, gr, go, gsg, gfz, ggd, ggp, ggf = wp.grad(fusion.sphere_cost)(
          c.source[0], c.anchor[0], c.radius[0], offset, sign, job.freeze_frame_vjp, gd, gp, gf)
      source[0] = gs
      anchor[0] = ga
    elif c.code[0] != 0:
      gs, ga, gx, gr, gc, go, gsg, gfz, ggd, ggp, ggf = wp.grad(fusion.feature_cost)(
          c.source[0], c.anchor[0], c.axis[0], c.radius[0], c.code[0], offset, sign,
          job.freeze_frame_vjp, gd, gp, gf)
      source[0] = gs
      anchor[0] = ga
      axis[0] = gx
  else:
    source, anchor, axis = fusion.fuse_vjp(
        c, model.contact_topk, model.threshold, offset, model.contact_normal_sign[row],
        job.freeze_frame_vjp, adj.contacts.distance[world, output],
        adj.contacts.position[world, output], adj.contacts.frame[world, output])
  for rank in range(fusion.MAX_RANKS):
    if rank < ranks:
      gradients.source[world, enabled * ranks + rank] = source[rank]
      gradients.anchor[world, enabled * ranks + rank] = anchor[rank]
      gradients.axis[world, enabled * ranks + rank] = axis[rank]


@wp.kernel(enable_backward=False)
def _pair_body(model: collision.search.GaussianCollisionModel,
               job: collision.CollisionJob, adj: collision.CollisionJob,
               gradients: PairGradients):
  world, body = wp.tid()
  position = wp.vec3(0.0)
  matrix = wp.mat33(0.0)
  ranks = job.selected.candidate_source.shape[2]
  # Fixed slot-then-rank order keeps the accumulation deterministic.
  for index in range(gradients.source.shape[1]):
    enabled = index / ranks
    rank = index - enabled * ranks
    source_id = job.selected.candidate_source[world, enabled, rank]
    if source_id >= 0:
      if model.source_body_ids[source_id] == body:
        gradient = gradients.source[world, index]
        position += gradient
        matrix += wp.outer(gradient, model.source_centers[source_id])
      if model.target_body_id == body:
        gradient = gradients.anchor[world, index]
        anchor = job.selected.candidate_anchor[world, enabled, rank]
        axis = job.selected.candidate_axis[world, enabled, rank]
        position += gradient
        matrix += wp.outer(gradient, collision.search._sphere_center(anchor))
        if wp.abs(int(axis[3])) != fusion.POINT_FEATURE:
          matrix += wp.outer(gradients.axis[world, index], wp.vec3(axis[0], axis[1], axis[2]))
  if adj.state.xpos:
    adj.state.xpos[world, body] += position
  if adj.state.xmat:
    adj.state.xmat[world, body] += matrix


@wp.func
def _plane_cost(center: wp.vec3, plane: wp.vec3, normal: wp.vec3,
                  radius: float, offset: float, sign: float, frozen: int,
                  gd: float, gp: wp.vec3, gf: wp.mat33) -> wp.float64:
  position = collision.search._inset_contact_position(
      center - normal * radius, normal, offset)
  frame = collision._selected_frame(normal * sign, frozen)
  return wp.float64(gd * wp.dot(center - plane, normal) + wp.dot(gp, position)
                    + _matrix_dot(gf, frame))


@wp.kernel(enable_backward=False)
def _plane_body(model: collision.search.GaussianCollisionModel,
                  job: collision.CollisionJob, adj: collision.CollisionJob):
  world = wp.tid()
  position, plane_position, normal_gradient = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
  matrix = wp.mat33(0.0)
  body = model.target_body_id
  geom = model.plane_geom_id
  plane_matrix = job.state.geom_xmat[world, geom]
  normal = wp.vec3(plane_matrix[0, 2], plane_matrix[1, 2], plane_matrix[2, 2])
  for row in range(collision.PLANE_CONTACTS):
    target_id = job.selected.target_id[world, row]
    if target_id >= 0:
      sphere = model.target_spheres[target_id]
      local = collision.search._sphere_center(sphere)
      center = job.state.xpos[world, body] + job.state.xmat[world, body] @ local
      output = job.contact_offset + row
      gf = adj.contacts.frame[world, output]
      if job.stop_frame_vjp == 1:
        gf = wp.mat33(0.0)
      gc, gpl, gn, gr, go, gs, gfr, ggd, ggp, ggf = wp.grad(_plane_cost)(
          center, job.state.geom_xpos[world, geom], normal, sphere[3],
          collision.search._distance_offset(model, world), model.contact_normal_sign[row],
          job.freeze_frame_vjp, adj.contacts.distance[world, output],
          adj.contacts.position[world, output], gf)
      position += gc
      matrix += wp.outer(gc, local)
      plane_position += gpl
      normal_gradient += gn
  if adj.state.xpos:
    adj.state.xpos[world, body] += position
  if adj.state.xmat:
    adj.state.xmat[world, body] += matrix
  if adj.state.geom_xpos:
    adj.state.geom_xpos[world, geom] += plane_position
  if adj.state.geom_xmat:
    adj.state.geom_xmat[world, geom] += wp.outer(normal_gradient, wp.vec3(0.0, 0.0, 1.0))


def detached_struct(value, paths):
  """Hide selected primal .grad pointers without copying or changing values."""
  result = value._cls()
  for name in value._cls.vars:
    item = getattr(value, name)
    tails = [path.split('.', 1)[1] for path in paths if path.startswith(name + '.')]
    if tails:
      item = detached_struct(item, tails)
    elif name in paths:
      item = wp.array(ptr=item.ptr, dtype=item.dtype, shape=item.shape,
                      strides=item.strides, device=item.device, requires_grad=False)
    setattr(result, name, item)
  return result


def has_gradient(value):
  if isinstance(value, wp.array):
    return bool(value.ptr)
  if hasattr(value, '_cls'):
    return any(has_gradient(getattr(value, name)) for name in value._cls.vars)
  return False


class GeometryTape(wp.Tape):
  def record_launch(self, kernel, dim, max_blocks, inputs, outputs, device,
                    block_dim=0, metadata=None):
    from . import continuous_contact
    if kernel is continuous_contact.fuse:
      continuous_contact.record_backward(self, inputs, device)
      return
    if kernel not in (frames._local_frames, collision._narrow_pairs, collision._narrow_plane):
      super().record_launch(kernel, dim, max_blocks, inputs, outputs, device,
                            block_dim=block_dim, metadata=metadata)
      return
    model, job = inputs
    model_adj = self.get_adjoint(model)
    adj = self.get_adjoint(job)
    frame_kernel = kernel is frames._local_frames
    paths = (['body_pose.position', 'body_pose.quaternion'] if frame_kernel else
             ['state.xpos', 'state.xmat', 'state.geom_xpos', 'state.geom_xmat'])
    detached = detached_struct(job, paths)
    detached_adj = self.get_adjoint(detached)
    parameter_gradients = has_gradient(model_adj)
    if kernel is collision._narrow_pairs:
      shape = (job.state.xpos.shape[0], dim[0])
      columns = (shape[0], dim[0] * job.selected.candidate_source.shape[2])
      gradients = PairGradients()
      gradients.source = wp.empty(columns, dtype=wp.vec3, device=device)
      gradients.anchor = wp.empty(columns, dtype=wp.vec3, device=device)
      gradients.axis = wp.empty(columns, dtype=wp.vec3, device=device)

    def backward_geometry():
      if frame_kernel:
        wp.launch(_frame_body, dim=job.body_pose.position.shape,
                  inputs=[model, job, adj], device=device)
      elif kernel is collision._narrow_pairs:
        wp.launch(_pair_gradients, dim=shape, inputs=[model, job, adj, gradients], device=device)
        wp.launch(_pair_body, dim=job.state.xpos.shape, inputs=[model, job, adj, gradients],
                  device=device)
      else:
        wp.launch(_plane_body, dim=job.state.xpos.shape[0],
                  inputs=[model, job, adj], device=device)
      if parameter_gradients:
        # The native adjoint consumes output seeds, so run it after the gathers.
        wp.launch(kernel, dim=dim, inputs=[model, detached], outputs=outputs,
                  adj_inputs=[model_adj, detached_adj], adj_outputs=[],
                  adjoint=True, device=device, max_blocks=max_blocks, block_dim=block_dim)

    self.launches.append(backward_geometry)
