"""Fixed-layout MuJoCo contact and scalar joint-limit rows."""

from __future__ import annotations

from dataclasses import dataclass
import os

import mujoco
import numpy as np
import warp as wp

from comfree_warp.comfree_core._src import support as production_support
from comfree_warp.mujoco_warp._src.types import vec5
from .gaussian_collision import FixedContacts


FREE = int(mujoco.mjtJoint.mjJNT_FREE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SUPPORTED_CONTACT_DIMS = frozenset((3, 6))
ROW_TILE_SIZE = 32
CONTACT_DISABLE_MASK = int(
    mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    | mujoco.mjtDisableBit.mjDSBL_CONTACT)
LIMIT_DISABLE_MASK = int(
    mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    | mujoco.mjtDisableBit.mjDSBL_LIMIT)


@wp.struct
class ContactParameters:
    body_dof_mask: wp.array2d(dtype=int)
    dof_jntid: wp.array(dtype=int)
    jnt_type: wp.array(dtype=int)
    jnt_dofadr: wp.array(dtype=int)
    jnt_bodyid: wp.array(dtype=int)
    geom_bodyid: wp.array(dtype=int)
    contact_geom: wp.array(dtype=wp.vec2i)
    contact_includemargin: wp.array(dtype=float)
    contact_friction: wp.array(dtype=vec5)
    friction_scale: wp.array(dtype=float)
    row_contact: wp.array(dtype=int)
    row_edge: wp.array(dtype=int)
    limit_joint: wp.array(dtype=int)
    jnt_qposadr: wp.array(dtype=int)
    jnt_dofadr: wp.array(dtype=int)
    jnt_range: wp.array2d(dtype=wp.vec2)
    jnt_margin: wp.array2d(dtype=float)
    contact_row_count: int
    limit_count: int
    dof_count: int


@wp.struct
class ContactMotion:
    joint_anchor: wp.array2d(dtype=wp.vec3)
    joint_axis: wp.array2d(dtype=wp.vec3)
    body_matrix: wp.array2d(dtype=wp.mat33)


@wp.struct
class ConstraintRows:
    jacobian: wp.array3d(dtype=float)
    position: wp.array2d(dtype=float)
    active: wp.array2d(dtype=int)


@wp.struct
class StaticSparseLayout:
    row_offsets: wp.array(dtype=int)
    row_nonzero_count: wp.array(dtype=int)
    column_indices: wp.array(dtype=int)
    entry_rows: wp.array(dtype=int)
    column_offsets: wp.array(dtype=int)
    csc_positions: wp.array(dtype=int)
    csc_rows: wp.array(dtype=int)
    row_count: int
    row_capacity: int
    nonzero_count: int
    dof_count: int


@wp.struct
class SparseConstraintRows:
    jacobian: wp.array2d(dtype=float)
    position: wp.array2d(dtype=float)
    active: wp.array2d(dtype=int)


@wp.struct
class ContactRowJob:
    contacts: FixedContacts
    motion: ContactMotion
    rows: ConstraintRows


@wp.struct
class SparseContactRowJob:
    contacts: FixedContacts
    motion: ContactMotion
    rows: SparseConstraintRows
    qpos: wp.array2d(dtype=float)


@wp.struct
class LimitRowJob:
    rows: ConstraintRows
    qpos: wp.array2d(dtype=float)


@wp.struct
class PointDofQuery:
    point: wp.vec3
    body: int
    dof: int
    world: int


@dataclass(frozen=True)
class CompiledContactRows:
    parameters: ContactParameters
    contact_count: int
    contact_row_count: int
    limit_count: int
    row_count: int
    row_capacity: int
    sparse: bool
    sparse_layout: object | None
    device: object


@dataclass(frozen=True)
class ContactRowWorkspace:
    rows: object


@dataclass(frozen=True)
class ContactRowBuildInput:
    contacts: FixedContacts
    motion: ContactMotion
    qpos: wp.array


@dataclass(frozen=True)
class _HostSparseLayout:
    row_offsets: np.ndarray
    row_nonzero_count: np.ndarray
    column_indices: np.ndarray
    entry_rows: np.ndarray
    column_offsets: np.ndarray
    csc_positions: np.ndarray
    csc_rows: np.ndarray
    row_count: int
    row_capacity: int
    dof_count: int


@dataclass(frozen=True)
class _HostRowMapping:
    edge_count: np.ndarray
    contact: np.ndarray
    edge: np.ndarray
    limit_joint: np.ndarray


@wp.func
def _basis(index: int) -> wp.vec3:
    if index == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if index == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def _matrix_axis(matrix: wp.mat33, index: int) -> wp.vec3:
    return wp.vec3(matrix[0, index], matrix[1, index], matrix[2, index])


@wp.func
def _point_velocity(params: ContactParameters, motion: ContactMotion,
                    query: PointDofQuery) -> wp.vec3:
    if params.body_dof_mask[query.body, query.dof] == 0:
        return wp.vec3(0.0)
    joint = params.dof_jntid[query.dof]
    joint_type = params.jnt_type[joint]
    offset = query.dof - params.jnt_dofadr[joint]
    if joint_type == SLIDE:
        return motion.joint_axis[query.world, joint]
    if joint_type == HINGE:
        axis = motion.joint_axis[query.world, joint]
    elif joint_type == FREE and offset < 3:
        return _basis(offset)
    else:
        joint_body = params.jnt_bodyid[joint]
        axis = _matrix_axis(motion.body_matrix[query.world, joint_body],
                            offset - wp.where(joint_type == FREE, 3, 0))
    anchor = motion.joint_anchor[query.world, joint]
    return wp.cross(axis, query.point - anchor)


@wp.func
def _angular_velocity(params: ContactParameters, motion: ContactMotion,
                      query: PointDofQuery) -> wp.vec3:
    if params.body_dof_mask[query.body, query.dof] == 0:
        return wp.vec3(0.0)
    joint = params.dof_jntid[query.dof]
    joint_type = params.jnt_type[joint]
    offset = query.dof - params.jnt_dofadr[joint]
    if joint_type == SLIDE or (joint_type == FREE and offset < 3):
        return wp.vec3(0.0)
    if joint_type == HINGE:
        return motion.joint_axis[query.world, joint]
    joint_body = params.jnt_bodyid[joint]
    axis_offset = offset - wp.where(joint_type == FREE, 3, 0)
    return _matrix_axis(
        motion.body_matrix[query.world, joint_body], axis_offset)


@wp.func
def _frame_row(frame: wp.mat33, row: int) -> wp.vec3:
    return wp.vec3(frame[row, 0], frame[row, 1], frame[row, 2])


@wp.func
def _penetrating(contacts: FixedContacts, params: ContactParameters,
                 world: int, contact: int) -> bool:
    return (contacts.active[world, contact] != 0
            and contacts.distance[world, contact]
            < params.contact_includemargin[contact])


@wp.func
def _constraint_position(contacts: FixedContacts, params: ContactParameters,
                         world: int, contact: int) -> float:
    return (contacts.distance[world, contact]
            - params.contact_includemargin[contact])


@wp.func
def _edge_jacobian(contacts: FixedContacts, params: ContactParameters,
                   index: wp.vec3i, linear: wp.vec3,
                   angular: wp.vec3) -> float:
    world = index[0]
    contact = index[1]
    edge = index[2]
    frame = contacts.frame[world, contact]
    friction_axis = edge // 2 + 1
    sign = wp.where(edge % 2 == 0, 1.0, -1.0)
    friction = params.contact_friction[contact][friction_axis - 1]
    friction *= params.friction_scale[world % params.friction_scale.shape[0]]
    normal = wp.dot(_frame_row(frame, 0), linear)
    if friction_axis < 3:
        tangent = wp.dot(_frame_row(frame, friction_axis), linear)
    else:
        tangent = wp.dot(_frame_row(frame, friction_axis - 3), angular)
    return normal + sign * friction * tangent


@wp.kernel
def _build_jacobian(params: ContactParameters, job: ContactRowJob):
    world, row, dof = wp.tid()
    contact = params.row_contact[row]
    edge = params.row_edge[row]
    penetrating = _penetrating(job.contacts, params, world, contact)
    if not penetrating:
        job.rows.jacobian[world, row, dof] = 0.0
        return
    geoms = params.contact_geom[contact]
    body1 = params.geom_bodyid[geoms[0]]
    body2 = params.geom_bodyid[geoms[1]]
    point = job.contacts.position[world, contact]
    query1 = PointDofQuery()
    query1.point = point
    query1.body = body1
    query1.dof = dof
    query1.world = world
    query2 = PointDofQuery()
    query2.point = point
    query2.body = body2
    query2.dof = dof
    query2.world = world
    velocity1 = _point_velocity(params, job.motion, query1)
    velocity2 = _point_velocity(params, job.motion, query2)
    linear = velocity2 - velocity1
    angular = (_angular_velocity(params, job.motion, query2)
               - _angular_velocity(params, job.motion, query1))
    job.rows.jacobian[world, row, dof] = _edge_jacobian(
        job.contacts, params, wp.vec3i(world, contact, edge), linear, angular)


@wp.kernel
def _build_position(params: ContactParameters, job: ContactRowJob):
    world, row = wp.tid()
    contact = params.row_contact[row]
    penetrating = _penetrating(job.contacts, params, world, contact)
    job.rows.position[world, row] = _constraint_position(
        job.contacts, params, world, contact)
    job.rows.active[world, row] = wp.where(penetrating, 1, 0)


@wp.kernel
def _build_limit_position(params: ContactParameters, job: LimitRowJob):
    world, limit = wp.tid()
    row = params.contact_row_count + limit
    joint = params.limit_joint[limit]
    qpos = job.qpos[world, params.jnt_qposadr[joint]]
    model = world % params.jnt_range.shape[0]
    interval = params.jnt_range[model, joint]
    margin = params.jnt_margin[
        world % params.jnt_margin.shape[0], joint]
    position = wp.min(qpos - interval[0], interval[1] - qpos) - margin
    job.rows.position[world, row] = position
    job.rows.active[world, row] = wp.where(position < 0.0, 1, 0)


@wp.kernel
def _build_limit_jacobian(params: ContactParameters, job: LimitRowJob):
    world, limit, dof = wp.tid()
    row = params.contact_row_count + limit
    joint = params.limit_joint[limit]
    qpos = job.qpos[world, params.jnt_qposadr[joint]]
    interval = params.jnt_range[
        world % params.jnt_range.shape[0], joint]
    sign = wp.where(qpos - interval[0] < interval[1] - qpos, 1.0, -1.0)
    active = job.rows.active[world, row] != 0
    target = params.jnt_dofadr[joint]
    job.rows.jacobian[world, row, dof] = wp.where(
        active and dof == target, sign, 0.0)


@wp.kernel
def _build_sparse_jacobian(params: ContactParameters,
                           layout: StaticSparseLayout,
                           job: SparseContactRowJob):
    world, position = wp.tid()
    row = layout.entry_rows[position]
    if row >= params.contact_row_count:
        limit = row - params.contact_row_count
        joint = params.limit_joint[limit]
        qpos = job.qpos[world, params.jnt_qposadr[joint]]
        interval = params.jnt_range[
            world % params.jnt_range.shape[0], joint]
        sign = wp.where(
            qpos - interval[0] < interval[1] - qpos, 1.0, -1.0)
        dof = layout.column_indices[position]
        target = params.jnt_dofadr[joint]
        job.rows.jacobian[world, position] = wp.where(
            job.rows.active[world, row] != 0 and dof == target, sign, 0.0)
        return
    contact = params.row_contact[row]
    edge = params.row_edge[row]
    dof = layout.column_indices[position]
    penetrating = _penetrating(job.contacts, params, world, contact)
    if not penetrating:
        job.rows.jacobian[world, position] = 0.0
        return
    geoms = params.contact_geom[contact]
    point = job.contacts.position[world, contact]
    query1 = PointDofQuery()
    query1.point = point
    query1.body = params.geom_bodyid[geoms[0]]
    query1.dof = dof
    query1.world = world
    query2 = PointDofQuery()
    query2.point = point
    query2.body = params.geom_bodyid[geoms[1]]
    query2.dof = dof
    query2.world = world
    linear = (_point_velocity(params, job.motion, query2)
              - _point_velocity(params, job.motion, query1))
    angular = (_angular_velocity(params, job.motion, query2)
               - _angular_velocity(params, job.motion, query1))
    job.rows.jacobian[world, position] = _edge_jacobian(
        job.contacts, params, wp.vec3i(world, contact, edge), linear, angular)


@wp.kernel
def _build_sparse_position(params: ContactParameters,
                           job: SparseContactRowJob):
    world, row = wp.tid()
    if row >= params.contact_row_count:
        limit = row - params.contact_row_count
        joint = params.limit_joint[limit]
        qpos = job.qpos[world, params.jnt_qposadr[joint]]
        interval = params.jnt_range[
            world % params.jnt_range.shape[0], joint]
        margin = params.jnt_margin[
            world % params.jnt_margin.shape[0], joint]
        position = wp.min(qpos - interval[0], interval[1] - qpos) - margin
        job.rows.position[world, row] = position
        job.rows.active[world, row] = wp.where(position < 0.0, 1, 0)
        return
    contact = params.row_contact[row]
    penetrating = _penetrating(job.contacts, params, world, contact)
    job.rows.position[world, row] = _constraint_position(
        job.contacts, params, world, contact)
    job.rows.active[world, row] = wp.where(penetrating, 1, 0)


def _ancestors(model: mujoco.MjModel, body: int) -> set[int]:
    result: set[int] = set()
    current = body
    while current not in result:
        result.add(current)
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return result


def _body_dof_mask(model: mujoco.MjModel) -> np.ndarray:
    mask = np.zeros((model.nbody, model.nv), np.int32)
    for body in range(model.nbody):
        ancestors = _ancestors(model, body)
        mask[body] = np.isin(model.dof_bodyid, list(ancestors))
    return mask


def _linear_solver_request() -> str:
    value = os.environ.get(
        production_support.LINEAR_SOLVER_ENV,
        production_support.DEFAULT_LINEAR_SOLVER).strip().lower()
    return production_support.validate_linear_solver(value)


def _use_matrix_free_solver() -> bool:
    request = _linear_solver_request()
    if production_support.L == 0:
        return False
    return request == production_support.LINEAR_SOLVER_PCG


def _host_row_mapping(
        dimensions: np.ndarray,
        limit_joint: np.ndarray | None = None) -> _HostRowMapping:
    if limit_joint is None:
        limit_joint = np.zeros(0, dtype=np.int32)
    edge_count = (2 * (dimensions - 1)).astype(np.int32)
    contact = np.repeat(
        np.arange(dimensions.size, dtype=np.int32), edge_count)
    edge = np.concatenate([
        np.arange(count, dtype=np.int32) for count in edge_count
    ]) if edge_count.size else np.zeros(0, np.int32)
    return _HostRowMapping(edge_count, contact, edge, limit_joint)


def _joint_ancestor_dofs(model: mujoco.MjModel, joint: int) -> np.ndarray:
    current = int(model.jnt_dofadr[joint])
    columns: list[int] = []
    while current >= 0:
        columns.append(current)
        current = int(model.dof_parentid[current])
    return np.asarray(columns[::-1], dtype=np.int32)


def _csr_layout(model: mujoco.MjModel, collision,
                mapping: _HostRowMapping, *,
                row_capacity: int) -> tuple[np.ndarray, ...]:
    body_mask = _body_dof_mask(model).astype(bool)
    contact_geom = np.asarray(collision.contact_geom.numpy(), dtype=np.int32)
    enabled_contacts = mapping.edge_count.size
    if enabled_contacts == 0:
        contact_geom = contact_geom[:0]
    elif enabled_contacts != contact_geom.shape[0]:
        raise ValueError(
            "native contact layout must enable either all contacts or none")
    rows: list[np.ndarray] = []
    for geom_pair, contact_rows in zip(
            contact_geom, mapping.edge_count, strict=True):
        bodies = np.asarray(model.geom_bodyid[geom_pair], dtype=np.int32)
        columns = np.flatnonzero(body_mask[bodies[0]] | body_mask[bodies[1]])
        rows.extend([columns.astype(np.int32)] * int(contact_rows))
    rows.extend([
        _joint_ancestor_dofs(model, int(joint))
        for joint in mapping.limit_joint
    ])
    rows.extend([np.zeros(0, np.int32)] * (row_capacity - len(rows)))
    counts = np.asarray([row.size for row in rows], dtype=np.int32)
    offsets = np.zeros(row_capacity + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    columns = np.concatenate(rows) if rows else np.zeros(0, np.int32)
    entry_rows = np.repeat(np.arange(row_capacity, dtype=np.int32), counts)
    return offsets, counts, columns, entry_rows


def _csc_layout(columns: np.ndarray, entry_rows: np.ndarray,
                dof_count: int) -> tuple[np.ndarray, ...]:
    order = np.argsort(columns, kind="stable").astype(np.int32)
    counts = np.bincount(columns, minlength=dof_count).astype(np.int32)
    offsets = np.zeros(dof_count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    return offsets, order, entry_rows[order]


def _host_sparse_layout(model: mujoco.MjModel, collision,
                        mapping: _HostRowMapping, *,
                        row_capacity: int) -> _HostSparseLayout:
    csr = _csr_layout(model, collision, mapping, row_capacity=row_capacity)
    csc = _csc_layout(csr[2], csr[3], model.nv)
    return _HostSparseLayout(
        *csr, *csc, int(mapping.contact.size + mapping.limit_joint.size),
        row_capacity, model.nv)


def _device_sparse_layout(host: _HostSparseLayout, device) -> StaticSparseLayout:
    result = StaticSparseLayout()
    for name in (
            "row_offsets", "row_nonzero_count", "column_indices", "entry_rows",
            "column_offsets", "csc_positions", "csc_rows"):
        setattr(result, name, wp.array(
            getattr(host, name), dtype=int, device=device))
    result.row_count = host.row_count
    result.row_capacity = host.row_capacity
    result.nonzero_count = int(host.column_indices.size)
    result.dof_count = host.dof_count
    return result


def _validated_contact_dimensions(model: mujoco.MjModel,
                                  collision) -> np.ndarray:
    dimensions = np.asarray(collision.contact_dim.numpy(), dtype=np.int32)
    expected_shape = (int(collision.contact_count),)
    if dimensions.shape != expected_shape:
        raise ValueError(
            f"contact_dim must have shape {expected_shape}, got {dimensions.shape}")
    if int(model.opt.disableflags) & CONTACT_DISABLE_MASK:
        return dimensions[:0]
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("only MuJoCo pyramidal friction cones are supported")
    unsupported = sorted(set(np.unique(dimensions)) - SUPPORTED_CONTACT_DIMS)
    if unsupported:
        raise ValueError(
            f"fixed contact rows support condim={sorted(SUPPORTED_CONTACT_DIMS)}, "
            f"got {unsupported}")
    return dimensions


def _limited_joints(model: mujoco.MjModel) -> np.ndarray:
    disabled = int(model.opt.disableflags) & LIMIT_DISABLE_MASK
    if disabled:
        return np.zeros(0, dtype=np.int32)
    joints = np.flatnonzero(np.asarray(model.jnt_limited)).astype(np.int32)
    types = np.asarray(model.jnt_type, dtype=np.int32)[joints]
    supported = (types == SLIDE) | (types == HINGE)
    if not np.all(supported):
        invalid = joints[~supported].tolist()
        raise ValueError(
            f"native joint limits support slide/hinge joints, got {invalid}")
    return joints


def compile_contact_rows(cpu_model: mujoco.MjModel, device_model,
                         collision) -> CompiledContactRows:
    dimensions = _validated_contact_dimensions(cpu_model, collision)
    mapping = _host_row_mapping(
        dimensions,
        _limited_joints(cpu_model))
    device = device_model.qpos0.device
    params = ContactParameters()
    params.body_dof_mask = wp.array(
        _body_dof_mask(cpu_model), dtype=int, device=device)
    for name in ("dof_jntid", "jnt_type", "jnt_dofadr", "jnt_bodyid",
                 "geom_bodyid"):
        setattr(params, name, getattr(device_model, name))
    params.contact_geom = collision.contact_geom
    params.contact_includemargin = collision.contact_includemargin
    params.contact_friction = collision.contact_friction
    params.friction_scale = wp.ones(1, dtype=float, device=device)
    params.row_contact = wp.array(mapping.contact, dtype=int, device=device)
    params.row_edge = wp.array(mapping.edge, dtype=int, device=device)
    params.limit_joint = wp.array(
        mapping.limit_joint, dtype=int, device=device)
    for name in (
            "jnt_qposadr", "jnt_dofadr", "jnt_range", "jnt_margin"):
        setattr(params, name, getattr(device_model, name))
    params.contact_row_count = int(mapping.contact.size)
    params.limit_count = int(mapping.limit_joint.size)
    params.dof_count = cpu_model.nv
    count = int(collision.contact_count)
    row_count = params.contact_row_count + params.limit_count
    row_capacity = ((row_count + ROW_TILE_SIZE - 1) // ROW_TILE_SIZE
                    * ROW_TILE_SIZE)
    sparse = _use_matrix_free_solver()
    layout = None
    if sparse:
        host = _host_sparse_layout(
            cpu_model, collision, mapping, row_capacity=row_capacity)
        layout = _device_sparse_layout(host, device)
    return CompiledContactRows(
        parameters=params,
        contact_count=count,
        contact_row_count=params.contact_row_count,
        limit_count=params.limit_count,
        row_count=row_count,
        row_capacity=row_capacity,
        sparse=sparse,
        sparse_layout=layout,
        device=device)


def allocate_workspace(compiled: CompiledContactRows, worlds: int,
                       *, gradient: bool) -> ContactRowWorkspace:
    if compiled.sparse:
        return _allocate_sparse_workspace(compiled, worlds, gradient=gradient)
    rows = ConstraintRows()
    rows.jacobian = wp.zeros(
        (worlds, compiled.row_capacity, compiled.parameters.dof_count),
        dtype=float, device=compiled.device, requires_grad=gradient,
        retain_grad=gradient)
    rows.position = wp.zeros(
        (worlds, compiled.row_capacity), dtype=float, device=compiled.device,
        requires_grad=gradient, retain_grad=gradient)
    rows.active = wp.zeros(
        (worlds, compiled.row_capacity), dtype=int, device=compiled.device)
    return ContactRowWorkspace(rows)


def _allocate_sparse_workspace(compiled: CompiledContactRows, worlds: int,
                               *, gradient: bool) -> ContactRowWorkspace:
    rows = SparseConstraintRows()
    rows.jacobian = wp.zeros(
        (worlds, compiled.sparse_layout.nonzero_count), dtype=float,
        device=compiled.device, requires_grad=gradient, retain_grad=gradient)
    rows.position = wp.zeros(
        (worlds, compiled.row_capacity), dtype=float, device=compiled.device,
        requires_grad=gradient, retain_grad=gradient)
    rows.active = wp.zeros(
        (worlds, compiled.row_capacity), dtype=int, device=compiled.device)
    return ContactRowWorkspace(rows)


def build_rows(compiled: CompiledContactRows, inputs: ContactRowBuildInput,
               workspace: ContactRowWorkspace):
    if compiled.sparse:
        return _build_sparse_rows(compiled, inputs, workspace)
    job = ContactRowJob()
    job.contacts = inputs.contacts
    job.motion = inputs.motion
    job.rows = workspace.rows
    worlds = inputs.contacts.distance.shape[0]
    contact_shape = (worlds, compiled.contact_row_count)
    wp.launch(_build_position, dim=contact_shape,
              inputs=[compiled.parameters, job], device=compiled.device)
    wp.launch(
        _build_jacobian,
        dim=(*contact_shape, compiled.parameters.dof_count),
        inputs=[compiled.parameters, job], device=compiled.device)
    if compiled.limit_count == 0:
        return job.rows
    limit_job = LimitRowJob()
    limit_job.rows = workspace.rows
    limit_job.qpos = inputs.qpos
    limit_shape = (worlds, compiled.limit_count)
    wp.launch(_build_limit_position, dim=limit_shape,
              inputs=[compiled.parameters, limit_job], device=compiled.device)
    wp.launch(
        _build_limit_jacobian,
        dim=(*limit_shape, compiled.parameters.dof_count),
        inputs=[compiled.parameters, limit_job], device=compiled.device)
    return job.rows


def _build_sparse_rows(compiled: CompiledContactRows,
                       inputs: ContactRowBuildInput,
                       workspace: ContactRowWorkspace):
    job = SparseContactRowJob()
    job.contacts = inputs.contacts
    job.motion = inputs.motion
    job.rows = workspace.rows
    job.qpos = inputs.qpos
    worlds = inputs.contacts.distance.shape[0]
    shape = (worlds, compiled.row_count)
    wp.launch(_build_sparse_position, dim=shape,
              inputs=[compiled.parameters, job], device=compiled.device)
    wp.launch(
        _build_sparse_jacobian,
        dim=(worlds, compiled.sparse_layout.nonzero_count),
        inputs=[compiled.parameters, compiled.sparse_layout, job],
        device=compiled.device)
    return job.rows
