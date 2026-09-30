"""Complete ForceAware-Warp MPC objective and analytic gradients."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from itertools import product
import mujoco
import numpy as np
import warp as wp
from forceaware.hand_specs import ASSET_HANDS
from forceaware.config import RetargetConfig
from forceaware.control import (
    FINGER_PALM_SMOOTHING_RATIO,
    HAND_STATE_ACCELERATION_RATIO,
    HAND_STATE_BASE_POSITION_ACCELERATION_SCALE_M_S2,
    HAND_STATE_BASE_POSITION_VELOCITY_SCALE_M_S,
    HAND_STATE_BASE_ROTATION_ACCELERATION_SCALE_RAD_S2,
    HAND_STATE_BASE_ROTATION_VELOCITY_SCALE_RAD_S,
    HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2,
    HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S,
    SMOOTHING_PROFILE,
)
from forceaware.numerics import LOSS_GRADIENT_SCALE
from .targets import ModelIndex
from .targets import hand_qvel_indices

GRASP_QUALITY_PROFILE = "sampled_gws_worst_margin_v6"


GRASP_WRENCH_MARGIN = 0.05


GRASP_SUPPORT_FLOOR = 0.0


GRASP_MIN_GUIDANCE_CONTACTS = 3


GRASP_MIN_PHYSICAL_CONTACTS = 3


GRASP_WRENCH_DIMENSION = 6


GRASP_FRICTION_EDGE_COUNT = 4


GRASP_WRENCH_DIRECTION_COUNT = 256


GRASP_DENSE_DIRECTION_SEED = 731


STABLE_GUIDANCE_WEIGHT = 1.0 - 1.0e-6


CAPSULE_RADIAL_QUANTILE = 0.9


CAPSULE_SPHERE_FIELD_COUNT = 4


CAPSULE_MINIMUM_SPHERE_COUNT = 3


def grasp_wrench_directions() -> np.ndarray:
    """Return deterministic, centrally symmetric samples of the 6-D sphere."""
    axes = np.eye(GRASP_WRENCH_DIMENSION, dtype=np.float32)
    directions = [axis * sign for axis in axes for sign in (-1.0, 1.0)]
    pair_scale = np.float32(1.0 / np.sqrt(2.0))
    for left, right in combinations(range(GRASP_WRENCH_DIMENSION), 2):
        for left_sign, right_sign in product((-1.0, 1.0), repeat=2):
            value = np.zeros(GRASP_WRENCH_DIMENSION, np.float32)
            value[left] = left_sign * pair_scale
            value[right] = right_sign * pair_scale
            directions.append(value)
    sparse = np.asarray(directions, np.float32)
    dense_pair_count = (GRASP_WRENCH_DIRECTION_COUNT - sparse.shape[0]) // 2
    generator = np.random.default_rng(GRASP_DENSE_DIRECTION_SEED)
    dense = generator.standard_normal(
        (dense_pair_count, GRASP_WRENCH_DIMENSION)
    ).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    result = np.concatenate((sparse, dense, -dense), axis=0)
    if result.shape != (GRASP_WRENCH_DIRECTION_COUNT, GRASP_WRENCH_DIMENSION):
        raise ValueError("grasp wrench direction count must preserve symmetry")
    return result


MILLIMETERS = wp.constant(1000.0)


LOSS_EPSILON = wp.constant(1.0e-12)


NORMALIZATION_EPSILON = wp.constant(1.0e-8)


TEMPERATURE_EPSILON = wp.constant(1.0e-6)


HAND_ACCELERATION_RATIO = wp.constant(HAND_STATE_ACCELERATION_RATIO)


@wp.struct
class LossModel:
    hand_qpos: wp.array(dtype=int)
    finger_joint_body: wp.array(dtype=int)
    finger_joint_start: wp.array(dtype=wp.vec3)
    finger_joint_end: wp.array(dtype=wp.vec3)
    finger_joint_radius: wp.array(dtype=float)
    finger_joint_pair_left: wp.array(dtype=int)
    finger_joint_pair_right: wp.array(dtype=int)
    output_contact_rows: wp.array(dtype=int)
    output_includemargin: wp.array(dtype=float)
    output_force_sign: wp.array(dtype=float)
    output_friction: wp.array(dtype=wp.vec2)
    wrench_force_direction: wp.array(dtype=wp.vec3)
    wrench_torque_direction: wp.array(dtype=wp.vec3)
    object_com_local: wp.vec3
    object_radius: float
    object_qpos: int
    object_body: int
    action_dim: int
    guidance_count: int
    output_count: int
    finger_joint_pair_count: int
    wrench_direction_count: int
    contact_thresholds: wp.array(dtype=float)


@wp.struct
class LossWeights:
    reference_pose_scale: float  # Scales object pose tracking only.
    object_position: float
    object_rotation: float
    action_base_position: float
    action_base_rotation: float
    action_finger: float
    contact_robustness: float
    marked_query: float
    finger_joint_collision: float
    finger_joint_surface_margin_mm: float
    soft_penetration: float
    penetration: float
    target_phi: float
    soft_penetration_limit: float
    penetration_limit: float
    marked_distance_mm: float
    marked_temperature_mm: float


@wp.struct
class LossTarget:
    action: wp.array(dtype=float)
    object_qpos: wp.array(dtype=float)
    contact_mask: wp.array(dtype=float)
    contact_weight: wp.array(dtype=float)
    contact_position_object: wp.array(dtype=wp.vec3)
    contact_output: wp.array(dtype=int)
    query_mask: wp.array(dtype=float)
    query_weight: wp.array(dtype=float)
    query_position_object: wp.array(dtype=wp.vec3)
    query_body: wp.array(dtype=int)
    query_local: wp.array(dtype=wp.vec3)
    grasp_enabled: int


@wp.struct
class LossState:
    qpos: wp.array2d(dtype=float)
    body_position: wp.array2d(dtype=wp.vec3)
    body_matrix: wp.array2d(dtype=wp.mat33)
    contact_distance: wp.array2d(dtype=float)
    contact_position: wp.array2d(dtype=wp.vec3)
    contact_frame: wp.array2d(dtype=wp.mat33)
    contact_active: wp.array2d(dtype=int)


@wp.struct
class LossAuxiliary:
    marked_query_weight: wp.array2d(dtype=float)


@wp.struct
class LossJob:
    model: LossModel
    weights: LossWeights
    target: LossTarget
    state: LossState
    auxiliary: LossAuxiliary
    guidance_enabled: int
    penetration_enabled: int
    query_enabled: int
    reference_scale: float


@dataclass(frozen=True)
class LossWorkspace:
    target: LossTarget
    auxiliary: LossAuxiliary
    job: LossJob
    loss: wp.array
    seed: wp.array
    tape: wp.Tape


@wp.func
def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + wp.exp(-value))


@wp.func
def _collision_output(job: LossJob, world: int, output: int):
    row = job.model.output_contact_rows[output]
    if row >= 0 and job.state.contact_active[world, row] != 0:
        return (
            job.state.contact_distance[world, row],
            job.state.contact_position[world, row],
        )
    return job.model.contact_thresholds[output], wp.vec3(0.0)


@wp.func
def _finger_joint_world(
    job: LossJob,
    world: int,
    joint: int,
    local: wp.vec3,
) -> wp.vec3:
    body = job.model.finger_joint_body[joint]
    return (
        job.state.body_position[world, body]
        + job.state.body_matrix[world, body] @ local
    )


@wp.func
def _point_segment_distance_squared(
    point: wp.vec3,
    start: wp.vec3,
    end: wp.vec3,
) -> float:
    segment = end - start
    denominator = wp.dot(segment, segment)
    fraction = wp.clamp(
        wp.dot(point - start, segment) / wp.max(denominator, LOSS_EPSILON),
        0.0,
        1.0,
    )
    delta = point - (start + fraction * segment)
    return wp.dot(delta, delta)


@wp.func
def _segment_distance_squared(
    left_start: wp.vec3,
    left_end: wp.vec3,
    right_start: wp.vec3,
    right_end: wp.vec3,
) -> float:
    left = left_end - left_start
    right = right_end - right_start
    offset = left_start - right_start
    left_squared = wp.dot(left, left)
    cross = wp.dot(left, right)
    right_squared = wp.dot(right, right)
    left_offset = wp.dot(left, offset)
    right_offset = wp.dot(right, offset)
    denominator = left_squared * right_squared - cross * cross
    endpoint_distance = wp.min(
        _point_segment_distance_squared(left_start, right_start, right_end),
        _point_segment_distance_squared(left_end, right_start, right_end),
    )
    endpoint_distance = wp.min(
        endpoint_distance,
        _point_segment_distance_squared(right_start, left_start, left_end),
    )
    endpoint_distance = wp.min(
        endpoint_distance,
        _point_segment_distance_squared(right_end, left_start, left_end),
    )
    scale = left_squared * right_squared
    if denominator > NORMALIZATION_EPSILON * scale:
        left_fraction = (
            cross * right_offset - right_squared * left_offset
        ) / denominator
        right_fraction = (
            left_squared * right_offset - cross * left_offset
        ) / denominator
        if (
            left_fraction >= 0.0
            and left_fraction <= 1.0
            and right_fraction >= 0.0
            and right_fraction <= 1.0
        ):
            interior_delta = offset + left_fraction * left - right_fraction * right
            endpoint_distance = wp.min(
                endpoint_distance, wp.dot(interior_delta, interior_delta)
            )
    return endpoint_distance


@wp.func
def _finger_joint_collision_loss(job: LossJob, world: int) -> float:
    if job.weights.finger_joint_collision <= 0.0:
        return 0.0
    loss = float(0.0)
    for pair in range(job.model.finger_joint_pair_count):
        left = job.model.finger_joint_pair_left[pair]
        right = job.model.finger_joint_pair_right[pair]
        left_start = _finger_joint_world(
            job, world, left, job.model.finger_joint_start[left]
        )
        left_end = _finger_joint_world(
            job, world, left, job.model.finger_joint_end[left]
        )
        right_start = _finger_joint_world(
            job, world, right, job.model.finger_joint_start[right]
        )
        right_end = _finger_joint_world(
            job, world, right, job.model.finger_joint_end[right]
        )
        axis_distance_mm = (
            wp.sqrt(
                _segment_distance_squared(left_start, left_end, right_start, right_end)
                + LOSS_EPSILON
            )
            * MILLIMETERS
        )
        radius_mm = (
            job.model.finger_joint_radius[left] + job.model.finger_joint_radius[right]
        ) * MILLIMETERS
        surface_gap_mm = axis_distance_mm - radius_mm
        deficit_mm = wp.max(
            job.weights.finger_joint_surface_margin_mm - surface_gap_mm, 0.0
        )
        loss += deficit_mm * deficit_mm
    return job.weights.finger_joint_collision * loss


@wp.func
def _contact(job: LossJob, world: int, contact: int):
    output = job.target.contact_output[contact]
    return _collision_output(job, world, output)


@wp.func
def _query_target_world(job: LossJob, world: int, contact: int) -> wp.vec3:
    body = job.model.object_body
    local = job.target.query_position_object[contact]
    return (
        job.state.body_position[world, body]
        + job.state.body_matrix[world, body] @ local
    )


@wp.func
def _query_world(job: LossJob, world: int, contact: int) -> wp.vec3:
    body = job.target.query_body[contact]
    local = job.target.query_local[contact]
    return (
        job.state.body_position[world, body]
        + job.state.body_matrix[world, body] @ local
    )


@wp.kernel(enable_backward=False)
def _prepare_auxiliary(job: LossJob):
    world, contact = wp.tid()
    error = (
        _query_world(job, world, contact) - _query_target_world(job, world, contact)
    ) * MILLIMETERS
    distance = wp.sqrt(wp.dot(error, error) + LOSS_EPSILON)
    marked_argument = distance - job.weights.marked_distance_mm
    marked_argument /= wp.max(job.weights.marked_temperature_mm, TEMPERATURE_EPSILON)
    job.auxiliary.marked_query_weight[world, contact] = _sigmoid(marked_argument)


@wp.func
def _hand_reference_squared(job: LossJob, world: int) -> float:
    squared = float(0.0)
    for action in range(job.model.action_dim):
        qpos = job.state.qpos[world, job.model.hand_qpos[action]]
        error = qpos - job.target.action[action]
        weight = job.weights.action_finger
        if action < 3:
            weight = job.weights.action_base_position
        elif action < 6:
            weight = job.weights.action_base_rotation
        weighted_error = weight * error
        squared += weighted_error * weighted_error
    return squared


@wp.func
def _normalized_quaternion_angle(
    quaternion: wp.vec4,
    target: wp.vec4,
) -> float:
    quaternion /= wp.sqrt(wp.dot(quaternion, quaternion)) + NORMALIZATION_EPSILON
    target /= wp.sqrt(wp.dot(target, target)) + NORMALIZATION_EPSILON
    target *= wp.where(wp.dot(quaternion, target) < 0.0, -1.0, 1.0)
    relative_vector = wp.vec3(
        target[0] * quaternion[1]
        - target[1] * quaternion[0]
        - target[2] * quaternion[3]
        + target[3] * quaternion[2],
        target[0] * quaternion[2]
        + target[1] * quaternion[3]
        - target[2] * quaternion[0]
        - target[3] * quaternion[1],
        target[0] * quaternion[3]
        - target[1] * quaternion[2]
        + target[2] * quaternion[1]
        - target[3] * quaternion[0],
    )
    sine = wp.sqrt(wp.max(wp.dot(relative_vector, relative_vector), LOSS_EPSILON))
    return 2.0 * wp.atan2(sine, wp.dot(quaternion, target))


@wp.func
def _quaternion_angle(job: LossJob, world: int) -> float:
    address = job.model.object_qpos + 3
    return _normalized_quaternion_angle(
        wp.vec4(
            job.state.qpos[world, address],
            job.state.qpos[world, address + 1],
            job.state.qpos[world, address + 2],
            job.state.qpos[world, address + 3],
        ),
        wp.vec4(
            job.target.object_qpos[3],
            job.target.object_qpos[4],
            job.target.object_qpos[5],
            job.target.object_qpos[6],
        ),
    )


@wp.func
def _object_reference_squared(job: LossJob, world: int) -> float:
    squared = float(0.0)
    for axis in range(3):
        error = (
            job.state.qpos[world, job.model.object_qpos + axis]
            - job.target.object_qpos[axis]
        )
        weighted_error = job.weights.object_position * error
        squared += weighted_error * weighted_error
    rotation = job.weights.object_rotation * _quaternion_angle(job, world)
    squared += rotation * rotation
    return squared


@wp.func
def _reference_pose_loss(job: LossJob, world: int) -> float:
    hand_squared = _hand_reference_squared(job, world)
    object_squared = _object_reference_squared(job, world)
    return (
        MILLIMETERS
        * MILLIMETERS
        * (hand_squared + job.weights.reference_pose_scale * object_squared)
    )


@wp.func
def _contact_loss(job: LossJob, world: int) -> float:
    if (
        job.guidance_enabled == 0
        and job.penetration_enabled == 0
        and job.query_enabled == 0
    ):
        return 0.0
    loss = float(0.0)
    for contact in range(job.model.guidance_count):
        if job.guidance_enabled == 1:
            loss += _physical_contact_loss(job, world, contact)
        if job.query_enabled == 1:
            loss += _marked_query_loss(job, world, contact)
    if job.penetration_enabled == 1:
        loss += _penetration_loss(job, world)
    return loss


@wp.func
def _physical_contact_loss(job: LossJob, world: int, contact: int) -> float:
    phi, _ = _contact(job, world, contact)
    mask = job.target.contact_mask[contact] * job.target.contact_weight[contact]
    depth_deficit = wp.max((phi - job.weights.target_phi) * MILLIMETERS, 0.0)
    loss = job.weights.contact_robustness * mask * depth_deficit * depth_deficit
    return loss


@wp.func
def _penetration_loss(job: LossJob, world: int) -> float:
    loss = float(0.0)
    for output in range(job.model.output_count):
        phi, _ = _collision_output(job, world, output)
        soft = wp.max((-phi - job.weights.soft_penetration_limit) * MILLIMETERS, 0.0)
        deep = wp.max((-phi - job.weights.penetration_limit) * MILLIMETERS, 0.0)
        loss += job.weights.soft_penetration * soft * soft
        loss += job.weights.penetration * deep * deep
    return loss


@wp.func
def _marked_query_loss(job: LossJob, world: int, contact: int) -> float:
    mask = job.target.query_mask[contact] * job.target.query_weight[contact]
    error = (
        _query_world(job, world, contact) - _query_target_world(job, world, contact)
    ) * MILLIMETERS
    marked = job.auxiliary.marked_query_weight[world, contact]
    return job.weights.marked_query * mask * marked * wp.dot(error, error)


@wp.kernel
def _loss(job: LossJob, output: wp.array(dtype=float)):
    world = wp.tid()
    output[world] = (
        job.reference_scale * _reference_pose_loss(job, world)
        + _contact_loss(job, world)
        + _finger_joint_collision_loss(job, world)
    )


def make_weights(cfg: RetargetConfig) -> LossWeights:
    loss = cfg.loss
    contact = cfg.contact
    weights = LossWeights()
    weights.reference_pose_scale = loss.reference_pose_scale
    weights.object_position = loss.obj_pos
    weights.object_rotation = loss.obj_rot
    weights.action_base_position = loss.action_ref_base
    weights.action_base_rotation = loss.action_ref_base_rot
    weights.action_finger = loss.action_ref_finger
    weights.contact_robustness = loss.contact_robustness
    weights.marked_query = loss.marked_query_pos
    weights.finger_joint_collision = loss.finger_joint_collision
    weights.finger_joint_surface_margin_mm = contact.finger_joint_surface_margin_mm
    weights.soft_penetration = loss.soft_penetration
    weights.penetration = loss.penetration
    weights.target_phi = contact.target_phi
    weights.soft_penetration_limit = contact.soft_penetration_limit
    weights.penetration_limit = contact.penetration_limit
    weights.marked_distance_mm = contact.marked_query_dist_mm
    weights.marked_temperature_mm = contact.marked_query_temp_mm
    return weights


def _grasp_output_properties(cpu_model, index: ModelIndex, collision):
    rows = collision.output_contact_rows.numpy().astype(np.int32)
    layout = collision.contact_layout
    contact_geom = layout.contact_geom.numpy().astype(np.int32)
    body_pairs = cpu_model.geom_bodyid[contact_geom]
    source_bodies = collision.output_body_ids.numpy().astype(np.int32)
    normal_sign = layout.contact_normal_sign.numpy().astype(np.float32)
    friction = layout.contact_friction.numpy().astype(np.float32)
    includemargin = layout.contact_includemargin.numpy().astype(np.float32)
    valid = rows >= 0
    for output in np.flatnonzero(valid):
        row = rows[output]
        bodies = body_pairs[row]
        expected = {int(source_bodies[output]), index.object_body}
        if set(map(int, bodies)) != expected:
            raise ValueError(
                "grasp-quality output is not a hand-object contact: "
                f"output={output}, row={row}, bodies={bodies.tolist()}, "
                f"expected={sorted(expected)}"
            )
    signs = np.zeros(rows.shape, np.float32)
    margins = np.zeros(rows.shape, np.float32)
    output_friction = np.zeros((rows.size, 2), np.float32)
    signs[valid] = normal_sign[rows[valid]]
    margins[valid] = includemargin[rows[valid]]
    output_friction[valid] = friction[rows[valid], :2]
    return signs, margins, output_friction


def _object_grasp_geometry(cpu_model, index: ModelIndex, collision):
    center = np.asarray(cpu_model.body_ipos[index.object_body], np.float32)
    radius = 0.0
    for batch in collision.batches:
        model = batch.model
        if int(model.target_body_id) != index.object_body:
            continue
        spheres = model.target_spheres.numpy().astype(np.float32)
        candidate = np.linalg.norm(spheres[:, :3] - center, axis=1) + spheres[:, 3]
        radius = max(radius, float(candidate.max(initial=0.0)))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"object collision radius must be positive and finite, got {radius}"
        )
    return center, radius


def _asset_hand_spec(hand: str):
    matches = tuple(spec for spec in ASSET_HANDS if spec.name == hand.lower())
    if len(matches) != 1:
        supported = ", ".join(sorted(spec.name for spec in ASSET_HANDS))
        raise ValueError(
            f"Unsupported self-collision hand {hand!r}; supported: {supported}"
        )
    return matches[0]


def _collision_source_clouds(collision):
    clouds = {}
    for batch in collision.batches:
        model = batch.model
        centers = model.source_centers.numpy().astype(np.float32)
        radii = model.source_radii.numpy().astype(np.float32)
        body_ids = model.source_body_ids.numpy().astype(np.int32)
        if centers.shape != (radii.size, 3) or body_ids.shape != radii.shape:
            raise ValueError(
                "Gaussian source collision arrays have inconsistent shapes"
            )
        spheres = np.column_stack((centers, radii))
        for body in np.unique(body_ids):
            clouds.setdefault(int(body), []).append(spheres[body_ids == body])
    return {
        body: np.unique(np.concatenate(parts, axis=0), axis=0)
        for body, parts in clouds.items()
    }


def _finger_body_chain(cpu_model, palm_body: int, tip_site_name: str):
    site = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_SITE, tip_site_name)
    if site < 0:
        raise ValueError(f"Missing fingertip site in planner model: {tip_site_name}")
    chain = []
    body = int(cpu_model.site_bodyid[site])
    while body != palm_body and body != 0:
        chain.append(body)
        body = int(cpu_model.body_parentid[body])
    if body != palm_body:
        raise ValueError(
            f"Fingertip site {tip_site_name!r} is not below the configured palm"
        )
    return tuple(reversed(chain))


def _finger_body_chains(cpu_model, hand: str):
    spec = _asset_hand_spec(hand)
    palm = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, spec.palm_body)
    if palm < 0:
        raise ValueError(f"Missing palm body in planner model: {spec.palm_body}")
    return tuple(
        _finger_body_chain(cpu_model, palm, site) for site in spec.tip_sites.values()
    )


def _fit_collision_capsule(spheres: np.ndarray):
    if (
        spheres.ndim != 2
        or spheres.shape[1] != CAPSULE_SPHERE_FIELD_COUNT
        or spheres.shape[0] < CAPSULE_MINIMUM_SPHERE_COUNT
    ):
        raise ValueError("A joint capsule requires at least three collision spheres")
    centers = np.asarray(spheres[:, :3], np.float64)
    sphere_radii = np.asarray(spheres[:, 3], np.float64)
    mean = centers.mean(axis=0)
    centered = centers - mean
    _, principal_axes = np.linalg.eigh(centered.T @ centered)
    direction = principal_axes[:, -1]
    axial = centered @ direction
    transverse = centered - axial[:, None] * direction
    radial_extent = np.linalg.norm(transverse, axis=1) + sphere_radii
    radius = float(np.quantile(radial_extent, CAPSULE_RADIAL_QUANTILE))
    lower = float(np.min(axial - sphere_radii))
    upper = float(np.max(axial + sphere_radii))
    midpoint = 0.5 * (lower + upper)
    half_segment = max(0.5 * (upper - lower) - radius, 0.0)
    start = mean + (midpoint - half_segment) * direction
    end = mean + (midpoint + half_segment) * direction
    values = np.concatenate((start, end, np.asarray([radius])))
    if not np.isfinite(values).all() or radius <= 0.0:
        raise ValueError("Joint collision capsule must be finite with positive radius")
    return start.astype(np.float32), end.astype(np.float32), np.float32(radius)


def _finger_joint_geometry(cpu_model, collision, hand: str):
    clouds = _collision_source_clouds(collision)
    bodies, starts, ends, radii, finger_ids = [], [], [], [], []
    for finger_id, chain in enumerate(_finger_body_chains(cpu_model, hand)):
        collision_chain = tuple(body for body in chain if body in clouds)
        if not collision_chain:
            raise ValueError(f"Finger {finger_id} has no collision-bearing bodies")
        for body in collision_chain:
            start, end, radius = _fit_collision_capsule(clouds[body])
            bodies.append(body)
            starts.append(start)
            ends.append(end)
            radii.append(radius)
            finger_ids.append(finger_id)
    pairs = np.asarray(
        [
            (left, right)
            for left, right in combinations(range(len(bodies)), 2)
            if finger_ids[left] != finger_ids[right]
        ],
        np.int32,
    )
    if pairs.size == 0:
        raise ValueError("Finger-joint collision requires at least two fingers")
    return (
        np.asarray(bodies, np.int32),
        np.asarray(starts, np.float32),
        np.asarray(ends, np.float32),
        np.asarray(radii, np.float32),
        pairs,
    )


def compile_model(
    cpu_model, index: ModelIndex, collision, *, hand: str, guidance_count: int, device
) -> LossModel:
    force_sign, includemargin, friction = _grasp_output_properties(
        cpu_model, index, collision
    )
    object_com, object_radius = _object_grasp_geometry(cpu_model, index, collision)
    (
        finger_joint_body,
        finger_joint_start,
        finger_joint_end,
        finger_joint_radius,
        finger_joint_pairs,
    ) = _finger_joint_geometry(cpu_model, collision, hand)
    directions = grasp_wrench_directions()
    model = LossModel()
    model.hand_qpos = wp.array(index.hand_qpos, dtype=int, device=device)
    model.finger_joint_body = wp.array(finger_joint_body, dtype=int, device=device)
    model.finger_joint_start = wp.array(
        finger_joint_start, dtype=wp.vec3, device=device
    )
    model.finger_joint_end = wp.array(finger_joint_end, dtype=wp.vec3, device=device)
    model.finger_joint_radius = wp.array(
        finger_joint_radius, dtype=float, device=device
    )
    model.finger_joint_pair_left = wp.array(
        finger_joint_pairs[:, 0], dtype=int, device=device
    )
    model.finger_joint_pair_right = wp.array(
        finger_joint_pairs[:, 1], dtype=int, device=device
    )
    model.output_contact_rows = collision.output_contact_rows
    model.output_includemargin = wp.array(includemargin, dtype=float, device=device)
    model.output_force_sign = wp.array(force_sign, dtype=float, device=device)
    model.output_friction = wp.array(friction, dtype=wp.vec2, device=device)
    model.wrench_force_direction = wp.array(
        directions[:, :3], dtype=wp.vec3, device=device
    )
    model.wrench_torque_direction = wp.array(
        directions[:, 3:], dtype=wp.vec3, device=device
    )
    model.object_com_local = wp.vec3(*object_com)
    model.object_radius = object_radius
    model.object_qpos = index.object_qpos
    model.object_body = index.object_body
    model.action_dim = index.action_dim
    model.guidance_count = guidance_count
    model.output_count = collision.output_count
    model.finger_joint_pair_count = finger_joint_pairs.shape[0]
    model.wrench_direction_count = directions.shape[0]
    model.contact_thresholds = collision.output_thresholds
    return model


def allocate_target(action_dim: int, contact_count: int, device) -> LossTarget:
    target = LossTarget()
    target.action = wp.empty(action_dim, dtype=float, device=device)
    target.object_qpos = wp.empty(7, dtype=float, device=device)
    target.contact_mask = wp.empty(contact_count, dtype=float, device=device)
    target.contact_weight = wp.empty(contact_count, dtype=float, device=device)
    target.contact_position_object = wp.empty(
        contact_count, dtype=wp.vec3, device=device
    )
    target.contact_output = wp.empty(contact_count, dtype=int, device=device)
    target.query_mask = wp.empty(contact_count, dtype=float, device=device)
    target.query_weight = wp.empty(contact_count, dtype=float, device=device)
    target.query_position_object = wp.empty(contact_count, dtype=wp.vec3, device=device)
    target.query_body = wp.empty(contact_count, dtype=int, device=device)
    target.query_local = wp.empty(contact_count, dtype=wp.vec3, device=device)
    target.grasp_enabled = 0
    return target


def build_workspace(job: LossJob, worlds: int, device) -> LossWorkspace:
    auxiliary = LossAuxiliary()
    shape = (worlds, job.model.guidance_count)
    auxiliary.marked_query_weight = wp.empty(shape, dtype=float, device=device)
    job.auxiliary = auxiliary
    loss = wp.empty(worlds, dtype=float, device=device, requires_grad=True)
    seed = wp.full(worlds, value=LOSS_GRADIENT_SCALE, dtype=float, device=device)
    tape = wp.Tape()
    # Targets and rollout states are populated before the first evaluation.
    tape.record_launch(_loss, worlds, 0, [job], [loss], wp.get_device(device), block_dim=256)
    return LossWorkspace(job.target, auxiliary, job, loss, seed, tape)


def evaluate(workspace: LossWorkspace) -> None:
    worlds = workspace.loss.shape[0]
    device = workspace.loss.device
    wp.launch(
        _prepare_auxiliary,
        dim=(worlds, workspace.job.model.guidance_count),
        inputs=[workspace.job],
        device=device,
    )
    wp.launch(
        _loss,
        dim=worlds,
        inputs=[workspace.job],
        outputs=[workspace.loss],
        device=device,
    )


def _zero_state_gradients(state: LossState) -> None:
    differentiable_state = (
        state.qpos,
        state.body_position,
        state.body_matrix,
        state.contact_distance,
        state.contact_position,
        state.contact_frame,
    )
    for value in differentiable_state:
        if value.grad is not None:
            value.grad.zero_()


def backward(workspace: LossWorkspace) -> None:
    workspace.tape.zero()
    _zero_state_gradients(workspace.job.state)
    workspace.tape.backward(grads={workspace.loss: workspace.seed})


def stable_grasp_guidance(mask, weight, query_body) -> bool:
    stable = (np.asarray(mask) > 0.0) & (np.asarray(weight) >= STABLE_GUIDANCE_WEIGHT)
    bodies = np.asarray(query_body, np.int32)[stable]
    return np.unique(bodies).size >= GRASP_MIN_GUIDANCE_CONTACTS


def assign_target(
    target: LossTarget,
    values,
    action_step: int,
    *,
    contact_step: int | None = None,
    query_step: int | None = None,
    grasp: bool = True,
) -> None:
    contact_step = action_step if contact_step is None else contact_step
    query_step = contact_step if query_step is None else query_step
    target.action.assign(values.dense_actions[action_step])
    target.object_qpos.assign(values.object_qpos[action_step])
    target.contact_mask.assign(values.contact_mask[contact_step])
    target.contact_weight.assign(values.contact_weight[contact_step])
    target.contact_position_object.assign(values.contact_position_object[contact_step])
    target.contact_output.assign(values.contact_output[contact_step])
    target.query_mask.assign(values.contact_mask[query_step])
    target.query_weight.assign(values.contact_weight[query_step])
    target.query_position_object.assign(values.contact_position_object[query_step])
    target.query_body.assign(values.query_body[query_step])
    target.query_local.assign(values.query_local[query_step])
    target.grasp_enabled = int(
        grasp
        and stable_grasp_guidance(
            values.contact_mask[contact_step],
            values.contact_weight[contact_step],
            values.query_body[contact_step],
        )
    )


BASE_ACTION_DIM = 6


INITIAL_POSE_TERM = 0


CONTROL_VELOCITY_TERM = 1


CONTROL_ACCELERATION_TERM = 2


HAND_STATE_SMOOTHNESS_TERM = 3


REGULARIZATION_TERM_NAMES = (
    "initial_pose",
    "control_velocity",
    "control_acceleration",
    "hand_state_smoothness",
)


REGULARIZATION_TERM_COUNT = len(REGULARIZATION_TERM_NAMES)


@wp.struct
class RegularizationModel:
    init_bound: wp.array(dtype=float)
    control_velocity_scale: wp.array(dtype=float)
    control_velocity_weight: wp.array(dtype=float)
    control_acceleration_scale: wp.array(dtype=float)
    state_velocity_scale: wp.array(dtype=float)
    state_acceleration_scale: wp.array(dtype=float)
    state_component_ratio: wp.array(dtype=float)
    hand_qvel: wp.array(dtype=int)
    action_dim: int


@wp.struct
class InitialPoseRegularizationJob:
    model: RegularizationModel
    raw: wp.array2d(dtype=float)
    raw_gradient: wp.array2d(dtype=float)
    components: wp.array2d(dtype=float)
    allow_init: float
    init_weight: float
    gradient_scale: float


@wp.struct
class ControlSmoothnessJob:
    model: RegularizationModel
    previous_previous: wp.array2d(dtype=float)
    previous: wp.array2d(dtype=float)
    current: wp.array2d(dtype=float)
    raw_gradient: wp.array2d(dtype=float)
    components: wp.array2d(dtype=float)
    previous_previous_step: int
    previous_step: int
    current_step: int
    acceleration_weight: float
    previous_dt: float
    current_dt: float
    gradient_scale: float


@wp.struct
class StateSmoothnessJob:
    model: RegularizationModel
    qvel: wp.array2d(dtype=float)
    qacc: wp.array2d(dtype=float)
    qvel_cotangent: wp.array2d(dtype=float)
    qacc_cotangent: wp.array2d(dtype=float)
    components: wp.array2d(dtype=float)
    weight: float
    gradient_scale: float


@wp.kernel(enable_backward=False)
def _initial_pose_regularization_loss(job: InitialPoseRegularizationJob):
    world = wp.tid()
    job.components[world, CONTROL_VELOCITY_TERM] = 0.0
    job.components[world, CONTROL_ACCELERATION_TERM] = 0.0
    job.components[world, HAND_STATE_SMOOTHNESS_TERM] = 0.0
    initial_value = float(0.0)
    for action in range(job.model.action_dim):
        initial = job.model.init_bound[action] * wp.tanh(job.raw[world, action])
        initial_value += job.init_weight * job.allow_init * initial * initial
    job.components[world, INITIAL_POSE_TERM] = initial_value


@wp.kernel(enable_backward=False)
def _control_smoothness_loss(job: ControlSmoothnessJob):
    world = wp.tid()
    velocity_value = float(0.0)
    acceleration_value = float(0.0)
    for action in range(job.model.action_dim):
        velocity_denominator = job.model.control_velocity_scale[action] * job.current_dt
        velocity = (
            job.current[world, action] - job.previous[world, action]
        ) / velocity_denominator
        velocity_value += (
            job.model.control_velocity_weight[action] * velocity * velocity
        )
        previous_velocity = (
            job.previous[world, action] - job.previous_previous[world, action]
        ) / job.previous_dt
        current_velocity = (
            job.current[world, action] - job.previous[world, action]
        ) / job.current_dt
        acceleration = (
            2.0
            * (current_velocity - previous_velocity)
            / (job.previous_dt + job.current_dt)
            / job.model.control_acceleration_scale[action]
        )
        acceleration_value += job.acceleration_weight * acceleration * acceleration
    job.components[world, CONTROL_VELOCITY_TERM] += velocity_value
    job.components[world, CONTROL_ACCELERATION_TERM] += acceleration_value


@wp.kernel(enable_backward=False)
def _state_smoothness_loss(job: StateSmoothnessJob):
    world = wp.tid()
    velocity_value = float(0.0)
    acceleration_value = float(0.0)
    for action in range(job.model.action_dim):
        qvel_index = job.model.hand_qvel[action]
        component_weight = job.weight * job.model.state_component_ratio[action]
        normalized_velocity = (
            job.qvel[world, qvel_index] / job.model.state_velocity_scale[action]
        )
        normalized_acceleration = (
            job.qacc[world, qvel_index] / job.model.state_acceleration_scale[action]
        )
        velocity_value += component_weight * normalized_velocity * normalized_velocity
        acceleration_value += (
            component_weight
            * HAND_ACCELERATION_RATIO
            * normalized_acceleration
            * normalized_acceleration
        )
    job.components[world, HAND_STATE_SMOOTHNESS_TERM] += (
        velocity_value + acceleration_value
    )


@wp.kernel(enable_backward=False)
def _sum_components(
    components: wp.array2d(dtype=float),
    output: wp.array(dtype=float),
):
    world = wp.tid()
    value = float(0.0)
    for term in range(REGULARIZATION_TERM_COUNT):
        value += components[world, term]
    output[world] = value


@wp.kernel(enable_backward=False)
def _initial_pose_regularization_gradient(
    job: InitialPoseRegularizationJob,
):
    world, action = wp.tid()
    tangent = wp.tanh(job.raw[world, action])
    delta = job.model.init_bound[action] * tangent
    derivative = job.model.init_bound[action] * (1.0 - tangent * tangent)
    gradient = 2.0 * job.init_weight * job.allow_init * delta * derivative
    job.raw_gradient[world, action] += job.gradient_scale * gradient


@wp.kernel(enable_backward=False)
def _control_smoothness_gradient(job: ControlSmoothnessJob):
    world, action = wp.tid()
    difference = job.current[world, action] - job.previous[world, action]
    denominator = job.model.control_velocity_scale[action] * job.current_dt
    gradient = (
        2.0
        * job.gradient_scale
        * job.model.control_velocity_weight[action]
        * difference
        / (denominator * denominator)
    )
    current_offset = job.model.action_dim * (1 + job.current_step) + action
    job.raw_gradient[world, current_offset] += gradient
    if job.previous_step >= 0:
        previous_offset = job.model.action_dim * (1 + job.previous_step) + action
        job.raw_gradient[world, previous_offset] -= gradient
    acceleration_factor = 2.0 / (job.previous_dt + job.current_dt)
    previous_velocity = (
        job.previous[world, action] - job.previous_previous[world, action]
    ) / job.previous_dt
    current_velocity = (
        job.current[world, action] - job.previous[world, action]
    ) / job.current_dt
    acceleration = acceleration_factor * (current_velocity - previous_velocity)
    scale = job.model.control_acceleration_scale[action]
    acceleration_gradient = (
        2.0
        * job.gradient_scale
        * job.acceleration_weight
        * acceleration
        / (scale * scale)
    )
    current_offset = job.model.action_dim * (1 + job.current_step) + action
    job.raw_gradient[world, current_offset] += (
        acceleration_gradient * acceleration_factor / job.current_dt
    )
    if job.previous_step >= 0:
        previous_offset = job.model.action_dim * (1 + job.previous_step) + action
        job.raw_gradient[world, previous_offset] -= acceleration_gradient * (
            acceleration_factor / job.current_dt + acceleration_factor / job.previous_dt
        )
    if job.previous_previous_step >= 0:
        previous_previous_offset = (
            job.model.action_dim * (1 + job.previous_previous_step) + action
        )
        job.raw_gradient[world, previous_previous_offset] += (
            acceleration_gradient * acceleration_factor / job.previous_dt
        )


@wp.kernel(enable_backward=False)
def _state_smoothness_gradient(job: StateSmoothnessJob):
    world, action = wp.tid()
    qvel_index = job.model.hand_qvel[action]
    component_weight = job.weight * job.model.state_component_ratio[action]
    velocity_scale = job.model.state_velocity_scale[action]
    acceleration_scale = job.model.state_acceleration_scale[action]
    job.qvel_cotangent[world, qvel_index] += (
        2.0
        * job.gradient_scale
        * component_weight
        * job.qvel[world, qvel_index]
        / (velocity_scale * velocity_scale)
    )
    job.qacc_cotangent[world, qvel_index] += (
        2.0
        * job.gradient_scale
        * component_weight
        * HAND_ACCELERATION_RATIO
        * job.qacc[world, qvel_index]
        / (acceleration_scale * acceleration_scale)
    )


def _grouped_values(
    action_dim: int,
    base_position: float,
    base_rotation: float,
    finger: float,
) -> np.ndarray:
    if action_dim <= BASE_ACTION_DIM:
        raise ValueError(
            f"ForceAware smoothing requires finger actions, got {action_dim} DOFs"
        )
    values = np.full(action_dim, finger, np.float32)
    values[:3] = base_position
    values[3:BASE_ACTION_DIM] = base_rotation
    return values


def _compile_model(cpu_model, index, cfg: RetargetConfig, device):
    smoothing = cfg.smoothing
    model = RegularizationModel()
    model.init_bound = wp.array(
        _grouped_values(
            index.action_dim,
            cfg.bounds.init_base_pos,
            cfg.bounds.init_base_rot,
            cfg.bounds.init_finger,
        ),
        dtype=float,
        device=device,
    )
    model.control_velocity_scale = wp.array(
        _grouped_values(
            index.action_dim,
            smoothing.control_velocity_base_position_scale,
            smoothing.control_velocity_base_rotation_scale,
            smoothing.control_velocity_finger_scale,
        ),
        dtype=float,
        device=device,
    )
    model.control_velocity_weight = wp.array(
        _grouped_values(
            index.action_dim,
            smoothing.control_velocity_weight,
            smoothing.control_velocity_weight,
            smoothing.control_velocity_weight * FINGER_PALM_SMOOTHING_RATIO,
        ),
        dtype=float,
        device=device,
    )
    model.control_acceleration_scale = wp.array(
        _grouped_values(
            index.action_dim,
            smoothing.control_acceleration_base_position_scale,
            smoothing.control_acceleration_base_rotation_scale,
            smoothing.control_acceleration_finger_scale,
        ),
        dtype=float,
        device=device,
    )
    model.state_velocity_scale = wp.array(
        _grouped_values(
            index.action_dim,
            HAND_STATE_BASE_POSITION_VELOCITY_SCALE_M_S,
            HAND_STATE_BASE_ROTATION_VELOCITY_SCALE_RAD_S,
            HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S,
        ),
        dtype=float,
        device=device,
    )
    model.state_acceleration_scale = wp.array(
        _grouped_values(
            index.action_dim,
            HAND_STATE_BASE_POSITION_ACCELERATION_SCALE_M_S2,
            HAND_STATE_BASE_ROTATION_ACCELERATION_SCALE_RAD_S2,
            HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2,
        ),
        dtype=float,
        device=device,
    )
    model.state_component_ratio = wp.array(
        _grouped_values(
            index.action_dim,
            1.0,
            1.0,
            FINGER_PALM_SMOOTHING_RATIO,
        ),
        dtype=float,
        device=device,
    )
    model.hand_qvel = wp.array(
        hand_qvel_indices(cpu_model, index),
        dtype=int,
        device=device,
    )
    model.action_dim = index.action_dim
    return model


@dataclass(frozen=True)
class RegularizationRuntime:
    components: wp.array
    total: wp.array


def _regularization_action(
    name: str,
    value: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    action = np.asarray(value, np.float32)
    expected = (action_dim,)
    if action.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {action.shape}")
    if not np.isfinite(action).all():
        raise FloatingPointError(f"{name} must be finite")
    return action


class MpcRegularizer:
    """Own every non-trajectory term and its analytic gradient."""

    def __init__(self, cpu_model, index, cfg: RetargetConfig, *, worlds: int, device):
        self.cfg = cfg
        self.model = _compile_model(cpu_model, index, cfg, device)
        self.worlds = worlds
        self.device = device
        self._previous_previous_cpu: np.ndarray | None = None
        self._previous_previous = wp.empty(
            (worlds, index.action_dim), dtype=float, device=device
        )
        self.runtime = RegularizationRuntime(
            components=wp.empty(
                (worlds, REGULARIZATION_TERM_COUNT), dtype=float, device=device
            ),
            total=wp.empty(worlds, dtype=float, device=device),
        )

    def prepare_window(
        self,
        previous_action: np.ndarray,
        *,
        previous_previous_action: np.ndarray | None = None,
    ) -> None:
        previous = _regularization_action(
            "previous action", previous_action, self.model.action_dim
        )
        if previous_previous_action is None:
            if self._previous_previous_cpu is None:
                self._previous_previous_cpu = previous.copy()
            previous_previous = self._previous_previous_cpu
        else:
            previous_previous = _regularization_action(
                "previous-previous action",
                previous_previous_action,
                self.model.action_dim,
            )
        history = np.broadcast_to(
            previous_previous, (self.worlds, self.model.action_dim)
        )
        self._previous_previous.assign(history)

    def commit_window(self, previous_action: np.ndarray) -> None:
        previous = _regularization_action(
            "committed previous action", previous_action, self.model.action_dim
        )
        self._previous_previous_cpu = previous.copy()

    def evaluate(self, control, steps, *, allow_init: bool) -> wp.array:
        runtime = self.runtime
        job = InitialPoseRegularizationJob()
        job.model = self.model
        job.raw = control.raw
        job.raw_gradient = control.gradient
        job.components = runtime.components
        job.allow_init = float(allow_init)
        job.init_weight = self.cfg.loss.init
        job.gradient_scale = LOSS_GRADIENT_SCALE
        wp.launch(
            _initial_pose_regularization_loss,
            dim=self.worlds,
            inputs=[job],
            device=self.device,
        )
        if self.cfg.smoothing.enabled:
            self._evaluate_smoothness(control, steps)
        wp.launch(
            _sum_components,
            dim=self.worlds,
            inputs=[runtime.components],
            outputs=[runtime.total],
            device=self.device,
        )
        return runtime.total

    def _evaluate_smoothness(self, control, steps) -> None:
        self._evaluate_control_smoothness(control)
        self._evaluate_state_smoothness(steps)

    def _evaluate_control_smoothness(self, control) -> None:
        smoothing = self.cfg.smoothing
        previous = control.previous_action
        previous_previous = self._previous_previous
        previous_dt = self.cfg.simulator.time_grid.action_dt
        current_dt = self.cfg.simulator.time_grid.knot_dt
        for current in control.actions:
            job = ControlSmoothnessJob()
            job.model = self.model
            job.previous_previous = previous_previous
            job.previous = previous
            job.current = current
            job.components = self.runtime.components
            job.acceleration_weight = smoothing.control_acceleration_weight
            job.previous_dt = previous_dt
            job.current_dt = current_dt
            wp.launch(
                _control_smoothness_loss,
                dim=self.worlds,
                inputs=[job],
                device=self.device,
            )
            previous_previous, previous = previous, current
            previous_dt = current_dt

    def _evaluate_state_smoothness(self, steps) -> None:
        smoothing = self.cfg.smoothing
        for step in steps:
            job = StateSmoothnessJob()
            job.model = self.model
            job.qvel = step.recorded.result.qvel
            job.qacc = step.recorded.result.qacc
            job.qvel_cotangent = step.cotangent.qvel
            job.qacc_cotangent = step.cotangent.qacc
            job.components = self.runtime.components
            job.weight = smoothing.hand_state_weight
            job.gradient_scale = LOSS_GRADIENT_SCALE
            wp.launch(
                _state_smoothness_loss,
                dim=self.worlds,
                inputs=[job],
                device=self.device,
            )

    def add_state_cotangent(self, steps) -> None:
        if not self.cfg.smoothing.enabled:
            return
        smoothing = self.cfg.smoothing
        for step in steps:
            job = StateSmoothnessJob()
            job.model = self.model
            job.qvel = step.recorded.result.qvel
            job.qacc = step.recorded.result.qacc
            job.qvel_cotangent = step.cotangent.qvel
            job.qacc_cotangent = step.cotangent.qacc
            job.components = self.runtime.components
            job.weight = smoothing.hand_state_weight
            job.gradient_scale = LOSS_GRADIENT_SCALE
            wp.launch(
                _state_smoothness_gradient,
                dim=(self.worlds, self.model.action_dim),
                inputs=[job],
                device=self.device,
            )

    def add_regularization_gradient(self, control, *, allow_init: bool) -> None:
        job = InitialPoseRegularizationJob()
        job.model = self.model
        job.raw = control.raw
        job.raw_gradient = control.gradient
        job.components = self.runtime.components
        job.allow_init = float(allow_init)
        job.init_weight = self.cfg.loss.init
        job.gradient_scale = LOSS_GRADIENT_SCALE
        wp.launch(
            _initial_pose_regularization_gradient,
            dim=(self.worlds, self.model.action_dim),
            inputs=[job],
            device=self.device,
        )

        if not self.cfg.smoothing.enabled:
            return
        self._add_smooth_control_gradient(control)

    def _add_smooth_control_gradient(self, control) -> None:
        previous = control.previous_action
        previous_previous = self._previous_previous
        previous_step = -1
        previous_previous_step = -1
        previous_dt = self.cfg.simulator.time_grid.action_dt
        current_dt = self.cfg.simulator.time_grid.knot_dt
        for current_step, current in enumerate(control.actions):
            job = ControlSmoothnessJob()
            job.model = self.model
            job.previous_previous = previous_previous
            job.previous = previous
            job.current = current
            job.raw_gradient = control.gradient
            job.components = self.runtime.components
            job.previous_previous_step = previous_previous_step
            job.previous_step = previous_step
            job.current_step = current_step
            job.acceleration_weight = self.cfg.smoothing.control_acceleration_weight
            job.previous_dt = previous_dt
            job.current_dt = current_dt
            job.gradient_scale = LOSS_GRADIENT_SCALE
            wp.launch(
                _control_smoothness_gradient,
                dim=(self.worlds, self.model.action_dim),
                inputs=[job],
                device=self.device,
            )
            previous_previous, previous = previous, current
            previous_previous_step, previous_step = previous_step, current_step
            previous_dt = current_dt


def sampled_grasp_supports_numpy(
    contact_position: np.ndarray,
    contact_frame: np.ndarray,
    physical_contact: np.ndarray,
    force_sign: np.ndarray,
    friction: np.ndarray,
    object_position: np.ndarray,
    object_matrix: np.ndarray,
    object_com_local: np.ndarray,
    object_radius: float,
) -> np.ndarray:
    """Reference sampled GWS support values used for metrics and tests."""
    wrenches = grasp_wrenches_numpy(
        contact_position,
        contact_frame,
        physical_contact,
        force_sign,
        friction,
        object_position,
        object_matrix,
        object_com_local,
        object_radius,
    )
    directions = grasp_wrench_directions().astype(np.float64)
    if wrenches.shape[0] == 0:
        return np.zeros(directions.shape[0], np.float64)
    return np.maximum(np.max(directions @ wrenches.T, axis=1), GRASP_SUPPORT_FLOOR)


def grasp_wrenches_numpy(
    contact_position: np.ndarray,
    contact_frame: np.ndarray,
    physical_contact: np.ndarray,
    force_sign: np.ndarray,
    friction: np.ndarray,
    object_position: np.ndarray,
    object_matrix: np.ndarray,
    object_com_local: np.ndarray,
    object_radius: float,
) -> np.ndarray:
    """Build the physical primitive object-frame wrenches."""
    position = np.asarray(contact_position, np.float64)
    frame = np.asarray(contact_frame, np.float64)
    physical = np.asarray(physical_contact, bool)
    signs = np.asarray(force_sign, np.float64)
    coefficients = np.asarray(friction, np.float64)
    rotation = np.asarray(object_matrix, np.float64)
    origin = np.asarray(object_position, np.float64)
    com = np.asarray(object_com_local, np.float64)
    if position.shape != (physical.size, 3):
        raise ValueError("contact positions must have shape (outputs, 3)")
    if frame.shape != (physical.size, 3, 3):
        raise ValueError("contact frames must have shape (outputs, 3, 3)")
    if signs.shape != physical.shape or coefficients.shape != (physical.size, 2):
        raise ValueError("grasp contact properties do not match collision outputs")
    if not np.isfinite(object_radius) or object_radius <= 0.0:
        raise ValueError("object_radius must be positive and finite")
    wrenches = []
    for output in np.flatnonzero(physical & (signs != 0.0)):
        arm = rotation.T @ (position[output] - origin)
        arm = (arm - com) / object_radius
        normal = rotation.T @ (signs[output] * frame[output, 0])
        for edge in range(GRASP_FRICTION_EDGE_COUNT):
            axis, edge_sign = divmod(edge, 2)
            tangent = rotation.T @ frame[output, axis + 1]
            mu = coefficients[output, axis]
            force = normal + (1.0 if edge_sign == 0 else -1.0) * mu * tangent
            force /= np.sqrt(1.0 + mu * mu)
            wrenches.append(np.concatenate((force, np.cross(arm, force))))
    if not wrenches:
        return np.empty((0, GRASP_WRENCH_DIMENSION), np.float64)
    return np.asarray(wrenches, np.float64)


def grasp_wrench_volume_numpy(wrenches: np.ndarray) -> float:
    """Return the geometric mean eigenvalue of the wrench Gram matrix."""
    matrix = np.asarray(wrenches, np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != GRASP_WRENCH_DIMENSION:
        raise ValueError("wrenches must have shape (primitives, 6)")
    if matrix.shape[0] == 0:
        gram = np.zeros((GRASP_WRENCH_DIMENSION, GRASP_WRENCH_DIMENSION))
    else:
        gram = matrix.T @ matrix / matrix.shape[0]
    sign, log_determinant = np.linalg.slogdet(gram)
    if sign <= 0.0:
        return 0.0
    return float(np.exp(log_determinant / GRASP_WRENCH_DIMENSION))


OBJECTIVE_SCHEMA_VERSION = 16


@dataclass(frozen=True)
class TrainingLossRuntime:
    immediate: tuple
    terminal: LossWorkspace
    trajectory_terms: wp.array
    regularization: wp.array
    total: wp.array



@wp.kernel(enable_backward=False)
def _store_trajectory_loss(
    source: wp.array(dtype=float),
    column: int,
    terms: wp.array2d(dtype=float),
):
    world = wp.tid()
    terms[world, column] = source[world]


@wp.kernel(enable_backward=False)
def _sum_objective(
    trajectory_terms: wp.array2d(dtype=float),
    regularization: wp.array(dtype=float),
    total: wp.array(dtype=float),
):
    world = wp.tid()
    value = regularization[world]
    for column in range(trajectory_terms.shape[1]):
        value += trajectory_terms[world, column]
    total[world] = value


def _workspace(
    cpu_model,
    compiled,
    index,
    runtime,
    *,
    cfg: RetargetConfig,
    worlds: int,
    guidance_enabled: bool,
    penetration_enabled: bool,
    query_enabled: bool,
    guidance_count: int,
    qpos=None,
    reference_scale: float = 1.0,
    contact_active=None,
) -> LossWorkspace:
    result = runtime.recorded.result
    state = LossState()
    state.qpos = result.qpos if qpos is None else qpos
    state.body_position = result.body_position
    state.body_matrix = result.body_matrix
    state.contact_distance = result.contact_distance
    state.contact_position = result.contact_position
    state.contact_frame = result.contact_frame
    if contact_active is None:
        contact_active = runtime.workspace.forward.base.collision.contacts.active
    state.contact_active = contact_active
    model = compile_model(
        cpu_model,
        index,
        compiled.base.collision,
        hand=cfg.sequence.hand,
        guidance_count=guidance_count,
        device=compiled.base.device,
    )
    target = allocate_target(index.action_dim, guidance_count, compiled.base.device)
    job = LossJob()
    job.model = model
    job.weights = make_weights(cfg)
    job.target = target
    job.state = state
    job.guidance_enabled = int(guidance_enabled)
    job.penetration_enabled = int(penetration_enabled)
    job.query_enabled = int(query_enabled)
    job.reference_scale = reference_scale
    return build_workspace(job, worlds, compiled.base.device)


class MpcObjective:
    """Allocate, assign, evaluate, and differentiate the complete MPC objective."""

    def __init__(
        self,
        cpu_model,
        compiled,
        index,
        steps,
        terminal_observation,
        *,
        cfg: RetargetConfig,
        guidance_count: int,
        worlds: int,
    ):
        self.cpu_model = cpu_model
        self.cfg = cfg
        self.compiled = compiled
        self.index = index
        self.steps = steps
        self.terminal_observation = terminal_observation
        self.worlds = worlds
        self.dense_steps = len(steps)
        self.device = compiled.base.device
        self.regularizer = self._make_regularizer(
            cpu_model,
            index,
            cfg,
            worlds=worlds,
            device=self.device,
        )
        self.training = self._allocate_training(guidance_count)

    def _make_regularizer(
        self,
        cpu_model,
        index,
        cfg: RetargetConfig,
        *,
        worlds: int,
        device,
    ) -> MpcRegularizer:
        return MpcRegularizer(
            cpu_model,
            index,
            cfg,
            worlds=worlds,
            device=device,
        )

    def _allocate_training(self, guidance_count: int) -> TrainingLossRuntime:
        # Contacts use pre-step geometry; final tracking uses the terminal workspace.
        immediate = tuple(
            _workspace(
                self.cpu_model,
                self.compiled,
                self.index,
                runtime,
                cfg=self.cfg,
                worlds=self.worlds,
                guidance_enabled=step > 0,
                penetration_enabled=True,
                query_enabled=step > 0,
                guidance_count=guidance_count,
                reference_scale=(0.0 if step == self.dense_steps - 1 else 1.0),
            )
            for step, runtime in enumerate(self.steps)
        )
        terminal = self._terminal_workspace(guidance_count)
        columns = self.dense_steps + 1
        return TrainingLossRuntime(
            immediate=immediate,
            terminal=terminal,
            trajectory_terms=wp.empty(
                (self.worlds, columns), dtype=float, device=self.device
            ),
            regularization=self.regularizer.runtime.total,
            total=wp.empty(self.worlds, dtype=float, device=self.device),
        )

    def _terminal_workspace(self, guidance_count: int) -> LossWorkspace:
        final = self.steps[-1].recorded.result
        observation = self.terminal_observation
        return _workspace(
            self.cpu_model,
            self.compiled,
            self.index,
            observation,
            cfg=self.cfg,
            worlds=self.worlds,
            guidance_enabled=True,
            penetration_enabled=True,
            query_enabled=True,
            guidance_count=guidance_count,
            qpos=final.qpos,
            reference_scale=self.cfg.loss.terminal,
            contact_active=observation.workspace.collision.contacts.active,
        )

    def prepare_window(
        self,
        targets,
        previous_action,
        *,
        previous_previous_action=None,
    ) -> None:
        self._prepare_regularizer(
            targets, previous_action, previous_previous_action=previous_previous_action
        )
        self._assign_targets(self.training, targets)

    def _prepare_regularizer(
        self,
        targets,
        previous_action,
        *,
        previous_previous_action=None,
    ) -> None:
        del targets
        self.regularizer.prepare_window(
            previous_action, previous_previous_action=previous_previous_action
        )

    def _assign_targets(self, runtime, targets) -> None:
        for step, workspace in enumerate(runtime.immediate):
            assign_target(
                workspace.target,
                targets,
                step,
                contact_step=max(step - 1, 0),
                grasp=False,
            )
        assign_target(runtime.terminal.target, targets, self.dense_steps - 1)

    def evaluate(self, control, *, allow_init: bool) -> None:
        self._evaluate_trajectory(self.training)
        self.regularizer.evaluate(control, self.steps, allow_init=allow_init)
        wp.launch(
            _sum_objective,
            dim=self.worlds,
            inputs=[
                self.training.trajectory_terms,
                self.training.regularization,
            ],
            outputs=[self.training.total],
            device=self.device,
        )


    def _evaluate_trajectory(self, runtime) -> None:
        for column, workspace in enumerate(runtime.immediate):
            evaluate(workspace)
            wp.launch(
                _store_trajectory_loss,
                dim=self.worlds,
                inputs=[workspace.loss, column],
                outputs=[runtime.trajectory_terms],
                device=self.device,
            )
        evaluate(runtime.terminal)
        wp.launch(
            _store_trajectory_loss,
            dim=self.worlds,
            inputs=[runtime.terminal.loss, self.dense_steps],
            outputs=[runtime.trajectory_terms],
            device=self.device,
        )

    def backward_tracking(self) -> None:
        for workspace in self.training.immediate:
            backward(workspace)

    def backward_terminal(self) -> None:
        # Read immediate gradients before the shared terminal buffer is cleared.
        backward(self.training.terminal)

    def add_state_cotangent(self) -> None:
        self.regularizer.add_state_cotangent(self.steps)

    def add_regularization_gradient(self, control, *, allow_init: bool) -> None:
        self.regularizer.add_regularization_gradient(control, allow_init=allow_init)

    def commit_window(self, previous_action) -> None:
        self.regularizer.commit_window(previous_action)


def objective_metadata(cfg: RetargetConfig) -> dict[str, object]:
    """Canonical, unit-explicit objective metadata for rollout artifacts."""
    loss = cfg.loss
    contact = cfg.contact
    smoothing = cfg.smoothing
    return {
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "w_obj_pos": loss.obj_pos,
        "w_obj_rot": loss.obj_rot,
        "w_terminal": loss.terminal,
        "reference_pose_scale": loss.reference_pose_scale,
        "w_init": loss.init,
        "w_action_ref_base": loss.action_ref_base,
        "w_action_ref_base_rot": loss.action_ref_base_rot,
        "w_action_ref_finger": loss.action_ref_finger,
        "w_contact_robustness": loss.contact_robustness,
        "w_marked_query_pos": loss.marked_query_pos,
        "w_finger_joint_collision": loss.finger_joint_collision,
        "finger_joint_surface_margin_mm": (contact.finger_joint_surface_margin_mm),
        "grasp_quality_profile": GRASP_QUALITY_PROFILE,
        "grasp_wrench_margin": GRASP_WRENCH_MARGIN,
        "grasp_wrench_direction_count": grasp_wrench_directions().shape[0],
        "w_soft_penetration": loss.soft_penetration,
        "w_penetration": loss.penetration,
        "contact_target_phi": contact.target_phi,
        "soft_penetration_limit": contact.soft_penetration_limit,
        "penetration_limit": contact.penetration_limit,
        "marked_query_dist_mm": contact.marked_query_dist_mm,
        "marked_query_temp_mm": contact.marked_query_temp_mm,
        "contact_age_ramp_frames": contact.age_ramp_frames,
        "smoothing_enabled": smoothing.enabled,
        "smoothing_profile": SMOOTHING_PROFILE,
        "w_control_velocity": smoothing.control_velocity_weight,
        "w_control_acceleration": smoothing.control_acceleration_weight,
        "w_hand_state": smoothing.hand_state_weight,
        "finger_palm_smoothing_ratio": FINGER_PALM_SMOOTHING_RATIO,
    }
