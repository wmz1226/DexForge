"""Backend-neutral ForceAware target loading and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from forceaware.config import RetargetConfig
from forceaware.contact_mapping import map_guidance_to_gs_outputs
from forceaware.contact_schedule import contact_age_ramp
from forceaware.time_grid import TimeGrid


BASE_ACTION_DIM = 6
QUATERNION_EPSILON = 1.0e-8
SURFACE_NORMAL_EPSILON = 1.0e-10


@dataclass(frozen=True)
class ModelIndex:
    hand_qpos: np.ndarray
    hand_ctrl: np.ndarray
    object_qpos: int
    object_body: int

    @property
    def action_dim(self) -> int:
        return int(self.hand_qpos.size)


@dataclass(frozen=True)
class ContactGuidance:
    mask: np.ndarray
    weight: np.ndarray
    position_object: np.ndarray
    output_index: np.ndarray
    query_body: np.ndarray
    query_local: np.ndarray


@dataclass(frozen=True)
class SequenceData:
    hand: np.ndarray
    object_pose_xyzw: np.ndarray
    contact: ContactGuidance
    hand_index_per_frame: float = 1.0
    hand_source: str = "contactaware"


@dataclass(frozen=True)
class WindowTargets:
    actions: np.ndarray
    dense_actions: np.ndarray
    object_qpos: np.ndarray
    contact_mask: np.ndarray
    contact_weight: np.ndarray
    contact_position_object: np.ndarray
    contact_output: np.ndarray
    query_body: np.ndarray
    query_local: np.ndarray
    frames: np.ndarray
    knot_frames: np.ndarray


@dataclass(frozen=True)
class WindowFrameGrid:
    knots: np.ndarray
    dense: np.ndarray


def load_sequence(
    cfg: RetargetConfig,
    *,
    output_body_ids: np.ndarray,
    enabled_output_indices: np.ndarray,
    object_surface_spheres: np.ndarray | None = None,
) -> SequenceData:
    output = cfg.sequence.contact_guidance_npz.parent
    hand = cfg.sequence.hand
    guidance_path = cfg.sequence.contact_guidance_npz
    if not guidance_path.exists():
        raise FileNotFoundError(f"Missing contact guidance: {guidance_path}")
    with np.load(guidance_path, allow_pickle=False) as guidance:
        contact = _load_contact_guidance(
            guidance,
            ramp_frames=cfg.contact.age_ramp_frames,
            output_body_ids=output_body_ids,
            enabled_output_indices=enabled_output_indices,
            object_surface_spheres=object_surface_spheres,
            surface_offset=cfg.simulator.gs_distance_offset,
        )
    hand_trajectory = _load_array(output / f"{hand}_qpos.npy")
    hand_index_per_frame, hand_source = 1.0, "contactaware"
    sequence = SequenceData(
        hand_trajectory,
        _load_array(output / "object_pose_7.npy"),
        contact,
        hand_index_per_frame,
        hand_source,
    )
    _validate_sequence(sequence)
    return sequence


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing contactaware trajectory: {path}")
    return np.load(path, allow_pickle=False).astype(np.float32)


def _load_contact_guidance(
    guidance,
    *,
    ramp_frames: int,
    output_body_ids: np.ndarray,
    enabled_output_indices: np.ndarray,
    object_surface_spheres: np.ndarray | None = None,
    surface_offset: float = 0.0,
) -> ContactGuidance:
    mask = np.asarray(guidance["contact_mask"], np.float32)
    weight = np.asarray(guidance["contact_weight"], np.float32)
    position_object = np.asarray(guidance["contact_pos_obj"], np.float32)
    query_body = np.asarray(guidance["hand_query_body_ids"], np.int32)
    query_local = np.asarray(guidance["hand_query_local_pos"], np.float32)
    if mask.ndim != 2:
        raise ValueError(f"contact mask must be rank-2, got {mask.shape}")
    if weight.shape != mask.shape:
        raise ValueError("contact weight must match contact mask")
    expected_points = mask.shape + (3,)
    if position_object.shape != expected_points:
        raise ValueError("object contact positions must match contact mask")
    if query_body.shape != mask.shape:
        raise ValueError("query body IDs must match contact mask")
    if query_local.shape != expected_points:
        raise ValueError("query-local positions must match contact mask")
    output_index = map_guidance_to_gs_outputs(
        np.asarray(output_body_ids, np.int32),
        query_body,
        mask,
    )
    _validate_enabled_outputs(output_index, mask, enabled_output_indices)
    age_ramp = contact_age_ramp(mask, ramp_frames)
    if surface_offset != 0.0:
        position_object = offset_contact_positions(
            position_object,
            mask,
            object_surface_spheres,
            offset=surface_offset,
        )
    return ContactGuidance(
        mask=mask,
        weight=weight * age_ramp,
        position_object=position_object,
        output_index=output_index,
        query_body=query_body,
        query_local=query_local,
    )


def offset_contact_positions(
    position_object: np.ndarray,
    mask: np.ndarray,
    object_surface_spheres: np.ndarray | None,
    *,
    offset: float,
) -> np.ndarray:
    """Move active anchors inward along their original GS surface normals."""
    result = np.asarray(position_object, np.float32).copy()
    active = np.asarray(mask) > 0.0
    if offset == 0.0 or not active.any():
        return result
    spheres = _validate_surface_spheres(object_surface_spheres)
    unique, inverse = np.unique(result[active], axis=0, return_inverse=True)
    shifted = np.stack(
        [_offset_surface_point(point, spheres, offset=offset) for point in unique]
    ).astype(np.float32)
    result[active] = shifted[inverse]
    return result


def _validate_surface_spheres(spheres: np.ndarray | None) -> np.ndarray:
    if spheres is None:
        raise ValueError("Contact anchor inset requires object Gaussian spheres")
    values = np.asarray(spheres, np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError(
            f"object Gaussian spheres must have shape (N, 4), got {values.shape}"
        )
    if not np.isfinite(values).all() or np.any(values[:, 3] <= 0.0):
        raise ValueError("object Gaussian spheres must be finite with positive radii")
    return values


def _offset_surface_point(
    point: np.ndarray,
    spheres: np.ndarray,
    *,
    offset: float,
) -> np.ndarray:
    delta = np.asarray(point, np.float64) - spheres[:, :3]
    distance = np.linalg.norm(delta, axis=1)
    support = int(np.argmin(distance - spheres[:, 3]))
    support_distance = float(distance[support])
    if support_distance <= SURFACE_NORMAL_EPSILON:
        raise ValueError("Cannot determine GS surface normal at a sphere center")
    outward_normal = delta[support] / support_distance
    return np.asarray(point, np.float64) - offset * outward_normal


def _validate_enabled_outputs(
    mapped: np.ndarray,
    mask: np.ndarray,
    enabled_output_indices: np.ndarray,
) -> None:
    active_outputs = np.unique(mapped[mask > 0.0])
    enabled = np.asarray(enabled_output_indices, np.int32)
    disabled = np.setdiff1d(active_outputs, enabled)
    if disabled.size:
        raise ValueError(
            "Guidance maps to GS outputs disabled for hand-object collision: "
            f"{disabled.tolist()}"
        )


def _validate_sequence(sequence: SequenceData) -> None:
    if sequence.hand.ndim != 2:
        raise ValueError(f"hand trajectory must be rank-2, got {sequence.hand.shape}")
    frames = sequence.object_pose_xyzw.shape[0]
    if sequence.object_pose_xyzw.shape != (frames, 7):
        raise ValueError(
            "object trajectory must have seven values per frame: "
            f"{sequence.object_pose_xyzw.shape}"
        )
    if sequence.contact.mask.shape[0] != frames:
        raise ValueError(
            "contact guidance and object trajectory must have equal frame counts: "
            f"contact={sequence.contact.mask.shape[0]}, object={frames}"
        )
    if sequence.hand_index_per_frame <= 0.0:
        raise ValueError("hand_index_per_frame must be positive")
    covered = 1.0 + (sequence.hand.shape[0] - 1) / sequence.hand_index_per_frame
    # Clamp the sub-frame tail to the final reference state.
    if covered + 1.0 < frames:
        raise ValueError(
            "hand reference does not span the object trajectory: covers "
            f"{covered:g} frames, object has {frames}"
        )


def build_index(model: mujoco.MjModel) -> ModelIndex:
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    if object_joint < 0:
        raise ValueError("scene is missing joint 'obj_joint'")
    object_qpos = int(model.jnt_qposadr[object_joint])
    hand_joints = _hand_joints(model, object_qpos)
    hand_qpos = np.asarray(
        [model.jnt_qposadr[joint] for joint in hand_joints], np.int32
    )
    actuator_by_joint = {
        int(model.actuator_trnid[actuator, 0]): actuator for actuator in range(model.nu)
    }
    missing = [joint for joint in hand_joints if joint not in actuator_by_joint]
    if missing:
        raise ValueError(f"hand joints without actuators: {missing}")
    hand_ctrl = np.asarray(
        [actuator_by_joint[joint] for joint in hand_joints], np.int32
    )
    return ModelIndex(
        hand_qpos,
        hand_ctrl,
        object_qpos,
        int(model.jnt_bodyid[object_joint]),
    )


def hand_qvel_indices(
    model: mujoco.MjModel,
    index: ModelIndex,
) -> np.ndarray:
    qpos_to_qvel = {
        int(model.jnt_qposadr[joint]): int(model.jnt_dofadr[joint])
        for joint in range(model.njnt)
    }
    missing = [int(qpos) for qpos in index.hand_qpos if int(qpos) not in qpos_to_qvel]
    if missing:
        raise ValueError(f"hand qpos addresses have no joint DOF mapping: {missing}")
    return np.asarray([qpos_to_qvel[int(qpos)] for qpos in index.hand_qpos], np.int32)


def initial_state(
    model: mujoco.MjModel,
    index: ModelIndex,
    sequence: SequenceData,
    *,
    start_frame: float,
    ref_dt: float,
    hold: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(ref_dt) or ref_dt <= 0.0:
        raise ValueError("Reference timestep must be finite and positive")
    qpos = _reference_qpos(model, index, sequence, start_frame)
    qvel = np.zeros(model.nv, np.float64)
    next_frame = min(np.floor(start_frame) + 1.0, len(sequence.object_pose_xyzw) - 1)
    if not hold and next_frame > start_frame:
        following = _reference_qpos(model, index, sequence, next_frame)
        mujoco.mj_differentiatePos(
            model, qvel, (next_frame - start_frame) * ref_dt,
            qpos.astype(np.float64), following.astype(np.float64),
        )
    return qpos, qvel.astype(np.float32)


def _reference_qpos(model, index, sequence, start_frame):
    qpos = np.asarray(model.qpos0, np.float32).copy()
    qpos[index.hand_qpos] = _interpolate(
        sequence.hand,
        np.asarray([start_frame], np.float32) * sequence.hand_index_per_frame,
    )[0]
    pose = _interpolate_pose(
        sequence.object_pose_xyzw, np.asarray([start_frame], np.float32)
    )[0]
    qpos[index.object_qpos : index.object_qpos + 3] = pose[4:7]
    qpos[index.object_qpos + 3 : index.object_qpos + 7] = pose[[3, 0, 1, 2]]
    return qpos


def make_window(
    sequence: SequenceData,
    *,
    executed_steps: int,
    horizon: int,
    time_grid: TimeGrid,
    start_frame: float,
    end_frame: int,
    hold_frame: int = -1,
) -> WindowTargets:
    frame_grid = _window_frame_grids(
        executed_steps,
        horizon,
        knot_substeps=time_grid.knot_substeps,
        start_frame=start_frame,
        end_frame=end_frame,
        knot_frame_step=time_grid.knot_dt / time_grid.ref_dt,
        dense_frame_step=time_grid.mpc_dt / time_grid.ref_dt,
        hold_frame=hold_frame,
    )
    object_qpos = _object_qpos(sequence.object_pose_xyzw, frame_grid.dense)
    contact_ids = np.clip(
        np.rint(frame_grid.dense).astype(np.int32),
        0,
        sequence.contact.mask.shape[0] - 1,
    )
    scale = sequence.hand_index_per_frame
    return WindowTargets(
        actions=_interpolate(sequence.hand, frame_grid.knots * scale),
        dense_actions=_interpolate(sequence.hand, frame_grid.dense * scale),
        object_qpos=object_qpos,
        contact_mask=sequence.contact.mask[contact_ids],
        contact_weight=sequence.contact.weight[contact_ids],
        contact_position_object=sequence.contact.position_object[contact_ids],
        contact_output=sequence.contact.output_index[contact_ids],
        query_body=sequence.contact.query_body[contact_ids],
        query_local=sequence.contact.query_local[contact_ids],
        frames=frame_grid.dense,
        knot_frames=frame_grid.knots,
    )


def _window_frame_grids(
    executed_steps: int,
    horizon: int,
    *,
    knot_substeps: int,
    start_frame: float,
    end_frame: int,
    knot_frame_step: float,
    dense_frame_step: float,
    hold_frame: int,
) -> WindowFrameGrid:
    if executed_steps < 0:
        raise ValueError("executed_steps must be nonnegative")
    if hold_frame >= 0:
        knot = np.full(horizon, float(hold_frame), np.float32)
        dense = np.full(horizon * knot_substeps, float(hold_frame), np.float32)
        return WindowFrameGrid(knot, dense)
    if executed_steps % knot_substeps == 0:
        knot_origin = executed_steps // knot_substeps
    else:
        knot_origin = np.float32(executed_steps) / np.float32(knot_substeps)
    knot_offsets = knot_origin + np.arange(1, horizon + 1, dtype=np.float32)
    dense_offsets = executed_steps + np.arange(
        1, horizon * knot_substeps + 1, dtype=np.float32
    )
    knot = start_frame + knot_offsets * knot_frame_step
    dense = start_frame + dense_offsets * dense_frame_step
    return WindowFrameGrid(
        np.minimum(knot, float(end_frame)),
        np.minimum(dense, float(end_frame)),
    )


def _hand_joints(model: mujoco.MjModel, object_qpos: int) -> list[int]:
    joints = [
        joint
        for joint in range(model.njnt)
        if int(model.jnt_qposadr[joint]) < object_qpos
    ]
    joints.sort(key=lambda joint: int(model.jnt_qposadr[joint]))
    if len(joints) != object_qpos or len(joints) <= BASE_ACTION_DIM:
        raise ValueError("hand joints before obj_joint must all be scalar")
    return joints


def _interpolate(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    lower = np.floor(frames).astype(np.int32)
    lower = np.clip(lower, 0, values.shape[0] - 1)
    upper = np.minimum(lower + 1, values.shape[0] - 1)
    fraction = np.clip(frames - lower.astype(np.float32), 0.0, 1.0)[:, None]
    return ((1.0 - fraction) * values[lower] + fraction * values[upper]).astype(
        np.float32
    )


def _object_qpos(poses: np.ndarray, frames: np.ndarray) -> np.ndarray:
    pose = _interpolate_pose(poses, frames)
    return np.concatenate((pose[:, 4:7], pose[:, [3, 0, 1, 2]]), axis=1)


def _interpolate_pose(poses: np.ndarray, frames: np.ndarray) -> np.ndarray:
    position = _interpolate(poses[:, 4:7], frames)
    lower = np.clip(np.floor(frames).astype(np.int32), 0, poses.shape[0] - 1)
    quaternion = np.stack(
        [
            _quaternion_nlerp(
                poses[lo, :4],
                poses[min(lo + 1, len(poses) - 1), :4],
                float(np.clip(frame - lo, 0.0, 1.0)),
            )
            for frame, lo in zip(frames, lower)
        ]
    )
    return np.concatenate((quaternion, position), axis=1)


def _quaternion_nlerp(
    left: np.ndarray, right: np.ndarray, fraction: float
) -> np.ndarray:
    right = -right if float(np.dot(left, right)) < 0.0 else right
    value = (1.0 - fraction) * left + fraction * right
    return (value / (float(np.linalg.norm(value)) + QUATERNION_EPSILON)).astype(
        np.float32
    )
