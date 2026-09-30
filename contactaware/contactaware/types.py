"""Shared immutable value types passed between contact-aware domains."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

TERMINAL_PAD = "terminal_pad"
FULL_TERMINAL_SURFACE = "full_terminal_surface"

if TYPE_CHECKING:
    from contactaware.solver.hand import MujocoHand


@dataclass(frozen=True)
class SafeConfig:
    safe_distance: float
    activation_distance: float
    gamma: float
    joint_gamma: float
    fingertip_weight: float
    palm_position_weight: float
    palm_rotation_weight: float
    base_translation_step_weight: float
    base_rotation_step_weight: float
    joint_step_weight: float
    base_translation_velocity_weight: float
    base_rotation_velocity_weight: float
    joint_velocity_weight: float
    tracking_relative_tolerance: float
    sqp_stagnation_patience: int
    sqp_progress_relative_tolerance: float
    min_hessian: float
    qp_maxiter: int
    first_frame_max_iterations: int
    first_frame_ftol: float
    qp_absolute_tolerance: float
    qp_relative_tolerance: float
    qp_validation_tolerance: float
    geometry_feasibility_tolerance_m: float
    joint_limit_tolerance_rad: float
    base_translation_trust_m: float
    base_rotation_trust_rad: float
    joint_trust_rad: float
    contact_anchor_weight: float
    contact_normal_weight_scale: float
    hand_object_activation: float
    hand_object_safe_distance: float
    pre_contact_safe_distance: float
    key_release_contact_error: float
    hand_object_gamma: float
    hand_object_topk_per_link: int
    mano_shape_weight: float
    base_align_iters: int
    base_align_step: float
    pad_min_axial: float
    pad_max_axial: float
    pad_min_palmar: float
    descriptor_axial_cost_weight: float
    descriptor_azimuth_cost_weight: float


@dataclass(frozen=True)
class Capsule:
    geom: str
    body: str
    finger: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class QueryBodySpec:
    file_stem: str
    body: str
    finger_id: int
    link_id: int
    is_tip: bool = False


@dataclass(frozen=True)
class HandProfile:
    name: str
    mesh_xml: Path
    qp_dir: Path
    base_qpos_dim: int
    palm_body: str
    palm_anchor_offset: tuple[float, float, float]
    track_sites: tuple[str, ...]
    track_finger_ids: tuple[int, ...]
    query_specs: tuple[QueryBodySpec, ...]
    capsules: tuple[Capsule, ...]
    chain_points: Mapping[str, tuple[tuple[str, str], ...]]
    tip_references: Mapping[int, tuple[str, tuple[float, float, float]]]
    merged_terminal_finger_ids: tuple[int, ...]
    default_qpos: np.ndarray
    terminal_flexion_signs: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryPointSet:
    local_pos: np.ndarray
    local_normal: np.ndarray
    body_ids: np.ndarray
    finger_ids: np.ndarray
    link_ids: np.ndarray
    is_tip: np.ndarray


@dataclass(frozen=True)
class SurfaceDescriptorSet:
    values: np.ndarray
    azimuth_valid: np.ndarray


@dataclass(frozen=True)
class ManoSurfaceMapping:
    descriptors: SurfaceDescriptorSet
    merged_terminal_split_by_finger: Mapping[int, float]


@dataclass(frozen=True)
class ManoTrajectory:
    joints: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    vertex_fingers: np.ndarray
    surface_mapping: ManoSurfaceMapping
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class ObjectModel:
    # Static spheres of the collision target (Gaussians, or mesh convex-hull vertices), (N, 4).
    support_spheres: np.ndarray
    xml_path: Path
    distance_offset: float
    query_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class ContactAnchorSet:
    mask: np.ndarray
    anchor_pos_obj: np.ndarray
    anchor_normal_obj: np.ndarray
    finger_ids: np.ndarray
    mano_vertex_ids: np.ndarray
    summary: Mapping[str, object]


@dataclass(frozen=True)
class ContactRuntime:
    anchors: ContactAnchorSet
    hand_query_ids: np.ndarray
    participation: np.ndarray
    contact_target_pos_obj: np.ndarray
    guidance_blend: np.ndarray


@dataclass(frozen=True)
class SequencePolicy:
    dataset: str
    source: str
    coordinate_transform: str
    world_pose_policy: str
    ground_stabilization: bool
    contact_query_region: str


@dataclass(frozen=True)
class RetargetInputs:
    mano_joints: np.ndarray
    mano_vertices: np.ndarray
    mano_faces: np.ndarray
    mano_surface_mapping: ManoSurfaceMapping
    obj_pose: np.ndarray
    camera_pose: np.ndarray
    camera_fovy: float
    hand: MujocoHand
    obj: ObjectModel
    anchors: ContactAnchorSet
    sequence_policy: SequencePolicy
