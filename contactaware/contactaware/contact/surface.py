"""Generic canonical surface frames, descriptors, and matching costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contactaware.types import SurfaceDescriptorSet


SURFACE_EPS = 1e-9
SURFACE_DESCRIPTOR_SIZE = 6
AXIAL_COMPONENT = 0
AZIMUTH_COMPONENTS = slice(1, 3)
PALMAR_AZIMUTH_COMPONENT = 1
PALMAR_NORMAL_COMPONENT = 4
NORMAL_COMPONENTS = slice(3, 6)


@dataclass(frozen=True)
class TerminalFrame:
    root: np.ndarray
    tip: np.ndarray
    basis: np.ndarray


def normalize_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= SURFACE_EPS:
        raise ValueError(f"Cannot normalize degenerate {name}: {vector}")
    return vector / length


def make_terminal_frame(
    root: np.ndarray,
    tip: np.ndarray,
    palmar_direction: np.ndarray,
) -> TerminalFrame:
    root = np.asarray(root, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    longitudinal = normalize_vector(tip - root, name="terminal axis")
    palmar = np.asarray(palmar_direction, dtype=np.float64)
    palmar = palmar - float(palmar @ longitudinal) * longitudinal
    palmar = normalize_vector(palmar, name="projected palmar direction")
    lateral = normalize_vector(np.cross(palmar, longitudinal), name="lateral axis")
    basis = np.stack([lateral, palmar, longitudinal], axis=1)
    return TerminalFrame(root, tip, basis)


def mesh_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangle faces with shape (F, 3), got {faces.shape}")
    edge_left = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_right = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = np.cross(edge_left, edge_right)
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= SURFACE_EPS):
        bad = np.flatnonzero(lengths <= SURFACE_EPS)
        raise ValueError(f"Degenerate mesh vertex normals at ids: {bad.tolist()}")
    return normals / lengths[:, None]


def surface_descriptors(
    points: np.ndarray,
    normals: np.ndarray,
    frame: TerminalFrame,
) -> SurfaceDescriptorSet:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError(
            f"Surface points/normals must both have shape (N, 3), got "
            f"{points.shape} and {normals.shape}"
        )
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(normals)) or np.any(normal_lengths <= SURFACE_EPS):
        raise ValueError("Surface descriptors require finite non-zero normals")
    normals = normals / normal_lengths[:, None]
    longitudinal = frame.basis[:, 2]
    terminal_length = float(np.linalg.norm(frame.tip - frame.root))
    relative = points - frame.root
    axial_distance = relative @ longitudinal
    axial_coordinate = axial_distance / terminal_length
    radial = relative - axial_distance[:, None] * longitudinal
    radial_x = radial @ frame.basis[:, 0]
    radial_y = radial @ frame.basis[:, 1]
    radial_length = np.hypot(radial_x, radial_y)
    azimuth_valid = radial_length > SURFACE_EPS
    azimuth = np.zeros((len(points), 2), dtype=np.float64)
    azimuth[azimuth_valid, 0] = radial_y[azimuth_valid] / radial_length[azimuth_valid]
    azimuth[azimuth_valid, 1] = radial_x[azimuth_valid] / radial_length[azimuth_valid]
    local_normals = normals @ frame.basis
    values = np.column_stack([axial_coordinate, azimuth, local_normals])
    return SurfaceDescriptorSet(values, azimuth_valid)


def subset_descriptors(
    descriptors: SurfaceDescriptorSet,
    indices: np.ndarray,
) -> SurfaceDescriptorSet:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"Descriptor indices must be one-dimensional, got {indices.shape}")
    if np.any(indices < 0) or np.any(indices >= len(descriptors.values)):
        raise IndexError(
            f"Descriptor indices out of range [0, {len(descriptors.values)}): {indices.tolist()}"
        )
    return SurfaceDescriptorSet(
        descriptors.values[indices],
        descriptors.azimuth_valid[indices],
    )


def terminal_face_masks(descriptors: SurfaceDescriptorSet, minimum_alignment: float):
    """Identify outward-facing finger pads and backs; exclude the sides."""
    values = descriptors.values
    palmar = (descriptors.azimuth_valid
              & (values[:, PALMAR_AZIMUTH_COMPONENT] >= minimum_alignment)
              & (values[:, PALMAR_NORMAL_COMPONENT] >= minimum_alignment))
    dorsal = (descriptors.azimuth_valid
              & (values[:, PALMAR_AZIMUTH_COMPONENT] <= -minimum_alignment)
              & (values[:, PALMAR_NORMAL_COMPONENT] <= -minimum_alignment))
    return palmar, dorsal


def contact_surface_descriptor_cost(reference, candidates, *, minimum_palmar,
                                    axial_weight, azimuth_weight):
    """Match a MANO finger-pad contact only to the robot finger pad."""
    palmar_reference, _ = terminal_face_masks(reference, minimum_palmar)
    if not palmar_reference[0]:
        raise ValueError("MANO contact is not on the finger pad")
    allowed, _ = terminal_face_masks(candidates, minimum_palmar)
    cost = surface_descriptor_cost(reference, candidates, axial_weight=axial_weight,
                                   azimuth_weight=azimuth_weight)
    return np.where(allowed, cost, np.inf)


def surface_descriptor_cost(
    reference: SurfaceDescriptorSet,
    candidates: SurfaceDescriptorSet,
    *,
    axial_weight: float,
    azimuth_weight: float,
) -> np.ndarray:
    expected_reference_shape = (1, SURFACE_DESCRIPTOR_SIZE)
    if reference.values.shape != expected_reference_shape or reference.azimuth_valid.shape != (1,):
        raise ValueError("Surface descriptor reference must contain exactly one point")
    if candidates.values.ndim != 2 or candidates.values.shape[1] != SURFACE_DESCRIPTOR_SIZE:
        raise ValueError(
            f"Candidate descriptor shape must be (N, {SURFACE_DESCRIPTOR_SIZE}), "
            f"got {candidates.values.shape}"
        )
    if not np.all(np.isfinite(reference.values)) or not np.all(np.isfinite(candidates.values)):
        raise ValueError("Cannot compare non-finite surface descriptors")
    ref = reference.values[0]
    axial_cost = float(axial_weight) * np.square(
        candidates.values[:, AXIAL_COMPONENT] - ref[AXIAL_COMPONENT]
    )
    azimuth_cost = np.zeros(len(candidates.values), dtype=np.float64)
    if bool(reference.azimuth_valid[0]):
        invalid = ~candidates.azimuth_valid
        azimuth_dot = np.clip(
            candidates.values[:, AZIMUTH_COMPONENTS] @ ref[AZIMUTH_COMPONENTS], -1.0, 1.0
        )
        azimuth_cost = float(azimuth_weight) * (1.0 - azimuth_dot)
        azimuth_cost[invalid] = np.inf
    # Normal matching excludes end and back faces.
    normal_dot = np.clip(
        candidates.values[:, NORMAL_COMPONENTS] @ ref[NORMAL_COMPONENTS],
        -1.0, 1.0,
    )
    normal_cost = float(azimuth_weight) * (1.0 - normal_dot)
    return axial_cost + azimuth_cost + normal_cost
