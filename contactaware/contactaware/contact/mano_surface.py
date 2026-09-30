"""Canonical MANO terminal-surface descriptors and semantic link mapping."""

from __future__ import annotations

import numpy as np

from contactaware.types import ManoSurfaceMapping, SurfaceDescriptorSet
from contactaware.contact.surface import (
    AXIAL_COMPONENT,
    SURFACE_DESCRIPTOR_SIZE,
    SURFACE_EPS,
    TerminalFrame,
    make_terminal_frame,
    mesh_vertex_normals,
    normalize_vector,
    subset_descriptors,
    surface_descriptors,
)


# MANO pad landmarks; left and right models share vertex IDs.
MANO_PAD_VERTEX_BY_FINGER = {0: 328, 1: 438, 2: 566, 3: 760, 5: 687}
MANO_JOINT_CHAIN_BY_FINGER = {
    0: (1, 2, 3),
    1: (4, 5, 6),
    2: (10, 11, 12),
    3: (13, 14, 15),
    5: (7, 8, 9),
}
MANO_TIP_VERTEX_BY_FINGER = {
    0: 317,
    1: 444,
    2: 556,
    3: 745,
    5: 673,
}


def canonical_mano_surface_mapping(
    mano,
    vertex_fingers: np.ndarray,
) -> ManoSurfaceMapping:
    vertices, joints, faces = mano.canonical_geometry()
    normals = mesh_vertex_normals(vertices, faces)
    vertex_fingers = np.asarray(vertex_fingers, dtype=np.int32)
    values = np.full((len(vertices), SURFACE_DESCRIPTOR_SIZE), np.nan, dtype=np.float64)
    azimuth_valid = np.zeros(len(vertices), dtype=bool)
    merged_terminal_splits = {}
    for finger_id, joint_chain in MANO_JOINT_CHAIN_BY_FINGER.items():
        vertex_ids = np.flatnonzero(vertex_fingers == int(finger_id))
        frame = mano_terminal_frame(
            vertices,
            joints,
            vertex_ids,
            finger_id=int(finger_id),
            root_joint=int(joint_chain[-1]),
            palmar_normal=normals[MANO_PAD_VERTEX_BY_FINGER[int(finger_id)]],
            side=mano.side,
        )
        middle_length = float(np.linalg.norm(joints[joint_chain[-1]] - joints[joint_chain[-2]]))
        distal_length = float(np.linalg.norm(frame.tip - frame.root))
        merged_terminal_splits[int(finger_id)] = middle_length / (
            middle_length + distal_length
        )
        finger_desc = surface_descriptors(vertices[vertex_ids], normals[vertex_ids], frame)
        values[vertex_ids] = finger_desc.values
        azimuth_valid[vertex_ids] = finger_desc.azimuth_valid
    return ManoSurfaceMapping(
        descriptors=SurfaceDescriptorSet(values, azimuth_valid),
        merged_terminal_split_by_finger=merged_terminal_splits,
    )


def mano_reference_descriptor(
    mapping: ManoSurfaceMapping,
    *,
    vertex_id: int,
    finger_id: int,
    merged_terminal: bool,
) -> SurfaceDescriptorSet:
    reference = subset_descriptors(
        mapping.descriptors,
        np.asarray([vertex_id], dtype=np.int64),
    )
    if not merged_terminal:
        return reference
    if finger_id not in mapping.merged_terminal_split_by_finger:
        raise KeyError(f"Missing MANO middle/distal split for finger_id={finger_id}")
    split = float(mapping.merged_terminal_split_by_finger[finger_id])
    if not 0.0 < split < 1.0:
        raise ValueError(f"Invalid MANO middle/distal split for finger_id={finger_id}: {split}")
    values = reference.values.copy()
    # Embed distal u after the virtual DIP inside a rigid middle+distal link.
    values[:, AXIAL_COMPONENT] = split + (1.0 - split) * values[:, AXIAL_COMPONENT]
    return SurfaceDescriptorSet(values, reference.azimuth_valid.copy())


def mano_terminal_frame(
    vertices: np.ndarray,
    joints: np.ndarray,
    vertex_ids: np.ndarray,
    *,
    finger_id: int,
    root_joint: int,
    palmar_normal: np.ndarray,
    side: str = "right",
) -> TerminalFrame:
    if vertex_ids.size == 0:
        raise ValueError(f"No MANO distal vertices for finger_id={finger_id}")
    root = joints[int(root_joint)]
    tip_vertex = MANO_TIP_VERTEX_BY_FINGER[int(finger_id)]
    longitudinal = normalize_vector(vertices[tip_vertex] - root, name="MANO terminal axis")
    length = float(np.max((vertices[vertex_ids] - root) @ longitudinal))
    if length <= SURFACE_EPS:
        raise ValueError(f"Invalid MANO terminal length for finger_id={finger_id}: {length}")
    tip = root + length * longitudinal
    frame = make_terminal_frame(root, tip, palmar_normal)
    if side == "left":
        # Preserve anatomical azimuth under left-to-right reflection.
        frame.basis[:, 0] *= -1.0
    return frame
