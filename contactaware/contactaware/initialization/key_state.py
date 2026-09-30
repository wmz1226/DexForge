"""Seed-frame selection and the seed's anchor mapping into the trajectory runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from contactaware.types import ContactAnchorSet, ContactRuntime
from contactaware.solver.contact_position import refined_contact_targets

DEFAULT_CONTACT_AXIAL_CENTER = 0.5


@dataclass(frozen=True)
class ContactKeySelection:
    source_frame: int
    internal_frame: int
    active_finger_count: int
    active_finger_ids: np.ndarray
    active_anchor_columns: np.ndarray
    per_source_frame_finger_counts: np.ndarray


@dataclass(frozen=True)
class ContactKeySeed:
    qpos: np.ndarray
    query_ids: np.ndarray
    anchor_columns: np.ndarray
    anchor_shifts_obj: np.ndarray
    contact_normals_obj: np.ndarray
    source_frame: int
    internal_frame: int


    def adapt_runtime(self, runtime: ContactRuntime) -> ContactRuntime:
        query_ids = runtime.hand_query_ids.copy()
        query_ids[:, self.anchor_columns] = self.query_ids[None, :]
        targets = runtime.contact_target_pos_obj.copy()
        targets[:, self.anchor_columns] = refined_contact_targets(
            targets[:, self.anchor_columns], runtime.guidance_blend[:, self.anchor_columns],
            self.anchor_shifts_obj)
        normals = runtime.anchors.anchor_normal_obj.copy()
        normals[:, self.anchor_columns] = self.contact_normals_obj[None, :, :]
        return replace(
            runtime,
            anchors=replace(runtime.anchors, anchor_normal_obj=normals),
            hand_query_ids=query_ids,
            contact_target_pos_obj=targets,
        )


def stable_finger_counts(anchors: ContactAnchorSet) -> np.ndarray:
    """Count distinct stable-contact fingers at each source frame."""
    mask = np.asarray(anchors.mask, dtype=bool)
    fingers = np.asarray(anchors.finger_ids, dtype=np.int32)
    if mask.ndim != 2 or mask.shape[1] != len(fingers):
        raise ValueError("Stable-contact mask and finger IDs have incompatible shapes")
    return np.asarray(
        [len(np.unique(fingers[row])) for row in mask],
        dtype=np.int32,
    )


def select_contact_key(
    anchors: ContactAnchorSet,
    *,
    interpolation_ratio: int,
) -> ContactKeySelection:
    """Select the first source frame attaining the maximum stable-finger count."""
    if interpolation_ratio <= 0:
        raise ValueError("Interpolation ratio must be positive")
    counts = stable_finger_counts(anchors)
    maximum = int(np.max(counts, initial=0))
    if maximum == 0:
        raise ValueError("Contact-key initialization requires a stable contact frame")
    source_frame = int(np.flatnonzero(counts == maximum)[0])
    columns = np.flatnonzero(anchors.mask[source_frame]).astype(np.int32)
    fingers = np.unique(anchors.finger_ids[columns]).astype(np.int32)
    return ContactKeySelection(
        source_frame=source_frame,
        internal_frame=source_frame * interpolation_ratio,
        active_finger_count=maximum,
        active_finger_ids=fingers,
        active_anchor_columns=columns,
        per_source_frame_finger_counts=counts,
    )


