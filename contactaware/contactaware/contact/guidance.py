"""Contact-guidance container and robot-trajectory accessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from contactaware.settings import FINGER_LABELS, LINK_LABELS
from contactaware.types import ObjectModel

if TYPE_CHECKING:
    from contactaware.solver.hand import MujocoHand

POSITION_DECIMALS = 6
METRIC_DECIMALS = 3


@dataclass(frozen=True)
class HandContactSequence:
    hand: MujocoHand
    qpos: np.ndarray
    obj_pose: np.ndarray
    obj: ObjectModel


def empty_contact_guidance(
    frame_count: int,
    link_count: int,
    labels: np.ndarray,
) -> dict:
    return {
        "contact_mask": np.zeros((frame_count, link_count), dtype=np.float32),
        "contact_weight": np.zeros((frame_count, link_count), dtype=np.float32),
        "contact_pos_obj": np.zeros((frame_count, link_count, 3), dtype=np.float32),
        "hand_query_indices": np.zeros((frame_count, link_count), dtype=np.int32),
        "hand_query_body_ids": np.zeros((frame_count, link_count), dtype=np.int32),
        "hand_query_local_pos": np.zeros((frame_count, link_count, 3), dtype=np.float32),
        "link_labels": labels,
    }


def link_label(finger: int, link: int) -> str:
    finger_label = FINGER_LABELS.get(finger, f"finger{finger}")
    link_name = LINK_LABELS.get(link, f"link{link}")
    return f"{finger_label}_{link_name}"
