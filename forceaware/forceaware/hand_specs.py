"""Robot palm and fingertip names used to derive collision capsules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HandSpec:
    name: str
    palm_body: str
    tip_sites: dict[str, str]


ASSET_HANDS = (
    HandSpec(
        "leaphand",
        "palm",
        {"th": "th_tip", "if": "if_tip", "mf": "mf_tip", "rf": "rf_tip"},
    ),
    HandSpec(
        "allegro",
        "right_palm",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
        },
    ),
    HandSpec(
        "ability",
        "right_base",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
            "pk": "right_pinky_tip",
        },
    ),
    HandSpec(
        "inspire",
        "right_hand_base",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
            "pk": "right_pinky_tip",
        },
    ),
    HandSpec(
        "metahand",
        "right_palm",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
        },
    ),
    HandSpec(
        "schunk",
        "right_hand_base_link",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
            "pk": "right_pinky_tip",
        },
    ),
    HandSpec(
        "xhand",
        "right_hand_link",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
            "pk": "right_pinky_tip",
        },
    ),
    HandSpec(
        "sharpa",
        "right_hand_C_MC",
        {
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
            "pk": "right_pinky_tip",
        },
    ),
    HandSpec(
        "demo", "palm", {"th": "th_tip", "if": "if_tip", "mf": "mf_tip", "rf": "rf_tip"}
    ),
)
