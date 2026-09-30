"""Declarative asset locations and topology metadata for supported robot hands."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from contactaware.settings import ROOT

MERGED_NON_THUMB_FINGERS = ("if", "mf", "rf", "pk")
SCHUNK_SHAPE_POINTS = {
    "th": (
        ("body", "right_hand_a"),
        ("body", "right_hand_b"),
        ("body", "right_hand_c"),
        ("site", "right_thumb_tip"),
    ),
    "if": (
        ("body", "right_hand_virtual_l"),
        ("body", "right_hand_p"),
        ("body", "right_hand_t"),
        ("site", "right_index_tip"),
    ),
    "mf": (
        ("body", "right_hand_k"),
        ("body", "right_hand_o"),
        ("body", "right_hand_s"),
        ("site", "right_middle_tip"),
    ),
    "rf": (
        ("body", "right_hand_virtual_j"),
        ("body", "right_hand_n"),
        ("body", "right_hand_r"),
        ("site", "right_ring_tip"),
    ),
    "pk": (
        ("body", "right_hand_virtual_i"),
        ("body", "right_hand_m"),
        ("body", "right_hand_q"),
        ("site", "right_pinky_tip"),
    ),
}


@dataclass(frozen=True)
class AssetHandSpec:
    name: str
    mesh_xml: Path
    qp_dir: Path
    palm_body: str
    tip_sites: dict[str, str]
    shape_points: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    palm_site: str = "right_palm"
    default_qpos: tuple[float, ...] | None = None
    merged_terminal_fingers: tuple[str, ...] = ()
    # Use -1 for fingers whose terminal hinge closes as q decreases.
    terminal_flexion_signs: dict[str, float] = field(default_factory=dict)


def asset_path(*parts: str) -> Path:
    return Path(os.environ.get("HAND_ASSETS_ROOT", ROOT.parent / "assets")).joinpath(*parts)


def standard_tip_sites() -> dict[str, str]:
    return {
        "th": "right_thumb_tip",
        "if": "right_index_tip",
        "mf": "right_middle_tip",
        "rf": "right_ring_tip",
        "pk": "right_pinky_tip",
    }


ASSET_HANDS = (
    AssetHandSpec(
        name="leaphand",
        mesh_xml=asset_path("leaphand", "leaphand_mesh.xml"),
        qp_dir=asset_path("leaphand", "qp"),
        palm_body="palm",
        tip_sites={"th": "th_tip", "if": "if_tip", "mf": "mf_tip", "rf": "rf_tip"},
        shape_points={
            "th": (("body", "th_bs"), ("body", "th_px"),
                   ("body", "th_ds"), ("site", "th_tip")),
            "if": (("body", "if_px"), ("body", "if_md"),
                   ("body", "if_ds"), ("site", "if_tip")),
            "mf": (("body", "mf_px"), ("body", "mf_md"),
                   ("body", "mf_ds"), ("site", "mf_tip")),
            "rf": (("body", "rf_px"), ("body", "rf_md"),
                   ("body", "rf_ds"), ("site", "rf_tip")),
        },
        palm_site="",
        default_qpos=(
            0, 0, 0, 0, 0, 0,
            0.45, 0.00, 0.35, 0.25,
            0.45, 0.00, 0.35, 0.25,
            0.45, 0.00, 0.35, 0.25,
            0.40, 0.40, 0.35, 0.20,
        ),
    ),
    AssetHandSpec(
        name="allegro",
        mesh_xml=asset_path("allegro", "right.xml"),
        qp_dir=asset_path("allegro", "qp_right"),
        palm_body="right_palm",
        tip_sites={
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
        },
        default_qpos=(
            0, 0, 0, 0, 0, 0,
            0.00, 0.45, 0.45, 0.35,
            0.00, 0.45, 0.45, 0.35,
            0.00, 0.45, 0.45, 0.35,
            0.50, 0.40, 0.50, 0.40,
        ),
    ),
    AssetHandSpec(
        name="ability",
        mesh_xml=asset_path("ability", "right.xml"),
        qp_dir=asset_path("ability", "qp_right"),
        palm_body="right_base",
        tip_sites=standard_tip_sites(),
    ),
    AssetHandSpec(
        name="inspire",
        mesh_xml=asset_path("inspire", "right.xml"),
        qp_dir=asset_path("inspire", "qp_right"),
        palm_body="right_hand_base",
        tip_sites=standard_tip_sites(),
        merged_terminal_fingers=MERGED_NON_THUMB_FINGERS,
    ),
    AssetHandSpec(
        name="metahand",
        mesh_xml=asset_path("metahand", "right.xml"),
        qp_dir=asset_path("metahand", "qp_right"),
        palm_body="right_palm",
        tip_sites={
            "th": "right_thumb_tip",
            "if": "right_index_tip",
            "mf": "right_middle_tip",
            "rf": "right_ring_tip",
        },
    ),
    AssetHandSpec(
        name="schunk",
        mesh_xml=asset_path("schunk", "right.xml"),
        qp_dir=asset_path("schunk", "qp_right"),
        palm_body="right_hand_base_link",
        tip_sites=standard_tip_sites(),
        shape_points=SCHUNK_SHAPE_POINTS,
    ),
    AssetHandSpec(
        name="xhand",
        mesh_xml=asset_path("xhand", "right.xml"),
        qp_dir=asset_path("xhand", "qp_right"),
        palm_body="right_hand_link",
        tip_sites=standard_tip_sites(),
        merged_terminal_fingers=MERGED_NON_THUMB_FINGERS,
    ),
    AssetHandSpec(
        name="sharpa",
        mesh_xml=asset_path("sharpa", "right.xml"),
        qp_dir=asset_path("sharpa", "qp_right"),
        palm_body="right_hand_C_MC",
        tip_sites=standard_tip_sites(),
    ),
)
