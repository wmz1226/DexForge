"""CLI options and immutable configuration for contact-aware retargeting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from contactaware.types import SafeConfig

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
# The simulator shared with ForceAware: collision queries, contact fusion and object settling.
COMFREE_WARP_ROOT = Path(os.environ.get("COMFREE_WARP_ROOT", ROOT.parent / "forceaware" / "third_party" / "comfree_warp")).resolve()

np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.str = np.str_
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_

CONTACT_GUIDANCE_NAME = "contact_guidance.npz"
BASE_TRANSLATION_DIM = 3
BASE_ROTATION_DIM = 3
PALM_ANCHORS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.0, 0.04, 0.0],
        [0.0, 0.0, 0.04],
    ],
    dtype=np.float64,
)
MM = 1000.0
MANO_BASE_JOINT_BY_FINGER = {0: 5, 1: 9, 2: 13, 3: 1, 5: 17}
MANO_DISTAL_WEIGHT_JOINT_BY_FINGER = {0: 3, 1: 6, 2: 12, 3: 15, 5: 9}
MANO_TIP_JOINT_BY_FINGER = {0: 8, 1: 12, 2: 16, 3: 4, 5: 20}
FINGER_LABELS = {0: "if", 1: "mf", 2: "rf", 3: "th", 4: "palm", 5: "pk"}
LINK_LABELS = {0: "bs", 1: "px", 2: "md", 3: "ds", 4: "palm_4", 5: "palm_5"}
GROUND_PLANE_POS = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
GROUND_PLANE_NORMAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
XYZ_DIM = 3
MANO_CHAIN_JOINTS = {
    "th": (1, 2, 3, 4),
    "if": (5, 6, 7, 8),
    "mf": (9, 10, 11, 12),
    "rf": (13, 14, 15, 16),
    "pk": (17, 18, 19, 20),
}


def mano_base_joint_ids(finger_ids) -> np.ndarray:
    return np.asarray(
        [MANO_BASE_JOINT_BY_FINGER[int(x)] for x in finger_ids],
        dtype=np.int64,
    )


def mano_tip_joint_ids(finger_ids) -> np.ndarray:
    return np.asarray(
        [MANO_TIP_JOINT_BY_FINGER[int(x)] for x in finger_ids],
        dtype=np.int64,
    )


CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "contactaware.yaml"


def read_config_file(path: Path) -> dict:
    """Load one grouped hyperparameter file and flatten it to leaf names."""
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Hyperparameter file must be a mapping: {path}")
    values: dict = {}
    for group, entries in document.items():
        if not isinstance(entries, dict):
            raise ValueError(f"{path}: group {group!r} must be a mapping")
        for key, value in entries.items():
            if key in values:
                raise ValueError(f"{path}: duplicate hyperparameter {key!r}")
            values[key] = value
    return values


# The YAML is the single source of truth for hyperparameters.
DEFAULTS = read_config_file(DEFAULT_CONFIG_PATH)


def resolve_hyperparameters(
    config_path: Path | None,
    overrides: list[str],
) -> tuple[dict, dict]:
    values = dict(DEFAULTS)
    source = {
        "default": str(DEFAULT_CONFIG_PATH),
        "config": None,
        "overrides": {},
    }
    if config_path is not None:
        extra = read_config_file(config_path)
        reject_unknown_keys(extra, path=Path(config_path))
        values.update(extra)
        source["config"] = str(Path(config_path).resolve())
    parsed = parse_overrides(overrides)
    values.update(parsed)
    source["overrides"] = parsed
    return values, source


def reject_unknown_keys(values: dict, *, path: Path) -> None:
    """Fail on names contactaware.yaml does not define, exposing typos."""
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"{path} sets hyperparameters that {DEFAULT_CONFIG_PATH.name} "
            f"does not define: {unknown}"
        )


def parse_overrides(overrides: list[str]) -> dict:
    values: dict = {}
    for item in overrides:
        key, separator, raw = str(item).partition("=")
        if not separator:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key = key.strip()
        if key not in DEFAULTS:
            raise ValueError(f"--set uses an unknown hyperparameter: {key!r}")
        values[key] = yaml.safe_load(raw)
    return values


def config_report(args) -> dict:
    return {
        key: str(value) if isinstance(value := getattr(args, key), Path) else value
        for key in DEFAULTS
    }


def parse_args() -> argparse.Namespace:
    """Parse the run target; every hyperparameter comes from the YAML config."""
    parser = argparse.ArgumentParser(
        description=(
            "Contact-aware MANO-to-robot-hand retargeting. Hyperparameters are "
            f"defined in {DEFAULT_CONFIG_PATH}; use --config/--set to change them."
        )
    )
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <sequence-dir>/retarget/<hand>/contactaware")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="hyperparameter file whose values override contactaware.yaml",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a single hyperparameter (repeatable)",
    )
    args = parser.parse_args()
    values, source = resolve_hyperparameters(args.config, args.overrides)
    for key, value in values.items():
        setattr(args, key, value)
    args.hyperparameter_source = source
    return args


def make_config(args) -> SafeConfig:
    return SafeConfig(
        **{key: getattr(args, key) for key in SafeConfig.__dataclass_fields__}
    )
