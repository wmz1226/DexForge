from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

from .time_grid import SolveRequest, TimeGrid


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/forceaware.yaml"
EXEC_BACKENDS = ("comfree",)
CONTACT_DYNAMICS_MODES = ("hard",)
MIN_TRAJECTORY_FRAMES = 2
OBJECT_POSE_DIM = 7
CONFIG_SCHEMA = {
    "sequence": frozenset(("hand", "scene", "start", "hold_frame", "hold_seconds")),
    "simulator": frozenset(
        (
            "exec_backend",
            "mpc_dt",
            "action_dt",
            "knot_dt",
            "exec_dt",
            "ref_dt",
        )
    ),
    "contact_physics": frozenset(("gs_distance_offset", "contact_topk")),
    "optimizer": frozenset(
        (
            "horizon",
            "n_iter",
            "contact_dynamics",
            "lr_max",
            "lr_min",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "grad_clip",
            "warm_shift_scale",
            "multistart_count",
            "multistart_seed",
            "multistart_init_std",
            "multistart_ctrl_std",
            "multistart_init_raw_limit",
        )
    ),
    "loss": frozenset(
        (
            "obj_pos",
            "obj_rot",
            "terminal",
            "reference_pose_scale",
            "init",
            "action_ref_base",
            "action_ref_base_rot",
            "action_ref_finger",
            "contact_robustness",
            "marked_query_pos",
            "finger_joint_collision",
            "soft_penetration",
            "penetration",
        )
    ),
    "smoothing": frozenset(
        (
            "enabled",
            "control_velocity_weight",
            "control_acceleration_weight",
            "hand_state_weight",
            "control_velocity_base_position_scale",
            "control_velocity_base_rotation_scale",
            "control_velocity_finger_scale",
            "control_acceleration_base_position_scale",
            "control_acceleration_base_rotation_scale",
            "control_acceleration_finger_scale",
        )
    ),
    "contact": frozenset(
        (
            "marked_query_dist_mm",
            "marked_query_temp_mm",
            "age_ramp_frames",
            "target_phi",
            "penetration_limit",
            "finger_joint_surface_margin_mm",
        )
    ),
    "bounds": frozenset(("init_base_pos", "init_base_rot", "init_finger")),
    "video": frozenset(("fps",)),
}


def _format_key_error(label: str, keys: set[str]) -> str:
    return f"{label}: {', '.join(sorted(keys))}"


def validate_config_schema(config: dict) -> None:
    expected_sections = set(CONFIG_SCHEMA)
    actual_sections = set(config)
    missing_sections = expected_sections - actual_sections
    unexpected_sections = actual_sections - expected_sections
    if missing_sections:
        raise KeyError(_format_key_error("Missing config sections", missing_sections))
    if unexpected_sections:
        raise KeyError(
            _format_key_error("Unexpected config sections", unexpected_sections)
        )
    for section, expected_keys in CONFIG_SCHEMA.items():
        values = config[section]
        if not isinstance(values, dict):
            raise TypeError(f"Config section must be a mapping: {section}")
        actual_keys = set(values)
        missing_keys = set(expected_keys) - actual_keys
        unexpected_keys = actual_keys - set(expected_keys)
        if missing_keys:
            raise KeyError(
                _format_key_error(f"Missing config keys in {section}", missing_keys)
            )
        if unexpected_keys:
            raise KeyError(
                _format_key_error(
                    f"Unexpected config keys in {section}", unexpected_keys
                )
            )


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing retarget config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Retarget config must be a mapping: {path}")
    validate_config_schema(data)
    return data


def config_value(config: dict, path: str):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config key: {path}")
        value = value[key]
    if value is None:
        raise ValueError(f"Config key must not be null: {path}")
    return value


def cfg_str(config: dict, path: str) -> str:
    value = config_value(config, path)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Config key must be a non-empty string: {path}")
    return value


def cfg_int(config: dict, path: str) -> int:
    value = config_value(config, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key must be an integer: {path}")
    return value


def cfg_float(config: dict, path: str) -> float:
    value = config_value(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Config key must be numeric: {path}")
    return float(value)


def cfg_bool(config: dict, path: str) -> bool:
    value = config_value(config, path)
    if not isinstance(value, bool):
        raise TypeError(f"Config key must be boolean: {path}")
    return value


def object_xml_stem(seq_dir: Path) -> str:
    meta = json.loads((seq_dir / "mano_raw/meta.json").read_text(encoding="utf-8"))
    object_name = str(meta["object"])
    prefix, _, rest = object_name.partition("_")
    return rest if prefix.isdigit() and rest else object_name


def contactaware_last_frame(contact_guidance_npz: Path, hand: str) -> int:
    output_dir = contact_guidance_npz.parent
    hand_path = output_dir / f"{hand}_qpos.npy"
    object_path = output_dir / "object_pose_7.npy"
    for label, path in (
        ("hand trajectory", hand_path),
        ("object trajectory", object_path),
        ("contact guidance", contact_guidance_npz),
    ):
        if path.exists():
            continue
        raise FileNotFoundError(f"Missing ContactAware {label}: {path}")

    hand_trajectory = np.load(hand_path, mmap_mode="r", allow_pickle=False)
    object_trajectory = np.load(object_path, mmap_mode="r", allow_pickle=False)
    with np.load(contact_guidance_npz, allow_pickle=False) as guidance:
        if "contact_mask" not in guidance:
            raise KeyError(
                f"Contact guidance has no contact_mask: {contact_guidance_npz}"
            )
        contact_mask = guidance["contact_mask"]

    if hand_trajectory.ndim != 2:
        raise ValueError(
            f"ContactAware hand trajectory must be rank-2: {hand_path} "
            f"shape={hand_trajectory.shape}"
        )
    frame_count = int(hand_trajectory.shape[0])
    if frame_count < MIN_TRAJECTORY_FRAMES:
        raise ValueError(
            "ContactAware trajectory must contain at least two frames: "
            f"{hand_path} shape={hand_trajectory.shape}"
        )
    if object_trajectory.shape != (frame_count, OBJECT_POSE_DIM):
        raise ValueError(
            "ContactAware hand/object frame counts must match: "
            f"hand={hand_trajectory.shape}, object={object_trajectory.shape}"
        )
    if contact_mask.ndim != 2 or contact_mask.shape[0] != frame_count:
        raise ValueError(
            "ContactAware hand/contact frame counts must match: "
            f"hand={hand_trajectory.shape}, contact={contact_mask.shape}"
        )
    return frame_count - 1


@dataclass(frozen=True)
class LossWeights:
    obj_pos: float
    obj_rot: float
    terminal: float
    reference_pose_scale: float
    init: float
    action_ref_base: float
    action_ref_base_rot: float
    action_ref_finger: float
    contact_robustness: float
    marked_query_pos: float
    soft_penetration: float
    penetration: float
    finger_joint_collision: float


@dataclass(frozen=True)
class SmoothingConfig:
    enabled: bool
    control_velocity_weight: float
    control_acceleration_weight: float
    hand_state_weight: float
    control_velocity_base_position_scale: float
    control_velocity_base_rotation_scale: float
    control_velocity_finger_scale: float
    control_acceleration_base_position_scale: float
    control_acceleration_base_rotation_scale: float
    control_acceleration_finger_scale: float


@dataclass(frozen=True)
class BoundsConfig:
    init_base_pos: float
    init_base_rot: float
    init_finger: float


@dataclass(frozen=True)
class OptimizerConfig:
    horizon: int
    n_iter: int
    contact_dynamics: str
    lr_max: float
    lr_min: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    grad_clip: float
    warm_shift_scale: float
    multistart_count: int
    multistart_seed: int
    multistart_init_std: float
    multistart_ctrl_std: float
    multistart_init_raw_limit: float


@dataclass(frozen=True)
class ContactConfig:
    target_phi: float
    penetration_limit: float
    marked_query_dist_mm: float
    marked_query_temp_mm: float
    age_ramp_frames: int
    finger_joint_surface_margin_mm: float

    @property
    def soft_penetration_limit(self) -> float:
        """Shared penetration depth for contact encouragement and its soft penalty."""
        return -self.target_phi


@dataclass(frozen=True)
class SimConfig:
    time_grid: TimeGrid
    exec_backend: str
    comfree_warp_root: Path
    gs_distance_offset: float
    contact_topk: int


@dataclass(frozen=True)
class SeqConfig:
    seq_dir: Path
    hand: str
    xml: Path
    contact_guidance_npz: Path
    start: int
    end: int
    hold_frame: int
    hold_seconds: float
    output_dir_name: str


@dataclass(frozen=True)
class VideoConfig:
    fps: float


@dataclass(frozen=True)
class RetargetConfig:
    loss: LossWeights
    smoothing: SmoothingConfig
    bounds: BoundsConfig
    optimizer: OptimizerConfig
    contact: ContactConfig
    simulator: SimConfig
    sequence: SeqConfig
    video: VideoConfig


def _read_sequence(
    config: dict,
    sequence_dir: Path,
    contactaware_dir: Path | None = None,
) -> SeqConfig:
    seq_dir = sequence_dir.expanduser().resolve()
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")
    hand = cfg_str(config, "sequence.hand")
    contactaware_dir = (seq_dir / "retarget" / hand / "contactaware" if contactaware_dir is None
                        else Path(contactaware_dir).expanduser().resolve())
    contact_npz = contactaware_dir / "contact_guidance.npz"
    scene = cfg_str(config, "sequence.scene")
    xml = seq_dir / "scene" / hand / f"{object_xml_stem(seq_dir)}_{scene}.xml"
    start = cfg_int(config, "sequence.start")
    last_frame = contactaware_last_frame(contact_npz, hand)
    hold_frame = cfg_int(config, "sequence.hold_frame")
    hold_seconds = cfg_float(config, "sequence.hold_seconds")
    if hold_frame > last_frame:
        raise ValueError(
            "sequence.hold_frame exceeds ContactAware last frame: "
            f"hold_frame={hold_frame}, last_frame={last_frame}"
        )
    return SeqConfig(
        seq_dir=seq_dir,
        hand=hand,
        xml=xml,
        contact_guidance_npz=contact_npz,
        start=start,
        end=last_frame,
        hold_frame=hold_frame,
        hold_seconds=hold_seconds,
        output_dir_name="",
    )


def _read_optimizer(config: dict) -> OptimizerConfig:
    return OptimizerConfig(
        horizon=cfg_int(config, "optimizer.horizon"),
        n_iter=cfg_int(config, "optimizer.n_iter"),
        contact_dynamics=cfg_str(config, "optimizer.contact_dynamics"),
        lr_max=cfg_float(config, "optimizer.lr_max"),
        lr_min=cfg_float(config, "optimizer.lr_min"),
        adam_beta1=cfg_float(config, "optimizer.adam_beta1"),
        adam_beta2=cfg_float(config, "optimizer.adam_beta2"),
        adam_epsilon=cfg_float(config, "optimizer.adam_epsilon"),
        grad_clip=cfg_float(config, "optimizer.grad_clip"),
        warm_shift_scale=cfg_float(config, "optimizer.warm_shift_scale"),
        multistart_count=cfg_int(config, "optimizer.multistart_count"),
        multistart_seed=cfg_int(config, "optimizer.multistart_seed"),
        multistart_init_std=cfg_float(config, "optimizer.multistart_init_std"),
        multistart_ctrl_std=cfg_float(config, "optimizer.multistart_ctrl_std"),
        multistart_init_raw_limit=cfg_float(
            config, "optimizer.multistart_init_raw_limit"
        ),
    )


def _read_loss(config: dict) -> LossWeights:
    return LossWeights(
        obj_pos=cfg_float(config, "loss.obj_pos"),
        obj_rot=cfg_float(config, "loss.obj_rot"),
        terminal=cfg_float(config, "loss.terminal"),
        reference_pose_scale=cfg_float(config, "loss.reference_pose_scale"),
        init=cfg_float(config, "loss.init"),
        action_ref_base=cfg_float(config, "loss.action_ref_base"),
        action_ref_base_rot=cfg_float(config, "loss.action_ref_base_rot"),
        action_ref_finger=cfg_float(config, "loss.action_ref_finger"),
        contact_robustness=cfg_float(config, "loss.contact_robustness"),
        marked_query_pos=cfg_float(config, "loss.marked_query_pos"),
        soft_penetration=cfg_float(config, "loss.soft_penetration"),
        penetration=cfg_float(config, "loss.penetration"),
        finger_joint_collision=cfg_float(config, "loss.finger_joint_collision"),
    )


def _read_smoothing(config: dict) -> SmoothingConfig:
    return SmoothingConfig(
        enabled=cfg_bool(config, "smoothing.enabled"),
        control_velocity_weight=cfg_float(config, "smoothing.control_velocity_weight"),
        control_acceleration_weight=cfg_float(
            config, "smoothing.control_acceleration_weight"
        ),
        hand_state_weight=cfg_float(config, "smoothing.hand_state_weight"),
        control_velocity_base_position_scale=cfg_float(
            config, "smoothing.control_velocity_base_position_scale"
        ),
        control_velocity_base_rotation_scale=cfg_float(
            config, "smoothing.control_velocity_base_rotation_scale"
        ),
        control_velocity_finger_scale=cfg_float(
            config, "smoothing.control_velocity_finger_scale"
        ),
        control_acceleration_base_position_scale=cfg_float(
            config, "smoothing.control_acceleration_base_position_scale"
        ),
        control_acceleration_base_rotation_scale=cfg_float(
            config, "smoothing.control_acceleration_base_rotation_scale"
        ),
        control_acceleration_finger_scale=cfg_float(
            config, "smoothing.control_acceleration_finger_scale"
        ),
    )


def _read_bounds(config: dict) -> BoundsConfig:
    return BoundsConfig(
        init_base_pos=cfg_float(config, "bounds.init_base_pos"),
        init_base_rot=cfg_float(config, "bounds.init_base_rot"),
        init_finger=cfg_float(config, "bounds.init_finger"),
    )


def _read_contact(config: dict) -> ContactConfig:
    return ContactConfig(
        target_phi=cfg_float(config, "contact.target_phi"),
        penetration_limit=cfg_float(config, "contact.penetration_limit"),
        marked_query_dist_mm=cfg_float(config, "contact.marked_query_dist_mm"),
        marked_query_temp_mm=cfg_float(config, "contact.marked_query_temp_mm"),
        age_ramp_frames=cfg_int(config, "contact.age_ramp_frames"),
        finger_joint_surface_margin_mm=cfg_float(
            config, "contact.finger_joint_surface_margin_mm"
        ),
    )


def _read_simulator(
    config: dict,
    time_grid: TimeGrid,
) -> SimConfig:
    exec_backend = cfg_str(config, "simulator.exec_backend")
    if exec_backend not in EXEC_BACKENDS:
        raise ValueError(f"simulator.exec_backend must be one of {EXEC_BACKENDS}")
    return SimConfig(
        time_grid=time_grid,
        exec_backend=exec_backend,
        comfree_warp_root=ROOT / "third_party/comfree_warp",
        gs_distance_offset=cfg_float(config, "contact_physics.gs_distance_offset"),
        contact_topk=cfg_int(config, "contact_physics.contact_topk"),
    )


def _fill_output_dir(
    seq: SeqConfig,
    simulator: SimConfig,
    output_method: str,
) -> SeqConfig:
    method = output_method
    if simulator.exec_backend != "comfree":
        method = f"{method}_{simulator.exec_backend}"
    output = f"retarget/{seq.hand}/{method}"
    return replace(seq, output_dir_name=output)


def resolved_config_dict(config: RetargetConfig) -> dict:
    value = asdict(config)
    simulator = value["simulator"]
    time_grid = simulator.pop("time_grid")
    time_grid.pop("executor_substeps_per_mpc_step")
    simulator.update(time_grid)
    simulator["exec_substeps"] = config.simulator.time_grid.executor_substeps_per_action
    value["optimizer"]["n_mpc_steps"] = len(
        make_solve_request(config).execution_steps_per_window
    )
    return value


def _require_finite_fields(section: str, value: object) -> None:
    for field in fields(value):
        scalar = getattr(value, field.name)
        if isinstance(scalar, (int, float)) and not np.isfinite(scalar):
            raise ValueError(f"{section}.{field.name} must be finite, got {scalar}")


def _require_nonnegative_fields(section: str, value: object) -> None:
    _require_finite_fields(section, value)
    for field in fields(value):
        scalar = getattr(value, field.name)
        if isinstance(scalar, (int, float)) and scalar < 0:
            raise ValueError(f"{section}.{field.name} must be >= 0, got {scalar}")


def _validate_bounds(bounds: BoundsConfig) -> None:
    _require_finite_fields("bounds", bounds)
    for field in fields(bounds):
        value = getattr(bounds, field.name)
        if value <= 0.0:
            raise ValueError(f"bounds.{field.name} must be positive")


def _validate_optimizer(optimizer: OptimizerConfig) -> None:
    _require_finite_fields("optimizer", optimizer)
    limits = (
        (("horizon", "n_iter", "multistart_count"), 1.0, True, ">= 1"),
        (
            ("lr_max", "lr_min", "adam_epsilon", "multistart_init_raw_limit"),
            0.0,
            False,
            "> 0",
        ),
        (
            (
                "grad_clip",
                "warm_shift_scale",
                "multistart_init_std",
                "multistart_ctrl_std",
            ),
            0.0,
            True,
            ">= 0",
        ),
    )
    for names, bound, inclusive, requirement in limits:
        for name in names:
            value = getattr(optimizer, name)
            invalid = value < bound if inclusive else value <= bound
            if invalid:
                raise ValueError(f"optimizer.{name} must be {requirement}, got {value}")
    if optimizer.lr_min > optimizer.lr_max:
        raise ValueError("optimizer.lr_min must be <= optimizer.lr_max")
    for name in ("adam_beta1", "adam_beta2"):
        value = getattr(optimizer, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"optimizer.{name} must be in [0, 1), got {value}")
    if optimizer.contact_dynamics not in CONTACT_DYNAMICS_MODES:
        raise ValueError(
            "optimizer.contact_dynamics must be one of "
            f"{CONTACT_DYNAMICS_MODES}, got {optimizer.contact_dynamics!r}"
        )


def _validate_contact(contact: ContactConfig) -> None:
    _require_finite_fields("contact", contact)
    limits = (
        contact.soft_penetration_limit,
        contact.penetration_limit,
    )
    if min(limits) < 0.0 or limits != tuple(sorted(limits)):
        raise ValueError("contact penetration limits must be nonnegative and ordered")
    nonnegative = (
        contact.marked_query_dist_mm,
        contact.finger_joint_surface_margin_mm,
    )
    if min(nonnegative) < 0.0:
        raise ValueError("contact distance thresholds must be nonnegative")
    if contact.marked_query_temp_mm <= 0.0:
        raise ValueError("contact temperatures must be positive")
    if contact.age_ramp_frames < 0:
        raise ValueError("contact.age_ramp_frames must be >= 0")


def _validate_smoothing(smoothing: SmoothingConfig) -> None:
    _require_nonnegative_fields("smoothing", smoothing)
    scales = (
        smoothing.control_velocity_base_position_scale,
        smoothing.control_velocity_base_rotation_scale,
        smoothing.control_velocity_finger_scale,
        smoothing.control_acceleration_base_position_scale,
        smoothing.control_acceleration_base_rotation_scale,
        smoothing.control_acceleration_finger_scale,
    )
    if min(scales) <= 0.0:
        raise ValueError("smoothing normalization scales must be positive")


def validate_retarget_config(config: RetargetConfig) -> RetargetConfig:
    sequence = config.sequence
    if sequence.start < 0 or sequence.end <= sequence.start:
        raise ValueError("sequence must satisfy 0 <= start < end")
    if sequence.hold_seconds < 0.0 or not np.isfinite(sequence.hold_seconds):
        raise ValueError("sequence.hold_seconds must be finite and nonnegative")
    if sequence.hold_frame >= 0 and sequence.hold_seconds <= 0.0:
        raise ValueError("sequence.hold_frame requires positive hold_seconds")
    _require_nonnegative_fields("loss", config.loss)
    _validate_smoothing(config.smoothing)
    _validate_bounds(config.bounds)
    _validate_optimizer(config.optimizer)
    _validate_contact(config.contact)
    if config.video.fps <= 0.0 or not np.isfinite(config.video.fps):
        raise ValueError("video.fps must be finite and positive")
    return config


def resolve_retarget_config(
    config: dict,
    sequence_dir: Path,
    *,
    output_method: str = "forceaware",
    contactaware_dir: Path | None = None,
) -> RetargetConfig:
    validate_config_schema(config)
    seq = _read_sequence(config, sequence_dir, contactaware_dir)
    time_grid = TimeGrid.create(
        mpc_dt=cfg_float(config, "simulator.mpc_dt"),
        action_dt=cfg_float(config, "simulator.action_dt"),
        knot_dt=cfg_float(config, "simulator.knot_dt"),
        exec_dt=cfg_float(config, "simulator.exec_dt"),
        ref_dt=cfg_float(config, "simulator.ref_dt"),
    )
    simulator = _read_simulator(config, time_grid)
    seq = _fill_output_dir(seq, simulator, output_method)
    return validate_retarget_config(
        RetargetConfig(
            loss=_read_loss(config),
            smoothing=_read_smoothing(config),
            bounds=_read_bounds(config),
            optimizer=_read_optimizer(config),
            contact=_read_contact(config),
            simulator=simulator,
            sequence=seq,
            video=VideoConfig(cfg_float(config, "video.fps")),
        )
    )


def make_solve_request(config: RetargetConfig) -> SolveRequest:
    simulator = config.simulator
    sequence = config.sequence
    if sequence.hold_frame >= 0:
        duration_seconds = sequence.hold_seconds
    else:
        duration_seconds = (sequence.end - sequence.start) * simulator.time_grid.ref_dt
    return SolveRequest.create(
        simulator.time_grid,
        duration_seconds=duration_seconds,
    )


def load_retarget_config(
    sequence_dir: Path, config_path: Path = CONFIG_PATH, *, hand: str | None = None,
    contactaware_dir: Path | None = None
) -> RetargetConfig:
    values = load_config(config_path)
    if hand is not None:
        values = {**values, "sequence": {**values["sequence"], "hand": hand}}
    cfg = resolve_retarget_config(values, sequence_dir, contactaware_dir=contactaware_dir)
    if not cfg.sequence.xml.is_file():
        raise FileNotFoundError(f"Missing scene: {cfg.sequence.xml}")
    return cfg


def config_dict(cfg: RetargetConfig) -> dict:
    return resolved_config_dict(cfg)
