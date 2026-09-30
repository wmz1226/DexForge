"""One collision configuration for geometry queries, observations and simulation."""
from dataclasses import dataclass
import math

from .collision_targets import MAX_TOPK, SPHERE_TARGET

CONTACT_FRAME_VJP_PROFILE = "gs_hard_stop_frame_other_full_derivative"
OBSERVATION_FRAME_VJP_PROFILE = CONTACT_FRAME_VJP_PROFILE
FREEZE_TANGENT_GAUGE_VJP = False


@dataclass(frozen=True)
class CollisionConfig:
    contact_topk: int = 1
    distance_offset: float = 0.

    def __post_init__(self):
        if int(self.contact_topk) != self.contact_topk or not 1 <= self.contact_topk <= MAX_TOPK:
            raise ValueError(f"contact_topk must be in [1, {MAX_TOPK}]")
        if not math.isfinite(self.distance_offset):
            raise ValueError("Contact distance offset must be finite")


def stop_normal_gradient(target_kind, contact_topk):
    """Only hard Gaussian contact frames are detached; distance/position remain differentiable."""
    # GS hard selects a single Gaussian, so surface normals can jump when the
    # active Gaussian changes. This lack of smoothness often makes normal
    # gradients unreliable over finite optimization steps. Stop the normal/frame
    # path while preserving distance and contact-position derivatives.
    return int(target_kind) == SPHERE_TARGET and int(contact_topk) == 1


def configure_collision(device_model, config: CollisionConfig):
    collision = getattr(device_model, 'gaussian_collision', None)
    if collision is None:
        raise ValueError("Scene does not contain query-point collision data")
    for batch in collision if isinstance(collision, tuple) else (collision,):
        batch.distance_offset.fill_(float(config.distance_offset))
        batch.contact_topk = int(config.contact_topk)
