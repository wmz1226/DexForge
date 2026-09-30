"""Soft contact-position weights over the approach/release schedule."""


import numpy as np


def refined_contact_targets(reference_targets, blend, anchor_shifts):
    """Apply the same optimized-anchor displacement at keys and runtime frames."""
    return reference_targets + np.asarray(blend, dtype=np.float64)[..., None] * anchor_shifts


def contact_position_weights(cfg, participation, blend, multiplier):
    return (cfg.contact_anchor_weight * np.asarray(participation, dtype=np.float64)
            * (1.0 + (multiplier - 1.0) * np.asarray(blend, dtype=np.float64)))


def validate_position_multiplier(multiplier):
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("Contact position multiplier must be finite and at least one")


