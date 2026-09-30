"""Unit conversion for ForceAware objective and gradient scaling."""

from __future__ import annotations


MILLIMETERS_PER_METER = 1000.0
LOSS_GRADIENT_SCALE = 1.0 / (MILLIMETERS_PER_METER * MILLIMETERS_PER_METER)
