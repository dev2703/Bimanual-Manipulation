"""Physical dinner-skill success predicates shared by expert and policy eval."""

from __future__ import annotations

import numpy as np


def mug_placed(
    initial: np.ndarray,
    final: np.ndarray,
    target: np.ndarray,
    *,
    peak_height: float,
    carried_to_target: bool,
    upright_cosine: float,
    gripper_opening: float,
) -> bool:
    """Require a real carried transfer and a released, upright final mug."""
    return bool(
        np.linalg.norm(initial[:2] - target[:2]) > 0.10
        and peak_height > initial[2] + 0.10
        and carried_to_target
        and np.linalg.norm(final[:2] - target[:2]) < 0.045
        and abs(final[2] - 0.029) < 0.012
        and upright_cosine > 0.90
        and gripper_opening > 0.03
    )
