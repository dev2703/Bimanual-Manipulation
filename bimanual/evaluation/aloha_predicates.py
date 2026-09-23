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


def plate_placed(
    initial: np.ndarray,
    final: np.ndarray,
    target: np.ndarray,
    *,
    peak_height: float,
    carried_to_target: bool,
    upright_cosine: float,
    gripper_opening: float,
) -> bool:
    """Require an airborne transfer followed by an upright released plate."""
    return bool(
        np.linalg.norm(initial[:2] - target[:2]) > 0.08
        and peak_height > initial[2] + 0.08
        and carried_to_target
        and np.linalg.norm(final[:2] - target[:2]) < 0.04
        and abs(final[2] - 0.011) < 0.01
        and upright_cosine > 0.95
        and gripper_opening > 0.03
    )


def drawer_opened(
    initial_opening: float,
    peak_opening: float,
    final_opening: float,
    *,
    handle_contact_steps: int,
    gripper_opening: float,
) -> bool:
    """Require a released, retained drawer opening made with handle contact."""
    return bool(
        abs(initial_opening) < 0.01
        and peak_opening > 0.12
        and final_opening > 0.11
        and handle_contact_steps >= 20
        and gripper_opening > 0.03
    )
