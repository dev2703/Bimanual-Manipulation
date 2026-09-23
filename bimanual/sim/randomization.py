"""Randomization levels L0-L2 (plan.md section 26; L3-L5 are later phases).

Each level returns a SceneConfig built from a seeded RNG, so the same seed
always yields the same scene (decision: env.reset(seed) must be
deterministic, gate-tested in tests/test_scene.py).

L0: nominal -- fixed scene, fixed object placement (SceneConfig defaults).
L1: geometric -- object xy jitter, drawer initial opening.
L2: L1 + visual -- object/table color jitter.

L3 (physical: mass/friction) and L4 (dynamic: mid-episode perturbation)
land in Phase 3 alongside experts/recovery.py, since they need an expert
policy running to be meaningful.
"""

from __future__ import annotations

import numpy as np

from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, SceneConfig

LEVELS = ("L0", "L1", "L2")

# Meters. Half-width of the uniform jitter box applied to each object's
# nominal (x, y) at L1+.
XY_JITTER = 0.03


def _jitter_xy(rng: np.random.Generator, xy: tuple[float, float]) -> tuple[float, float]:
    dx, dy = rng.uniform(-XY_JITTER, XY_JITTER, size=2)
    return (xy[0] + dx, xy[1] + dy)


def _jitter_rgba(
    rng: np.random.Generator, rgba: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    r, g, b, a = rgba
    delta = rng.uniform(-0.15, 0.15, size=3)
    r, g, b = (float(np.clip(c + d, 0.0, 1.0)) for c, d in zip((r, g, b), delta))
    return (r, g, b, a)


def sample_scene_config(level: str, seed: int) -> SceneConfig:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")
    rng = np.random.default_rng(seed)
    base = SceneConfig()

    if level == "L0":
        return base

    plate_xy = _jitter_xy(rng, base.plate_xy)
    mug_xy = _jitter_xy(rng, base.mug_xy)
    bottle_xy = _jitter_xy(rng, base.bottle_xy)
    fork_xy = _jitter_xy(rng, base.fork_xy)
    spoon_xy = _jitter_xy(rng, base.spoon_xy)
    # Drawer starts anywhere from closed to half open at L1+, matching the
    # "state ambiguity" concern in docs/plan_review.md A7: the policy must
    # not assume the episode always starts with the drawer closed.
    drawer_opening = float(rng.uniform(0.0, DRAWER_OPEN_DIST * 0.5))

    if level == "L1":
        return SceneConfig(
            plate_xy=plate_xy,
            mug_xy=mug_xy,
            bottle_xy=bottle_xy,
            fork_xy=fork_xy,
            spoon_xy=spoon_xy,
            drawer_opening=drawer_opening,
        )

    # L2: L1 geometry + visual jitter (table color, object colors).
    table_rgba = _jitter_rgba(rng, base.table_rgba)
    overrides = {
        "plate": _jitter_rgba(rng, (0.9, 0.9, 0.9, 1.0)),
        "mug": _jitter_rgba(rng, (0.8, 0.2, 0.2, 1.0)),
        "bottle": _jitter_rgba(rng, (0.1, 0.4, 0.7, 0.6)),
    }
    return SceneConfig(
        plate_xy=plate_xy,
        mug_xy=mug_xy,
        bottle_xy=bottle_xy,
        fork_xy=fork_xy,
        spoon_xy=spoon_xy,
        drawer_opening=drawer_opening,
        table_rgba=table_rgba,
        object_rgba_overrides=overrides,
    )
