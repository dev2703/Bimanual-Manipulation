"""Symbolic predicates over privileged simulator state.

Training labels and evaluation oracles ONLY (decision A2/A3): these all
consume BimanualTableEnv.oracle_state(), never proprio(). At inference the
executor (Phase 5) uses perception/verifier.py's RGB-estimated versions of
these same predicates instead.
"""

from __future__ import annotations

import numpy as np

from bimanual.sim.scene_builder import (
    DRAWER_OPEN_DIST,
    OBJECT_SPEC,
    REGIONS,
    TABLE_HEIGHT,
    resting_center_z,
)

DRAWER_OPEN_THRESHOLD = 0.8  # fraction of DRAWER_OPEN_DIST counted as "open"
HOLDING_XY_TOL = 0.04  # meters, EE-to-object planar distance for "holding"
HOLDING_Z_TOL = 0.06
GRIPPER_CLOSED_FRACTION = 0.5  # gripper qpos below this fraction of its range = closed
ARM_ARM_EXCLUDE_PREFIXES = ("left_", "right_")  # for filtering self-contacts out


# Closed-drawer handle world y (scene_builder.CABINET_POS[1] - 0.245,
# with drawer_opening=0). Used as a fixed reference so drawer_open() can
# measure TOTAL displacement (static SceneConfig.drawer_opening offset +
# slide-joint travel) via the handle's absolute position, rather than
# just the joint's own qpos -- an earlier version used qpos alone and
# misjudged L1+ episodes that already started partway open, since the
# joint's qpos doesn't include that static baseline (docs/decisions.md
# Phase 3 notes).
CLOSED_DRAWER_HANDLE_Y = 0.39 - 0.245


def drawer_open(oracle_state: dict, threshold: float = DRAWER_OPEN_THRESHOLD) -> bool:
    if "drawer_handle_pos" in oracle_state:
        total_opening = CLOSED_DRAWER_HANDLE_Y - oracle_state["drawer_handle_pos"][1]
    else:
        total_opening = -oracle_state["drawer_opening"]  # qpos-only fallback
    return total_opening >= threshold * DRAWER_OPEN_DIST


def in_region(oracle_state: dict, obj: str, region: str, z_tol: float = 0.03) -> bool:
    """An object counts as placed only if it is BOTH inside the region's
    xy radius AND actually resting at its own resting height.

    The height check is per-object (scene_builder.resting_center_z), not a
    blanket tolerance around TABLE_HEIGHT: the original +-0.15m window was
    loose enough that an object sitting on TOP OF THE CABINET (z=0.515)
    still read as "on table" (docs/decisions.md Phase 4 notes). Objects
    whose geometry isn't in OBJECT_SPEC (e.g. synthetic names in unit
    tests) fall back to the old loose check.
    """
    rx, ry, radius = REGIONS[region]
    pos = oracle_state[f"{obj}_pos"]
    xy_dist = float(np.hypot(pos[0] - rx, pos[1] - ry))
    if obj in OBJECT_SPEC:
        resting = resting_center_z(obj)
        # Cutlery is placed onto the TABLE, but its spec's resting height
        # is defined against the drawer tray it spawns on, so compare
        # against the table-relative resting height here.
        table_resting = TABLE_HEIGHT + OBJECT_SPEC[obj]["half_height"]
        at_rest = min(abs(pos[2] - resting), abs(pos[2] - table_resting)) < z_tol
    else:
        at_rest = abs(pos[2] - TABLE_HEIGHT) < z_tol + 0.10
    return xy_dist <= radius and at_rest


def holding(
    proprio: dict,
    oracle_state: dict,
    arm: str,
    obj: str,
    gripper_range: tuple[float, float],
) -> bool:
    """Requires BOTH proprioception (gripper closed) and oracle state
    (object near the EE) -- this predicate itself is oracle-only (uses
    object position), but note perception/verifier.py (Phase 5) has to
    learn to estimate it from RGB alone since object position won't be
    available at inference."""
    ee_pos = proprio[f"{arm}_ee_pos"]
    obj_pos = oracle_state[f"{obj}_pos"]
    xy_close = float(np.hypot(ee_pos[0] - obj_pos[0], ee_pos[1] - obj_pos[1])) < HOLDING_XY_TOL
    z_close = abs(ee_pos[2] - obj_pos[2]) < HOLDING_Z_TOL
    lo, hi = gripper_range
    closed_thresh = lo + GRIPPER_CLOSED_FRACTION * (hi - lo)
    gripper_closed = proprio[f"{arm}_gripper_opening"] < closed_thresh
    return xy_close and z_close and gripper_closed


def pour_success(
    oracle_state: dict,
    bottle_tilt_rad: float,
    tilt_threshold: float = 1.0,
    mug_xy_tol: float = 0.05,
) -> bool:
    """Proxy metric (decision A4): relative pose + tilt, no fluid
    particles. bottle_tilt_rad is the angle between the bottle's own +z
    axis and world +z, computed by the caller from oracle_state's
    bottle_mat (kept out of this function so it stays a pure predicate
    over already-extracted scalars, easing unit testing)."""
    mug_pos = oracle_state["mug_pos"]
    bottle_pos = oracle_state["bottle_pos"]
    xy_dist = float(np.hypot(mug_pos[0] - bottle_pos[0], mug_pos[1] - bottle_pos[1]))
    return xy_dist <= mug_xy_tol and bottle_tilt_rad >= tilt_threshold


def bottle_tilt_from_mat(bottle_mat: np.ndarray) -> float:
    """Angle (radians) between the bottle's local +z axis and world +z."""
    local_z_world = bottle_mat[:, 2]
    cos_angle = float(np.clip(local_z_world[2], -1.0, 1.0))
    return float(np.arccos(cos_angle))


def arm_arm_collision(oracle_state: dict) -> bool:
    """True if any contact pair involves bodies from both a left_* arm and
    a right_* arm (their own kinematic-chain self-contacts are already
    excluded by MuJoCo's default parent/child adjacency rule, and we don't
    want to misreport those as inter-arm collisions). Contacts are keyed
    by body name, not geom name -- see env.py's oracle_state()."""
    for b1, b2 in oracle_state["contacts"]:
        b1_is_left = bool(b1) and b1.startswith("left_")
        b2_is_left = bool(b2) and b2.startswith("left_")
        b1_is_right = bool(b1) and b1.startswith("right_")
        b2_is_right = bool(b2) and b2.startswith("right_")
        if (b1_is_left and b2_is_right) or (b1_is_right and b2_is_left):
            return True
    return False
