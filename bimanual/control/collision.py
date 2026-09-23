"""Arm-arm and arm-table collision avoidance for the IK solve, plus a
post-hoc contact query for evaluation.

mink.CollisionAvoidanceLimit needs explicit geom groups. Most of the
vendored SO-101's mesh geoms have no `name` attribute (only their parent
bodies are named -- see env.py's oracle_state()), so groups are built via
mink.get_subtree_geom_ids() from each arm's base body, rather than by
listing geom names.
"""

from __future__ import annotations

import mink
import mujoco

ARM_BASE_BODY = {"left": "left_base", "right": "right_base"}
MIN_DISTANCE = 0.01  # meters, kept clear of self/table during IK
OBJECT_BODIES = ("plate", "mug", "bottle", "fork", "spoon")


def build_arm_collision_avoidance_limits(
    model: mujoco.MjModel,
    min_distance: float = MIN_DISTANCE,
) -> list[mink.CollisionAvoidanceLimit]:
    """One CollisionAvoidanceLimit for left-vs-right arm geoms, and one per
    arm for arm-vs-table, so the IK velocity solve steers away from both
    before a contact is ever detected by mj_step."""
    left_geoms = mink.get_subtree_geom_ids(model, model.body(ARM_BASE_BODY["left"]).id)
    right_geoms = mink.get_subtree_geom_ids(model, model.body(ARM_BASE_BODY["right"]).id)
    table_geoms = mink.get_subtree_geom_ids(model, model.body("table").id)

    limits = [
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(left_geoms, right_geoms)],
            minimum_distance_from_collisions=min_distance,
        ),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(left_geoms, table_geoms)],
            minimum_distance_from_collisions=min_distance,
        ),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(right_geoms, table_geoms)],
            minimum_distance_from_collisions=min_distance,
        ),
    ]
    return limits


def build_object_avoidance_limits(
    model: mujoco.MjModel,
    prefix: str,
    exclude_objects: tuple[str, ...] = (),
    min_distance: float = MIN_DISTANCE,
    detection_distance: float = 0.10,
) -> list[mink.CollisionAvoidanceLimit]:
    """Keeps one arm's geoms clear of movable tabletop objects during a
    transit, other than any in exclude_objects (e.g. the object that
    segment is actually trying to approach/grasp). Without this, the IK
    solve only constrains the end-effector site -- nothing stops the
    elbow or forearm from swinging through an object sitting between the
    arm's current pose and the target (Phase 3 finding, docs/decisions.md:
    this is exactly how the mug got knocked across the table by a
    "safe" straight-up-then-over EE path)."""
    arm_geoms = mink.get_subtree_geom_ids(model, model.body(ARM_BASE_BODY[prefix]).id)
    limits = []
    for obj in OBJECT_BODIES:
        if obj in exclude_objects:
            continue
        obj_geoms = mink.get_subtree_geom_ids(model, model.body(obj).id)
        limits.append(
            mink.CollisionAvoidanceLimit(
                model=model,
                geom_pairs=[(arm_geoms, obj_geoms)],
                minimum_distance_from_collisions=min_distance,
                collision_detection_distance=detection_distance,
            )
        )
    return limits


def inter_group_contacts(
    contacts: list[tuple[str, str]], prefix_a: str, prefix_b: str
) -> list[tuple[str, str]]:
    """Generalization of oracle_predicates.arm_arm_collision: returns the
    contact pairs (by body name, see env.py's oracle_state()) that cross
    between two body-name-prefix groups, e.g. inter_group_contacts(c,
    "left_", "right_") for arm-arm, or inter_group_contacts(c, "left_",
    "cabinet") for an arm hitting the cabinet."""
    hits = []
    for b1, b2 in contacts:
        a_in_1 = bool(b1) and b1.startswith(prefix_a)
        a_in_2 = bool(b2) and b2.startswith(prefix_a)
        b_in_1 = bool(b1) and b1.startswith(prefix_b)
        b_in_2 = bool(b2) and b2.startswith(prefix_b)
        if (a_in_1 and b_in_2) or (a_in_2 and b_in_1):
            hits.append((b1, b2))
    return hits
