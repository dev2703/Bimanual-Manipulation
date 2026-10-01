"""Shared geometry for a bounded ballistic pour-stream proxy.

This checks a 4 mm stream leaving the lower lip at 0–0.2 m/s. It does not
simulate fluids, splashing, or bottle fill level.
"""
from __future__ import annotations

import numpy as np

MOUTH_OFFSET = .081
MUG_RIM_OFFSET = .030
OPENING_RADIUS = .028
OUTLET_RADIUS = .013
STREAM_RADIUS = .004
MAX_EXIT_SPEED = .20
MIN_GAP = .015
MAX_GAP = .20
APPROACH_GAP = .08
GRAVITY = 9.81
DWELL_STEPS = 20


def lower_lip(mouth, up):
    down = np.array([0., 0., -1.])
    tangent = down - np.dot(down, up) * up
    length = np.linalg.norm(tangent)
    return np.asarray(mouth) + (OUTLET_RADIUS * tangent / length if length > 1e-8 else 0.)


def landing_points(source, bottle_up, rim, mug_rotation):
    """Intersect ballistic paths with the actual glass-rim plane."""
    normal = mug_rotation[:, 2]
    height = float(np.dot(np.asarray(source) - rim, normal))
    if height <= 0 or normal[2] <= 0:
        return None
    velocities = np.linspace(0., MAX_EXIT_SPEED, 17)[:, None] * bottle_up
    vn = velocities @ normal
    times = (vn + np.sqrt(vn**2 + 2*GRAVITY*normal[2]*height)) / (GRAVITY*normal[2])
    points = source + velocities * times[:, None]
    points[:, 2] -= .5 * GRAVITY * times**2
    return (points - rim) @ mug_rotation


def stream_metrics(mouth, bottle_up, rim, mug_rotation):
    source = lower_lip(mouth, bottle_up)
    points = landing_points(source, bottle_up, rim, mug_rotation)
    margin = None if points is None else float(
        OPENING_RADIUS - STREAM_RADIUS - np.max(np.linalg.norm(points[:, :2], axis=1))
    )
    return {
        "mouth_xy_error": float(np.linalg.norm((mouth-rim)[:2])),
        "rim_gap": float(mouth[2]-rim[2]),
        "mug_upright_cosine": float(mug_rotation[2, 2]),
        "bottle_upright_cosine": float(bottle_up[2]),
        "stream_margin": margin,
        "stream_radius": STREAM_RADIUS,
        "max_exit_speed": MAX_EXIT_SPEED,
    }


def bottle_outlet(env):
    body = env.model.body("bottle").id
    up = env.data.xmat[body].reshape(3, 3)[:, 2].copy()
    return env.data.xpos[body].copy() + up*MOUTH_OFFSET, up


def pour_alignment(env):
    mouth, up = bottle_outlet(env)
    body = env.model.body("mug").id
    rotation = env.data.xmat[body].reshape(3, 3)
    rim = env.data.xpos[body] + rotation[:, 2]*MUG_RIM_OFFSET
    return stream_metrics(mouth, up, rim, rotation)


def outlet_aligned(metrics):
    margin = metrics["stream_margin"]
    return bool(
        margin is not None and margin >= 0.
        and MIN_GAP <= metrics["rim_gap"] <= MAX_GAP
        and metrics["mug_upright_cosine"] >= .98
        and metrics["bottle_upright_cosine"] < .88
    )


def pour_sample_valid(env):
    joints = env.state_vector()
    return bool(outlet_aligned(pour_alignment(env))
                and joints[13] > .030 and .005 < joints[6] < .034
                and not glass_blocking_contacts(env)
                and all(f"left/{side}_finger_link" in bottle_contact_bodies(env)
                        for side in ("left", "right")))


def glass_blocking_contacts(env):
    pairs = set()
    for contact in env.data.contact:
        bodies = [env.model.body(env.model.geom_bodyid[g]).name for g in contact.geom]
        if contact.dist < 0 and "mug" in bodies and any(b.startswith("left/") for b in bodies):
            pairs.add(" / ".join(sorted(bodies)))
    return sorted(pairs)


def bottle_contact_bodies(env):
    bodies = set()
    for contact in env.data.contact:
        pair = [env.model.body(env.model.geom_bodyid[g]).name for g in contact.geom]
        if contact.dist <= 0 and "bottle" in pair:
            bodies.update(b for b in pair if b.startswith("left/"))
    return sorted(bodies)


def pour_succeeded(env, dwell):
    return dwell >= DWELL_STEPS and pour_sample_valid(env)


def outlet_target(rim, up, gap=APPROACH_GAP):
    """Center the bounded stream envelope, compensating for lateral travel."""
    mouth = np.asarray(rim) + [0., 0., gap]
    source = lower_lip(mouth, up)
    points = landing_points(source, up, np.asarray(rim), np.eye(3))
    midpoint = .5 * (points[0, :2] + points[-1, :2])
    mouth[:2] -= midpoint
    return mouth
