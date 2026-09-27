"""Kinematic spec for the SO-101 arm used in the bimanual scene.

The arm itself is the real vendored model from
assets/robots/so101/so101.xml (see NOTICE.md there), loaded and
instantiated twice by bimanual/sim/so101_loader.py. This module is just
the shared naming/limits contract the rest of the codebase (env.py,
control/ik.py in Phase 2, the dataset schema in Phase 4) depends on, kept
in one place so it can't drift from the vendored file.

Decision A1 (docs/decisions.md): action/joint space is 5 revolute joints
plus a gripper per arm, matching the physical SO-101, not a 6-DOF EE pose.
"""

from __future__ import annotations

from dataclasses import dataclass

# Real joint names from the vendored SO-101 MJCF (onshape-to-robot export).
# Order matters: it is the fixed order used for observation.state / action
# vectors everywhere else in the codebase.
JOINT_SUFFIXES = (
    "shoulder_pan",   # waist yaw
    "shoulder_lift",  # shoulder pitch
    "elbow_flex",     # elbow pitch
    "wrist_flex",     # wrist pitch
    "wrist_roll",     # wrist roll
)
GRIPPER_SUFFIX = "gripper"

# Joint limits (radians), copied from assets/robots/so101/so101.xml so
# callers don't need to parse the MJCF just to get a range.
@dataclass(frozen=True)
class JointLimit:
    lo: float
    hi: float


JOINT_LIMITS: dict[str, JointLimit] = {
    "shoulder_pan": JointLimit(-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift": JointLimit(-1.7453292519943224, 1.7453292519943366),
    "elbow_flex": JointLimit(-1.69, 1.69),
    "wrist_flex": JointLimit(-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": JointLimit(-2.7438472969992493, 2.841206309382605),
}
GRIPPER_LIMIT = JointLimit(-0.17453297762778586, 1.7453291995659765)
# Verified by rendering both extremes (docs/decisions.md Phase 3 notes):
# the vendored joint's convention is the OPPOSITE of what its numeric
# range order suggests -- .lo is fully OPEN, .hi is fully CLOSED.
GRIPPER_OPEN = GRIPPER_LIMIT.lo
GRIPPER_CLOSED = GRIPPER_LIMIT.hi

N_ARM_JOINTS = len(JOINT_SUFFIXES)
N_ARM_ACTIONS = N_ARM_JOINTS + 1  # + gripper
N_BIMANUAL_ACTIONS = 2 * N_ARM_ACTIONS  # 12-D joint-target baseline (A1)


def joint_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def arm_joint_names(prefix: str) -> list[str]:
    return [joint_name(prefix, s) for s in JOINT_SUFFIXES]


def gripper_joint_name(prefix: str) -> str:
    return joint_name(prefix, GRIPPER_SUFFIX)


def ee_site_name(prefix: str) -> str:
    # The vendored model's own end-effector reference frame.
    return f"{prefix}_gripperframe"
