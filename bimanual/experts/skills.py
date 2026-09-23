"""Scripted single-arm skills built on the Phase 2 control stack.

Each skill returns a PhasedTrajectory: an ordered list of (phase, ArmTrajectory)
segments for ONE arm. experts/table_setting.py concatenates these across
arms/skills into a full episode and uses the phase labels as the dataset's
per-frame phase field (plan.md's Long-VLA-inspired phase vocabulary).

Lift-clear waypoints, done properly (Phase 2 and Phase 3 findings, see
docs/decisions.md): a single straight-line 3D move from A to B is NOT
object-aware and, worse, conflates vertical and horizontal motion -- early
in a min-jerk interpolation the end-effector can sweep sideways at low
height before it has risen clear, sweeping straight through whatever is
sitting between the start and end XY. Phase 2 lost a bottle to exactly
this. Phase 3 first-draft skills also knocked the mug across the table
during "APPROACH" for the same reason.

The fix used throughout this module: every transit between two points at
different (x, y) and comparable-to-table z always goes through an
explicit SAFE_Z waypoint via THREE separate straight segments (rise in
place, move horizontally at safe height, descend in place), never a
single diagonal interpolation.

Grasp orientation: the gripper approaches top-down. Orientation is a
soft task (decision A1), so a downward-pointing target quat only weakly
biases the solver rather than forcing an exact approach angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.bimanual import ArmTrajectory, plan_arm_motion
from bimanual.control.collision import build_object_avoidance_limits
from bimanual.control.ik import BimanualIK
from bimanual.control.trajectories import gripper_fraction_to_qpos
from bimanual.sim import robot_spec as rs
from bimanual.sim.scene_builder import TABLE_HEIGHT, table_center_z

SAFE_Z = TABLE_HEIGHT + 0.18  # transport height, clears all tabletop objects
RISE_STEPS = 12
TRANSIT_STEPS = 20
# These are PATH RESOLUTION only. The position-controlled arm lags its
# commanded trajectory, so a descend can end before the EE has really
# arrived (measured: 68-100mm residual at the end of APPROACH on
# failing seeds, larger than env.py's AUTO_GRASP_Z_TOL, so the grasp
# never fires). That lag is absorbed by executor.py's adaptive settle
# rather than by padding these counts: simply raising them to 20/18
# took the plate 79%->93% but collapsed the mug 100%->6%, because a
# fixed budget that suits one object over-holds another.
DESCEND_STEPS = 12
GRASP_STEPS = 8
RELEASE_STEPS = 8

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # soft target, decision A1

# How far above an object's BODY CENTER the gripper closes. Must be
# consistent between pick() and place(): pick() grasps at center +
# this, so the kinematic pin carries the object exactly this far below
# the EE, and place() targets resting_center + this to put it back down
# at its resting height. Values track each object's half-height (grab
# near the top of a tall object, level with a flat one) -- the plate's
# was previously the 0.03 default, i.e. the jaws closed on air 2.4cm
# above a 1.2cm-thick disc.
GRASP_HEIGHT_ABOVE_BASE = {
    "plate": 0.01,
    "mug": 0.03,
    "bottle": 0.05,
    "fork": 0.0,
    "spoon": 0.0,
}

# Released this far above the object's resting height, so it settles the
# last few mm under gravity instead of being driven into the table.
RELEASE_CLEARANCE = 0.005


def place_target(object_kind: str, region_xy: tuple[float, float]) -> np.ndarray:
    """The (x, y, z) a caller should hand to place() for `object_kind`.

    z is the object's body-center height when RESTING ON THE TABLE (all
    placements target the table, including cutlery, which merely spawns
    in the drawer).
    place() offsets by the grasp height on top of this, which -- because
    pick() grasps at object_center + grasp_height -- lands the object's
    center back at its resting height. Passing TABLE_HEIGHT directly (as
    every caller originally did) commands the object a half-height INTO
    the table: the descend then stalls fighting the table contact and the
    object gets released ~15cm up and rolls away (measured, docs/
    decisions.md Phase 4 notes).
    """
    return np.array([region_xy[0], region_xy[1], table_center_z(object_kind) + RELEASE_CLEARANCE])


@dataclass
class PhaseSegment:
    phase: str
    trajectory: ArmTrajectory


PhasedTrajectory = list[PhaseSegment]


def _hold(gripper_frac: float, n_steps: int, joint_targets: np.ndarray) -> ArmTrajectory:
    return ArmTrajectory(
        joint_targets=np.tile(joint_targets, (n_steps, 1)),
        gripper_targets=np.full(n_steps, gripper_fraction_to_qpos(gripper_frac)),
        active=np.ones(n_steps, dtype=bool),
    )


def _straight_move(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_steps: int,
    gripper_frac: float,
    avoid_objects_except: tuple[str, ...] = (),
) -> tuple[ArmTrajectory, np.ndarray]:
    """One straight-line segment at a fixed gripper state. Keeps this
    arm's whole body clear of every tabletop object except those in
    avoid_objects_except (Phase 3 finding: point-only IK doesn't stop the
    forearm/elbow from sweeping through an object even on a "safe"
    end-effector path -- see module docstring). Returns the trajectory
    and the qpos to warm-start the NEXT segment from."""
    extra_limits = build_object_avoidance_limits(ik.model, prefix, exclude_objects=avoid_objects_except)
    traj = plan_arm_motion(
        ik, qpos, prefix, start_pos, DOWN_QUAT, end_pos, DOWN_QUAT, n_steps,
        gripper_open_at_start=(gripper_frac > 0.5),
        close_gripper_during_motion=False,
        extra_limits=extra_limits,
    )
    traj.gripper_targets[:] = gripper_fraction_to_qpos(gripper_frac)
    next_qpos = qpos.copy()
    for suffix, val in zip(rs.JOINT_SUFFIXES, traj.joint_targets[-1]):
        next_qpos[ik.model.joint(rs.joint_name(prefix, suffix)).qposadr[0]] = val
    return traj, next_qpos


def _rise_transit_descend(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    start_pos: np.ndarray,
    end_xy_z: np.ndarray,
    gripper_frac: float,
    target_object: str,
    safe_z: float = SAFE_Z,
) -> tuple[PhasedTrajectory, np.ndarray]:
    """Rise straight up in place to safe_z, move horizontally at safe_z,
    descend straight down in place to end_xy_z. Never interpolates XY and
    Z at the same time (see module docstring). RISE/TRANSPORT avoid every
    tabletop object including target_object; the final descend leg
    excludes target_object from avoidance so the gripper can actually
    reach it, while still avoiding the other objects."""
    segments: PhasedTrajectory = []

    rise_target = start_pos.copy()
    rise_target[2] = safe_z
    rise_traj, q = _straight_move(
        ik, qpos, prefix, start_pos, rise_target, RISE_STEPS, gripper_frac,
        avoid_objects_except=(target_object,) if gripper_frac < 0.5 else (),
    )
    segments.append(PhaseSegment("RISE", rise_traj))

    transit_target = end_xy_z.copy()
    transit_target[2] = safe_z
    transit_traj, q = _straight_move(
        ik, q, prefix, rise_target, transit_target, TRANSIT_STEPS, gripper_frac,
        avoid_objects_except=(target_object,) if gripper_frac < 0.5 else (),
    )
    segments.append(PhaseSegment("TRANSPORT", transit_traj))

    descend_traj, q = _straight_move(
        ik, q, prefix, transit_target, end_xy_z, DESCEND_STEPS, gripper_frac,
        avoid_objects_except=(target_object,),
    )
    segments.append(PhaseSegment("APPROACH" if gripper_frac > 0.5 else "PLACE", descend_traj))

    return segments, q


def pick(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    ee_pos: np.ndarray,
    object_base_pos: np.ndarray,
    object_kind: str,
) -> PhasedTrajectory:
    """Rise/transit/descend to directly above the object (gripper open),
    then descend onto it, close, and lift straight up. Leaves the arm at
    SAFE_Z above the object's original XY, holding it."""
    grasp_h = GRASP_HEIGHT_ABOVE_BASE.get(object_kind, 0.03)
    grasp_pos = object_base_pos.copy()
    grasp_pos[2] = object_base_pos[2] + grasp_h
    above_xy = grasp_pos.copy()
    above_xy[2] = SAFE_Z

    segments, q = _rise_transit_descend(ik, qpos, prefix, ee_pos, grasp_pos, gripper_frac=1.0, target_object=object_kind)

    grasp = _hold(0.0, GRASP_STEPS, segments[-1].trajectory.joint_targets[-1])
    segments.append(PhaseSegment("GRASP", grasp))

    lift_traj, _ = _straight_move(
        ik, q, prefix, grasp_pos, above_xy, RISE_STEPS, gripper_frac=0.0,
        avoid_objects_except=(object_kind,),
    )
    segments.append(PhaseSegment("LIFT", lift_traj))

    return segments


def place(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    ee_pos: np.ndarray,
    target_xy_z_table: np.ndarray,
    object_kind: str,
) -> PhasedTrajectory:
    """Assumes the arm is at SAFE_Z holding an object (as pick() leaves
    it). Transits at safe height to above the target, descends, opens,
    retracts straight up."""
    grasp_h = GRASP_HEIGHT_ABOVE_BASE.get(object_kind, 0.03)
    place_pos = target_xy_z_table.copy()
    place_pos[2] = target_xy_z_table[2] + grasp_h

    segments, q = _rise_transit_descend(ik, qpos, prefix, ee_pos, place_pos, gripper_frac=0.0, target_object=object_kind)

    release = _hold(1.0, RELEASE_STEPS, segments[-1].trajectory.joint_targets[-1])
    segments.append(PhaseSegment("RELEASE", release))

    above_target = place_pos.copy()
    above_target[2] = SAFE_Z
    retract_traj, _ = _straight_move(
        ik, q, prefix, place_pos, above_target, RISE_STEPS, gripper_frac=1.0,
        avoid_objects_except=(object_kind,),
    )
    segments.append(PhaseSegment("RETRACT", retract_traj))

    return segments


def open_drawer(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    ee_pos: np.ndarray,
    drawer_open_dist: float,
    handle_pos: np.ndarray,
) -> PhasedTrajectory:
    """Approach the drawer handle, close the gripper around it, pull
    straight back (-y, toward the arm) by drawer_open_dist, release.
    handle_pos must be the handle's CURRENT world position (e.g. from
    env.data.geom_xpos[model.geom('drawer_handle').id]) -- not a fixed
    formula, since L1+ randomization can start the drawer partway open
    (an earlier version assumed drawer_opening=0 and consistently missed
    the handle at L1, docs/decisions.md Phase 3 notes)."""
    handle_pos = handle_pos.copy()
    pulled_pos = handle_pos.copy()
    pulled_pos[1] -= drawer_open_dist

    segments, q = _rise_transit_descend(
        ik, qpos, prefix, ee_pos, handle_pos, gripper_frac=1.0, target_object="cabinet_handle_na",
    )

    grasp = _hold(0.0, GRASP_STEPS, segments[-1].trajectory.joint_targets[-1])
    segments.append(PhaseSegment("GRASP", grasp))

    pull_traj, q = _straight_move(ik, q, prefix, handle_pos, pulled_pos, TRANSIT_STEPS, gripper_frac=0.0)
    segments.append(PhaseSegment("PULL", pull_traj))

    release = _hold(1.0, RELEASE_STEPS, pull_traj.joint_targets[-1])
    segments.append(PhaseSegment("RELEASE", release))

    retract_target = pulled_pos.copy()
    retract_target[2] = SAFE_Z
    retract_traj, _ = _straight_move(ik, q, prefix, pulled_pos, retract_target, RISE_STEPS, gripper_frac=1.0)
    segments.append(PhaseSegment("RETRACT", retract_traj))

    return segments
