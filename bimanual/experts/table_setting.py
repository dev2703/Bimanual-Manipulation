"""Task dependency graph and deterministic arm assignment for the full
table-setting episode (plan.md section 9), and a function to run it
end to end against a live BimanualTableEnv.

Deterministic arm assignment: an object is assigned to whichever arm's
base is on the same side of the table (by x sign), matching how
reachability_study.py laid out the scene (left objects on the -x half,
right on the +x half) -- this is the "implicit arm assignment from
reachability" the executor design (decision A3) calls for, done here
with the oracle x-position rather than the RGB verifier (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bimanual.control.ik import BimanualIK
from bimanual.control.trajectories import min_jerk_trajectory
from bimanual.experts.executor import execute, execute_open_drawer
from bimanual.experts.skills import open_drawer, pick, place, place_target
from bimanual.logging_utils import get_logger
from bimanual.sim import robot_spec as rs
from bimanual.sim.env import HOME_POSE, BimanualTableEnv
from bimanual.sim.oracle_predicates import drawer_open, in_region
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, REGIONS

log = get_logger(__name__)

# (object, region, requires_drawer_open)
#
# fork/spoon were removed per decision A5's pre-agreed descope rule
# ("if cutlery grasp is below 80% at the end of Phase 3, ship the full
# task with cutlery pre-placed on the table and record the decision").
# Extracting a flat capsule from inside the drawer measured 0-33% even
# after the physics fixes, and the reachable table area cannot fit a
# 5th spawn+region pair without overlapping the others. They still
# spawn on the table as scene dressing/collision realism.
TASK_GRAPH = [
    ("mug", "mug_region", False),
    ("bottle", "bottle_region", False),
    ("plate", "plate_region", False),
]


def assign_arm(object_x: float) -> str:
    return "left" if object_x < 0 else "right"


# assign_arm(x) is ambiguous right at x~=0 (L1 jitter can push plate's
# spawn x to either sign per seed, docs/decisions.md), and plate_region
# was calibrated (grid-searched) for a SPECIFIC arm's convergence -- an
# episode where jitter flips plate to the other arm would try to reach
# plate_region from the wrong side entirely. Pin plate to the arm the
# region was calibrated for instead of trusting its near-zero spawn sign.
FIXED_ARM_OVERRIDE = {"plate": "right"}


def _resolve_arm(obj: str, object_x: float) -> str:
    return FIXED_ARM_OVERRIDE.get(obj) or assign_arm(object_x)


RETURN_HOME_STEPS = 20


def _return_arm_to_home(env: BimanualTableEnv, prefix: str, n_steps: int = RETURN_HOME_STEPS) -> None:
    """Drives `prefix`'s joints back to HOME_POSE via a smooth min-jerk
    joint-space trajectory after each object's pick+place.

    Without this, an object's pick()/place() leaves the arm wherever its
    own task happened to end, and the NEXT object's motion plan warm-
    starts from that arbitrary, episode-order-dependent pose instead of
    the known-good HOME_POSE baseline every region/skill was calibrated
    against. Found empirically this session: plate_region was grid-
    searched (and worked reliably) starting from HOME_POSE, but plate
    still failed almost every real full-episode run because by the time
    its turn came, the same right arm was still wherever mug's own
    attempt (which runs earlier in TASK_GRAPH) had left it.
    """
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    current = np.array([env._get_joint_qpos(rs.joint_name(prefix, s)) for s in rs.JOINT_SUFFIXES])
    target = np.array([HOME_POSE[s] for s in rs.JOINT_SUFFIXES])
    traj = min_jerk_trajectory(current, target, n_steps)
    for jt in traj:
        ctrl = env.data.ctrl.copy()
        for suffix, val in zip(rs.JOINT_SUFFIXES, jt):
            ctrl[names.index(rs.joint_name(prefix, suffix))] = val
        ctrl[names.index(rs.gripper_joint_name(prefix))] = rs.GRIPPER_OPEN
        env.step(ctrl)


@dataclass
class StepResult:
    step: str
    success: bool
    detail: dict = field(default_factory=dict)


def run_full_episode(env: BimanualTableEnv, ik: BimanualIK) -> list[StepResult]:
    """Runs open_drawer, then each pick+place in TASK_GRAPH order,
    against the given (already-reset) env. Returns one StepResult per
    step so callers (expert_success.py) can report per-skill AND
    composed success without re-simulating."""
    results: list[StepResult] = []

    drawer_opened_ok = False
    # Still opened every episode even though cutlery is now pre-placed
    # (A5 descope) and no TASK_GRAPH entry depends on it: "open the
    # drawer" remains part of the table-setting task and is reported as
    # its own skill by scripts/expert_success.py.
    if True:
        prefix = "left"
        ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()
        handle_pos = env.data.geom_xpos[env.model.geom("drawer_handle").id].copy()
        segs = open_drawer(ik, env.data.qpos, prefix, ee_pos, DRAWER_OPEN_DIST, handle_pos)
        execute_open_drawer(env, prefix, segs, DRAWER_OPEN_DIST)
        for _ in range(10):
            env.step(env.data.ctrl)
        drawer_opened_ok = drawer_open(env.oracle_state())
        results.append(StepResult("open_drawer", drawer_opened_ok))
        _return_arm_to_home(env, "left")

    for obj, region, requires_drawer in TASK_GRAPH:
        if requires_drawer and not drawer_opened_ok:
            # drawer never opened -- cutlery is unreachable, record and skip
            results.append(StepResult(f"pick_place_{obj}", False, {"reason": "drawer_not_open"}))
            continue

        obj_pos = env.oracle_state()[f"{obj}_pos"].copy()
        prefix = _resolve_arm(obj, obj_pos[0])
        ee_pos = env.proprio()[f"{prefix}_ee_pos"].copy()

        try:
            pick_segs = pick(ik, env.data.qpos, prefix, ee_pos, obj_pos, obj)
            execute(env, prefix, pick_segs, obj_name=obj)

            rx, ry, _ = REGIONS[region]
            target = place_target(obj, (rx, ry))
            ee2 = env.proprio()[f"{prefix}_ee_pos"].copy()
            place_segs = place(ik, env.data.qpos, prefix, ee2, target, obj)
            execute(env, prefix, place_segs, obj_name=obj)

            for _ in range(10):
                env.step(env.data.ctrl)
            ok = in_region(env.oracle_state(), obj, region)
        except Exception as e:  # noqa: BLE001 -- a planning/IK failure counts as a failed skill, not a crash
            log.exception("pick_place_%s failed during planning/execution", obj)
            if env._held[prefix] is not None:
                env.release(prefix)  # don't carry a half-grasped object back to home
            _return_arm_to_home(env, prefix)
            results.append(StepResult(f"pick_place_{obj}", False, {"error": repr(e)}))
            continue

        if env._held[prefix] is not None:
            env.release(prefix)
        _return_arm_to_home(env, prefix)
        results.append(StepResult(f"pick_place_{obj}", ok))

    return results
