"""Executes a PhasedTrajectory (from experts/skills.py) against a live
BimanualTableEnv, one control tick at a time.

This is where the GRASP/RELEASE phase labels turn into actual
env.grasp()/env.release() calls (the scripted/kinematic attach
mechanism -- see env.py's _held docstring note) -- skills.py itself only
plans joint trajectories and has no env access.
"""

from __future__ import annotations

import numpy as np

from bimanual.experts.skills import PhasedTrajectory
from bimanual.sim import robot_spec as rs
from bimanual.sim.env import BimanualTableEnv

# Phases after which the arm is allowed extra ticks to actually arrive at
# the pose that was commanded, before anything depends on it having
# arrived (the grasp firing, or the object being let go at the right
# height).
SETTLE_PHASES = ("APPROACH", "PLACE")
SETTLE_MAX_TICKS = 25
SETTLE_TOL = 5e-4  # metres of EE movement per tick that counts as "stopped"
DATASET_HZ = 10
DATASET_PERIOD = 1.0 / DATASET_HZ


def _snapshot(env: BimanualTableEnv, prefix: str, phase: str, action: np.ndarray) -> dict:
    """Capture a synchronized observation before ``action`` is applied.

    Keeping capture here, in the live control loop, prevents historical
    proprioception from being paired with images rendered from the final
    state of the episode.
    """
    return {
        "phase": phase,
        "arm": prefix,
        "proprio": env.proprio(),
        "oracle_state": env.oracle_state(),
        "action": action.copy(),
        "frames": env.render(),
        "timestamp": float(env.data.time),
    }


def _record_due(record: list | None, sim_time: float) -> bool:
    return record is not None and (not record or sim_time + 1e-9 >= record[-1]["timestamp"] + DATASET_PERIOD)


def commanded_action_vector(env: BimanualTableEnv, ctrl: np.ndarray) -> np.ndarray:
    """The 12-D COMMANDED joint/gripper target for this tick, in the same
    layout as observation.state (left 5 joints + gripper, then right).

    This is what an imitation policy must learn to output. It is NOT the
    same as the measured state: the position-controlled arm lags its
    command by design (the lag is exactly what executor._settle() waits
    out). Recording `action = state` instead -- which the dataset writer
    originally did -- trains the policy on the identity function: it fits
    with a very low loss and then commands "stay exactly where you are"
    at inference, so the arm never moves. Measured: ACT reached L1 0.065
    in training and 0/20 closed-loop before this was separated out.
    """
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    out = []
    for prefix in ("left", "right"):
        for suffix in rs.JOINT_SUFFIXES:
            out.append(ctrl[names.index(rs.joint_name(prefix, suffix))])
        out.append(ctrl[names.index(rs.gripper_joint_name(prefix))])
    return np.asarray(out, dtype=np.float32)


def _settle(
    env: BimanualTableEnv,
    prefix: str,
    ctrl: np.ndarray,
    phase: str,
    record: list | None,
) -> None:
    """Holds the last commanded ctrl until the end-effector stops moving.

    The arm is position-controlled and lags its commanded trajectory, so
    a fixed-length descend routinely ends tens of mm short (measured:
    68-100mm residual at the end of APPROACH on failing seeds, which is
    outside env.py's AUTO_GRASP_Z_TOL, so the grasp silently never
    fires). Padding skills.py's step counts instead was tried and is a
    worse fix: 20/18 steps took the plate 79%->93% but collapsed the mug
    100%->6%, because one fixed budget cannot suit objects of different
    sizes. Waiting for actual convergence adapts per object and per seed,
    and matches the closed-loop re-observation that docs/decisions.md's
    Phase 2 notes already called for.
    """
    last = env.proprio()[f"{prefix}_ee_pos"].copy()
    for _ in range(SETTLE_MAX_TICKS):
        if _record_due(record, float(env.data.time)):
            record.append(_snapshot(env, prefix, phase, commanded_action_vector(env, ctrl)))
        env.step(ctrl)
        now = env.proprio()[f"{prefix}_ee_pos"].copy()
        if float(np.linalg.norm(now - last)) < SETTLE_TOL:
            return
        last = now


def execute(
    env: BimanualTableEnv,
    prefix: str,
    segments: PhasedTrajectory,
    obj_name: str | None = None,
    record: list | None = None,
) -> None:
    """Runs every segment's trajectory through env.step(). If obj_name is
    given, calls env.grasp(prefix, obj_name) at the end of a "GRASP"
    phase and env.release(prefix) at the end of a "RELEASE" phase. If
    `record` is a list, appends one dict per control tick (proprio +
    phase label + oracle state) -- the Phase 4 dataset generator's hook."""
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    other_prefix = "right" if prefix == "left" else "left"

    for seg in segments:
        for jt, gt in zip(seg.trajectory.joint_targets, seg.trajectory.gripper_targets):
            ctrl = env.data.ctrl.copy()
            for suffix, val in zip(rs.JOINT_SUFFIXES, jt):
                ctrl[names.index(rs.joint_name(prefix, suffix))] = val
            ctrl[names.index(rs.gripper_joint_name(prefix))] = gt
            if _record_due(record, float(env.data.time)):
                record.append(_snapshot(env, prefix, seg.phase, commanded_action_vector(env, ctrl)))
            env.step(ctrl)

        if seg.phase in SETTLE_PHASES:
            _settle(env, prefix, ctrl, seg.phase, record)

        if obj_name is not None and seg.phase == "GRASP" and not env.is_physical:
            env.grasp(prefix, obj_name)
        elif obj_name is not None and seg.phase == "RELEASE" and not env.is_physical:
            env.release(prefix)


def execute_open_drawer(
    env: BimanualTableEnv,
    prefix: str,
    segments: PhasedTrajectory,
    drawer_open_dist: float,
) -> None:
    """Like execute(), but during the "PULL" phase also ramps the
    drawer_act actuator from closed to drawer_open_dist in sync with the
    arm's retreat. The drawer has a 1-DOF slide joint, not a freejoint,
    so it can't use the generic grasp()/release() kinematic-attach path
    (that assumes a 7-DOF pos+quat object) -- it must be driven by its
    own actuator directly."""
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    drawer_idx = names.index("drawer_act")

    for seg in segments:
        n = seg.trajectory.n_steps
        for i, (jt, gt) in enumerate(zip(seg.trajectory.joint_targets, seg.trajectory.gripper_targets)):
            ctrl = env.data.ctrl.copy()
            for suffix, val in zip(rs.JOINT_SUFFIXES, jt):
                ctrl[names.index(rs.joint_name(prefix, suffix))] = val
            ctrl[names.index(rs.gripper_joint_name(prefix))] = gt
            if seg.phase == "PULL":
                frac = (i + 1) / n
                ctrl[drawer_idx] = -drawer_open_dist * frac
            env.step(ctrl)
