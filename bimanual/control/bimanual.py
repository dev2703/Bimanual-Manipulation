"""Synchronized dual-arm trajectory execution.

Combines control/ik.py (per-waypoint joint targets) and
control/trajectories.py (min-jerk timing + gripper profiles) into a single
ArmTrajectory / BimanualTrajectory representation that
BimanualTableEnv.step() can be driven from directly, one control tick at a
time. This is the shared plumbing Phase 3's scripted experts build skills
on top of (open_drawer, pick, place, handoff, pour).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bimanual.control.ik import BimanualIK
from bimanual.control.trajectories import (
    gripper_fraction_to_qpos,
    gripper_profile,
    min_jerk_trajectory,
)
from bimanual.sim import robot_spec as rs


@dataclass
class ArmTrajectory:
    """Per-control-tick joint targets (rs.N_ARM_JOINTS,) and gripper qpos
    targets (scalar) for one arm, plus which ticks it's actually active
    on (pause/resume support: an arm holds its last commanded pose on any
    tick it isn't active for)."""

    joint_targets: np.ndarray  # (n_steps, 5)
    gripper_targets: np.ndarray  # (n_steps,)
    active: np.ndarray  # (n_steps,) bool

    @property
    def n_steps(self) -> int:
        return len(self.active)


@dataclass
class BimanualTrajectory:
    arms: dict[str, ArmTrajectory] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return max((t.n_steps for t in self.arms.values()), default=0)

    def actuator_ctrl(self, model, tick: int, drawer_ctrl: float = 0.0) -> np.ndarray:
        """Builds the full model.nu actuator vector for one control tick,
        holding an arm's last target on ticks where it isn't active."""
        ctrl = np.zeros(model.nu)
        actuator_names = [model.actuator(i).name for i in range(model.nu)]
        for prefix, traj in self.arms.items():
            idx = min(tick, traj.n_steps - 1)
            joints = traj.joint_targets[idx]
            gripper = traj.gripper_targets[idx]
            for suffix, val in zip(rs.JOINT_SUFFIXES, joints):
                name = rs.joint_name(prefix, suffix)  # actuator names match joint names (vendored model)
                ctrl[actuator_names.index(name)] = val
            ctrl[actuator_names.index(rs.gripper_joint_name(prefix))] = gripper
        if "drawer_act" in actuator_names:
            ctrl[actuator_names.index("drawer_act")] = drawer_ctrl
        return ctrl


def plan_arm_motion(
    ik: BimanualIK,
    qpos: np.ndarray,
    prefix: str,
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    end_pos: np.ndarray,
    end_quat: np.ndarray,
    n_steps: int,
    gripper_open_at_start: bool,
    close_gripper_during_motion: bool = False,
    extra_limits: list | None = None,
) -> ArmTrajectory:
    """Solves IK at each min-jerk-interpolated EE waypoint along a single
    straight-line Cartesian path (approach/transport moves; pick/place
    skills in Phase 3 chain several of these end to end).

    Orientation is interpolated with simple linear slerp-free blending
    (nlerp) since it's a soft task anyway (decision A1) -- not worth a
    full SO3 slerp for a low-weight target.
    """
    waypoints = min_jerk_trajectory(start_pos, end_pos, n_steps)
    quats = _nlerp_quats(start_quat, end_quat, n_steps)

    joint_targets = np.zeros((n_steps, rs.N_ARM_JOINTS))
    current_qpos = qpos.copy()
    for i in range(n_steps):
        result = ik.solve(current_qpos, {prefix: (waypoints[i], quats[i])}, extra_limits=extra_limits)
        joint_targets[i] = result.joint_targets[prefix]
        for suffix, val in zip(rs.JOINT_SUFFIXES, joint_targets[i]):
            current_qpos[ik.model.joint(rs.joint_name(prefix, suffix)).qposadr[0]] = val

    gripper_frac = gripper_profile(n_steps, open_at_start=gripper_open_at_start) \
        if close_gripper_during_motion else np.full(n_steps, 0.0 if not gripper_open_at_start else 1.0)
    gripper_targets = gripper_fraction_to_qpos(gripper_frac)

    return ArmTrajectory(
        joint_targets=joint_targets,
        gripper_targets=np.asarray(gripper_targets),
        active=np.ones(n_steps, dtype=bool),
    )


def _nlerp_quats(q0: np.ndarray, q1: np.ndarray, n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - t) * q0[None, :] + t * q1[None, :]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def pause_arm(traj: ArmTrajectory, start_tick: int, end_tick: int) -> ArmTrajectory:
    """Marks [start_tick, end_tick) as inactive -- the arm holds its
    pose at start_tick's target for that window (e.g. carrier arm waiting
    for the receiver during a handoff, decision/section 12 in plan.md)."""
    active = traj.active.copy()
    active[start_tick:end_tick] = False
    joint_targets = traj.joint_targets.copy()
    gripper_targets = traj.gripper_targets.copy()
    joint_targets[start_tick:end_tick] = joint_targets[start_tick]
    gripper_targets[start_tick:end_tick] = gripper_targets[start_tick]
    return ArmTrajectory(joint_targets=joint_targets, gripper_targets=gripper_targets, active=active)
