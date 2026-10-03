"""Contact-only two-arm transfer of a graspable dinner-service baton.

Every motion goes through the position actuators; the expert never writes
joint or object state, so recorded actions replay the transfer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from bimanual.control.aloha_ik import AlohaIK, top_down_quaternion
from bimanual.control.trajectories import min_jerk_trajectory
from bimanual.evaluation.aloha_contacts import body_touches
from bimanual.evaluation.aloha_predicates import handoff_succeeded
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaPhysicalEnv

NECK = 0.042
# Position actuators settle about 1.6cm below the commanded site. Aim high so
# the fingertip site lands on the neck.
SITE_SAG = np.array([0.0, 0.0, 0.018])
# The receiver's fingertip site sits this far above the neck when pinching.
PINCH_HEIGHT = 0.012


@dataclass(frozen=True)
class AlohaHandoffResult:
    success: bool
    initial_height: float
    peak_height: float
    after_release_height: float
    final_height: float
    carrier_opening: float
    receiver_opening: float
    record: list[dict]
    receiver_contact_steps: int = 0
    carrier: str = "left"


def grip_quaternion(arm: str) -> np.ndarray:
    """Top-down grasp with the gripper housing pointing outboard.

    The housing reaches 11.7cm to one side of the fingertips, wider than the
    gap between the baton's necks. With one shared orientation one housing
    always overhangs the other gripper, so the left gripper is turned 180
    degrees about its approach axis.
    """
    if arm == "right":
        return top_down_quaternion()
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion


def _hold(env: AlohaPhysicalEnv, steps: int, arm: str, phase: str, record: list[dict] | None,
          grips: dict[int, float] | None = None) -> None:
    for _ in range(steps):
        action = env.data.ctrl.copy()
        for index, value in (grips or {}).items():
            action[index] = value
        step_recorded(env, action, arm, phase, record)


def _nudge(
    env: AlohaPhysicalEnv, ik: AlohaIK, goal: np.ndarray, gripper: float,
    arm: str, phase: str, record: list[dict] | None,
) -> None:
    """Position-first fallback when a strict orientation solve will not converge."""
    result = ik.solve(env.data.qpos, {arm: (goal, grip_quaternion(arm))}, orientation_tolerance=0.6)
    if result.position_error[arm] > 0.015:
        return
    start = 0 if arm == "left" else 7
    trajectory = min_jerk_trajectory(env.data.ctrl[start:start + 6], result.joint_targets[arm], 16)
    for joints in trajectory:
        action = env.data.ctrl.copy()
        action[start:start + 6] = joints
        action[start + 6] = gripper
        step_recorded(env, action, arm, phase, record)


def _reach(
    env: AlohaPhysicalEnv, ik: AlohaIK, target: Callable[[], np.ndarray], gripper: float,
    arm: str, phase: str, record: list[dict] | None, rounds: int = 5,
) -> None:
    """Drive the fingertip site onto a moving target through the actuators.

    Each round re-reads the target, then shifts the IK goal by the measured
    site error, which cancels actuator sag without touching joint state.
    """
    start = 0 if arm == "left" else 7
    site = env.model.site(f"{arm}/gripper").id
    offset = np.zeros(3)
    for _ in range(rounds):
        goal = target()
        result = ik.solve(env.data.qpos, {arm: (goal + offset, grip_quaternion(arm))},
                          orientation_tolerance=0.5)
        for joints in min_jerk_trajectory(env.data.ctrl[start:start + 6], result.joint_targets[arm], 20):
            action = env.data.ctrl.copy()
            action[start:start + 6] = joints
            action[start + 6] = gripper
            step_recorded(env, action, arm, phase, record)
        _hold(env, 10, arm, phase, record)
        error = goal - env.data.site_xpos[site]
        if float(np.linalg.norm(error)) < 0.005:
            return
        offset += error


def _servo_site(
    env: AlohaPhysicalEnv, ik: AlohaIK, target: np.ndarray, gripper: float,
    arm: str, phase: str, record: list[dict] | None, rounds: int = 6,
) -> None:
    """Walk the gripper site toward a target in short steps.

    One IK solve from a distant pose can overshoot, and the position actuators
    sag a couple of centimetres, so each round commands only the next 2.5cm.
    """
    for _ in range(rounds):
        site = env.data.site_xpos[env.model.site(f"{arm}/gripper").id].copy()
        error = np.asarray(target, dtype=np.float64) - site
        if float(np.linalg.norm(error)) < 0.008:
            return
        goal = site + np.clip(error, -0.02, 0.02)
        try:
            move_arm(env, ik, goal, gripper, 24, arm, phase, record, quaternion=grip_quaternion(arm))
        except RuntimeError:
            _nudge(env, ik, goal, gripper, arm, phase, record)


def run_baton_handoff(
    env: AlohaPhysicalEnv, record_frames: bool = False, carrier: str = "left",
) -> AlohaHandoffResult:
    """Carrier lifts, receiver pinches the far end, carrier unloads, receiver holds."""
    if carrier not in {"left", "right"}:
        raise ValueError(f"unsupported carrier {carrier!r}")
    receiver = "right" if carrier == "left" else "left"
    carrier_grip = 6 if carrier == "left" else 13
    receiver_grip = 13 if carrier == "left" else 6
    sign = -1.0 if carrier == "left" else 1.0
    ik = AlohaIK(env.model)
    record: list[dict] | None = [] if record_frames else None
    carrier_pose = grip_quaternion(carrier)
    initial = env.oracle_state()["task_block_pos"].copy()
    carrier_grasp = initial + [sign * NECK, 0, 0.0]
    move_arm(env, ik, carrier_grasp + [0, 0, 0.14], OPEN, 50, carrier, "CARRIER_APPROACH", record,
             quaternion=carrier_pose)
    _servo_site(env, ik, carrier_grasp + SITE_SAG, OPEN, carrier, "CARRIER_PRE_GRASP", record)
    _hold(env, 100, carrier, "CARRIER_GRASP", record, {carrier_grip: CLOSED})
    carrier_site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
    move_arm(env, ik, carrier_site + [0, 0, 0.14], CLOSED, 100, carrier, "CARRIER_LIFT", record,
             quaternion=carrier_pose)
    peak = float(env.oracle_state()["task_block_pos"][2])
    _hold(env, 40, carrier, "CARRIER_SETTLE", record)

    # Pull toward the bases in y only. Sliding along x walks the neck out of
    # the fingers; y is across the pinch and stays inside both workspaces.
    for _ in range(6):
        site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
        if abs(site[1] - 0.10) < 0.012:
            break
        dy = float(np.clip(0.10 - site[1], -0.025, 0.025))
        try:
            move_arm(env, ik, site + np.array([0.0, dy, SITE_SAG[2]]), CLOSED, 24, carrier,
                     "CARRIER_PRESENT", record, quaternion=carrier_pose)
        except RuntimeError:
            break
    _hold(env, 20, carrier, "PRESENT_SETTLE", record)

    # The baton droops slowly in a single pinch, so the receiver tracks the
    # live position of the free neck instead of a snapshot.
    body = env.model.body("task_block").id

    def neck_above(height: float) -> Callable[[], np.ndarray]:
        def target() -> np.ndarray:
            axis = env.data.xmat[body].reshape(3, 3)[:, 0]
            return env.data.xpos[body] + axis * (-sign * NECK) + [0.0, 0.0, height]
        return target

    _reach(env, ik, neck_above(0.14), OPEN, receiver, "RECEIVER_APPROACH", record)
    _reach(env, ik, neck_above(0.06), OPEN, receiver, "RECEIVER_APPROACH", record)
    _reach(env, ik, neck_above(PINCH_HEIGHT), OPEN, receiver, "RECEIVER_PRE_GRASP", record)
    _hold(env, 120, receiver, "RECEIVER_CLOSE", record, {receiver_grip: CLOSED})
    receiver_contacts = 0
    for _ in range(80):
        _hold(env, 1, receiver, "DUAL_HOLD", record, {receiver_grip: CLOSED})
        receiver_contacts += int(body_touches(env.model, env.data, "task_block", f"{receiver}/"))
    released = receiver_contacts >= 20 and float(env.state_vector()[receiver_grip]) > 0.010
    if released:
        _hold(env, 40, carrier, "CARRIER_RELEASE", record, {carrier_grip: OPEN})
        receiver_site = env.data.site_xpos[env.model.site(f"{receiver}/gripper").id].copy()
        try:
            move_arm(env, ik, receiver_site + [0, 0, 0.08], CLOSED, 40, receiver, "RECEIVER_LIFT",
                     record, quaternion=grip_quaternion(receiver))
        except RuntimeError:
            pass
    _hold(env, 30, receiver, "RECEIVER_SETTLE", record, {receiver_grip: CLOSED})
    after_release = float(env.oracle_state()["task_block_pos"][2])
    carrier_site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
    try:
        move_arm(env, ik, carrier_site + [0, 0, 0.08], OPEN if released else CLOSED, 40, carrier,
                 "CARRIER_RETREAT", record, quaternion=carrier_pose)
    except RuntimeError:
        pass
    final = float(env.oracle_state()["task_block_pos"][2])
    state = env.state_vector()
    success = handoff_succeeded(
        float(initial[2]), peak, after_release, final,
        float(state[carrier_grip]), float(state[receiver_grip]), receiver_contacts,
    )
    return AlohaHandoffResult(
        success, float(initial[2]), peak, after_release, final,
        float(state[carrier_grip]), float(state[receiver_grip]), record or [], receiver_contacts,
        carrier,
    )
