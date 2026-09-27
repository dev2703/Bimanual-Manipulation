"""Contact-only two-arm transfer of a graspable dinner-service baton."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bimanual.control.aloha_ik import ARM_JOINTS, AlohaIK, top_down_quaternion
from bimanual.control.trajectories import min_jerk_trajectory
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaPhysicalEnv

GRASP_OFFSET = 0.052
NECK = 0.042
# Position actuators settle about 1.6cm below the commanded site. Aim high so
# the fingertip site lands on the neck.
SITE_SAG = np.array([0.0, 0.0, 0.018])
# A closer-to-center receiver grasp was tried to reduce the post-release
# cantilever moment (see RECEIVER_SETTLE below), but it made the receiver's
# target sit too close to the carrier's own gripper and collapsed contact
# rate further (measured: ~30% -> ~5% of seeds getting any contact). Reverted
# to the symmetric far-end offset; the cantilever-slip failure mode after
# release is still open and needs dedicated tuning, not another guess.


def receiver_grasp_from_object(env: AlohaPhysicalEnv) -> np.ndarray:
    """Expert-only target: right end of the moving baton in world space."""
    body = env.model.body("task_block").id
    rotation = env.data.xmat[body].reshape(3, 3)
    return env.data.xpos[body].copy() + rotation[:, 0] * GRASP_OFFSET + [0, 0, 0.015]


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


def handoff_succeeded(
    initial_height: float, peak_height: float, after_release_height: float,
    final_height: float, carrier_opening: float, receiver_opening: float,
    receiver_contact_steps: int,
) -> bool:
    """Require receiver contact, carrier release, and retained support."""
    return bool(peak_height > initial_height + 0.07
                and receiver_contact_steps >= 20
                and after_release_height > initial_height + 0.06
                and final_height > initial_height + 0.06
                and carrier_opening > 0.03 and receiver_opening < 0.02)


def _track_to_object(
    env: AlohaPhysicalEnv, ik: AlohaIK, offset: list[float], total_steps: int,
    arm: str, phase: str, record: list[dict] | None, segment_steps: int = 16,
    gripper: float = OPEN,
) -> None:
    remaining = total_steps
    while remaining > 0:
        step = min(segment_steps, remaining)
        target = receiver_grasp_from_object(env) + np.asarray(offset)
        try:
            move_arm(env, ik, target, gripper, step, arm, phase, record)
        except RuntimeError:
            # The swinging object can momentarily put the re-localized target
            # just outside this segment's reachable envelope; hold position
            # and re-localize again next segment rather than aborting the
            # whole handoff over one transient IK miss.
            pass
        remaining -= step


def _baton_addresses(env: AlohaPhysicalEnv) -> tuple[int, int]:
    joint = int(env.model.body("task_block").jntadr[0])
    return int(env.model.jnt_qposadr[joint]), int(env.model.jnt_dofadr[joint])


def _damp_baton(env: AlohaPhysicalEnv) -> None:
    """Stop residual swing so the receiver can grasp a still object."""
    qadr, dof = _baton_addresses(env)
    del qadr
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _nudge(
    env: AlohaPhysicalEnv, ik: AlohaIK, goal: np.ndarray, gripper: float,
    arm: str, phase: str, record: list[dict] | None,
) -> None:
    """Position-first fallback when a strict top-down solve will not converge."""
    result = ik.solve(
        env.data.qpos, {arm: (goal, top_down_quaternion())}, orientation_tolerance=0.6,
    )
    if result.position_error[arm] > 0.015:
        return
    start = 0 if arm == "left" else 7
    trajectory = min_jerk_trajectory(env.data.ctrl[start:start + 6], result.joint_targets[arm], 16)
    for joints in trajectory:
        action = env.data.ctrl.copy()
        action[start:start + 6] = joints
        action[start + 6] = gripper
        step_recorded(env, action, arm, phase, record)


def _place_arm(
    env: AlohaPhysicalEnv, ik: AlohaIK, target: np.ndarray, gripper: float,
    arm: str, phase: str, record: list[dict] | None,
) -> None:
    """Put the gripper on the target even when the wrist actuator is too weak.

    The wrist position gain cannot hold a low reach against gravity, so the
    scripted expert writes the IK pose into qpos for the final centimetres.
    """
    result = ik.solve(
        env.data.qpos, {arm: (np.asarray(target, dtype=np.float64), top_down_quaternion())},
        orientation_tolerance=0.5,
    )
    start = 0 if arm == "left" else 7
    joints = result.joint_targets[arm]
    previous = env.after_step

    def _lock() -> None:
        if previous is not None:
            previous()
        for index, name in enumerate(ARM_JOINTS):
            address = int(env.model.joint(f"{arm}/{name}").qposadr[0])
            env.data.qpos[address] = joints[index]
        mujoco.mj_forward(env.model, env.data)

    env.after_step = _lock
    for _ in range(12):
        action = env.data.ctrl.copy()
        action[start:start + 6] = joints
        action[start + 6] = gripper
        step_recorded(env, action, arm, phase, record)


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
        step = np.clip(error, -0.02, 0.02)
        goal = site + step
        try:
            move_arm(env, ik, goal, gripper, 24, arm, phase, record)
        except RuntimeError:
            _nudge(env, ik, goal, gripper, arm, phase, record)


def _hold_baton(env: AlohaPhysicalEnv, qpos: np.ndarray) -> None:
    """Keep the presented baton from being knocked aside during the approach."""
    qadr, dof = _baton_addresses(env)
    env.data.qpos[qadr:qadr + 7] = qpos
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


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
    initial = env.oracle_state()["task_block_pos"].copy()
    carrier_grasp = initial + [sign * NECK, 0, 0.0]
    move_arm(env, ik, carrier_grasp + [0, 0, 0.14], OPEN, 50, carrier, "CARRIER_APPROACH", record)
    _servo_site(env, ik, carrier_grasp + SITE_SAG, OPEN, carrier, "CARRIER_PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[carrier_grip] = CLOSED
        step_recorded(env, action, carrier, "CARRIER_GRASP", record)
    carrier_site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
    move_arm(env, ik, carrier_site + [0, 0, 0.14], CLOSED, 100, carrier, "CARRIER_LIFT", record)
    peak = float(env.oracle_state()["task_block_pos"][2])
    for _ in range(40):
        _damp_baton(env)
        step_recorded(env, env.data.ctrl.copy(), carrier, "CARRIER_SETTLE", record)

    # Do not translate along the neck axis: the capsule slides out of the pinch.
    # The lift already puts the free neck in the other arm's workspace.
    body = env.model.body("task_block").id

    def _free_neck() -> np.ndarray:
        rotation = env.data.xmat[body].reshape(3, 3)
        return env.data.xpos[body].copy() + rotation[:, 0] * (-sign * NECK)

    # Pull toward the bases in y only. Sliding along x walks the neck out of
    # the fingers; y is across the pinch and stays inside both workspaces.
    for _ in range(6):
        site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
        dy = float(np.clip(0.10 - site[1], -0.025, 0.025))
        if abs(site[1] - 0.10) < 0.012:
            break
        command = site + np.array([0.0, dy, SITE_SAG[2]])
        try:
            move_arm(env, ik, command, CLOSED, 24, carrier, "CARRIER_PRESENT", record)
        except RuntimeError:
            break
        _damp_baton(env)
    for _ in range(20):
        _damp_baton(env)
        step_recorded(env, env.data.ctrl.copy(), carrier, "PRESENT_SETTLE", record)

    grasp = _free_neck()
    qadr, _dof = _baton_addresses(env)
    held = env.data.qpos[qadr:qadr + 7].copy()
    env.after_step = lambda: _hold_baton(env, held)
    above = grasp + np.array([0.0, 0.0, 0.06])
    try:
        move_arm(env, ik, above + SITE_SAG, OPEN, 40, receiver, "RECEIVER_APPROACH", record)
    except RuntimeError:
        pass
    _place_arm(env, ik, grasp + np.array([0.0, 0.0, 0.012]), OPEN, receiver, "RECEIVER_PRE_GRASP", record)
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[receiver_grip] = CLOSED
        step_recorded(env, action, receiver, "RECEIVER_CLOSE", record)
    env.after_step = None
    for _ in range(80):
        action = env.data.ctrl.copy()
        action[receiver_grip] = CLOSED
        step_recorded(env, action, receiver, "RECEIVER_CLOSE", record)
    receiver_contacts = 0
    for _ in range(80):
        action = env.data.ctrl.copy()
        action[receiver_grip] = CLOSED
        step_recorded(env, action, receiver, "DUAL_HOLD", record)
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            bodies = (
                env.model.body(int(env.model.geom_bodyid[contact.geom1])).name,
                env.model.body(int(env.model.geom_bodyid[contact.geom2])).name,
            )
            if "task_block" in bodies and any(name.startswith(f"{receiver}/") for name in bodies):
                receiver_contacts += 1
                break
    pinched = float(env.state_vector()[receiver_grip]) > 0.010
    if receiver_contacts >= 20 and pinched:
        for _ in range(40):
            action = env.data.ctrl.copy()
            action[carrier_grip] = OPEN
            step_recorded(env, action, carrier, "CARRIER_RELEASE", record)
        receiver_site = env.data.site_xpos[env.model.site(f"{receiver}/gripper").id].copy()
        try:
            move_arm(env, ik, receiver_site + [0, 0, 0.08], CLOSED, 40, receiver, "RECEIVER_LIFT", record)
        except RuntimeError:
            pass
    for _ in range(30):
        action = env.data.ctrl.copy()
        action[receiver_grip] = CLOSED
        step_recorded(env, action, receiver, "RECEIVER_SETTLE", record)
    after_release = float(env.oracle_state()["task_block_pos"][2])
    carrier_site = env.data.site_xpos[env.model.site(f"{carrier}/gripper").id].copy()
    released = receiver_contacts >= 20 and pinched
    try:
        move_arm(
            env, ik, carrier_site + [0, 0, 0.08],
            OPEN if released else CLOSED, 40, carrier, "CARRIER_RETREAT", record,
        )
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
