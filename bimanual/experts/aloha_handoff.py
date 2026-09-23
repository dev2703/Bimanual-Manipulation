"""Contact-only two-arm transfer of a graspable dinner-service baton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaPhysicalEnv

GRASP_OFFSET = 0.052


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


def run_baton_handoff(env: AlohaPhysicalEnv, record_frames: bool = False) -> AlohaHandoffResult:
    """Left carries, right pinches, left unloads, right retains and retreats."""
    ik = AlohaIK(env.model)
    record: list[dict] | None = [] if record_frames else None
    initial = env.oracle_state()["task_block_pos"].copy()
    left_grasp = initial + [-GRASP_OFFSET, 0, 0.038]
    move_arm(env, ik, left_grasp + [0, 0, 0.16], OPEN, 50, "left", "CARRIER_APPROACH", record)
    move_arm(env, ik, left_grasp, OPEN, 50, "left", "CARRIER_PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[6] = CLOSED
        step_recorded(env, action, "left", "CARRIER_GRASP", record)
    left_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(env, ik, left_site + [0, 0, 0.10], CLOSED, 100, "left", "CARRIER_LIFT", record)
    peak = float(env.oracle_state()["task_block_pos"][2])

    # The receiver target is updated from the carried object's actual pose.
    # This privileged localization belongs only to the scripted expert.
    right_grasp = receiver_grasp_from_object(env)
    move_arm(env, ik, right_grasp + [0, 0, 0.06], OPEN, 80, "right", "RECEIVER_APPROACH", record)
    # The carrier can move while the receiver approaches. Recompute the
    # target from the *current* carried object rather than replaying the
    # original fixed waypoint.
    right_grasp = receiver_grasp_from_object(env)
    move_arm(env, ik, right_grasp, OPEN, 100, "right", "RECEIVER_PRE_GRASP", record)
    receiver_contacts = 0
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[13] = CLOSED
        step_recorded(env, action, "right", "DUAL_HOLD", record)
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            names = (env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name)
            if "handoff_baton" in names and any(name.startswith("right/") for name in names):
                receiver_contacts += 1
                break
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[6] = OPEN
        step_recorded(env, action, "left", "CARRIER_RELEASE", record)
    after_release = float(env.oracle_state()["task_block_pos"][2])
    left_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(env, ik, left_site + [-0.08, 0, 0.08], OPEN, 80, "left", "CARRIER_RETREAT", record)
    right_site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    move_arm(env, ik, right_site + [0.06, 0, 0.06], CLOSED, 80, "right", "RECEIVER_RETREAT", record)
    for _ in range(30):
        step_recorded(env, env.data.ctrl.copy(), "right", "RECEIVER_HOLD", record)
    final = float(env.oracle_state()["task_block_pos"][2])
    state = env.state_vector()
    success = handoff_succeeded(float(initial[2]), peak, after_release, final,
                                float(state[6]), float(state[13]), receiver_contacts)
    return AlohaHandoffResult(success, float(initial[2]), peak, after_release,
                              final, float(state[6]), float(state[13]), record or [], receiver_contacts)
