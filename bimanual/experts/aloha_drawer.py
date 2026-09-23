"""Open the clearance-correct ALOHA drawer using handle contact only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.evaluation.aloha_contacts import right_gripper_touches_drawer_handle
from bimanual.evaluation.aloha_predicates import drawer_opened
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv


@dataclass(frozen=True)
class AlohaDrawerResult:
    success: bool
    initial_opening: float
    peak_opening: float
    final_opening: float
    handle_contact_steps: int
    record: list[dict]


def run_drawer_open(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaDrawerResult:
    """Pinch the handle, pull its slide joint, release, and verify retention."""
    arm = "right"
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    initial = float(env.oracle_state()["drawer_opening"][0])
    handle = env.data.geom_xpos[env.model.geom("drawer_handle").id].copy()
    grasp = handle + [0.025, 0, 0.026]

    move_arm(env, ik, grasp + [0, 0, 0.15], OPEN, 60, arm, "APPROACH", record)
    move_arm(env, ik, grasp, OPEN, 60, arm, "PRE_GRASP", record)
    for _ in range(80):
        action = env.data.ctrl.copy()
        action[13] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    contact_steps = 0
    previous_after_step = env.after_step

    def count_handle_contact() -> None:
        nonlocal contact_steps
        if previous_after_step is not None:
            previous_after_step()
        contact_steps += int(right_gripper_touches_drawer_handle(env.model, env.data))

    site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    try:
        env.after_step = count_handle_contact
        move_arm(env, ik, site + [0, -0.15, 0], CLOSED, 200, arm, "PULL", record)
    finally:
        env.after_step = previous_after_step
    peak = float(env.oracle_state()["drawer_opening"][0])

    for _ in range(40):
        action = env.data.ctrl.copy()
        action[13] = OPEN
        step_recorded(env, action, arm, "RELEASE", record)
    site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    move_arm(env, ik, site + [0, 0, 0.12], OPEN, 70, arm, "RETRACT", record)
    for _ in range(40):
        step_recorded(env, env.data.ctrl.copy(), arm, "SETTLE", record)

    final = float(env.oracle_state()["drawer_opening"][0])
    success = drawer_opened(
        initial, peak, final,
        handle_contact_steps=contact_steps,
        gripper_opening=float(env.state_vector()[13]),
    )
    return AlohaDrawerResult(success, initial, peak, final, contact_steps, record or [])
