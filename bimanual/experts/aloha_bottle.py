"""Contact-only bottle grasp/lift primitive for eventual cooperative pouring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv


@dataclass(frozen=True)
class AlohaBottleResult:
    success: bool
    initial_position: np.ndarray
    final_position: np.ndarray
    peak_height: float
    record: list[dict]


def run_bottle_grasp_lift(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaBottleResult:
    arm = "left"
    gripper_index = 6
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    initial = env.oracle_state()["bottle_pos"].copy()

    move_arm(env, ik, initial + [0, 0, 0.18], OPEN, 50, arm, "APPROACH", record)
    move_arm(env, ik, initial + [0, 0, 0.03], OPEN, 50, arm, "PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
    move_arm(env, ik, site + [0, 0, 0.15], CLOSED, 120, arm, "LIFT", record)
    peak = float(env.oracle_state()["bottle_pos"][2])
    for _ in range(45):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "HOLD", record)

    final = env.oracle_state()["bottle_pos"].copy()
    success = bool(
        peak > initial[2] + 0.10
        and final[2] > initial[2] + 0.10
        # The bottle's finite width prevents the fingers reaching the closed
        # command. A squeeze below 3.4 cm is the observed contact regime.
        and env.state_vector()[gripper_index] < 0.034
    )
    return AlohaBottleResult(success, initial, final, peak, record or [])
