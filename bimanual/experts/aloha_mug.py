"""Contact-only mug pick-and-place expert for the dinner-table scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.evaluation.aloha_predicates import mug_placed
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv


@dataclass(frozen=True)
class AlohaMugResult:
    success: bool
    initial_position: np.ndarray
    final_position: np.ndarray
    target_position: np.ndarray
    peak_height: float
    transit_height: float
    final_upright_cosine: float
    record: list[dict]


def run_mug_pick_place(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaMugResult:
    """Place the mug to the right of the plate using only joint/gripper control."""
    arm = "right"
    gripper_index = 13
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    initial = env.oracle_state()["mug_pos"].copy()
    target = env.data.site_xpos[env.model.site("mug_region").id].copy()

    move_arm(env, ik, initial + [0, 0, 0.18], OPEN, 50, arm, "APPROACH", record)
    # Grip low enough to carry laterally, but above the table contact plane.
    move_arm(env, ik, initial + [0, 0, 0.030], OPEN, 50, arm, "PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    site_id = env.model.site("right/gripper").id
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.15], CLOSED, 120, arm, "LIFT", record)
    peak_height = float(env.oracle_state()["mug_pos"][2])

    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, np.array([target[0], target[1], site[2]]), CLOSED, 120, arm, "TRANSPORT", record)
    transit_position = env.oracle_state()["mug_pos"].copy()
    transit_height = float(transit_position[2])
    move_arm(env, ik, np.array([target[0], target[1], 0.075]), CLOSED, 70, arm, "PLACE", record)

    for _ in range(30):
        action = env.data.ctrl.copy()
        action[gripper_index] = OPEN
        step_recorded(env, action, arm, "RELEASE", record)
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.14], OPEN, 80, arm, "RETRACT", record)
    for _ in range(30):
        step_recorded(env, env.data.ctrl.copy(), arm, "SETTLE", record)

    final = env.oracle_state()["mug_pos"].copy()
    upright = env.mug_upright_cosine()
    success = mug_placed(
        initial, final, target,
        peak_height=peak_height,
        carried_to_target=bool(
            transit_height > initial[2] + 0.08
            and np.linalg.norm(transit_position[:2] - target[:2]) < 0.05
        ),
        upright_cosine=upright,
        gripper_opening=float(env.state_vector()[gripper_index]),
    )
    return AlohaMugResult(
        success, initial, final, target, peak_height, transit_height, upright,
        record or [],
    )
