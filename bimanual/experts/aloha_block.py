"""Contact-only ALOHA block grasp/lift expert and physical gate runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaPhysicalEnv


@dataclass(frozen=True)
class AlohaBlockResult:
    success: bool
    arm: str
    initial_position: np.ndarray
    final_position: np.ndarray
    peak_height: float
    retained_height: float
    record: list[dict]


def run_block_grasp_lift(env: AlohaPhysicalEnv, record_frames: bool = False) -> AlohaBlockResult:
    """Approach, friction-grasp, lift, and hold using only actuator commands."""
    ik = AlohaIK(env.model)
    initial = env.oracle_state()["task_block_pos"].copy()
    arm = "left" if initial[0] <= 0 else "right"
    start = 0 if arm == "left" else 7
    record: list[dict] | None = [] if record_frames else None
    above = initial + np.array([0.0, 0.0, 0.18])
    grasp = initial + np.array([0.0, 0.0, 0.038])

    move_arm(env, ik, above, OPEN, 50, arm, "APPROACH", record)
    move_arm(env, ik, grasp, OPEN, 50, arm, "PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[start + 6] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    current_site = env.data.site_xpos[env.model.site(f"{arm}/gripper").id].copy()
    move_arm(
        env, ik, current_site + np.array([0.0, 0.0, 0.15]), CLOSED, 120,
        arm, "LIFT", record,
    )
    peak = float(env.oracle_state()["task_block_pos"][2])

    for _ in range(30):
        action = env.data.ctrl.copy()
        action[start + 6] = CLOSED
        step_recorded(env, action, arm, "HOLD", record)
    final = env.oracle_state()["task_block_pos"].copy()
    retained = float(final[2])
    success = peak >= 0.10 and retained >= 0.08
    return AlohaBlockResult(success, arm, initial, final, peak, retained, record or [])
