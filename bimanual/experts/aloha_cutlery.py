"""Contact-only fork and spoon placement in the combined dinner scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def _cutlery_grasp_quaternion() -> np.ndarray:
    """Top-down grasp rotated 90 degrees so the jaws cross the long handle."""
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion


@dataclass(frozen=True)
class AlohaCutleryResult:
    success: bool
    utensil: str
    initial_position: np.ndarray
    final_position: np.ndarray
    target_position: np.ndarray
    peak_height: float
    record: list[dict]


def run_cutlery_place(
    env: AlohaTableSettingEnv, utensil: str, record_frames: bool = False,
) -> AlohaCutleryResult:
    if utensil not in {"fork", "spoon"}:
        raise ValueError(f"unsupported utensil: {utensil}")
    arm = "left" if utensil == "fork" else "right"
    gripper_index = 6 if arm == "left" else 13
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    orientation = _cutlery_grasp_quaternion()
    initial = env.oracle_state()[f"{utensil}_pos"].copy()
    target = env.data.site_xpos[env.model.site(f"{utensil}_region").id].copy()

    move_arm(env, ik, initial + [0, 0, 0.16], OPEN, 60, arm, "APPROACH", record, orientation)
    move_arm(env, ik, initial + [0, 0, 0.026], OPEN, 60, arm, "PRE_GRASP", record, orientation)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    site_id = env.model.site(f"{arm}/gripper").id
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.13], CLOSED, 100, arm, "LIFT", record, orientation)
    peak = float(env.oracle_state()[f"{utensil}_pos"][2])
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, np.array([target[0], target[1], site[2]]), CLOSED, 140, arm, "TRANSPORT", record, orientation)
    move_arm(env, ik, np.array([target[0], target[1], 0.040]), CLOSED, 80, arm, "PLACE", record, orientation)
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[gripper_index] = OPEN
        step_recorded(env, action, arm, "RELEASE", record)
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.12], OPEN, 70, arm, "RETRACT", record, orientation)
    for _ in range(30):
        step_recorded(env, env.data.ctrl.copy(), arm, "SETTLE", record)

    final = env.oracle_state()[f"{utensil}_pos"].copy()
    success = bool(
        peak > initial[2] + 0.07
        and np.linalg.norm(final[:2] - target[:2]) < 0.035
        and final[2] < 0.035
        and env.state_vector()[gripper_index] > 0.03
    )
    return AlohaCutleryResult(success, utensil, initial, final, target, peak, record or [])


def run_fork_place(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaCutleryResult:
    return run_cutlery_place(env, "fork", record_frames)


def run_spoon_place(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaCutleryResult:
    return run_cutlery_place(env, "spoon", record_frames)
