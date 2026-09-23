"""Contact-only handled-plate placement in the isolated dinner scene variant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.evaluation.aloha_predicates import plate_placed
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.sim.perturbation import move_ungrasped_object

HANDLE_X_OFFSET = 0.072
GRASP_Z_OFFSET = 0.025


@dataclass(frozen=True)
class AlohaPlateResult:
    success: bool
    initial_position: np.ndarray
    final_position: np.ndarray
    target_position: np.ndarray
    peak_height: float
    transit_height: float
    final_upright_cosine: float
    record: list[dict]
    perturbed: bool = False
    replanned: bool = False


def run_plate_pick_place(
    env: AlohaTableSettingEnv, record_frames: bool = False,
    perturbation_xy: np.ndarray | None = None, replan_after_perturb: bool = True,
) -> AlohaPlateResult:
    """Pinch the physical serving-plate handle, carry, place, and release."""
    arm = "right"
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    initial = env.oracle_state()["plate_pos"].copy()
    target = env.data.site_xpos[env.model.site("plate_region").id].copy()
    grasp = initial + [HANDLE_X_OFFSET, 0, GRASP_Z_OFFSET]

    move_arm(env, ik, grasp + [0, 0, 0.16], OPEN, 50, arm, "APPROACH", record)
    if perturbation_xy is not None:
        move_ungrasped_object(env, "plate", perturbation_xy)
        if replan_after_perturb:
            # The event happens after APPROACH, before PRE_GRASP. A fresh
            # visual observation is recorded by the next control step; this
            # expert's privileged re-localization is an upper bound for a
            # learned visual recovery policy.
            grasp = env.oracle_state()["plate_pos"].copy() + [HANDLE_X_OFFSET, 0, GRASP_Z_OFFSET]
    move_arm(env, ik, grasp, OPEN, 50, arm, "PRE_GRASP", record)
    for _ in range(100):
        action = env.data.ctrl.copy()
        action[13] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)

    site_id = env.model.site("right/gripper").id
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.14], CLOSED, 120, arm, "LIFT", record)
    peak_height = float(env.oracle_state()["plate_pos"][2])

    site = env.data.site_xpos[site_id].copy()
    move_arm(
        env, ik, np.array([target[0] + HANDLE_X_OFFSET, target[1], site[2]]),
        CLOSED, 180, arm, "TRANSPORT", record,
    )
    transit_position = env.oracle_state()["plate_pos"].copy()
    transit_height = float(transit_position[2])
    move_arm(
        env, ik, np.array([target[0] + HANDLE_X_OFFSET, target[1], 0.055]),
        CLOSED, 100, arm, "PLACE", record,
    )
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[13] = OPEN
        step_recorded(env, action, arm, "RELEASE", record)
    site = env.data.site_xpos[site_id].copy()
    move_arm(env, ik, site + [0, 0, 0.13], OPEN, 80, arm, "RETRACT", record)
    for _ in range(40):
        step_recorded(env, env.data.ctrl.copy(), arm, "SETTLE", record)

    final = env.oracle_state()["plate_pos"].copy()
    upright = env.object_upright_cosine("plate")
    success = plate_placed(
        initial, final, target,
        peak_height=peak_height,
        carried_to_target=bool(
            transit_height > initial[2] + 0.06
            and np.linalg.norm(transit_position[:2] - target[:2]) < 0.04
        ),
        upright_cosine=upright,
        gripper_opening=float(env.state_vector()[13]),
    )
    return AlohaPlateResult(
        success, initial, final, target, peak_height, transit_height, upright, record or [],
        perturbation_xy is not None, perturbation_xy is not None and replan_after_perturb,
    )
