"""Scripted pour-pose expert.

The right arm holds the mug. The left arm holds the bottle and tips it so the
mouth dwells over the mug. This measures a geometric pour pose. It does not
claim that liquid moved. Bead transfer is a separate scene and metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv

MOUTH_OFFSET = 0.081
TILT = 0.95  # radians, about 54 degrees from vertical


@dataclass(frozen=True)
class AlohaPourResult:
    success: bool
    mug_height: float
    bottle_height: float
    tilt_cosine: float
    mouth_xy_error: float
    dwell_steps: int
    record: list[dict]


def pour_pose_reached(
    mug_height: float, bottle_height: float, tilt_cosine: float,
    mouth_xy_error: float, dwell_steps: int,
    mug_opening: float, bottle_opening: float,
) -> bool:
    """Bottle mouth dwells above a mug that is still being held."""
    return bool(
        mug_height > 0.08
        and bottle_height > 0.10
        and tilt_cosine < 0.88
        and mouth_xy_error < 0.10
        and dwell_steps >= 20
        # The gripper may pinch either the mug body or its 12 mm handle. An
        # empty fully closed gripper settles near the 2 mm control limit.
        and 0.005 < mug_opening < 0.036
        and bottle_opening < 0.034
    )


def _tilted_quaternion(angle: float) -> np.ndarray:
    """Tip the top-down grasp about the finger axis."""
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    tip = np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    top = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, (tip @ top).ravel())
    return quaternion


def _nudge_mouth_toward_mug(env, ik, record) -> None:
    """Slide the tipped bottle so its mouth moves toward the mug, a few centimetres at a time."""
    site_id = env.model.site("left/gripper").id
    for _ in range(3):
        mouth, _up = _mouth(env)
        mug = env.oracle_state()["mug_pos"]
        delta = mug[:3] - mouth
        delta[2] = 0.0
        if float(np.linalg.norm(delta)) < 0.04:
            return
        delta = np.clip(delta, -0.035, 0.035)
        site = env.data.site_xpos[site_id].copy()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, env.data.site_xmat[site_id].reshape(3, 3).ravel().copy())
        try:
            move_arm(env, ik, site + delta, CLOSED, 24, "left", "MOUTH_NUDGE", record, quaternion)
        except RuntimeError:
            return


def _pitch_left(env, delta: float, record, release: bool = True) -> None:
    """Tip the left wrist with the position actuators so the bottle stays pinched."""
    del release
    start = env.data.ctrl.copy()
    goal = start.copy()
    goal[4] = float(np.clip(start[4] + delta, -1.6, 2.0))
    for alpha in np.linspace(0.0, 1.0, 50):
        action = start.copy()
        action[:6] = (1.0 - alpha) * start[:6] + alpha * goal[:6]
        action[6] = CLOSED
        action[13] = CLOSED
        step_recorded(env, action, "left", "BOTTLE_TILT", record)


def _mouth(env: AlohaTableSettingEnv) -> tuple[np.ndarray, np.ndarray]:
    body = int(env.model.body("bottle").id)
    rotation = env.data.xmat[body].reshape(3, 3)
    up = rotation[:, 2].copy()
    return env.data.xpos[body].copy() + up * MOUTH_OFFSET, up


def _carry(env, ik, arm: str, object_name: str, desired: np.ndarray, record, phase: str) -> None:
    """Translate a grasped object by moving the gripper site by the same delta."""
    obj = env.oracle_state()[f"{object_name}_pos"].copy()
    site = env.data.site_xpos[env.model.site(f"{arm}/gripper").id].copy()
    move_arm(env, ik, site + (np.asarray(desired) - obj), CLOSED, 60, arm, phase, record)


def _grasp(env, ik, object_name: str, arm: str, gripper_index: int, height: float, record) -> None:
    initial = env.oracle_state()[f"{object_name}_pos"].copy()
    move_arm(env, ik, initial + [0, 0, 0.16], OPEN, 45, arm, "APPROACH", record)
    move_arm(env, ik, initial + [0, 0, height], OPEN, 45, arm, "PRE_GRASP", record)
    for _ in range(80):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)


def run_pour_pose(env: AlohaTableSettingEnv, record_frames: bool = False) -> AlohaPourResult:
    """Hold the mug with the right arm and tip the bottle over it with the left."""
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    _grasp(env, ik, "mug", "right", 13, 0.030, record)
    mug_site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    move_arm(env, ik, mug_site + [0, 0, 0.14], CLOSED, 80, "right", "MUG_LIFT", record)
    # Stow the mug on the right, clear of the left arm's bottle pickup.
    # An open drawer occupies the original carry corridor. Keep the mug's
    # bottom above its front and lift the bottle before translating across it.
    drawer_open = float(env.oracle_state()["drawer_opening"][0]) > 0.11
    anchor = np.array([0.11, 0.11, 0.23 if drawer_open else 0.20])
    move_arm(env, ik, anchor, CLOSED, 80, "right", "MUG_PRESENT", record)
    _grasp(env, ik, "bottle", "left", 6, 0.035, record)
    if drawer_open:
        bottle_site = env.data.site_xpos[env.model.site("left/gripper").id].copy()
        move_arm(env, ik, bottle_site + [0, 0, 0.16], CLOSED, 80,
                 "left", "BOTTLE_LIFT", record)
    try:
        _carry(env, ik, "left", "bottle",
               np.array([-0.10, 0.08, 0.18 if drawer_open else 0.16]), record, "BOTTLE_PARK")
    except RuntimeError:
        pass
    # This target stays below the tilt threshold under actuator control. A
    # larger command reaches farther initially but rebounds during the dwell.
    _pitch_left(env, -0.90, record, release=False)
    dwell = 0
    best_tilt = 1.0
    best_error = 1.0
    for _ in range(25):
        action = env.data.ctrl.copy()
        action[6] = CLOSED
        action[13] = CLOSED
        step_recorded(env, action, "left", "POUR_DWELL", record)
        mouth, up = _mouth(env)
        mug = env.oracle_state()["mug_pos"]
        error = float(np.linalg.norm(mouth[:2] - mug[:2]))
        best_tilt = min(best_tilt, float(up[2]))
        best_error = min(best_error, error)
        if float(up[2]) < 0.88 and error < 0.10 and float(mug[2]) > 0.08:
            dwell += 1
    mug = env.oracle_state()["mug_pos"]
    bottle = env.oracle_state()["bottle_pos"]
    state = env.state_vector()
    success = pour_pose_reached(
        float(mug[2]), float(bottle[2]), best_tilt, best_error, dwell,
        float(state[13]), float(state[6]),
    )
    return AlohaPourResult(
        success, float(mug[2]), float(bottle[2]), best_tilt, best_error, dwell, record or [],
    )
