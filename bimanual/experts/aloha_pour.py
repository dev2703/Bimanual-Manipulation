"""Scripted pour-pose expert.

The right arm places the mug on the table. The left arm holds the bottle so the
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
POUR_GAP = 0.0254
MUG_RIM_OFFSET = 0.030


def pour_alignment(env) -> dict[str, float]:
    """Measure the outlet relative to the center of the glass rim."""
    mouth, bottle_up = _mouth(env)
    mug_id = env.model.body("mug").id
    mug_up = env.data.xmat[mug_id].reshape(3, 3)[:, 2]
    rim = env.data.xpos[mug_id] + mug_up * MUG_RIM_OFFSET
    delta = mouth - rim
    return {
        "mouth_xy_error": float(np.linalg.norm(delta[:2])),
        "rim_gap": float(delta[2]),
        "mug_upright_cosine": float(mug_up[2]),
        "bottle_upright_cosine": float(bottle_up[2]),
    }


def outlet_aligned(metrics: dict[str, float]) -> bool:
    """Centered within 5 mm, one inch above the rim within 3 mm."""
    return bool(
        metrics["mouth_xy_error"] <= 0.005
        and abs(metrics["rim_gap"] - POUR_GAP) <= 0.003
        and metrics["mug_upright_cosine"] >= 0.98
        and metrics["bottle_upright_cosine"] < 0.88
    )


@dataclass(frozen=True)
class AlohaPourResult:
    success: bool
    mug_height: float
    bottle_height: float
    tilt_cosine: float
    mouth_xy_error: float
    dwell_steps: int
    record: list[dict]
    rim_gap: float = float("nan")
    mug_upright_cosine: float = float("nan")
    target_gap: float = POUR_GAP
    failure_reason: str | None = None


def pour_pose_reached(
    mug_height: float, bottle_height: float, tilt_cosine: float,
    mouth_xy_error: float, dwell_steps: int,
    mug_opening: float, bottle_opening: float,
    table_supported: bool = False,
) -> bool:
    """Bottle mouth dwells above a mug that is still being held."""
    return bool(
        mug_height > (0.02 if table_supported else 0.08)
        and bottle_height > (0.02 if table_supported else 0.10)
        and tilt_cosine < 0.88
        and mouth_xy_error < 0.10
        and dwell_steps >= 20
        # The gripper may pinch either the mug body or its 12 mm handle. An
        # empty fully closed gripper settles near the 2 mm control limit.
        and (mug_opening > 0.030 if table_supported else 0.005 < mug_opening < 0.036)
        and 0.005 < bottle_opening < 0.034
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


def _level_mug(env, ik, record) -> None:
    """Correct small handle-grasp sag without translating the cup."""
    for _ in range(8):
        body = env.model.body("mug").id
        up = env.data.xmat[body].reshape(3, 3)[:, 2]
        angle = float(np.arccos(np.clip(up[2], -1, 1)))
        current_center = env.data.xpos[body].copy()
        translation = np.zeros(3)
        if angle < .04 and np.linalg.norm(translation) < .002:
            return
        axis = np.cross(up, [0., 0., 1.])
        if np.linalg.norm(axis) < 1e-8:
            return
        axis /= np.linalg.norm(axis)
        quaternion = np.empty(4)
        mujoco.mju_axisAngle2Quat(quaternion, axis, min(angle, .10))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        site_id = env.model.site("right/gripper").id
        center = env.data.xpos[body].copy()
        target = center + translation + rotation @ (env.data.site_xpos[site_id] - center)
        mujoco.mju_mat2Quat(quaternion, (rotation @ env.data.site_xmat[site_id].reshape(3, 3)).ravel())
        try:
            move_arm(env, ik, target, CLOSED, 30, "right", "LEVEL_MUG", record, quaternion)
        except RuntimeError:
            return


def _align_outlet(env, ik, record, desired_rotation) -> str | None:
    """Approach the measured rim in small translations under joint control."""
    for _ in range(40):
        if env.mug_upright_cosine() < .98:
            return "glass_not_upright"
        if env.state_vector()[6] <= .005:
            return "bottle_grasp_lost"
        mouth, _ = _mouth(env)
        mug_id = env.model.body("mug").id
        rim = env.data.xpos[mug_id] + env.data.xmat[mug_id].reshape(3, 3)[:, 2] * MUG_RIM_OFFSET
        delta = rim + [0, 0, POUR_GAP] - mouth
        if np.linalg.norm(delta) < .002:
            return
        delta = np.clip(delta, -.008, .008)
        site_id = env.model.site("left/gripper").id
        quaternion = np.empty(4)
        body_id = env.model.body("bottle").id
        correction = desired_rotation @ env.data.xmat[body_id].reshape(3, 3).T
        target = mouth + delta + correction @ (env.data.site_xpos[site_id] - mouth)
        mujoco.mju_mat2Quat(quaternion, (correction @ env.data.site_xmat[site_id].reshape(3, 3)).ravel())
        solution = ik.solve(env.data.qpos, {"left": (target, quaternion)},
                            max_iters=240, position_tolerance=.005)
        if not solution.converged["left"]:
            return "outlet_target_unreachable"
        start = env.data.ctrl.copy()
        for alpha in np.linspace(0., 1., 24):
            action = start.copy()
            action[:6] = (1-alpha)*start[:6] + alpha*solution.joint_targets["left"]
            action[6] = CLOSED
            step_recorded(env, action, "left", "ALIGN_OUTLET", record)
    return "alignment_iteration_limit"


def _grasp(env, ik, object_name: str, arm: str, gripper_index: int, height: float, record,
           lateral_offset: float = 0.0) -> None:
    initial = env.oracle_state()[f"{object_name}_pos"].copy()
    # Grip the mug's handle explicitly rather than catching its rim and
    # allowing the cup to hang at an angle during presentation.
    body_id = env.model.body(object_name).id
    initial += env.data.xmat[body_id].reshape(3, 3)[:, 0] * lateral_offset
    move_arm(env, ik, initial + [0, 0, 0.16], OPEN, 45, arm, "APPROACH", record)
    move_arm(env, ik, initial + [0, 0, height], OPEN, 45, arm, "PRE_GRASP", record)
    for _ in range(80):
        action = env.data.ctrl.copy()
        action[gripper_index] = CLOSED
        step_recorded(env, action, arm, "GRASP", record)


def run_pour_pose(
    env: AlohaTableSettingEnv, record_frames: bool = False,
    bottle_grasp_height: float = 0.025,
    mug_anchor: np.ndarray | None = None,
    tilt_delta: float = -1.134464,
    bottle_anchor: np.ndarray | None = None,
) -> AlohaPourResult:
    """Place the glass, clear the right arm, then align a side-grasped bottle."""
    record: list[dict] | None = [] if record_frames else None
    ik = AlohaIK(env.model)
    _grasp(env, ik, "mug", "right", 13, 0.010, record, lateral_offset=0.040)
    mug_site = env.data.site_xpos[env.model.site("right/gripper").id].copy()
    move_arm(env, ik, mug_site + [0, 0, 0.14], CLOSED, 80, "right", "MUG_LIFT", record)
    # Stow the mug on the right, clear of the left arm's bottle pickup.
    # An open drawer occupies the original carry corridor. Keep the mug's
    # bottom above its front and lift the bottle before translating across it.
    anchor = (np.array([0.11, 0.11, 0.24])
              if mug_anchor is None else np.asarray(mug_anchor, dtype=np.float64))
    move_arm(env, ik, anchor, CLOSED, 80, "right", "MUG_PRESENT", record)
    _level_mug(env, ik, record)
    site_id = env.model.site("right/gripper").id
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, env.data.site_xmat[site_id].copy())
    desired_mug = np.array([.06, .06, .033])
    target = env.data.site_xpos[site_id].copy() + desired_mug - env.oracle_state()["mug_pos"]
    move_arm(env, ik, target, CLOSED, 100, "right", "GLASS_PLACE", record, quaternion)
    for _ in range(40):
        action = env.data.ctrl.copy()
        action[13] = OPEN
        step_recorded(env, action, "right", "GLASS_RELEASE", record)
    move_arm(env, ik, env.data.site_xpos[site_id].copy() + [0, 0, .16],
             OPEN, 60, "right", "GLASS_RETRACT", record, quaternion)
    move_arm(env, ik, np.array([.30, .12, .20]), OPEN, 70,
             "right", "CLEAR_POUR_WORKSPACE", record)
    bottle = env.oracle_state()["bottle_pos"].copy()
    side = np.array([1., 0., 0., 0.])
    for target, gripper, steps, phase in (
        (bottle + [0, 0, .16], OPEN, 60, "BOTTLE_APPROACH"),
        (bottle + [0, 0, bottle_grasp_height], OPEN, 60, "BOTTLE_PRE_GRASP"),
        (bottle + [0, 0, bottle_grasp_height], CLOSED, 80, "BOTTLE_GRASP"),
        (bottle + [0, 0, bottle_grasp_height + .20], CLOSED, 100, "BOTTLE_LIFT"),
    ):
        move_arm(env, ik, target, gripper, steps, "left", phase, record, side)
    tilt = abs(tilt_delta)
    final_up = np.array([np.sin(tilt), 0., np.cos(tilt)])
    mug_id = env.model.body("mug").id
    rim = env.data.xpos[mug_id] + env.data.xmat[mug_id].reshape(3, 3)[:, 2] * MUG_RIM_OFFSET
    center = (rim + [0, 0, POUR_GAP + .030] - final_up * MOUTH_OFFSET
              if bottle_anchor is None else np.asarray(bottle_anchor))
    site_id = env.model.site("left/gripper").id
    target = env.data.site_xpos[site_id].copy() + center - env.oracle_state()["bottle_pos"]
    move_arm(env, ik, target, CLOSED, 80, "left", "BOTTLE_PARK", record, side)
    for angle in np.linspace(0, abs(tilt_delta), 7)[1:]:
        c, s = np.cos(angle), np.sin(angle)
        desired_rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        body_id = env.model.body("bottle").id
        correction = desired_rotation @ env.data.xmat[body_id].reshape(3, 3).T
        target = center + correction @ (env.data.site_xpos[site_id] - env.data.xpos[body_id])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (correction @ env.data.site_xmat[site_id].reshape(3, 3)).ravel())
        move_arm(env, ik, target, CLOSED, 35, "left", "BOTTLE_TILT", record, quaternion)
    failure_reason = _align_outlet(env, ik, record, desired_rotation)
    dwell = 0
    best_tilt = 1.0
    best_error = 1.0
    for _ in range(25):
        action = env.data.ctrl.copy()
        action[6] = CLOSED
        action[13] = OPEN
        step_recorded(env, action, "left", "POUR_DWELL", record)
        mouth, up = _mouth(env)
        mug = env.oracle_state()["mug_pos"]
        error = float(np.linalg.norm(mouth[:2] - mug[:2]))
        best_tilt = min(best_tilt, float(up[2]))
        best_error = min(best_error, error)
        if outlet_aligned(pour_alignment(env)) and float(mug[2]) > 0.02:
            dwell += 1
        else:
            dwell = 0
    mug = env.oracle_state()["mug_pos"]
    bottle = env.oracle_state()["bottle_pos"]
    state = env.state_vector()
    alignment = pour_alignment(env)
    success = outlet_aligned(alignment) and pour_pose_reached(
        float(mug[2]), float(bottle[2]), best_tilt, best_error, dwell,
        float(state[13]), float(state[6]), table_supported=True,
    )
    return AlohaPourResult(
        success, float(mug[2]), float(bottle[2]), float(up[2]),
        alignment["mouth_xy_error"], dwell, record or [],
        alignment["rim_gap"], alignment["mug_upright_cosine"],
        POUR_GAP, None if success else (failure_reason or "dwell_alignment_failed"),
    )
