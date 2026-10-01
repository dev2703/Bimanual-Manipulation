"""Scripted pour-pose expert.

The right arm places the mug on the table. The left arm holds the bottle so the
mouth dwells over the mug. This measures a geometric pour pose. It does not
claim that liquid moved. Bead transfer is a separate scene and metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from bimanual.control.aloha_ik import AlohaIK
from bimanual.experts.aloha_motion import CLOSED, OPEN, move_arm, step_recorded
from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.evaluation.pour_geometry import (
    APPROACH_GAP, MOUTH_OFFSET, MUG_RIM_OFFSET, bottle_outlet,
    outlet_target, pour_alignment, pour_sample_valid, pour_succeeded,
    bottle_contact_bodies, glass_blocking_contacts,
)

# Kept for the static preview's requested example gap, not a pass criterion.
POUR_GAP = .0254


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
    diagnostics: dict = field(default_factory=dict)
    stream_margin: float | None = None


def _mouth(env: AlohaTableSettingEnv) -> tuple[np.ndarray, np.ndarray]:
    return bottle_outlet(env)


def _level_mug(env, ik, record) -> None:
    """Correct small handle-grasp sag without translating the cup."""
    for _ in range(8):
        body = env.model.body("mug").id
        up = env.data.xmat[body].reshape(3, 3)[:, 2]
        angle = float(np.arccos(np.clip(up[2], -1, 1)))
        if angle < .04:
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
        target = center + rotation @ (env.data.site_xpos[site_id] - center)
        mujoco.mju_mat2Quat(quaternion, (rotation @ env.data.site_xmat[site_id].reshape(3, 3)).ravel())
        try:
            move_arm(env, ik, target, CLOSED, 30, "right", "LEVEL_MUG", record, quaternion)
        except RuntimeError:
            return


def _align_outlet(env, ik, record, desired_rotation, diagnostics) -> str | None:
    """Approach the measured rim in small translations under joint control."""
    for _ in range(40):
        blockers = glass_blocking_contacts(env)
        if blockers:
            diagnostics["blocking_contacts"] = sorted(set(blockers))
            return "gripper_glass_collision"
        if env.mug_upright_cosine() < .98:
            return "glass_not_upright"
        if env.state_vector()[6] <= .005:
            return "bottle_grasp_lost"
        if pour_sample_valid(env):
            return None
        mouth, _ = _mouth(env)
        mug_id = env.model.body("mug").id
        rim = env.data.xpos[mug_id] + env.data.xmat[mug_id].reshape(3, 3)[:, 2] * MUG_RIM_OFFSET
        delta = outlet_target(rim, desired_rotation[:, 2]) - mouth
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
        diagnostics["last_ik_position_error"] = solution.position_error["left"]
        diagnostics["last_ik_orientation_error"] = solution.orientation_error["left"]
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
    bottle_grasp_height: float = 0.075,
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
    center = (outlet_target(rim, final_up) + [0, 0, .030] - final_up * MOUTH_OFFSET
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
    diagnostics = {}
    failure_reason = _align_outlet(env, ik, record, desired_rotation, diagnostics)
    dwell = 0
    for _ in range(25):
        action = env.data.ctrl.copy()
        action[6] = CLOSED
        action[13] = OPEN
        step_recorded(env, action, "left", "POUR_DWELL", record)
        mouth, up = _mouth(env)
        mug = env.oracle_state()["mug_pos"]
        if pour_sample_valid(env):
            dwell += 1
        else:
            dwell = 0
    mug = env.oracle_state()["mug_pos"]
    bottle = env.oracle_state()["bottle_pos"]
    alignment = pour_alignment(env)
    contacts = bottle_contact_bodies(env)
    diagnostics["bottle_contact_bodies"] = contacts
    diagnostics["bottle_has_two_finger_contacts"] = all(
        f"left/{finger}_finger_link" in contacts for finger in ("left", "right")
    )
    success = failure_reason is None and pour_succeeded(env, dwell)
    return AlohaPourResult(
        success, float(mug[2]), float(bottle[2]), float(up[2]),
        alignment["mouth_xy_error"], dwell, record or [],
        alignment["rim_gap"], alignment["mug_upright_cosine"],
        APPROACH_GAP, None if success else (failure_reason or "dwell_alignment_failed"),
        diagnostics, alignment["stream_margin"],
    )
