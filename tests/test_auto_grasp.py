"""Tests for BimanualTableEnv's auto-grasp/release mechanism (env.py):
proximity + closed-gripper attach, independent of the scripted expert's
explicit grasp()/release() calls. Added after discovering (docs/
decisions.md Phase 4 notes) that contact friction alone provides ~0
holding force with the vendored gripper, which meant a learned policy
(no access to phase-label-driven grasp() calls) could never hold
anything. See env.py's GRASPABLE_OBJECTS/AUTO_GRASP_* docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.env import AUTO_GRASP_CLOSE_FRACTION
from bimanual.sim import robot_spec as rs


def _set_gripper_ctrl(env: BimanualTableEnv, prefix: str, qpos_value: float) -> np.ndarray:
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    ctrl = env.data.ctrl.copy()
    ctrl[names.index(rs.gripper_joint_name(prefix))] = qpos_value
    return ctrl


def test_gripper_closed_fraction_endpoints():
    env = BimanualTableEnv()
    env._set_joint_qpos(rs.gripper_joint_name("left"), rs.GRIPPER_OPEN)
    assert env._gripper_closed_fraction("left") == 0.0
    env._set_joint_qpos(rs.gripper_joint_name("left"), rs.GRIPPER_CLOSED)
    assert env._gripper_closed_fraction("left") == 1.0


def test_auto_grasp_does_not_engage_when_far_from_every_object():
    env = BimanualTableEnv()
    ctrl = _set_gripper_ctrl(env, "left", rs.GRIPPER_CLOSED)
    for _ in range(5):
        env.step(ctrl)
    # HOME_POSE keeps the arm well clear of every object (docs/decisions.md
    # Phase 3 notes), so closing the gripper there must not grab anything.
    assert env._held["left"] is None


def _pin_object_to_left_ee(env: BimanualTableEnv, obj: str) -> None:
    """Teleports `obj` to the left EE's current position with zero
    velocity -- a free-floating object placed mid-air would otherwise
    fall/drift away from the EE under gravity before the (real, finite-
    speed) gripper actuator has time to actually close, which isn't what
    these tests are checking."""
    import mujoco

    ee_pos = env.proprio()["left_ee_pos"].copy()
    adr = env._obj_qpos_adr(obj)
    env.data.qpos[adr : adr + 3] = ee_pos
    joint_id = env.model.body(obj).jntadr[0]
    dof_start = env.model.jnt_dofadr[joint_id]
    env.data.qvel[dof_start : dof_start + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_auto_grasp_engages_when_ee_is_placed_at_an_object_and_gripper_closes():
    env = BimanualTableEnv()
    ctrl = _set_gripper_ctrl(env, "left", rs.GRIPPER_CLOSED)
    for _ in range(10):  # position actuator needs several ticks to actually close
        _pin_object_to_left_ee(env, "mug")
        env.step(ctrl)
    assert env._held["left"] is not None
    assert env._held["left"][0] == "mug"


def test_auto_release_engages_when_gripper_reopens():
    env = BimanualTableEnv()
    ctrl_closed = _set_gripper_ctrl(env, "left", rs.GRIPPER_CLOSED)
    for _ in range(10):
        _pin_object_to_left_ee(env, "mug")
        env.step(ctrl_closed)
    assert env._held["left"] is not None

    for _ in range(15):
        env.step(_set_gripper_ctrl(env, "left", rs.GRIPPER_OPEN))
    assert env._held["left"] is None


def test_auto_grasp_never_double_assigns_object_to_both_arms():
    env = BimanualTableEnv()
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    ctrl = env.data.ctrl.copy()
    ctrl[names.index(rs.gripper_joint_name("left"))] = rs.GRIPPER_CLOSED
    ctrl[names.index(rs.gripper_joint_name("right"))] = rs.GRIPPER_CLOSED
    for _ in range(10):
        _pin_object_to_left_ee(env, "mug")
        env.step(ctrl)

    held_objs = [h[0] for h in env._held.values() if h is not None]
    assert held_objs.count("mug") <= 1


def test_physical_mode_enables_arm_object_contact_and_disables_attachment_api():
    env = BimanualTableEnv(ResetOptions(interaction_mode="physical_contact"))
    arm_geom = next(
        i for i in range(env.model.ngeom)
        if env.model.body(env.model.geom_bodyid[i]).name.startswith("left_")
    )
    object_body = env.model.body("mug").id
    object_geom = next(i for i in range(env.model.ngeom) if env.model.geom_bodyid[i] == object_body)
    assert env.model.geom_contype[arm_geom] & env.model.geom_conaffinity[object_geom]
    assert env.model.geom_contype[object_geom] & env.model.geom_conaffinity[arm_geom]
    with pytest.raises(RuntimeError, match="disabled in physical_contact"):
        env.grasp("left", "mug")
