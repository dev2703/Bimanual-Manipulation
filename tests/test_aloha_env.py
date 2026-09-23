from __future__ import annotations

import numpy as np
import pytest

from bimanual.sim.aloha_env import (
    ALOHA_BIMANUAL, TABLE_SETTING_OBJECTS, AlohaPhysicalEnv,
    AlohaTableSettingEnv, CONTROL_HZ,
)


def test_aloha_environment_has_declared_14d_contract_and_no_attachment_api():
    env = AlohaPhysicalEnv()
    assert env.model.nu == ALOHA_BIMANUAL.action_dim == 14
    assert env.state_vector().shape == (14,)
    assert not hasattr(env, "grasp")
    assert not hasattr(env, "release")


def test_aloha_step_has_exact_control_period_and_block_is_dynamic():
    env = AlohaPhysicalEnv()
    before = float(env.data.time)
    env.step(env.data.ctrl.copy())
    assert float(env.data.time - before) == pytest.approx(1 / CONTROL_HZ, abs=1e-12)
    block_joint = env.model.body("task_block").jntadr[0]
    assert env.model.jnt_type[block_joint] == 0  # mjJNT_FREE


def test_aloha_neutral_hold_is_finite():
    env = AlohaPhysicalEnv()
    observation = env.step(env.data.ctrl.copy())
    assert np.isfinite(observation["observation.state"]).all()
    assert np.isfinite(env.oracle_state()["task_block_pos"]).all()


def test_showcase_geometry_cannot_affect_physics_or_policy_cameras():
    env = AlohaPhysicalEnv()
    assert env.model.camera("audience_cam").id >= 0
    names = (
        "studio_back_wall", "studio_side_left", "studio_side_right",
        "work_surface_mat", "target_outer", "target_inner",
    )
    for name in names:
        geom = env.model.geom(name)
        assert geom.group == 5
        assert geom.contype == 0
        assert geom.conaffinity == 0
    assert env._policy_scene_option.geomgroup[2] == 1  # Menagerie robot visuals
    assert env._policy_scene_option.geomgroup[5] == 0  # audience-only dressing
    assert env._policy_scene_option.sitegroup[5] == 0  # hidden goal markers
    env.close()


def test_aloha_table_setting_scene_has_full_task_entities_and_14d_policy_contract():
    env = AlohaTableSettingEnv()
    env.reset(seed=9, randomize_objects=True)
    oracle = env.oracle_state()
    assert env.model.nu == 14
    assert env.state_vector().shape == (14,)
    assert set(TABLE_SETTING_OBJECTS) == {key.removesuffix("_pos") for key in oracle if key.endswith("_pos")}
    assert env.model.joint("drawer_slide").range[1] > 0
    assert env.model.geom("drawer_handle").id >= 0
    for name in TABLE_SETTING_OBJECTS:
        assert np.isfinite(oracle[f"{name}_pos"]).all()
    for name in ("room_back_wall", "table_linen", "place_mat", "water_glass"):
        geom = env.model.geom(name)
        assert geom.group == 5
        assert geom.contype == geom.conaffinity == 0
    env.close()
