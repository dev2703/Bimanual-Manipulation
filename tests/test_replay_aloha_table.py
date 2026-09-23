"""Held-action dinner dataset replay must not credit an untouched table."""

from pathlib import Path

import numpy as np

from bimanual.data.replay_aloha_table import replay_episode
from bimanual.sim.aloha_env import TABLE_SETTING_OBJECTS, AlohaTableSettingEnv


def test_replay_no_action_does_not_place_mug():
    env = AlohaTableSettingEnv()
    try:
        env.reset(seed=100_000, randomize_objects=True)
        row = {
            "privileged.scene_seed": [100_000],
            "observation.state": [env.state_vector().tolist()],
            "action": [env.data.ctrl.copy().tolist()],
            "privileged.object_positions": [[
                value for name in TABLE_SETTING_OBJECTS
                for value in env.oracle_state()[f"{name}_pos"]
            ]],
        }
    finally:
        env.close()
    result = replay_episode(row, 0)
    assert not result.final_success
    assert np.isclose(result.max_target_error, 0.0)


def test_replay_no_action_does_not_place_plate():
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_plate_v2.xml"
    env = AlohaTableSettingEnv(scene)
    try:
        env.reset(seed=100_000, randomize_objects=True)
        row = {
            "privileged.scene_seed": [100_000],
            "observation.state": [env.state_vector().tolist()],
            "action": [env.data.ctrl.copy().tolist()],
            "privileged.object_positions": [[
                value for name in TABLE_SETTING_OBJECTS
                for value in env.oracle_state()[f"{name}_pos"]
            ]],
        }
    finally:
        env.close()
    result = replay_episode(row, 0, "plate_pick_place")
    assert result.target_name == "plate"
    assert not result.final_success
    assert np.isclose(result.max_target_error, 0.0)


def test_replay_no_action_does_not_open_drawer():
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_drawer_v2.xml"
    env = AlohaTableSettingEnv(scene)
    try:
        env.reset(seed=100_000, randomize_objects=True)
        row = {
            "privileged.scene_seed": [100_000],
            "observation.state": [env.state_vector().tolist()],
            "action": [env.data.ctrl.copy().tolist()],
            "privileged.drawer_opening": [[0.0]],
            "privileged.phase": ["PULL"],
        }
    finally:
        env.close()
    result = replay_episode(row, 0, "drawer_open")
    assert result.target_name == "drawer"
    assert not result.final_success
    assert np.isclose(result.max_target_error, 0.0)
