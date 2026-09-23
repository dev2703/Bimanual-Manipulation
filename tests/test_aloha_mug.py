from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def test_dinner_act_uses_bounded_action_normalization():
    config = Path(__file__).parents[1] / "bimanual/training/configs/act_aloha_mug.yaml"
    policy = yaml.safe_load(config.read_text())["policy"]
    assert policy["chunk_size"] == 20 and policy["n_action_steps"] == 8
    assert policy["normalization_mapping"]["ACTION"] == "MIN_MAX"
    assert policy["normalization_mapping"]["STATE"] == "MIN_MAX"


def test_mug_expert_physically_releases_upright_mug_at_distinct_target():
    for seed in (0, 13, 41):
        env = AlohaTableSettingEnv()
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = run_mug_pick_place(env)
            assert result.success
            assert result.transit_height > 0.10
            assert result.final_upright_cosine > 0.9
            assert np.linalg.norm(result.initial_position[:2] - result.target_position[:2]) > 0.1
            assert np.linalg.norm(result.final_position[:2] - result.target_position[:2]) < 0.045
            assert env.state_vector()[13] > 0.03  # gripper is actually open
        finally:
            env.close()


def test_no_action_does_not_satisfy_mug_placement():
    env = AlohaTableSettingEnv()
    try:
        env.reset(seed=17, randomize_objects=True)
        target = env.data.site_xpos[env.model.site("mug_region").id]
        for _ in range(90):
            env.step(env.data.ctrl.copy())
        position = env.oracle_state()["mug_pos"]
        assert np.linalg.norm(position[:2] - target[:2]) > 0.10
    finally:
        env.close()


def test_dinner_scene_settles_without_drawer_creep_or_object_falls():
    for seed in (0, 1, 100_000):
        env = AlohaTableSettingEnv()
        try:
            env.reset(seed=seed, randomize_objects=True)
            initial = env.oracle_state()
            for _ in range(90):
                env.step(env.data.ctrl.copy())
            final = env.oracle_state()
            assert float(final["drawer_opening"][0]) < 0.003
            for name in ("plate", "mug", "bottle", "fork", "spoon"):
                assert abs(final[f"{name}_pos"][2] - initial[f"{name}_pos"][2]) < 0.003
        finally:
            env.close()
