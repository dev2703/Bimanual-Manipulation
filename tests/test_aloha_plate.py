"""Physical handled-plate placement and false-positive guards."""

from pathlib import Path

import numpy as np

from bimanual.evaluation.aloha_predicates import plate_placed
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv

PLATE_SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_plate_v2.xml"


def test_plate_variant_does_not_replace_mug_scene():
    baseline = AlohaTableSettingEnv()
    variant = AlohaTableSettingEnv(PLATE_SCENE)
    try:
        assert baseline.model.geom("plate_geom").size[1] == 0.007
        assert variant.model.geom("plate_geom").size[1] == 0.012
        assert variant.model.geom("plate_handle").id >= 0
        assert baseline.oracle_state()["plate_pos"][2] == 0.007
        assert variant.oracle_state()["plate_pos"][2] == 0.012
    finally:
        baseline.close()
        variant.close()


def test_plate_expert_carries_and_releases_on_held_out_scenes():
    for seed in (100_000, 100_013, 100_041):
        env = AlohaTableSettingEnv(PLATE_SCENE)
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = run_plate_pick_place(env)
            assert result.success
            assert result.peak_height > result.initial_position[2] + 0.08
            assert result.transit_height > result.initial_position[2] + 0.06
            assert np.linalg.norm(result.final_position[:2] - result.target_position[:2]) < 0.04
            assert result.final_upright_cosine > 0.95
            assert env.state_vector()[13] > 0.03
        finally:
            env.close()


def test_untouched_or_unreleased_plate_cannot_pass():
    initial = np.array([0.15, -0.02, 0.012])
    target = np.array([0.0, -0.105, 0.009])
    common = dict(
        peak_height=0.15, carried_to_target=True,
        upright_cosine=1.0, gripper_opening=0.037,
    )
    assert not plate_placed(initial, initial, target, **common)
    assert not plate_placed(initial, np.array([0.01, -0.105, 0.011]), target,
                            **{**common, "carried_to_target": False})
    assert not plate_placed(initial, np.array([0.01, -0.105, 0.011]), target,
                            **{**common, "gripper_opening": 0.005})
