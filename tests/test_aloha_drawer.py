"""Drawer opening requires real contact with a graspable handle."""

from pathlib import Path

from bimanual.evaluation.aloha_predicates import drawer_opened
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.sim.aloha_env import AlohaTableSettingEnv

DRAWER_SCENE = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_drawer_v2.xml"


def test_drawer_handle_variant_has_clearance_from_front_panel():
    baseline = AlohaTableSettingEnv()
    variant = AlohaTableSettingEnv(DRAWER_SCENE)
    try:
        original_y = baseline.data.geom_xpos[baseline.model.geom("drawer_handle").id][1]
        improved_y = variant.data.geom_xpos[variant.model.geom("drawer_handle").id][1]
        assert original_y - improved_y > 0.03
        assert variant.model.geom("drawer_handle_left_mount").id >= 0
    finally:
        baseline.close()
        variant.close()


def test_drawer_expert_grasps_handle_and_keeps_drawer_open():
    for seed in (100_000, 100_013, 100_041):
        env = AlohaTableSettingEnv(DRAWER_SCENE)
        try:
            env.reset(seed=seed, randomize_objects=True)
            result = run_drawer_open(env)
            assert result.success
            assert result.handle_contact_steps >= 20
            assert result.peak_opening > 0.12
            assert result.final_opening > 0.11
            assert env.state_vector()[13] > 0.03
        finally:
            env.close()


def test_original_decorative_handle_does_not_pass_contact_gate():
    env = AlohaTableSettingEnv()
    try:
        result = run_drawer_open(env)
        assert result.handle_contact_steps == 0
        assert not result.success
    finally:
        env.close()


def test_no_contact_or_no_release_cannot_pass_drawer_predicate():
    common = dict(initial_opening=0.0, peak_opening=0.149, final_opening=0.147)
    assert not drawer_opened(**common, handle_contact_steps=0, gripper_opening=0.037)
    assert not drawer_opened(**common, handle_contact_steps=175, gripper_opening=0.005)
    assert not drawer_opened(0.0, 0.02, 0.02, handle_contact_steps=175, gripper_opening=0.037)
