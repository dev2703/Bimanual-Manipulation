from pathlib import Path

import numpy as np

from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def test_four_core_skills_compose_in_one_contact_scene():
    scene = Path(__file__).parents[1] / "assets/robots/aloha/task_table_setting_combined_v2.xml"
    env = AlohaTableSettingEnv(scene)
    try:
        for name in ("plate_handle", "drawer_handle", "drawer_handle_left_mount", "drawer_handle_right_mount"):
            geom = env.model.geom(name)
            assert geom.contype and geom.conaffinity
        env.reset(seed=100_000, randomize_objects=True)
        assert run_drawer_open(env).success
        assert run_plate_pick_place(env).success
        assert run_mug_pick_place(env).success
        assert run_bottle_grasp_lift(env).success
        state = env.oracle_state()
        assert state["drawer_opening"][0] > 0.11
        for name, radius in (("plate", 0.04), ("mug", 0.045)):
            target = env.data.site_xpos[env.model.site(f"{name}_region").id]
            assert np.linalg.norm(state[f"{name}_pos"][:2] - target[:2]) < radius
    finally:
        env.close()
