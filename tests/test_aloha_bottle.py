from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.sim.aloha_env import AlohaTableSettingEnv


def test_bottle_is_physically_lifted_and_held_on_randomized_scene():
    env = AlohaTableSettingEnv()
    try:
        env.reset(seed=23, randomize_objects=True)
        result = run_bottle_grasp_lift(env)
        assert result.success
        assert result.peak_height > result.initial_position[2] + 0.10
        assert result.final_position[2] > result.initial_position[2] + 0.10
    finally:
        env.close()
