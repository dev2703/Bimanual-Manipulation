import numpy as np
import pytest

from bimanual.sim.aloha_env import AlohaTableSettingEnv
from bimanual.sim.perturbation import close_drawer_exogenously, move_ungrasped_object


def test_plate_disturbance_changes_object_only():
    env = AlohaTableSettingEnv("assets/robots/aloha/task_table_setting_plate_v2.xml")
    try:
        before = env.oracle_state()["plate_pos"].copy()
        state = env.state_vector().copy()
        move_ungrasped_object(env, "plate", np.array([0.025, -0.015]))
        np.testing.assert_allclose(env.oracle_state()["plate_pos"][:2] - before[:2], [0.025, -0.015])
        np.testing.assert_array_equal(env.state_vector(), state)
        with pytest.raises(ValueError):
            move_ungrasped_object(env, "plate", np.array([0.1, 0.0]))
    finally:
        env.close()


def test_exogenous_drawer_closure_does_not_command_an_actuator():
    env = AlohaTableSettingEnv("assets/robots/aloha/task_table_setting_drawer_v2.xml")
    try:
        joint = env.model.joint("drawer_slide")
        env.data.qpos[joint.qposadr[0]] = -0.12
        import mujoco

        mujoco.mj_forward(env.model, env.data)
        ctrl = env.data.ctrl.copy()
        close_drawer_exogenously(env)
        assert env.data.qpos[joint.qposadr[0]] == 0.0
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()
