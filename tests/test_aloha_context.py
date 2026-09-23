import numpy as np

from bimanual.control.aloha_context import (
    ALOHA_COOPERATIVE, COOPERATIVE_STATE_NAMES, INTER_ARM_NAMES,
    cooperative_state, inter_arm_context,
)
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def test_inter_arm_context_is_finite_and_has_declared_order():
    env = AlohaPhysicalEnv()
    try:
        context = inter_arm_context(env)
        state = cooperative_state(env)
        assert len(INTER_ARM_NAMES) == 23
        assert len(COOPERATIVE_STATE_NAMES) == 37
        assert context.shape == (23,)
        assert state.shape == (37,)
        np.testing.assert_array_equal(state[:14], env.state_vector())
        np.testing.assert_array_equal(state[14:], context)
        np.testing.assert_allclose(context[-2:], env.state_vector()[[6, 13]])
        assert np.isfinite(state).all()
        ALOHA_COOPERATIVE.validate(state, env.state_vector())
    finally:
        env.close()


def test_inter_arm_context_ignores_object_only_motion():
    env = AlohaPhysicalEnv()
    try:
        before = inter_arm_context(env)
        joint_id = env.model.body("task_block").jntadr[0]
        env.data.qpos[env.model.jnt_qposadr[joint_id]] += 0.04
        import mujoco

        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(inter_arm_context(env), before, atol=1e-6)
    finally:
        env.close()
