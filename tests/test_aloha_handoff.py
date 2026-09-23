from bimanual.experts.aloha_handoff import handoff_succeeded, receiver_grasp_from_object
from bimanual.sim.aloha_env import AlohaPhysicalEnv


def test_handoff_gate_requires_receiver_support_after_carrier_release():
    assert handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.02, 0.02, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 0)


def test_receiver_target_tracks_object_frame():
    env = AlohaPhysicalEnv("assets/robots/aloha/task_handoff_baton.xml")
    try:
        first = receiver_grasp_from_object(env)
        joint_id = env.model.body("task_block").jntadr[0]
        env.data.qpos[env.model.jnt_qposadr[joint_id]] += 0.025
        import mujoco

        mujoco.mj_forward(env.model, env.data)
        second = receiver_grasp_from_object(env)
        assert abs(second[0] - first[0] - 0.025) < 1e-6
    finally:
        env.close()
