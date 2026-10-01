import numpy as np
from bimanual.experts.aloha_pour import POUR_GAP
from bimanual.evaluation.pour_geometry import outlet_aligned, outlet_target, stream_metrics


def test_stream_target_accounts_for_horizontal_exit_velocity():
    up = np.array([np.sin(1.13), 0., np.cos(1.13)])
    rim = np.zeros(3)
    mouth = outlet_target(rim, up, gap=.10)
    assert outlet_aligned(stream_metrics(mouth, up, rim, np.eye(3)))
    # A source directly over the center can still overshoot the opening.
    assert not outlet_aligned(stream_metrics(np.array([0., 0., .10]), up, rim, np.eye(3)))


def test_stream_rejects_tableward_and_below_rim_paths():
    up = np.array([np.sin(1.13), 0., np.cos(1.13)])
    for mouth in [np.array([.08, 0., .08]), np.array([0., 0., -.02])]:
        assert not outlet_aligned(stream_metrics(mouth, up, np.zeros(3), np.eye(3)))


def test_static_preview_is_aligned_but_not_a_physical_rollout():
    from scripts.preview_pour_pose import preview_env, set_preview_pose
    env = preview_env()
    try:
        report = set_preview_pose(env)
        assert report["alignment_passed"]
        assert abs(report["rim_gap"] - POUR_GAP) < 1e-8
        assert report["mouth_xy_error"] < 1e-8
        assert all(report["arm_ik_converged"].values())
        assert not report["physical_rollout"]
    finally:
        env.close()


def test_stream_rejects_tipped_glass_and_untilted_bottle():
    up = np.array([np.sin(1.13), 0., np.cos(1.13)])
    rim = np.zeros(3)
    metrics = stream_metrics(outlet_target(rim, up), up, rim, np.eye(3))
    assert not outlet_aligned({**metrics, "mug_upright_cosine": .70})
    assert not outlet_aligned({**metrics, "bottle_upright_cosine": 1.})


def test_alignment_stops_on_real_finger_glass_contact_before_moving():
    import mujoco
    import numpy as np
    from bimanual.experts.aloha_pour import _align_outlet
    from bimanual.skills.registry import get_skill
    env = get_skill("pour_pose").make_env()
    try:
        address = env.model.joint("mug_free").qposadr[0]
        env.data.qpos[address:address + 3] = env.data.geom_xpos[env.model.geom("left/right_g1").id]
        mujoco.mj_forward(env.model, env.data)
        diagnostics = {}
        time_before = env.data.time
        assert _align_outlet(env, None, None, np.eye(3), diagnostics) == "gripper_glass_collision"
        assert diagnostics["blocking_contacts"]
        assert env.data.time == time_before
    finally:
        env.close()
