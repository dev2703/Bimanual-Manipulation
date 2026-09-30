from bimanual.experts.aloha_pour import pour_pose_reached
from bimanual.experts.aloha_pour import outlet_aligned, POUR_GAP


def test_outlet_requires_centered_one_inch_gap_and_upright_glass():
    metrics = dict(mouth_xy_error=0., rim_gap=POUR_GAP,
                   mug_upright_cosine=1., bottle_upright_cosine=.42)
    assert outlet_aligned(metrics)
    for key, value in [("mouth_xy_error", .006), ("rim_gap", -.01),
                       ("rim_gap", .05), ("mug_upright_cosine", .7),
                       ("bottle_upright_cosine", 1.)]:
        assert not outlet_aligned({**metrics, key: value})


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


def test_pour_pose_requires_a_lean_dwell_over_a_held_mug():
    assert pour_pose_reached(0.12, 0.20, 0.80, 0.07, 25, 0.025, 0.030)
    assert pour_pose_reached(0.12, 0.20, 0.80, 0.07, 25, 0.008, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.80, 0.07, 25, 0.002, 0.030)
    assert not pour_pose_reached(0.03, 0.20, 0.80, 0.07, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.97, 0.07, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.80, 0.20, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.80, 0.07, 2, 0.025, 0.030)
