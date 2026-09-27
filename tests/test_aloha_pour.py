from bimanual.experts.aloha_pour import pour_pose_reached


def test_pour_pose_requires_a_lean_dwell_over_a_held_mug():
    assert pour_pose_reached(0.12, 0.20, 0.80, 0.07, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.03, 0.20, 0.80, 0.07, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.97, 0.07, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.80, 0.20, 25, 0.025, 0.030)
    assert not pour_pose_reached(0.12, 0.20, 0.80, 0.07, 2, 0.025, 0.030)
