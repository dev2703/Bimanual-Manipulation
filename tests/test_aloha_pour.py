import numpy as np
import pytest
from bimanual.evaluation.pour_geometry import (
    MAX_GAP, OPENING_RADIUS, STREAM_RADIUS, landing_points, lower_lip, outlet_aligned,
    outlet_target, stream_metrics,
)

TILTED_UP = np.array([np.sin(1.13), 0., np.cos(1.13)])


def _shifted_stream(offset):
    rim = np.zeros(3)
    return stream_metrics(outlet_target(rim, TILTED_UP) + offset, TILTED_UP, rim, np.eye(3))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_stream_shifted_past_the_opening_lands_on_the_table(axis, sign):
    offset = np.zeros(3)
    offset[axis] = sign * OPENING_RADIUS
    metrics = _shifted_stream(offset)
    assert metrics["stream_margin"] < 0
    assert not outlet_aligned(metrics)


def test_stream_margin_changes_sign_at_the_opening_edge():
    centered = _shifted_stream(np.zeros(3))["stream_margin"]
    assert 0 < centered < OPENING_RADIUS - STREAM_RADIUS
    # Shift along the pour direction, where the margin is exactly the slack to the edge.
    inside = np.array([.99 * centered, 0., 0.])
    outside = np.array([1.01 * centered + 1e-4, 0., 0.])
    assert outlet_aligned(_shifted_stream(inside))
    assert not outlet_aligned(_shifted_stream(outside))


def test_source_at_or_below_rim_or_inverted_glass_has_no_landing():
    rim = np.zeros(3)
    source = lower_lip(np.array([0., 0., .05]), TILTED_UP)
    assert landing_points(source - [0., 0., .06], TILTED_UP, rim, np.eye(3)) is None
    assert landing_points(rim, TILTED_UP, rim, np.eye(3)) is None
    inverted = np.diag([1., -1., -1.])
    assert landing_points(source, TILTED_UP, rim, inverted) is None


def test_every_exit_speed_must_land_inside_not_just_the_mean():
    rim = np.zeros(3)
    mouth = outlet_target(rim, TILTED_UP)
    points = landing_points(lower_lip(mouth, TILTED_UP), TILTED_UP, rim, np.eye(3))
    assert len(points) > 2
    assert np.all(np.linalg.norm(points[:, :2], axis=1) <= OPENING_RADIUS - STREAM_RADIUS)


def test_source_too_high_above_rim_is_rejected():
    rim = np.zeros(3)
    mouth = outlet_target(rim, TILTED_UP, gap=MAX_GAP + .02)
    assert not outlet_aligned(stream_metrics(mouth, TILTED_UP, rim, np.eye(3)))


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
    from scripts.preview_pour_pose import POUR_GAP, preview_env, set_preview_pose
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


def test_glass_staging_is_skipped_only_for_an_upright_glass_at_the_target():
    import mujoco
    from bimanual.experts.aloha_pour import GLASS_SET_TOLERANCE, _glass_already_set
    from bimanual.skills.registry import get_skill
    env = get_skill("pour_pose").make_env()
    try:
        env.reset(seed=0)
        mug = env.oracle_state()["mug_pos"].copy()
        assert _glass_already_set(env, mug)
        assert not _glass_already_set(env, mug + [2 * GLASS_SET_TOLERANCE, 0., 0.])
        address = env.model.joint("mug_free").qposadr[0]
        env.data.qpos[address + 3:address + 7] = (np.cos(.3), np.sin(.3), 0., 0.)
        mujoco.mj_forward(env.model, env.data)
        assert not _glass_already_set(env, mug)
    finally:
        env.close()


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
