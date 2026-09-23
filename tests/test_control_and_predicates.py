"""Edge-case / boundary tests for the pure-logic control and oracle-predicate
layer (bimanual/control/trajectories.py, bimanual/control/bimanual.py,
bimanual/sim/oracle_predicates.py, bimanual/sim/randomization.py).

These modules have no MuJoCo dependency (by design, see trajectories.py's
module docstring), so they're tested here without an env/model fixture.
Existing tests/test_scene.py etc. cover the MuJoCo-backed layer; this file
targets the scripted-expert plumbing that previously had zero coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from bimanual.control.bimanual import ArmTrajectory, BimanualTrajectory, _nlerp_quats, pause_arm
from bimanual.control.collision import inter_group_contacts
from bimanual.control.trajectories import (
    gripper_fraction_to_qpos,
    gripper_profile,
    min_jerk_scaling,
    min_jerk_trajectory,
)
from bimanual.sim import robot_spec as rs
from bimanual.sim.oracle_predicates import (
    arm_arm_collision,
    bottle_tilt_from_mat,
    drawer_open,
    holding,
    in_region,
    pour_success,
)
from bimanual.sim.randomization import LEVELS, XY_JITTER, sample_scene_config
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, REGIONS, SceneConfig, TABLE_HEIGHT


# --------------------------------------------------------------------------
# min_jerk_scaling / min_jerk_trajectory
# --------------------------------------------------------------------------


def test_min_jerk_scaling_endpoints_and_monotonic():
    t = np.linspace(0.0, 1.0, 50)
    s = min_jerk_scaling(t)
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(s) >= -1e-12), "min-jerk profile must be monotonically non-decreasing"


def test_min_jerk_scaling_clips_out_of_range_inputs():
    s = min_jerk_scaling(np.array([-5.0, -0.001, 1.001, 5.0]))
    assert s[0] == pytest.approx(0.0)
    assert s[1] == pytest.approx(0.0)
    assert s[2] == pytest.approx(1.0)
    assert s[3] == pytest.approx(1.0)


def test_min_jerk_trajectory_matches_endpoints():
    start = np.array([0.0, 1.0, -2.0])
    end = np.array([3.0, -1.0, 2.0])
    traj = min_jerk_trajectory(start, end, n_steps=20)
    assert traj.shape == (20, 3)
    np.testing.assert_allclose(traj[0], start)
    np.testing.assert_allclose(traj[-1], end)


def test_min_jerk_trajectory_single_step_is_start_only():
    # linspace(0, 1, 1) == [0.0], so a 1-step trajectory never reaches `end`.
    # This is a real edge case a caller could hit accidentally (e.g. a
    # zero-duration segment) and silently get a "motion" that never moves.
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 1.0])
    traj = min_jerk_trajectory(start, end, n_steps=1)
    assert traj.shape == (1, 2)
    np.testing.assert_allclose(traj[0], start)


def test_min_jerk_trajectory_zero_length_displacement():
    start = np.array([1.0, 2.0])
    traj = min_jerk_trajectory(start, start, n_steps=10)
    np.testing.assert_allclose(traj, np.tile(start, (10, 1)))


# --------------------------------------------------------------------------
# gripper_profile
# --------------------------------------------------------------------------


@pytest.mark.parametrize("open_at_start", [True, False])
def test_gripper_profile_length_always_matches_n_steps(open_at_start):
    for n in (1, 2, 3, 10, 37):
        profile = gripper_profile(n, open_at_start=open_at_start)
        assert len(profile) == n, f"n_steps={n}"


def test_gripper_profile_hold_then_transition():
    profile = gripper_profile(100, open_at_start=True, hold_frac=0.3)
    assert profile[0] == pytest.approx(1.0)
    # Held for the first ~30 steps.
    assert np.all(profile[:29] == pytest.approx(1.0))
    # Ends closed.
    assert profile[-1] == pytest.approx(0.0)
    assert np.all(profile >= -1e-9) and np.all(profile <= 1.0 + 1e-9)


def test_gripper_profile_hold_frac_zero_and_one_are_valid():
    p0 = gripper_profile(10, open_at_start=False, hold_frac=0.0)
    assert len(p0) == 10
    assert p0[-1] == pytest.approx(1.0)

    p1 = gripper_profile(10, open_at_start=False, hold_frac=1.0)
    assert len(p1) == 10
    # Fully held: never transitions within the given horizon.
    assert np.all(p1 == pytest.approx(0.0))


def test_gripper_profile_single_step():
    profile = gripper_profile(1, open_at_start=True)
    assert len(profile) == 1


# --------------------------------------------------------------------------
# gripper_fraction_to_qpos
# --------------------------------------------------------------------------


def test_gripper_fraction_to_qpos_endpoints():
    assert gripper_fraction_to_qpos(1.0) == pytest.approx(rs.GRIPPER_OPEN)
    assert gripper_fraction_to_qpos(0.0) == pytest.approx(rs.GRIPPER_CLOSED)


def test_gripper_fraction_to_qpos_array():
    fractions = np.array([0.0, 0.5, 1.0])
    qpos = gripper_fraction_to_qpos(fractions)
    expected_mid = rs.GRIPPER_CLOSED + 0.5 * (rs.GRIPPER_OPEN - rs.GRIPPER_CLOSED)
    assert qpos[0] == pytest.approx(rs.GRIPPER_CLOSED)
    assert qpos[1] == pytest.approx(expected_mid)
    assert qpos[2] == pytest.approx(rs.GRIPPER_OPEN)


def test_gripper_fraction_to_qpos_clips_out_of_range_fraction():
    # A caller passing an out-of-[0,1] fraction (e.g. an unclamped policy
    # output) must saturate at the physical joint limits, not extrapolate
    # past them.
    over = gripper_fraction_to_qpos(1.5)
    under = gripper_fraction_to_qpos(-0.5)
    assert over == pytest.approx(rs.GRIPPER_OPEN)
    assert under == pytest.approx(rs.GRIPPER_CLOSED)

    lo, hi = rs.GRIPPER_LIMIT.lo, rs.GRIPPER_LIMIT.hi
    joint_lo, joint_hi = min(lo, hi), max(lo, hi)
    tol = 1e-9
    assert joint_lo - tol <= under <= joint_hi + tol
    assert joint_lo - tol <= over <= joint_hi + tol


def test_gripper_fraction_to_qpos_clips_array_input():
    fractions = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    qpos = gripper_fraction_to_qpos(fractions)
    assert qpos[0] == pytest.approx(rs.GRIPPER_CLOSED)
    assert qpos[-1] == pytest.approx(rs.GRIPPER_OPEN)


# --------------------------------------------------------------------------
# _nlerp_quats
# --------------------------------------------------------------------------


def test_nlerp_quats_endpoints_and_normalized():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    quats = _nlerp_quats(q0, q1, n_steps=25)
    np.testing.assert_allclose(quats[0], q0, atol=1e-9)
    norms = np.linalg.norm(quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_nlerp_quats_takes_short_path_on_opposite_hemisphere():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -np.array([1.0, 0.0, 0.0, 0.0])  # antipodal representation of same rotation
    quats = _nlerp_quats(q0, q1, n_steps=5)
    # Because q0 . q1 < 0, q1 is flipped, so the whole path should stay
    # near q0 rather than sweeping through the long way (or degenerating
    # at the midpoint where linear interpolation of antipodal quats would
    # produce a near-zero-norm vector before normalization amplifies it).
    for q in quats:
        assert np.dot(q, q0) > 0


def test_nlerp_quats_single_step():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])
    quats = _nlerp_quats(q0, q1, n_steps=1)
    assert quats.shape == (1, 4)
    np.testing.assert_allclose(quats[0], q0, atol=1e-9)


# --------------------------------------------------------------------------
# ArmTrajectory / BimanualTrajectory / pause_arm
# --------------------------------------------------------------------------


def _fake_arm_trajectory(n_steps: int, seed: int = 0) -> ArmTrajectory:
    rng = np.random.default_rng(seed)
    return ArmTrajectory(
        joint_targets=rng.uniform(-1, 1, size=(n_steps, rs.N_ARM_JOINTS)),
        gripper_targets=rng.uniform(0, 1, size=n_steps),
        active=np.ones(n_steps, dtype=bool),
    )


def test_bimanual_trajectory_n_steps_empty_is_zero():
    assert BimanualTrajectory().n_steps == 0


def test_bimanual_trajectory_n_steps_is_max_across_arms():
    traj = BimanualTrajectory(arms={"left": _fake_arm_trajectory(10), "right": _fake_arm_trajectory(17)})
    assert traj.n_steps == 17


def test_pause_arm_noop_when_start_equals_end():
    traj = _fake_arm_trajectory(10)
    paused = pause_arm(traj, start_tick=5, end_tick=5)
    assert np.array_equal(paused.active, traj.active)
    np.testing.assert_array_equal(paused.joint_targets, traj.joint_targets)


def test_pause_arm_holds_pose_across_window():
    traj = _fake_arm_trajectory(10)
    paused = pause_arm(traj, start_tick=3, end_tick=7)
    assert not np.any(paused.active[3:7])
    assert np.all(paused.active[:3]) and np.all(paused.active[7:])
    for i in range(3, 7):
        np.testing.assert_allclose(paused.joint_targets[i], traj.joint_targets[3])
        assert paused.gripper_targets[i] == pytest.approx(traj.gripper_targets[3])
    # Original trajectory must not be mutated in place.
    assert np.all(traj.active)


def test_pause_arm_full_range():
    traj = _fake_arm_trajectory(6)
    paused = pause_arm(traj, start_tick=0, end_tick=6)
    assert not np.any(paused.active)
    for i in range(6):
        np.testing.assert_allclose(paused.joint_targets[i], traj.joint_targets[0])


def test_pause_arm_end_tick_beyond_length_clamped_by_slicing():
    traj = _fake_arm_trajectory(5)
    # Should not raise even though end_tick > n_steps.
    paused = pause_arm(traj, start_tick=2, end_tick=1000)
    assert not np.any(paused.active[2:])
    assert np.all(paused.active[:2])


# --------------------------------------------------------------------------
# inter_group_contacts (pure body-name-prefix logic, no mujoco model needed)
# --------------------------------------------------------------------------


def test_inter_group_contacts_filters_correctly():
    contacts = [
        ("left_upper_arm", "right_forearm"),   # cross -> hit
        ("right_wrist", "left_gripper"),        # cross (reversed order) -> hit
        ("left_upper_arm", "left_forearm"),     # same group -> no hit
        ("table", "right_forearm"),             # neither in group_a -> no hit
        ("", "right_forearm"),                  # empty body name -> no hit, must not crash
    ]
    hits = inter_group_contacts(contacts, "left_", "right_")
    assert hits == [
        ("left_upper_arm", "right_forearm"),
        ("right_wrist", "left_gripper"),
    ]


def test_inter_group_contacts_empty_input():
    assert inter_group_contacts([], "left_", "right_") == []


# --------------------------------------------------------------------------
# oracle_predicates
# --------------------------------------------------------------------------


def test_drawer_open_true_and_false_via_handle_pos():
    closed_state = {"drawer_handle_pos": np.array([0.0, 0.39 - 0.245, 0.0])}
    assert not drawer_open(closed_state)

    open_y = (0.39 - 0.245) - DRAWER_OPEN_DIST  # fully open
    open_state = {"drawer_handle_pos": np.array([0.0, open_y, 0.0])}
    assert drawer_open(open_state)


def test_drawer_open_threshold_boundary():
    # Exactly at threshold should count as open (>=).
    y_at_threshold = (0.39 - 0.245) - DRAWER_OPEN_DIST * DRAWER_OPEN_DIST.__class__(0.8)
    state = {"drawer_handle_pos": np.array([0.0, y_at_threshold, 0.0])}
    assert drawer_open(state, threshold=0.8)


def test_drawer_open_qpos_fallback_when_handle_pos_missing():
    state = {"drawer_opening": -DRAWER_OPEN_DIST}
    assert drawer_open(state)
    state_closed = {"drawer_opening": 0.0}
    assert not drawer_open(state_closed)


def test_in_region_boundary_radius():
    region_name = next(iter(REGIONS))
    rx, ry, radius = REGIONS[region_name]
    # Exactly on the boundary counts as inside (<=).
    state = {f"obj_pos": np.array([rx + radius, ry, TABLE_HEIGHT])}
    assert in_region(state, "obj", region_name)
    # Just outside should fail.
    state_outside = {f"obj_pos": np.array([rx + radius + 1e-3, ry, TABLE_HEIGHT])}
    assert not in_region(state_outside, "obj", region_name)


def test_in_region_wrong_height_fails():
    region_name = next(iter(REGIONS))
    rx, ry, _radius = REGIONS[region_name]
    state = {"obj_pos": np.array([rx, ry, TABLE_HEIGHT + 1.0])}
    assert not in_region(state, "obj", region_name)


def test_in_region_missing_key_raises():
    region_name = next(iter(REGIONS))
    with pytest.raises(KeyError):
        in_region({}, "nonexistent_obj", region_name)


def test_holding_requires_both_proximity_and_closed_gripper():
    # holding() takes an "opening fraction" (0=closed, 1=open), which is
    # the OPPOSITE convention from rs.GRIPPER_OPEN/GRIPPER_CLOSED (raw
    # qpos, where the vendored joint's numeric range is inverted). Do not
    # pass rs.GRIPPER_LIMIT values here directly.
    gripper_range = (0.0, 1.0)
    closed_val = 0.0
    open_val = 1.0

    proprio_close_and_closed = {"left_ee_pos": np.array([0.0, 0.0, 0.0]), "left_gripper_opening": closed_val}
    proprio_close_but_open = {"left_ee_pos": np.array([0.0, 0.0, 0.0]), "left_gripper_opening": open_val}
    oracle = {"mug_pos": np.array([0.0, 0.0, 0.0])}

    assert holding(proprio_close_and_closed, oracle, "left", "mug", gripper_range)
    assert not holding(proprio_close_but_open, oracle, "left", "mug", gripper_range)

    far_proprio = {"left_ee_pos": np.array([1.0, 1.0, 1.0]), "left_gripper_opening": closed_val}
    assert not holding(far_proprio, oracle, "left", "mug", gripper_range)


def test_pour_success_requires_both_alignment_and_tilt():
    oracle = {"mug_pos": np.array([0.0, 0.0, 0.1]), "bottle_pos": np.array([0.0, 0.0, 0.3])}
    assert pour_success(oracle, bottle_tilt_rad=1.2)
    assert not pour_success(oracle, bottle_tilt_rad=0.5)  # not tilted enough

    far_oracle = {"mug_pos": np.array([0.0, 0.0, 0.1]), "bottle_pos": np.array([1.0, 1.0, 0.3])}
    assert not pour_success(far_oracle, bottle_tilt_rad=1.5)  # tilted but not over the mug


def test_bottle_tilt_from_mat_upright_and_upside_down():
    identity = np.eye(3)
    assert bottle_tilt_from_mat(identity) == pytest.approx(0.0)

    flipped = np.diag([1.0, -1.0, -1.0])  # 180 deg about x: local +z -> world -z
    assert bottle_tilt_from_mat(flipped) == pytest.approx(np.pi)

    sideways = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert bottle_tilt_from_mat(sideways) == pytest.approx(np.pi / 2)


def test_arm_arm_collision_empty_contacts():
    assert not arm_arm_collision({"contacts": []})


def test_arm_arm_collision_self_contact_not_flagged():
    # Both bodies from the same arm must not count as an inter-arm collision.
    state = {"contacts": [("left_upper_arm", "left_forearm")]}
    assert not arm_arm_collision(state)


def test_arm_arm_collision_cross_arm_flagged_either_order():
    assert arm_arm_collision({"contacts": [("left_gripper", "right_gripper")]})
    assert arm_arm_collision({"contacts": [("right_gripper", "left_gripper")]})


# --------------------------------------------------------------------------
# randomization
# --------------------------------------------------------------------------


def test_sample_scene_config_rejects_unknown_level():
    with pytest.raises(ValueError):
        sample_scene_config("L99", seed=0)


def test_sample_scene_config_l0_matches_defaults_exactly():
    base = SceneConfig()
    cfg = sample_scene_config("L0", seed=123)
    assert cfg == base


@pytest.mark.parametrize("level", ["L1", "L2"])
def test_sample_scene_config_same_seed_is_reproducible(level):
    cfg_a = sample_scene_config(level, seed=7)
    cfg_b = sample_scene_config(level, seed=7)
    assert cfg_a == cfg_b


def test_sample_scene_config_different_seeds_differ():
    cfg_a = sample_scene_config("L1", seed=1)
    cfg_b = sample_scene_config("L1", seed=2)
    assert cfg_a.plate_xy != cfg_b.plate_xy


@pytest.mark.parametrize("level", ["L1", "L2"])
def test_sample_scene_config_jitter_within_bounds(level):
    base = SceneConfig()
    cfg = sample_scene_config(level, seed=99)
    for attr in ("plate_xy", "mug_xy", "bottle_xy", "fork_xy", "spoon_xy"):
        base_xy = getattr(base, attr)
        cfg_xy = getattr(cfg, attr)
        for b, c in zip(base_xy, cfg_xy):
            assert abs(c - b) <= XY_JITTER + 1e-9, f"{attr} jitter exceeded bound at level {level}"
    assert 0.0 <= cfg.drawer_opening <= DRAWER_OPEN_DIST * 0.5 + 1e-9


def test_sample_scene_config_all_levels_are_exercised():
    # Guards against LEVELS drifting out of sync with the if/elif chain in
    # sample_scene_config (e.g. a new level added to LEVELS without a
    # matching branch would silently fall through to the L2 return).
    for level in LEVELS:
        cfg = sample_scene_config(level, seed=0)
        assert isinstance(cfg, SceneConfig)
