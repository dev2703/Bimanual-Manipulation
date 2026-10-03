import numpy as np
import pytest

from bimanual.evaluation.aloha_predicates import mug_placed
from bimanual.evaluation.skill_gate import reset_for_skill
from bimanual.skills.registry import RolloutState, SKILLS, get_skill, handoff_carrier, recovery_shift


def test_handoff_instruction_names_the_arms_of_each_episode():
    skill = get_skill("baton_handoff")
    for seed in (100000, 100001):
        carrier = handoff_carrier(seed)
        receiver = "right" if carrier == "left" else "left"
        assert skill.instruction_for(seed) == f"Hand the baton from the {carrier} gripper to the {receiver} gripper."
    assert skill.instruction_for(2) != skill.instruction_for(3)
    assert get_skill("mug_pick_place").instruction_for(7) == get_skill("mug_pick_place").instruction


def test_memory_instruction_lists_dinner_goals_already_done():
    assert get_skill("drawer_open").memory_instruction_for(0) == (
        "Task: Set the dinner table. Done: nothing yet. Now: drawer open with right arm.")
    assert get_skill("spoon_place").memory_instruction_for(0) == (
        "Task: Set the dinner table. Done: drawer open, plate pick place, fork place. "
        "Now: spoon place with right arm.")
    assert (get_skill("plate_recovery").memory_instruction_for(3)
            == get_skill("plate_pick_place").memory_instruction_for(3))
    assert get_skill("block_lift").memory_instruction_for(0) == (
        "Task: Lift the block off the table. Done: nothing yet. Now: block lift with both arms.")


def test_visual_randomization_changes_colours_but_not_the_scene_state():
    skill = get_skill("plate_pick_place")
    env = skill.make_env()
    try:
        robot = np.array([env.model.body(int(b)).name.startswith(("left/", "right/"))
                          for b in env.model.geom_bodyid])
        reset_for_skill(skill, env, 7, jitter=0.015)
        plain_qpos, plain_rgba = env.data.qpos.copy(), env.model.geom_rgba.copy()
        plain_mat, plain_light = env.model.mat_rgba.copy(), env.model.light_diffuse.copy()
        reset_for_skill(skill, env, 7, jitter=0.015, visuals=True)
        np.testing.assert_array_equal(env.data.qpos, plain_qpos)
        np.testing.assert_array_equal(env.model.geom_rgba[robot], plain_rgba[robot])
        assert (not np.array_equal(env.model.mat_rgba, plain_mat)
                or not np.array_equal(env.model.geom_rgba, plain_rgba))
        assert not np.array_equal(env.model.light_diffuse, plain_light)
        drawn = env.model.light_diffuse.copy()
        reset_for_skill(skill, env, 7, jitter=0.015, visuals=True)
        np.testing.assert_array_equal(env.model.light_diffuse, drawn)
        reset_for_skill(skill, env, 7, jitter=0.015)
        np.testing.assert_array_equal(env.model.geom_rgba, plain_rgba)
        np.testing.assert_array_equal(env.model.light_diffuse, plain_light)
    finally:
        env.close()


def test_recovery_shift_is_seeded_and_fixed_length():
    assert np.allclose(recovery_shift(5), recovery_shift(5))
    assert not np.allclose(recovery_shift(5), recovery_shift(6))
    assert np.isclose(np.linalg.norm(recovery_shift(5)), 0.035)


@pytest.mark.parametrize("name", sorted(SKILLS))
def test_untouched_scene_fails_every_skill_check(name):
    """Idle arms must never be scored as success, and every skill must be scorable."""
    skill = get_skill(name)
    env = skill.make_env()
    try:
        reset_for_skill(skill, env, 100000, jitter=0.015)
        state = skill.begin(env)
        for _ in range(30):
            env.step(env.data.ctrl.copy())
            skill.update(env, state)
        assert not skill.succeeded(env, state)
    finally:
        env.close()


def test_registry_covers_the_dinner_skills():
    assert get_skill("mug_pick_place").gate_passed
    assert get_skill("baton_handoff").gate_passed
    # Passed under the ballistic stream proxy (no simulated liquid).
    assert get_skill("pour_pose").gate_passed
    try:
        get_skill("missing")
    except KeyError as exc:
        assert "mug_pick_place" in str(exc)


def test_mug_rollout_check_matches_mug_placed():
    skill = get_skill("mug_pick_place")
    state = RolloutState(
        initial=np.array([0.3, 0.12, 0.03]),
        peak_height=0.16,
        position=np.array([0.15, -0.02, 0.029]),
        target=np.array([0.15, -0.02, 0.0]),
        carried_to_target=True,
        initial_height=0.03,
    )

    class _Env:
        def mug_upright_cosine(self):
            return 0.99

        def state_vector(self):
            return np.zeros(14)

    env = _Env()
    env.state_vector = lambda: np.concatenate([np.zeros(13), [0.037]])
    assert skill.succeeded(env, state) is mug_placed(
        state.initial, state.position, state.target,
        peak_height=state.peak_height, carried_to_target=True,
        upright_cosine=0.99, gripper_opening=0.037,
    )
