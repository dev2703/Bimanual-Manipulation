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
