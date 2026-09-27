import numpy as np

from bimanual.evaluation.aloha_predicates import mug_placed
from bimanual.skills.registry import RolloutState, SKILLS, get_skill


def test_registry_covers_the_dinner_skills():
    assert get_skill("mug_pick_place").gate_passed
    assert get_skill("baton_handoff").gate_passed
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
