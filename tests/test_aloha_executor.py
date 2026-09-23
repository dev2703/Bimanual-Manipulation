import numpy as np
import pytest

from bimanual.evaluation.aloha_executor import run_verified_goals


class _Env:
    def __init__(self):
        self.render_count = 0

    def render(self):
        self.render_count += 1
        return {"overhead_cam": np.zeros((2, 2, 3), dtype=np.uint8)}

    def oracle_state(self):
        raise AssertionError("executor must not inspect privileged state")


def test_verified_goals_retry_and_serialize_current_memory():
    env = _Env()
    calls = []
    readings = iter((False, True, True))

    def skill(_env, goal, prompt):
        calls.append((goal, prompt))

    def verifier(frames):
        assert "overhead_cam" in frames
        return {"place_mug": next(readings), "open_drawer": True}

    memory, log = run_verified_goals(
        env, ["place_mug", "open_drawer"], skill, verifier, max_attempts=2,
    )
    assert [row.verified for row in log] == [False, True, True]
    assert memory.state.completed == ["place_mug", "open_drawer"]
    assert len(memory.state.failures) == 1
    assert "Done: place mug" in calls[-1][1]
    assert env.render_count == 3


def test_missing_rgb_predicate_is_an_error():
    with pytest.raises(KeyError, match="place_mug"):
        run_verified_goals(_Env(), ["place_mug"], lambda *_: None, lambda _: {}, max_attempts=1)


def test_failed_goal_never_becomes_completed():
    memory, log = run_verified_goals(
        _Env(), ["place_mug"], lambda *_: None,
        lambda _: {"place_mug": False}, max_attempts=2,
    )
    assert not memory.state.completed
    assert len(log) == 2
    assert len(memory.state.failures) == 2
