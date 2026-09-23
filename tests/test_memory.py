"""Tests for bimanual/memory/* (Phase 5, decision A7): pure logic, no
simulation needed.
"""

from __future__ import annotations

import pytest

from bimanual.memory.serialization import serialize_instruction
from bimanual.memory.task_memory import TaskMemory


def test_task_memory_requires_at_least_one_goal():
    with pytest.raises(ValueError):
        TaskMemory("set the table", [])


def test_task_memory_starts_on_first_goal():
    mem = TaskMemory("set the table", ["open_drawer", "pick_place_mug"])
    assert mem.state.active == "open_drawer"
    assert mem.state.pending == ["pick_place_mug"]
    assert mem.state.completed == []
    assert not mem.is_done


def test_mark_success_advances_and_records_completed():
    mem = TaskMemory("set the table", ["open_drawer", "pick_place_mug"])
    mem.mark_success()
    assert mem.state.completed == ["open_drawer"]
    assert mem.state.active == "pick_place_mug"
    assert mem.state.pending == []
    assert not mem.is_done


def test_mark_success_on_last_goal_sets_done():
    mem = TaskMemory("set the table", ["open_drawer"])
    mem.mark_success()
    assert mem.is_done
    assert mem.state.active is None
    assert mem.state.completed == ["open_drawer"]


def test_mark_success_with_no_active_goal_raises():
    mem = TaskMemory("set the table", ["open_drawer"])
    mem.mark_success()
    with pytest.raises(RuntimeError):
        mem.mark_success()


def test_mark_failure_with_no_active_goal_raises():
    mem = TaskMemory("set the table", ["open_drawer"])
    mem.mark_success()
    with pytest.raises(RuntimeError):
        mem.mark_failure("anything")


def test_skip_active_records_failure_not_completion_and_advances():
    mem = TaskMemory("set the table", ["open_drawer", "pick_place_fork"])
    mem.skip_active("drawer_not_open")
    assert mem.state.completed == []
    assert len(mem.state.failures) == 1
    assert mem.state.failures[0].goal == "open_drawer"
    assert mem.state.failures[0].reason == "drawer_not_open"
    assert mem.state.active == "pick_place_fork"


def test_step_index_increments_across_goals():
    mem = TaskMemory("t", ["a", "b", "c"])
    mem.mark_success()
    mem.skip_active("nope")
    assert mem.state.failures[0].step_index == 1
    assert mem.state.active == "c"


def test_failures_accumulate_across_multiple_goals():
    mem = TaskMemory("t", ["a", "b"])
    mem.mark_failure("first_try_failed")
    mem.skip_active("gave_up")
    assert len(mem.state.failures) == 2
    assert [f.reason for f in mem.state.failures] == ["first_try_failed", "gave_up"]


# --------------------------------------------------------------------------
# serialize_instruction
# --------------------------------------------------------------------------


def test_serialize_instruction_initial_state():
    mem = TaskMemory("set the table", ["open_drawer", "pick_place_mug"])
    s = serialize_instruction(mem, arm=None)
    assert s == "Task: set the table. Done: nothing yet. Now: open drawer."


def test_serialize_instruction_includes_arm_when_given():
    mem = TaskMemory("set the table", ["pick_place_mug"])
    s = serialize_instruction(mem, arm="right")
    assert s == "Task: set the table. Done: nothing yet. Now: pick place mug with right."


def test_serialize_instruction_lists_completed_goals_in_order():
    mem = TaskMemory("set the table", ["open_drawer", "pick_place_mug", "pick_place_plate"])
    mem.mark_success()
    mem.mark_success()
    s = serialize_instruction(mem, arm="left")
    assert s == "Task: set the table. Done: open drawer, pick place mug. Now: pick place plate with left."


def test_serialize_instruction_when_done():
    mem = TaskMemory("set the table", ["open_drawer"])
    mem.mark_success()
    s = serialize_instruction(mem)
    assert s == "Task: set the table. Done: open drawer. Now: done."


def test_serialize_instruction_underscore_goal_names_are_humanized():
    mem = TaskMemory("t", ["pick_place_spoon"])
    s = serialize_instruction(mem)
    assert "pick place spoon" in s
    assert "_" not in s
