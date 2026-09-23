"""Per-frame instruction serialization (decision A7, docs/decisions.md).

`Task: <original>. Done: <completed goals>. Now: <active subgoal> with
<arm>.` The dataset generator and the inference-time executor must build
this exact same string format from the same TaskMemory state, or
train/inference distributions diverge (A7's stated failure mode).
"""

from __future__ import annotations

from bimanual.memory.task_memory import TaskMemory


def _humanize(goal: str) -> str:
    return goal.replace("_", " ")


def serialize_instruction(memory: TaskMemory, arm: str | None = None) -> str:
    completed = memory.state.completed
    done = ", ".join(_humanize(g) for g in completed) if completed else "nothing yet"

    if memory.is_done:
        now = "done"
    else:
        now = _humanize(memory.state.active)
        if arm is not None:
            now = f"{now} with {arm}"

    return f"Task: {memory.state.original_instruction}. Done: {done}. Now: {now}."
