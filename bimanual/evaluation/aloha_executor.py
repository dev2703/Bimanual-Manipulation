"""Verifier-driven goal transitions for the physical ALOHA dinner task.

The caller supplies an expert or learned skill executor. A scripted expert may
use simulator state for planning; this loop never uses it to decide whether a
goal has completed. A separately trained RGB verifier supplies that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from bimanual.memory.serialization import serialize_instruction
from bimanual.memory.task_memory import TaskMemory


class RGBVerifier(Protocol):
    def __call__(self, frames: Mapping[str, np.ndarray]) -> Mapping[str, bool]: ...


@dataclass(frozen=True)
class GoalAttempt:
    goal: str
    instruction: str
    attempt: int
    verified: bool
    error: str | None = None


def run_verified_goals(
    env,
    goals: list[str],
    execute_skill: Callable[[object, str, str], None],
    verify_rgb: RGBVerifier,
    *,
    instruction: str = "Set the dinner table",
    max_attempts: int = 2,
) -> tuple[TaskMemory, list[GoalAttempt]]:
    """Execute and verify each goal, retrying from fresh observations.

    `execute_skill` receives the current serialized instruction so learned
    policies see exactly the memory state used by the controller. The verifier
    receives rendered RGB only. A missing predicate is an interface error,
    never silently treated as success or failure.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    memory = TaskMemory(instruction, goals)
    attempts: list[GoalAttempt] = []
    while not memory.is_done:
        goal = memory.state.active
        assert goal is not None
        for attempt in range(1, max_attempts + 1):
            prompt = serialize_instruction(memory)
            error = None
            try:
                execute_skill(env, goal, prompt)
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            readings = verify_rgb(env.render())
            if goal not in readings:
                raise KeyError(f"RGB verifier did not predict {goal!r}")
            verified = bool(readings[goal])
            attempts.append(GoalAttempt(goal, prompt, attempt, verified, error))
            if verified:
                memory.mark_success()
                break
            if attempt < max_attempts:
                memory.mark_failure("rgb_verification_failed_retry")
            else:
                memory.skip_active("rgb_verification_failed")
    return memory, attempts
