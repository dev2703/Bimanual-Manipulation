"""TaskMemory: the mutable controller over a TaskMemoryState (decision A7).

One goal is "active" at a time; completing or giving up on it advances to
the next pending goal. This is deliberately simpler than
experts/recovery.py's perturb-and-replan mechanism -- TaskMemory tracks
*what the executor believes has happened so far*, not how to recover a
failed motion. The executor (evaluation/executor.py) is what decides
whether a failure means "skip and move on" or "retry."
"""

from __future__ import annotations

from bimanual.memory.schema import FailureEvent, TaskMemoryState


class TaskMemory:
    def __init__(self, instruction: str, goals: list[str]) -> None:
        if not goals:
            raise ValueError("goals must be non-empty")
        self.state = TaskMemoryState(
            original_instruction=instruction,
            active=goals[0],
            pending=list(goals[1:]),
        )
        self._step_index = 0

    @property
    def is_done(self) -> bool:
        return self.state.active is None

    def mark_success(self) -> None:
        if self.state.active is None:
            raise RuntimeError("no active goal to complete")
        self.state.completed.append(self.state.active)
        self._advance()

    def mark_failure(self, reason: str) -> None:
        if self.state.active is None:
            raise RuntimeError("no active goal to fail")
        self.state.failures.append(
            FailureEvent(goal=self.state.active, reason=reason, step_index=self._step_index)
        )

    def skip_active(self, reason: str) -> None:
        """Gives up on the active goal (recorded as a failure, NOT marked
        complete) and advances -- e.g. cutlery goals when the drawer never
        opened, matching experts/table_setting.py's existing
        'drawer_not_open' skip behavior."""
        self.mark_failure(reason)
        self._advance()

    def _advance(self) -> None:
        self._step_index += 1
        self.state.active = self.state.pending.pop(0) if self.state.pending else None
