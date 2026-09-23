"""Data structures for task memory (Phase 5, decision A7, docs/decisions.md).

Kept as plain dataclasses (no behavior) so they're trivially
JSON-serializable for logging/replay. `TaskMemory` (task_memory.py) owns
all state transitions; nothing here mutates itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailureEvent:
    goal: str
    reason: str
    step_index: int


@dataclass
class TaskMemoryState:
    """Rewritten in place per goal as the episode progresses (A7: "world
    memory reduces to the verifier's predicate vector plus timestamps...
    rewrite fields, never append"). `failures` is the sole exception --
    a deliberate append-only history of what went wrong and when, not
    current belief."""

    original_instruction: str
    completed: list[str] = field(default_factory=list)
    active: str | None = None
    pending: list[str] = field(default_factory=list)
    failures: list[FailureEvent] = field(default_factory=list)
