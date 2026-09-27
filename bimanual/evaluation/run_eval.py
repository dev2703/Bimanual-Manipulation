"""Composed evaluation records for a dinner episode.

Writes one JSON file with per-stage success, the stage curve, and the memory
instruction that was active at each stage. A scripted expert or a policy
callable can fill `execute`.
"""

from __future__ import annotations

import json
from pathlib import Path

from bimanual.memory.serialization import serialize_instruction
from bimanual.memory.task_memory import TaskMemory


def run_episode(env, stages: list[tuple[str, object]], instruction: str, seed: int) -> dict:
    """Run named callables `(env) -> result` and record a stage curve."""
    memory = TaskMemory(instruction, [name for name, _execute in stages])
    rows = []
    for name, execute in stages:
        text = serialize_instruction(memory, arm=None)
        try:
            result = execute(env)
            success = bool(result.success)
            error = None
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        if success:
            memory.mark_success()
        else:
            memory.mark_failure("stage failed")
        rows.append({
            "stage": name,
            "success": success,
            "instruction": text,
            "error": error,
        })
        if not success:
            break
    return {
        "seed": seed,
        "instruction": instruction,
        "stages": rows,
        "success": bool(rows) and all(row["success"] for row in rows) and len(rows) == len(stages),
        "completed": [row["stage"] for row in rows if row["success"]],
    }


def write_report(path: Path, episodes: list[dict], threshold: float) -> dict:
    successes = sum(1 for episode in episodes if episode["success"])
    curve: dict[str, int] = {}
    for episode in episodes:
        for row in episode["stages"]:
            if row["success"]:
                curve[row["stage"]] = curve.get(row["stage"], 0) + 1
    report = {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "threshold": threshold,
        "passed": bool(episodes) and successes / len(episodes) >= threshold,
        "stage_successes": curve,
        "episodes_detail": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
