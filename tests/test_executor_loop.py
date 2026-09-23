"""Tests for bimanual/evaluation/executor.py (Phase 5): the observe ->
serialize -> execute -> verify -> update loop, run against a live env
with the oracle-backed verify_fn (test-only per the module's own
docstring -- the real Phase 5 gate needs the RGB verifier).
"""

from __future__ import annotations

import numpy as np
import pytest

from bimanual.control.ik import BimanualIK
from bimanual.evaluation.executor import oracle_verify_fn, run_executor
from bimanual.sim.env import BimanualTableEnv, ResetOptions
from bimanual.sim.randomization import sample_scene_config


def _fresh_env_and_ik(seed: int = 0):
    cfg = sample_scene_config("L1", seed=seed)
    env = BimanualTableEnv(ResetOptions(scene_cfg=cfg))
    ik = BimanualIK(env.model)
    return env, ik


def test_run_executor_covers_every_goal_exactly_once():
    env, ik = _fresh_env_and_ik(seed=0)
    logs = run_executor(env, ik, oracle_verify_fn)

    goals = [entry.goal for entry in logs]
    assert goals[0] == "open_drawer"
    assert set(goals[1:]) == {"pick_place_mug", "pick_place_bottle", "pick_place_plate"}
    assert len(goals) == len(set(goals))  # each goal appears exactly once


def test_run_executor_instruction_strings_track_progress():
    env, ik = _fresh_env_and_ik(seed=0)
    logs = run_executor(env, ik, oracle_verify_fn, instruction="set the table")

    for entry in logs:
        assert entry.instruction.startswith("Task: set the table. Done: ")
        assert f"Now: {entry.goal.replace('_', ' ')}" in entry.instruction or "Now: done" in entry.instruction

    # Instructions should show strictly non-decreasing "Done" progress:
    # once a goal is mentioned as done, it stays mentioned.
    done_lists = []
    for entry in logs:
        done_part = entry.instruction.split("Done: ")[1].split(". Now")[0]
        done_lists.append(done_part)
    for earlier, later in zip(done_lists, done_lists[1:]):
        if earlier != "nothing yet":
            assert earlier.split(", ")[0] in later or earlier == later


def test_run_executor_continues_past_a_failed_drawer(env_and_ik=None):
    """With cutlery pre-placed (A5 descope) nothing depends on the
    drawer, so a drawer whose verifier reports failure must not stop the
    remaining goals from being attempted."""
    env, ik = _fresh_env_and_ik(seed=0)

    def fail_drawer_only(env, goal):
        return False if goal == "open_drawer" else oracle_verify_fn(env, goal)

    logs = run_executor(env, ik, fail_drawer_only)
    by_goal = {entry.goal: entry for entry in logs}
    assert by_goal["open_drawer"].verified_success is False
    assert {"pick_place_mug", "pick_place_bottle", "pick_place_plate"} <= set(by_goal)


def test_run_executor_always_success_verifier_marks_all_goals_completed():
    env, ik = _fresh_env_and_ik(seed=0)
    logs = run_executor(env, ik, lambda env, goal: True)
    assert all(entry.verified_success for entry in logs)


def test_run_executor_always_failure_verifier_skips_everything_without_crashing():
    env, ik = _fresh_env_and_ik(seed=0)
    logs = run_executor(env, ik, lambda env, goal: False)
    assert not any(entry.verified_success for entry in logs)
    assert len(logs) == 4  # open_drawer + 3 manipulated objects (cutlery is pre-placed, A5)


def test_oracle_verify_fn_matches_in_region_predicate_directly():
    env, ik = _fresh_env_and_ik(seed=0)
    # Before anything happens, the drawer should read as closed.
    assert oracle_verify_fn(env, "open_drawer") == False  # noqa: E712 (np.bool_, not a Python bool)
