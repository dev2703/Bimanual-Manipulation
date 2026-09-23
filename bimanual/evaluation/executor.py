"""Phase 5 executor: observe -> serialize memory -> execute -> verify ->
update loop (docs/phases.md Phase 5, docs/decisions.md A7).

Uses the scripted expert (bimanual/experts/skills.py) as the "policy" per
the Phase 5 gate, which deliberately isolates verifier quality from
policy quality: subgoal success is judged by `verify_fn`, not by
directly trusting execution. At inference, `verify_fn` must be backed by
perception/verifier.py's RGB predicate model (A3) -- `oracle_verify_fn`
below reads bimanual.sim.oracle_predicates directly and exists ONLY for
testing this loop's control flow; using it to report the Phase 5 gate
metric would be cheating (verifying with privileged state defeats the
point of having a verifier at all).

This loop does not replan a failed motion (that is
experts/recovery.py's job) -- on verifier failure it records the failure
in TaskMemory and moves on, matching table_setting.py's existing
"drawer_not_open" skip behavior for dependent goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bimanual.control.ik import BimanualIK
from bimanual.experts.executor import execute, execute_open_drawer
from bimanual.experts.skills import open_drawer, pick, place, place_target
from bimanual.experts.table_setting import TASK_GRAPH, assign_arm
from bimanual.logging_utils import get_logger
from bimanual.memory.serialization import serialize_instruction
from bimanual.memory.task_memory import TaskMemory
from bimanual.sim.env import BimanualTableEnv
from bimanual.sim.oracle_predicates import drawer_open, in_region
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, REGIONS

log = get_logger(__name__)

VerifyFn = Callable[[BimanualTableEnv, str], bool]
# VerifyFn(env, goal_name) -> whether that goal now reads as satisfied.

_REGION_BY_OBJ = {obj: region for obj, region, _requires_drawer in TASK_GRAPH}
_REQUIRES_DRAWER_BY_OBJ = {obj: requires_drawer for obj, _region, requires_drawer in TASK_GRAPH}


def _goal_list() -> list[str]:
    # open_drawer stays in the goal list unconditionally: cutlery is
    # pre-placed now (decision A5 descope) so no pick-place goal depends
    # on it, but opening the drawer is still part of the task and is
    # still measured as its own skill.
    goals = ["open_drawer"]
    goals.extend(f"pick_place_{obj}" for obj, _region, _requires_drawer in TASK_GRAPH)
    return goals


def oracle_verify_fn(env: BimanualTableEnv, goal: str) -> bool:
    """Testing-only verify_fn backed by privileged simulator state. See
    module docstring: never use this to report the Phase 5 gate metric."""
    if goal == "open_drawer":
        return drawer_open(env.oracle_state())
    obj = goal.removeprefix("pick_place_")
    return in_region(env.oracle_state(), obj, _REGION_BY_OBJ[obj])


@dataclass
class ExecutorStepLog:
    goal: str
    instruction: str
    verified_success: bool


def run_executor(
    env: BimanualTableEnv,
    ik: BimanualIK,
    verify_fn: VerifyFn,
    instruction: str = "set the table",
) -> list[ExecutorStepLog]:
    memory = TaskMemory(instruction, _goal_list())
    logs: list[ExecutorStepLog] = []
    drawer_opened_ok = False

    while not memory.is_done:
        goal = memory.state.active
        arm: str | None = None
        ok = False

        if goal == "open_drawer":
            arm = "left"
            ee_pos = env.proprio()[f"{arm}_ee_pos"].copy()
            handle_pos = env.data.geom_xpos[env.model.geom("drawer_handle").id].copy()
            segs = open_drawer(ik, env.data.qpos, arm, ee_pos, DRAWER_OPEN_DIST, handle_pos)
            execute_open_drawer(env, arm, segs, DRAWER_OPEN_DIST)
            for _ in range(10):
                env.step(env.data.ctrl)
            ok = verify_fn(env, goal)
            drawer_opened_ok = ok
        else:
            obj = goal.removeprefix("pick_place_")
            if _REQUIRES_DRAWER_BY_OBJ[obj] and not drawer_opened_ok:
                logs.append(ExecutorStepLog(goal, serialize_instruction(memory, arm), False))
                memory.skip_active("drawer_not_open")
                continue

            obj_pos = env.oracle_state()[f"{obj}_pos"].copy()
            arm = assign_arm(obj_pos[0])
            ee_pos = env.proprio()[f"{arm}_ee_pos"].copy()
            try:
                pick_segs = pick(ik, env.data.qpos, arm, ee_pos, obj_pos, obj)
                execute(env, arm, pick_segs, obj_name=obj)

                rx, ry, _radius = REGIONS[_REGION_BY_OBJ[obj]]
                target = place_target(obj, (rx, ry))
                ee2 = env.proprio()[f"{arm}_ee_pos"].copy()
                place_segs = place(ik, env.data.qpos, arm, ee2, target, obj)
                execute(env, arm, place_segs, obj_name=obj)

                for _ in range(10):
                    env.step(env.data.ctrl)
                ok = verify_fn(env, goal)
            except Exception:
                log.exception("%s planning/execution failed", goal)
                ok = False

        instruction_str = serialize_instruction(memory, arm=arm)
        logs.append(ExecutorStepLog(goal, instruction_str, ok))
        if ok:
            memory.mark_success()
        else:
            memory.skip_active("verifier_reported_failure")

    return logs
