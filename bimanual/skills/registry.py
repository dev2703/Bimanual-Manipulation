"""One declaration per ALOHA skill, shared by data generation and rollouts.

`gate_passed` records whether the scripted expert has cleared its held-out
gate. Handoff has not. Pour is registered only after its expert exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from bimanual.evaluation.aloha_predicates import mug_placed, plate_placed
from bimanual.experts.aloha_block import run_block_grasp_lift
from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_cutlery import run_fork_place, run_spoon_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_handoff import run_baton_handoff
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_pour import (
    outlet_aligned, pour_alignment, pour_pose_reached, run_pour_pose,
)
from bimanual.sim.aloha_env import AlohaPhysicalEnv, AlohaTableSettingEnv

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robots" / "aloha"


@dataclass
class RolloutState:
    initial: np.ndarray
    peak_height: float
    position: np.ndarray
    target: np.ndarray | None = None
    carried_to_target: bool = False
    initial_height: float = 0.0
    pour_dwell_steps: int = 0


@dataclass(frozen=True)
class Skill:
    name: str
    bucket: str
    scene_name: str
    instruction: str
    source_expert: str
    run: Callable
    arms: tuple[str, ...]
    object_key: str | None
    max_policy_steps: int = 240
    default_prefix: int = 20
    gate_passed: bool = False
    env_kind: str = "table"
    randomize: str = "objects"
    on_step: Callable | None = None
    check: Callable | None = None
    lift_threshold: float = 0.08

    def scene_path(self) -> Path:
        return ASSETS / self.scene_name

    def make_env(self):
        if self.env_kind == "table":
            return AlohaTableSettingEnv(self.scene_path())
        return AlohaPhysicalEnv(self.scene_path())

    def begin(self, env) -> RolloutState:
        if self.object_key is None:
            raise ValueError(f"{self.name} has no single object for a policy rollout")
        initial = env.oracle_state()[self.object_key].copy()
        target = None
        if self.env_kind == "table" and f"{self.object_key.split('_')[0]}_region" in [
            env.model.site(i).name for i in range(env.model.nsite)
        ]:
            site = f"{self.object_key.split('_')[0]}_region"
            target = env.data.site_xpos[env.model.site(site).id].copy()
        return RolloutState(initial, float(initial[2]), initial.copy(), target, initial_height=float(initial[2]))

    def update(self, env, state: RolloutState) -> None:
        if self.on_step is not None:
            self.on_step(env, state)
            return
        if self.object_key is None:
            return
        position = env.oracle_state()[self.object_key].copy()
        state.position = position
        state.peak_height = max(state.peak_height, float(position[2]))

    def succeeded(self, env, state: RolloutState) -> bool:
        if self.check is not None:
            return bool(self.check(env, state))
        return float(state.position[2]) >= state.initial_height + self.lift_threshold


def _pour_step(env, state: RolloutState) -> None:
    _track_height(env, state, "mug_pos")
    if outlet_aligned(pour_alignment(env)) and state.position[2] > .02:
        state.pour_dwell_steps += 1
    else:
        state.pour_dwell_steps = 0


def _pour_ok(env, state: RolloutState) -> bool:
    alignment = pour_alignment(env)
    joints = env.state_vector()
    return outlet_aligned(alignment) and pour_pose_reached(
        float(env.oracle_state()["mug_pos"][2]),
        float(env.oracle_state()["bottle_pos"][2]),
        alignment["bottle_upright_cosine"], alignment["mouth_xy_error"],
        state.pour_dwell_steps, float(joints[13]), float(joints[6]), table_supported=True,
    )


def _track_height(env, state: RolloutState, key: str) -> None:
    position = env.oracle_state()[key].copy()
    state.position = position
    state.peak_height = max(state.peak_height, float(position[2]))


def _mug_step(env, state: RolloutState) -> None:
    _track_height(env, state, "mug_pos")
    height = float(state.position[2])
    state.carried_to_target = state.carried_to_target or bool(
        height > state.initial_height + 0.08
        and np.linalg.norm(state.position[:2] - state.target[:2]) < 0.05
    )


def _mug_ok(env, state: RolloutState) -> bool:
    return mug_placed(
        state.initial, state.position, state.target,
        peak_height=state.peak_height,
        carried_to_target=state.carried_to_target,
        upright_cosine=env.mug_upright_cosine(),
        gripper_opening=float(env.state_vector()[13]),
    )


def _plate_step(env, state: RolloutState) -> None:
    _track_height(env, state, "plate_pos")
    height = float(state.position[2])
    state.carried_to_target = state.carried_to_target or bool(
        height > state.initial_height + 0.08
        and np.linalg.norm(state.position[:2] - state.target[:2]) < 0.05
    )


def _plate_ok(env, state: RolloutState) -> bool:
    return plate_placed(
        state.initial, state.position, state.target,
        peak_height=state.peak_height,
        carried_to_target=state.carried_to_target,
        upright_cosine=env.object_upright_cosine("plate"),
        gripper_opening=float(env.state_vector()[13]),
    )


def _run_handoff_either_direction(env, record_frames: bool = False):
    """Even gate seeds transfer left to right; odd seeds transfer right to left."""
    seed = int(getattr(env, "gate_seed", 0))
    carrier = "left" if seed % 2 == 0 else "right"
    return run_baton_handoff(env, record_frames=record_frames, carrier=carrier)


SKILLS: dict[str, Skill] = {
    "block_lift": Skill(
        "block_lift", "block", "task_block.xml",
        "Lift the block off the table.",
        "aloha_contact_block_v1", run_block_grasp_lift, ("left", "right"), "task_block_pos",
        max_policy_steps=160, gate_passed=True, env_kind="physical", randomize="block",
    ),
    "mug_pick_place": Skill(
        "mug_pick_place", "mug", "task_table_setting.xml",
        "Set the dinner table: place the blue mug to the right of the plate.",
        "aloha_contact_mug_v1", run_mug_pick_place, ("right",), "mug_pos",
        gate_passed=True, on_step=_mug_step, check=_mug_ok,
    ),
    "plate_pick_place": Skill(
        "plate_pick_place", "plate", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_v2", run_plate_pick_place, ("right",), "plate_pos",
        gate_passed=True, on_step=_plate_step, check=_plate_ok,
    ),
    "plate_recovery": Skill(
        "plate_recovery", "plate_recovery", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_moved_recovery_v1", run_plate_pick_place, ("right",), "plate_pos",
        gate_passed=True, on_step=_plate_step, check=_plate_ok,
    ),
    "drawer_open": Skill(
        "drawer_open", "drawer", "task_table_setting_drawer_v2.xml",
        "Open the top drawer of the dinner cabinet.",
        "aloha_contact_drawer_v2", run_drawer_open, ("right",), None,
        gate_passed=True, env_kind="table",
    ),
    "bottle_grasp_lift": Skill(
        "bottle_grasp_lift", "bottle", "task_table_setting.xml",
        "Lift the bottle from the dinner table.",
        "aloha_contact_bottle_v1", run_bottle_grasp_lift, ("left",), "bottle_pos",
        gate_passed=True,
        check=lambda env, state: float(state.position[2]) > state.initial_height + 0.10
        and float(env.state_vector()[6]) < 0.034,
    ),
    "fork_place": Skill(
        "fork_place", "fork", "task_table_setting_combined_v2.xml",
        "Place the fork beside the plate.",
        "aloha_contact_cutlery_v1", run_fork_place, ("left",), "fork_pos",
        gate_passed=True,
    ),
    "spoon_place": Skill(
        "spoon_place", "spoon", "task_table_setting_combined_v2.xml",
        "Place the spoon beside the plate.",
        "aloha_contact_cutlery_v1", run_spoon_place, ("right",), "spoon_pos",
        gate_passed=True,
    ),
    "pour_pose": Skill(
        "pour_pose", "pour", "task_table_setting.xml",
        "Place the glass upright on the table and hold the bottle outlet centered one inch above its rim.",
        "aloha_pour_pose_v3", run_pour_pose, ("left", "right"), "mug_pos",
        gate_passed=False,
        on_step=_pour_step, check=_pour_ok,
    ),
    "baton_handoff": Skill(
        "baton_handoff", "handoff", "task_handoff_baton.xml",
        "Hand the baton from the left gripper to the right gripper.",
        "aloha_contact_handoff_v1", _run_handoff_either_direction, ("left", "right"), "task_block_pos",
        gate_passed=True, env_kind="physical", randomize="block", default_prefix=5,
    ),
}


def get_skill(name: str) -> Skill:
    try:
        return SKILLS[name]
    except KeyError:
        known = ", ".join(sorted(SKILLS))
        raise KeyError(f"unknown skill {name!r}; known: {known}") from None
