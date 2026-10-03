"""One declaration per ALOHA skill, shared by gates, recording, replay and policy rollouts.

To add a skill: write an expert `run(env, record_frames=False) -> result with
.success`, add a `Skill` entry below with a state-based success check, then run
`scripts/gate.py --skill <name>`. `gate_passed` records whether the scripted
expert has cleared its held-out gate; see `outputs/gates/<name>.json`.

Success checks read simulator state only, so the same check scores an expert,
a replayed dataset and a learned policy. `Skill.update` runs every control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from bimanual.evaluation.aloha_contacts import body_touches, right_gripper_touches_drawer_handle
from bimanual.evaluation.aloha_predicates import (
    cutlery_placed, drawer_opened, handoff_succeeded, mug_placed, plate_placed,
)
from bimanual.evaluation.pour_geometry import pour_sample_valid, pour_succeeded
from bimanual.experts.aloha_block import run_block_grasp_lift
from bimanual.experts.aloha_bottle import run_bottle_grasp_lift
from bimanual.experts.aloha_cutlery import run_fork_place, run_spoon_place
from bimanual.experts.aloha_drawer import run_drawer_open
from bimanual.experts.aloha_handoff import run_baton_handoff
from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.experts.aloha_plate import run_plate_pick_place
from bimanual.experts.aloha_pour import run_pour_pose
from bimanual.sim.aloha_env import AlohaPhysicalEnv, AlohaTableSettingEnv

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robots" / "aloha"
GRIPPER_INDEX = {"left": 6, "right": 13}
RECOVERY_SHIFT = 0.035


@dataclass
class RolloutState:
    initial: np.ndarray
    peak_height: float
    position: np.ndarray
    target: np.ndarray | None = None
    carried_to_target: bool = False
    initial_height: float = 0.0
    pour_dwell_steps: int = 0
    contact_steps: int = 0
    initial_opening: float = 0.0
    peak_opening: float = 0.0


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
    instruction_fn: Callable[[int], str] | None = None
    recovery: bool = False
    setup: Callable | None = None  # unrecorded precondition run after reset

    def scene_path(self) -> Path:
        return ASSETS / self.scene_name

    def make_env(self):
        if self.env_kind == "table":
            return AlohaTableSettingEnv(self.scene_path())
        return AlohaPhysicalEnv(self.scene_path())

    def instruction_for(self, seed: int) -> str:
        """Language label for one episode; some skills vary with the seed."""
        return self.instruction if self.instruction_fn is None else self.instruction_fn(seed)

    def begin(self, env) -> RolloutState:
        oracle = env.oracle_state()
        initial = oracle[self.object_key].copy() if self.object_key else np.zeros(3)
        target = None
        if self.object_key and self.env_kind == "table":
            site = f"{self.object_key.split('_')[0]}_region"
            if site in {env.model.site(i).name for i in range(env.model.nsite)}:
                target = env.data.site_xpos[env.model.site(site).id].copy()
        state = RolloutState(initial, float(initial[2]), initial.copy(), target, initial_height=float(initial[2]))
        if "drawer_opening" in oracle:
            state.initial_opening = state.peak_opening = float(oracle["drawer_opening"][0])
        return state

    def update(self, env, state: RolloutState) -> None:
        if self.on_step is not None:
            self.on_step(env, state)
        elif self.object_key is not None:
            _track_height(env, state, self.object_key)

    def succeeded(self, env, state: RolloutState) -> bool:
        if self.check is not None:
            return bool(self.check(env, state))
        return float(state.position[2]) >= state.initial_height + self.lift_threshold


def _track_height(env, state: RolloutState, key: str) -> None:
    position = env.oracle_state()[key].copy()
    state.position = position
    state.peak_height = max(state.peak_height, float(position[2]))


def _carry_tracker(key: str):
    def step(env, state: RolloutState) -> None:
        _track_height(env, state, key)
        state.carried_to_target = state.carried_to_target or bool(
            float(state.position[2]) > state.initial_height + 0.08
            and np.linalg.norm(state.position[:2] - state.target[:2]) < 0.05
        )
    return step


def _mug_ok(env, state: RolloutState) -> bool:
    return mug_placed(
        state.initial, state.position, state.target,
        peak_height=state.peak_height, carried_to_target=state.carried_to_target,
        upright_cosine=env.mug_upright_cosine(),
        gripper_opening=float(env.state_vector()[GRIPPER_INDEX["right"]]),
    )


def _plate_ok(env, state: RolloutState) -> bool:
    return plate_placed(
        state.initial, state.position, state.target,
        peak_height=state.peak_height, carried_to_target=state.carried_to_target,
        upright_cosine=env.object_upright_cosine("plate"),
        gripper_opening=float(env.state_vector()[GRIPPER_INDEX["right"]]),
    )


def _drawer_step(env, state: RolloutState) -> None:
    state.peak_opening = max(state.peak_opening, float(env.oracle_state()["drawer_opening"][0]))
    state.contact_steps += int(right_gripper_touches_drawer_handle(env.model, env.data))


def _drawer_ok(env, state: RolloutState) -> bool:
    return drawer_opened(
        state.initial_opening, state.peak_opening, float(env.oracle_state()["drawer_opening"][0]),
        handle_contact_steps=state.contact_steps,
        gripper_opening=float(env.state_vector()[GRIPPER_INDEX["right"]]),
    )


def _cutlery_ok(arm: str):
    def check(env, state: RolloutState) -> bool:
        return cutlery_placed(state.initial, state.position, state.target, peak_height=state.peak_height,
                              gripper_opening=float(env.state_vector()[GRIPPER_INDEX[arm]]))
    return check


def _pour_step(env, state: RolloutState) -> None:
    _track_height(env, state, "mug_pos")
    state.pour_dwell_steps = state.pour_dwell_steps + 1 if pour_sample_valid(env) else 0


def _pour_ok(env, state: RolloutState) -> bool:
    return pour_succeeded(env, state.pour_dwell_steps)


def handoff_carrier(seed: int) -> str:
    """Even seeds hand left to right, odd seeds right to left."""
    return "left" if seed % 2 == 0 else "right"


def _handoff_arms(env) -> tuple[str, str]:
    carrier = handoff_carrier(int(getattr(env, "gate_seed", 0)))
    return carrier, "right" if carrier == "left" else "left"


def _handoff_instruction(seed: int) -> str:
    carrier = handoff_carrier(seed)
    receiver = "right" if carrier == "left" else "left"
    return f"Hand the baton from the {carrier} gripper to the {receiver} gripper."


def _run_handoff(env, record_frames: bool = False):
    return run_baton_handoff(env, record_frames=record_frames, carrier=_handoff_arms(env)[0])


def _handoff_step(env, state: RolloutState) -> None:
    _track_height(env, state, "task_block_pos")
    receiver = _handoff_arms(env)[1]
    state.contact_steps += int(body_touches(env.model, env.data, "task_block", f"{receiver}/"))


def _handoff_ok(env, state: RolloutState) -> bool:
    carrier, receiver = _handoff_arms(env)
    joints = env.state_vector()
    height = float(state.position[2])
    return handoff_succeeded(
        state.initial_height, state.peak_height, height, height,
        float(joints[GRIPPER_INDEX[carrier]]), float(joints[GRIPPER_INDEX[receiver]]), state.contact_steps,
    )


def _open_drawer_first(env) -> None:
    """Cutlery starts inside the drawer, so its skills begin from an opened drawer."""
    if not run_drawer_open(env).success:
        raise RuntimeError("setup failed: drawer did not open")


def recovery_shift(seed: int) -> np.ndarray:
    """Exogenous plate displacement applied after the approach, fixed by the seed."""
    angle = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    return RECOVERY_SHIFT * np.array([np.cos(angle), np.sin(angle)])


def _run_plate_recovery(env, record_frames: bool = False):
    return run_plate_pick_place(env, record_frames=record_frames,
                                perturbation_xy=recovery_shift(int(getattr(env, "gate_seed", 0))),
                                replan_after_perturb=True)


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
        gate_passed=True, on_step=_carry_tracker("mug_pos"), check=_mug_ok,
    ),
    "plate_pick_place": Skill(
        "plate_pick_place", "plate", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_v2", run_plate_pick_place, ("right",), "plate_pos",
        gate_passed=True, on_step=_carry_tracker("plate_pos"), check=_plate_ok,
    ),
    "plate_recovery": Skill(
        "plate_recovery", "plate_recovery", "task_table_setting_plate_v2.xml",
        "Set the dinner table: place the serving plate in the centre.",
        "aloha_contact_plate_moved_recovery_v1", _run_plate_recovery, ("right",), "plate_pos",
        gate_passed=True, on_step=_carry_tracker("plate_pos"), check=_plate_ok, recovery=True,
    ),
    "drawer_open": Skill(
        "drawer_open", "drawer", "task_table_setting_drawer_v2.xml",
        "Open the top drawer of the dinner cabinet.",
        "aloha_contact_drawer_v2", run_drawer_open, ("right",), None,
        gate_passed=True, on_step=_drawer_step, check=_drawer_ok,
    ),
    "bottle_grasp_lift": Skill(
        "bottle_grasp_lift", "bottle", "task_table_setting.xml",
        "Lift the bottle from the dinner table.",
        "aloha_contact_bottle_v1", run_bottle_grasp_lift, ("left",), "bottle_pos",
        gate_passed=True,
        check=lambda env, state: float(state.position[2]) > state.initial_height + 0.10
        and float(env.state_vector()[GRIPPER_INDEX["left"]]) < 0.034,
    ),
    "fork_place": Skill(
        "fork_place", "fork", "task_table_setting_combined_v2.xml",
        "Place the fork beside the plate.",
        "aloha_contact_cutlery_v1", run_fork_place, ("left",), "fork_pos",
        gate_passed=True, check=_cutlery_ok("left"), setup=_open_drawer_first,
    ),
    "spoon_place": Skill(
        "spoon_place", "spoon", "task_table_setting_combined_v2.xml",
        "Place the spoon beside the plate.",
        "aloha_contact_cutlery_v1", run_spoon_place, ("right",), "spoon_pos",
        gate_passed=True, check=_cutlery_ok("right"), setup=_open_drawer_first,
    ),
    "pour_pose": Skill(
        "pour_pose", "pour", "task_table_setting.xml",
        "Place the glass upright and aim the pour stream inside its opening.",
        "aloha_pour_stream_pose_v4", run_pour_pose, ("left", "right"), "mug_pos",
        gate_passed=True, on_step=_pour_step, check=_pour_ok,
    ),
    "baton_handoff": Skill(
        "baton_handoff", "handoff", "task_handoff_baton.xml",
        "Hand the baton from one gripper to the other.",
        "aloha_contact_handoff_v1", _run_handoff, ("left", "right"), "task_block_pos",
        gate_passed=True, env_kind="physical", randomize="block", default_prefix=5,
        on_step=_handoff_step, check=_handoff_ok, instruction_fn=_handoff_instruction,
    ),
}


def get_skill(name: str) -> Skill:
    try:
        return SKILLS[name]
    except KeyError:
        known = ", ".join(sorted(SKILLS))
        raise KeyError(f"unknown skill {name!r}; known: {known}") from None
