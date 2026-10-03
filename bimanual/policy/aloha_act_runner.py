"""Processor-correct, rate-correct ACT execution in the ALOHA contact task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from bimanual.policy.lerobot_io import frame_to_batch_image as _frame_to_batch_image
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, CONTROL_HZ, AlohaPhysicalEnv, AlohaTableSettingEnv
from bimanual.skills.registry import Skill, get_skill

POLICY_HZ = ALOHA_BIMANUAL.policy_hz
CONTROL_STEPS_PER_ACTION = CONTROL_HZ // POLICY_HZ


@dataclass(frozen=True)
class AlohaRolloutResult:
    success: bool
    policy_steps: int
    control_steps: int
    initial_height: float
    peak_height: float
    retained_height: float
    task: str = "block_lift"
    final_xy_error: float | None = None


def _raw_observation(env: AlohaPhysicalEnv, device: str, instruction: str | None = None) -> dict[str, Any]:
    frames = env.render()
    observation = {
        "observation.images.global": _frame_to_batch_image(frames["overhead_cam"], device).squeeze(0),
        "observation.images.left_wrist": _frame_to_batch_image(frames["wrist_cam_left"], device).squeeze(0),
        "observation.images.right_wrist": _frame_to_batch_image(frames["wrist_cam_right"], device).squeeze(0),
        "observation.state": torch.from_numpy(env.state_vector()).to(device),
    }
    if instruction is not None:
        observation["task"] = instruction
    return observation


def run_aloha_act_episode(
    env: AlohaPhysicalEnv,
    policy: Any,
    *,
    preprocessor: Any | None = None,
    postprocessor: Any | None = None,
    device: str = "mps",
    max_policy_steps: int = 160,
    retain_steps: int = 5,
    task: str = "block_lift",
    skill: Skill | None = None,
    instruction: str | None = None,
    after_control_step: Callable[[], None] | None = None,
) -> AlohaRolloutResult:
    """Execute a policy at 10 Hz while stepping the ALOHA controller at 30 Hz."""
    if CONTROL_HZ % POLICY_HZ:
        raise ValueError("control frequency must be divisible by policy frequency")
    skill = skill or get_skill(task)
    if skill.name == "mug_pick_place" and not isinstance(env, AlohaTableSettingEnv):
        raise TypeError("mug_pick_place requires AlohaTableSettingEnv")
    if instruction is not None and preprocessor is None:
        raise ValueError("language-conditioned inference requires the saved preprocessor")
    policy.reset()
    state = skill.begin(env)
    retained = 0
    for policy_step in range(1, max_policy_steps + 1):
        raw = _raw_observation(env, device, instruction)
        batch = preprocessor(raw) if preprocessor is not None else {k: v.unsqueeze(0) for k, v in raw.items()}
        with torch.no_grad():
            action = policy.select_action(batch)
        if postprocessor is not None:
            action = postprocessor(action)
        action_np = np.asarray(action.squeeze(0).detach().cpu(), dtype=np.float64)
        ALOHA_BIMANUAL.validate(env.state_vector(), action_np)
        for _ in range(CONTROL_STEPS_PER_ACTION):
            env.step(action_np)
            skill.update(env, state)
            if after_control_step is not None:
                after_control_step()
        retained = retained + 1 if skill.succeeded(env, state) else 0
        xy_error = None if state.target is None else float(np.linalg.norm(state.position[:2] - state.target[:2]))
        if retained >= retain_steps:
            return AlohaRolloutResult(
                True, policy_step, policy_step * CONTROL_STEPS_PER_ACTION,
                float(state.initial[2]), state.peak_height, float(state.position[2]), skill.name,
                xy_error,
            )
    return AlohaRolloutResult(
        False, max_policy_steps, max_policy_steps * CONTROL_STEPS_PER_ACTION,
        float(state.initial[2]), state.peak_height, float(state.position[2]), skill.name,
        None if state.target is None else float(np.linalg.norm(state.position[:2] - state.target[:2])),
    )
