"""Processor-correct, rate-correct ACT execution in the ALOHA contact task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from bimanual.policy.act_runner import _frame_to_batch_image
from bimanual.evaluation.aloha_predicates import mug_placed
from bimanual.sim.aloha_env import ALOHA_BIMANUAL, CONTROL_HZ, AlohaPhysicalEnv, AlohaTableSettingEnv

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
    lift_threshold: float = 0.08,
    retain_steps: int = 5,
    task: str = "block_lift",
    instruction: str | None = None,
    after_control_step: Callable[[], None] | None = None,
) -> AlohaRolloutResult:
    """Execute ACT at 10 Hz while stepping the ALOHA controller at 30 Hz."""
    if CONTROL_HZ % POLICY_HZ:
        raise ValueError("control frequency must be divisible by policy frequency")
    if task not in {"block_lift", "mug_pick_place"}:
        raise ValueError(f"unsupported ALOHA task {task!r}")
    if task == "mug_pick_place" and not isinstance(env, AlohaTableSettingEnv):
        raise TypeError("mug_pick_place requires AlohaTableSettingEnv")
    if instruction is not None and preprocessor is None:
        raise ValueError("language-conditioned inference requires the saved preprocessor")
    policy.reset()
    object_key = "mug_pos" if task == "mug_pick_place" else "task_block_pos"
    initial_position = env.oracle_state()[object_key].copy()
    initial_height = float(initial_position[2])
    target = env.data.site_xpos[env.model.site("mug_region").id].copy() if task == "mug_pick_place" else None
    peak_height = initial_height
    retained = 0
    carried_to_target = False
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
            if after_control_step is not None:
                after_control_step()
        position = env.oracle_state()[object_key]
        height = float(position[2])
        peak_height = max(peak_height, height)
        if task == "mug_pick_place":
            assert target is not None and isinstance(env, AlohaTableSettingEnv)
            carried_to_target |= bool(
                height > initial_height + 0.08
                and np.linalg.norm(position[:2] - target[:2]) < 0.05
            )
            successful_now = mug_placed(
                initial_position, position, target,
                peak_height=peak_height,
                carried_to_target=carried_to_target,
                upright_cosine=env.mug_upright_cosine(),
                gripper_opening=float(env.state_vector()[13]),
            )
        else:
            successful_now = height >= initial_height + lift_threshold
        retained = retained + 1 if successful_now else 0
        if retained >= retain_steps:
            return AlohaRolloutResult(
                True, policy_step, policy_step * CONTROL_STEPS_PER_ACTION,
                initial_height, peak_height, height, task,
                float(np.linalg.norm(position[:2] - target[:2])) if target is not None else None,
            )
    position = env.oracle_state()[object_key]
    height = float(position[2])
    return AlohaRolloutResult(
        False, max_policy_steps, max_policy_steps * CONTROL_STEPS_PER_ACTION,
        initial_height, peak_height, height, task,
        float(np.linalg.norm(position[:2] - target[:2])) if target is not None else None,
    )
