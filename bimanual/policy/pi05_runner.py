"""Pi0.5 checkpoint loading and closed-loop ALOHA dinner inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bimanual.policy.aloha_act_runner import AlohaRolloutResult, run_aloha_act_episode
from bimanual.sim.aloha_env import AlohaTableSettingEnv

MUG_INSTRUCTION = "Set the dinner table: place the blue mug to the right of the plate."


def load_pi05_bundle(checkpoint: str | Path, device: str = "cuda") -> tuple[Any, Any, Any]:
    """Load Pi0.5 weights with the exact saved normalization processors."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    checkpoint = Path(checkpoint)
    policy = PI05Policy.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    overrides = {"device_processor": {"device": device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides=overrides,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides=overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def run_pi05_mug_episode(
    env: AlohaTableSettingEnv,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    *,
    instruction: str = MUG_INSTRUCTION,
    device: str = "cuda",
    max_policy_steps: int = 240,
) -> AlohaRolloutResult:
    return run_aloha_act_episode(
        env,
        policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        max_policy_steps=max_policy_steps,
        task="mug_pick_place",
        instruction=instruction,
    )
