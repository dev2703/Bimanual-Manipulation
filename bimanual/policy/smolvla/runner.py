"""SmolVLA checkpoint loading and closed-loop ALOHA dinner inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bimanual.policy.aloha_act_runner import AlohaRolloutResult, run_aloha_act_episode
from bimanual.sim.aloha_env import AlohaTableSettingEnv

MUG_INSTRUCTION = "Set the dinner table: place the blue mug to the right of the plate."


def load_smolvla_bundle(checkpoint: str | Path, device: str = "mps") -> tuple[Any, Any, Any]:
    """Load weights and their saved tokenization/normalization processors."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    checkpoint = Path(checkpoint)
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    overrides = {"device_processor": {"device": device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json", local_files_only=True,
        overrides=overrides,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_postprocessor.json", local_files_only=True,
        overrides=overrides, to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    if policy.config.chunk_size != 20 or policy.config.n_action_steps != 8:
        raise ValueError("dinner SmolVLA checkpoint must declare chunk 20 / prefix 8")
    return policy, preprocessor, postprocessor


def run_smolvla_mug_episode(
    env: AlohaTableSettingEnv,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    *,
    instruction: str = MUG_INSTRUCTION,
    device: str = "mps",
    max_policy_steps: int = 240,
) -> AlohaRolloutResult:
    """Same stepping, stopping, and physical metric as the ACT mug control."""
    return run_aloha_act_episode(
        env, policy, preprocessor=preprocessor, postprocessor=postprocessor,
        device=device, max_policy_steps=max_policy_steps, task="mug_pick_place",
        instruction=instruction,
    )
