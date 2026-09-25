"""Compact-160M checkpoint loading and closed-loop ALOHA mug inference.

Reuses the same stepping/success logic as ACT and SmolVLA
(bimanual.policy.aloha_act_runner) via a thin policy/preprocessor adapter,
so closed-loop numbers are directly comparable across all three policies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from bimanual.policy.aloha_act_runner import AlohaRolloutResult, run_aloha_act_episode
from bimanual.policy.compact_vla import ByteTokenizer, CompactVLA, CompactVLAConfig
from bimanual.sim.aloha_env import AlohaTableSettingEnv

MUG_INSTRUCTION = "Set the dinner table: place the blue mug to the right of the plate."
CAMERA_KEYS = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")


class _CompactPreprocessor:
    def __init__(self, tokenizer: ByteTokenizer, device: str) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, raw: dict) -> dict:
        images = torch.stack([raw[key] for key in CAMERA_KEYS], dim=0).unsqueeze(0).unsqueeze(0).to(self.device)
        state = raw["observation.state"].unsqueeze(0).to(self.device)
        language = self.tokenizer([raw.get("task", "")], device=self.device)
        return {"images": images, "state": state, "language": language}


class _CompactPolicyAdapter:
    def __init__(self, model: CompactVLA, device: str) -> None:
        self.model = model
        self.device = device

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        output = self.model.sample_actions(batch["images"], batch["state"], batch["language"])
        return output.actions[:, 0]


def load_compact_bundle(checkpoint: str | Path, device: str = "mps") -> tuple[Any, Any, None]:
    """Load weights and build a policy/preprocessor pair matching load_*_bundle elsewhere."""
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "config.json").read_text())
    model = CompactVLA(CompactVLAConfig(**metadata["config"])).to(device).eval()
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device, weights_only=True))
    tokenizer = ByteTokenizer(model.config.max_language_tokens)
    return _CompactPolicyAdapter(model, device), _CompactPreprocessor(tokenizer, device), None


def run_compact_mug_episode(
    env: AlohaTableSettingEnv,
    policy: Any,
    preprocessor: Any,
    *,
    instruction: str = MUG_INSTRUCTION,
    device: str = "mps",
    max_policy_steps: int = 240,
) -> AlohaRolloutResult:
    return run_aloha_act_episode(
        env, policy, preprocessor=preprocessor, postprocessor=None,
        device=device, max_policy_steps=max_policy_steps, task="mug_pick_place",
        instruction=instruction,
    )
