"""Shared LeRobot image conversion and ACT checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def frame_to_batch_image(frame: np.ndarray, device: str) -> torch.Tensor:
    """uint8 (H, W, 3) -> float32 (1, 3, H, W) in [0, 1]."""
    image = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    return image.unsqueeze(0).to(device)


def load_act_bundle(checkpoint: str | Path, device: str = "mps") -> tuple[Any, Any, Any]:
    """Load ACT and the exact normalization pipelines saved with it."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    checkpoint = Path(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
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
