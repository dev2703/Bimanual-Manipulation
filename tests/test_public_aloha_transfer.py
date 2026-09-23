from __future__ import annotations

import torch
from safetensors.torch import save_file

from scripts.infer_public_aloha_transfer import legacy_stats


def test_legacy_public_act_stats_are_read_from_checkpoint(tmp_path):
    tensors = {
        "normalize_inputs.buffer_observation_images_top.mean": torch.zeros(3, 1, 1),
        "normalize_inputs.buffer_observation_images_top.std": torch.ones(3, 1, 1),
        "normalize_inputs.buffer_observation_state.mean": torch.arange(14, dtype=torch.float32),
        "normalize_inputs.buffer_observation_state.std": torch.ones(14),
        "unnormalize_outputs.buffer_action.mean": torch.arange(14, dtype=torch.float32) + 1,
        "unnormalize_outputs.buffer_action.std": torch.ones(14) * 2,
    }
    save_file(tensors, tmp_path / "model.safetensors")
    stats = legacy_stats(tmp_path, "cpu")
    assert torch.equal(stats["state_mean"], tensors["normalize_inputs.buffer_observation_state.mean"])
    assert torch.equal(stats["action_mean"], tensors["unnormalize_outputs.buffer_action.mean"])
    assert torch.equal(stats["action_std"], tensors["unnormalize_outputs.buffer_action.std"])
