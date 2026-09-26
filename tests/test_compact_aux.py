from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from bimanual.policy.compact_aux import (
    PHASES, AuxWeights, aux_metrics, aux_targets, auxiliary_loss,
)
from bimanual.policy.compact_vla import ByteTokenizer, CompactVLA, CompactVLAConfig


def _batch():
    positions = torch.zeros(3, 15)
    positions[0, 3:5] = torch.tensor([0.30, 0.12])   # mug
    positions[1, 0:2] = torch.tensor([-0.5, -0.25])  # plate at table corner
    positions[2, 6:8] = torch.tensor([9.0, 9.0])     # bottle off table
    return {
        "privileged.phase": ["APPROACH", "TRANSPORT", "PULL"],
        "privileged.arm": ["right", "right", "left"],
        "privileged.object_positions": positions,
        "frame_index": torch.tensor([0, 50, 99]),
        "episode_index": torch.tensor([7, 7, 3]),
    }


def test_targets_follow_privileged_labels():
    targets = aux_targets(_batch(), {7: 101, 3: 100}, "cpu")
    assert targets["phase"].tolist() == [PHASES.index("APPROACH"), PHASES.index("TRANSPORT"), PHASES.index("PULL")]
    assert torch.allclose(targets["progress"].squeeze(-1), torch.tensor([0.0, 0.5, 1.0]))
    assert targets["holding"].tolist() == [[0, 0], [0, 1], [1, 0]]
    assert torch.allclose(targets["grounding"][0, 1], torch.tensor([0.8, 0.37 / 0.75]), atol=1e-6)
    assert targets["grounding"][1, 0].tolist() == [0.0, 0.0]
    assert targets["grounding"][2, 2].tolist() == [1.0, 1.0]


def test_unknown_phase_and_missing_columns_fail_loudly():
    batch = _batch()
    batch["privileged.phase"] = ["APPROACH", "WIGGLE", "PULL"]
    with pytest.raises(ValueError, match="WIGGLE"):
        aux_targets(batch, {7: 101, 3: 100}, "cpu")
    batch = _batch()
    del batch["privileged.arm"]
    with pytest.raises(KeyError):
        aux_targets(batch, {7: 101, 3: 100}, "cpu")


def _model_and_inputs():
    torch.manual_seed(0)
    config = replace(CompactVLAConfig.smoke(14, 14), num_phases=len(PHASES))
    model = CompactVLA(config)
    images = torch.rand(3, 1, 3, 3, 32, 32)
    state = torch.rand(3, 14)
    language = ByteTokenizer(96)(["place the mug"] * 3)
    actions = torch.rand(3, 20, 14)
    return model, images, state, language, actions


def test_shared_encoding_flow_loss_matches_action_only_loss():
    model, images, state, language, actions = _model_and_inputs()
    torch.manual_seed(5)
    baseline = model.flow_matching_loss(images, state, language, actions)
    torch.manual_seed(5)
    shared, _ = model.flow_and_heads(images, state, language, actions)
    assert torch.allclose(baseline, shared)


def test_aux_loss_trains_heads_and_zero_weight_is_inert():
    model, images, state, language, actions = _model_and_inputs()
    targets = aux_targets(_batch(), {7: 101, 3: 100}, "cpu")
    _, heads = model.flow_and_heads(images, state, language, actions)
    total, parts = auxiliary_loss(heads, targets, AuxWeights())
    assert set(parts) == {"phase", "progress", "holding", "grounding"}
    total.backward()
    assert model.phase_head.weight.grad is not None
    assert model.grounding_head[0].weight.grad is not None

    zero, _ = auxiliary_loss(heads, targets, AuxWeights().scaled(0.0))
    assert float(zero.detach()) == 0.0
    metrics = aux_metrics(heads, targets)
    assert 0.0 <= metrics["phase_accuracy"] <= 1.0
    assert metrics["grounding_mae_m"] >= 0.0
