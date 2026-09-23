from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bimanual.policy.compact_vla import ByteTokenizer, CompactVLA, CompactVLAConfig


def _inputs(batch: int = 2):
    images = torch.rand(batch, 2, 3, 3, 64, 64)
    state = torch.rand(batch, 12)
    language = ByteTokenizer(96)(["place the plate", "move the mug"][:batch])
    return images, state, language


def test_compact_vla_output_contract_and_parameter_report():
    model = CompactVLA(CompactVLAConfig.smoke())
    output = model(*_inputs())
    assert output.actions.shape == (2, 20, 12)
    assert output.phase_logits.shape == (2, 8)
    assert output.progress.shape == (2, 1)
    assert output.grounding.shape == (2, 4, 2)
    assert output.holding_logits.shape == (2, 2)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["total"] > 0


def test_research_preset_is_in_declared_compact_parameter_band():
    counts = CompactVLA(CompactVLAConfig()).parameter_counts()
    assert 100_000_000 <= counts["trainable"] <= 160_000_000


def test_flow_matching_loss_backpropagates():
    model = CompactVLA(CompactVLAConfig.smoke())
    images, state, language = _inputs()
    target = torch.rand(2, 20, 12)
    loss = model.flow_matching_loss(images, state, language, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.flow_output[-1].weight.grad is not None


def test_flow_matching_ignores_padded_actions():
    torch.manual_seed(4)
    model = CompactVLA(CompactVLAConfig.smoke())
    images, state, language = _inputs()
    target = torch.rand(2, 20, 12)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[:, 10:] = True
    torch.manual_seed(9)
    first = model.flow_matching_loss(images, state, language, target, action_is_pad=pad)
    changed = target.clone()
    changed[:, 10:] = 1000
    torch.manual_seed(9)
    second = model.flow_matching_loss(images, state, language, changed, action_is_pad=pad)
    assert torch.allclose(first, second)


def test_visual_and_language_inputs_change_shared_policy_output():
    torch.manual_seed(0)
    model = CompactVLA(CompactVLAConfig.smoke()).eval()
    images, state, language = _inputs(batch=1)
    baseline = model(images, state, language).actions
    changed_image = model(torch.zeros_like(images), state, language).actions
    changed_language = model(images, state, ByteTokenizer(96)(["open the drawer"])).actions
    assert not torch.allclose(baseline, changed_image)
    assert not torch.allclose(baseline, changed_language)


def test_language_pooling_ignores_extra_padding():
    model = CompactVLA(CompactVLAConfig.smoke()).eval()
    images, state, _ = _inputs(batch=1)
    short = ByteTokenizer(16)(["lift"])
    padded = torch.nn.functional.pad(short, (0, 80))
    _, short_latent = model._encode(images, state, short)
    _, padded_latent = model._encode(images, state, padded)
    assert torch.allclose(short_latent, padded_latent)


def test_vision_encoder_preserves_spatial_patch_tokens():
    config = CompactVLAConfig.smoke()
    model = CompactVLA(config).eval()
    images, state, language = _inputs(batch=1)
    tokens, _ = model._encode(images, state, language)
    expected = 2 * 3 * config.vision_grid_size**2 + 2  # visual + language + state
    assert tokens.shape[1] == expected


def test_sample_actions_produces_chunk():
    model = CompactVLA(CompactVLAConfig.smoke()).eval()
    output = model.sample_actions(*_inputs(batch=1), integration_steps=2)
    assert output.actions.shape == (1, 20, 12)
