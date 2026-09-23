"""Tests for bimanual/perception/verifier.py's architecture and
preprocessing -- shape/contract checks only. Training accuracy (the real
Phase 5 gate: >=95% verifier-vs-oracle agreement) requires a real dataset
run through bimanual/training/train_verifier.py, not covered here (see
that module's docstring for why the current 2-episode dataset can't
produce a meaningful number yet).
"""

from __future__ import annotations

import numpy as np
import torch

from bimanual.perception.verifier import PREDICATE_NAMES, PredicateVerifier, preprocess_frame


def test_preprocess_frame_shape_and_range():
    frame = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    t = preprocess_frame(frame)
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_preprocess_frame_rejects_non_uint8():
    frame = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        preprocess_frame(frame)
        assert False, "expected an assertion error for non-uint8 input"
    except AssertionError:
        pass


def test_predicate_verifier_forward_output_shape():
    model = PredicateVerifier()
    batch = 4
    imgs = [torch.rand(batch, 3, 256, 256) for _ in range(3)]
    logits = model(*imgs)
    assert logits.shape == (batch, len(PREDICATE_NAMES))


def test_predicate_verifier_forward_handles_non_multiple_of_16_size():
    # 256 is a nice power of two; guard against the backbone silently
    # assuming that and breaking on an odd input size (e.g. a future
    # non-256 render resolution).
    model = PredicateVerifier()
    imgs = [torch.rand(1, 3, 130, 130) for _ in range(3)]
    logits = model(*imgs)
    assert logits.shape == (1, len(PREDICATE_NAMES))


def test_predicate_verifier_predict_returns_bool_dict_with_expected_keys():
    model = PredicateVerifier()
    imgs = [torch.rand(3, 256, 256) for _ in range(3)]
    preds = model.predict(*imgs)
    assert set(preds) == set(PREDICATE_NAMES)
    for v in preds.values():
        assert isinstance(v, bool)


def test_predicate_verifier_predict_threshold_is_respected():
    model = PredicateVerifier(predicate_names=("only_one",))
    # Force known logits by overriding the head with a fixed bias.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(10.0)  # sigmoid(10) >> 0.5
    imgs = [torch.zeros(3, 32, 32) for _ in range(3)]
    preds = model.predict(*imgs, threshold=0.5)
    assert preds["only_one"] is True

    with torch.no_grad():
        model.head.bias.fill_(-10.0)
    preds = model.predict(*imgs, threshold=0.5)
    assert preds["only_one"] is False


def test_predicate_names_cover_drawer_and_all_five_objects():
    assert "drawer_open" in PREDICATE_NAMES
    for obj in ("plate", "mug", "bottle", "fork", "spoon"):
        assert f"{obj}_in_region" in PREDICATE_NAMES
