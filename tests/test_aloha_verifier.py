import numpy as np
import pytest
import torch

from bimanual.perception.aloha_verifier import (
    AlohaRGBVerifier, balanced_indices, binary_agreement, labels_from_recorded_state,
)
from bimanual.perception.verifier import PredicateVerifier
from bimanual.training.train_aloha_verifier import CounterfactualPairs


def test_labels_use_recorded_position_and_drawer_opening():
    positions = np.zeros((3, 15))
    positions[:, 3:6] = [[0.0, 0.0, 0.029], [0.3, 0.1, 0.029], [0.3, 0.1, 0.08]]
    opening = np.array([0.0, 0.12, 0.0])
    target = np.array([0.3, 0.1, 0.0])
    np.testing.assert_array_equal(labels_from_recorded_state("mug_pick_place", positions, opening, target), [0, 1, 0])
    np.testing.assert_array_equal(labels_from_recorded_state("drawer_open", positions, opening), [0, 1, 0])


def test_balance_and_metrics_reject_constant_label_shortcuts():
    labels = np.array([0] * 60 + [1] * 30)
    indices = balanced_indices(labels, max_per_class=25, seed=1)
    assert len(indices) == 50 and labels[indices].sum() == 25
    result = binary_agreement(np.zeros(90), labels)
    assert result["positive_recall"] == 0.0
    assert result["negative_recall"] == 1.0
    with pytest.raises(ValueError, match="both classes"):
        balanced_indices(np.zeros(5), max_per_class=2)


def test_rgb_verifier_receives_only_frames():
    class ShapeCheckingVerifier(PredicateVerifier):
        def predict(self, global_img, left_img, right_img, threshold=0.5):
            assert global_img.shape == left_img.shape == right_img.shape == (3, 96, 96)
            return {"mug_in_region": False}

    model = ShapeCheckingVerifier(("mug_in_region",))
    adapter = AlohaRGBVerifier(model, "mug_in_region")
    frames = {name: np.zeros((32, 32, 3), dtype=np.uint8)
              for name in ("overhead_cam", "wrist_cam_left", "wrist_cam_right")}
    assert adapter(frames) == {"mug_in_region": False}


def test_counterfactual_archive_requires_matched_scene_pairs(tmp_path):
    images = np.zeros((2, 3, 256, 256, 3), dtype=np.uint8)
    archive = tmp_path / "pairs.npz"
    np.savez_compressed(archive, images=images, labels=[1, 0], seeds=[7, 7])
    pairs = CounterfactualPairs(archive)
    assert len(pairs) == 2
    assert pairs[0][0].shape == (3, 96, 96)
    np.savez_compressed(archive, images=images, labels=[1, 0], seeds=[7, 8])
    with pytest.raises(ValueError, match="share a scene seed"):
        CounterfactualPairs(archive)
