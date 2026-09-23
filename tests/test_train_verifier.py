"""Tests for bimanual/training/train_verifier.py's compute_labels() --
the pure oracle-label-from-privileged-fields function, without needing a
real dataset or a training run.
"""

from __future__ import annotations

import numpy as np
import torch

from bimanual.perception.verifier import PREDICATE_NAMES
from bimanual.sim.scene_builder import DRAWER_OPEN_DIST, OBJECT_SPEC, REGIONS, TABLE_HEIGHT, resting_center_z
from bimanual.training.train_verifier import OBJECT_ORDER, compute_labels


def _fake_batch(object_poses: np.ndarray, drawer_opening: np.ndarray) -> dict:
    return {
        "privileged.object_poses": torch.from_numpy(object_poses.astype(np.float32)),
        "privileged.drawer_opening": torch.from_numpy(drawer_opening.astype(np.float32)),
    }


def test_compute_labels_shape():
    poses = np.zeros((3, 5, 3))
    drawer = np.zeros((3, 1))
    labels = compute_labels(_fake_batch(poses, drawer))
    assert labels.shape == (3, len(PREDICATE_NAMES))


def test_compute_labels_drawer_open_threshold():
    poses = np.zeros((2, 5, 3))
    # index 0: closed; index 1: fully open (negative direction, per
    # scene_builder.py's DRAWER_OPEN_DIST convention).
    drawer = np.array([[0.0], [-DRAWER_OPEN_DIST]])
    labels = compute_labels(_fake_batch(poses, drawer))
    idx = PREDICATE_NAMES.index("drawer_open")
    assert labels[0, idx] == 0.0
    assert labels[1, idx] == 1.0


def test_compute_labels_in_region_matches_object_position():
    poses = np.zeros((1, 5, 3))
    for i, obj in enumerate(OBJECT_ORDER):
        region_name = {
            "plate": "plate_region", "mug": "mug_region", "bottle": "bottle_region",
            "fork": "left_cutlery_region", "spoon": "right_cutlery_region",
        }[obj]
        rx, ry, _radius = REGIONS[region_name]
        expected_z = (
            TABLE_HEIGHT + OBJECT_SPEC[obj]["half_height"]
            if obj in ("fork", "spoon")
            else resting_center_z(obj)
        )
        poses[0, i] = [rx, ry, expected_z]
    drawer = np.zeros((1, 1))
    labels = compute_labels(_fake_batch(poses, drawer))

    for obj in OBJECT_ORDER:
        idx = PREDICATE_NAMES.index(f"{obj}_in_region")
        assert labels[0, idx] == 1.0, f"{obj} should read in-region when placed exactly at its region center"


def test_compute_labels_far_object_is_not_in_region():
    poses = np.full((1, 5, 3), 100.0)  # absurdly far from every region
    drawer = np.zeros((1, 1))
    labels = compute_labels(_fake_batch(poses, drawer))
    for obj in OBJECT_ORDER:
        idx = PREDICATE_NAMES.index(f"{obj}_in_region")
        assert labels[0, idx] == 0.0
