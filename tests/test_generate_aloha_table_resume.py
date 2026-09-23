"""Interrupted dinner datasets resume only with matching seeds and scene."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bimanual.experts.generate_aloha_table import (
    SKILLS,
    _compatible_features,
    _recover_manifests,
    _save_manifests,
    make_features,
)


def _partial(root, seeds=(0, 1)):
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0, 0, 1, 1],
        "privileged.scene_seed": [seeds[0], seeds[0], seeds[1], seeds[1]],
    }), root / "data/partial.parquet")


def test_recover_partial_dataset_and_save_manifests(tmp_path):
    _partial(tmp_path)
    hashes = {"task_table_setting_plate_v2.xml": "sha256:test"}
    manifests = _recover_manifests(tmp_path, 2, 0, SKILLS["plate_pick_place"], "train", hashes)
    assert [item["seed"] for item in manifests] == [0, 1]
    _save_manifests(tmp_path, manifests)
    assert json.loads((tmp_path / "episode_manifests.json").read_text()) == manifests
    assert _recover_manifests(tmp_path, 2, 0, SKILLS["plate_pick_place"], "train", hashes) == manifests


def test_resume_rejects_corrupt_seed_or_changed_scene(tmp_path):
    _partial(tmp_path, seeds=(0, 9))
    skill = SKILLS["plate_pick_place"]
    with pytest.raises(ValueError, match="unexpected scene seed"):
        _recover_manifests(tmp_path, 2, 0, skill, "train", {"scene": "sha256:a"})

    _partial_root = tmp_path / "valid"
    _partial(_partial_root)
    manifests = _recover_manifests(_partial_root, 2, 0, skill, "train", {"scene": "sha256:a"})
    _save_manifests(_partial_root, manifests)
    with pytest.raises(ValueError, match="manifests disagree"):
        _recover_manifests(_partial_root, 2, 0, skill, "train", {"scene": "sha256:b"})


def test_feature_check_allows_lerobot_indices_but_rejects_wrong_action_shape():
    declared = make_features()
    actual = {name: {**feature} for name, feature in declared.items()}
    actual["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    assert _compatible_features(actual, declared)
    actual["action"]["shape"] = (12,)
    assert not _compatible_features(actual, declared)


def test_resume_empty_dataset_needs_no_parquet(tmp_path):
    assert _recover_manifests(
        tmp_path, 0, 0, SKILLS["drawer_open"], "train", {"scene": "sha256:a"},
    ) == []
