from __future__ import annotations

import numpy as np
import pytest

from bimanual.policy.types import EpisodeManifest, SO101_BIMANUAL, file_sha256


def test_so101_embodiment_validates_dimensions():
    SO101_BIMANUAL.validate(np.zeros(12), np.zeros(12))
    with pytest.raises(ValueError, match="state dim 12"):
        SO101_BIMANUAL.validate(np.zeros(14))
    with pytest.raises(ValueError, match="action dim 12"):
        SO101_BIMANUAL.validate(np.zeros(12), np.zeros(14))


def test_episode_manifest_is_serializable_and_records_rates():
    manifest = EpisodeManifest(
        environment="bimanual_table",
        embodiment=SO101_BIMANUAL.name,
        model_revision=SO101_BIMANUAL.model_revision,
        seed=7,
        split="train",
        source_expert="mink_waypoint_v1",
    ).to_dict()
    assert manifest["physics_hz"] == 600
    assert manifest["control_hz"] == 30
    assert manifest["policy_hz"] == 10


def test_dataset_schema_contains_auditable_sim_time():
    from bimanual.experts.generate import make_features

    assert "privileged.sim_time" in make_features()


def test_file_sha256_tracks_artifact_contents(tmp_path):
    artifact = tmp_path / "model.xml"
    artifact.write_text("first")
    first = file_sha256(artifact)
    artifact.write_text("second")
    assert file_sha256(artifact) != first
