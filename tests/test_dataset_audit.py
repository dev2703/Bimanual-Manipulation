import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bimanual.data.audit import assert_disjoint_splits, audit_dataset


def _dataset(root, *, identity=False, seeds=(1, 2)):
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    episodes = np.repeat(np.arange(len(seeds)), 3)
    state = [np.array([i, -i], dtype=np.float32) for i in range(len(episodes))]
    action = [x.copy() if identity else x + np.array([0.1, 0.0], np.float32) for x in state]
    sim_time = np.tile(np.arange(3, dtype=np.float32) / 10, len(seeds))
    frame_seeds = np.repeat(seeds, 3)
    pq.write_table(
        pa.table({
            "episode_index": episodes,
            "observation.state": state,
            "action": action,
            "privileged.sim_time": sim_time,
            "privileged.scene_seed": frame_seeds,
        }),
        root / "data/data.parquet",
    )
    (root / "meta/info.json").write_text(json.dumps({"total_frames": len(episodes), "total_episodes": len(seeds)}))
    (root / "episode_manifests.json").write_text(json.dumps([{"seed": x} for x in seeds]))


def test_audit_accepts_synchronized_nonidentity_dataset(tmp_path):
    root = tmp_path / "train"
    _dataset(root)
    result = audit_dataset(root)
    assert result.episodes == 2
    assert result.mean_dt == pytest.approx(0.1)
    assert result.action_state_mae == pytest.approx(0.05)


def test_audit_rejects_identity_actions(tmp_path):
    root = tmp_path / "identity"
    _dataset(root, identity=True)
    with pytest.raises(ValueError, match="copied"):
        audit_dataset(root)


def test_split_audit_rejects_seed_leakage(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    _dataset(train, seeds=(1, 2))
    _dataset(val, seeds=(2, 3))
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_splits(train, val)
