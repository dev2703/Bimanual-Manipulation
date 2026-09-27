from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimanual.data.verify_index import missing_videos, verify_entry


def _dataset(root: Path, cameras: tuple[str, ...] = ("observation.images.global",)) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"features": {name: {"dtype": "video"} for name in cameras}}))


def test_missing_videos_names_absent_cameras(tmp_path: Path):
    root = tmp_path / "set"
    _dataset(root, ("observation.images.global", "observation.images.left_wrist"))
    video = root / "videos/observation.images.global/chunk-000"
    video.mkdir(parents=True)
    (video / "file-000.mp4").write_bytes(b"not-empty")
    assert missing_videos(root) == ["observation.images.left_wrist"]


def test_empty_video_file_counts_as_missing(tmp_path: Path):
    root = tmp_path / "set"
    _dataset(root)
    video = root / "videos/observation.images.global/chunk-000"
    video.mkdir(parents=True)
    (video / "file-000.mp4").write_bytes(b"")
    assert missing_videos(root) == ["observation.images.global"]


def test_unknown_audit_mode_is_rejected(tmp_path: Path):
    root = tmp_path / "set"
    _dataset(root)
    video = root / "videos/observation.images.global"
    video.mkdir(parents=True)
    (video / "file.mp4").write_bytes(b"x")
    with pytest.raises(ValueError, match="audit_mode"):
        verify_entry({"root": str(root), "audit_mode": "guess"})
