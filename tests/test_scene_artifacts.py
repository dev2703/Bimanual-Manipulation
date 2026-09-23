"""Scene manifests must include the physics and visual files used at capture."""

import json

import pytest

from bimanual.data.scene_artifacts import assert_scene_compatible, scene_artifact_hashes


def _scene(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "cup.stl").write_bytes(b"mesh-v1")
    (assets / "cloth.png").write_bytes(b"texture-v1")
    (tmp_path / "robot.xml").write_text(
        '<mujoco><compiler meshdir="assets" texturedir="assets"/>'
        '<asset><mesh file="cup.stl"/><texture file="cloth.png"/></asset></mujoco>'
    )
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="robot.xml"/></mujoco>')
    return scene


def test_scene_hashes_track_includes_meshes_and_textures(tmp_path):
    scene = _scene(tmp_path)
    baseline = scene_artifact_hashes(scene)
    assert set(baseline) == {"scene.xml", "robot.xml", "assets/cup.stl", "assets/cloth.png"}

    for name in ("robot.xml", "assets/cup.stl", "assets/cloth.png"):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + (b"<!-- changed -->" if path.suffix == ".xml" else b"changed"))
        changed = scene_artifact_hashes(scene)
        assert changed[name] != baseline[name]
        path.write_bytes(original)


def test_scene_compatibility_rejects_changed_asset_and_mixed_manifests(tmp_path):
    scene = _scene(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    hashes = scene_artifact_hashes(scene)
    manifest_path = dataset / "episode_manifests.json"
    manifest_path.write_text(json.dumps([{"artifact_hashes": hashes}]))
    assert_scene_compatible(dataset, scene)

    (tmp_path / "assets/cup.stl").write_bytes(b"mesh-v2")
    with pytest.raises(ValueError, match="assets/cup.stl"):
        assert_scene_compatible(dataset, scene)

    manifest_path.write_text(json.dumps([{"artifact_hashes": hashes}, {"artifact_hashes": {}}]))
    with pytest.raises(ValueError, match="disagree"):
        assert_scene_compatible(dataset, scene)


def test_legacy_root_scene_hash_remains_readable(tmp_path):
    scene = _scene(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    only_root = {"scene.xml": scene_artifact_hashes(scene)["scene.xml"]}
    (dataset / "episode_manifests.json").write_text(json.dumps([{"artifact_hashes": only_root}]))
    assert_scene_compatible(dataset, scene)
