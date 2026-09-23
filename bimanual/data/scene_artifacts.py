"""Fingerprint every local file that contributes to a MuJoCo scene."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from bimanual.policy.types import file_sha256


def scene_artifact_hashes(scene_path: str | Path) -> dict[str, str]:
    """Hash the scene, recursive includes, and referenced mesh/texture assets.

    Keys are relative to the entry scene's directory, so manifests remain
    portable when a checkout is moved. Files outside that directory are rejected
    rather than recorded under machine-specific absolute paths.
    """
    scene = Path(scene_path).resolve()
    root_dir = scene.parent
    visited: set[Path] = set()

    def visit(xml_path: Path) -> None:
        xml_path = xml_path.resolve()
        if xml_path in visited:
            return
        visited.add(xml_path)
        tree = ET.parse(xml_path)
        for include in tree.iter("include"):
            visit(xml_path.parent / include.attrib["file"])

        compiler = tree.find("compiler")
        options = compiler.attrib if compiler is not None else {}
        asset_dir = options.get("assetdir", "")
        for tag, option in (("mesh", "meshdir"), ("texture", "texturedir"), ("hfield", "assetdir")):
            directory = options.get(option, asset_dir)
            for element in tree.findall(f".//asset/{tag}"):
                filename = element.get("file")
                if filename:
                    visited.add((xml_path.parent / directory / filename).resolve())

    visit(scene)
    hashes = {}
    for path in sorted(visited):
        try:
            key = path.relative_to(root_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"scene dependency is outside {root_dir}: {path}") from exc
        hashes[key] = f"sha256:{file_sha256(path)}"
    return hashes


def assert_scene_compatible(dataset_root: str | Path, scene_path: str | Path) -> None:
    """Reject replay if a dataset's recorded scene dependencies have changed."""
    manifests = json.loads((Path(dataset_root) / "episode_manifests.json").read_text())
    if not manifests:
        raise ValueError("dataset has no episode manifests")
    expected = manifests[0].get("artifact_hashes")
    if not expected:
        raise ValueError("episode manifest has no scene artifact hashes")
    if any(item.get("artifact_hashes") != expected for item in manifests[1:]):
        raise ValueError("episode manifests disagree on scene artifact hashes")
    actual = scene_artifact_hashes(scene_path)
    for key, digest in expected.items():
        if actual.get(key) != digest:
            raise ValueError(f"scene artifact mismatch: {key}")
