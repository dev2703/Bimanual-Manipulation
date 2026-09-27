"""Check every dataset listed in outputs/DATA_INDEX.json.

Full audits replay the timing and action checks. Video checks catch the
failure where parquet survived but camera files were deleted. Datasets merged
without privileged columns are listed as videos_only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bimanual.data.audit import assert_disjoint_splits, audit_dataset

INDEX_PATH = Path("outputs/DATA_INDEX.json")
IMAGE_PREFIX = "observation.images."


def video_keys(root: Path) -> list[str]:
    info = json.loads((root / "meta/info.json").read_text())
    return [key for key in info["features"] if key.startswith(IMAGE_PREFIX)]


def missing_videos(root: Path) -> list[str]:
    missing = []
    for key in video_keys(root):
        files = [path for path in (root / "videos" / key).rglob("*.mp4") if path.stat().st_size > 0]
        if not files:
            missing.append(key)
    return missing


def verify_entry(entry: dict) -> dict:
    root = Path(entry["root"])
    result = {"root": entry["root"], "mode": entry["audit_mode"]}
    if not root.is_dir():
        raise FileNotFoundError(root)
    missing = missing_videos(root)
    if missing:
        raise ValueError(f"{root} is missing camera videos: {missing}")
    result["cameras"] = video_keys(root)
    if entry["audit_mode"] == "full":
        report = audit_dataset(root)
        if report.episodes != entry["episodes"]:
            raise ValueError(f"{root} has {report.episodes} episodes, index expects {entry['episodes']}")
        manifests = json.loads((root / "episode_manifests.json").read_text())
        hashes = manifests[0].get("artifact_hashes")
        if hashes != entry["scene_hashes"]:
            raise ValueError(f"{root} scene hashes no longer match the index")
        result["audit"] = asdict(report)
    elif entry["audit_mode"] != "videos_only":
        raise ValueError(f"unknown audit_mode {entry['audit_mode']!r}")
    return result


def verify_index(index_path: Path = INDEX_PATH) -> list[dict]:
    index = json.loads(index_path.read_text())
    results = [verify_entry(entry) for entry in index["datasets"]]
    groups: dict[str, list[str]] = {}
    for entry in index["datasets"]:
        if entry["audit_mode"] != "full":
            continue
        groups.setdefault(entry["pair"], []).append(entry["root"])
    for roots in groups.values():
        if len(roots) > 1:
            assert_disjoint_splits(*roots)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(INDEX_PATH))
    args = parser.parse_args()
    results = verify_index(Path(args.index))
    print(json.dumps({"datasets": len(results), "ok": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
