"""Publish and hash-verify the final 20k multitask checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

from bimanual.training.rocm.publish_huggingface import upload_verified


def main() -> None:
    root = Path("/workspace")
    api = HfApi()
    account = api.whoami()["name"]
    results = []

    def publish(repo: str, path: Path, prefix: str = "") -> None:
        results.append(upload_verified(api, f"{account}/{repo}", "model", path, prefix=prefix))
        status_path.write_text(json.dumps({"account": account, "uploads": results}, indent=2) + "\n")

    status_path = root / "logs/multitask_20k_hub_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    publish("bimanual-compact-vla-160m-multitask-20k", root / "checkpoints/compact-160m-multitask-20k")
    checkpoint = root / "checkpoints/pi05-multitask-20k/checkpoints/020000"
    publish("bimanual-pi05-multitask-20k", checkpoint / "pretrained_model")
    publish("bimanual-pi05-multitask-20k", checkpoint / "training_state", "training_state")
    status = {"account": account, "private": True, "complete": True, "uploads": results}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
