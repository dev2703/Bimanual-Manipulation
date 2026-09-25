"""Publish completed multitask GPU experiments from a droplet.

The caller supplies Hugging Face authentication through HF_HOME. All repos
are private, and upload_verified checks every remote file hash.
"""

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

    def publish(repo: str, kind: str, path: Path, prefix: str = "") -> None:
        results.append(upload_verified(api, f"{account}/{repo}", kind, path, prefix=prefix))
        (root / "logs/multitask_hub_status.json").write_text(
            json.dumps({"account": account, "private": True, "uploads": results}, indent=2) + "\n"
        )

    compact = root / "checkpoints/compact-160m-multitask-5k"
    pi = root / "checkpoints/pi05-multitask-5k/checkpoints/005000"
    publish("bimanual-compact-vla-160m-multitask-5k", "model", compact)
    publish("bimanual-pi05-multitask-5k", "model", pi / "pretrained_model")
    publish("bimanual-pi05-multitask-5k", "model", pi / "training_state", "training_state")
    publish("bimanual-aloha-multitask-train", "dataset", root / "datasets/aloha_multitask_train")
    publish("bimanual-aloha-multitask-val", "dataset", root / "datasets/aloha_multitask_val")
    status = {"account": account, "private": True, "complete": True, "uploads": results}
    (root / "logs/multitask_hub_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
