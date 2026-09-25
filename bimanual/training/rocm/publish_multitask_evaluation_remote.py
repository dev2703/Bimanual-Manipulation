"""Publish and hash-verify closed-loop evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

from bimanual.training.rocm.publish_huggingface import upload_verified


def main() -> None:
    root = Path("/workspace/evaluation-results")
    api = HfApi()
    account = api.whoami()["name"]
    result = upload_verified(api, f"{account}/bimanual-multitask-evaluation-20k", "model", root)
    status = {"account": account, "private": True, "complete": True, "upload": result}
    (root / "huggingface_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
