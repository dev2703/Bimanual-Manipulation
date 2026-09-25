"""Run locally. Wait for the 160M and pi0.5 multitask training containers on
the droplet to finish, pull their checkpoints, then publish the checkpoints
and the merged multitask dataset to the private Hub.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "bimanual/training/rocm"))
from publish_huggingface import upload_verified  # noqa: E402

JOBS = {
    "compact160m-multitask-5k": "checkpoints/compact-160m-multitask-5k",
    "pi05-multitask-5k": "checkpoints/pi05-multitask-5k",
}


def ssh(host: str, key: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", key, "-o", "BatchMode=yes", host, *args],
        capture_output=True, text=True,
    )


def container_running(host: str, key: str, name: str) -> bool:
    result = ssh(host, key, "docker", "inspect", "-f", "{{.State.Running}}", name)
    return result.returncode == 0 and result.stdout.strip() == "true"


def container_exit_code(host: str, key: str, name: str) -> int:
    result = ssh(host, key, "docker", "inspect", "-f", "{{.State.ExitCode}}", name)
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="root@129.212.176.178")
    parser.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519"))
    parser.add_argument("--local-root", default="outputs/droplet_129_212_176_178")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_hours * 3600
    pending = set(JOBS)
    while pending and time.monotonic() < deadline:
        for name in list(pending):
            if not container_running(args.host, args.key, name):
                code = container_exit_code(args.host, args.key, name)
                print(f"{name}: finished with exit code {code}", flush=True)
                pending.discard(name)
        if pending:
            time.sleep(args.poll_seconds)
    if pending:
        raise RuntimeError(f"Timed out waiting for: {sorted(pending)}")

    local_root = Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    account_repo_suffix = "multitask-5k"

    for name, checkpoint_subdir in JOBS.items():
        destination = local_root / checkpoint_subdir
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-az", "-e", f"ssh -i {args.key}",
            f"{args.host}:/root/bimanual-training/{checkpoint_subdir}/", f"{str(destination)}/",
        ], check=True)
        print(f"pulled checkpoint for {name} -> {destination}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    account = api.whoami()["name"]

    upload_verified(
        api, f"{account}/bimanual-compact-vla-160m-{account_repo_suffix}", "model",
        local_root / JOBS["compact160m-multitask-5k"],
    )
    pi05_checkpoints = sorted(
        (local_root / JOBS["pi05-multitask-5k"] / "checkpoints").glob("[0-9]*"),
        key=lambda p: p.name,
    )
    if pi05_checkpoints:
        latest = pi05_checkpoints[-1]
        upload_verified(
            api, f"{account}/bimanual-pi05-{account_repo_suffix}", "model",
            latest / "pretrained_model",
        )
    upload_verified(
        api, f"{account}/bimanual-aloha-multitask-train", "dataset",
        "outputs/aloha_multitask_train",
    )
    upload_verified(
        api, f"{account}/bimanual-aloha-multitask-val", "dataset",
        "outputs/aloha_multitask_val",
    )
    print("All multitask training artifacts verified on the Hub.", flush=True)


if __name__ == "__main__":
    main()
