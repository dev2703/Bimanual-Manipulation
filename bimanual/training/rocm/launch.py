"""Config-driven GPU job launcher.

A job file is `rocm/jobs/<name>.yaml`. The real chain on a droplet is train,
hash-verified Hub upload, eval, then power-off. `--smoke` runs the same config
for a few local steps on CPU or MPS and writes a report. It does not upload
or power anything off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

JOBS = Path(__file__).resolve().parent / "jobs"


def load_job(name: str) -> dict:
    path = JOBS / f"{name}.yaml"
    job = yaml.safe_load(path.read_text())
    required = ("name", "policy", "instruction", "steps", "chunk_size", "n_action_steps", "smoke_steps")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"{path.name} missing {missing}")
    if job["instruction"] not in {"plain", "memory"}:
        raise ValueError("instruction must be plain or memory")
    if int(job["n_action_steps"]) > int(job["chunk_size"]):
        raise ValueError("n_action_steps cannot exceed chunk_size")
    job["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return job


def smoke(job: dict, output: Path) -> dict:
    """Twenty optimizer steps so the job config is exercised on this machine."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    layer = torch.nn.Linear(int(job["chunk_size"]), int(job["n_action_steps"])).to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    losses = []
    for _ in range(int(job["smoke_steps"])):
        optimizer.zero_grad()
        batch = torch.randn(4, int(job["chunk_size"]), device=device)
        loss = layer(batch).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    report = {
        "job": job["name"],
        "instruction": job["instruction"],
        "device": device,
        "smoke_steps": int(job["smoke_steps"]),
        "n_action_steps": int(job["n_action_steps"]),
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "config_sha256": job["config_sha256"],
        "chain": ["train", "hash_verified_upload", "eval", "power_off"],
        "scope": "local smoke of the job config; GPU train is not started",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    job = load_job(args.job)
    if not args.smoke:
        raise SystemExit("Refusing to start a GPU chain from this machine. Re-run with --smoke, or launch on the droplet.")
    output = args.output or Path("outputs/gates") / f"job_smoke_{job['name']}.json"
    smoke(job, output)


if __name__ == "__main__":
    main()
