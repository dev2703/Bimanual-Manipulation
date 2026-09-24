"""Pinned SmolVLA smoke/fine-tune job for the local physical mug dataset.

Requires the optional `modal` extra and an authenticated Modal CLI. Upload the
audited local dataset first, for example:

  modal volume create bimanual-dinner
  modal volume put bimanual-dinner outputs/aloha_mug_train datasets/aloha_mug_train
  modal run bimanual/training/modal_app.py --steps 200

The public base model needs no Hugging Face write token. The dataset and all
checkpoints stay private in the Modal volume; no credentials are embedded.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "bimanual-dinner"
DATASET_ROOT = "/workspace/datasets/aloha_mug_train"
OUTPUT_ROOT = "/workspace/checkpoints"
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lerobot[smolvla,dataset]==0.6.1",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "transformers==5.5.4",
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("bimanual-dinner-smolvla", image=image)


@app.function(volumes={"/workspace": volume}, timeout=300)
def verify_dataset() -> dict:
    """Exercise the uploaded dataset and video decoder before reserving a GPU."""
    import json
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(DATASET_ROOT)
    info = json.loads((root / "meta/info.json").read_text())
    if int(info["total_episodes"]) != 50 or int(info["fps"]) != 10:
        raise ValueError("expected the audited 50-episode, 10 Hz mug dataset")
    dataset = LeRobotDataset("local/aloha-dinner-mug", root=root, video_backend="pyav")
    sample = dataset[0]
    for key in ("observation.images.global", "observation.images.left_wrist",
                "observation.images.right_wrist", "observation.state", "action"):
        if key not in sample:
            raise KeyError(f"missing training feature: {key}")
    if sample["observation.state"].shape != sample["action"].shape or sample["action"].numel() != 14:
        raise ValueError("ALOHA state and action must each have 14 ordered values")
    return {"episodes": len(dataset.meta.episodes), "frames": len(dataset),
            "image_shape": list(sample["observation.images.global"].shape),
            "state_shape": list(sample["observation.state"].shape),
            "action_shape": list(sample["action"].shape)}


@app.function(gpu="A10G", volumes={"/workspace": volume}, timeout=7200)
def fine_tune(steps: int = 200) -> str:
    import json
    from pathlib import Path
    import subprocess

    if steps < 1 or steps > 20_000:
        raise ValueError("steps must be between 1 and 20000")
    root = Path(DATASET_ROOT)
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"upload the dataset to {DATASET_ROOT} first")
    info = json.loads((root / "meta/info.json").read_text())
    if int(info["total_episodes"]) < 50 or int(info["fps"]) != 10:
        raise ValueError("SmolVLA requires the audited 50-episode 10 Hz mug dataset")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    output = f"{OUTPUT_ROOT}/smolvla_mug_{steps}"
    command = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.chunk_size=20",
        "--policy.n_action_steps=8",
        "--policy.train_expert_only=true",
        "--dataset.repo_id=local/aloha-dinner-mug",
        f"--dataset.root={root}",
        "--dataset.video_backend=pyav",
        "--batch_size=8",
        f"--steps={steps}",
        f"--output_dir={output}",
        "--job_name=smolvla_aloha_mug",
        "--save_checkpoint=true",
        f"--save_freq={steps}",
        "--wandb.enable=false",
    ]
    subprocess.run(command, check=True)
    volume.commit()
    return output


@app.local_entrypoint()
def main(steps: int = 200, verify_only: bool = False) -> None:
    print(verify_dataset.remote())
    if not verify_only:
        print(fine_tune.remote(steps))
