

from __future__ import annotations

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lerobot[smolvla,dataset]==0.6.1",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "transformers==5.5.4",
)
volume = modal.Volume.from_name("bimanual-dinner")
app = modal.App("bimanual-dinner-dataset-check", image=image)


@app.function(volumes={"/workspace": volume}, timeout=300)
def verify_dataset() -> dict:
    import json
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("/workspace/datasets/aloha_mug_train")
    info = json.loads((root / "meta/info.json").read_text())
    if int(info["total_episodes"]) != 50 or int(info["fps"]) != 10:
        raise ValueError("expected the audited 50-episode, 10 Hz mug dataset")
    dataset = LeRobotDataset("local/aloha-dinner-mug", root=root, video_backend="pyav")
    sample = dataset[0]
    cameras = ("global", "left_wrist", "right_wrist")
    for camera in cameras:
        if sample[f"observation.images.{camera}"].shape != (3, 256, 256):
            raise ValueError(f"unexpected {camera} image shape")
    if sample["observation.state"].shape != (14,) or sample["action"].shape != (14,):
        raise ValueError("ALOHA state and action must each have 14 ordered values")
    return {"episodes": len(dataset.meta.episodes), "frames": len(dataset),
            "cameras": cameras, "state_shape": list(sample["observation.state"].shape),
            "action_shape": list(sample["action"].shape)}


@app.local_entrypoint()
def main() -> None:
    print(verify_dataset.remote())
