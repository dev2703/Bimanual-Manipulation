"""Train the local Redwood-inspired compact VLA on a validated dataset.

The command refuses legacy datasets without episode manifests by default so
the known final-frame image bug cannot silently contaminate a new experiment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from bimanual.data.audit import audit_dataset, assert_disjoint_splits
from bimanual.policy.compact_vla import ByteTokenizer, CompactVLA, CompactVLAConfig


CAMERA_KEYS = (
    "observation.images.global",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _make_model_config(name: str, state_dim: int, action_dim: int) -> CompactVLAConfig:
    if name == "smoke":
        return CompactVLAConfig.smoke(state_dim, action_dim)
    if name == "research160":
        return CompactVLAConfig.research_160m(state_dim, action_dim)
    return CompactVLAConfig(state_dim=state_dim, action_dim=action_dim)


def batch_inputs(batch: dict, tokenizer: ByteTokenizer, device: str) -> tuple:
    # Initial curriculum uses the current frame. History frames are added by
    # changing only this adapter, not the model contract.
    images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=1).unsqueeze(1).to(device)
    state = batch["observation.state"].to(device)
    tasks = list(batch["task"])
    language = tokenizer(tasks, device=device)
    actions = batch["action"].to(device)
    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    pad = batch.get("action_is_pad")
    if pad is not None:
        pad = pad.to(device)
    return images, state, language, actions, pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="local/bimanual-physical")
    parser.add_argument("--output", default="outputs/compact_vla")
    parser.add_argument("--config", choices=["smoke", "research", "research160"], default="smoke")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-legacy-dataset", action="store_true")
    parser.add_argument("--val-root")
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=500)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_freq, args.val_batches) < 1:
        parser.error('steps, batch-size, save-freq, and val-batches must be positive')

    root = Path(args.root)
    manifest_path = root / "episode_manifests.json"
    if not manifest_path.exists() and not args.allow_legacy_dataset:
        raise RuntimeError(
            f"{root} has no episode_manifests.json and may predate synchronized recording; "
            "regenerate it or pass --allow-legacy-dataset for diagnostics only"
        )
    if manifest_path.exists():
        report = audit_dataset(root)
        print(f"dataset_audit={json.dumps(asdict(report), sort_keys=True)}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(args.seed)
    probe = LeRobotDataset(repo_id=args.repo_id, root=root, episodes=args.episodes, video_backend="pyav")
    state_dim = int(probe.meta.features["observation.state"]["shape"][0])
    action_dim = int(probe.meta.features["action"]["shape"][0])
    config = _make_model_config(args.config, state_dim, action_dim)
    del probe

    delta_timestamps = {"action": [i / 10 for i in range(config.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers)
    model = CompactVLA(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tokenizer = ByteTokenizer(config.max_language_tokens)
    val_loader = None
    if args.val_root:
        audit_dataset(args.val_root)
        assert_disjoint_splits(root, args.val_root)
        validation = LeRobotDataset('local/aloha-dinner-mug-val', root=Path(args.val_root),
                                    delta_timestamps=delta_timestamps, video_backend='pyav')
        indices = torch.linspace(0, len(validation) - 1,
                                 min(len(validation), args.val_batches * args.batch_size)).long().tolist()
        val_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    def evaluate():
        if val_loader is None:
            return None
        model.eval()
        losses = []
        devices = [torch.cuda.current_device()] if args.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(12345)
            for index, batch in enumerate(val_loader):
                if index >= args.val_batches:
                    break
                images, state, language, actions, pad = batch_inputs(batch, tokenizer, args.device)
                losses.append(float(model.flow_matching_loss(
                    images, state, language, actions, action_is_pad=pad).item()))
        model.train()
        return sum(losses) / len(losses)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    initial_val_loss = evaluate()
    print(f'parameters={model.parameter_counts()} initial_val_loss={initial_val_loss}', flush=True)

    iterator = iter(loader)
    model.train()
    first_loss = None
    final_loss = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, state, language, actions, pad = batch_inputs(batch, tokenizer, args.device)
        loss = model.flow_matching_loss(images, state, language, actions, action_is_pad=pad)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        if first_loss is None:
            first_loss = float(loss.item())
        final_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step} flow_loss={loss.item():.6f}", flush=True)
        if step % args.save_freq == 0 or step == args.steps:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'step': step, 'config': asdict(config)}, output / 'latest.tmp')
            (output / 'latest.tmp').replace(output / 'latest.pt')

    final_val_loss = evaluate()
    torch.save(model.state_dict(), output / "model.pt")
    metadata = {
        "architecture": "original_redwood_inspired_compact_vla",
        "not_a_neo_reproduction": True,
        "config": asdict(config),
        "parameters": model.parameter_counts(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "episodes": args.episodes,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "initial_val_loss": initial_val_loss,
            "final_val_loss": final_val_loss,
            "val_batches": args.val_batches,
        },
        "dataset_root": str(root),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
