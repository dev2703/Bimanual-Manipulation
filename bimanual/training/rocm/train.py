"""Validate the AMD runtime and mug data, then fine-tune SmolVLA privately."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20_000:
        raise ValueError('steps must be between 1 and 20000')
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm-enabled PyTorch and an accessible AMD GPU are required')
    print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
                      'gpu': torch.cuda.get_device_name(0)}), flush=True)
    # Exercise both forward and backward kernels before loading the policy.
    x = torch.randn(128, 128, device='cuda', requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    source = Path('/workspace/datasets/aloha_mug_train')
    root = Path('/workspace/datasets/aloha_mug_train_stable_stats')
    if not root.exists():
        shutil.copytree(source, root)
    # Float32 aggregate statistics in the recorded metadata can report zero
    # variance for almost stationary joints, magnifying rounding error by 1e8.
    # Recompute only state/action mean and std in float64 in a training copy.
    table = pq.read_table(sorted((source / 'data').rglob('*.parquet')))
    stats = json.loads((source / 'meta/stats.json').read_text())
    for key in ('observation.state', 'action'):
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        stats[key]['mean'] = values.mean(axis=0).tolist()
        stats[key]['std'] = np.maximum(values.std(axis=0), 1e-4).tolist()
    (root / 'meta/stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    (root / 'normalization_provenance.json').write_text(json.dumps({
        'source': str(source), 'method': 'float64 population mean/std',
        'std_floor': 1e-4, 'features': ['observation.state', 'action'],
    }, indent=2) + '\n')
    info = json.loads((root / 'meta/info.json').read_text())
    if info['total_episodes'] != 50 or info['fps'] != 10:
        raise ValueError('Expected the audited 50-episode, 10 Hz mug dataset')
    dataset = LeRobotDataset('local/aloha-dinner-mug', root=root, video_backend='pyav')
    sample = dataset[0]
    for camera in ('global', 'left_wrist', 'right_wrist'):
        image = sample[f'observation.images.{camera}']
        if image.shape != (3, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f'Invalid camera sample: {camera}')
    if sample['observation.state'].shape != (14,) or sample['action'].shape != (14,):
        raise ValueError('Expected 14-D ALOHA state and action')
    print(f'Dataset verified: {len(dataset)} frames', flush=True)
    subprocess.run([
        'lerobot-train', '--policy.path=lerobot/smolvla_base',
        # Null makes LeRobot infer features from the dataset. A dict override
        # merges with the base config and retains its unused camera names.
        '--policy.input_features=null',
        '--policy.device=cuda', '--policy.chunk_size=20',
        '--policy.n_action_steps=8', '--policy.train_expert_only=true',
        '--policy.push_to_hub=false',
        '--dataset.repo_id=local/aloha-dinner-mug', f'--dataset.root={root}',
        '--dataset.video_backend=pyav', '--batch_size=8', '--num_workers=4',
        f'--steps={args.steps}', f'--output_dir={args.output}',
        '--job_name=smolvla_aloha_mug_rocm', '--save_checkpoint=true',
        f'--save_freq={min(args.steps, 1000)}', '--log_freq=20',
        '--wandb.enable=false',
    ], check=True)


if __name__ == '__main__':
    main()
