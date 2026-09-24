"""Fixed held-out expert-frame action error; this is not a rollout success test."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

from bimanual.policy.compact_vla import CompactVLA, CompactVLAConfig, ByteTokenizer
from bimanual.training.train_compact_vla import batch_inputs
from bimanual.data.audit import assert_disjoint_splits


def main():
    workspace = Path('/workspace')
    assert_disjoint_splits(workspace / 'datasets/aloha_mug_train', workspace / 'datasets/aloha_mug_val')
    dataset = LeRobotDataset('local/aloha-dinner-mug-val',
                            root=workspace / 'datasets/aloha_mug_val', video_backend='pyav')
    indices = torch.linspace(0, len(dataset)-1, 80).long().tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=8, num_workers=4)
    results = {'metric': 'held-out expert-frame first-action MSE in physical units',
               'frames': len(indices), 'rollout_success_evaluated': False}
    for name in ('smolvla-mug-20k', 'compact-160m-2k'):
        try:
            directory = workspace / 'checkpoints' / name
            if name.startswith('smolvla'):
                checkpoint = directory / 'checkpoints/last/pretrained_model'
                model = SmolVLAPolicy.from_pretrained(checkpoint).to('cuda').eval()
                pre = PolicyProcessorPipeline.from_pretrained(
                    checkpoint, config_filename='policy_preprocessor.json', local_files_only=True,
                    overrides={'device_processor': {'device': 'cuda'}})
                post = PolicyProcessorPipeline.from_pretrained(
                    checkpoint, config_filename='policy_postprocessor.json', local_files_only=True,
                    overrides={'device_processor': {'device': 'cuda'}},
                    to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
            else:
                metadata = json.loads((directory / 'config.json').read_text())
                model = CompactVLA(CompactVLAConfig(**metadata['config'])).to('cuda').eval()
                model.load_state_dict(torch.load(directory / 'model.pt', map_location='cuda', weights_only=True))
                tokenizer = ByteTokenizer(model.config.max_language_tokens)
            errors = []
            torch.manual_seed(12345)
            with torch.inference_mode():
                for batch in loader:
                    target = batch['action'].to('cuda').clone()
                    if name.startswith('smolvla'):
                        model.reset()
                        predicted = post(model.select_action(pre(batch)))
                    else:
                        images, state, language, _, _ = batch_inputs(batch, tokenizer, 'cuda')
                        predicted = model.sample_actions(images, state, language).actions[:, 0]
                    if predicted.shape != target.shape or not torch.isfinite(predicted).all():
                        raise ValueError('Invalid predicted action')
                    errors.append((predicted-target).square().cpu())
            error = torch.cat(errors)
            joints = [i for i in range(14) if i not in (6, 13)]
            results[name] = {'joint_mse_rad2': error[:, joints].mean().item(),
                             'gripper_mse_m2': error[:, [6, 13]].mean().item(),
                             'per_dimension_mse': error.mean(0).tolist()}
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            results[name] = {'error': f'{type(exc).__name__}: {exc}'}
    (workspace / 'logs/heldout_metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
