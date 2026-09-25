#!/usr/bin/env bash
# Full 20k-step Pi0.5 run on the merged mug/plate/drawer/block dataset.
set -euo pipefail
python -m pip install -c /opt/rocm-constraints.txt 'lerobot[pi]==0.6.1' 'transformers==5.5.4'
python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
tokenizer_dir = snapshot_download('google/paligemma-3b-pt-224')
AutoTokenizer.from_pretrained(tokenizer_dir)
snapshot_download('lerobot/pi05_base')
print('Pi0.5 weights and tokenizer are available on droplet disk', flush=True)
PY
lerobot-train \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.chunk_size=20 --policy.n_action_steps=8 \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true --policy.push_to_hub=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --dataset.repo_id=local/aloha-multitask \
  --dataset.root=/workspace/datasets/aloha_multitask_train \
  --dataset.video_backend=pyav --batch_size=4 --num_workers=4 \
  --steps=20000 --save_freq=5000 --log_freq=100 --save_checkpoint=true \
  --output_dir=/workspace/checkpoints/pi05-multitask-20k \
  --job_name=pi05_multitask_rocm_20k --wandb.enable=false
