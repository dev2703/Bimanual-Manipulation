#!/usr/bin/env bash
set -euo pipefail
python -m pip install -c /opt/rocm-constraints.txt 'lerobot[pi]==0.6.1' 'transformers==5.5.4'
python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', local_files_only=True)
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
  --dataset.repo_id=local/aloha-dinner-mug \
  --dataset.root=/workspace/datasets/aloha_mug_train_stable_stats \
  --dataset.video_backend=pyav --batch_size=4 --num_workers=4 \
  --steps=200 --save_freq=200 --log_freq=20 --save_checkpoint=true \
  --output_dir=/workspace/checkpoints/pi05-mug-smoke \
  --job_name=pi05_mug_rocm_smoke --wandb.enable=false
