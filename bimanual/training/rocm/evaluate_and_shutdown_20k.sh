#!/usr/bin/env bash
set -euo pipefail

workspace=/root/bimanual-training
log="$workspace/logs/evaluate_and_shutdown_20k.log"
result_dir="$workspace/evaluation-results"
exec >>"$log" 2>&1

echo "$(date -Is) waiting for verified 20k uploads"
while [[ ! -f "$workspace/logs/multitask_20k_hub_status.json" ]]; do
  sleep 30
done
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("complete") else 1)' \
  "$workspace/logs/multitask_20k_hub_status.json"

mkdir -p "$result_dir"
docker rm -f evaluate-multitask-20k 2>/dev/null || true
docker run --name evaluate-multitask-20k \
  --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=16g \
  -e MUJOCO_GL=osmesa -e HF_HOME=/training/hf-cache -e PYTHONPATH=/workspace \
  -v "$workspace/eval-src:/workspace" -v "$workspace:/training" -w /workspace \
  bimanual-eval:rocm python eval_aloha_mug_closed_loop.py \
  --smolvla-checkpoint /training/eval-checkpoints/smolvla-mug \
  --compact-checkpoint /training/checkpoints/compact-160m-multitask-20k \
  --pi05-checkpoint /training/checkpoints/pi05-multitask-20k/checkpoints/020000/pretrained_model \
  --episodes 10 --max-policy-steps 240 --device cuda \
  --output /training/evaluation-results/aloha_mug_closed_loop_20k.json

docker rm -f publish-multitask-evaluation-hf 2>/dev/null || true
docker run --name publish-multitask-evaluation-hf \
  -e HF_HOME=/workspace/hf-upload-cache -e PYTHONUNBUFFERED=1 \
  -v "$workspace:/workspace" \
  -v /dev/shm/bimanual-hf-auth/token:/workspace/hf-upload-cache/token:ro \
  bimanual-smolvla:rocm \
  python -m bimanual.training.rocm.publish_multitask_evaluation_remote
echo "$(date -Is) evaluation and upload complete; powering off"
/usr/sbin/poweroff
