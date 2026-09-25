#!/usr/bin/env bash
set -euo pipefail

workspace=/root/bimanual-training
log="$workspace/logs/finalize_multitask_20k.log"
exec >>"$log" 2>&1

echo "$(date -Is) waiting for 20k training containers"
while :; do
  compact_status=$(docker inspect -f '{{.State.Status}}' compact160m-multitask-20k)
  pi_status=$(docker inspect -f '{{.State.Status}}' pi05-multitask-20k)
  if [[ "$compact_status" != running && "$pi_status" != running ]]; then
    break
  fi
  sleep 30
done

compact_exit=$(docker inspect -f '{{.State.ExitCode}}' compact160m-multitask-20k)
pi_exit=$(docker inspect -f '{{.State.ExitCode}}' pi05-multitask-20k)
echo "$(date -Is) training exits compact=$compact_exit pi05=$pi_exit"
if [[ "$compact_exit" != 0 || "$pi_exit" != 0 ]]; then
  exit 1
fi

docker rm -f publish-multitask-20k-hf 2>/dev/null || true
docker run --name publish-multitask-20k-hf \
  -e HF_HOME=/workspace/hf-upload-cache \
  -e PYTHONUNBUFFERED=1 \
  -v "$workspace:/workspace" \
  -v /dev/shm/bimanual-hf-auth/token:/workspace/hf-upload-cache/token:ro \
  bimanual-smolvla:rocm \
  python -m bimanual.training.rocm.publish_multitask_20k_remote
echo "$(date -Is) final uploads complete"
