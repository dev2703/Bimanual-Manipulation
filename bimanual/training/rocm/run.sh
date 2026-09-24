#!/usr/bin/env bash
# Run on the droplet after copying train.py and datasets into WORKSPACE.
set -euo pipefail
workspace=${WORKSPACE:-/root/bimanual-training}
steps=${1:-200}
name=${2:-smolvla-mug-smoke}
docker run -d --name "$name" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size=16g \
  -v "$workspace:/workspace" \
  bimanual-smolvla:rocm \
  python /workspace/train.py --steps "$steps" \
  --output "/workspace/checkpoints/$name"
