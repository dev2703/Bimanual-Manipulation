# The droplet's local ROCm image supplies Python 3.12 and ROCm userspace.
ARG BASE_IMAGE=rocm:latest
FROM ${BASE_IMAGE}
RUN python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/rocm7.2
RUN python -c "import importlib.metadata as m; from pathlib import Path; Path('/opt/rocm-constraints.txt').write_text(''.join(f'{p}=={m.version(p)}\n' for p in ('torch', 'torchvision')))"
RUN python -m pip install --no-cache-dir -c /opt/rocm-constraints.txt \
    'lerobot[smolvla,dataset]==0.6.1' 'transformers==5.5.4'
ENV HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
CMD ["bash"]
