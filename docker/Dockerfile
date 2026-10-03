# Local stand-in for the Kaggle image + sorokin/pip-install-unsloth-flash-patch
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends patch git && rm -rf /var/lib/apt/lists/*

RUN python --version && echo "torch==2.8.0" > /constraints.txt \
 && pip install --no-cache-dir -c /constraints.txt unsloth==2025.9.7 unsloth_zoo==2025.9.9 numpy==2.2.6 \
 && pip install --no-cache-dir --no-deps https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-linux_x86_64.whl

COPY qwen3.patch /qwen3.patch
RUN patch --binary "$(python -c 'import importlib.util as u; print(u.find_spec("unsloth").submodule_search_locations[0])')/models/qwen3.py" /qwen3.patch

# torchao pulled in by the resolver targets a newer torch and breaks peft imports
RUN pip uninstall -y torchao

RUN python -c "import torch, flash_attn, peft, transformers; print(torch.__version__, flash_attn.__version__, peft.__version__, transformers.__version__)"
# triton compiles a small C helper at first use
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
ENV CC=gcc
WORKDIR /work
