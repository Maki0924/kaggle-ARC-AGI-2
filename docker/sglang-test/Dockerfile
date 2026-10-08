# Stand-in for the Kaggle NVARC image (Ubuntu 22.04, CUDA 12.8 toolkit, Python 3.11) to test the offline SGLang install
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
