"""Assemble the Kaggle submission notebook from src/nvarc."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nvarc"

import sys

VARIANTS = {
    "baseline": ("koumeimaki/arc2-nvarc-baseline", "ARC2 NVARC baseline", ""),
    "fast1": ("koumeimaki/arc2-nvarc-fast1", "ARC2 NVARC fast1", "ARC_DECODE_BATCH=8 ARC_MERGE_LORA=1 "),
}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "baseline"
KERNEL_ID, TITLE, RUN_ENV = VARIANTS[VARIANT]
OUT = ROOT / "kaggle" / KERNEL_ID.split("/")[1]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def writefile(name):
    return code(f"%%writefile {name}\n" + (SRC / name).read_text(encoding="utf-8"))


cells = [
    {"cell_type": "markdown", "metadata": {}, "source": "NVARC 2025 pipeline (Qwen3-4B + per-task LoRA TTT) with error handling per puzzle, a stable scoring seed and timing logs."},
    code("# 10-minute buffer for writing the submission.\nimport time\nglobal_end_time = time.time() + 12 * 3600 - 600"),
    code("!pip uninstall -y tensorflow"),
    writefile("arc_loader.py"),
    writefile("arc_decoder.py"),
    writefile("arc_solver.py"),
    writefile("starter.py"),
    code(f"!{RUN_ENV}PYTHONHASHSEED=0 UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {{global_end_time}}"),
    code((SRC / "make_submission.py").read_text(encoding="utf-8")),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
        "kaggle": {
            "accelerator": "nvidiaL4",
            "dockerImageVersionId": 31090,
            "isGpuEnabled": True,
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

metadata = {
    "id": KERNEL_ID,
    "title": TITLE,
    "code_file": f"{KERNEL_ID.split('/')[1]}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [],
    "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
    "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
    "machine_shape": "NvidiaL4",
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{KERNEL_ID.split('/')[1]}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8", newline="\n")
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
print("wrote", OUT)
