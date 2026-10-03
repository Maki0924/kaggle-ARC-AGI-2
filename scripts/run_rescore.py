"""Launch scripts/rescore.py in the local Docker image.

    uv run --no-project python scripts/run_rescore.py <run-name> <lora-tag|base> --subset timing20
"""
import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL = pathlib.Path.home() / "arc2-local"

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("lora_tag")
parser.add_argument("--subset", default="timing20")
args = parser.parse_args()

split = json.load(open(ROOT / "splits" / "eval_split.json"))
keys = split[args.subset] if args.subset in split else args.subset.split(",")

cmd = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g",
       "-v", f"{ROOT / 'src' / 'nvarc'}:/work:ro", "-v", f"{ROOT / 'scripts'}:/scripts:ro",
       "-v", f"{ROOT / 'data'}:/data:ro", "-v", f"{LOCAL}:/local",
       "-e", "ARC_MODEL_DIR=/local/models/qwen3_4b_grids15_sft139", "-e", "ARC_DATA_DIR=/data",
       "-e", "HF_HUB_OFFLINE=1", "-e", "UNSLOTH_DISABLE_STATISTICS=1", "-e", "PYTHONUNBUFFERED=1",
       "arc2-nvarc:local", "python", "/scripts/rescore.py", f"/local/runs/{args.run}", f"/local/loras/{args.lora_tag}", args.lora_tag, ",".join(keys)]
raise SystemExit(subprocess.call(cmd))
