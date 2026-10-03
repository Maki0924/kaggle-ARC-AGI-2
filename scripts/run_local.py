"""Run the NVARC pipeline on the local GPU inside the arc2-nvarc:local image.

    uv run --no-project python scripts/run_local.py <run-name> --subset timing20 --hours 6
"""
import argparse
import hashlib
import json
import shutil
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL = pathlib.Path.home() / "arc2-local"  # models and run outputs stay out of OneDrive

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--subset", default="timing20", help="key in splits/eval_split.json, or comma separated task ids")
parser.add_argument("--hours", type=float, default=12.0)
parser.add_argument("--env", action="append", default=[], help="extra ARC_* settings, e.g. ARC_N_TRAIN_AUG=8")
parser.add_argument("--no-lora-cache", action="store_true", help="do not save or reuse per-puzzle LoRAs")
args = parser.parse_args()

split = json.load(open(ROOT / "splits" / "eval_split.json"))
keys = split[args.subset] if args.subset in split else args.subset.split(",")

run_dir = LOCAL / "runs" / args.name
run_dir.mkdir(parents=True, exist_ok=True)

# The job runs from its own copy of the sources, so editing src/ while it runs cannot change it.
src_dir = run_dir / "src"
if src_dir.exists():
    shutil.rmtree(src_dir)
shutil.copytree(ROOT / "src" / "nvarc", src_dir, ignore=shutil.ignore_patterns("__pycache__"))

# LoRAs are only reusable under identical test-time-training settings.
TTT_KEYS = ["ARC_MODEL_DIR", "ARC_N_TRAIN_AUG", "ARC_TTT_SEED", "ARC_LR", "ARC_EPOCHS"]
extra = dict(e.split("=", 1) for e in args.env)
ttt = {k: extra[k] for k in TTT_KEYS if k in extra}
lora_tag = "stock" if not ttt else "ttt-" + hashlib.sha1(json.dumps(ttt, sort_keys=True).encode()).hexdigest()[:10]

(run_dir / "config.json").write_text(json.dumps({"subset": args.subset, "keys": keys, "env": args.env, "hours": args.hours, "lora_tag": lora_tag, "ttt": ttt}, indent=1))

env = {
    "ARC_MODEL_DIR": "/local/models/qwen3_4b_grids15_sft139",
    "ARC_DATA_DIR": "/data",
    "ARC_WORK_DIR": f"/local/runs/{args.name}",
    "ARC_NUM_GPUS": "1",
    "ARC_TASKS": ",".join(keys),
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "OMP_NUM_THREADS": "12",
    "HF_HUB_OFFLINE": "1",
}
if not args.no_lora_cache:
    env["ARC_LORA_DIR"] = f"/local/loras/{lora_tag}"
env.update(extra)

end_time = time.time() + args.hours * 3600
cmd = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g",
       "-v", f"{src_dir}:/work:ro",
       "-v", f"{ROOT / 'data'}:/data:ro",
       "-v", f"{LOCAL}:/local"]
for k, v in env.items():
    cmd += ["-e", f"{k}={v}"]
cmd += ["arc2-nvarc:local", "bash", "-c",
        f"cd /work && python starter.py --end-time {end_time} 2>&1 | tee /local/runs/{args.name}/log.txt && cd /local/runs/{args.name} && PYTHONPATH=/work python /work/make_submission.py 2>&1 | tee summary.txt"]
raise SystemExit(subprocess.call(cmd))
