"""Per-task report for a local run: timing, selection result and how much the candidate pool could give.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_run.py <run-name>
"""
import bz2
import json
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
run_dir = pathlib.Path.home() / "arc2-local" / "runs" / sys.argv[1]
sys.path.insert(0, str(ROOT / "src" / "nvarc"))
from arc_decoder import score_full_probmul_3, score_kgmon  # noqa: E402

solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
config = json.load(open(run_dir / "config.json"))

timing = {}
for path in run_dir.glob("timing_rank*.jsonl"):
    for line in open(path):
        row = json.loads(line)
        timing[row["key"]] = row

# base key "<task>_<test index>" -> {subkey.outN: candidate}
pools = defaultdict(dict)
for path in (run_dir / "inference_outputs").iterdir():
    with bz2.BZ2File(path) as f:
        for i, sample in enumerate(pickle.load(f)):
            pools[path.name.split(".")[0]][f"{path.name}.out{i}"] = sample

totals = defaultdict(float)
print(f"{'task':10} {'sec':>6} {'train':>6} {'uniq':>5} {'kgmon':>5} {'probm':>5} {'pool':>4} {'rank':>4}  status")
for key in config["keys"]:
    n_test = len(solutions[key])
    row = timing.get(key, {})
    for i in range(n_test):
        truth = np.array(solutions[key][i])
        pool = pools.get(f"{key}_{i}", {})
        ranked = score_kgmon(pool) if pool else []
        ranked_pm = score_full_probmul_3(pool) if pool else []
        hit = [np.array_equal(g, truth) for g in ranked]
        hit_pm = [np.array_equal(g, truth) for g in ranked_pm]
        rank = hit.index(True) + 1 if any(hit) else 0
        totals["kgmon"] += any(hit[:2]) / n_test
        totals["probmul"] += any(hit_pm[:2]) / n_test
        totals["either"] += (any(hit[:2]) or any(hit_pm[:2])) / n_test
        totals["pool"] += any(hit) / n_test
        totals["top10"] += any(hit[:10]) / n_test
        totals["empty"] += (not pool) / n_test
        print(f"{key}_{i:<1} {row.get('seconds', 0):6.0f} {row.get('train_seconds', 0):6.0f} {len(ranked):5d} "
              f"{int(any(hit[:2])):5d} {int(any(hit_pm[:2])):5d} {int(any(hit)):4d} {rank:4d}  "
              f"{row.get('status', 'not-run')}{' timeout' if row.get('timeout') else ''}")

n = len(config["keys"])
seconds = [timing[k]["seconds"] for k in config["keys"] if k in timing]
print()
print(f"tasks: {n}  run: {len(seconds)}")
for name in ["kgmon", "probmul", "either", "top10", "pool", "empty"]:
    print(f"  {name:8} {totals[name]:6.2f}  ({100 * totals[name] / n:5.1f}%)")
if seconds:
    print(f"  seconds per task: mean {np.mean(seconds):.0f}  median {np.median(seconds):.0f}  max {np.max(seconds):.0f}  total {np.sum(seconds) / 3600:.2f} h")
