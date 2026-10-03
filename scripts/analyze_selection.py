"""CPU-only comparison of candidate-selection rules on saved pools.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_selection.py t20_stock dev60_stock
"""
import bz2
import json
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))

pools, keys = defaultdict(dict), set()
for name in sys.argv[1:]:
    run_dir = pathlib.Path.home() / "arc2-local" / "runs" / name
    for line in open(run_dir / "timing_rank0.jsonl"):
        row = json.loads(line)
        if row.get("status") == "ok":
            keys.add(row["key"])
    for path in (run_dir / "inference_outputs").iterdir():
        with bz2.BZ2File(path) as f:
            for i, s in enumerate(pickle.load(f)):
                pools[path.name.split(".")[0]][f"{path.name}.out{i}"] = s
keys = sorted(keys)


def grouped(pool):
    """Unique grids with the list of (beam_score, score_aug list) that produced them."""
    groups = defaultdict(list)
    for s in pool.values():
        groups[tuple(map(tuple, s["solution"]))].append(s)
    return groups


def rule_kgmon(g):
    return len(g) - np.mean([np.mean(s["score_aug"]) for s in g])


def rule_probmul(g, baseline=3):
    return np.sum([baseline - s["beam_score"] for s in g]) + np.mean([np.sum([baseline - x for x in s["score_aug"]]) for s in g])


def rule_votes(g):
    return len(g) - 1e-3 * np.mean([np.mean(s["score_aug"]) for s in g])


def rule_nll(g):
    return -np.mean([np.mean(s["score_aug"]) for s in g])


def rule_nll_min(g):
    return -np.min([np.mean(s["score_aug"]) for s in g])


def rule_nll_per_cell(g):
    grid = g[0]["solution"]
    return -np.mean([np.mean(s["score_aug"]) for s in g]) / (grid.shape[0] * grid.shape[1])


def rule_kgmon_per_cell(g):
    grid = g[0]["solution"]
    return len(g) - np.mean([np.mean(s["score_aug"]) for s in g]) / np.sqrt(grid.shape[0] * grid.shape[1])


def rule_logvotes_nll(g):
    return 2 * np.log1p(len(g)) - np.mean([np.mean(s["score_aug"]) for s in g])


def rule_kgmon_median(g):
    return len(g) - np.mean([np.median(s["score_aug"]) for s in g])


rules = {n[5:]: f for n, f in globals().items() if n.startswith("rule_")}

print(f"{'rule':16} {'pass@1':>7} {'pass@2':>7}   (pool upper bound counted on the same outputs)")
pool_bound = n_outputs = 0.0
results = {}
for name, rule in rules.items():
    p1 = p2 = 0.0
    for key in keys:
        n_test = len(solutions[key])
        for i in range(n_test):
            truth = np.array(solutions[key][i])
            groups = grouped(pools.get(f"{key}_{i}", {}))
            ranked = sorted(groups.items(), key=lambda kv: -rule(kv[1]))
            hit = [np.array_equal(np.array(grid), truth) for grid, _ in ranked]
            p1 += any(hit[:1]) / n_test
            p2 += any(hit[:2]) / n_test
            if name == "kgmon":
                pool_bound += any(hit) / n_test
    results[name] = p2
    print(f"{name:16} {p1:7.2f} {p2:7.2f}")
print(f"{'pool bound':16} {'':7} {pool_bound:7.2f}   over {len(keys)} tasks")

# Where the rules disagree: tasks solved by one rule but not the other.
base = "kgmon"
for name in rules:
    if name == base:
        continue
    won, lost = [], []
    for key in keys:
        n_test = len(solutions[key])
        for i in range(n_test):
            truth = np.array(solutions[key][i])
            groups = grouped(pools.get(f"{key}_{i}", {}))
            a = any(np.array_equal(np.array(g), truth) for g, _ in sorted(groups.items(), key=lambda kv: -rules[base](kv[1]))[:2])
            b = any(np.array_equal(np.array(g), truth) for g, _ in sorted(groups.items(), key=lambda kv: -rules[name](kv[1]))[:2])
            if b and not a:
                won.append(f"{key}_{i}")
            if a and not b:
                lost.append(f"{key}_{i}")
    if won or lost:
        print(f"{name:16} vs {base}: gains {won}  losses {lost}")
