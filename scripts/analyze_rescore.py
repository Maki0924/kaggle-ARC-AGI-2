"""Combine the solver's own candidate scores with re-scores from other adapters and compare selection rules.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_rescore.py t20_stock ttt-7638d7cfa9 base
"""
import bz2
import json
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
run = pathlib.Path.home() / "arc2-local" / "runs" / sys.argv[1]
tags = sys.argv[2:]
solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
config = json.load(open(run / "config.json"))

pools = defaultdict(dict)
for p in (run / "inference_outputs").iterdir():
    with bz2.BZ2File(p) as f:
        for i, s in enumerate(pickle.load(f)):
            pools[p.name.split(".")[0]][f"{p.name}.out{i}"] = s
rescored = {t: pickle.load(open(run / f"rescore_{t}.pkl", "rb")) for t in tags}

# candidate -> votes, own NLL (8 views), extra NLLs per tag
rows = {}
for key in config["keys"]:
    for i in range(len(solutions[key])):
        bk = f"{key}_{i}"
        groups = defaultdict(list)
        for s in pools.get(bk, {}).values():
            groups[tuple(map(tuple, s["solution"]))].append(s)
        cands = {}
        for g, ss in groups.items():
            extra = {t: rescored[t][bk]["candidates"][g] for t in tags if bk in rescored[t] and g in rescored[t][bk]["candidates"]}
            cands[g] = {"votes": len(ss), "own": float(np.mean([np.mean(s["score_aug"]) for s in ss])), **{t: float(np.mean(v)) for t, v in extra.items()}}
        rows[bk] = cands


def evaluate(rule, label):
    p1 = p2 = 0.0
    for key in config["keys"]:
        n = len(solutions[key])
        for i in range(n):
            bk = f"{key}_{i}"
            truth = np.array(solutions[key][i])
            ranked = sorted(rows[bk].items(), key=lambda kv: -rule(kv[1]))
            hit = [np.array_equal(np.array(g), truth) for g, _ in ranked]
            p1 += any(hit[:1]) / n
            p2 += any(hit[:2]) / n
    print(f"  {label:44} pass@1 {p1:5.2f}  pass@2 {p2:5.2f}")


def mean_nll(c, use):
    vals = [c[t] for t in use if t in c]
    return float(np.mean(vals)) if vals else c["own"]


print(f"tasks {len(config['keys'])}  rescored with: {tags}")
evaluate(lambda c: c["votes"] - c["own"], "stock: votes - own NLL")
evaluate(lambda c: -c["own"], "own NLL only")
for t in tags:
    evaluate(lambda c, t=t: c["votes"] - mean_nll(c, [t]), f"votes - NLL[{t}]")
    evaluate(lambda c, t=t: c["votes"] - mean_nll(c, ['own', t]) if False else c["votes"] - (c["own"] + mean_nll(c, [t])) / 2, f"votes - mean(own, {t})")
    evaluate(lambda c, t=t: -(c["own"] + mean_nll(c, [t])) / 2, f"mean(own, {t}) NLL only")
if len(tags) > 1:
    evaluate(lambda c: c["votes"] - (c["own"] + sum(mean_nll(c, [t]) for t in tags)) / (1 + len(tags)), "votes - mean(own, all tags)")

# Truth NLL as a TTT-quality signal: does it separate solved from unsolved outputs?
print("\nNLL of the true output (mean over 8 views), by whether stock pass@2 solved the output:")
for t in tags:
    solved, unsolved = [], []
    for key in config["keys"]:
        for i in range(len(solutions[key])):
            bk = f"{key}_{i}"
            if bk not in rescored[t]:
                continue
            truth = np.array(solutions[key][i])
            ranked = sorted(rows[bk].items(), key=lambda kv: -(kv[1]["votes"] - kv[1]["own"]))
            hit = any(np.array_equal(np.array(g), truth) for g, _ in ranked[:2])
            (solved if hit else unsolved).append(np.mean(rescored[t][bk]["truth"]))
    print(f"  {t:16} solved: median {np.median(solved):7.2f} (n={len(solved)})   unsolved: median {np.median(unsolved):7.2f} (n={len(unsolved)})")
