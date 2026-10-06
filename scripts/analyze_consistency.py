"""Evaluate the consistency signal (scripts/consistency.py) as a candidate-selection rule.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_consistency.py <run-name> <tag>
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
tag = sys.argv[2]
solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
cons = pickle.load(open(run / f"consistency_{tag}.pkl", "rb"))

pools = defaultdict(dict)
for p in (run / "inference_outputs").iterdir():
    with bz2.BZ2File(p) as f:
        for i, s in enumerate(pickle.load(f)):
            pools[p.name.split(".")[0]][f"{p.name}.out{i}"] = s

rows = {}
for bk, entry in cons.items():
    groups = defaultdict(list)
    for s in pools[bk].values():
        groups[tuple(map(tuple, s["solution"]))].append(s)
    cands = {}
    for g, with_nll in entry["candidates"].items():
        ss = groups[g]
        r = float(np.mean(np.array(entry["without"]) - np.array(with_nll))) if entry["without"] else 0.0
        cands[g] = {"votes": len(ss), "own": float(np.mean([np.mean(s["score_aug"]) for s in ss])), "r": r,
                    "with": float(np.mean(with_nll)) if with_nll else 0.0}
    rows[bk] = cands


def evaluate(rule, label):
    p1 = p2 = 0.0
    n_tasks = len({bk.rsplit("_", 1)[0] for bk in rows})
    for bk, cands in rows.items():
        key, i = bk.rsplit("_", 1)
        n = len(solutions[key])
        truth = np.array(solutions[key][int(i)])
        ranked = sorted(cands.items(), key=lambda kv: -rule(kv[1]))
        hit = [np.array_equal(np.array(g), truth) for g, _ in ranked]
        p1 += any(hit[:1]) / n
        p2 += any(hit[:2]) / n
    print(f"  {label:40} pass@1 {p1:5.2f}  pass@2 {p2:5.2f}   ({n_tasks} tasks, top-{max(len(c) for c in rows.values())} candidates)")


print(f"run {sys.argv[1]}  tag {tag}  outputs {len(rows)}")
evaluate(lambda c: c["votes"] - c["own"], "stock (votes - own NLL)")
evaluate(lambda c: c["r"], "consistency r only")
evaluate(lambda c: -c["with"], "NLL of held-out pairs with candidate")
for lam in [0.5, 1, 2, 5]:
    evaluate(lambda c, lam=lam: c["votes"] - c["own"] + lam * c["r"], f"stock + {lam} * r")
evaluate(lambda c: -c["own"] + c["r"], "-own NLL + r")

# Does r separate correct from wrong candidates at all?
corr, wrong = [], []
for bk, cands in rows.items():
    key, i = bk.rsplit("_", 1)
    truth = np.array(solutions[key][int(i)])
    for g, c in cands.items():
        (corr if np.array_equal(np.array(g), truth) else wrong).append(c["r"])
print(f"\nr for correct candidates: n={len(corr)} median {np.median(corr) if corr else float('nan'):.2f}   wrong: n={len(wrong)} median {np.median(wrong):.2f}")
# rank of the correct candidate by r within each output that has one
ranks = []
for bk, cands in rows.items():
    key, i = bk.rsplit("_", 1)
    truth = np.array(solutions[key][int(i)])
    order = [g for g, _ in sorted(cands.items(), key=lambda kv: -kv[1]["r"])]
    for pos, g in enumerate(order, 1):
        if np.array_equal(np.array(g), truth):
            ranks.append(pos)
print("rank of correct candidate by r:", sorted(ranks))
