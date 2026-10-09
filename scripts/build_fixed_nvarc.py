"""Build the fixed NVARC inputs for the A100 Qwen comparison (docs/A100_PLAN.md, step 1).

Reads the saved dev80 candidate pools (stock 16-view runs on the 4090), rebuilds NVARC's submission and confidence
exactly as the hybrid handoff cell does, checks that dev80 pass@2 reproduces 24.00, and draws the explore / confirm
cohorts by stratum without looking at the answers.

    uv run --no-project --with numpy==2.2.6 python scripts/build_fixed_nvarc.py
Output: a100/fixed_nvarc/{nvarc_submission,nvarc_confidence,cohort_manifest,qwen_eligible_explore,qwen_eligible_confirm}.json
"""
import bz2
import json
import pathlib
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "nvarc"))
from arc_decoder import score_kgmon  # noqa: E402

RUNS = pathlib.Path.home() / "arc2-local" / "runs"
OUT = ROOT / "a100" / "fixed_nvarc"
SEED = 20261010

split = json.load(open(ROOT / "splits" / "eval_split.json"))
challenges = json.load(open(ROOT / "data" / "arc-agi_evaluation_challenges.json"))

pools = defaultdict(dict)
for name in ["t20_stock", "dev60_stock"]:
    for p in (RUNS / name / "inference_outputs").iterdir():
        with bz2.BZ2File(p) as f:
            for i, s in enumerate(pickle.load(f)):
                pools[p.name.split(".")[0]][f"{name}:{p.name}.out{i}"] = s

submission, confidence = {}, {}
for task in split["dev"]:
    rows = []
    for i in range(len(challenges[task]["test"])):
        pool = pools.get(f"{task}_{i}", {})
        votes = defaultdict(int)
        for s in pool.values():
            votes[tuple(map(tuple, s["solution"]))] += 1
        ranked = score_kgmon(pool) if pool else []
        v = [votes[tuple(map(tuple, g))] for g in ranked]
        confidence[f"{task}_{i}"] = [len(ranked), v[0] if v else 0, v[1] if len(v) > 1 else 0]
        guesses = [g.tolist() for g in ranked[:2]] + [[[0]]] * 2
        rows.append({"attempt_1": guesses[0], "attempt_2": guesses[1]})
    submission[task] = rows

# sanity: dev80 pass@2 must reproduce the stock 24.00 (answers are used only here)
solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
score = sum(1 / len(solutions[t]) for t in split["dev"] for i, r in enumerate(solutions[t])
            if r in (submission[t][i]["attempt_1"], submission[t][i]["attempt_2"]))
print(f"dev80 NVARC pass@2 reproduced: {score:.2f} / 80 (expected 24.00)")
assert abs(score - 24.0) < 1e-6, "pools do not reproduce the stock run"


def stratum(task):
    """Z: some output without a candidate, O: some output with one candidate, W: some weak attempt_2, else None."""
    rows = [confidence[f"{task}_{i}"] for i in range(len(challenges[task]["test"]))]
    if any(n == 0 for n, _, _ in rows):
        return "Z"
    if any(n == 1 for n, _, _ in rows):
        return "O"
    if any(v2 <= 2 for _, _, v2 in rows):
        return "W"
    return None


def prompt_size(task):
    t = challenges[task]
    return sum(len(g) * len(g[0]) for ex in t["train"] for g in ex.values()) + sum(len(ex["input"]) * len(ex["input"][0]) for ex in t["test"])


pool_tasks = [t for t in split["dev"] if t not in split["screen8"]]
by_stratum = defaultdict(list)
for t in pool_tasks:
    s = stratum(t)
    if s:
        by_stratum[s].append(t)
print("eligible tasks by stratum (dev80 minus screen8):", {k: len(v) for k, v in sorted(by_stratum.items())},
      "| not eligible:", sum(1 for t in pool_tasks if stratum(t) is None))

# confirm 12 first (4 per stratum where possible), then explore up to 8 / 10 / 18; prompt sizes spread within a stratum
rng = random.Random(SEED)
targets_confirm = {"Z": 3, "O": 3, "W": 6}
targets_explore = {"Z": 8, "O": 10, "W": 18}
confirm, explore = [], []
for s in ["Z", "O", "W"]:
    tasks = sorted(by_stratum[s], key=prompt_size)
    rng.shuffle(tasks)
    if s == "W":  # Z and O are scarce: W fills the explore set up to 36 tasks
        targets_explore["W"] = 36 - len(explore)
    c = tasks[:targets_confirm[s]]
    e = tasks[targets_confirm[s]:targets_confirm[s] + targets_explore[s]]
    confirm += c
    explore += e
leftover = [t for s in by_stratum for t in by_stratum[s] if t not in confirm + explore]


def eligible_outputs(tasks):
    out = []
    for t in tasks:
        for i in range(len(challenges[t]["test"])):
            n, v1, v2 = confidence[f"{t}_{i}"]
            if n == 0 or n < 2 or v2 <= 2:
                out.append([t, i])
    return out


OUT.mkdir(parents=True, exist_ok=True)
manifest = {
    "seed": SEED, "source_runs": ["t20_stock", "dev60_stock"], "nvarc": "stock 16 views, score_kgmon",
    "explore": sorted(explore), "confirm": sorted(confirm), "leftover": sorted(leftover),
    "strata": {t: stratum(t) for t in explore + confirm},
    "explore_outputs": len(eligible_outputs(explore)), "confirm_outputs": len(eligible_outputs(confirm)),
}
(OUT / "nvarc_submission.json").write_text(json.dumps(submission))
(OUT / "nvarc_confidence.json").write_text(json.dumps(confidence))
(OUT / "cohort_manifest.json").write_text(json.dumps(manifest, indent=1))
(OUT / "qwen_eligible_explore.json").write_text(json.dumps(eligible_outputs(explore)))
(OUT / "qwen_eligible_confirm.json").write_text(json.dumps(eligible_outputs(confirm)))
print(f"explore: {len(explore)} tasks / {manifest['explore_outputs']} Qwen outputs | confirm: {len(confirm)} tasks / {manifest['confirm_outputs']} outputs | leftover {len(leftover)}")
print("strata explore:", {s: sum(1 for t in explore if stratum(t) == s) for s in 'ZOW'}, "confirm:", {s: sum(1 for t in confirm if stratum(t) == s) for s in 'ZOW'})
