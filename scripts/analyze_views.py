"""CPU-only: time breakdown and what fewer decode views would have given, from saved runs.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_views.py <run-name> [<run-name> ...]
"""
import bz2
import json
import pathlib
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "nvarc"))
from arc_decoder import score_kgmon  # noqa: E402

solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
pools, timing = defaultdict(dict), {}
for name in sys.argv[1:]:
    run_dir = pathlib.Path.home() / "arc2-local" / "runs" / name
    for line in open(run_dir / "timing_rank0.jsonl"):
        row = json.loads(line)
        if row.get("status") == "ok":
            timing[row["key"]] = row
    for path in (run_dir / "inference_outputs").iterdir():
        with bz2.BZ2File(path) as f:
            pools[path.name.split(".")[0]][path.name] = pickle.load(f)

keys = sorted(timing)
tot = sum(r["seconds"] for r in timing.values())
train = sum(r.get("train_seconds", 0) for r in timing.values())
dfs = sum(d["seconds"] for r in timing.values() for d in r.get("dfs", []))
score = sum(d.get("score_seconds", 0) for r in timing.values() for d in r.get("dfs", []))
calls = sum(d["calls"] for r in timing.values() for d in r.get("dfs", []))
print(f"tasks {len(keys)}  total {tot / 3600:.2f} h  mean {tot / len(keys):.0f} s")
print(f"  train {100 * train / tot:.0f}%  dfs {100 * dfs / tot:.0f}%  rescoring {100 * score / tot:.0f}%  other {100 * (tot - train - dfs - score) / tot:.0f}%")
print(f"  dfs calls {calls}  ->  {1000 * dfs / calls:.1f} ms per call")


def view_kind(subkey):  # e.g. 'abc_0.transpose.rot90.permute0123456789.ex012' -> ('transpose', n_rot)
    ops = subkey.split(".")[1:]
    return ("transpose" in ops, ops.count("rot90"))


def evaluate(select):
    solved = pool_hit = 0.0
    for key in keys:
        n_test = len(solutions[key])
        for i in range(n_test):
            truth = np.array(solutions[key][i])
            views = sorted(pools.get(f"{key}_{i}", {}))
            chosen = select(views)
            flat = {f"{v}.out{j}": s for v in chosen for j, s in enumerate(pools[f"{key}_{i}"][v])}
            ranked = score_kgmon(flat) if flat else []
            hit = [np.array_equal(g, truth) for g in ranked]
            solved += any(hit[:2]) / n_test
            pool_hit += any(hit) / n_test
    return solved, pool_hit


def first_per_geometry(views):
    seen, out = set(), []
    for v in views:
        if view_kind(v) not in seen:
            seen.add(view_kind(v))
            out.append(v)
    return out


variants = {
    "all views (16)": lambda v: v,
    "8 views: one colour permutation per geometry": first_per_geometry,
    "8 views: no transpose": lambda v: [x for x in v if not view_kind(x)[0]],
    "4 views: no transpose, one permutation": lambda v: [x for x in first_per_geometry(v) if not view_kind(x)[0]],
    "4 views: rot 0 and 180, no transpose": lambda v: [x for x in v if not view_kind(x)[0] and view_kind(x)[1] in (0, 2)],
    "2 views: identity geometry": lambda v: [x for x in v if view_kind(x) == (False, 0)],
}
print()
for name, select in variants.items():
    solved, pool_hit = evaluate(select)
    print(f"  {name:48} pass@2 {solved:5.2f} ({100 * solved / len(keys):4.1f}%)   pool {pool_hit:5.2f} ({100 * pool_hit / len(keys):4.1f}%)")
print("\n(views that produced no valid grid are not saved, so counts can be below the nominal number)")
