"""Summarize NLL-only TTT runs (truth NLL per test output) against the stock run's solved/unsolved split.

    uv run --no-project --with numpy==2.2.6 python scripts/analyze_nll_sweep.py nll_stock nll_aug8 ...
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

RUNS = pathlib.Path.home() / "arc2-local" / "runs"
sol = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))

# which outputs the stock decode run solved (pass@2), from any run that has pools
pools = defaultdict(dict)
for name in ["t20_stock", "dev60_stock"]:
    for p in (RUNS / name / "inference_outputs").iterdir():
        with bz2.BZ2File(p) as f:
            for i, s in enumerate(pickle.load(f)):
                pools[p.name.split(".")[0]][f"{p.name}.out{i}"] = s
solved = {}
for bk, pool in pools.items():
    k, i = bk.rsplit("_", 1)
    t = np.array(sol[k][int(i)])
    solved[bk] = any(np.array_equal(g, t) for g in score_kgmon(pool)[:2])

per = {}
print(f"{'run':22} {'n':>3} {'solved med':>10} {'unsolved med':>12} {'all mean':>9} {'all med':>8} {'train_loss':>10} {'sec/task':>8}")
for r in sys.argv[1:]:
    f = RUNS / r / "timing_rank0.jsonl"
    if not f.exists():
        print(f"{r:22} missing")
        continue
    rows = [json.loads(l) for l in open(f)]
    rows = [x for x in rows if x.get("status") == "ok" and "truth_nll" in x]
    vals = {bk: float(np.mean(v)) for x in rows for bk, v in x["truth_nll"].items()}
    per[r] = vals
    s = [v for bk, v in vals.items() if solved.get(bk)]
    u = [v for bk, v in vals.items() if bk in solved and not solved[bk]]
    tl = np.mean([x["train_loss"] for x in rows if "train_loss" in x]) if rows else float("nan")
    print(f"{r:22} {len(rows):3d} {np.median(s):10.2f} {np.median(u):12.2f} {np.mean(list(vals.values())):9.2f} {np.median(list(vals.values())):8.2f} {tl:10.4f} {np.mean([x['seconds'] for x in rows]):8.0f}")

base = sys.argv[1]
print(f"\npaired against {base} (negative = truth more likely):")
for r in per:
    if r == base:
        continue
    d = [per[r][bk] - per[base][bk] for bk in per[base] if bk in per[r]]
    du = [per[r][bk] - per[base][bk] for bk in per[base] if bk in per[r] and not solved.get(bk)]
    ds = [per[r][bk] - per[base][bk] for bk in per[base] if bk in per[r] and solved.get(bk)]
    print(f"  {r:22} better {sum(x < 0 for x in d):2d} / worse {sum(x > 0 for x in d):2d}   median {np.median(d):+.2f}   unsolved median {np.median(du):+.2f}   solved median {np.median(ds):+.2f}")
