"""Merge Qwen samples into the fixed NVARC submission and score (docs/A100_PLAN.md, step 6). CPU only.

The merge is the production hybrid5 publish policy (scripts/build_hybrid.py), replayed in (task, output, slot) order
instead of completion order so it is deterministic:
  - start from NVARC's rows; a duplicate attempt_2 is replaced by the runner's distinct fallback
  - a grid equal to either current attempt is skipped
  - no NVARC candidate: first model grid -> attempt_1 (attempt_2 = distinct fallback), next one -> attempt_2
  - NVARC weak (n < 2 or v2 <= 2) and attempt_2 still NVARC's: grid -> attempt_2
  - a grid from a sample that did not finish with "stop" is discarded when NVARC has a candidate
Gain and loss are counted per output (1 / outputs of the task, as Kaggle scores), so they cannot cancel inside a task.
Refuses to score a run that does not cover its whole cohort unless --partial.

    python a100/score.py runs/medium_32k [runs/xhigh_32k ...] [--samples 1] [--partial]
"""
import argparse
import json
import pathlib
import sys
import tempfile
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import runner_parts as rp  # noqa: E402
from hybrid_policy import natural_answer, output_priority, output_stratum

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--fixed", default=str(HERE / "fixed_nvarc"))
ap.add_argument("--challenges", default=str(ROOT / "data" / "arc-agi_evaluation_challenges.json"))
ap.add_argument("--solutions", default=str(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
ap.add_argument("--samples", type=int, default=0, help="use only slots < N (e.g. 1 to read a 2-sample run as 1)")
ap.add_argument("--partial", action="store_true", help="score an incomplete run (not a valid comparison)")
ap.add_argument("--policy", choices=["legacy", "strict"], default="legacy",
                help="legacy reproduces hybrid5; strict requires natural stop for every output (hybrid6)")
ap.add_argument("--weak-vote-limit", type=int, choices=[1, 2], default=2,
                help="exploratory merge gate for outputs with at least two candidates")
ap.add_argument("--limit-outputs", type=int, default=0,
                help="counterfactual: only merge the first N outputs in weak-first order; not a wall-time estimate")
ap.add_argument("--report-dir", help="write reports here/<run name> instead of overwriting historical run reports")
args = ap.parse_args()
if args.samples < 0 or args.limit_outputs < 0:
    ap.error("samples and limit-outputs must be nonnegative")
if (args.policy != "legacy" or args.weak_vote_limit != 2 or args.limit_outputs or args.samples) and not args.report_dir:
    ap.error("changed policies/subsets require --report-dir to preserve the original result")

fixed = pathlib.Path(args.fixed)
challenges = json.load(open(args.challenges))
solutions = json.load(open(args.solutions))
NVARC = json.load(open(fixed / "nvarc_submission.json"))
CONF = {k: tuple(v) for k, v in json.load(open(fixed / "nvarc_confidence.json")).items()}
STRATA = json.load(open(fixed / "cohort_manifest.json"))["strata"]
AGG = rp.load_module_from_source("qwen38_freeform_aggregator", pathlib.Path(tempfile.gettempdir()) / "qwen38_agg_score.py",
                                 rp.AGGREGATOR_SOURCE)


def merge(rows, tasks):
    fallback = AGG.guaranteed_fallback_submission({t: challenges[t] for t in tasks})
    sub = {t: json.loads(json.dumps(NVARC[t])) for t in tasks}
    for t in tasks:
        for i, row in enumerate(sub[t]):
            if row["attempt_1"] == row["attempt_2"]:
                row["attempt_2"] = rp.select_distinct_fallback(row["attempt_1"], fallback[t][i])
    nvarc_rows = json.loads(json.dumps(sub))
    answered, taken = set(), 0
    for r in rows:
        t, i, grid = r["task_id"], r["query_index"], r["grid"]
        n_cand, v1, v2 = CONF.get(f"{t}_{i}", (0, 0, 0))
        if grid is None or (not natural_answer(r) if args.policy == "strict"
                            else r["finish"] != "stop" and n_cand > 0):
            continue
        if args.weak_vote_limit < 2 and n_cand >= 2 and v2 > args.weak_vote_limit:
            continue
        old, nv = sub[t][i], nvarc_rows[t][i]
        if grid in (old["attempt_1"], old["attempt_2"]):
            continue
        if n_cand == 0 or (args.policy == "legacy" and nv["attempt_1"] == [[0]]):
            if (t, i) not in answered:
                new = {"attempt_1": grid, "attempt_2": rp.select_distinct_fallback(grid, fallback[t][i])}
            else:
                new = {"attempt_1": old["attempt_1"], "attempt_2": grid}
        elif (n_cand < 2 or v2 <= 2) and old["attempt_2"] == nv["attempt_2"]:
            new = {"attempt_1": old["attempt_1"], "attempt_2": grid}
        else:
            continue
        sub[t][i] = new
        answered.add((t, i))
        taken += 1
    return sub, nvarc_rows, taken


def hit(sub, t, i):
    return solutions[t][i] in (sub[t][i]["attempt_1"], sub[t][i]["attempt_2"])


def report(run):
    run = pathlib.Path(run)
    man = json.load(open(run / "manifest.json"))
    cond = man["condition"]
    slots = min(args.samples or cond["samples"], cond["samples"])
    expected = {(t, i, s) for t, i in map(tuple, man["fixed"]["outputs"]) for s in range(slots)}
    rows = sorted((r for r in map(json.loads, open(run / "samples.jsonl")) if r["slot"] < slots),
                  key=lambda r: (r["task_id"], r["query_index"], r["slot"]))
    have = {(r["task_id"], r["query_index"], r["slot"]) for r in rows}
    missing = sorted(expected - have)
    if have - expected or len(have) != len(rows):
        sys.exit(f"{run}: duplicate or unexpected sample keys")
    if missing and not args.partial:
        sys.exit(f"{run}: {len(missing)} of {len(expected)} samples missing (e.g. {missing[:3]}); rerun qwen_ab.py "
                 f"or pass --partial")
    tasks = sorted(man["fixed"]["tasks"])
    ordered_outputs = sorted(map(tuple, man["fixed"]["outputs"]),
                             key=lambda o: (*output_priority(CONF[f"{o[0]}_{o[1]}"]), *o))
    selected = set(ordered_outputs[:args.limit_outputs] if args.limit_outputs else ordered_outputs)
    rows = [r for r in rows if (r["task_id"], r["query_index"]) in selected]
    sub, base_sub, taken = merge(rows, tasks)
    gain = loss = 0.0
    gained, lost = set(), set()
    by = defaultdict(float)
    by_output = defaultdict(lambda: {"outputs": 0, "gain": 0.0, "loss": 0.0, "net": 0.0,
                                     "completion_tokens": 0, "sample_wall_s": 0.0})
    for t in tasks:
        for i in range(len(solutions[t])):
            d = hit(sub, t, i) - hit(base_sub, t, i)
            w = d / len(solutions[t])
            gain += max(w, 0)
            loss += max(-w, 0)
            by[STRATA.get(t, "?")] += w
            stat = by_output[output_stratum(CONF.get(f"{t}_{i}", (0, 0, 0)))]
            stat["outputs"] += 1
            stat["gain"] += max(w, 0)
            stat["loss"] += max(-w, 0)
            stat["net"] += w
            (gained if d > 0 else lost if d < 0 else set()).add(t)
    base = sum(hit(base_sub, t, i) / len(solutions[t]) for t in tasks for i in range(len(solutions[t])))
    toks = sum(r["completion_tokens"] for r in rows)
    selected_keys = {f"{t}_{i}_{s}" for t, i in selected for s in range(slots)}
    toks_all = sum(r["completion_tokens"] for r in map(json.loads, open(run / "turns.jsonl"))
                   if r["key"] in selected_keys) if (run / "turns.jsonl").exists() else toks
    fin = defaultdict(int)
    for r in rows:
        fin[r["finish"].split(":")[0]] += 1
        stat = by_output[output_stratum(CONF[f"{r['task_id']}_{r['query_index']}"])]
        stat["completion_tokens"] += r["completion_tokens"]
        stat["sample_wall_s"] += r["wall_s"]
    right = {(r["task_id"], r["query_index"]) for r in rows if r["grid"] is not None and r["grid"] == solutions[r["task_id"]][r["query_index"]]}
    nv_wrong = {(t, i) for t in tasks for i in range(len(solutions[t])) if not hit(base_sub, t, i)}
    out = {
        "run": run.name, "condition": cond, "complete": not missing, "tasks": len(tasks), "samples": len(rows),
        "nvarc": round(base, 2), "merged": round(base + gain - loss, 2),
        "gain": round(gain, 2), "loss": round(loss, 2), "net": round(gain - loss, 2),
        "gained_tasks": sorted(gained), "lost_tasks": sorted(lost),
        "net_by_stratum": {k: round(v, 2) for k, v in sorted(by.items())},
        "net_by_output_stratum": {k: {name: round(value, 3) for name, value in v.items()}
                                  for k, v in sorted(by_output.items())},
        "replay": {"policy": args.policy, "weak_vote_limit": args.weak_vote_limit,
                   "limit_outputs": args.limit_outputs, "selected_outputs": len(selected),
                   "order": "W/O/Z then v2,v1,task_id,query_index",
                   "wall_time_note": "sample walls overlap; their sum is NOT elapsed time; subset speed is unmeasured"},
        "grids_taken": taken,
        "stop_with_grid": sum(1 for r in rows if r["finish"] == "stop" and r["grid"] is not None),
        "finish": dict(sorted(fin.items())),
        "qwen_correct_outputs": len(right), "qwen_correct_where_nvarc_wrong": len(right & nv_wrong),
        "completion_tokens": toks, "completion_tokens_incl_failed_attempts": toks_all,
        "net_per_1M_tokens": round((gain - loss) / toks_all * 1e6, 3) if toks_all else None,
        "mean_turns": round(sum(r["turns"] for r in rows) / max(1, len(rows)), 2),
        "tool_calls": sum(r["tool_calls"] for r in rows),
    }
    destination = pathlib.Path(args.report_dir) / run.name if args.report_dir else run
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "merged_submission.json").write_text(json.dumps(sub))
    (destination / "scores.json").write_text(json.dumps(out, indent=1))
    return out


results = [report(r) for r in args.runs]
cols = ["run", "complete", "tasks", "samples", "nvarc", "merged", "gain", "loss", "net", "net_per_1M_tokens",
        "stop_with_grid", "qwen_correct_where_nvarc_wrong", "completion_tokens_incl_failed_attempts"]
print(" | ".join(cols))
for r in results:
    print(" | ".join(str(r[c]) for c in cols))
for r in results:
    print(f"\n{r['run']}: gained {r['gained_tasks']} lost {r['lost_tasks']} by stratum {r['net_by_stratum']} "
          f"finish {r['finish']}")
