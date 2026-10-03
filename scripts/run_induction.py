"""Run the program-synthesis leg against a local OpenAI-compatible server (vLLM) and score it.

    uv run --no-project python scripts/run_induction.py <run-name> --subset screen8 --model <served-model-name>
"""
import argparse
import json
import pathlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "induction"))
from solver import solve_task  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--subset", default="screen8")
parser.add_argument("--model", required=True)
parser.add_argument("--base-url", default="http://localhost:8000/v1")
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--repairs", type=int, default=1)
parser.add_argument("--max-tokens", type=int, default=1500)
parser.add_argument("--parallel", type=int, default=4)
args = parser.parse_args()

split = json.load(open(ROOT / "splits" / "eval_split.json"))
keys = split[args.subset] if args.subset in split else args.subset.split(",")
tasks = json.load(open(ROOT / "data" / "arc-agi_evaluation_challenges.json"))
solutions = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))
out_dir = pathlib.Path.home() / "arc2-local" / "runs" / args.name
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(vars(args) | {"keys": keys}, indent=1))


def run(key):
    path = out_dir / f"{key}.json"
    if path.exists():
        return key, json.loads(path.read_text())
    try:
        record = solve_task(tasks[key], args.base_url, args.model, args.samples, args.repairs, max_tokens=args.max_tokens)
    except Exception as e:
        record = {"programs": [], "verified": [], "tokens": 0, "seconds": 0, "error": f"{type(e).__name__}: {e}"}
    path.write_text(json.dumps(record))
    return key, record


with ThreadPoolExecutor(args.parallel) as pool:
    results = dict(pool.map(run, keys))

print(f"{'task':10} {'sec':>5} {'progs':>5} {'best':>5} {'verif':>5} {'correct':>7}  note")
total = verified_tasks = false_verified = 0.0
for key in keys:
    r = results[key]
    n_test = len(solutions[key])
    # Verified programs vote on each test output; the two most common outputs are the attempts.
    score = 0.0
    for i in range(n_test):
        votes = Counter(json.dumps(v[i]) for v in r["verified"])
        attempts = [json.loads(g) for g, _ in votes.most_common(2)]
        score += any(a == solutions[key][i] for a in attempts) / n_test
    total += score
    verified_tasks += bool(r["verified"])
    false_verified += bool(r["verified"]) and score == 0
    best = max([p["train_ok"] for p in r["programs"]], default=0)
    print(f"{key:10} {r['seconds']:5.0f} {len(r['programs']):5d} {best:>3}/{len(tasks[key]['train'])} {len(r['verified']):5d} {score:7.2f}  {r.get('error', '')}")

n = len(keys)
print(f"\ntasks {n}  solved {total:.2f} ({100 * total / n:.1f}%)  with a verified program {verified_tasks:.0f}  verified but wrong {false_verified:.0f}")
print(f"mean seconds {sum(r['seconds'] for r in results.values()) / n:.0f}  total tokens {sum(r['tokens'] for r in results.values())}")
