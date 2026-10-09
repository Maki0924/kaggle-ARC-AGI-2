"""Qwen-only A/B runner for the A100 comparison (docs/A100_PLAN.md, steps 5-7).

Runs the production agent loop (same prompt, tool, parser and seeds as hybrid5; see runner_parts.py) on the fixed
NVARC-eligible outputs, with one change of rules: each sample gets a TOTAL completion-token budget summed over all
turns, and runs until it answers or the budget / context is used up. There is no wall-clock deadline, no forced-final
turn, no truncation recovery and no context-error halving; the turn limit is a high safety stop, not a budget.
Each request's max_tokens is min(budget left, context left), with the prompt counted by the server's own tokenizer
(/tokenize). Servers are started separately (serve.sh writes servers.json); this script only talks to them.

    python a100/qwen_ab.py --cohort explore --effort medium --budget 32768 --out runs/medium_32k \
        --servers server_logs/servers.json --concurrency 8

Files in --out: manifest.json (condition + input/server fingerprints, fixed at the first start and checked on every
resume), samples.jsonl (finished samples only), failures.jsonl (infrastructure failures; retried on the next start),
turns.jsonl (every request, including failed attempts), raw/<task>_<output>_<slot>.json (full conversation).
Score with a100/score.py.
"""
import argparse
import concurrent.futures as cf
import datetime
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner_parts as rp  # noqa: E402

ROOT = HERE.parent
MODEL_NAME = rp.RUNTIME_CONFIG["model"]["served_name"]
SETTINGS = rp.RUNTIME_CONFIG["sampling"]

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", required=True, help="explore | confirm | screen | tid:qi,tid:qi,...")
ap.add_argument("--effort", required=True, choices=["low", "medium", "high", "xhigh"])
ap.add_argument("--budget", type=int, required=True, help="total completion tokens per sample, all turns")
ap.add_argument("--samples", type=int, default=1, help="independent samples per output (slots 0..n-1)")
ap.add_argument("--tools", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--max-turns", type=int, default=64, help="safety stop only; the token budget is the real limit")
ap.add_argument("--context-len", type=int, default=131072)
ap.add_argument("--servers", required=True, help="servers.json written by serve.sh")
ap.add_argument("--concurrency", type=int, default=8, help="in-flight samples in total (spread over servers)")
ap.add_argument("--tool-concurrency", type=int, default=8)
ap.add_argument("--fixed", default=str(HERE / "fixed_nvarc"))
ap.add_argument("--challenges", default=str(ROOT / "data" / "arc-agi_evaluation_challenges.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0, help="run only the first N pending samples (smoke tests)")
ap.add_argument("--stop-launch-at", type=float, default=0, help="unix time after which no new sample starts (left pending)")
ap.add_argument("--allow-visible-solutions", action="store_true", help="smoke tests only")
args = ap.parse_args()

# the model's Python tool can read files: no answer file may be reachable from the inference side
visible = [p for d in {str(ROOT), str(pathlib.Path(args.challenges).resolve().parent), "/kaggle/input"}
           for p in glob.glob(os.path.join(d, "**", "*solution*.json"), recursive=True)]
if visible and not args.allow_visible_solutions:
    sys.exit(f"answer files are visible to the inference side, move them away first: {visible[:3]}")

OUT = pathlib.Path(args.out)
(OUT / "raw").mkdir(parents=True, exist_ok=True)
challenges = json.load(open(args.challenges))
fixed = pathlib.Path(args.fixed)
if args.cohort in ("explore", "confirm"):
    outputs = [tuple(x) for x in json.load(open(fixed / f"qwen_eligible_{args.cohort}.json"))]
    tasks = json.load(open(fixed / "cohort_manifest.json"))[args.cohort]
elif args.cohort == "screen":
    split = json.load(open(HERE / "screen8.json"))
    conf = json.load(open(fixed / "nvarc_confidence.json"))
    outputs = [(t, i) for t in split for i in range(len(challenges[t]["test"]))
               if conf[f"{t}_{i}"][0] < 2 or conf[f"{t}_{i}"][2] <= 2]
    tasks = split
else:
    outputs = [(x.split(":")[0], int(x.split(":")[1])) for x in args.cohort.split(",")]
    tasks = sorted({t for t, _ in outputs})
work = [(t, i, s) for s in range(args.samples) for t, i in outputs]
servers = json.load(open(args.servers))
URLS = [s["url"] for s in servers["servers"]]


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else json.dumps(b, sort_keys=True).encode()).hexdigest()[:16]


def post(url, path, body, timeout):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


models = {}
for u in URLS:
    with urllib.request.urlopen(u + "/v1/models", timeout=30) as r:
        m = json.loads(r.read())
    assert any(d.get("id") == MODEL_NAME for d in m["data"]), f"{u} does not serve {MODEL_NAME}"
    models[u] = [{k: d.get(k) for k in ("id", "root", "max_model_len")} for d in m["data"]]
assert len({json.dumps(v, sort_keys=True) for v in models.values()}) == 1, f"servers differ: {models}"

# everything that defines the experiment; a resume must match it exactly
FIXED = {
    "condition": {k: getattr(args, k) for k in ["cohort", "effort", "budget", "samples", "tools", "max_turns", "context_len"]},
    "outputs": [list(o) for o in outputs], "tasks": sorted(tasks),
    "inputs": {"challenges": sha({t: challenges[t] for t in tasks}), "runner_parts": sha((HERE / "runner_parts.py").read_bytes()),
               "qwen_ab": sha(pathlib.Path(__file__).read_bytes()),
               "nvarc_submission": sha((fixed / "nvarc_submission.json").read_bytes()),
               "nvarc_confidence": sha((fixed / "nvarc_confidence.json").read_bytes())},
    "server": {k: servers[k] for k in servers if k != "servers"}, "model": next(iter(models.values())),
}
manifest_path = OUT / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"fixed": FIXED, "starts": []}
if manifest["fixed"] != FIXED:
    diff = [k for k in FIXED if manifest["fixed"].get(k) != FIXED[k]]
    sys.exit(f"{OUT} was started with a different setup ({diff}); use a new --out")
manifest["condition"] = FIXED["condition"]

done = set()
if (OUT / "samples.jsonl").exists():
    for line in open(OUT / "samples.jsonl"):
        r = json.loads(line)
        done.add((r["task_id"], r["query_index"], r["slot"]))
todo = [w for w in work if w not in done]
if args.limit:
    todo = todo[:args.limit]

TMP = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
SANDBOX = rp.load_module_from_source("qwen38_safe_python_tool", TMP / "qwen38_safe_python_tool.py", rp.SANDBOX_SOURCE) \
    if args.tools else None
AGG = rp.load_module_from_source("qwen38_freeform_aggregator", TMP / "qwen38_freeform_aggregator.py", rp.AGGREGATOR_SOURCE)
TOOL_SLOTS = threading.BoundedSemaphore(args.tool_concurrency)
WRITE_LOCK = threading.Lock()


def chat_kwargs():
    return {"enable_thinking": SETTINGS["enable_thinking"], "reasoning_effort": args.effort}


def count_prompt(url, messages):
    """Exact prompt length from the server's tokenizer and chat template (tools included)."""
    body = {"model": MODEL_NAME, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": chat_kwargs()}
    if args.tools:
        body["tools"] = [SANDBOX.TOOL_SCHEMA]
    return int(post(url, "/tokenize", body, 120)["count"])


def valid_grid(grid):
    if grid is None:
        return None
    try:
        AGG.validate_grid(grid, context="a100 model grid")
        return grid
    except Exception:
        return None


def append(name, row):
    with WRITE_LOCK, open(OUT / name, "a") as f:
        f.write(json.dumps(row) + "\n")


class InfraError(Exception):
    pass


def solve(tid, qi, slot, url):
    test_input = challenges[tid]["test"][qi]["input"]
    task = {"train": challenges[tid]["train"], "test_input": test_input}
    prompt = rp.build_agent_prompt(task) if args.tools else rp.build_prompt(task)
    messages = [{"role": "user", "content": prompt}]
    context = {"train": task["train"], "test_input": test_input}
    seed = rp.stable_sample_seed(tid, qi, slot)
    key, attempt = f"{tid}_{qi}_{slot}", uuid.uuid4().hex[:8]
    used = turns = tool_calls = tool_errors = 0
    tool_s = 0.0
    finish, final, grid_any = "max_turns", "", None
    raw = []
    t0 = time.time()
    for turn in range(args.max_turns):
        try:
            prompt_tokens = count_prompt(url, messages)
        except Exception as exc:
            raise InfraError(f"tokenize: {type(exc).__name__}: {str(exc)[:200]}") from exc
        if args.budget - used <= 0:
            finish = "budget_exhausted"
            break
        max_tokens = min(args.budget - used, args.context_len - prompt_tokens)
        if max_tokens <= 0:
            finish = "context_exhausted"
            break
        body = {"model": MODEL_NAME, "messages": messages, "temperature": SETTINGS["temperature"],
                "top_p": SETTINGS["top_p"], "top_k": SETTINGS["top_k"], "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": chat_kwargs()}
        if args.tools:
            body["tools"] = [SANDBOX.TOOL_SCHEMA]
            body["tool_choice"] = "auto"
        ts = time.time()
        try:
            p = post(url, "/v1/chat/completions", body, 4 * 3600)
        except urllib.error.HTTPError as exc:
            raise InfraError("http%d: %s" % (exc.code, exc.read().decode(errors="replace")[:300])) from exc
        except Exception as exc:
            raise InfraError(f"chat: {type(exc).__name__}: {str(exc)[:200]}") from exc
        turns += 1
        ch = p["choices"][0]
        msg = ch["message"]
        usage = p.get("usage") or {}
        n_out = usage.get("completion_tokens") or 0
        used += n_out
        content = msg.get("content") if isinstance(msg.get("content"), str) else ""
        reasoning = rp.response_reasoning(msg)
        tcs = msg.get("tool_calls") or []
        turn_finish = ch.get("finish_reason")
        g = valid_grid(rp.parse_model_grid(content, require_closed_fence=turn_finish == "length"))
        grid_any = g or grid_any
        append("turns.jsonl", {"key": key, "attempt": attempt, "turn": turn, "prompt_tokens": prompt_tokens,
                               "usage_prompt_tokens": usage.get("prompt_tokens"), "max_tokens": max_tokens,
                               "completion_tokens": n_out, "finish_reason": turn_finish, "tool_calls": len(tcs),
                               "reasoning_chars": len(reasoning), "content_chars": len(content),
                               "wall_s": round(time.time() - ts, 1)})
        raw.append({"turn": turn, "finish_reason": turn_finish, "message": msg, "usage": usage})
        if not tcs and ("<tool_call>" in reasoning or "<tool_call>" in content):
            finish, final = "protocol_error:tool_call_markup", content
            break
        if tcs and args.tools and turn_finish != "length":
            messages.append(rp.assistant_replay_message(msg))
            for tc in tcs:
                fn = tc.get("function") or {}
                code = rp.decode_tool_code(fn) if fn.get("name") == "run_python" else ""
                tool_calls += 1
                if not code:
                    out = "ERROR: run_python arguments did not contain non-empty code" if fn.get("name") == "run_python" \
                        else "error: unknown tool %r" % fn.get("name")
                    tool_errors += 1
                else:
                    with TOOL_SLOTS:
                        res = SANDBOX.run_python(code, context)
                    tool_s += res.wall_s or 0
                    out = res.stdout if res.ok else "ERROR: " + (res.error or "")
                    tool_errors += 0 if res.ok else 1
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "name": fn.get("name") or "unknown_tool",
                                 "content": out or "(no output)"})
            continue
        final = content
        finish = turn_finish or "stop"
        break
    grid = valid_grid(rp.parse_model_grid(final, require_closed_fence=finish == "length"))
    row = {"task_id": tid, "query_index": qi, "slot": slot, "seed": seed, "attempt": attempt, "finish": finish,
           "grid": grid, "grid_any_turn": grid_any if grid is None else None,
           "completion_tokens": used, "turns": turns, "tool_calls": tool_calls, "tool_errors": tool_errors,
           "tool_s": round(tool_s, 1), "wall_s": round(time.time() - t0, 1), "server": url}
    (OUT / "raw" / f"{key}.json").write_text(json.dumps({"messages": messages, "responses": raw, "result": row}))
    append("samples.jsonl", row)
    return row


class NotStarted(Exception):
    pass


def run_one(t, i, s, url):
    if args.stop_launch_at and time.time() > args.stop_launch_at:
        raise NotStarted()
    try:
        return solve(t, i, s, url)
    except Exception as exc:  # not written to samples.jsonl, so the next start retries it
        append("failures.jsonl", {"task_id": t, "query_index": i, "slot": s, "server": url,
                                  "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
        raise


try:
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    git = None
manifest["starts"].append({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git": git, "urls": URLS,
                           "concurrency": args.concurrency, "todo": len(todo), "done_before": len(done)})
manifest_path.write_text(json.dumps(manifest, indent=1))

print(f"{len(work)} samples ({len(outputs)} outputs x {args.samples}); {len(done)} done; running {len(todo)}", flush=True)
t_start = time.time()
tokens, failed, skipped = [0], [0], [0]
with cf.ThreadPoolExecutor(args.concurrency) as ex:
    futs = {ex.submit(run_one, t, i, s, URLS[n % len(URLS)]): (t, i, s) for n, (t, i, s) in enumerate(todo)}
    for n, fut in enumerate(cf.as_completed(futs), 1):
        try:
            r = fut.result()
        except NotStarted:
            skipped[0] += 1
            continue
        except Exception as exc:
            failed[0] += 1
            print("FAILED", futs[fut], type(exc).__name__, str(exc)[:300], flush=True)
            continue
        tokens[0] += r["completion_tokens"]
        el = time.time() - t_start
        print(f"[{n}/{len(todo)}] {r['task_id']}:{r['query_index']}#{r['slot']} {r['finish']} grid={r['grid'] is not None} "
              f"tok={r['completion_tokens']} turns={r['turns']} tools={r['tool_calls']} {r['wall_s']}s | "
              f"agg {tokens[0] / el:.0f} tok/s", flush=True)
print(f"finished: {len(todo) - failed[0] - skipped[0]} ok, {failed[0]} failed, {skipped[0]} not started "
      "(rerun the same command to retry / continue)", flush=True)
sys.exit(1 if failed[0] or skipped[0] else 0)
