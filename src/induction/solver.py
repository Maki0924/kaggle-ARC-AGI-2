"""Program synthesis leg: ask a code model for `transform(grid)`, keep programs that reproduce every train pair."""
import json
import re
import time
import urllib.request

from sandbox import run_program

SYSTEM = (
    "You solve ARC puzzles by writing Python. Each puzzle has input/output grid pairs that all follow one rule. "
    "Grids are lists of lists of integers 0-9 (colours). Work out the rule, then write a function "
    "`transform(grid)` that takes an input grid (list of lists of int) and returns the output grid. "
    "The function must be general: it is checked on the examples and then applied to new inputs. "
    "Do not hard-code the example outputs. You may use numpy and the standard library. "
    "Reason briefly first, then give the complete code in one ```python block."
)


def grid_text(grid):
    return "\n".join("".join(str(c) for c in row) for row in grid)


def task_prompt(task):
    parts = []
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} input ({len(ex['input'])}x{len(ex['input'][0])}):\n{grid_text(ex['input'])}\n"
                     f"Example {i} output ({len(ex['output'])}x{len(ex['output'][0])}):\n{grid_text(ex['output'])}")
    for i, ex in enumerate(task["test"], 1):
        parts.append(f"Test input {i} ({len(ex['input'])}x{len(ex['input'][0])}):\n{grid_text(ex['input'])}")
    return "\n\n".join(parts) + "\n\nWrite `transform(grid)`."


def feedback_prompt(task, outputs):
    lines = ["Your function does not reproduce the examples:"]
    for i, (ex, out) in enumerate(zip(task["train"], outputs), 1):
        if out == ex["output"]:
            lines.append(f"Example {i}: correct.")
        elif isinstance(out, dict):
            lines.append(f"Example {i}: {out['error']}")
        else:
            wrong = sum(a != b for ra, rb in zip(out, ex["output"]) for a, b in zip(ra, rb)) if (len(out), len(out[0])) == (len(ex["output"]), len(ex["output"][0])) else None
            detail = f"{wrong} cells differ" if wrong is not None else f"shape {len(out)}x{len(out[0])} instead of {len(ex['output'])}x{len(ex['output'][0])}"
            lines.append(f"Example {i}: wrong ({detail}). Your output:\n{grid_text(out)}")
    lines.append("Reconsider the rule and give the corrected complete code in one ```python block.")
    return "\n".join(lines)


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        if "def transform" in block:
            return block
    return None


CONTEXT_LIMIT = int(__import__("os").getenv("ARC_LLM_CONTEXT", "0"))  # 0 = let the server decide


def chat(base_url, model, messages, n, temperature, max_tokens, timeout=3600):
    if CONTEXT_LIMIT:
        # Rough token estimate so that prompt + answer fits a small context window.
        prompt_tokens = sum(len(m["content"]) for m in messages) // 3
        max_tokens = max(256, min(max_tokens, CONTEXT_LIMIT - prompt_tokens - 64))
    body = json.dumps({"model": model, "messages": messages, "n": n, "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return [c["message"]["content"] or "" for c in out["choices"]], out.get("usage", {})


def solve_task(task, base_url, model, n_samples=4, n_repairs=1, temperature=0.8, max_tokens=1500, prog_timeout=4.0):
    """Returns {"programs": [...], "verified": [test outputs per verified program], "seconds", "tokens"}."""
    start = time.time()
    train_inputs = [ex["input"] for ex in task["train"]]
    test_inputs = [ex["input"] for ex in task["test"]]
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_prompt(task)}]
    record = {"programs": [], "verified": [], "tokens": 0}

    def check(code, stage):
        outputs = run_program(code, train_inputs + test_inputs, timeout=prog_timeout)
        train_out, test_out = outputs[:len(train_inputs)], outputs[len(train_inputs):]
        n_ok = sum(o == ex["output"] for o, ex in zip(train_out, task["train"]))
        ok = n_ok == len(train_inputs) and not any(isinstance(o, dict) for o in test_out)
        record["programs"].append({"stage": stage, "code": code, "train_ok": n_ok, "verified": ok})
        if ok:
            record["verified"].append(test_out)
        return ok, train_out

    replies, usage = chat(base_url, model, base, n_samples, temperature, max_tokens)
    record["tokens"] += usage.get("total_tokens", 0)
    failures = []
    for reply in replies:
        code = extract_code(reply)
        if code is None:
            record["programs"].append({"stage": "sample", "code": None, "train_ok": 0, "verified": False})
            continue
        ok, train_out = check(code, "sample")
        if not ok:
            n_ok = record["programs"][-1]["train_ok"]
            failures.append((n_ok, reply, train_out))

    # Repair only when nothing verified, starting from the attempts that got the most examples right.
    if not record["verified"]:
        for _, reply, train_out in sorted(failures, key=lambda f: -f[0])[:2] * n_repairs:
            messages = base + [{"role": "assistant", "content": reply}, {"role": "user", "content": feedback_prompt(task, train_out)}]
            fixes, usage = chat(base_url, model, messages, 1, 0.4, max_tokens)
            record["tokens"] += usage.get("total_tokens", 0)
            code = extract_code(fixes[0])
            if code is not None and check(code, "repair")[0]:
                break

    record["seconds"] = time.time() - start
    return record
