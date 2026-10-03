"""Run a generated `transform(grid)` on grids in a separate process with time and memory limits."""
import json
import subprocess
import sys

RUNNER = r'''
import json, sys
try:
    import resource
    limit = int(sys.argv[1]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except Exception:
    pass  # not available on Windows
payload = json.load(sys.stdin)
results = []
try:
    scope = {}
    exec(payload["code"], scope)
    transform = scope["transform"]
except BaseException as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.exit(0)
import copy
for grid in payload["grids"]:
    try:
        out = transform(copy.deepcopy(grid))
        if hasattr(out, "tolist"):
            out = out.tolist()
        out = [[int(c) for c in row] for row in out]
        ok = 0 < len(out) <= 30 and 0 < len(out[0]) <= 30 and all(len(r) == len(out[0]) for r in out) and all(0 <= c <= 9 for r in out for c in r)
        results.append(out if ok else {"error": "output is not a rectangular grid of digits 0-9 up to 30x30"})
    except BaseException as e:
        results.append({"error": f"{type(e).__name__}: {e}"})
print(json.dumps({"results": results}))
'''


def run_program(code, grids, timeout=4.0, memory_mb=2048):
    """Returns one entry per grid: the output grid, or {"error": ...}."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", RUNNER, str(memory_mb)],
            input=json.dumps({"code": code, "grids": grids}),
            capture_output=True, text=True, timeout=timeout,
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return [{"error": "timeout"}] * len(grids)
    except Exception as e:
        return [{"error": f"runner failed: {type(e).__name__}"}] * len(grids)
    if "error" in out:
        return [{"error": out["error"]}] * len(grids)
    return out["results"]
