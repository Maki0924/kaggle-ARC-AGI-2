"""Build the hybrid Kaggle notebook: NVARC fast1 on all tasks, then Qwen3.8-27B (russcore runner) on the
tasks NVARC is least confident about, merged by a vote-based policy.

    uv run --no-project python scripts/build_hybrid.py [variant]
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nvarc"
RUNNER_NB = ROOT / "kaggle" / "qwen38-probe" / "arc2-qwen38-probe.ipynb"  # patched fork (MTP flags)

VARIANTS = {
    # name: (kernel id, title, NVARC hours, NVARC extra env, runner hard wall seconds)
    "hybrid1": ("koumeimaki/arc2-hybrid1-nvarc-qwen38", "ARC2 hybrid1 nvarc+qwen38", 7.5, "", 40500),  # wall 11.25h from notebook start
    # same as hybrid1, plus NVARC's candidates in the Qwen prompt as hypotheses to verify
    "hybrid2": ("koumeimaki/arc2-hybrid2-nvarc-qwen38-hint", "ARC2 hybrid2 nvarc+qwen38 hint", 7.5, "", 40500),
}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "hybrid1"
WHEELHOUSE = "koumeimaki/vllm019-cp311-cu128-wheelhouse"
KERNEL_ID, TITLE, NVARC_HOURS, NVARC_ENV, HARD_WALL = VARIANTS[VARIANT]
OUT = ROOT / "kaggle" / KERNEL_ID.split("/")[1]
SCREEN8 = ",".join(json.load(open(ROOT / "splits" / "eval_split.json"))["screen8"])


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def writefile(name):
    return code(f"%%writefile {name}\n" + (SRC / name).read_text(encoding="utf-8"))


# --- the Qwen runner cell, patched -----------------------------------------------------------------
runner_nb = json.loads(RUNNER_NB.read_text(encoding="utf-8"))
runner_src = "".join(runner_nb["cells"][2]["source"])


def patch(old, new, count=1):
    global runner_src
    assert runner_src.count(old) == count, (old[:60], runner_src.count(old))
    runner_src = runner_src.replace(old, new)


# the 12-hour wall is measured from the notebook start, not from the runner cell start
patch("RUN_STARTED_S = time.time()", "RUN_STARTED_S = globals().get('NOTEBOOK_START', time.time())")
patch("'hard_wall_seconds': 43020", f"'hard_wall_seconds': {HARD_WALL}")
# our own Python 3.11 wheelhouse (vLLM 0.19.0 + torch 2.10 cu128), installed after NVARC has finished
patch("'wheelhouse_hint': 'arc3-vllm-h100-wheelhouse-v3'", f"'wheelhouse_hint': '{WHEELHOUSE.split('/')[1]}'")
patch(").hexdigest() != RUNTIME_CONFIG_SHA256:", ").hexdigest() != RUNTIME_CONFIG_SHA256 and False:  # config edited above")
# install vLLM into an isolated directory (the NVARC utility packages live on a read-only path that pip would try to
# replace), and run the server with that directory first on PYTHONPATH
VLLM_ENV = "/tmp/vllm_env"
# the server must not see the system site-packages (sklearn/numpy built for the NVARC stack): python -S + PYTHONPATH
patch('base_serve = [sys.executable,"-m","vllm.entrypoints.openai.api_server",', 'base_serve = [sys.executable,"-S","-m","vllm.entrypoints.openai.api_server",')
patch('"vllm==" + EXPECTED_VLLM_VERSION],', f'"--target", "{VLLM_ENV}", "vllm==" + EXPECTED_VLLM_VERSION],')
patch('            runtime_versions["vllm"] = importlib.metadata.version("vllm")',
      f'            sys.path.insert(0, "{VLLM_ENV}")\n            runtime_versions["vllm"] = importlib.metadata.version("vllm")')
patch("                    server = subprocess.Popen(\n                        serve, stdout=srv_log, stderr=subprocess.STDOUT,\n                        start_new_session=True)",
      "                    server = subprocess.Popen(\n                        serve, stdout=srv_log, stderr=subprocess.STDOUT,\n                        start_new_session=True,\n"
      f"                        env={{**os.environ, 'PYTHONPATH': '{VLLM_ENV}', 'PATH': '{VLLM_ENV}/bin:' + os.environ.get('PATH', '')}})")
# tasks in NVARC's low-confidence-first order; in diagnostic mode the limit is the handoff list length
patch("task_ids = list(challenges)\nif TASK_LIMIT is not None:\n    task_ids = task_ids[:TASK_LIMIT]",
      "task_ids = [t for t in globals().get('QWEN_TASK_ORDER', list(challenges)) if t in challenges]\n"
      "if TASK_LIMIT is not None and 'QWEN_TASK_ORDER' not in globals():\n    task_ids = task_ids[:TASK_LIMIT]")

# --- selective Qwen: only outputs the merge policy can use, in handoff order, capped to what fits at full episode budget
patch('''    queries = [(tid, qi, q["input"]) for tid in task_ids for qi, q in enumerate(challenges[tid]["test"])]
    remaining_generation_s = max(0, GENERATION_DEADLINE_S - time.time())''',
      '''    queries = [(tid, qi, challenges[tid]["test"][qi]["input"]) for tid, qi in QWEN_ELIGIBLE if tid in challenges and qi < len(challenges[tid]["test"])]
    remaining_generation_s = max(0, GENERATION_DEADLINE_S - time.time())
    _waves = max(1, int(remaining_generation_s // (CONFIGURED_PASS_POLICIES["xhigh"]["episode_budget_s"] + COVERAGE_BUDGET_ADJUSTMENT["per_wave_overhead_s"])))
    queries = queries[:_waves * CONCURRENCY]
    log("selective Qwen: eligible outputs", len(QWEN_ELIGIBLE), "| scheduled", len(queries), "| waves", _waves)''')
# --- the runner's submission starts from NVARC's answers, so every incremental save is already merged
patch('''submission = AGG.guaranteed_fallback_submission(challenges)
fallback_submission = json.loads(json.dumps(submission))''',
      '''submission = json.load(open("/kaggle/working/nvarc_submission.json"))
NVARC_CONF = {k: tuple(v) for k, v in json.load(open("/kaggle/working/nvarc_confidence.json")).items()}
fallback_submission = AGG.guaranteed_fallback_submission(challenges)
# the runner validator requires distinct attempts: fill NVARC's duplicate / empty attempt_2 from the distinct floor
for _tid, _rows in submission.items():
    for _qi, _row in enumerate(_rows):
        if _row["attempt_1"] == _row["attempt_2"]:
            _row["attempt_2"] = select_distinct_fallback(_row["attempt_1"], fallback_submission[_tid][_qi])
NVARC_ROWS = json.loads(json.dumps(submission))''')
# --- vote-based merge policy at publish time
patch('''    key = (tid, qi)
    old_row = submission[tid][qi]
    if key not in model_answered:
        new_row = {
            "attempt_1": grid,
            "attempt_2": select_distinct_fallback(
                grid, fallback_submission[tid][qi]),
        }
        published_slot = "attempt_1"
    else:
        if grid == old_row["attempt_1"] or grid == old_row["attempt_2"]:
            return None
        new_row = {"attempt_1": old_row["attempt_1"], "attempt_2": grid}
        published_slot = "attempt_2"''',
      '''    key = (tid, qi)
    old_row = submission[tid][qi]
    nv = NVARC_ROWS[tid][qi]
    n_cand, v1, v2 = NVARC_CONF.get(f"{tid}_{qi}", (0, 0, 0))
    if grid == old_row["attempt_1"] or grid == old_row["attempt_2"]:
        return None
    if n_cand == 0 or nv["attempt_1"] == [[0]]:
        if key not in model_answered:
            new_row = {"attempt_1": grid, "attempt_2": select_distinct_fallback(grid, fallback_submission[tid][qi])}
            published_slot = "attempt_1"
        else:
            new_row = {"attempt_1": old_row["attempt_1"], "attempt_2": grid}
            published_slot = "attempt_2"
    elif (n_cand < 2 or v2 <= 2) and old_row["attempt_2"] == nv["attempt_2"]:
        new_row = {"attempt_1": old_row["attempt_1"], "attempt_2": grid}
        published_slot = "attempt_2"
    else:
        return None''')

# --- v9: only naturally finished answers may replace NVARC (forced answers were 0/10 correct); lower concurrency so
# each sequence gets enough tokens to finish; higher per-turn cap
patch('''    grid = validated_model_grid(o.get("grid"))
    published_slot = None
    publish_error = None
    if grid is not None:
        try:
            published_slot = publish_model_grid(tid, qi, grid)''',
      '''    grid = validated_model_grid(o.get("grid"))
    published_slot = None
    publish_error = None
    if grid is not None and o.get("finish") != "stop" and NVARC_CONF.get(f"{tid}_{qi}", (0, 0, 0))[0] > 0:
        log("forced/recovered answer discarded:", tid, qi, o.get("finish"))
        # keep it for offline scoring: is the recovery accurate enough to accept next time?
        with open("/kaggle/working/qwen_discarded.jsonl", "a") as _f:
            _f.write(json.dumps({"task_id": tid, "query_index": qi, "finish": o.get("finish"), "grid": grid}) + "\\n")
        grid = None
    if grid is not None:
        try:
            published_slot = publish_model_grid(tid, qi, grid)''')
patch("'scheduler': {'concurrency': 24,", "'scheduler': {'concurrency': 8,")
patch("'xhigh': {'reasoning_effort': 'xhigh', 'turn_max_tokens': 40960,", "'xhigh': {'reasoning_effort': 'xhigh', 'turn_max_tokens': 57344,")

# --- v10: recovery sees (almost) the whole reasoning, fp8 KV + Triton attention with retry, concurrency 16
patch('"[my reasoning, truncated at the token limit]\\n" + prior_reasoning[-8000:]',
      '"[my reasoning, truncated at the token limit]\\n" + prior_reasoning[-max(8000, (CONTEXT_LEN - 8192) * 2 - len(prompt)):]')
patch('''        attempts = (([tool_flags] if TOOLS else []) + [[]])''',
      '''        fp8_flags = ["--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN"]  # L4: fp8 KV needs the Triton backend
        os.environ["VLLM_FORCE_ATTN_BACKEND"] = "TRITON_ATTN"
        attempts = (([tool_flags + fp8_flags, tool_flags] if TOOLS else []) + [[]])''')
patch("'scheduler': {'concurrency': 8,", "'scheduler': {'concurrency': 16,")

# --- hybrid2: show Qwen NVARC's candidate grids as hypotheses to verify, fix or reject (verification is cheaper than solving)
if VARIANT == "hybrid2":
    patch('''def solve_one_agent(tid, qi, test_input, slot, policy_name):''',
          '''def nvarc_hint(tid, qi):
    """Text listing NVARC's distinct candidate grids for this test output, or '' if it has none."""
    global _NVARC_ORIG
    try:
        _NVARC_ORIG
    except NameError:
        _NVARC_ORIG = json.load(open("/kaggle/working/nvarc_submission.json"))
    row = _NVARC_ORIG.get(tid, [{}] * (qi + 1))[qi]
    cands = []
    for name in ("attempt_1", "attempt_2"):
        g = row.get(name)
        if isinstance(g, list) and g and g != [[0]] and g not in cands:
            cands.append(g)
    if not cands:
        return ""
    parts = ["\\nA separate solver proposed the candidate output(s) below for the FINAL INPUT. They may be wrong. "
             "Check each candidate against the rule you infer from EVERY training example (run_python is the "
             "fastest way: apply your rule to the training inputs and compare). Then either confirm a candidate, "
             "correct it, or give your own answer if none fits.\\n"]
    for i, g in enumerate(cands):
        parts.append("Candidate %s:\\n%s\\n" % ("AB"[i], render(g)))
    return "\\n".join(parts)

def solve_one_agent(tid, qi, test_input, slot, policy_name):''')
    patch('''    prompt = build_agent_prompt(task)
    messages = [{"role":"user","content":prompt}]''',
          '''    prompt = build_agent_prompt(task) + nvarc_hint(tid, qi)
    messages = [{"role":"user","content":prompt}]''')

runner_cell = code(
    "# Qwen3.8 pass on the handoff tasks. SystemExit/exceptions must not stop the notebook: the merge cell below\n"
    "# rebuilds submission.json from NVARC's answers plus whatever the runner published.\n"
    "import json, os, time, subprocess, sys\n"
    "# NVARC is finished: remove system packages whose numpy ABI clashes with the isolated vLLM stack (optional deps of transformers)\n"
    "# a shim sklearn at the front of the isolated path shadows the system one (transformers only touches roc_curve)\n"
    "os.makedirs('/tmp/vllm_env/sklearn', exist_ok=True)\n"
    "open('/tmp/vllm_env/sklearn/__init__.py', 'w').write('__version__ = \"0.0.0\"\\n')\n"
    "open('/tmp/vllm_env/sklearn/metrics.py', 'w').write('def roc_curve(*a, **k):\\n    raise RuntimeError(\"sklearn shim\")\\n')\n"
    "os.environ['ARC_FREEFORM_DIAGNOSTIC'] = '' if os.getenv('KAGGLE_IS_COMPETITION_RERUN') else '1'\n"
    "QWEN_TASK_ORDER = json.load(open('/kaggle/working/qwen_task_order.json'))\n"
    "QWEN_ELIGIBLE = [tuple(x) for x in json.load(open('/kaggle/working/qwen_eligible.json'))]\n"
    "print('runner start at', round((time.time() - NOTEBOOK_START) / 3600, 2), 'h; tasks in handoff order:', len(QWEN_TASK_ORDER))\n"
    "_runner = open('/kaggle/working/qwen_runner.py', encoding='utf-8').read()\n"
    "try:\n    exec(compile(_runner, 'qwen_runner.py', 'exec'), globals())\n"
    "except BaseException as e:\n    print('runner stopped:', type(e).__name__, str(e)[:300])\n"
)

handoff_cell = code(r'''
# NVARC is done (or out of time). Write its submission, then rank tasks for Qwen: outputs with no candidate first,
# then by the vote count of the second candidate (weak attempt_2 is what Qwen may replace).
import bz2, json, os, pickle, shutil
from collections import defaultdict
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_kgmon

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
data_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
data = ArcDataset.from_file(os.path.join(data_dir, "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json"))
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
submission = data.get_submission()
try:
    decoder.load_decoded_results("/kaggle/inference_outputs")
    ArcDataset.fill_submission(decoder.run_selection_algo(), submission)
except Exception as e:
    print("*** NVARC selection failed:", type(e).__name__, e)
json.dump(submission, open("/kaggle/working/nvarc_submission.json", "w"))
json.dump(submission, open("/kaggle/working/submission.json", "w"))

conf = {}   # (task, test index) -> (n unique candidates, votes of attempt_1, votes of attempt_2)
for bk, pool in decoder.decoded_results.items():
    groups = defaultdict(int)
    for s in pool.values():
        groups[tuple(map(tuple, s["solution"]))] += 1
    ranked = score_kgmon(pool)
    votes = [groups[tuple(map(tuple, g))] for g in ranked]
    task, idx = bk.rsplit("_", 1)
    conf[(task, int(idx))] = (len(ranked), votes[0] if votes else 0, votes[1] if len(votes) > 1 else 0)
json.dump({f"{t}_{i}": v for (t, i), v in conf.items()}, open("/kaggle/working/nvarc_confidence.json", "w"))

def task_priority(task):
    rows = [conf.get((task, i), (0, 0, 0)) for i in range(len(data.queries[task]["test"]))]
    no_candidate = any(r[0] == 0 for r in rows)
    weak_second = min(r[2] for r in rows)
    return (0 if no_candidate else 1, weak_second, min(r[1] for r in rows))

keys = sorted(data.keys, key=task_priority)
if not rerun_mode:
    keys = [k for k in keys if k in os.getenv("ARC_TASKS", ",".join(keys)).split(",")]
json.dump(keys, open("/kaggle/working/qwen_task_order.json", "w"))

# outputs the merge policy can use, most promising first: no candidate -> one candidate -> weak attempt_2 (votes <= 2)
eligible = []
for task in keys:
    for i in range(len(data.queries[task]["test"])):
        n, v1, v2 = conf.get((task, i), (0, 0, 0))
        if n == 0 or n < 2 or v2 <= 2:
            eligible.append(((0 if n == 0 else 1 if n < 2 else 2), v2, v1, task, i))
eligible.sort()
json.dump([[t, i] for _, _, _, t, i in eligible], open("/kaggle/working/qwen_eligible.json", "w"))
print("eligible outputs for Qwen:", len(eligible), "of", sum(len(data.queries[k]["test"]) for k in keys))
print("handoff order (first 12):", [(k, task_priority(k)) for k in keys[:12]])
print("NVARC done at", round((time.time() - NOTEBOOK_START) / 3600, 2), "h")
''')

merge_cell = code(r'''
# The runner's submission already holds NVARC's answers plus Qwen grids applied by policy. Validate it; fall back to NVARC.
import json, os
nvarc = json.load(open("/kaggle/working/nvarc_submission.json"))
final = nvarc
try:
    cand = json.load(open("/kaggle/working/submission.json"))
    ok = set(cand) == set(nvarc) and all(len(cand[t]) == len(nvarc[t]) and all(isinstance(r.get("attempt_1"), list) and isinstance(r.get("attempt_2"), list) for r in cand[t]) for t in nvarc)
    if ok:
        final = cand
    else:
        print("runner submission invalid; using NVARC")
except Exception as e:
    print("no runner submission:", type(e).__name__, e)
json.dump(final, open("/kaggle/working/submission.json", "w"))
print("outputs changed by Qwen:", sum(1 for t in nvarc for i, r in enumerate(nvarc[t]) if final[t][i] != r))
try:
    acc = json.load(open("/kaggle/working/freeform_receipt.json")).get("prediction_accounting", {})
    print("model-backed attempt_1:", acc.get("model_backed_attempt_1"), "attempt_2:", acc.get("model_backed_attempt_2"))
except Exception:
    pass
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    from arc_loader import ArcDataset
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")
    keys = json.load(open("/kaggle/working/qwen_task_order.json"))
    score = lambda sub: sum(1 / len(data.replies[k]) for k in keys for i, r in enumerate(data.replies[k]) if any(r == sub[k][i][a] for a in ("attempt_1", "attempt_2")))
    print(f"on {len(keys)} handoff tasks: NVARC {score(nvarc):.2f}  ->  hybrid {score(final):.2f}")
    if os.path.exists("/kaggle/working/qwen_discarded.jsonl"):
        rows = [json.loads(l) for l in open("/kaggle/working/qwen_discarded.jsonl")]
        hits = [r for r in rows if r["grid"] == data.replies[r["task_id"]][r["query_index"]]]
        print(f"discarded forced/recovered answers: {len(rows)}, correct: {len(hits)}", [(r["task_id"], r["query_index"]) for r in hits])
''')

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": "Hybrid: NVARC fast1 (Qwen3-4B TTT, batch 8, merged LoRA) on every task within the NVARC budget, then Qwen3.8-27B-FP8 (vLLM 0.19, TP=4, MTP) on the tasks NVARC is least sure about until the wall, merged by vote policy."},
    code(f"import time, os, json\nNOTEBOOK_START = time.time()\nNVARC_HOURS = {NVARC_HOURS}\nglobal_end_time = NOTEBOOK_START + NVARC_HOURS * 3600 - 120\n"
         f"if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n    os.environ['ARC_TASKS'] = '{SCREEN8}'  # commit test: NVARC and Qwen on the same 8 dev tasks\n"
         "print('start', NOTEBOOK_START)"),
    code("!pip uninstall -y tensorflow"),
    writefile("arc_loader.py"), writefile("arc_decoder.py"), writefile("arc_solver.py"), writefile("starter.py"),
    code(f"!ARC_DECODE_BATCH=8 ARC_MERGE_LORA=1 {NVARC_ENV}PYTHONHASHSEED=0 UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {{global_end_time}}"),
    handoff_cell,
    code("%%writefile /kaggle/working/qwen_runner.py\n" + runner_src),
    runner_cell,
    merge_cell,
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                                    "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
      "nbformat": 4, "nbformat_minor": 4}
meta = {
    "id": KERNEL_ID, "title": TITLE, "code_file": f"{KERNEL_ID.split('/')[1]}.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
    "dataset_sources": [WHEELHOUSE],
    "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
    "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"],
    # the NVARC image (Python 3.11, torch 2.8 cu128); the Qwen phase installs its own torch 2.10 afterwards
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
    "machine_shape": "NvidiaL4",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{KERNEL_ID.split('/')[1]}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
print("wrote", OUT)
