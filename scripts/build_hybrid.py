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
    "hybrid1": ("koumeimaki/arc2-hybrid1-nvarc-qwen38", "ARC2 hybrid1 nvarc+qwen38", 7.5, "", 42300),
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
print("handoff order (first 12):", [(k, task_priority(k)) for k in keys[:12]])
print("NVARC done at", round((time.time() - NOTEBOOK_START) / 3600, 2), "h")
''')

merge_cell = code(r'''
# Final merge: NVARC answers, with Qwen's model-backed grids applied by policy.
import json, os
nvarc = json.load(open("/kaggle/working/nvarc_submission.json"))
conf = json.load(open("/kaggle/working/nvarc_confidence.json"))
final = json.loads(json.dumps(nvarc))
stats = {"qwen_outputs": 0, "to_attempt_1": 0, "to_attempt_2": 0, "agree": 0, "kept_nvarc": 0}
try:
    receipt = json.load(open("/kaggle/working/freeform_receipt.json"))
    qwen_sub = json.load(open("/kaggle/working/submission.json"))
    for task, per_query in receipt.get("per_task", {}).items():
        for qi, records in per_query.items():
            i = int(qi)
            slots = [r.get("published_slot") for r in records if r.get("grid") and r.get("published_slot")]
            if not slots:
                continue
            grid = qwen_sub[task][i][slots[0]]
            stats["qwen_outputs"] += 1
            n_cand, v1, v2 = conf.get(f"{task}_{i}", [0, 0, 0])
            a1, a2 = final[task][i]["attempt_1"], final[task][i]["attempt_2"]
            if n_cand == 0 or a1 == [[0]]:
                final[task][i]["attempt_1"] = grid; stats["to_attempt_1"] += 1
            elif grid == a1:
                stats["agree"] += 1
            elif n_cand < 2 or v2 <= 2:
                final[task][i]["attempt_2"] = grid; stats["to_attempt_2"] += 1
            else:
                stats["kept_nvarc"] += 1
except Exception as e:
    print("merge: no usable Qwen output:", type(e).__name__, e)
json.dump(final, open("/kaggle/working/submission.json", "w"))
print("merge stats:", stats)
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    from arc_loader import ArcDataset
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")
    keys = json.load(open("/kaggle/working/qwen_task_order.json"))
    sub_n = {k: nvarc[k] for k in keys}; sub_f = {k: final[k] for k in keys}
    score = lambda sub: sum(1 / len(data.replies[k]) for k in keys for i, r in enumerate(data.replies[k]) if any(r == sub[k][i][a] for a in ("attempt_1", "attempt_2")))
    print(f"on {len(keys)} handoff tasks: NVARC {score(sub_n):.2f}  ->  hybrid {score(sub_f):.2f}")
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
