"""Build a Kaggle L4x4 notebook that runs the A100 harness (a100/qwen_ab.py) on Kaggle itself.

The comparison of docs/A100_PLAN.md (fixed NVARC, Qwen only, total token budget) on the production hardware: one
vLLM server over 4 L4 (TP=4, fp8 KV + Triton attention, MTP, as in hybrid5), then the conditions in priority order
until a launch cutoff. The competition data must be attached to get L4x4 (without it the kernel gets 2 GPUs); its
solutions file is kept from the model by the tool file guard in qwen_ab.py. Challenges are embedded. Download /kaggle/working/runs and score locally:

    uv run --no-project python scripts/build_qwen_ab_notebook.py
    uvx kaggle kernels push -p kaggle/arc2-qwen-ab-l4
    uvx kaggle kernels output koumeimaki/arc2-qwen-ab-l4 -p results/qwen_ab/<date>
    python a100/score.py results/qwen_ab/<date>/runs/*
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL_ID = "koumeimaki/arc2-qwen-ab-l4"
OUT = ROOT / "kaggle" / KERNEL_ID.split("/")[1]
# priority order of docs/A100_PLAN.md section 3; later ones run only if time is left
CONDITIONS = [("explore", "medium", 32768), ("explore", "xhigh", 32768), ("explore", "medium", 65536), ("explore", "xhigh", 65536)]
LAUNCH_CUTOFF_H = 10.5   # no new sample after this; in-flight ones finish (32k at ~30 tok/s per stream < 20 min)
CONCURRENCY = 16

fixed = ROOT / "a100" / "fixed_nvarc"
manifest = json.load(open(fixed / "cohort_manifest.json"))
challenges = json.load(open(ROOT / "data" / "arc-agi_evaluation_challenges.json"))
screen8 = json.load(open(ROOT / "a100" / "screen8.json"))
tasks = sorted(set(manifest["explore"] + manifest["confirm"] + screen8))
files = {f"a100/{p.name}": p.read_text(encoding="utf-8") for p in [ROOT / "a100" / n for n in
         ["runner_parts.py", "qwen_ab.py", "serve.sh", "screen8.json"]]}
files.update({f"a100/fixed_nvarc/{p.name}": p.read_text(encoding="utf-8") for p in sorted(fixed.glob("*.json"))})
files["data/arc-agi_evaluation_challenges.json"] = json.dumps({t: challenges[t] for t in tasks})


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


cells = [{"cell_type": "markdown", "metadata": {}, "source":
          "Qwen3.8-27B A/B on fixed NVARC inputs (docs/A100_PLAN.md) on L4x4: total token budget per sample, "
          "no forced final, conditions " + ", ".join(f"{e} {b // 1024}k" for _, e, b in CONDITIONS) + "."},
         code("import os, time\nNOTEBOOK_START = time.time()\n"
              "for d in ['a100/fixed_nvarc', 'data']:  # %%writefile does not create directories\n"
              "    os.makedirs(f'/kaggle/working/ab/{d}', exist_ok=True)\nprint('start', NOTEBOOK_START)")]
# as in the hybrid notebooks: TensorFlow in the image breaks transformers' imports in vLLM's inspection subprocess
cells.append(code("!pip uninstall -y tensorflow"))
for path, text in files.items():
    cells.append(code(f"%%writefile /kaggle/working/ab/{path}\n" + text))
cells.append(code(r'''import glob, os, subprocess, sys
print("answer files attached (unreadable by the tool, see qwen_ab.py):", glob.glob("/kaggle/input/**/*solution*.json", recursive=True))
print("GPUs:", subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
wh = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/vllm-0.19.0*.whl", recursive=True)})
assert len(wh) == 1, wh
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-warn-conflicts", "--disable-pip-version-check",
                    "--find-links", wh[0], "--target", "/tmp/vllm_env", "vllm==0.19.0"], capture_output=True, text=True)
print(r.returncode, r.stderr[-1500:])
os.makedirs("/tmp/vllm_env/sklearn", exist_ok=True)   # numpy ABI clash in the NVARC image (see build_hybrid.py)
open("/tmp/vllm_env/sklearn/__init__.py", "w").write('__version__ = "0.0.0"\n')
open("/tmp/vllm_env/sklearn/metrics.py", "w").write('def roc_curve(*a, **k):\n    raise RuntimeError("shim")\n')
cfgs = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "qwen3-8-27b-fp8" in p.lower()]
assert cfgs, "model not attached"
MODEL = os.path.dirname(min(cfgs, key=len)); print("model", MODEL, "| candidates", cfgs)'''))
cells.append(code(f'''!cd /kaggle/working/ab && MODEL="{{MODEL}}" VENV=/tmp/vllm_env TP=4 SEQS={CONCURRENCY} KVFP8=1 MTP=1 CTX=131072 MEM=0.90 LOGDIR=/kaggle/working/server_logs bash a100/serve.sh'''))
cells.append(code(f'''import subprocess, sys, time
CONDITIONS = {CONDITIONS!r}
cutoff = NOTEBOOK_START + {LAUNCH_CUTOFF_H} * 3600
for cohort, effort, budget in CONDITIONS:
    if time.time() > cutoff:
        print("cutoff reached; skipped", cohort, effort, budget); continue
    out = f"/kaggle/working/runs/{{cohort}}_{{effort}}_{{budget // 1024}}k"
    t0 = time.time()
    r = subprocess.run([sys.executable, "a100/qwen_ab.py", "--cohort", cohort, "--effort", effort, "--budget", str(budget),
                        "--concurrency", "{CONCURRENCY}", "--servers", "/kaggle/working/server_logs/servers.json",
                        "--out", out, "--stop-launch-at", str(cutoff)], cwd="/kaggle/working/ab",
                       capture_output=True, text=True)
    print(f"=== {{out}} exit {{r.returncode}} in {{(time.time() - t0) / 3600:.2f}} h")
    print("\\n".join(r.stdout.splitlines()[-6:]), r.stderr[-1500:])'''))
cells.append(code('''!grep -E "SpecDecoding|Avg prompt throughput" /kaggle/working/server_logs/vllm_8000.log | tail -3
!du -sh /kaggle/working/runs/*; ls /kaggle/working/runs'''))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False,
                                              "language": "python", "sourceType": "notebook"}},
      "nbformat": 4, "nbformat_minor": 4}
meta = {
    "id": KERNEL_ID, "title": "ARC2 qwen ab l4", "code_file": f"{KERNEL_ID.split('/')[1]}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"], "dataset_sources": ["koumeimaki/vllm019-cp311-cu128-wheelhouse"], "kernel_sources": [],
    # the competition attachment is what gets the L4x4 machine (without it: 2 GPUs)
    "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"],
    # same Python 3.11 image as the hybrid notebooks (the wheelhouse is cp311)
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
    "machine_shape": "NvidiaL4",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{KERNEL_ID.split('/')[1]}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
print("wrote", OUT, "| tasks embedded:", len(tasks), "| notebook size", (OUT / f"{KERNEL_ID.split('/')[1]}.ipynb").stat().st_size)
