"""Build a self-contained Colab notebook that runs the program-synthesis leg with a large model on an A100."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
split = json.load(open(ROOT / "splits" / "eval_split.json"))
solver = (ROOT / "src" / "induction" / "solver.py").read_text(encoding="utf-8")
sandbox = (ROOT / "src" / "induction" / "sandbox.py").read_text(encoding="utf-8")
runner = (ROOT / "scripts" / "run_induction.py").read_text(encoding="utf-8")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# ARC-AGI-2 program synthesis on Colab\n\nRuntime: A100 (80GB preferred). Upload `arc-agi_evaluation_challenges.json` and `arc-agi_evaluation_solutions.json` to `/content/data/` first (or mount Drive and copy them)."),
    code("MODEL = 'Qwen/Qwen3.8-27B-FP8'   # 80GB A100. On a 40GB A100 use 'cyankiwi/Qwen3.8-27B-AWQ-INT4'\nSUBSET = 'screen8'               # 'screen8', 'timing20' or 'dev'\nSAMPLES, REPAIRS, MAX_TOKENS, PARALLEL = 4, 1, 12000, 8"),
    code("!pip -q install vllm\n!mkdir -p /content/data /content/work/induction /content/runs"),
    code("import json, os\nsplit = " + json.dumps({k: split[k] for k in ["screen8", "timing20", "dev"]}) + "\nos.makedirs('/content/splits', exist_ok=True)\njson.dump(split, open('/content/splits/eval_split.json', 'w'))\nassert os.path.exists('/content/data/arc-agi_evaluation_challenges.json'), 'upload the evaluation json files to /content/data'"),
    code("%%writefile /content/work/induction/sandbox.py\n" + sandbox),
    code("%%writefile /content/work/induction/solver.py\n" + solver),
    code("%%writefile /content/work/run_induction.py\n" + runner.replace('ROOT = pathlib.Path(__file__).resolve().parents[1]', "ROOT = pathlib.Path('/content')").replace('pathlib.Path.home() / "arc2-local" / "runs"', 'pathlib.Path("/content/runs")').replace('ROOT / "src" / "induction"', 'ROOT / "work" / "induction"')),
    code("import subprocess, time, urllib.request\nserver = subprocess.Popen(['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL, '--served-model-name', 'llm', '--max-model-len', '32768', '--gpu-memory-utilization', '0.92', '--max-num-seqs', str(PARALLEL)], stdout=open('/content/vllm.log', 'w'), stderr=subprocess.STDOUT)\nfor _ in range(240):\n    time.sleep(10)\n    try:\n        urllib.request.urlopen('http://localhost:8000/v1/models', timeout=3); print('server ready'); break\n    except Exception:\n        if server.poll() is not None: print(open('/content/vllm.log').read()[-3000:]); raise SystemExit('vllm died')"),
    code("!cd /content && python work/run_induction.py ind_{MODEL.split('/')[-1]}_{SUBSET} --subset {SUBSET} --model llm --samples {SAMPLES} --repairs {REPAIRS} --max-tokens {MAX_TOKENS} --parallel {PARALLEL}"),
    code("# Download the results folder (or copy it to Drive) and put it under ~/arc2-local/runs/ locally.\n!cd /content/runs && zip -qr /content/induction_results.zip . && ls -la /content/induction_results.zip"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "colab_induction.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
print("wrote", out)
