"""Build a Colab notebook that runs the NVARC pipeline (src/nvarc) on an A100 as a second GPU worker."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# NVARC pipeline on Colab (A100)\n\n"
       "Needs Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (from kaggle.json) with the notebook access toggle on.\n"
       "Set the run below, run all cells, then download `/content/<RUN_NAME>.zip` and unzip it into `~/arc2-local/runs/`."),
    code("RUN_NAME = 'colab_nll_stock'\n"
         "SUBSET = 'timing20'            # key in splits/eval_split.json or comma separated task ids\n"
         "HOURS = 6\n"
         "ENV = {'ARC_NLL_ONLY': '1', 'ARC_MERGE_LORA': '1'}   # e.g. add 'ARC_N_TRAIN_AUG': '8', 'ARC_LR': '2.5e-5', 'ARC_EPOCHS': '2'\n"
         "GIT_REF = 'main'"),
    code("import os\nfrom google.colab import userdata\n"
         "os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\nos.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')\n"
         "!pip -q install kaggle\n"
         "!rm -rf /content/repo && git clone -q --depth 1 -b $GIT_REF https://github.com/Maki0924/kaggle-ARC-AGI-2.git /content/repo\n"
         "!mkdir -p /content/data /content/models /content/runs\n"
         "!cd /content/data && kaggle competitions download -c arc-prize-2026-arc-agi-2 -q && unzip -o -q arc-prize-2026-arc-agi-2.zip\n"
         "!kaggle models instances versions download sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 -p /content/models/qwen3_4b_grids15_sft139 --untar -q\n"
         "!ls /content/data /content/models/qwen3_4b_grids15_sft139 | head -20"),
    code("# Same package set as the Kaggle utility notebook sorokin/pip-install-unsloth-flash-patch.\n"
         "import torch, sys\nprint(torch.__version__, sys.version)\n"
         "!pip -q install 'unsloth==2025.9.7' 'unsloth_zoo==2025.9.9' 'numpy==2.2.6' 'transformers==4.55.4' 'peft==0.21.2' 'trl==0.22.2' 'datasets==3.6.0' 'accelerate==1.15.0' 'safetensors'\n"
         "!pip -q uninstall -y torchao\n"
         "# flash-attn prebuilt wheel for this torch/cuda/python if one exists; the Qwen3 patch is applied only when it imports.\n"
         "import importlib, subprocess, re\n"
         "tv = torch.__version__.split('+')[0]; tv = '.'.join(tv.split('.')[:2]); cu = torch.version.cuda.replace('.', '')[:3]; py = f'cp{sys.version_info.major}{sys.version_info.minor}'\n"
         "url = f'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu{cu}torch{tv}-{py}-{py}-linux_x86_64.whl'\n"
         "print('trying', url)\n"
         "r = subprocess.run(['pip', '-q', 'install', '--no-deps', url])\n"
         "try:\n    import flash_attn; print('flash_attn', flash_attn.__version__)\n"
         "    import unsloth.models.qwen3 as q3; patch_target = q3.__file__\n"
         "    subprocess.run(['patch', '--binary', '-N', patch_target, '/content/repo/docker/qwen3.patch'])\n"
         "except Exception as e:\n    print('flash_attn unavailable, running without the patch:', e)"),
    code("import json, os, time, subprocess, shutil\n"
         "split = json.load(open('/content/repo/splits/eval_split.json'))\n"
         "keys = split[SUBSET] if SUBSET in split else SUBSET.split(',')\n"
         "run_dir = f'/content/runs/{RUN_NAME}'; os.makedirs(run_dir, exist_ok=True)\n"
         "json.dump({'subset': SUBSET, 'keys': keys, 'env': [f'{k}={v}' for k, v in ENV.items()], 'hours': HOURS, 'colab': True}, open(f'{run_dir}/config.json', 'w'), indent=1)\n"
         "shutil.copytree('/content/repo/src/nvarc', f'{run_dir}/src', dirs_exist_ok=True)\n"
         "env = dict(os.environ, ARC_MODEL_DIR='/content/models/qwen3_4b_grids15_sft139', ARC_DATA_DIR='/content/data', ARC_WORK_DIR=run_dir, ARC_NUM_GPUS='1', ARC_TASKS=','.join(keys), PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', UNSLOTH_DISABLE_STATISTICS='1', HF_HUB_OFFLINE='1', **ENV)\n"
         "end_time = time.time() + HOURS * 3600\n"
         "with open(f'{run_dir}/log.txt', 'w') as log:\n"
         "    proc = subprocess.Popen(['python', 'starter.py', '--end-time', str(end_time)], cwd=f'{run_dir}/src', env=env, stdout=log, stderr=subprocess.STDOUT)\n"
         "    proc.wait()\n"
         "!tail -n 5 {run_dir}/log.txt; grep -c FAILED {run_dir}/log.txt; wc -l {run_dir}/timing_rank0.jsonl"),
    code("!cd /content/runs && zip -qr /content/{RUN_NAME}.zip {RUN_NAME} -x '{RUN_NAME}/src/*' && ls -la /content/{RUN_NAME}.zip\n"
         "# Optional: copy to Drive\n# from google.colab import drive; drive.mount('/content/drive'); !cp /content/{RUN_NAME}.zip /content/drive/MyDrive/"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "colab_nvarc.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
print("wrote", out)
