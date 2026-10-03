"""Re-score saved candidate pools under another adapter (or the base model), and record the NLL of the true answer.

Runs inside the arc2-nvarc image (see scripts/run_rescore.py). For every test output of the given run it loads the
LoRA <lora-dir>/<task>.safetensors (or none for "base"), scores each unique candidate grid on the same 8 augmented
views the solver uses, and also scores the ground-truth output. Results: <run>/rescore_<tag>.pkl
"""
import bz2
import json
import os
import pickle
import sys
import time
import zlib
from collections import defaultdict

sys.path.insert(0, "/work")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from arc_loader import ArcDataset, QwenFormatter  # noqa: E402
from arc_solver import calc_scores  # noqa: E402

run_dir, lora_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
keys = sys.argv[4].split(",")
max_seq_length = 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.environ["ARC_MODEL_DIR"], full_finetuning=False, load_in_4bit=False,
    local_files_only=True, use_gradient_checkpointing=False, max_seq_length=max_seq_length)
model = FastLanguageModel.get_peft_model(
    model, r=256, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=42, use_rslora=True, loftq_config=None)
for p in model.parameters():
    if p.dtype == torch.float32:
        p.data = p.data.to(torch.bfloat16)
default_weights = {k: v.clone().detach() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
model = FastLanguageModel.for_inference(model)
formatter = QwenFormatter(tokenizer=tokenizer)
max_new_tokens = formatter.max_new_tokens()

challenges = ArcDataset.from_file(os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_challenges.json"))
solutions = json.load(open(os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_solutions.json")))

pools = defaultdict(dict)
for name in os.listdir(os.path.join(run_dir, "inference_outputs")):
    with bz2.BZ2File(os.path.join(run_dir, "inference_outputs", name)) as f:
        for i, s in enumerate(pickle.load(f)):
            pools[name.split(".")[0]][f"{name}.out{i}"] = s

out_path = os.path.join(run_dir, f"rescore_{tag}.pkl")
results = pickle.load(open(out_path, "rb")) if os.path.exists(out_path) else {}


@torch.inference_mode()
def score_grid(puzzle_ds_multi, bk, grid):
    ds = ArcDataset(keys=[bk], queries={bk: puzzle_ds_multi.queries.get(bk)}, replies={bk: [grid.tolist()]})
    ds = ds.augment(seed=zlib.crc32(bk.encode()) % 1024**2)
    ds = ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
    samples = ds.as_list(formatter)
    q, a = [s["input"] for s in samples], [s["reply"] for s in samples]
    return calc_scores(q[:4], a[:4], tokenizer, model) + calc_scores(q[4:], a[4:], tokenizer, model)


for key in keys:
    if all(f"{key}_{i}" in results for i in range(len(solutions[key]))):
        continue
    start = time.time()
    set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")
    if tag != "base":
        path = os.path.join(lora_dir, f"{key}.safetensors")
        if not os.path.exists(path):
            print(f"no LoRA for {key}, skipping")
            continue
        set_peft_model_state_dict(model, load_file(path, device="cuda"), adapter_name="default")
    puzzle_ds_multi = challenges.change_keys([key]).split_multi_replies()
    for i in range(len(solutions[key])):
        bk = f"{key}_{i}"
        grids = {}
        for s in pools.get(bk, {}).values():
            grids.setdefault(tuple(map(tuple, s["solution"])), s["solution"])
        entry = {"candidates": {g: score_grid(puzzle_ds_multi, bk, grid) for g, grid in grids.items()},
                 "truth": score_grid(puzzle_ds_multi, bk, np.array(solutions[key][i]))}
        results[bk] = entry
    pickle.dump(results, open(out_path, "wb"))
    print(f"{key}: {sum(len(results[f'{key}_{i}']['candidates']) for i in range(len(solutions[key])))} candidates, {time.time() - start:.0f}s")
print("done", out_path)
