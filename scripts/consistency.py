"""Consistency score for saved candidates: does adding (test input, candidate) as an extra demonstration make a
held-out training pair easier to predict?  r(c) = mean_j [ NLL(y_j | D-j) - NLL(y_j | D-j + (x*, c)) ].

Runs inside the arc2-nvarc image via scripts/run_consistency.py. Scores the top-K candidates (stock ranking) of
every test output in <run>/inference_outputs with the base model (tag "base") or a LoRA from <lora-dir>.
Results: <run>/consistency_<tag>.pkl
"""
import bz2
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/work")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402

from arc_decoder import score_kgmon  # noqa: E402
from arc_loader import QwenFormatter  # noqa: E402
from arc_solver import calc_scores  # noqa: E402

run_dir, lora_dir, tag, keys, top_k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5])
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

challenges = json.load(open(os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_challenges.json")))
pools = defaultdict(dict)
for name in os.listdir(os.path.join(run_dir, "inference_outputs")):
    with bz2.BZ2File(os.path.join(run_dir, "inference_outputs", name)) as f:
        for i, s in enumerate(pickle.load(f)):
            pools[name.split(".")[0]][f"{name}.out{i}"] = s

out_path = os.path.join(run_dir, f"consistency_{tag}.pkl")
results = pickle.load(open(out_path, "rb")) if os.path.exists(out_path) else {}


def prompt(pairs, query_in, reply_out):
    """Chat text for demonstrations `pairs`, then the query; returns (input text, reply text)."""
    text = formatter.fmt_train(pairs) + formatter.fmt_query([{"input": query_in}])
    return text, formatter.fmt_reply([reply_out])


@torch.inference_mode()
def nll(queries, answers):
    out = []
    for i in range(0, len(queries), 4):
        out += calc_scores(queries[i:i+4], answers[i:i+4], tokenizer, model)
    return out


for key in keys:
    task = challenges[key]
    start = time.time()
    set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")
    if tag != "base":
        path = os.path.join(lora_dir, f"{key}.safetensors")
        if not os.path.exists(path):
            print(f"no LoRA for {key}, skipping")
            continue
        set_peft_model_state_dict(model, load_file(path, device="cuda"), adapter_name="default")
    train = task["train"]
    for i, test in enumerate(task["test"]):
        bk = f"{key}_{i}"
        if bk in results or bk not in pools:
            continue
        ranked = score_kgmon(pools[bk])[:top_k]
        # held-out pairs whose prompt (with one extra demonstration) fits the context
        folds = []
        for j in range(len(train)):
            others = [p for k, p in enumerate(train) if k != j]
            q, a = prompt(others + [{"input": test["input"], "output": train[j]["output"]}], train[j]["input"], train[j]["output"])
            if len(tokenizer.encode(q + a)) <= max_seq_length:
                folds.append(j)
        if not folds:
            results[bk] = {"folds": [], "without": [], "candidates": {}}
            continue
        base_q, base_a = zip(*[prompt([p for k, p in enumerate(train) if k != j], train[j]["input"], train[j]["output"]) for j in folds])
        without = nll(list(base_q), list(base_a))
        entry = {"folds": folds, "without": without, "candidates": {}}
        for grid in ranked:
            cand = grid.tolist()
            qs, ans = zip(*[prompt([p for k, p in enumerate(train) if k != j] + [{"input": test["input"], "output": cand}], train[j]["input"], train[j]["output"]) for j in folds])
            entry["candidates"][tuple(map(tuple, grid))] = nll(list(qs), list(ans))
        results[bk] = entry
    pickle.dump(results, open(out_path, "wb"))
    print(f"{key}: {time.time() - start:.0f}s")
print("done", out_path)
