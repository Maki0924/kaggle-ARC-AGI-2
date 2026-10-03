"""Where does a DFS step spend its time? Run inside the arc2-nvarc image:

    docker run --rm --gpus all -v <repo>/src/nvarc:/work:ro -v <repo>/scripts:/scripts:ro -v <repo>/data:/data:ro -v ~/arc2-local:/local \
        arc2-nvarc:local python /scripts/profile_dfs.py <task-id>
"""
import sys
import time

sys.path.insert(0, "/work")
from unsloth import FastLanguageModel  # noqa: E402
import torch  # noqa: E402
from peft import set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from arc_loader import ArcDataset, QwenFormatter  # noqa: E402
from arc_solver import ARC_TOKENS, inference_turbo_dfs, DFS_STATS  # noqa: E402
import numpy as np  # noqa: E402

key = sys.argv[1]
steps = 300
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/local/models/qwen3_4b_grids15_sft139", full_finetuning=False, load_in_4bit=False,
    local_files_only=True, use_gradient_checkpointing=False, max_seq_length=8192)
model = FastLanguageModel.get_peft_model(
    model, r=256, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=42, use_rslora=True, loftq_config=None)
for p in model.parameters():
    if p.dtype == torch.float32:
        p.data = p.data.to(torch.bfloat16)
set_peft_model_state_dict(model, load_file(f"/local/loras/stock/{key}.safetensors", device="cuda"), adapter_name="default")
model = FastLanguageModel.for_inference(model)

formatter = QwenFormatter(tokenizer=tokenizer)
ds = ArcDataset.from_file("/data/arc-agi_evaluation_challenges.json").change_keys([key]).split_multi_replies()
eval_ds = ds.augment(n=2, seed=2)
subkeys = [sorted(eval_ds.keys)[i] for i in (0, 1, 4, 5)]  # same pairing as the worker: equal lengths
tokens = [tokenizer.encode(eval_ds.get(k, formatter)["input"]) for k in subkeys]
assert len({len(t) for t in tokens}) == 1
arc_ids = torch.tensor(ARC_TOKENS, device="cuda")


@torch.inference_mode()
def greedy(label, repeat=1):
    """Greedy single-token steps with the same call signature the DFS uses."""
    input_ids = torch.tensor(tokens * repeat, device="cuda")
    bsz = input_ids.size(0)
    torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=input_ids, return_dict=True, use_cache=True)
    torch.cuda.synchronize(); prefill = time.time() - t0
    cache, pos = out.past_key_values, input_ids.size(1)
    nxt = arc_ids[out.logits[:, -1].index_select(-1, arc_ids).argmax(-1)]
    fwd = host = 0.0
    for _ in range(steps):
        t0 = time.time()
        out = model(input_ids=nxt.view(-1, 1), position_ids=torch.full((bsz, 1), pos, device="cuda"), past_key_values=cache, return_dict=True, use_cache=True)
        t1 = time.time()            # python + kernel launches
        torch.cuda.synchronize()
        t2 = time.time()            # wait for the GPU to finish
        host += t1 - t0; fwd += t2 - t0
        cache, pos = out.past_key_values, pos + 1
        nxt = arc_ids[out.logits[:, -1].index_select(-1, arc_ids).argmax(-1)]
    label = f"{label}, batch {bsz}"
    print(f"{label:28} prefix {input_ids.size(1)} tok  prefill {prefill:.2f}s  step {1000 * fwd / steps:.1f} ms  (host side {1000 * host / steps:.1f} ms, GPU wait {1000 * (fwd - host) / steps:.1f} ms)")


greedy("warmup")
greedy("LoRA unmerged")
greedy("LoRA unmerged", repeat=2)
greedy("LoRA unmerged", repeat=4)

t0 = time.time(); c0 = DFS_STATS["calls"]
with torch.inference_mode():
    res = inference_turbo_dfs(model, tokens, formatter.max_new_tokens(), -np.log(0.2), time.time() + 600)
calls = DFS_STATS["calls"] - c0
print(f"real turbo_dfs: {time.time() - t0:.1f}s for {calls} calls -> {1000 * (time.time() - t0) / max(calls, 1):.1f} ms per call")

t0 = time.time()
model.merge_adapter()
print(f"merge_adapter took {time.time() - t0:.2f}s; merged flags: {sum(getattr(m, 'merged', False) for m in model.modules())}")
greedy("LoRA merged")
greedy("LoRA merged", repeat=2)
greedy("LoRA merged", repeat=4)
print(f"peak VRAM {torch.cuda.max_memory_allocated() // 2**20} MB")
