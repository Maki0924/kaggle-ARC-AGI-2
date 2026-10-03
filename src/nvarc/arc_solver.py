from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

import bz2
import json
import pickle
import traceback
import zlib

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Minimal performance patch: preserve the baseline beam set and ranking, but transfer
# only the 12 ARC-token NLL values to CPU instead of every Qwen vocabulary logit.
_ARC_TOKEN_ID_CACHE = {}

# Forward passes made by the DFS, for timing logs.
DFS_STATS = {"calls": 0}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    # Algebraically identical to: scores - logits.float().cpu().log_softmax(-1),
    # restricted to the same ARC_TOKENS used by the baseline DFS loop.
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < 540 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        DFS_STATS["calls"] += 1

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    # Keep logits on GPU and gather only the target-token scores. KV cache is not
    # consumed by teacher-forced scoring, so disabling it removes redundant writes.
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


def worker(rank, queue, end_time):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.getenv("ARC_MODEL_DIR", "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(0.2)

    data_dir = os.getenv("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
    if rerun_mode:
        test_path = os.path.join(data_dir, "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")

    arc_test_set = ArcDataset.from_file(test_path)

    work_dir = os.getenv("ARC_WORK_DIR", "/kaggle")
    dir_outputs = os.path.join(work_dir, "inference_outputs")
    os.makedirs(dir_outputs, exist_ok=True)

    n_train_aug = int(os.getenv("ARC_N_TRAIN_AUG", "16"))
    ttt_seed = int(os.getenv("ARC_TTT_SEED", "1"))  # stock: augmentation seed 1, trainer seed 42
    train_args["seed"] = 42 if ttt_seed == 1 else ttt_seed
    task_cap = float(os.getenv("ARC_TASK_CAP", "1200"))
    # Decode views: colour permutations per geometry (stock 2 -> 16 views) and views per forward pass (stock 4).
    n_eval_aug = int(os.getenv("ARC_N_EVAL_AUG", "2"))
    decode_batch = int(os.getenv("ARC_DECODE_BATCH", "4"))
    # Peak VRAM grows with prompt length x batch (about 2.3 MB per prompt token at batch 8), so long prompts fall back to 4.
    big_batch_max_prefix = int(os.getenv("ARC_BIG_BATCH_MAX_PREFIX", "4200"))
    # Fold the LoRA into the base weights for decoding; the base weights are restored exactly after each puzzle.
    merge_lora = os.getenv("ARC_MERGE_LORA", "0") == "1"
    pristine = {}
    if merge_lora:
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
                pristine[name] = (module, module.base_layer.weight.detach().cpu().clone())

    # Stock behaviour only checks the cap between decode batches, so a puzzle can overrun it by a whole DFS.
    # With ARC_STRICT_CAP=1 the cap is also the deadline inside the DFS.
    strict_cap = os.getenv("ARC_STRICT_CAP", "0") == "1"

    # Per-puzzle LoRA weights are kept here (about 1 GB each) so later runs can skip test-time training.
    lora_dir = os.getenv("ARC_LORA_DIR", "")
    if lora_dir:
        os.makedirs(lora_dir, exist_ok=True)

    def solve(key):

        nonlocal model

        info = {}

        start_time = time.time()
        
        torch.cuda.reset_peak_memory_stats()

        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        puzzle_ds = arc_test_set.change_keys([key])

        lora_path = os.path.join(lora_dir, f"{key}.safetensors") if lora_dir else ""

        if lora_path and os.path.exists(lora_path):

            set_peft_model_state_dict(model, load_file(lora_path, device="cuda"), adapter_name="default")
            stats = f"loaded {lora_path}"
            info["lora_loaded"] = True

        else:

            model = FastLanguageModel.for_training(model)


            train_ds = puzzle_ds.augment(n=n_train_aug, shfl_keys=True, seed=ttt_seed)
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            
                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )

                stats = trainer.train()

                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

                del trainer

            if lora_path:
                trained = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
                save_file(trained, lora_path + ".tmp")
                os.replace(lora_path + ".tmp", lora_path)
                del trained

        model = FastLanguageModel.for_inference(model)

        info["train_seconds"] = time.time() - start_time
        
        gc.collect()
        torch.cuda.empty_cache()
            
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

        torch.cuda.reset_peak_memory_stats()
        
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        eval_ds = puzzle_ds_multi.augment(n=n_eval_aug, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        stock_batching = n_eval_aug == 2 and decode_batch == 4

        batches = []

        if not stock_batching:
            # Views are batched by prompt length (a forward pass costs about the same for 4 or 8 rows).
            # A transposed view has the same shape as a view rotated by 90 degrees, so they share batches.
            for test_id, subkeys in test_id_to_subkeys.items():
                by_length = defaultdict(list)
                for subkey in subkeys:
                    by_length[len(tokenizer.encode(eval_ds.get(subkey, formatter)["input"]))].append(subkey)
                for length, group in by_length.items():
                    size = decode_batch if length <= big_batch_max_prefix else min(decode_batch, 4)
                    for i in range(0, len(group), size):
                        batches.append(group[i:i+size])

        for test_id, subkeys in (test_id_to_subkeys.items() if stock_batching else []):
            # 0: permute x 2
            # 4: rot90.rot90.permute x 2
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 2: permute.rot90 x 2
            # 6: rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
        for test_id, subkeys in (test_id_to_subkeys.items() if stock_batching else []):
            # 8: transpose.permute x 2
            # 12: transpose.rot90.rot90.permute x 2
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 10: transpose.rot90.permute x 2
            # 14: transpose.rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)

        if merge_lora:
            model.merge_adapter()

        with torch.inference_mode():
                
            known_scores = {}

            for subkeys in batches:

                spend_time = time.time() - start_time
                if spend_time > task_cap or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    info["timeout"] = True
                    break

                print(f"[Rank {rank}] decoding {subkeys}")

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_start, dfs_calls = time.time(), DFS_STATS["calls"]

                dfs_end_time = min(end_time, start_time + task_cap) if strict_cap else end_time

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, dfs_end_time)

                info.setdefault("dfs", []).append({
                    "seconds": round(time.time() - dfs_start, 1),
                    "calls": DFS_STATS["calls"] - dfs_calls,
                    "prefix_tokens": max(len(t) for t in tokens),
                    "beams": sum(len(b) for _, b in dfs_result),
                })
                score_start = time.time()

                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    for beam_score, tokens in scored_beams:

                        array = formatter.convert_tokens_to_array(tokens)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=zlib.crc32(bk.encode()) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores
                        
                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                        })

                    if len(decoded_result):
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

                info["dfs"][-1]["score_seconds"] = round(time.time() - score_start, 1)

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")

        info["n_batches"] = len(batches)
        info["n_scored_grids"] = len(known_scores)
        info["inference_mb"] = memory_allocated
        return info

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        task_start = time.time()

        # One failing puzzle (e.g. OOM) must not take the whole GPU out for the rest of the run.
        try:
            info = solve(key)
            info["status"] = "ok"
        except Exception as e:
            print(f"[Rank {rank}] FAILED {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            info = {"status": f"error:{type(e).__name__}"}
            gc.collect()
            torch.cuda.empty_cache()

        if merge_lora:
            # Unmerging in bf16 would leave rounding error in the base model, so the saved weights are copied back.
            model.unmerge_adapter()
            with torch.no_grad():
                for module, weight in pristine.values():
                    module.base_layer.weight.copy_(weight)

        info.update(key=key, rank=rank, seconds=time.time() - task_start)
        with open(os.path.join(work_dir, f"timing_rank{rank}.jsonl"), "a") as f:
            f.write(json.dumps(info) + "\n")
