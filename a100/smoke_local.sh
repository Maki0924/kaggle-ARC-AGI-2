#!/usr/bin/env bash
# Local smoke test of the A100 harness on one 4090 with Qwen3.5-4B (same family, served under the Qwen3.8 name so the
# harness checks pass). Inference runs from a staging copy without the answer files; scoring runs from the repo.
#   docker run --rm --gpus all --shm-size 16g -v <wheelhouse_cp311>:/wh:ro -v <Qwen3.5-4B>:/model:ro \
#     -v <repo>:/repo -w /repo arc2-nvarc:local bash a100/smoke_local.sh
set -euo pipefail
python -m pip install -q --no-index --find-links /wh --target /tmp/vllm_env vllm==0.19.0 2>&1 | grep -v "requires\|WARNING" | tail -2 || true
mkdir -p /tmp/vllm_env/sklearn && echo '__version__ = "0.0.0"' > /tmp/vllm_env/sklearn/__init__.py \
  && printf 'def roc_curve(*a, **k):\n    raise RuntimeError("shim")\n' > /tmp/vllm_env/sklearn/metrics.py

rm -rf /tmp/a100_smoke; mkdir -p /tmp/a100_smoke
MODEL=/model VENV=/tmp/vllm_env SEQS=4 CTX=32768 MEM=0.80 LOGDIR=/tmp/a100_smoke/logs bash /repo/a100/serve.sh

# the repo's data/ holds the solutions, as /kaggle/input does on Kaggle: the tool guard must hold (self-test at start)
RUN="python /repo/a100/qwen_ab.py --cohort screen --effort medium --budget 4096 --context-len 32768 --concurrency 4 \
     --servers /tmp/a100_smoke/logs/servers.json --out /tmp/a100_smoke/run"
$RUN --limit 3 | tee /tmp/a100_smoke/first.txt || true
grep -q "tool file guard verified: [1-9]" /tmp/a100_smoke/first.txt
$RUN || true                             # resume: runs the rest only
$RUN --effort high 2> /tmp/a100_smoke/mismatch.txt && { echo "FAIL: resumed with another condition"; exit 1; } || true
grep -q "different setup" /tmp/a100_smoke/mismatch.txt
python /repo/a100/score.py /tmp/a100_smoke/run
python - <<'EOF'
import collections, json
d = "/tmp/a100_smoke/run/"
turns = [json.loads(l) for l in open(d + "turns.jsonl")]
samples = [json.loads(l) for l in open(d + "samples.jsonl")]
keys = [(s["task_id"], s["query_index"], s["slot"]) for s in samples]
assert len(keys) == len(set(keys)), "duplicate samples after resume"
bad = [t for t in turns if t["usage_prompt_tokens"] != t["prompt_tokens"]]
assert not bad, f"/tokenize differs from usage.prompt_tokens: {bad[:3]}"
per = collections.Counter()
for t in turns: per[(t["key"], t["attempt"])] += t["completion_tokens"]
assert max(per.values()) <= 4096, f"budget exceeded: {max(per.values())}"
assert all(s["completion_tokens"] <= 4096 for s in samples)
print(f"OK: {len(samples)} samples, {len(turns)} turns, max sample tokens {max(per.values())}, "
      f"tool calls {sum(s['tool_calls'] for s in samples)} (errors {sum(s['tool_errors'] for s in samples)}), "
      f"finish {collections.Counter(s['finish'] for s in samples)}")
EOF
mkdir -p /repo/a100/smoke_out && cp /tmp/a100_smoke/run/{scores.json,samples.jsonl,turns.jsonl,manifest.json} /tmp/a100_smoke/logs/servers.json /repo/a100/smoke_out/
echo SMOKE PASSED
