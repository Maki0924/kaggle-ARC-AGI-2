#!/usr/bin/env bash
# Start one vLLM 0.19 server per group of TP GPUs for a100/qwen_ab.py (docs/A100_PLAN.md, steps 3-4).
#
#   MODEL=/data/qwen3-8-27b-fp8 VENV=/tmp/vllm_env TP=1 SEQS=8 bash a100/serve.sh
#
# GPUs   the ones in CUDA_VISIBLE_DEVICES (as assigned by the scheduler), else all GPUs nvidia-smi lists
# TP     tensor parallel size per server (1 on 80GB, 2 on 40GB); the GPU count must be a multiple of it
# SEQS   max concurrent sequences per server (= harness --concurrency / servers)
# MTP=1  add MTP speculative decoding (start without it first, per the plan)
# KVFP8=1  fp8 KV cache with the Triton backend (the L4 production setting; off on A100 by default)
# TOOLS=0  start without the tool-call parser (only for the tools-off condition)
# VENV   optional pip --target dir holding vLLM (the Kaggle cp311 wheelhouse install); launched with python -S
# Writes $LOGDIR/servers.json (pass it to qwen_ab.py --servers) and $LOGDIR/vllm_<port>.log. Default LOGDIR=server_logs.
set -euo pipefail
: "${MODEL:?set MODEL to the Qwen3.8-27B-FP8 directory}"
TP=${TP:-1}; SEQS=${SEQS:-8}; CTX=${CTX:-131072}; MEM=${MEM:-0.92}; PORT0=${PORT0:-8000}
NAME=${NAME:-Qwen/Qwen3.8-27B-FP8}; LOGDIR=${LOGDIR:-server_logs}; PY=${PY:-python}
mkdir -p "$LOGDIR"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
((${#gpus[@]} > 0 && ${#gpus[@]} % TP == 0)) || { echo "GPUs (${gpus[*]}) not a positive multiple of TP=$TP"; exit 1; }
nserv=$((${#gpus[@]} / TP))

flags=(--served-model-name "$NAME" --tensor-parallel-size "$TP" --max-model-len "$CTX" --gpu-memory-utilization "$MEM"
       --enable-prefix-caching --max-num-seqs "$SEQS" --reasoning-parser qwen3 --language-model-only --dtype bfloat16)
[ "${TOOLS:-1}" = 1 ] && flags+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${MTP:-0}" = 1 ] && flags+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
if [ "${KVFP8:-0}" = 1 ]; then flags+=(--kv-cache-dtype fp8 --attention-backend TRITON_ATTN); export VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN; fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export USE_TF=0 TRANSFORMERS_NO_TF=1   # Kaggle images ship TensorFlow; transformers importing it crashes vLLM's model inspection
pyargs=(); if [ -n "${VENV:-}" ]; then pyargs=(-S); export PYTHONPATH="$VENV" PATH="$VENV/bin:$PATH"; fi
up() { python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$1/v1/models', timeout=5)" 2> /dev/null; }

pids=(); devs_list=()
for ((s = 0; s < nserv; s++)); do
  port=$((PORT0 + s)); devs=$(IFS=,; echo "${gpus[*]:$((s * TP)):$TP}")
  if up "$port"; then echo "port $port already serves something; stop it first"; exit 1; fi
  echo "server $s: GPUs $devs port $port flags: ${flags[*]}"
  CUDA_VISIBLE_DEVICES=$devs nohup "$PY" "${pyargs[@]}" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --port "$port" "${flags[@]}" > "$LOGDIR/vllm_$port.log" 2>&1 &
  pids+=($!); devs_list+=("$devs")
done
for ((s = 0; s < nserv; s++)); do
  port=$((PORT0 + s)); t0=$SECONDS
  until up "$port"; do
    sleep 5
    if ! kill -0 "${pids[$s]}" 2> /dev/null; then echo "server on port $port died"; tail -30 "$LOGDIR/vllm_$port.log"; exit 1; fi
    if ((SECONDS - t0 > 3600)); then echo "port $port not ready after 1h; see $LOGDIR/vllm_$port.log"; exit 1; fi
  done
  kill -0 "${pids[$s]}" || { echo "server on port $port exited after answering"; exit 1; }
  echo "ready: port $port (pid ${pids[$s]}) after $((SECONDS - t0))s"
  grep -iE "quantiz|marlin|kv cache|maximum concurrency" "$LOGDIR/vllm_$port.log" | tail -5 || true
done

vllm_version=$("$PY" "${pyargs[@]}" -c "import vllm; print(vllm.__version__)" 2> /dev/null || echo unknown)
gpu_names=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort -u | paste -sd ";")
python3 - "$LOGDIR/servers.json" "$MODEL" "$vllm_version" "$gpu_names" "$PORT0" "${devs_list[@]}" -- "${flags[@]}" <<'EOF'
import json, sys
out, model, version, gpus, port0, *rest = sys.argv[1:]
cut = rest.index("--")
devs, flags = rest[:cut], rest[cut + 1:]
json.dump({"model_path": model, "vllm": version, "gpus": gpus, "flags": flags,
           "servers": [{"url": f"http://127.0.0.1:{int(port0) + s}", "gpus": d} for s, d in enumerate(devs)]},
          open(out, "w"), indent=1)
print("wrote", out)
EOF
