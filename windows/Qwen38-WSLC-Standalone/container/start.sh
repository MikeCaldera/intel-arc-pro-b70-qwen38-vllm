#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/model}"
MODEL_NAME="${MODEL_NAME:-qwen38}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MTP_TOKENS="${MTP_TOKENS:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_TOOLS="${ENABLE_TOOLS:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

python /opt/qwen38/diagnose.py
python /opt/qwen38/patch_mtp_nightly.py
python /opt/qwen38/patch_mtp_boundary.py
python /opt/qwen38/patch_xpu_single_gpu_warmup.py
python /opt/qwen38/prepare_low_reasoning_template.py

args=(
  --quantization gptq \
  --dtype float16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --port 8000 \
  --host 0.0.0.0 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-enable-prefix-caching \
  --served-model-name "$MODEL_NAME" \
  --language-model-only \
  --chat-template /tmp/qwen38_chat_template_low_reasoning.jinja \
  --reasoning-parser qwen3
)

if (( KV_CACHE_MEMORY_BYTES > 0 )); then
  args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if (( ENABLE_TOOLS > 0 )); then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if (( MTP_TOKENS > 0 )); then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

exec vllm serve "$MODEL_PATH" "${args[@]}"
