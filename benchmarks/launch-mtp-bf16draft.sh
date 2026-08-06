#!/usr/bin/env bash
# A15c — native INT4 target + BF16-unquantized MTP draft patch
set -euo pipefail
LOG=/home/sergio/B70-DOCS/results/vllm-native-mtp-bf16draft-serve.log
MODEL_DIR=/mnt/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4
PATCH_V4=/home/sergio/B70-DOCS/scripts/tmp/vllm-xpu-int4-patch/patch_xpu_int4_moe_v4.py
PATCH_MTP=/home/sergio/B70-DOCS/scripts/tmp/vllm-xpu-int4-patch/patch_mtp_bf16_draft.py
SPEC_FILE=/tmp/b70-spec-mtp.json
printf '%s\n' '{"method":"mtp","num_speculative_tokens":1}' > "$SPEC_FILE"

echo 230000000 | sudo -n tee /sys/class/hwmon/hwmon4/power1_cap >/dev/null
systemctl --user stop llama-profile.service 2>/dev/null || true
sudo -n docker rm -f b70vllm 2>/dev/null || true
pkill -9 -f llama-server 2>/dev/null || true
sleep 16
pkill -9 -f llama-server 2>/dev/null || true
sleep 5
VRAM=$(sudo -n cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm | grep visible_avail | grep -oP '\d+' | head -1)
echo "VRAM=$VRAM"
RENDER_GID=$(stat -c "%g" /dev/dri/render* | head -n1)
: > "$LOG"

CID=$(sudo -n docker run -d --name b70vllm -p 8001:8000 \
  --device /dev/dri --group-add "${RENDER_GID}" \
  -v /dev/dri:/dev/dri:ro \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${PATCH_V4}:/patch_v4.py:ro" \
  -v "${PATCH_MTP}:/patch_mtp.py:ro" \
  -v "${SPEC_FILE}:/spec.json:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/intel/oneapi/2025.3/lib \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash intel/vllm:0.21.0-xpu-int4moe \
  -lc 'set -e; python /patch_v4.py; python /patch_mtp.py; SPEC=$(cat /spec.json); echo SPEC=$SPEC; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.92 --port 8000 --cudagraph-capture-sizes 1 2 4 8 16 32 --max-num-seqs 1 --max-num-batched-tokens 8192 --served-model-name Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 --language-model-only --speculative-config "$SPEC"')

echo "CID=$CID"
sudo -n docker logs -f b70vllm >>"$LOG" 2>&1 &
LPID=$!
for i in $(seq 1 55); do
  sleep 15
  if curl -s -m 3 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q '"id"'; then
    echo "UP after $((i*15))s"
    grep -E 'speculative_config|non-default args|B70|KeyError|ERROR|Graph capturing|native int4|MTP draft' "$LOG" | tail -50
    # keep log follower alive so post-startup crashes get captured
    exit 0
  fi
  ST=$(sudo -n docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' b70vllm 2>/dev/null || echo gone)
  echo "t=$((i*15))s status=$ST"
  if [[ "$ST" == exited* ]] || [[ "$ST" == gone* ]]; then
    echo DEAD
    grep -E 'KeyError|Error|error:|AssertionError|RuntimeError|B70|Traceback' "$LOG" | tail -60
    tail -50 "$LOG"
    kill $LPID 2>/dev/null || true
    exit 1
  fi
done
echo TIMEOUT
tail -50 "$LOG"
kill $LPID 2>/dev/null || true
exit 1
