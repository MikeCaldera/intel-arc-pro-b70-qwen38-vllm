# Intel Arc Pro B70 Quality Deployment

This document describes the tested quality-focused serving configuration for the fresh Qwen3.8-27B GPTQ INT4 G128 checkpoint on a single Intel Arc Pro B70 32 GB GPU.

## Tested serving profile

```text
Model:                  Qwen3.8-27B
Quantization:           GPTQ INT4
Group size:             G128
Speculative decoding:   MTP4
Context:                161000
KV cache:               FP8
GPU memory utilization: 0.90
Max sequences:          1
Max batched tokens:     8192
Prefix caching:         disabled
```

The 161,000-token context value was validated on the tested system. It is not a guarantee that every B70 host will reach the same limit.

## Model path

```text
~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH
```

## Launch vLLM XPU

From the repository root:

```bash
docker rm -f qwen38-quality-mtp4 2>/dev/null || true

docker run -d \
  --name qwen38-quality-mtp4 \
  --entrypoint bash \
  --device /dev/dri:/dev/dri \
  -v /dev/dri:/dev/dri \
  -v ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH:/model:ro \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 11444:8000 \
  vllm-xpu-b70:26.31-test \
  -lc "python /patch_mtp.py && python /patch_boundary.py && exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 161000 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --language-model-only --enable-auto-tool-choice --tool-call-parser hermes --served-model-name qwen38 --seed 0 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"
```

## Verify startup

```bash
docker logs -f qwen38-quality-mtp4
```

Expected indicators include:

```text
Using XPUwNa16LinearKernel for AutoGPTQLinearMethod
[B70] MTP draft: forcing unquantized build
Detected MTP model
Application startup complete
```

Approximate G128 startup values from the tested system:

```text
Model loading memory: 17.38 GiB
Available KV cache:   6.26 GiB
GPU KV cache size:    166,750 tokens
```

## Verify API

```bash
curl http://127.0.0.1:11444/v1/models
```

## Deterministic source-grounded profile

```text
temperature=0.0
top_p=1.0
top_k=-1
max_tokens=4096
```

## Run the quality benchmark

```bash
cd ~/intel-arc-pro-b70-qwen38-vllm/benchmarks/quality

python3 scripts/run-source-fidelity-30-thinking-deterministic.py \
  --config mtp4 \
  --port 11444 \
  --prompts prompts/desktop-source-fidelity-30-briefthink-v2.json \
  --max-tokens 4096
```

Expected completion:

```text
Tests: 30
Errors: 0
Truncations: 0
```

## Troubleshooting

If 161K does not fit, confirm no other large GPU workload is using the B70, confirm the G128 checkpoint is loaded, inspect available KV-cache memory, and reduce `max_model_len` if necessary.
