# Fresh Qwen3.8-27B GPTQ Quantization

This document describes the fresh GPTQ INT4 build used for the quality-focused Intel Arc Pro B70 tests.

## Source model

Base model:

```text
Qwen/Qwen3.8-27B
```

The quantization was performed directly from the original BF16 checkpoint.

Do not use an already-quantized GPTQ, GGUF, AWQ, or other derivative if attempting to reproduce this experiment.

## Calibration

Final controlled calibration file:

```text
quantization/calibration/c4-fixed-128x1024.json
```

Configuration:

```text
128 samples
1024 tokens per sample
131,072 total calibration tokens
```

SHA256:

```text
ddfc570e23458c048951501231c2ff75fa175440b120045bbeb1790bea5d2599
```

Verify:

```bash
sha256sum quantization/calibration/c4-fixed-128x1024.json
```

## GPTQ configuration

```text
bits          = 4
group_size    = 128
sym           = true
desc_act      = false
lm_head       = false
quant_method  = gptq
pack_dtype    = int32
```

MTP tensors are excluded from GPTQ quantization:

```python
dynamic={
    "-:.*mtp.*": {}
}
```

Additional tested settings:

```text
GPTQModel               7.3.2
damp_percent            0.05
damp_auto_increment     0.01
static_groups           false
true_sequential         true
mse                     0
act_group_aware         true
fallback strategy       RTN
fallback threshold      0.5%
hessian chunk_size      32
hessian staging_dtype   float32
dense_vram_strategy     balanced
```

## Important GPU-memory rule

Stop large inference containers before quantization.

```bash
docker stop qwen38-quality-mtp4 2>/dev/null || true
```

During testing, another vLLM container occupying B70 VRAM caused apparent Hessian OOM failures. Once inference workloads were stopped and VRAM was available, the controlled quantizations completed on the intended XPU path.

## Build G128

```bash
cd ~/intel-arc-pro-b70-qwen38-vllm

rm -rf ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH

docker run --rm -i \
  --device /dev/dri \
  -v ~/models:/models \
  -v "$PWD:/work" \
  -w /work \
  --entrypoint python \
  gptqmodel-xpu-b70:7.3.2-xpu-fallback-v2fixed \
  quantization/quantize-qwen38-gptq.py \
    --group-size 128 \
    --output /models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH
```

Successful output should include approximately:

```text
Merged 15 tensors with prefixes [mtp.] into the state dict
Pre-Quantized model size: ~51.75 GB
Quantized model size:     ~18.22 GB
DONE
```

## Verify the checkpoint

```bash
du -sh ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH

ls -lh ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH

cat ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH/quantize_config.json
```

Confirm at minimum:

```text
bits        = 4
group_size  = 128
sym         = true
desc_act    = false
```

and confirm that the MTP exclusion rule is present.

## Build G32 for comparison

Use the exact same quantizer and frozen calibration data, changing only:

```text
--group-size 32
```

The tested G32 checkpoint was approximately 19.54 GB.
