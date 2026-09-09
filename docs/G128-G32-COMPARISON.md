# G128 vs G32 Controlled Comparison

Two fresh GPTQ INT4 checkpoints were built directly from the same Qwen3.8-27B BF16 base model.

The experiment held constant:

- BF16 source model
- frozen calibration corpus
- exact calibration token stream
- GPTQModel version
- GPTQ parameters
- MTP exclusion rule
- vLLM/XPU serving stack
- MTP4 speculative decoding
- quality prompt

The controlled variable was:

```text
group_size
```

The tested values were G128 and G32.

## Model size

| Configuration | Quantized checkpoint size |
|---|---:|
| G128 | approximately 18.22 GB |
| G32 | approximately 19.54 GB |

G32 used approximately 1.32 GB more checkpoint space.

## G128 context capacity

G128 successfully served with:

```text
max_model_len=161000
gpu_memory_utilization=0.90
kv_cache_dtype=fp8
```

Observed startup values were approximately:

```text
Model loading memory: 17.38 GiB
Available KV cache:   6.26 GiB
GPU KV cache size:    166,750 tokens
```

This was sufficient for the tested 161,000-token context.

## G32 context capacity

G32 loaded successfully but could not start at 161,000 tokens with the same `gpu_memory_utilization=0.90` setting.

Observed values were approximately:

```text
Model loading memory: 18.45 GiB
Available KV cache:   5.0 GiB
KV required for 161K: 6.04 GiB
Estimated maximum:    128,128 tokens
```

The G32 quality test was therefore run with:

```text
max_model_len=128000
```

## Quality comparison

Both checkpoints completed the source-fidelity benchmark successfully enough for practical comparison.

The recurring conflicting-document case remained present in both models.

G32 therefore did not demonstrate a clear source-fidelity advantage sufficient to justify:

- approximately 1.32 GB greater checkpoint size
- higher VRAM pressure
- reduced KV-cache capacity
- reduction from 161K tested context to approximately 128K

## Performance comparison

The controlled thinking/source-fidelity runs showed no performance advantage large enough to offset G32's additional memory use.

In the measured quality runs:

```text
G128: approximately 48.6 tok/s in the initial sampled v2 run
G32:  approximately 46.2 tok/s in the sampled v2 comparison run
```

These figures are workload-specific and should not be compared directly with the separate short-context decode benchmark.

## Decision

**G128 is the preferred single-B70 checkpoint.**

It provides the best tested balance of:

- source-grounded answer quality
- checkpoint size
- VRAM use
- KV-cache capacity
- long-context capability
- generation performance

There is currently no strong evidence that another G32, G64, or similar group-size build would resolve the remaining reasoning edge case.
