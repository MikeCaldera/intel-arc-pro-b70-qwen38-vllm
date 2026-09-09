# Intel Arc Pro B70 + Qwen3.8-27B — vLLM XPU, GPTQ INT4, MTP4

Reproducible community work for running **Qwen3.8-27B** on a single **Intel Arc Pro B70 32 GB** with vLLM XPU.

This repository now documents two related tracks:

1. **Performance reproduction**
   - verified MTP4 throughput
   - 84.65 tok/s short-context reference result
   - verified long-context scaling
   - Intel XPU / vLLM deployment

2. **Fresh quality-focused GPTQ build**
   - quantization directly from the original BF16 model
   - controlled G128 vs G32 comparison
   - frozen calibration data
   - MTP4 serving
   - 161K tested context on G128
   - deterministic source-fidelity testing
   - documented failure cases and limitations

---

## Current recommended single-B70 quality configuration

| Setting | Value |
|---|---|
| Base model | Qwen3.8-27B BF16 |
| Quantization | GPTQ INT4 |
| Group size | **G128** |
| Symmetric | `true` |
| `desc_act` | `false` |
| MTP target weights | excluded from GPTQ quantization |
| Speculative decoding | **MTP4** |
| MTP draft | unquantized draft path |
| Context tested | **161,000 tokens** |
| KV cache | **FP8** |
| GPU memory utilization | `0.90` |
| Prefix caching | disabled |
| Max sequences | `1` |
| Max batched tokens | `8192` |
| Source-grounded prompt | brief-think v2 |
| Temperature | `0.0` |
| Top-p | `1.0` |
| Top-k | `-1` |
| Max generation | `4096` |

This is the best quality-focused configuration tested so far on one B70.

It is **not** claimed to be perfect or universally optimal.

---
## Ready-to-download model

Don't want to quantize the model yourself?

The tested fresh G128 GPTQ INT4 checkpoint is available directly on Hugging Face:

https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

No GPTQ quantization step is required.

This GitHub repository contains the full reproduction procedure, B70/vLLM XPU patches, calibration methodology, G128 vs G32 comparison, benchmarks, and deployment instructions.
## Performance reference

The earlier short-context MTP4 validation remains part of this project.

Verified result:

```text
Median decode: 84.65 tok/s
Mean decode:   84.49 tok/s
Prompt:        512 tokens
Generation:    128 tokens
```

That benchmark is separate from the newer thinking/source-fidelity benchmark. Do not directly compare the throughput numbers because the workloads differ.

The original detailed reproduction guide is preserved here:

[Original Qwen3.8 B70 replication guide](docs/ORIGINAL-QWEN38-B70-REPLICATION-GUIDE.md)

Additional benchmark detail:

[Qwen3.8-27B Intel Arc Pro B70 reproduction](benchmarks/QWEN38_B70_REPRODUCTION.md)

---

## Fresh GPTQ quantization work

The quality-focused checkpoints were quantized directly from the original Qwen3.8-27B BF16 model.

The controlled G128/G32 comparison kept constant:

- BF16 source model
- calibration corpus
- exact calibration token stream
- GPTQModel version
- GPTQ settings
- MTP exclusion rule
- serving software
- MTP4 configuration
- quality benchmark

Only the GPTQ group size changed.

### G128

```text
Quantized checkpoint: approximately 18.22 GB
Tested context:       161,000 tokens
```

### G32

```text
Quantized checkpoint: approximately 19.54 GB
Tested context:       128,000 tokens
```

At the same `gpu_memory_utilization=0.90`, G32 did not have enough remaining KV-cache memory to serve the same 161,000-token context.

G32 also did not show enough quality improvement to justify its additional memory use and reduced context capacity.

**G128 is therefore the preferred single-B70 configuration.**

Detailed procedures:

- [Fresh GPTQ quantization](docs/QUANTIZATION.md)
- [G128 vs G32 comparison](docs/G128-G32-COMPARISON.md)

---

## Frozen calibration and quality testing

The controlled G128/G32 quantization comparison used the same frozen calibration file:

```text
quantization/calibration/c4-fixed-128x1024.json
```

Calibration size:

```text
128 samples
1024 tokens per sample
131,072 total calibration tokens
```

SHA256:

```text
ddfc570e23458c048951501231c2ff75fa175440b120045bbeb1790bea5d2599
```

Verify locally with:

```bash
sha256sum quantization/calibration/c4-fixed-128x1024.json
```

Using the same calibration stream for both checkpoints keeps the group-size comparison controlled.

### Source-fidelity benchmark

The model was tested against 30 source-grounded prompts covering:

- conflicting documents
- missing information
- timeline ambiguity
- document hierarchy
- source provenance
- unsupported explanations

The preferred prompt is brief-think v2.

Recommended deterministic sampling:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 4096
}
```

Two back-to-back deterministic runs completed all 30 prompts with zero truncations.

Observed aggregate generation throughput was approximately:

```text
57.75 tok/s
```

This is a thinking/source-fidelity workload and should not be directly compared with the separate 84.65 tok/s short-context decode benchmark.

Detailed methodology:

- [Quality testing](docs/QUALITY-TESTING.md)

---

## Known source-conflict limitation

One benchmark case repeatedly produced a disputed interpretation.

The supplied records contained conflicting quantities, approximately:

```text
Invoice A:       240 units
Receiving Log B: 238 units
Document C:      no quantity
```

The benchmark expected the 240 vs 238 difference to remain unresolved.

The model repeatedly answered:

```text
238 units were definitively received.
```

This behavior occurred across:

- G128
- G32
- multiple prompt variants
- sampled decoding
- deterministic decoding

That makes a simple GPTQ group-size explanation unlikely.

There is also a benchmark-design caveat: an invoice quantity and a receiving-log quantity may describe different semantic facts rather than the exact same field.

A better future test should separate a true same-field contradiction from a shipped-versus-received distinction.

This limitation is documented rather than hidden.

---

## Production RAG recommendation

For higher-reliability document QA and RAG, use a deterministic evidence layer before LLM generation.

Recommended flow:

```text
source documents
    |
    v
extract facts
    |
    v
normalize fields and units
    |
    v
detect conflicts
    |
    v
classify evidence
    |
    +--> CONSISTENT
    +--> CONFLICTING
    +--> MISSING
    +--> AMBIGUOUS
    |
    v
construct grounded prompt
    |
    v
Qwen3.8-27B G128 + MTP4
    |
    v
final answer with source references
```

This is safer than relying on prompt engineering alone to identify every possible evidence conflict.

---

## Deployment guide

The recommended single-B70 quality serving profile is documented here:

[B70 quality deployment](docs/B70-QUALITY-DEPLOYMENT.md)

Tested profile:

```text
GPTQ INT4
G128
MTP4
161000 context
FP8 KV cache
gpu_memory_utilization=0.90
max_num_seqs=1
max_num_batched_tokens=8192
prefix caching disabled
```

---

## Repository structure

```text
.
├── README.md
├── Dockerfile.vllm-xpu-26.31
├── Dockerfile.gptqmodel-7.3.2-xpu
├── Dockerfile.gptqmodel-7.3.2-xpu-fallback-v2fixed
├── quantization/
│   ├── build-c4-calibration.py
│   ├── build-tokenized-calibration.py
│   ├── quantize-qwen38-gptq.py
│   └── calibration/
├── benchmarks/
│   └── quality/
│       ├── prompts/
│       ├── scripts/
│       └── results/
│           └── published/
└── docs/
    ├── ORIGINAL-QWEN38-B70-REPLICATION-GUIDE.md
    ├── QUANTIZATION.md
    ├── G128-G32-COMPARISON.md
    ├── QUALITY-TESTING.md
    └── B70-QUALITY-DEPLOYMENT.md
```

---

## Reproducibility checklist

Before claiming a reproduced result, record:

```text
[ ] GPU model
[ ] GPU driver
[ ] OS/kernel
[ ] Docker image or digest
[ ] vLLM version
[ ] vllm-xpu-kernels version
[ ] PyTorch version
[ ] Transformers version
[ ] GPTQModel version
[ ] base-model revision
[ ] calibration SHA256
[ ] quantization settings
[ ] group size
[ ] context length
[ ] KV-cache type
[ ] MTP speculative-token count
[ ] sampling settings
[ ] benchmark prompt file
[ ] max_tokens
[ ] benchmark result JSON
```

---

## Credits

This work builds on:

- **Qwen** — Qwen3.8-27B base model
- **GPTQModel** — GPTQ quantization
- **vLLM** — inference and OpenAI-compatible serving
- **Intel XPU ecosystem**
- **SergiioB / intel-arc-pro-b70-inference-cookbook** — Intel Arc Pro B70 vLLM/XPU/MTP groundwork

Upstream cookbook:

https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

This repository extends that groundwork with fresh GPTQ quantization, controlled G128/G32 comparison, long-context validation, deterministic quality testing, and source-fidelity analysis.

---

## License

See [LICENSE](LICENSE).
