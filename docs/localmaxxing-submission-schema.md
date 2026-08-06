# localmaxxing.com submission schema

The schema for `lmx benchmark submit` / the curl fallback. Flat JSON, single
object per submission.

## Schema (flat, working)

```json
{
  "hfId": "Qwen/Qwen3.6-35B-A3B",
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "gpuName": "Intel Arc Pro B70",
    "gpuCount": 1,
    "vramGb": 32.0,
    "os": "Linux"
  },
  "engineName": "vllm",
  "engineVersion": "0.21.1.dev18 XPU",
  "quantization": "GPTQ-Int4",
  "backend": "xpu",
  "tokSOut": 123.3,
  "tokSPrefill": 7261.0,
  "contextLength": 16384,
  "batchSize": 1,
  "notes": "Single-stream. vLLM 0.21 XPU + 4 in-container patches (github.com/SergiioB/intel-arc-pro-b70-inference-cookbook). MTP speculative, 1 layer, num_spec=1. q8_0/q4_1 KV-equivalent. 150W. KL audit vs eager pending.",
  "engineFlags": {
    "commandSnippet": "vllm serve /model --quantization gptq --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 1 --language-model-only --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1} --cudagraph-capture-sizes 1 2 4 8 16 32",
    "kvCacheDtype": "auto",
    "flashAttn": true,
    "numParallel": 1
  }
}
```

## Required fields

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `hfId` | string | `Qwen/Qwen3.6-35B-A3B` | HuggingFace org/model. Must match leaderboard model registry. |
| `hardware` | object | (see above) | `hwClass` ∈ {DISCRETE_GPU, IGPU, ...}. `gpuName` must match leaderboard hw registry. |
| `engineName` | string | `vllm` or `llama.cpp` | |
| `engineVersion` | string | `0.21.1.dev18 XPU` | Include build/commit info. |
| `quantization` | string | `GPTQ-Int4` / `Q4_K_XL` / `Q5_K_M` | |
| `backend` | string | `xpu` | |
| `tokSOut` | float | `123.3` | **Single-stream decode t/s** (engine rate, not wall-clock). |
| `tokSPrefill` | float | `7261.0` | Prefill t/s (≥1024-token prompt to be valid). |
| `contextLength` | int | `16384` | Max context the server was configured for. |
| `batchSize` | int | `1` | 1 for single-stream. |
| `notes` | string | | **Disclose patches + setup.** Reproducibility matters. |
| `engineFlags.commandSnippet` | string | | The exact launch command. |
| `engineFlags.kvCacheDtype` | string | `q8_0/q4_1` / `auto` | |
| `engineFlags.flashAttn` | bool | `true` | |
| `engineFlags.numParallel` | int | `1` | |

## Leaderboard weights (as of Jul 2026)

VRAM 30% / Model size 30% / Run count 25% / TPS 15%.

So submitting **multiple quants/models** (each valid run) helps the score more
than one high-TPS run. The vLLM MTP run is the standout (high TPS + large model
+ 32GB VRAM), but the llama.cpp MoE + dense runs add run-count weight.

## Honesty rules

localmaxxing values reproducibility. Every submission MUST disclose:

- **Patches applied** — link the repo if non-stock engine.
- **Speculative decoding** — note MTP/ngram + acceptance if known.
- **Power cap** — note if non-stock.
- **Single-stream vs concurrent** — `tokSOut` is single-stream. If you measured
  multi-user aggregate, that goes in notes, NOT `tokSOut`.

The vLLM MTP submission notes the 4 patches + pending KL audit. That's honest
and still a strong submission — the patches are open and reproducible.

## POST endpoint (curl fallback)

```bash
curl -sS -X POST https://www.localmaxxing.com/api/benchmarks \
  -H "Authorization: Bearer $LOCALMAXXING_API_KEY" \
  -H "Content-Type: application/json" \
  -d @submission.json
```

`LOCALMAXXING_API_KEY` (format `bhk_...`) from localmaxxing.com account settings.
Never commit the key.
