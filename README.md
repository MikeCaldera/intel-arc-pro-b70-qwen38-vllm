# Intel Arc Pro B60 / B70 Inference Cookbook 🚀

> Open recipes, engine patches, and benchmark harnesses for running LLMs on
> Intel Arc Pro B-series (Battlemage / Xe2) GPUs — **MoE 35B at 204.6 t/s decode
> (MTP4) and ~8.4K t/s prefill, single-stream, one card.**

[![Benchmark](https://img.shields.io/badge/MoE%20decode-204.6%20t%2Fs-10b981)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![Prefill](https://img.shields.io/badge/MoE%20prefill-8.4K%20t%2Fs-0ea5e9)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes%20please-22c55e)](#why-open)

> **2026-08-08 real-world update:** I reran the stack with the exact Pi system prompt across short chat, multi-turn, RAG append, cold/resident documents, mixed traffic, and exact long-context boundaries. Boundary-patched MTP4 now completes **p130944/g128 = 131,072 total tokens**. Read the [full real-world report and reproduction steps](docs/REAL-WORLD-PI-BENCHMARKS.md).

> **Image correction:** `intel/vllm:0.21.0-xpu-int4moe` was a local derived research image and was never published to Docker Hub. It is not a pullable prerequisite. The current recipe uses a public `vllm/vllm-openai-xpu` nightly pinned by digest. See the [image and patch matrix](docs/IMAGE-AND-PATCH-MATRIX.md).

## The headline

After a 24-run benchmark campaign (plus the MTP spec-token and prefill sweeps
below), here's what one Intel Arc Pro B70 (32 GB, Battlemage, ~€1,100 /
~$1,200) actually does on Qwen3.6-35B-A3B (MoE):

| Metric | vLLM XPU + MTP4 | vLLM + MTP1 | llama.cpp SYCL | vLLM best advantage |
|--------|---------------:|------------:|---------------:|-------------------:|
| **Decode (short/g32)** | **204.6 t/s** | 137.4 t/s | 74 t/s | **2.8×** |
| **Decode (p8k)** | **159.6 t/s** | 119.7 t/s | 58 t/s | **2.75×** |
| **Prefill (p4k)** | **8,153 t/s** | ~7,100 t/s | 1,662 t/s | **4.9×** |
| Power (decode) | 150-165W | 150W | 150W | — |
| Temp | 58°C | 58°C | 58°C | — |

> **Power note (corrected 2026-08-07):** prefill numbers are measured at a fixed
> 150W cap. An earlier version of this table claimed 9.0K t/s "MTP1@230W" as a
> power-boost effect — that was **prefix-cache contamination** (a constant-filler
> benchmark harness inflated prefill ~5× once cached), not a real power gain. A
> paired, alternating 150W-vs-230W A/B on the same warm server came back **flat at
> ±0.2%** across p2k/p4k/p8k. See the Power section below.

**The MTP spec-token lever:** the draft head is *recurrent* (`spec_step_idx %
mtp_num_hidden_layers`), so a single `mtp_num_hidden_layers: 1` layer emits
N draft tokens per step. `num_speculative_tokens` is NOT clamped by layer count
— bump it:
- **MTP4 = peak decode** (short/32: **204.6 t/s**, +49% vs MTP1; beats the
  community "145 t/s" claim by 41%). Prefill pays for it (-11%).
- **MTP2 = balanced** (decode +19% vs MTP1, prefill only -3-4%).
- **MTP1 = highest acceptance / least MTP overhead** (97.1% accept, but lowest
  throughput).

**Acceptance falls with N — that's fundamental, not a tuning problem.** Per
draft position (read directly from the `vllm:spec_decode_*` counters): pos0
92.6% → pos1 84.2% → pos2 72.7% → pos3 68.9% at N=4 (80.1% overall). Each draft
token is an autoregressive guess off the previous guess, so errors compound.
**Chasing 97-99% acceptance (N=1) costs 30% throughput** (137.4 vs 204.6 t/s).
For a single-layer draft head, throughput is the right objective, not
acceptance %. Combining high-N + high-acceptance needs a multi-layer draft
model (DeepSeek-V3-style), which this checkpoint lacks.

Full methodology + all grids: **[vLLM vs llama.cpp — The Full MoE + Dense Showdown](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)**.

## Concurrency (multi-user throughput)

Single-stream is one thing — serving many users at once is where vLLM's
continuous batching shines. Two measurements, both no-MTP native int4 v4
(MTP + concurrency is blocked by the GDN kernel):

**Quick sweep (64-token gens, max-num-seqs=64):**

| Concurrent users | Wall-agg tok/s |
|-----------------:|---------------:|
| 16 | 804.8 |
| 32 | 1,119.8 |
| 48 | 904.1 |
| **64** | **1,196.9** |

**Realistic (diverse 512-token prompts, median TPOT):**

| Concurrent users | Aggregate t/s | Gen t/s | Median TPOT |
|-----------------:|--------------:|--------:|------------:|
| 16 | 1,107.8 | 641.7 | 25ms |
| 32 | 1,523.8 | 882.7 | 36ms |
| 48 | 1,578.2 | 914.2 | 52ms |
| **64** | **1,967.7** | **1,139.8** | **56ms** |

**Max aggregate: 1,967.7 t/s @ C64** (1,139.8 gen t/s) with 64 concurrent
users, diverse 512-token prompts, median TPOT 56ms. The "145 t/s" community
single-stream claim sits far below vLLM's multi-user aggregate. Community
dual-B70 runs hit [912 tok/s at 50 users](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks) — we exceed that on a single B70.

*Note: this is the no-MTP path (Run 17/19). MTP + concurrency is **blocked** —
the XPU GDN `causal_conv1d` kernel rejects mixed spec/non-spec batches (Run 23),
so C2+ with MTP on crashes. The no-MTP concurrency numbers above are the
multi-user ceiling until that kernel is fixed.*

## Model reference

The vLLM path uses a specific checkpoint — not the stock Qwen release:

| Property | Value |
|----------|-------|
| **Architecture** | Qwen3.6-35B-A3B (MoE: 256 experts, 8 active + 1 shared, 3B active params) |
| **HF repo** | [`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4`](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4) |
| **Base model** | Qwen/Qwen3.6-35B-A3B (BF16, Apache 2.0) |
| **Post-train** | Heretic (uncensored/abliterated) + Native MTP-Preserved |
| **Quantization** | INT4 weights via GPTQ calibration (group_size=128, sym, desc_act=false) |
| **Quant tool** | GPTQModel 7.1.0-dev ([github.com/modelcloud/gptqmodel](https://github.com/modelcloud/gptqmodel)) |
| **MTP layer** | 1 layer, preserved at BF16 (not quantized) |
| **Size** | 21 GB (6 safetensors shards) |

> **Why this checkpoint?** The heretic variant from llmfan46 was the only
> GPTQ-Int4 with **preserved MTP weight tensors** at the time of benchmarking.
> The official `Qwen/Qwen3.6-35B-A3B-GPTQ-Int4` declares
> `mtp_num_hidden_layers: 1` in config but ships **zero MTP weight tensors**
> in the shards — MTP speculative decoding doesn't work without them. The
> heretic variant preserves these tensors, enabling the 133 t/s decode path.
> Quantization quality (KLD) is architecture-determined, not weight-determined
> — the heretic abliteration doesn't change how INT4 quantization affects the
> model's output distribution. See
> [docs/QUANTIZATION-QUALITY.md](docs/QUANTIZATION-QUALITY.md) for details.
>
> **INT4 is the Intel-optimized format.** The B70's XMX (Xe Matrix Extension)
> engines have native INT4 grouped GEMM support — this is Intel's equivalent
> of NVIDIA's NVFP4 on Tensor Cores. GPTQ is the algorithm that computes the
> INT4 weights (Hessian-based calibration); INT4 is the data format XMX
> accelerates. Any INT4 weights (GPTQ, AWQ, RTN) hit the same XMX fast path.
> See [research/quantization-format-strategy.md](research/quantization-format-strategy.md)
> for the full format comparison (GPTQ-Int4 vs MXFP4 vs FP8 vs GGUF K-quants).

The llama.cpp path uses Unsloth Dynamic GGUF quants of the same architecture:
[`Qwen3.6-35B-A3B-UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
(21 GB) and Q5_K_M (25 GB).

For a detailed KL-divergence comparison between GPTQ-Int4 and GGUF K-quants,
see **[docs/QUANTIZATION-QUALITY.md](docs/QUANTIZATION-QUALITY.md)**.

## What's in this repo

```
patches/
  patch_xpu_int4_moe_v4.py   ← historical vLLM 0.21 native-int4 load patch
  patch_mtp_bf16_draft.py    ← historical vLLM 0.21 MTP patch
  patch_mtp_nightly.py       ← pinned nightly BF16 MTP draft patch
  patch_mtp_boundary.py      ← exact-128K partial final MTP4 group fix
benchmarks/
  launch-mtp-bf16draft.sh    ← historical vLLM 0.21 launch
  launch-mtp4-128k-nightly.sh ← pinned nightly, both patches, 131072 context
  b70-generate-exact-prompts.py ← exact rendered-token calibration
  b70-realworld-context-harness.py ← TTFT/TTFC/TPOT/cache/MTP recorder
  b70-sync-monitor.py        ← energy-counter and named-temperature monitor
  pi-system-prompt.txt       ← exact public Pi workload prefix
docs/
  REAL-WORLD-PI-BENCHMARKS.md ← 2026-08-08 tables + exact reproduction
  IMAGE-AND-PATCH-MATRIX.md ← exact public image, version, and compatible patches
  POWER-SWEET-SPOTS.md       ← MoE=150W, Dense=180W
  CAMPAIGN-LOG.md            ← historical campaign narrative
  QUANTIZATION-QUALITY.md    ← GPTQ-Int4 vs GGUF K-quant analysis
results/
  realworld-pi-20260808-summary.json ← machine-readable public summary
```

## Quick start: current public nightly + MTP4 + exact 128K

**You need:** an Arc Pro B70, Docker, working `/dev/dri` access, and the MTP-preserved GPTQ-INT4 model:

```text
llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
```

Pulling `intel/vllm:0.21.0-xpu-int4moe` will fail because that historical tag was local and never published. Do not substitute `intel/vllm:0.21.0-xpu`; the old patches target different source.

The current launcher pins this public image:

```text
vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97
```

Observed inside: vLLM `v0.26.1rc1.dev457+gc810e5ee9`, `vllm-xpu-kernels 0.1.12`.

```bash
git pull
bash benchmarks/launch-mtp4-128k-nightly.sh \
  /path/to/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 8000
sudo docker logs -f b70-mtp4-128k
```

The launcher applies, in order:

1. `patches/patch_mtp_nightly.py`
2. `patches/patch_mtp_boundary.py`

Do not apply the historical v0.21 patches to this nightly. Full image/patch compatibility: [`docs/IMAGE-AND-PATCH-MATRIX.md`](docs/IMAGE-AND-PATCH-MATRIX.md). Exact p130944/g128 reproduction: [`docs/REAL-WORLD-PI-BENCHMARKS.md`](docs/REAL-WORLD-PI-BENCHMARKS.md).

## Current and historical patch sets

### Current pinned nightly

| Order | Patch | Fixes |
|---:|---|---|
| 1 | `patch_mtp_nightly.py` | Builds the checkpoint's preserved BF16 MTP draft outside the target GPTQ quant config. |
| 2 | `patch_mtp_boundary.py` | Handles the exact-128K partial final MTP4 group without padding or shortening output. |

### Historical vLLM 0.21 research path

| # | Patch | Fixes |
|---:|---|---|
| 1 | `patch_xpu_int4_moe_v4.py` | GPTQ uint8/int8 native-int4 load mismatch. |
| 2–4 | `patch_mtp_bf16_draft.py` | BF16 draft construction, obsolete MoE kwargs, and a metadata-only GDN guard. |

The historical `intel/vllm:0.21.0-xpu-int4moe` image was local and is not available on Docker Hub. These files remain for evidence preservation; they are not the current quick start.

## Power sweet spots (don't waste watts)

| Workload | Power cap | Why |
|----------|----------|-----|
| **MoE 35B (decode AND prefill)** | **150-165W** | Self-limits to ~140W draw; 230W gives -8% decode at +80W heat. Prefill is flat too — see the A/B below. |
| **Dense 27B** | **180W** sustained / 230W burst | Scales +18–30% (150→230W), but thermal cost (71→79°C). |

Set it once:
```bash
# 150W for MoE (decode + prefill both flat above this; 230W = pure waste heat)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap
```

### MoE prefill power A/B (the claim that didn't hold up)

An earlier version of this cookbook claimed prefill scaled +16-22% from
150W→230W on MoE+MTP. That came from **unpaired runs** with a constant-filler
harness whose reps 2-3 hit prefix caching and reported inflated prefill
(observed: p8k rising 8,640 → 42,733 t/s once cached — a 5× lie). A paired,
**alternating** A/B on the same warm server, cooled <52°C between rounds,
3 rounds, honest cold prefill (unique random prefix per call) returned:

| prompt | 150W mean | 230W mean | delta |
|--------|-----------|-----------|-------|
| p2k    | 7,216 t/s | 7,207 t/s | −0.1% |
| p4k    | 8,140 t/s | 8,135 t/s | −0.1% |
| p8k    | 8,384 t/s | 8,403 t/s | +0.2% |

**Flat at ±0.2%.** Live card-draw telemetry explains why: prefill p8k draws
~171W, decode ~113W, idle ~47W — so at a 165W cap the prefill is already
uncapped, and the grouped-GEMM is **bandwidth-gated, not power-gated**. Raising
the cap is pure waste heat on this workload. (Dense llama.cpp decode is the
opposite — it genuinely scales +18-30% from 150→230W; different workload.)

> **No clock locking either.** The B70 runs the `xe` driver (not i915), which
> exposes no clock-control sysfs — only read-only PMU counters. `xpu-smi` and
> `intel_gpu_frequency` both fail on this stack. The hwmon power cap is the only
> tunable, and as the A/B shows, it doesn't move MoE prefill.

## Consolidated results (Qwen3.6-35B-A3B-MTP GPTQ-Int4, one B70)

**Config:** vLLM 0.21.0-xpu-int4moe, native INT4 v4 + BF16 MTP draft,
single-stream, honest cold prefill (unique random prefix per call). All @165W.

### Decode (t/s) — grid @165W, steady-state

| Prompt → Gen | MTP1 | MTP2 | MTP4 |
|-------------|-----:|-----:|-----:|
| short → 32 | 137.4 | 152.3 | **204.6** |
| short → 64 | 134.8 | 155.2 | **190.6** |
| short → 128 | 130.9 | 145.6 | **175.7** |
| p1k → 64 | 127.7 | 151.8 | **179.8** |
| p2k → 64 | 125.9 | 139.1 | **175.8** |
| p4k → 64 | 121.8 | 141.2 | **173.1** |
| p8k → 64 | 118.5 | 137.1 | **165.9** |
| p8k → 128 | 119.7 | 133.4 | **159.6** |

### Prefill (t/s) — cold, no prefix cache, @165W

> **Corrected 2026-08-07:** the previous table here split MTP1@165W vs
> MTP1@230W with the 230W column showing 8.5-9.0K t/s. That was prefix-cache
> contamination, not a power effect (see the Power A/B above). Power is NOT a
> prefill lever on this workload. Honest re-measured prefill at 165W:

| Prompt | MTP1 @165W | MTP4 @165W |
|--------|-----------:|-----------:|
| p500 | ~5,000 | ~5,070 |
| p1k | ~6,900 | ~6,890 |
| p2k | ~7,260 | ~7,265 |
| p4k | ~8,150 | ~8,150 |
| p8k | ~8,390 | ~8,380 |

### Power / thermal @165W cap
| Config | card avg | card peak | prefill peak | temp pkg avg/peak |
|--------|---------:|----------:|-------------:|------------------:|
| MTP1 | 153.4W | 181.8W | ~237W | 66/73°C |
| MTP2 | 157.5W | 188.5W | — | 67/72°C |
| MTP4 | 159.8W | 190.5W | ~237W | 66/72°C |

### Headlines
- **MTP4 short/32 decode 204.6 t/s** (+49% vs MTP1, +41% vs community 145 claim)
- **MTP4 prefill p4k ~8.15K t/s** at 150W (power is NOT a lever here — see A/B)
- **Spec-N curve:** N=4 is the throughput optimum; N=1 reaches 97.1% acceptance
  but costs 30% throughput. Throughput, not acceptance %, is the goal.

### vs the "custom image + kernel" Reddit claim
| Metric | Our MTP4 (stock image + 2 patches) | Custom-build claim |
|--------|-----------------------------------:|-------------------:|
| tg32 decode | **204.6 t/s** | 174.54 ± 13.05 |
| pp4096 prefill | ~8.15K t/s | 9,268 ± 39.69 |

MTP4 beats his decode by 17%; his custom kernel edges us on prefill (~13%) — the
gap to close with a custom kernel.

## Real-world Pi and exact 128K update

The synthetic grids above measure clean engine surfaces. The 2026-08-08 campaign adds deployment-like Pi coordinates with actual endpoint tokens, TTFT, TTFC, post-first TPOT, cache deltas, MTP counters, power, thermals, and failures.

### Exact long-context MTP4

| Prompt | Output | Total | TTFT (s) | Client post-first (tok/s) | MTP accept | Result |
|---:|---:|---:|---:|---:|---:|---|
| 16,256 | 128 | 16,384 | 2.403 | 161.23 | 85.34% | Completed |
| 32,640 | 128 | 32,768 | 5.785 | 139.99 | 71.21% | Completed |
| 65,408 | 128 | 65,536 | 15.093 | 111.17 | 58.55% | Completed |
| 98,176 | 128 | 98,304 | 28.078 | 117.36 | 76.56% | Completed |
| 122,880 | 128 | 123,008 | 44.057 | 95.13 | 62.84% | Completed |
| **130,944** | **128** | **131,072** | **48.601** | **96.87** | **72.31%** | **Completed with patch 5** |

`Client post-first` is `(completion_tokens - 1) / (request_end - first_token)`, not an engine-native timing field. At exact 128K, one matched MTP2 observation reached 103.63 tok/s with 86.96% acceptance. MTP4 now works, but MTP2 may remain the better 128K production choice.

### Pi request flow

| State | Prompt | TTFT (s) | E2E (s) | Cache hits |
|---|---:|---:|---:|---:|
| Cold short chat | 595 | 0.811 | 1.746 | 0 |
| Warm multi-turn | 753 | 0.157 | 1.291 | 0 |
| RAG/tool append | 930 | 0.236 | 1.027 | 0 |
| Cold 32K document | 32,640 | 5.802 | 6.694 | 0 |
| Resident 32K follow-up | 32,795 | 0.676 | 1.726 | 30,464 |

The Pi system prefix is shorter than the model's 1,088-token cache page, so short warm requests can correctly have zero cache hits.

### Mixed traffic

MTP4 still crashes when long prefill and speculative decode share one XPU `causal_conv1d` invocation. No-spec completed one p65408/g128 document plus 20 g64 short requests, but short TTFT rose from 0.112 seconds p50 at baseline to 12.855 seconds p50 under the mix.

Mixed aggregate output was 74.46 tok/s: 1,374 output tokens divided by the full 18.452-second interval, including the cold 64K prefill. It is not per stream and not the maximum all-short aggregate decode result.

Full tables, patch order, commands, and metric definitions: [Real-World Pi Workloads and Exact 128K MTP4](docs/REAL-WORLD-PI-BENCHMARKS.md).

## Dense 27B status

vLLM dense FP8 is **blocked**: `KeyError: PlatformEnum.XPU` in
`choose_scaled_mm_linear_kernel` — there is no FP8 linear kernel registered for
XPU. Not slow, *absent*. **llama.cpp is the only working dense path** (Q4_K_M
@230W = 23 t/s). Help wanted on the XPU FP8 kernel — see
[docs/DENSE-FP8-GAP.md](docs/DENSE-FP8-GAP.md).

## Hardware this targets

- **Intel Arc Pro B70** (Battlemage, Xe2): 32 GB GDDR6, 608 GB/s, 256 XMX engines, ~€1,100 / ~$1,200
- **Intel Arc Pro B60** (same arch, 16 GB): should work with smaller models / lower ctx
- Ubuntu 24.04/26.04, Docker, oneAPI 2026.0 drivers
- We test on: B70 + Ryzen 7 5700X3D, 30 GB RAM

## Historical 122K resident-prefix observation (Run 22)

The old local vLLM 0.21 stack loaded a roughly 122K-token session in about 40 seconds, then returned a cached follow-up in about 1.4 seconds. This is historical evidence from the unavailable local-image generation, not the current quick-start proof. The current public-stack resident-prefix result is the exact 32K Pi follow-up in the real-world section above.

## On the localmaxxing leaderboard

**Historical accepted self-reported records (2026-08-07):**

| Submission | Output field | Submitted prefill field | Ctx | Batch | Status | Link |
|---|---:|---:|---:|---:|---|---|
| vLLM MTP4 single-stream | 204.6 t/s | 8,715 t/s, superseded cache-prone harness | 16K | 1 | Accepted self-report | [run](https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwpzf00a4qm01z18izmad) |
| vLLM concurrency max | 1,139.8 gen t/s | 8,715 t/s, superseded cache-prone harness | 16K | 64 | Accepted self-report | [run](https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwqmt00a9qm010iekvi3u) |

Prior batch (2026-08-06):

| Submission | Decode | Prefill | Status |
|------------|-------:|--------:|--------|
| **vLLM MTP (this repo)** | **132.9 t/s** | **7,535 t/s** | ✅ APPROVED |
| llama.cpp MoE Q4_K_XL | 69 t/s | 1,498 t/s | ✅ APPROVED |
| llama.cpp dense 27B Q4_K_M | 23 t/s | 1,007 t/s | ✅ APPROVED |

The accepted records are historical self-reports, not independent reproduction. Their 8,715 t/s prefill field is superseded by the honest cold-prefix measurements above. No August 8 Pi/128K result has been submitted.

`lmx benchmark submit` previously targeted `/api/benchmarks`; the working flat speed-test path was `POST /api/speed-tests`. Preserve the richer internal manifest before converting a result to the platform payload.

Submission payloads: `submissions/`. Schema + instructions:
`docs/localmaxxing-submission-schema.md`.

## Why open

The patches are derivative work — they build on Intel's open `xpu_kernels`,
vLLM, and llama.cpp (all MIT/Apache). Paywalling four patches that are 90% calls
into other people's open code would be ethically gray and reputationally risky.
**Open wins.** If this saved you a week of debugging, the best way to say thanks
is to [buy the card through our affiliate link](https://go.sergiiob.dev/arc-pro)
or [buy us a bench hour on Ko-fi](https://ko-fi.com/sergiiob) (both optional).

## Contributing

- **XPU FP8 dense kernel** — the #1 gap. If you can register an FP8 linear
  kernel for XPU in `choose_scaled_mm_linear_kernel`, dense 27B unblocks.
- **KL/acceptance audit** of the MTP path vs eager — the correctness gate.
- **More models** — DeepSeek-V4, Qwen3-Next, Gemma 4 MoE. Test + PR the configs.
- **B60 testing** — confirm the same patches work on 16 GB.

PRs welcome. See [docs/CAMPAIGN-LOG.md](docs/CAMPAIGN-LOG.md) for the full
19-run investigation narrative.

## License

MIT — see [LICENSE](LICENSE). Patches are derivative of vLLM (Apache 2.0) and
Intel xpu_kernels; benchmark harnesses are original work.

## More

- 📝 **Full showdown post:** [sergiiob.dev](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
- 📊 **All benchmark data:** raw JSONs in `benchmarks/results/` (coming soon)
- 🐦 **Thread:** [@sergiiob](https://x.com/sergiiob)
- 💬 **Issues:** Questions, failures, success stories — all welcome.

⭐ Star if this helped. It helps others find it.
