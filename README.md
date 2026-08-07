# Intel Arc Pro B60 / B70 Inference Cookbook 🚀

> Open recipes, engine patches, and benchmark harnesses for running LLMs on
> Intel Arc Pro B-series (Battlemage / Xe2) GPUs — **MoE 35B at 204.6 t/s decode
> (MTP4) and ~8.4K t/s prefill, single-stream, one card.**

[![Benchmark](https://img.shields.io/badge/MoE%20decode-204.6%20t%2Fs-10b981)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![Prefill](https://img.shields.io/badge/MoE%20prefill-8.4K%20t%2Fs-0ea5e9)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes%20please-22c55e)](#why-open)

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
  patch_xpu_int4_moe_v4.py   ← native int4 MoE: implement_zp→int8, MoeWNA16→XpuFusedMoe
  patch_mtp_bf16_draft.py    ← MTP unlock: BF16 draft, kwarg strip, GDN spec assert
benchmarks/
  launch-mtp-bf16draft.sh    ← one-shot server launch (vLLM MTP, 16K ctx)
  b70-vllm-reddit-bench-v2.py ← Reddit-mirror: tg32/pp2048 single-stream
  b70-moe-sweep.py           ← vLLM prefill×gen grid harness
  b70-moe-sweep-llamacpp.py  ← llama.cpp prefill×gen grid harness
docs/
  POWER-SWEET-SPOTS.md       ← MoE=150W, Dense=180W — why (with data)
  CAMPAIGN-LOG.md            ← 19-run narrative (A1→A16)
  QUANTIZATION-QUALITY.md    ← GPTQ-Int4 vs GGUF K-quant KL divergence analysis
  DENSE-FP8-GAP.md           ← Why vLLM dense is blocked (no XPU FP8 kernel)
```

## Quick start: vLLM MTP on Qwen3.6-35B-A3B (the 126 t/s path)

**You need:** an Arc Pro B60/B70, Docker, the B70 drivers + oneAPI runtime.

```bash
# 1. Pull the model (MTP-preserved GPTQ — has the real mtp.* tensors)
#    ~22 GB → https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4

# 2. Write the speculative config
echo '{"method":"mtp","num_speculative_tokens":1}' > /tmp/spec.json

# 3. Serve with both patches applied (in-container, no rebuild)
sudo docker run -d --name b70vllm -p 8000:8000 \
  --device /dev/dri --group-add $(stat -c "%g" /dev/dri/render* | head -n1) \
  -v /dev/dri:/dev/dri:ro \
  -v /path/to/model:/model:ro \
  -v $PWD/patches/patch_xpu_int4_moe_v4.py:/patch_v4.py:ro \
  -v $PWD/patches/patch_mtp_bf16_draft.py:/patch_mtp.py:ro \
  -v /tmp/spec.json:/spec.json:ro \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  --entrypoint bash intel/vllm:0.21.0-xpu-int4moe \
  -lc 'python /patch_v4.py && python /patch_mtp.py && SPEC=$(cat /spec.json) && \
       exec vllm serve /model --quantization gptq --dtype float16 \
       --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 1 \
       --language-model-only --speculative-config "$SPEC" \
       --cudagraph-capture-sizes 1 2 4 8 16 32'

# 4. Wait for "Application startup complete", then:
python3 benchmarks/b70-vllm-reddit-bench-v2.py \
  http://localhost:8000/v1/chat/completions \
  Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 /tmp/result.json "b70-mtp"
```

Look for `[B70] GDN XPU: spec decode active` in the logs — that's MTP running.

## The four patches (what they fix)

| # | Patch | Fixes | Root cause |
|---|-------|-------|------------|
| 1 | `patch_xpu_int4_moe_v4.py` | Native int4 path crash | C++ `is_B_int4 = (B_dtype == at::kChar)`; GPTQ packs uint8 → kernel treated weights as BF16. Store int8. |
| 2 | `patch_mtp_bf16_draft.py` (BF16 draft) | `KeyError: w2_weight` on MTP load | Draft inherits target's GPTQ quant_config; checkpoint's mtp experts are BF16 fused. Strip quant_config for any `mtp` prefix. |
| 3 | `patch_mtp_bf16_draft.py` (kwarg strip) | `XpuFusedMoe.__init__ unexpected kwarg is_fp8` | vLLM's xpu_moe.py passes is_fp8/is_mxfp4; the kernels auto-detect dtype from weights. Drop the kwargs. |
| 4 | `patch_mtp_bf16_draft.py` (GDN assert) | `AssertionError: spec_sequence_masks is None` | The XPU GDN SYCL kernel already takes explicit spec tensors; the boolean mask is metadata-only and never reaches the kernel. The assert was a guardrail, not a capability limit. |

Patch 4 is the headline: it refutes the prior "XPU GDN incompatible with
speculative decoding" verdict. It was an overcautious assert, not a real kernel
limit. Removing it unlocks MTP (and would unblock other spec methods too).

**Correctness verified:** greedy `temp=0` replays produce byte-identical output
(a corrupting spec path would diverge); factual probes (17×23=391, capital of
Australia=Canberra) correct. A full KL-divergence / acceptance-rate audit vs the
eager path is the remaining gate before production use — contributions welcome.

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

## Multi-turn 128K (Run 22)

With `--enable-prefix-caching`, a 128K session is interactive: load once (40s),
then follow-up turns respond in **1.4s at full 122K context** (28× faster than
cold, decode ~80 t/s). Harness: `benchmarks/b70-multiturn-128k-test.py`.

## On the localmaxxing leaderboard

**Newest (2026-08-07) — MTP4 + concurrency max, both APPROVED:**

| Submission | Decode | Prefill | Ctx | Batch | Status | Link |
|------------|-------:|--------:|----:|------:|--------|------|
| **vLLM MTP4 single-stream** | **204.6 t/s** | 8,715 t/s | 16K | 1 | ✅ APPROVED | [run](https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwpzf00a4qm01z18izmad) |
| **vLLM concurrency max** | **1,139.8 t/s** (gen) | 8,715 t/s | 16K | 64 | ✅ APPROVED | [run](https://www.localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmsiwwqmt00a9qm010iekvi3u) |

Prior batch (2026-08-06):

| Submission | Decode | Prefill | Status |
|------------|-------:|--------:|--------|
| **vLLM MTP (this repo)** | **132.9 t/s** | **7,535 t/s** | ✅ APPROVED |
| llama.cpp MoE Q4_K_XL | 69 t/s | 1,498 t/s | ✅ APPROVED |
| llama.cpp dense 27B Q4_K_M | 23 t/s | 1,007 t/s | ✅ APPROVED |

The MTP4 result (204.6 t/s single-stream) and the concurrency-max (1,139.8
gen t/s at batch 64) are the standout numbers — with full patch disclosure
in the notes. **Submission gotcha:** `lmx benchmark submit` hits the wrong
endpoint (`/api/benchmarks`, evals-only) and silently fails; the working path
is `POST /api/speed-tests`. See the `localmaxxing-submit` skill.

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
