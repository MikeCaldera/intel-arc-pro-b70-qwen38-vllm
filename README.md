# Intel Arc Pro B60 / B70 Inference Cookbook 🚀

> Open recipes, engine patches, and benchmark harnesses for running LLMs on
> Intel Arc Pro B-series (Battlemage / Xe2) GPUs — **MoE 35B at 133 t/s decode
> and 8.7K t/s prefill, single-stream, one card.**

[![Benchmark](https://img.shields.io/badge/MoE%20decode-133%20t%2Fs-10b981)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![Prefill](https://img.shields.io/badge/MoE%20prefill-8.7K%20t%2Fs-0ea5e9)](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes%20please-22c55e)](#why-open)

## The headline

After a 19-run benchmark campaign, here's what one Intel Arc Pro B70 (32 GB,
Battlemage, ~$600) actually does on Qwen3.6-35B-A3B (MoE):

| Metric | vLLM XPU + MTP | llama.cpp SYCL | vLLM advantage |
|--------|---------------:|---------------:|---------------:|
| **Decode (short/g32)** | **133 t/s** | 74 t/s | **1.8×** |
| **Prefill (p8k)** | **8,718 t/s** | 1,662 t/s | **5.2×** |
| **Decode (p8k/g512)** | **114 t/s** | 58 t/s | **1.96×** |
| Power | 150W | 150W | (same) |
| Temp | 58°C | 58°C | (same) |

And vs our previous best (Run 16, Triton path: 58 t/s decode / 5.3K prefill):
**2.17× decode jump, +40% prefill** — from four targeted in-container patches plus a batched-tokens cap fix.

Full methodology + all grids: **[vLLM vs llama.cpp — The Full MoE + Dense Showdown](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)**.

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
| **MoE 35B** | **150W** | Self-limits to ~140W draw; 230W gives -8% at +80W heat. Cooler = same speed. |
| **Dense 27B** | **180W** sustained / 230W burst | Scales +18–30% (150→230W), but thermal cost (71→79°C). |

Set it once:
```bash
# 150W for MoE (in microwatts)
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap
```

## Dense 27B status

vLLM dense FP8 is **blocked**: `KeyError: PlatformEnum.XPU` in
`choose_scaled_mm_linear_kernel` — there is no FP8 linear kernel registered for
XPU. Not slow, *absent*. **llama.cpp is the only working dense path** (Q4_K_M
@230W = 23 t/s). Help wanted on the XPU FP8 kernel — see
[docs/DENSE-FP8-GAP.md](docs/DENSE-FP8-GAP.md).

## Hardware this targets

- **Intel Arc Pro B70** (Battlemage, Xe2): 32 GB GDDR6, 608 GB/s, 256 XMX engines, ~$600
- **Intel Arc Pro B60** (same arch, 16 GB): should work with smaller models / lower ctx
- Ubuntu 24.04/26.04, Docker, oneAPI 2026.0 drivers
- We test on: B70 + Ryzen 7 5700X3D, 30 GB RAM

## Multi-turn 128K (Run 22)

With `--enable-prefix-caching`, a 128K session is interactive: load once (40s),
then follow-up turns respond in **1.4s at full 122K context** (28× faster than
cold, decode ~80 t/s). Harness: `benchmarks/b70-multiturn-128k-test.py`.

## On the localmaxxing leaderboard

Three submissions **APPROVED and live** (2026-08-06):

| Submission | Decode | Prefill | Status |
|------------|-------:|--------:|--------|
| **vLLM MTP (this repo)** | **132.9 t/s** | **7,535 t/s** | ✅ APPROVED |
| llama.cpp MoE Q4_K_XL | 69 t/s | 1,498 t/s | ✅ APPROVED |
| llama.cpp dense 27B Q4_K_M | 23 t/s | 1,007 t/s | ✅ APPROVED |

The vLLM MTP result is the standout — **highest decode t/s for this model on
the B70 on the leaderboard**, with full patch disclosure in the notes
(reproducibility matters; the 4 patches are in this repo).

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
