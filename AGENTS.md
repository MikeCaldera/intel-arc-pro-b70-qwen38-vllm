# AGENTS.md — Intel Arc Pro B60/B70 Inference Cookbook

> **For AI agents and ML engineers reproducing or extending this work.**
> This file is the authoritative setup guide. Follow it top-to-bottom on a
> fresh B60/B70 host. Last updated: 2026-08-06.

## 1. What this repo does

Open recipes to run LLMs on **Intel Arc Pro B60/B70 (Battlemage, Xe2)** GPUs:

- **vLLM XPU native int4 + MTP** — 126 t/s decode / 7.5K t/s prefill on Qwen3.6-35B-A3B MoE (single-stream, one B70). Requires four in-container patches.
- **llama.cpp SYCL** — the production single-user engine and the only working dense path.
- **Benchmark harnesses** — apples-to-apples prefill × generation grids for both engines.
- **Power-sweep tooling** — find the MoE (150W) vs Dense (180W) sweet spots.

Headline result and full methodology:
[sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/](https://sergiiob.dev/posts/intel-arc-b70-vllm-vs-llamacpp-moe-dense-showdown/)

## 2. Target hardware

- **Intel Arc Pro B70** (32 GB GDDR6, 608 GB/s, Xe2, ~$600) — primary test target.
- **Intel Arc Pro B60** (16 GB, same arch) — should work with smaller models / lower context. **Tested contributions welcome.**
- Ubuntu 24.04 / 26.04, x86_64.
- Reference rig: B70 + AMD Ryzen 7 5700X3D, 30 GB RAM, NVMe.

## 3. Host prerequisites (install once)

### 3.1 Drivers + oneAPI

The B70 needs the Intel GPU kernel driver + oneAPI runtime for both vLLM
(inside Docker) and llama.cpp (native SYCL).

```bash
# Intel GPU drivers (follow the official Intel guide for your distro):
#   https://dgpu-docs.intel.com/
# Verify the card is visible:
sudo apt install -y intel-level-zero-tools
sudo lszk  # or: lspci | grep -i vga

# oneAPI 2026.0 (for native llama.cpp SYCL builds):
#   Download from intel.com/content/www/us/en/developer/tools/oneapi/...
# Source it before any native SYCL work:
source /opt/intel/oneapi/setvars.sh --force
```

### 3.2 Docker (for vLLM)

```bash
# Standard Docker install, then verify GPU passthrough:
sudo docker run --rm --device /dev/dri --group-add $(stat -c "%g" /dev/dri/render* | head -n1) \
  -v /dev/dri:/dev/dri:ro --entrypoint bash intel/vllm:0.21.0-xpu-int4moe \
  -lc 'python -c "import torch; print(torch.xpu.device_count(), torch.xpu.get_device_name(0))"'
# Expect: 1 Intel Arc Pro B70
```

If that prints `0`, your driver/render-node permissions are wrong — fix before
proceeding. The user running Docker must be in the `render` group, or use
`--group-add $(stat -c "%g" /dev/dri/render*)`.

### 3.3 llama.cpp SYCL (native, for the dense + production path)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build-sycl && cd build-sycl
source /opt/intel/oneapi/setvars.sh --force
cmake .. -DGGML_SYCL=ON -DGGML_XMX=ON -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_TARGET_INTEL=ON \
  -DDETECT_ONEAPI_LICENSE=ON -DGGML_BACKEND_DL=ON -DLLAMA_CURL=ON
cmake --build . --config Release -j -- llama-server llama-bench
# Binary: ./bin/llama-server
```

The reference production build is `b10255+` (commit `071327508`). See
`docs/CAMPAIGN-LOG.md` for the exact flags that produced the headline numbers.

### 3.4 lmx (localmaxxing CLI) — optional, for leaderboard submission

```bash
curl -fsSLO https://github.com/LottoLottoLotto/localmaxxing-cli/releases/latest/download/lmx-linux-amd64.tar.gz
tar -xzf lmx-linux-amd64.tar.gz && sudo mv lmx /usr/local/bin/
lmx --version  # v0.1.30+ recommended
```

## 4. Reproducing the headline result (vLLM MTP, 126 t/s)

The full sequence — model download → patch → serve → bench.

### 4.1 Get the model

You need the **MTP-preserved** GPTQ checkpoint. Plain `Qwen3.6-35B-A3B-GPTQ-Int4`
has `mtp_num_hidden_layers: 1` in config but **zero MTP tensors** in the shards.
The preserved variant has the real `mtp.*` weights:

```bash
# Fastest: resolve the HF CDN URL, then aria2c -x16 (~86 MB/s)
CDN=$(curl -sI -L -o /dev/null -w '%{url_effective}' \
  'https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4/resolve/main/config.json')
# Download all 6 shards + sidecars from:
#   https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
# (~22 GB total)
```

### 4.2 Serve with both patches applied

```bash
# From the repo root:
bash benchmarks/launch-mtp-bf16draft.sh /path/to/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 8000
```

**Critical flag — `--max-num-batched-tokens 8192`:** MTP caps
`max_num_scheduled_tokens` to 2048 by default, which **chunks prefill** on any
prompt >2048 tokens and costs 21-28% prefill throughput. Setting it to 8192
clears the cap (verified: p4k prefill 6,626 → 8,484 t/s; p8k → 8,718 t/s).
Without this flag you lose the long-prompt prefill advantage.

The launch script:
1. Sets speculative config `{"method":"mtp","num_speculative_tokens":1}`
2. Runs `patch_xpu_int4_moe_v4.py` + `patch_mtp_bf16_draft.py` in-container
3. Starts `vllm serve` with the right flags
4. Polls until `/v1/models` responds

Watch for `[B70] GDN XPU: spec decode active` in the logs — that's MTP running.
Full graph capture + startup takes ~3-4 min.

### 4.3 Benchmark

```bash
python3 benchmarks/b70-vllm-reddit-bench-v2.py \
  http://localhost:8000/v1/chat/completions \
  Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 /tmp/result.json "b70-mtp"
# Expect: tg32 ~123-127 t/s, pp2048 ~7.2K t/s
```

For the full prefill × gen grid:
```bash
python3 benchmarks/b70-moe-sweep.py \
  http://localhost:8000/v1/chat/completions \
  Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 /tmp/grid.json vllm-mtp 2
```

## 5. The four patches — what each fixes

| # | Patch | File | Fixes |
|---|-------|------|-------|
| 1 | Native int4 dtype | `patches/patch_xpu_int4_moe_v4.py` | C++ `is_B_int4 = (B_dtype == at::kChar)`; GPTQ packs uint8 → kernel saw BF16. Store int8. |
| 2 | BF16 MTP draft | `patches/patch_mtp_bf16_draft.py` | Draft inherits GPTQ quant_config; checkpoint mtp experts are BF16 fused. Strip quant_config for `mtp` prefix. |
| 3 | XpuFusedMoe kwarg strip | `patches/patch_mtp_bf16_draft.py` | vLLM passes `is_fp8`/`is_mxfp4`; kernels auto-detect dtype. Drop the kwargs. |
| 4 | GDN spec assert | `patches/patch_mtp_bf16_draft.py` | `spec_sequence_masks` is metadata-only, never passed to SYCL. Assert was a guardrail, not a capability limit. |

Patches 2, 3, 4 are all in `patch_mtp_bf16_draft.py` (applied together). They
are **idempotent** — safe to re-run; they detect "already patched" and skip.

**Correctness:** patches 1-3 are pure load-path fixes (no math change). Patch 4
removes a guardrail; correctness verified by byte-identical greedy replays +
factual probes (17×23=391, capital of Australia=Canberra). A full KL-divergence
audit vs eager is the remaining gate before production — see
`docs/CAMPAIGN-LOG.md`.

## 6. Benchmarking discipline (read before measuring)

These rules are non-negotiable for valid data. Violating them produces
inflated/contaminated numbers.

1. **Sequential only.** Never run two inference processes concurrently — GPU
   contention corrupts metrics and can OOM.
2. **Warmup is mandatory.** First request after start includes SYCL JIT (30-60s).
   Discard rep 1; report best of reps 2+.
3. **Cooldown between runs.** Wait for GPU temp ≤52°C before the next run.
   Residual heat inflates temperatures.
4. **Measure engine rate, not wall-clock.** vLLM: stream with
   `include_usage`, decode = `completion_tokens / (total - ttft)`. llama.cpp:
   `timings.predicted_per_second` (the engine rate, not `tokens/total_time`).
5. **Power: set once, verify.** Set the cap, wait 3s, read it back. Don't
   change power while a server is running.

```bash
# Cooldown loop (hwmon temp2_input is the B70 sensor, in millidegrees C):
while [ $(($(cat /sys/class/hwmon/hwmon4/temp2_input)/1000)) -gt 52 ]; do sleep 2; done

# Power cap (microwatts): 150W=150000000, 180W=180000000, 230W=230000000
echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap
```

## 7. Submitting to localmaxxing.com

The `lmx` CLI validates and submits. There are two paths:

### 7.1 Automated (recommended) — lmx drives the running server

With the vLLM MTP server running on `localhost:8000`:

```bash
# Dry-run first to validate:
lmx benchmark run vllm \
  --mode remote --base-url http://localhost:8000 \
  --hf-id "Qwen/Qwen3.6-35B-A3B" \
  --quantization GPTQ-Int4 \
  --hardware b70-hardware.json \
  --max-tokens 256 --warmup 1 --iterations 3 \
  --out runs/qwen35-mtp-gptq/run.json --dry-run

# Real run (generates runs/.../run.json):
lmx benchmark run vllm \
  --mode remote --base-url http://localhost:8000 \
  --hf-id "Qwen/Qwen3.6-35B-A3B" \
  --quantization GPTQ-Int4 \
  --hardware b70-hardware.json \
  --max-tokens 256 --warmup 1 --iterations 3 \
  --out runs/qwen35-mtp-gptq/run.json

# Submit (requires API key from localmaxxing.com):
lmx auth --key bhk_...
lmx benchmark validate-local runs/qwen35-mtp-gptq/run.json
lmx benchmark runs submit runs/qwen35-mtp-gptq/run.json
```

### 7.2 Manual JSON (fallback, Cloudflare-safe via curl)

If `lmx` is unavailable, build the JSON per the schema in
`docs/localmaxxing-submission-schema.md` and POST via curl. The prior Jul 16
upload script (`upload_b70_localmaxxing_jul16.py`, in the private B70-DOCS) is
the reference implementation.

### 7.3 What to submit (the headline candidates)

| Engine | Model | Quant | tokSOut | tokSPrefill | Notes |
|--------|-------|-------|--------:|------------:|-------|
| **vLLM** | Qwen3.6-35B-A3B | GPTQ-Int4 + MTP | **~123-126** | **~7,200** | The breakthrough — first MTP-on-XPU-GDN. |
| llama.cpp | Qwen3.6-35B-A3B | Q4_K_XL | ~69 | ~1,500 | MoE baseline. |
| llama.cpp | Qwen3.6-35B-A3B | Q5_K_M | ~67 | ~1,690 | Prior Jul 16 submission. |
| llama.cpp | ThinkingCap-Qwen3.6-27B | Q4_K_M | ~23 | ~1,007 | Dense. |
| llama.cpp | ThinkingCap-Qwen3.6-27B | Q5_K_M-MTP | ~25 | ~613 | Dense + MTP. |

**Important honesty note for the vLLM submission:** the MTP path uses a patched
engine (GDN assert removed). The submission notes MUST disclose this —
localmaxxing values reproducibility, and the patches are open in this repo.
Suggested notes: *"vLLM 0.21 XPU + 4 in-container patches (see
github.com/SergiioB/intel-arc-pro-b70-inference-cookbook). MTP speculative,
1 layer, num_spec=1. Single-stream. KL audit vs eager pending."*

## 8. Known gaps & open work

- **Dense FP8 on vLLM** — blocked, no XPU kernel. See `docs/DENSE-FP8-GAP.md`.
- **KL-divergence / acceptance-rate audit** of the MTP path vs eager — the
  correctness gate before production use.
- **B60 testing** — same patches should work; needs confirmation.
- **num_spec=2** — clamps to 1 (checkpoint has 1 MTP layer). A 2-layer MTP
  checkpoint could push closer to 145 t/s.

## 9. Contributing

PRs welcome. Especially:

- **XPU FP8 dense kernel** — the #1 gap (see `docs/DENSE-FP8-GAP.md`).
- **KL/acceptance audit** of the MTP path.
- **More models** — DeepSeek-V4, Qwen3-Next, Gemma 4 MoE. Test + PR configs.
- **B60 confirmation** runs.

See `docs/CAMPAIGN-LOG.md` for the full 19-run investigation narrative.

## 10. Quick reference — environment variables

```bash
# Required for native SYCL (llama.cpp):
source /opt/intel/oneapi/setvars.sh --force
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZE_AFFINITY_MASK=0

# Inside Docker (vLLM) — set by the launch script:
#   VLLM_TARGET_DEVICE=xpu
#   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
#   ZE_AFFINITY_MASK=0
#   VLLM_XPU_ENABLE_XPU_GRAPH=1
#   PYTORCH_ALLOC_CONF=expandable_segments:True
```

## 11. Quick reference — power & thermal

| Setting | Value | Use when |
|---------|-------|----------|
| 150W | `150000000` | MoE (sweet spot — self-limits anyway) |
| 165W | `165000000` | Dense efficiency sweet spot (0.155 t/s/W) |
| 180W | `180000000` | Dense sustained |
| 230W | `230000000` | Dense burst only (79°C) |

```bash
cat /sys/class/hwmon/hwmon4/power1_cap                      # current cap (µW)
echo $(($(cat /sys/class/hwmon/hwmon4/temp2_input)/1000))°C # current temp
sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm   # VRAM (look for visible_avail)
```

**Never load a single GGUF >30 GB with `-ngl 99`** — it overflows 32 GB VRAM
and causes a hard system crash (verified; see campaign log). Use partial offload
(`-ngl <N`) for models that large.
