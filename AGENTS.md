# AGENTS.md — Intel Arc Pro B60/B70 Inference Cookbook

> **For AI agents and ML engineers reproducing or extending this work.**
> This file is the authoritative setup guide. Follow it top-to-bottom on a
> fresh B60/B70 host. Last updated: 2026-08-09.

## 1. What this repo does

Open recipes to run LLMs on **Intel Arc Pro B60/B70 (Battlemage, Xe2)** GPUs:

- **Current vLLM XPU GPTQ-INT4 + MTP4** — pinned public nightly, two current patches, exact 131,072-token completion, and real-world Pi workloads.
- **llama.cpp SYCL** — production single-user engine and dense-model path.
- **Benchmark harnesses** — exact rendered-token, serving-latency, cache, MTP, power, thermal, and mixed-load measurements.
- **Historical evidence** — old vLLM 0.21 scripts remain labeled; they are not the current quick start.

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
# Public image used by the current recipe:
IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
sudo docker run --rm --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')" \
  -v /dev/dri:/dev/dri:ro --entrypoint python "$IMAGE" \
  -c 'import torch; print(torch.xpu.device_count(), torch.xpu.get_device_name(0))'
# Expect: 1 Intel Arc Pro B70
```

`intel/vllm:0.21.0-xpu-int4moe` was a local derived research image and was never published. Do not substitute `intel/vllm:0.21.0-xpu`. Read `docs/IMAGE-AND-PATCH-MATRIX.md`.

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

## 4. Reproducing the current public MTP4 path

Use the public nightly by exact digest, the MTP-preserved model, and the two compatible current patches.

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

### 4.2 Serve with the current two-patch stack

```bash
bash benchmarks/qwen36-35a3/launch-mtp4-128k-nightly.sh \
  /path/to/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 8000
```

The launcher pins:

```text
vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97
```

Observed image contents: vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`. PyPI `vllm-xpu-kernels 0.1.12.2` is newer but untested.

Patch order:

1. `patch_mtp_nightly.py`
2. `patch_mtp_boundary.py`

Do not apply `patch_xpu_int4_moe_v4.py` or `patch_mtp_bf16_draft.py` to this nightly. Those target the historical local vLLM 0.21 image.

The tested scheduler budget is 8,192. A matched p9445/g128 sweep found no improvement at 9,600, 10,240, 12,288, or 16,384.

### 4.3 Reproduce the exact 128K request

Follow `docs/REAL-WORLD-PI-BENCHMARKS.md`:

1. Generate exact rendered p130944 prompts with `benchmarks/b70-generate-exact-prompts.py` inside the pinned tokenizer image.
2. Use one same-shape warmup with separate entropy.
3. Run `benchmarks/b70-realworld-context-harness.py` for p130944/g128.
4. Accept only 130,944 endpoint prompt tokens, 128 completion tokens, finish reason `length`, zero cache-hit delta, and no server error.

The expected single observation is TTFT 48.601 seconds, client post-first 96.87 tok/s, and MTP acceptance 72.31%. This is a single observation, not a median.

### 4.4 Reproduce the matched cache/spec matrix

Use the full public path:

```bash
bash benchmarks/qwen36-35a3/b70-pi-128k-cache-spec-matched.sh /path/to/model
```

It runs no-spec, MTP1, MTP2, and MTP4 with `--enable-prefix-caching` and `--no-enable-prefix-caching`. The negative flag is required because vLLM V1 in the pinned image defaults caching to on. Full setup, model download, patch verification, and all eight manual launch commands: `docs/FULL-SETUP-COMMANDS.md`.

Current machine-readable result: `results/cache-spec-matrix-20260808-summary.json`.


## 5. Current and historical patches

| Stack | Order | File | Purpose |
|---|---:|---|---|
| Current pinned nightly (Qwen Pi) | 1 | `patches/patch_mtp_nightly.py` | Build the preserved BF16 MTP draft outside the target GPTQ quant config |
| Current pinned nightly (Qwen Pi) | 2 | `patches/patch_mtp_boundary.py` | Complete an exact-128K partial final MTP4 group without padding |
| Qwen3.8 champion `f01e24f6` (optional Cn) | 3 | `patches/patch_gdn_mixed_split_v5.py` (also `patch_gdn_split_mixed.py`) | Compact+scatter mixed spec/non-spec `gdn_attention`. Mixed-batch keep; C1 speed-flat vs cookbook MTP4. |
| Qwen3.8 champion `f01e24f6` (optional MTP speed) | 4 | `patches/patch_draft_lmhead_int4.py` then `patch_draft_mtp_int4.py` | Draft-side INT4 RTN (S+M1). Env off unless set. Same-image n=5: p512/g128 112.65 vs 81.20 (+38.7%), p8192/g128 103.63 vs 77.52 (+33.7%), short agentic +32.8%. Isolated C1 128K p130944/g128 62.52 n=5 at 870 MiB free — not a serving cell. Speed only; accept 95.9%→94.4%. |
| Optional / legacy boards | — | `patches/patch_mtp_ptr_wrap.py` | Int64 `data_ptr` wrap only. Do not wrap uint64 `dst_ptrs`. Fail closed. |
| Nemotron DFlash (newer digest) | 1 | `patches/patch_xpu_grouped_topk_native_v2.py` | XPU native grouped-topk + `torch.compiler.disable` ([vllm#52159](https://github.com/vllm-project/vllm/pull/52159)) |
| Nemotron DFlash | 2 | `patches/ssu-b70-b8w4/` | B70 SSU B8/W4 JSON (device-specific, ~1%) |
| Optional kernels rebuild | — | `patches/vllm-xpu-kernels/0001-*.py` + `0002-*.py` | `at::zeros` + Muse tuple ([vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524)) |
| Historical local vLLM 0.21 | 1 | `patches/patch_xpu_int4_moe_v4.py` | GPTQ uint8/int8 native-int4 load correction |
| Historical local vLLM 0.21 | 2–4 | `patches/patch_mtp_bf16_draft.py` | BF16 draft, obsolete kwarg, and GDN metadata-guard corrections |

## 6. Benchmarking discipline (read before measuring)

These rules are non-negotiable for valid data. Violating them produces
inflated/contaminated numbers.

1. **One engine.** Never run two inference engines or unrelated GPU work. A declared concurrency campaign may send Cn requests to one server.
2. **Warm each shape.** Discard a generic JIT warmup and one same-length warmup with separate entropy. Normally retain at least five measured samples; report median, range, and `n`.
3. **Protect cold prefixes.** Put unique random entropy first and require zero token-level prefix-cache hit delta.
4. **Name exact metrics.** Client post-first rate is `(completion_tokens - 1) / (request_end - first_token)`. It is not engine-native. Retain TTFT, TTFC, TPOT/ITL, E2E, endpoint token counts, and server counters.
5. **Separate C1 and Cn.** Aggregate Cn output is all successful generated tokens divided by the full campaign interval. Never sum per-request rates.
6. **Measure real power.** Diff `energy1_input` over the request interval and sample verified named temperatures. Maximum sampled power is an interval average, not instantaneous.

```bash
# Cooldown loop (wait until GPU temp drops below 52°C):
HWMON=$(for h in /sys/class/hwmon/hwmon*; do [ -f "$h/name" ] && [ "$(cat "$h/name")" = "xe" ] && echo "$h"; done | head -n1)
while [ $(($(cat ${HWMON:-/sys/class/hwmon/hwmon0}/temp2_input 2>/dev/null || echo 0)/1000)) -gt 52 ]; do sleep 2; done
```

The public launchers and matrix runner do not change the host power cap. Record the configured cap in the evidence, but leave site-specific power policy to the operator.

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
`docs/localmaxxing-submission-schema.md` and POST via curl.

### 7.3 Current submission status

Files in `submissions/` are historical accepted LocalMaxxing payloads. They preserve the vLLM 0.21/MTP1 campaign and must not be copied as the current nightly recipe.

Any future current-stack note must include:

```text
Public image vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97.
Observed vLLM 0.26.1rc1.dev457+gc810e5ee9.xpu; vllm-xpu-kernels 0.1.12.
Patches: patch_mtp_nightly.py, then patch_mtp_boundary.py.
MTP4, single-stream unless declared otherwise.
```

Keep C1 decode, cold prefill, long-context latency, and aggregate Cn observations in separate internal evidence records even if the platform uses one flat payload.

## Long-context scaling (exact rendered tokens)

The 2026-08-08 Pi campaign used g128, unique cold prefixes, zero cache-hit delta, and C1:

| Spec | Prompt | Output | Total | TTFT (s) | Client post-first (tok/s) | Result |
|---|---:|---:|---:|---:|---:|---|
| MTP4 | 16,256 | 128 | 16,384 | 2.403 | 161.23 | Completed |
| MTP4 | 32,640 | 128 | 32,768 | 5.785 | 139.99 | Completed |
| MTP4 | 65,408 | 128 | 65,536 | 15.093 | 111.17 | Completed |
| MTP4 | 98,176 | 128 | 98,304 | 28.078 | 117.36 | Completed |
| MTP4 | 122,880 | 128 | 123,008 | 44.057 | 95.13 | Completed |
| MTP4 + boundary patch | 130,944 | 128 | **131,072** | 48.601 | 96.87 | **Completed** |
| MTP2, unpatched | 130,944 | 128 | **131,072** | 48.559 | 103.63 | Completed |

Unpatched MTP4 failed after 124 outputs because four remaining sequence slots truncated a required five-token speculative group. The boundary patch fixes that metadata path. MTP2 was 6.98% faster in the single matched exact-128K observations; MTP4 correctness does not make it the automatic production choice.

Launch exact MTP4 with `benchmarks/qwen36-35a3/launch-mtp4-128k-nightly.sh`. Reproduce the request through `docs/REAL-WORLD-PI-BENCHMARKS.md`.

## 8. Known gaps & open work

- **Dense FP8 on vLLM** — blocked, no XPU kernel. See `docs/qwen36-27/DENSE-FP8-GAP.md`.
- **KL-divergence / acceptance-rate audit** — pending before a broad production-correctness claim.
- **B60 testing** — current image and patches are not yet verified on the 16 GB card.
- **MTP mixed load** — long prefill plus speculative decode still fails in the XPU GDN `causal_conv1d` mixed-token path. Use no-spec for concurrent serving.

### Quantization format landscape on B70

GPTQ-Int4 is the optimal format for MoE on vLLM XPU — INT4 is what Intel's XMX
engines are built to accelerate. This is not a compromise vs NVIDIA's NVFP4;
it's the Intel-native equivalent. Full analysis in
`research/quantization-format-strategy.md`.

| Format | vLLM XPU | llama.cpp SYCL | Notes |
|--------|----------|----------------|-------|
| **INT4 (GPTQ)** | **133 t/s** | 69 t/s (Q4_K_XL) | XMX native fast path |
| MXFP4 | 10.4 t/s | N/A | Loads, correct output, bottlenecked by GDN Triton kernels (not the quant) |
| FP8 block | 0.75 t/s | N/A | Dequant fallback only, no native XPU kernel |
| GGUF Q5_K_M | N/A | 70 t/s | Best quality path, 2× lower KLD than Q4 |
| FP8 (native) | Blocked | N/A | Needs upstream `xpu_kernels` contribution |

## 9. Contributing

PRs welcome. Especially:

- **XPU FP8 dense kernel** — the #1 gap (see `docs/qwen36-27/DENSE-FP8-GAP.md`).
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
HWMON=$(for h in /sys/class/hwmon/hwmon*; do [ -f "$h/name" ] && [ "$(cat "$h/name")" = "xe" ] && echo "$h"; done | head -n1)
cat ${HWMON:-/sys/class/hwmon/hwmon0}/power1_cap                      # current cap (µW)
echo $(($(cat ${HWMON:-/sys/class/hwmon/hwmon0}/temp2_input 2>/dev/null || echo 0)/1000))°C # current temp
sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm   # VRAM (look for visible_avail)
```

**Never load a single GGUF >30 GB with `-ngl 99`** — it overflows 32 GB VRAM
and causes a hard system crash (verified; see campaign log). Use partial offload
(`-ngl <N`) for models that large.


## 12. Repeat the phase-separated matrix on another model

This is the operational repeat-and-publish checklist. The stable table and
evidence contract is [`docs/BENCHMARK-FORMAT.md`](docs/BENCHMARK-FORMAT.md).
Use [`.agentic/skills/b70-benchmark-visuals/SKILL.md`](.agentic/skills/b70-benchmark-visuals/SKILL.md) for data-driven dashboard and method SVG regeneration; its bundled renderer matches `benchmarks/render-prefill-decode-svg.py`.

### 12.1 Intake and calibration

1. Record a pullable image plus immutable digest, observed in-container vLLM and
   `vllm-xpu-kernels` versions, model repository/revision/quantization and hashes,
   tokenizer/chat template, ordered patches and hashes, context, scheduler,
   memory utilization, cache mode, configured cap, and UTC run ID.
2. Verify MTP tensors from the model shards. A config field alone is not proof.
   Run no-spec and each model-supported MTP depth; mark unsupported modes.
3. Calibrate rendered messages inside the tested image with
   `len(encoded["input_ids"])`. Save message and prompt-set hashes.
4. Use `benchmarks/benchmark-system-prompt.txt` for standard p512 through p8192
   cells. Use `benchmarks/pi-system-prompt.txt` for p9445 and full-context cells.
5. Generate six exact entropy-first prompts per coordinate: one warmup and five
   measured. Preserve the exact target length.

Any change to image, package, model revision, quantization, tokenizer, chat
template, prompt, patch, scheduler, context, or memory setting starts a new
result generation.

### 12.2 Exact matrix and request policy

Run:

- input p512/p2048/p4096/p6144/p8192/p131071, each g1;
- p512 and p8192 decode at g32/g128/g256/g512;
- p9445/g128 historical control;
- full p130944/g128 and p130560/g512;
- no-spec plus every supported MTP depth.

For every coordinate and mode, discard one full-output same-shape warmup, then
measure five C1 requests. Decode requests MUST use `ignore_eos=true`. Retain any
early-EOS original as an exclusion and rerun the full affected cell with forced
exact output. Never mix partial and exact samples.

This matrix is C1 cold-prefix work. Cache/resident-prefix and concurrency
campaigns are separate publications.

### 12.3 Cache, VRAM, and timing gates

Keep prefix caching enabled for the production-state matrix, but require unique
entropy first and zero `vllm:prefix_cache_hits` delta for each cold cell. Diff
`vllm:prefix_cache_queries` too and reconcile it with actual endpoint prompt
tokens. Retain proposed/accepted MTP counters; no-spec must propose zero tokens.

Before load, require no inference process/container and approximately 31,000 MiB
or more empty-card `visible_avail`. After load: below 500 MiB abort; 500–1,023
MiB is isolated C1 capacity only; 1,024–2,047 MiB is staged C1 research; at least
2,048 MiB is preferred C1 performance; at least 3,072 MiB is the
mixed/concurrency target. Logged KV capacity must exceed prompt plus output.

C1 means one active measured request and no queue:

- cold input rate = actual endpoint prompt tokens / client monotonic TTFT;
- client post-first rate =
  `(completion_tokens - 1) / (request_end - first_generated)`;
- TTFT, TTFC, TPOT/ITL, and E2E are client monotonic SSE measurements.

Cold input rate includes scheduling and first-token work. It is not llama-bench
`pp` or isolated engine prefill. Client post-first rate is not engine-native vLLM
decode and is not llama-bench `tg`.

### 12.4 Evidence, compiler, and publication

Store one root with `manifest.txt`, `package-versions.txt`, `prompts/`, one
directory per mode, `summary.json`, and `tables.md`. Each mode retains server
logs, metrics snapshots, monitor and VRAM data, harness logs, raw SSE/request
records, per-cell results, failures, exclusions, and cleanup state.

Compile before copying any number:

```bash
python3 benchmarks/b70-compile-prefill-decode-matrix.py "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json"
python3 benchmarks/render-prefill-decode-tables.py "$RUN_ROOT/summary.json" \
  > "$RUN_ROOT/tables.md"
```

The compiler must fail closed on incomplete modes, wrong tokens, prompt mismatch,
cache contamination, server errors, and missing replacement evidence.
`summary.json` is the numeric authority.

Before publication:

1. Show separate cold-input, p512 decode, p8192 decode, p9445 control,
   full-context, and exclusion views.
2. Put units, C1, `n=5`, cache state, timing source, configured cap, image and
   observed software versions beside the tables.
3. State prompt parity and output/correctness limits. Speed is not proof of token,
   logit/KL, or task-quality parity.
4. Update README, `FULL-SETUP-COMMANDS.md`, `BENCHMARK-FORMAT.md`,
   `REAL-WORLD-PI-BENCHMARKS.md`, compact public summary JSON, and every public
   page that repeats a changed number. Keep older results labeled and linked.