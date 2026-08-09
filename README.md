# Intel Arc Pro B70 Inference Cookbook

Repeatable vLLM XPU and llama.cpp SYCL recipes for Intel Arc Pro B60/B70 GPUs.

## Current tested stack

| Component | Exact tested value |
|---|---|
| Public image | `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` |
| vLLM observed in image | `0.26.1rc1.dev457+gc810e5ee9.xpu` |
| `vllm-xpu-kernels` observed in image | `0.1.12` |
| Model | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4` |
| Target / draft weights | GPTQ INT4 target / preserved BF16 MTP layer |
| Patches, in order | `patch_mtp_nightly.py`, then `patch_mtp_boundary.py` |
| Context / scheduler / memory | 131,072 / 8,192 / `gpu-memory-utilization=0.85` |

PyPI `vllm-xpu-kernels 0.1.12.2` is newer, but it was not installed or tested in this campaign. The historical `intel/vllm:0.21.0-xpu-int4moe` image was local and was never published.

## Short setup

```bash
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git
cd intel-arc-pro-b70-inference-cookbook
export MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
export MODEL_ID='llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4'
docker pull 'vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
curl -f http://127.0.0.1:8000/health
```

Use [Full setup commands](docs/FULL-SETUP-COMMANDS.md) for the render-device check, model download and verification, package check, patch hashes, endpoint checks, and full matrix.

Agents updating benchmark graphics should use [the B70 benchmark visuals skill](.agentic/skills/b70-benchmark-visuals/SKILL.md). It renders the dashboard and method diagram from canonical `summary.json`.

## MoE: Qwen3.6-35B-A3B — whole analysis

**Scope for every table below:** one Intel Arc Pro B70, C1, median of `n=5` after one same-output same-shape warmup, prefix cache enabled, unique entropy-first cold prefixes, zero cache-hit delta, scheduler 8,192, context 131,072, configured cap 165 W, client monotonic SSE timing, E2 provisional self-reported evidence. Independent reproduction is pending.

Two model checkpoints were verified on this stack: the `llmfan46/...MTP-Preserved` GPTQ-INT4 (all matrix tables) and the byte-exact `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` incl. `mtp.safetensors` (claim-reproduction tests, below).

![B70 phase-separated input and decode dashboard](docs/assets/b70-prefill-decode-dashboard.svg)

### Cold prefill proxy: actual input tokens / TTFT, tok/s

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| No spec | 5,156 | 6,674 | 7,197 | 7,451 | 7,576 | 3,144 |
| MTP1 | 4,840 | 7,377 | 6,999 | 7,189 | 7,264 | 2,679 |
| MTP2 | 4,843 | 7,341 | 7,002 | 7,140 | 7,229 | 2,683 |
| MTP4 | 4,532 | 7,401 | 6,868 | 7,057 | 7,197 | 2,678 |

This rate includes scheduling, uncached prompt processing, and first-token work. It is not isolated engine prefill and is not llama-bench `pp`.

### Decode at p512: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 97.43 | 96.79 | 96.60 | 96.13 |
| MTP1 | 122.21 | 124.57 | 123.82 | 120.58 |
| MTP2 | 162.90 | 153.17 | 148.31 | 141.80 |
| MTP4 | 178.34 | 170.91 | 167.85 | 148.35 |

### Decode at p8192: client post-first tok/s

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 85.92 | 90.34 | 90.91 | 91.26 |
| MTP1 | 108.41 | 118.41 | 118.49 | 117.45 |
| MTP2 | 143.95 | 145.43 | 143.82 | 135.61 |
| MTP4 | 156.28 | 164.36 | 163.89 | 138.03 |

### Historical control: p9445/g128

| Mode | Client post-first median (tok/s) |
|---|---:|
| No spec | 89.68 |
| MTP1 | 116.85 |
| MTP2 | 142.02 |
| MTP4 | 160.42 |

The MTP4 result of 160.42 tok/s reproduces the prior 158.83 tok/s scheduler-control result within 1.0%. It does not make the exact-128K cells equivalent.

### Scheduler and context findings (2026-08-09, focused probes)

`--max-num-batched-tokens` is a cap, not a target: at p4096 both 8,192 and 16,384 prefill in one chunk, yet the larger budget is measurably faster at the same 128K recipe, same seqs 64, same prompts (exact palmfuture checkpoint, 230 W, MTP4):

| Budget | p4096 prefill | g128 decode |
|---|---:|---:|
| 8,192 | 6,525 t/s | 133.1 t/s |
| 16,384 | 7,672 t/s | 149.1 t/s |
| Δ | **+17.6%** | **+12.0%** |

The gain is scheduler/memory-layout, not chunk count. **Do not blanket-adopt 16,384** without testing mixed long-prefill + short-chat loads: one 16K prefill step starves short requests (head-of-line), and activation spikes eat VRAM headroom (128K recipe already loads with ~1 GB free).

Prefill is essentially **flat across context** (p4096 input rate, batch 16,384, seqs 16): 8K ctx 7,727 t/s · 16K 6,622 · 32K 7,740 · 128K 7,672. Context length is not a prefill lever.

### The 12,400 tok/s LocalMaxxing claim — reproduction verdict

We reproduced the claimed config exactly (byte-identical `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` incl. `mtp.safetensors`, hash-verified; context 32,768; p4096/g1; batch 16,384; seqs 16; MTP4; fp8 KV; cache off; 230 W). Measured: **7,740 t/s median**, not 12,400. The claim's cited build hash `568afb3a1` is an upstream macOS-CI commit (#49901), not an XPU kernel change — not a meaningful reproduction target; their entry ran vLLM 0.26.1.dev0 on Windows 11, our stack is 0.26.1rc1.dev457 on Linux.

Verdict: `directional_only`, **not reproduced**. The 1.6× gap is build/OS or their prefill definition (their implied TTFT 0.330 s vs our 0.529 s). Evidence: `results/vllm-moe-12k-exact-20260809T190246Z-13717/` and `results/vllm-moe-12k-lowctx-20260809T193408Z-64922/` (raw SSE, monitor, summary.json in the private B70-DOCS repo).

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| No spec | 57.35 | n/a | 57.14 | n/a |
| MTP1 | 84.88 | 89.22% | 82.74 | 85.32% |
| MTP2 | 101.64 | 85.81% | 94.01 | 76.45% |
| MTP4 | 93.53 | 66.91% | 93.83 | 59.81% |

The original no-spec p130560/g512 cell stopped at EOS in three of five requests. It is excluded and retained in the evidence. The 57.14 tok/s row is the forced exact-output replacement.

`Client post-first` is `(completion tokens - 1) / (request end - first generated token)`. It is request-side timing, not engine-native vLLM decode.

## Choose a mode by workload

1. **Short C1 responses:** MTP4 was fastest in the p512 and p8192 g32/g128 cells.
2. **Exact 128K, g128:** MTP2 was fastest at 101.64 client post-first tok/s.
3. **Exact 128K, g512:** MTP2 and MTP4 were effectively tied at 94.01 and 93.83 tok/s in this campaign.
4. **Resident long sessions:** test cache reuse separately. The earlier matched cache campaign found MTP2 + cache on had the best resident end-to-end median.
5. **Mixed long prefill plus short requests:** use no-spec on this stack. The MTP mixed-token XPU path remains unsupported.

## Dense: Qwen3.6-27B — work in progress

**Status: WORK IN PROGRESS.** Two working paths, neither production-settled.

**llama.cpp SYCL (mature):** Q4_K_M ~21–23 t/s; Q6_K-MTP ~24–30 t/s with MTP-4; Q6_K @ 128K is the VRAM ceiling (0.7 GB free). 165 W is the efficiency sweet spot (0.155 t/s/W). See `docs/POWER-SWEET-SPOTS.md`.

**vLLM XPU dense INT4 (new, 2026-08-09, Run 29):** dense 27B GPTQ-INT4 now runs on the pinned nightly via `XPUwNa16LinearKernel` — 73.2 t/s C1 synthetic decode at p512 with 85% MTP acceptance (both MTP patches apply unchanged to the dense architecture). **fp8 KV is required** for 128K (fp16 KV doesn't fit); realistic Pi short-turn decode drops to 44–56 t/s (MTP acceptance 44–60% on real prompts). 128K is the safe ceiling; 200K loads but lands in the VRAM abort zone; 256K infeasible. Dense prefill is compute-bound and collapses at long context (p4096 ≈ 1,284 t/s, p130944 ≈ 547 t/s).

**Open blocker:** no FP8 *linear* kernel on XPU — `KeyError: PlatformEnum.XPU` in `choose_scaled_mm_linear_kernel`. FP8 checkpoints can't load; INT4 (GPTQ) dense works. See `docs/DENSE-FP8-GAP.md`.

Next: finish the dense four-mode matrix (no-spec/MTP1/MTP2 done at 230 W; MTP4 pending) and settle the scheduler budget under mixed load before calling dense production-ready.

## Reproduce the matrix

The runner does not change host power. `CONFIGURED_CAP_W` records the cap selected by the operator.

```bash
CONFIGURED_CAP_W=165 \
  bash benchmarks/b70-pi-prefill-decode-matrix.sh "$MODEL_DIR"
```

Evidence and format:

- [Machine-readable phase-separated result](results/prefill-decode-matrix-20260809-summary.json)
- [Stable cross-model benchmark format](docs/BENCHMARK-FORMAT.md)
- [Current result plus prior Pi campaigns](docs/REAL-WORLD-PI-BENCHMARKS.md)
- [Image and patch compatibility](docs/IMAGE-AND-PATCH-MATRIX.md)
- [Historical campaign log](docs/CAMPAIGN-LOG.md)

## Correctness limitation

Prompt hashes match across no-spec, MTP1, MTP2, and MTP4. Output parity does not. Depending on the longer-decode cell, only 0 to 4 of 5 repetitions matched exact output text across all four modes. The campaign shows speed and completed exact token shapes, not token, logit, KL, or task-quality parity. Do not use speed as correctness proof.

## Repository map

```text
benchmarks/   launchers, prompt generation, request harnesses, telemetry, compiler
patches/      current nightly patches and retained historical patches
docs/         setup, benchmark contract, methodology, compatibility, history
results/      compact machine-readable public summaries
research/     kernel and quantization investigations
submissions/  historical self-reported platform payloads
```

Code is MIT licensed. Measurement reports and prose are CC BY 4.0. See [LICENSE](LICENSE).
