# Qwen3.8-27B FP8 — Dual B70 TP2, W8A16 + Xe2 small-M kernel, MTP-8

Status: official-lab self-report (E2 — raw evidence retained, not independently
reproduced). Measured on a dual `Intel Arc Pro B70` (32 GB each) machine running
vLLM XPU in TP2, per-worker Level Zero affinity + SYS_PTRACE, power caps set to
230 W with per-card measured draw reported below.

## Result (C1, greedy diagnostic, 5 fresh samples)

| Cell | Median per-request tok/s | Max | Min | Run-level output tok/s | Mean TPOT | Cold-input prefill (in/TTFT) | Measured draw GPU0/GPU1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p512 / g128 | **53.42** | 63.70 | 36.28 | 49.93 | 16.78 ms | 1206.7 tok/s | 97.6 / 103.6 W |
| p1024 / g128 | **60.13** | 71.06 | 35.19 | 53.74 | 14.15 ms | 1736.2 tok/s | 109.5 / 118.9 W |
| p2048 / g128 (hybrid k=6) | **33.52** | 38.00 | 19.15 | 28.08 | 23.49 ms | 1315.1 tok/s | 135.1 / 147.2 W |

Raw per-request values: `results/qwen38-27-fp8-tp2-k8-n5/bench/`.

Sampling label: these are **greedy (temperature 0)** runs on random-token
prompts (`vllm bench serve --dataset-name random`). Speculative acceptance is
content- and numerics-dependent; the same stack under the model-card
recommended sampling (temp 0.7 / top_p 0.8 / top_k 20) is measurably slower
(spec-decode acceptance drops from ~5.3–5.8 accepted tokens/step to ~2; the
realistic-sampling cells are retained in the private campaign root). Decode
values here are the diagnostic maximum of this build, not a guaranteed serving
rate.

Prefill label: cold-input rate = actual endpoint prompt tokens ÷ client TTFT
(uncached, prefix-cache disabled; not engine-isolated prefill). The TTFT
decomposition is fixed overhead ~259 ms plus a marginal prefill rate of
~3094 tok/s across the p512→p1024 cells.

## Stack (recipe of record)

- Image digest (immutable, pullable): `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM 0.27.2rc1.dev77+gac7509e2b, vLLM-XPU kernels 0.1.12.3, torch 2.11).
- Target: `Qwen/Qwen3.8-27B-FP8`, TP2/PP1, container `--workdir /` so
  `import vllm` resolves the patched install.
- Serve flags: `--quantization fp8 --dtype bfloat16 --max-model-len 9216
  --max-num-seqs 1 --async-scheduling --block-size 64
  --mamba-ssm-cache-dtype float16 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching
  --language-model-only --speculative-config '{"method":"mtp","num_speculative_tokens":8}'`.
- Kernel wheel: build `vllm-xpu-kernels` from source at the generation matching
  the image (0.1.12.3), apply `patches/vllm-xpu-kernels/apply_smallm_patch.py`
  (Xe2 block-FP8 small-M op for M ≤ 40; `VXK_SRC` env selects the checkout) and
  `patches/vllm-xpu-kernels/fix_smallm_placement.py`, then mount the built
  `vllm_xpu_kernels` package over the container's `site-packages` path.
- Runtime patches (mount and run at container start, see headers):
  - `patches/vllm-xpu-kernels/patch_fp8_stack.py` — W8A16 reroute
    (`apply_input_quant=False`), M ≤ 40 dispatch to `xe2_block_fp8_small_m`,
    `B70_FP8_FORCE_W8A8` fallback.
  - `patches/vllm-xpu-kernels/patch_fp8_hybrid.py` — hybrid mode
    (`B70_FP8_MODE=hybrid`; W8A8 prefill/large-M, W8A16+small-M decode; used
    for the ≥2048 band).
  - `patches/patch_vllm_worker_affinity.py` — per-worker `ZE_AFFINITY_MASK` +
    SYS_PTRACE (required for TP2 on this memory class).
- Power: `energy1_input` deltas over the exact cell interval, 150 W cap at
  idle, 230 W declared campaign cap; hwmon devices resolved live.

## Known limitations

- Spec-decode acceptance is draw- and content-dependent on this W8A16 stack;
  long-context (>1536 prompt tokens) W8A16 prefill can collapse acceptance and
  is mitigated operationally by the hybrid mode (W8A8 prefill numerics), not
  by an engine-side fix.
- Cn (concurrent) MTP on one server is blocked upstream on this image family
  (GDN spec/non-spec mixing); these results are all C1.
- p1024 median exceeds p512 (acceptance benefit); the max-observed 71.06 tok/s
  is a single-sample maximum and not representative.

## Evidence

- Raw bench JSON + power + parsed summaries:
  `results/qwen38-27-fp8-tp2-k8-n5/bench/`
- Campaign root with full logs, monitor, counters, cleanup state: linked from
  the catalog record; private lab paths removed.
- Correctness: deterministic greedy output matches the no-spec target over the
  stable prompt set of the campaign (coherent-prefix parity, greedy); MTP is
  lossless-equivalent by construction.