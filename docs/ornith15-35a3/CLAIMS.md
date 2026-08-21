# Ornith-1.5-35B-A3B claims lock

Copy numbers only from this file and from
`results/ornith15-mixedcal-v2-summary/summary.json`. Do not invent LocalMaxxing
numbers or mix 150 W tables with the 230 W prefill A/B.

**Self-reported E2 with raw evidence.** Isolated C1, cache off, greedy
diagnostic. Not independently reproduced. LocalMaxxing payloads are
validate-local only and **not submitted**. Do not mix 150 W decode tables
with the 230 W prefill cells.

## Stack (do not substitute)

| Component | Exact tested value |
|---|---|
| GPU | Intel Arc Pro B70 32 GB, `xe` driver |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b` |
| `vllm-xpu-kernels` | `0.1.12.3` |
| Dtype | float16 |
| MoE backend observed | XPU WNA16 (`int_wna16`) |
| Cache | `--no-enable-prefix-caching` |
| Timing | client SSE post-first; cold input = actual endpoint tokens / client TTFT |
| Scope | C1 only |
| Power for the tables below | configured **150 W** unless a cell names another cap |

## Artifacts

| Artifact | RTN fallback | Contract |
|---|---|---|
| Original WikiText GPTQ | 7,605 / 30,720 = 24.76% | 30,720 expert `qweight`, 785 MTP BF16, 0 MTP leaks |
| MixedCal-v2 | 3,186 / 30,720 = 10.37% | same contract; 24,454,916,052 bytes |

Calibration for MixedCal-v2: 128 samples × 1,536 tokens (196,608), seed `15035260819`, corpus SHA-256 `e88ccd5f…`. Experts-only (`gate_proj`/`up_proj`/`down_proj`). MTP, attention/GDN, router, shared experts, embeddings, lm_head, vision not quantized.

## Speed — n=5 confirmation, no-spec, 32K, 150 W, cache off

Statistic: **median of three instance medians** (ORIG first each pair, then CAND; three pairs). Each instance is n=5 post-warmup.

| Cell | Metric | Original | MixedCal-v2 | Comparison |
|---|---|---:|---:|---|
| p512/g128 | client post-first tok/s | 70.80 | 70.74 | matched_except_quant; overlap / parity |
| p8192/g128 | client post-first tok/s | 64.86 | 64.95 | matched_except_quant; overlap / parity |
| p2048/g1 | cold input tok/s | 6935 | 6968 | matched_except_quant; overlap / parity |
| p8192/g1 | cold input tok/s | 6926 | 6963 | matched_except_quant; overlap / parity |

Do not call MixedCal-v2 “faster.” The n=5 confirmation is speed parity at 150 W.

## Speed — MTP1, 16K, 150 W, n=5

| Artifact | p512/g128 | p8192/g128 | Acceptance (accepted/draft) |
|---|---:|---:|---|
| Original | 97.88 | 91.06 | 78.5% (604/769) |
| MixedCal-v2 | 96.43 | 89.85 | 77.0% (595/773) |

## Speed — MixedCal-v2 MTP1 at 131,072, U=0.90, n=5

| Cell | client post-first | Acceptance |
|---|---:|---|
| p512/g128 | 98.16 | 85.3% (617/723) |
| p8192/g128 | 95.25 | same window |

## Exact-token capacity (MixedCal-v2, `/tokenize` calibrated)

These supersede the char-approx ladder rows (those landed at ~1.38× target tokens).

| Serve | Exact cell | n | client post-first median | cold input median |
|---|---|---:|---:|---:|
| 65,536 no-spec | p65408/g128 | 3 | 54.49 | — |
| 65,536 no-spec | p2048/g1 | 5 | — | 7600 |
| 65,536 no-spec | p8192/g1 | 5 | — | 7700 |
| 65,536 no-spec | p32768/g1 | 5 | — | 5845 |
| 131,072 no-spec | p130944/g128 | 3 | 45.84 | — |
| 131,072 MTP1 + boundary patch | p130944/g128 | 3 | 70.25 | — |
| 262,144 no-spec, `--kv-cache-memory 6623879680` | p262016/g128 | 3 | 35.35 | — |

Finish reason `length` on every retained sample. A successful 262K **serve** is not a 262K quality claim.

## Historical day-0 (original artifact, n=3, **230 W cap**)

Do not mix with the 150 W MixedCal tables.

- no-spec 32K p512/g128: 69.0 tok/s
- no-spec cold input p~2.9K/g1: **9,537 tok/s** (8,572–9,616)
- MTP1 16K p512/g128: 94.3 tok/s, pos0 accept 80.3%

The 9.5k prefill is **configured 230 W**, original artifact, n=3. The 150 W MixedCal p2048/g1 confirmation is ~7.0k. The paired MixedCal A/B is below.

## Speed — MTP2 / MTP4, 16K, 150 W, n=5 (depth A/B)

| Artifact | Depth | p512/g128 | p8192/g128 | Acceptance |
|---|---|---:|---:|---|
| Original | MTP2 | 82.03 | 77.47 | 41.1% (604/1468) |
| MixedCal-v2 | MTP2 | 84.16 | 75.86 | 41.7% (610/1464) |
| Original | MTP4 | 64.35 | 59.57 | 20.4% (602/2948) |
| MixedCal-v2 | MTP4 | 66.27 | 59.69 | 22.1% (628/2848) |

MTP4 on this head is **slower than no-spec** (~70.7 p512). MTP2 is between no-spec and MTP1. Winner remains **MTP1**. Comparison vs MixedCal MTP1 16K n=5 (96.43 / 89.85, 77.0%): matched_except_quant within each depth; depth comparison is matched except speculation.

## Speed — MixedCal-v2 MTP1 16K at 165 W configured cap, n=5

| Cell | 150 W (S8) | 165 W (D5) |
|---|---:|---:|
| p512/g128 | 96.43 | 97.19 |
| p8192/g128 | 89.85 | 92.68 |
| Acceptance | 77.0% | 82.0% (604/737) |

Not a paired A/B (different loads, different sessions). Directional only. Do not headline a 165 W win.

## Speed — paired 150↔230 W cold input, MixedCal-v2, no-spec 32K

One warm server, cache off, three alternating rounds (`r1` 150 then 230, `r2` 150 then 230, `r3` 150 then 230). Comparison: **matched except configured cap**. Statistic: per-round median of retained samples with a valid client TTFT (g1 rows with `ttft_s=null` excluded). Evidence: `results/ornith15-mtpdepth-power-20260821T075858Z/power-ab/`.

| Round | cap | p2048/g1 median (n valid) | p8192/g1 median (n valid) |
|---|---:|---:|---:|
| r1 | 150 W | 7271 (n=4) | 7036 (n=3) |
| r1 | 230 W | 9748 (n=5) | 9647 (n=3) |
| r2 | 150 W | 7212 (n=4) | 7050 (n=3) |
| r2 | 230 W | 9713 (n=5) | 9683 (n=5) |
| r3 | 150 W | 7055 (n=5) | 7062 (n=5) |
| r3 | 230 W | 9771 (n=5) | 9670 (n=3) |

230 W recovers the day-0 **~9.5–9.7k** cold-input class on MixedCal-v2. 150 W stays the **~7.1k** class. Prefill lever on this WNA16 image is **configured power cap**, not MixedCal vs original.

Campaign-window average card draw from `energy1_input` (whole alternating-round files, ~9–12 s each): **~150–152 W** at the 150 W cap and **~222 W** at the 230 W cap. Package temp peaked 61–64 °C at 150 W and 70–73 °C at 230 W. **GT clocks were not pinned and were not an independent A/B.** The independent variable is `power1_cap`. Do not claim a clock-only prefill win.

Caveat: after each 230→150 drop, the first retained p2048/g1 sample is systematically ~8.7k before later samples settle ~6.9k. Round medians include that sample. Do not treat ~8.7k as a 150 W sustained rate. 230 W p2048 is stable across the five retained samples (~9.65–9.80k).

## Speed — DraftINT4 S+M1 runtime overlay, MixedCal-v2, 16K, 150 W, n=5

Runtime overlay only (`B70_DRAFT_LMHEAD_INT4` + `B70_DRAFT_MTP_INT4` on
`qwen3_5_mtp.py`). Does **not** mutate the MixedCal-v2 artifact. MTP, lm_head,
attention, router remain BF16/FP16 on disk. Evidence:
`results/ornith15-draftint4-20260821T084132Z/`. DI0 dry-run anchors passed.
Logs show both MTP linears and lm_head quantized at load.

| Serve | vs | p512/g128 | p8192/g128 | Acceptance |
|---|---|---:|---:|---|
| DI1 MTP1 DraftINT4 | S8 MixedCal MTP1 BF16 draft | 106.27 vs 96.43 | 97.16 vs 89.85 | 81.9% (602/735) vs 77.0% (595/773) |
| DI2 MTP2 DraftINT4 | D2 MixedCal MTP2 BF16 draft | 96.05 vs 84.16 | 86.80 vs 75.86 | 44.8% (634/1414) vs 41.7% (610/1464) |

Acceptance gate (>3 pp drop fails): **pass** on both depths (acceptance rose).
Comparison is matched except draft-linear precision, **separate loads** (not a
paired same-server A/B). Do not treat +10 tok/s as a MixedCal conversion win.
MTP1 + overlay remains faster than MTP2 + overlay. Default research serve is
still MixedCal-v2 **MTP1**; overlay is optional and local-only.

VRAM after load: DI1 5501 MiB free, DI2 5395 MiB free (16K, U=0.85).

## LocalMaxxing speed-test — MixedCal-v2 MTP1, 16K, 150 W, **not submitted**

CLI remote C1, one warmup + n=3, 32 endpoint prompt tokens / 256 output, cache off,
BF16 MTP draft (no DraftINT4 overlay). Timing: LocalMaxxing client-observed HTTP/SSE.
Evidence: `results/ornith15-lmx-mixedcal-20260821T090450Z/`. `validate-local` **valid**.
**No submit.**

| Field | Value |
|---|---|
| `tokSOut` median | **94.1** (94.1 / 94.1 / 94.2) |
| `ttftMs` median | 48.88 (48.36–49.05) |
| `tokSPrefill` | 654.7 — **TTFT-derived from 32 prompt tokens; not a p2048/p8192 cold-input anchor** |
| Hardware `powerWatts` in payload | 150 (matches configured cap) |
| Post-load VRAM | 5471 MiB free |

Do not compare this 94.1 to the n=5 p512/g128 **96.43** cell: different prompt length (32 vs ~512–778), different g (256 vs 128), different timer, n=3 vs n=5. Original-artifact LMX at **230 W** was 85.8 tok/s on the same p32/g256 shape; that is **not** a matched A/B (cap + artifact both differ).

## LocalMaxxing + harness — MixedCal-v2 no-spec, 32K, **230 W**, long prompt, **not submitted**

Same image digest, cache off (`enable_prefix_caching: False`; prefix-cache hits/queries delta **0**), `--language-model-only`, `--max-model-len 32768`, no speculative config. Unique entropy at the start of the prompt. Evidence: `results/ornith15-lmx-prefill230-20260821T095259Z/`. `validate-local` **valid**. **No submit.**

This is the LocalMaxxing cell that belongs to the **~9.7k prefill class**. It is **not** the 150 W p32 `tokSPrefill` 654.7.

### Result — LMX `tokSPrefill`

**9,556.4 tok/s median cold input (LMX `tokSPrefill`)**, with **9,738.9 tok/s maximum observed `tokSTotal`**, on one Intel Arc Pro B70 using vLLM XPU `0.27.2rc1.dev77+gac7509e2b` / kernels `0.1.12.3` and MixedCal-v2 GPTQ INT4.

- Workload: C1, actual endpoint **2,899** prompt tokens, g=1 (`--max-tokens 1`)
- Statistic: LMX median of n=5 post-warmup; `ttftMs` p50 **303.36** (297.72–304.16)
- Cache: `warm_model_cold_prefix`; hits 0→0, queries 0→0
- Power: configured **230 W**; campaign-window `energy1_input` average **208.8 W** over the 12.2 s harness+LMX sampler (includes idle gaps; not a phase-isolated prefill wattage). Peak 1 s interval-average **236.8 W**. Tpkg 55→71 °C.
- Timing source: LocalMaxxing client HTTP/SSE; `tokSPrefill` estimated from TTFT and endpoint prompt tokens
- Correctness: coherent_output_smoke_tested only
- Validation: E2; independent reproduction pending
- Do **not** publish LMX `tokSOut` **19,153.8** — that is a 1-token completion divided by a sub-millisecond post-first window

### Same-load HTTP harness (authority for the 9.7k class on this serve)

`ornith-screen3.py`, one discarded same-shape warmup per cell, ignore_eos, finish=`length`. Target p2048 landed **~2,887–2,896** actual tokens (entropy prefix overshoot); target p8192 landed **~11,356–11,366**.

| Cell | n valid TTFT | median cold input | values |
|---|---:|---:|---|
| p2048/g1 | 4 of 5 (`ttft_s=null` excluded) | **9,428** | 9334 / 9410 / 9516 / 9445 |
| p8192/g1 | 5 of 5 | **9,608** | 9699 / 9634 / 9605 / 9607 / 9608 |

Comparison vs paired A/B MixedCal 230 W p2048 round medians 9748 / 9713 / 9771: **matched except load** (this is a later no-spec 32K serve, not the alternating-round A/B). Same ~9.5–9.7k class. One p2048 sample dropped for missing client TTFT.

## Not yet a claim

- native XpuFusedMoe int4 v4 / int8-store backport (the 204.6 Qwen3.6 path)
- DFlash2 (no Ornith/hidden-2048 draft; Rahul write-up is SGLang Qwen3.8-27B)
- LocalMaxxing **submission**
- BF16 logit/KL or task-quality suite

## Serving recommendation (research, 150 W)

MixedCal-v2 + `--speculative-config {"method":"mtp","num_speculative_tokens":1}`.
Optional local DraftINT4 S+M1 overlay screened faster than BF16 draft on this
image; it is not part of the published weight files. MTP4 is not the winner on
this single-layer MTP head (day-0 per-pos 81/15/2.5/0.5%).

## Evidence roots

- Conversion: `/mnt/models2/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T204436Z-r4/`
- n=5 no-spec: `results/ornith15-n5-maxkv-20260821T054323Z/`
- MTP1 + ladder: `results/ornith15-n5-maxkv-resume-20260821T062400Z/`
- Exact boundary: `results/ornith15-boundary-fixup-20260821T072410Z/`
- MTP depth + power A/B: `results/ornith15-mtpdepth-power-20260821T075858Z/`
- DraftINT4: `results/ornith15-draftint4-20260821T084132Z/`
- LMX MTP1 150 W p32: `results/ornith15-lmx-mixedcal-20260821T090450Z/`
- LMX + harness 230 W long-prompt: `results/ornith15-lmx-prefill230-20260821T095259Z/`
- Compiler: `results/ornith15-mixedcal-v2-summary/summary.json`
