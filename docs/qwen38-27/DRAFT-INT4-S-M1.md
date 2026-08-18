# Qwen3.8-27B — Draft INT4 (S + M1) research variant

Runtime patches that **requantize the speculative-draft side** of MTP to
INT4 g128 symmetric (round-to-nearest, not Hessian GPTQ) and route those
linears through `torch.ops._xpu_C.int4_gemm_w4a16`.

This is **not** the cookbook production recipe. The production Qwen3.8
image is `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`). These patches
were authored against the **legacy** `2c427ef` nightly (vLLM 0.26.1).
They fail closed if `qwen3_5_mtp.py` anchors differ.

## What is and is not lossless

| Piece | After S+M1 |
|---|---|
| Target body (GPTQ-INT4) | unchanged |
| Target / verify LM head | still FP16/BF16 |
| Draft LM head copy | runtime INT4 g128 RTN |
| Draft MTP 5 linears | runtime INT4 g128 RTN |

Greedy **emitted** tokens can match the BF16-draft baseline because the
target still verifies. Draft **logits are not** identical. Acceptance can
drop. Do not call this a lossless speedup.

## Hardware-feasible claim (not the original +51%)

Decode on the B70 is GDDR6-bound. Quantizing a 2.5 GB vocab head that is
reread on every draft pass is the right bottleneck class.

The original PR table compared **83.7 (champion `f01e24f6`, 2 patches)**
to **117.1 (legacy `2c427ef`, six patches, MBT 4096, prefix cache on)**.
That is `not_comparable` / `best_cell_cross_result`, not a Phase S+M1 A/B.
Do not headline +51% on the cookbook champion stack.

Same-image n=3 screen on `f01e24f6` (2026-08-18,
`B70-DOCS/results/qwen38-pr-ab-20260818T205619Z/`, C1 chat harness,
configured 230 W, cache off, MBT 8192, **PROVISIONAL**):

| Arm | p512/g128 median post-first | p8192/g1 input/TTFT | agentic 15-turn median |
|---|---:|---:|---:|
| cookbook MTP4 | 59.1 t/s (~p530) | 1748 t/s (~p4130) | 44.3 t/s |
| v5 only | 62.7 | 1746 | 44.4 |
| v5 + draft INT4 S+M1 | **83.9** | 1757 | **59.6** |

Prefill is flat. The 83.9 cell matches the published cookbook Run 40 card
(83.7 n=5 exact p512), it does **not** beat it. The +42% / +35% deltas
are versus this campaign’s cookbook arm on the same chat harness, not
versus 83.7. Acceptance counters were empty — do not claim lossless.

Do not replace the cookbook C1 card with these n=3 numbers.

## Required companion

Stack with **GDN mixed-split v5** (`patches/patch_gdn_mixed_split_v5.py`),
not the original PR #1 full-buffer split. The original split is OOB on
kernels 0.1.12.3 (host `narrow` + global `token_indx`).

Pointer wrap (`patch_mtp_ptr_wrap.py`) is optional and **int64-only**.
Do not signed-wrap uint64 `dst_ptrs_np`.

## Env

- `B70_DRAFT_LMHEAD_INT4=1` — Phase S (default off)
- `B70_DRAFT_MTP_INT4=1` — Phase M1 (default off)

## Status

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Same-image (`f01e24f6`) n=3
screen exists (table above). Still need n≥5 exact-token C1 plus
`spec_decode` accepted/proposed counters before a recipe keep.

Author self-report (E2, legacy image, +51%) is retained in the PR
discussion. It is not a cookbook result card.
