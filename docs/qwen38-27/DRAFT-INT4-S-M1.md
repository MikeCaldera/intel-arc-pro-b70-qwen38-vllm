# Qwen3.8-27B — Draft INT4 (S + M1)

Runtime patches that **requantize the speculative-draft side** of MTP to
INT4 g128 symmetric (round-to-nearest, not Hessian GPTQ) and route those
linears through `torch.ops._xpu_C.int4_gemm_w4a16`.

Production image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`). The patches fail
closed if `qwen3_5_mtp.py` anchors differ.

## What is and is not lossless

| Piece | After S+M1 |
|---|---|
| Target body (GPTQ-INT4) | unchanged |
| Target / verify LM head | still FP16/BF16 |
| Draft LM head copy | runtime INT4 g128 RTN |
| Draft MTP 5 linears | runtime INT4 g128 RTN |

The target still verifies. Draft **logits are not** identical. Acceptance
can drop. This is a **speed** keep, not a quality-parity claim.

## Same-image n=5 (2026-08-18)

Champion `f01e24f6`, MTP4, cache off, MBT 8192, U=0.88, fp8 KV, C1,
configured **230 W**. Timing: client post-first / input÷TTFT. Accept from
`vllm:spec_decode_*_total`. Raw: cookbook campaign sibling
`qwen38-pr-n5-20260818T214749Z` + long-ctx `qwen38-pr-long-20260818T221813Z`.

| Cell | BF16 draft (cookbook) | Draft INT4 S+M1 | Δ |
|---|---:|---:|---:|
| p512/g128 median n=5 | 81.20 tok/s (78.76–81.97) | **112.65** (111.58–117.73) | **+38.7%** |
| p8192/g128 median n=5 | 77.52 | **103.63** (99.10–107.70) | **+33.7%** |
| p8192/g1 cold input | 1691.1 | 1696.0 | +0.3% (flat) |
| short agentic 3×5 g256 | 43.27 | **57.45** | **+32.8%** |
| accept p512 / p8192 | 95.86% / 96.02% | 94.44% / 94.59% | −1.4 pp |
| measured median W p512 | 195.2 | 205.0 | — |

Long-context C1 agentic (g128, 0 crashes, actual first prompts ~8.3K /
~16.5K / ~32.9K): **+37%** at 8K (68.73 vs 50.06), **+26%** at 16K
(70.86 vs 56.29). 32K was isolated C1 (~245 MiB free) and only **+5%** —
not a serving headline.

Do **not** headline +51%. That table compared a different image, patch
list, MBT, and cache mode (`not_comparable`).

The n=3 chat-harness screen (59.1 → 83.9) used ~p530 filler, not exact
p512. It is superseded.

## Required companion

Stack with **GDN mixed-split v5** (`patches/patch_gdn_mixed_split_v5.py`)
if mixed spec + prefill can share an invocation. The original full-buffer
split is OOB on kernels 0.1.12.3.

Pointer wrap (`patch_mtp_ptr_wrap.py`) is optional and **int64-only**.

## Env

- `B70_DRAFT_LMHEAD_INT4=1` — Phase S (patch default off)
- `B70_DRAFT_MTP_INT4=1` — Phase M1 (patch default off)

Set both for the Qwen3.8-27B MTP4 speed keep. Leave unset to recover
BF16-draft acceptance.

## Status

Optional recipe keep on this champion image. Speed-only; no token / KL /
task-quality parity evidence.
