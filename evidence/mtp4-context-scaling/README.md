# Intel Arc Pro B70 — Qwen3.8-27B MTP4 Context-Scaling Evidence

This directory preserves the raw `results.json` files used for the
Qwen3.8-27B context-scaling benchmark on a single Intel Arc Pro B70.

## Model

- Qwen3.8-27B
- GPTQ INT4 target weights
- BF16 MTP draft
- Speculative decoding: MTP4
- 128 generated tokens per measured request

## Method

The context-scaling benchmark used:

- Exact rendered prompt-token counts
- Cold/unique prompts
- Prefix caching disabled
- Streaming generation
- `temperature=0`
- `ignore_eos`
- Client-side post-first-token decode timing

Decode throughput is calculated from completion-token counts and
client-side timing. It is not SSE chunk throughput and not prompt-prefill
throughput.

## Confirmed MTP4 results

| Prompt tokens | Measured reps | Median decode |
|---:|---:|---:|
| 512 | 5 | 81.2390 tok/s |
| 8,192 | 10 | 73.3014 tok/s |
| 16,384 | 10 | 74.7602 tok/s |
| 32,768 | 10 | 68.3402 tok/s |
| 65,536 | 10 | 66.1055 tok/s |
| 120,000 | 10 | 50.3087 tok/s |

Each raw dataset reports accepted speculative-token positions:

`[0, 1, 2, 3]`

This confirms that the context-scaling campaign used four MTP
speculative positions.

Every measured dataset also reports zero prefix-cache hits.

## Verification

From the repository root, run:

    python3 evidence/mtp4-context-scaling/verify_results.py

Expected verdict:

    VERDICT: ALL SIX DATASETS CONFIRMED MTP4

Verify file integrity with:

    cd evidence/mtp4-context-scaling
    sha256sum -c SHA256SUMS

## Important distinction: 84.65 tok/s reference result

The separately reported **84.65 tok/s** result was also MTP4, but came
from the original p512/g128 short-context validation/reference benchmark.

It used a different benchmark run methodology and therefore is not used
as the 512-token point in this context-scaling curve.

The correct 512-token median for this exact-context scaling campaign is:

**81.2390 tok/s**

The 84.65 tok/s result should be reported separately as the short-context
MTP4 reference/replication result.

## Sanitized benchmark evidence

The six `context-*-results.json` files in this directory are sanitized
metric-only extracts of the original benchmark `results.json` artifacts.
Prompt text, generated response text, reasoning text, entropy prefixes,
and raw SSE paths were intentionally removed before publication.

The larger per-request `.sse.jsonl` files are intentionally not included
because they contain generated response text and are not required to
verify the MTP4 configuration or the reported median throughput.
