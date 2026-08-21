# MixedCal-v2 conversion (Ornith-1.5-35B-A3B)

How the published GPTQ was built, what the calibration change
improves, and which percentages are observed. Speed numbers stay on
[CLAIMS.md](CLAIMS.md). Serve from [ORNITH-VLLM-XPU.md](ORNITH-VLLM-XPU.md).

**Self-reported E2.** Isolated C1 later confirmed speed **parity** at
150 W. MixedCal-v2 is not a tok/s win.

## What changed

There is no official Ornith GPTQ. Both B70 artifacts are local GPTQ INT4
symmetric G128 (`desc_act=false`) with MTP left in BF16. MixedCal-v2
keeps that **format and tensor scope** identical to the original
WikiText GPTQ so the only experimental axis is **calibration coverage**.

| Axis | Original (control) | MixedCal-v2 |
|---|---|---|
| Format | GPTQ INT4, G128, `sym=true`, `desc_act=false` | same |
| Quantized tensors | routed-expert `gate_proj` / `up_proj` / `down_proj` only | same |
| Expected expert `qweight` | 40 × 256 × 3 = **30,720** | same |
| MTP | 785 tensors, **0** `qweight` | same |
| Calibration | WikiText | mixed-domain, 128 × 1,536 tokens |
| Public id | local original (not overwritten) | [`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`](https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2) |

The original WikiText GPTQ and the official BF16 source
(`ornith-ai/Ornith-1.5-35B-A3B`, revision
`fbb995a79eedd569a5edc5f2af9644c0fa1124fc`) are immutable inputs.
MixedCal-v2 never overwrites either one.

## Experts-only contract

Quantize **only** routed experts. Do not quantize:

- `lm_head`
- embeddings (`model.language_model.embed_tokens`)
- attention / GDN
- router (`mlp.gate`)
- shared experts
- MTP draft tensors
- vision

GPTQModel 7.3.2 dynamic exclusions used for MixedCal-v2:

```python
{
    "-:.*attn.*": {},
    "-:.*mlp\\.gate$": {},
    "-:.*mtp.*": {},
    "-:.*shared_expert.*": {},
    "-:.*visual.*": {},
    "lm_head": {},
    "model.language_model.embed_tokens": {},
}
```

Staged contract before publication (run `r4`): **30,720** expert
`qweight`, **785** MTP tensors with **0** quantized, **0** forbidden
`qweight`. Artifact size **24,454,916,052** bytes, 6 shards.

## Calibration (the actual change)

| Input | Value |
|---|---|
| Samples | 128 |
| Tokens per sample | 1,536 |
| Total tokens | 196,608 |
| Seed | `15035260819` |
| Corpus SHA-256 | `e88ccd5f999a05a3c78bb9ec86a82256255a9ad97068c787dfc564907a377381` |
| Quantizer | GPTQModel 7.3.2 |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |

The corpus spans code, reasoning, agentic/tool content, SQL, networking,
technical prose, general text, and systems debugging. Routing is
**natural**: the process does not force router outputs or expert IDs.

WikiText under-covers late-layer experts on this 256-expert MoE. GPTQ
then cannot form a usable Hessian for those matrices and falls back to
round-to-nearest (RTN). MixedCal is a broader natural-routing set so
more expert matrices get a real GPTQ solution.

## What RTN fallback means

RTN here is **GPTQModel's weight-only fallback**, not a CPU/XPU kernel
fallback and not a serving-mode switch. A matrix that hits RTN is still
INT4 G128; it just was not GPTQ-solved from calibration.

Lower RTN % is the conversion win. It is **not** a decode or prefill
throughput claim.

## Observed percentages (run `r4`, 2026-08-21)

String-exact count from journald-captured quantizer lines
(30,719 of 30,720 module lines captured; the missing line does not
change the fallback count).

| Artifact | RTN fallbacks | Rate |
|---|---:|---:|
| Original WikiText GPTQ | 7,605 / 30,720 | **24.76%** |
| MixedCal-v2 | 3,186 / 30,720 | **10.37%** |

Relative to the original, MixedCal-v2 removed **58% of RTN fallbacks**
(`1 − 10.37 / 24.76`).

Per-projection MixedCal split is even: **1,062 / 1,062 / 1,062**
(`gate` / `up` / `down`).

Late-layer hotspot (layer 39, 768 expert matrices):

| Artifact | Layer 39 RTN | Rate |
|---|---:|---:|
| Original | 297 / 768 | **38.7%** |
| MixedCal-v2 | 75 / 768 | **9.8%** |

Worst MixedCal layers after the change: 34 (201), 37 (174), 32/36 (156).
The conversion did not zero RTN; it cut the WikiText tail, especially
late experts.

## What it does **not** improve

n=5 no-spec confirmation at 32K, 150 W, cache off, instance-median of
three loads (ORIG first each pair):

| Cell | Original | MixedCal-v2 |
|---|---:|---:|
| p512/g128 client post-first | 70.80 | 70.74 |
| p8192/g128 client post-first | 64.86 | 64.95 |
| p2048/g1 cold input | 6935 | 6968 |
| p8192/g1 cold input | 6926 | 6963 |

Do **not** call MixedCal-v2 faster. That table is speed **parity**.

Prefill **~9.7k** is a **configured 230 W** cell on the same WNA16
nightly, not a MixedCal-vs-WikiText effect. MTP1 ~96 tok/s is the 150 W
research serve. Copy those cells only from [CLAIMS.md](CLAIMS.md).

There is still no BF16 logit/KL or held-out task-quality suite
(`lmx eval suite list` HTTP 404). RTN coverage is a conversion metric,
not a substitute for KL.

## Fail-closed conversion (why this artifact exists)

Earlier WikiText-layout retries crossed the `/mnt/models` reserve at
layer 35. MixedCal-v2 therefore ran on a dedicated 80 GiB ext4
workspace loop-mounted at `/mnt/ornith-mixedcal-workspace`, with:

- unique run id + global artifact lock
- preflight that creates no scratch until capacity/inodes/GPU isolation pass
- `Restart=no` (no automatic retry)
- save to same-filesystem `.incomplete`, contract check, then atomic rename

Completed run: `mixedcal-v2-20260820T204436Z-r4` (7 h 40 m wall,
container exit 0). Host conversion notes:
`B70-DOCS/models/ornith/ornith-1.5-35b-a3b/mixedcal-v2-conversion.md`.

## Download check

```bash
huggingface-cli download SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2 \
  --local-dir "$HOME/models/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2"
```

Verify after download:

- 6 shards, ~22.77 GiB
- 30,720 routed-expert `qweight`
- 785 `mtp.*` tensors, **zero** MTP `qweight`
- `quantize_config.json`: 4-bit, `sym=true`, `group_size=128`, `desc_act=false`
