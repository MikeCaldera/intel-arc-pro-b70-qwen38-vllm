# Ornith-1.5-35B-A3B on the B70

Family index. Keep Qwen, Nemotron, and Muse numbers on their own pages.

| Path | What it is | Status |
|---|---|---|
| [ORNITH-VLLM-XPU.md](ORNITH-VLLM-XPU.md) | MixedCal-v2 recipe: MTP1 decode, 230 W no-spec prefill, graphs | Self-reported E2, C1 |
| [CLAIMS.md](CLAIMS.md) | Observed numbers only — copy from here | Source of truth for this family |
| `benchmarks/ornith15-35a3/launch-ornith-mtp1.sh` | MTP1 launcher (`MODE=no-spec\|mtp1\|mtp2\|mtp4`) | Same `f01e24f6` digest as Qwen3.8 |
| [Dashboard SVG](../assets/b70-ornith15-dashboard.svg) | Lane-1 decode + cold-input + capacity card | Rendered from the host compiler JSON |

## Artifact (do not rename)

Published under **`SergiioB`** (two i's):

- https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2

This is a **local** experts-only GPTQ INT4 G128 with MTP left in BF16. It is not an official Ornith GPTQ.

## Hard rules for this family

1. Same public digest as Qwen3.8-27B (`f01e24f6`), **not** Qwen3.6 Pi `2c427ef` and **not** the v0.21 native-int4moe image that produced Qwen3.6 MTP4 204.6. See [IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md).
2. Research spec default is **MTP1**. MTP4 is slower than no-spec on this single-layer head.
3. Prefill ~9.7k is **configured 230 W, no-spec**. Decode ~96 tok/s is **configured 150 W, MTP1**. Do not mix those cells.
4. DFlash / DFlash2 is **not** a serving path here. There is no Ornith hidden-2048 DFlash2 draft. `z-lab/Qwen3.5-35B-A3B-DFlash` matches hidden/vocab and is **not measured**.
5. LocalMaxxing MTP1 150 W p32/g256 `cmt2sl6eg0hdcmv01gre5o3ub` is APPROVED self-report (`tokSOut` 94.1). That is not the 9.7k prefill cell.
