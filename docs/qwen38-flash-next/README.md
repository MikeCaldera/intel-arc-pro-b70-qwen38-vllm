# Qwen3.8-Flash-Next on two Arc Pro B70s (llama.cpp SYCL)

This directory is the family hub for **Qwen3.8-Flash-Next**, not Qwen3.8-27B.
The 27B GPTQ/MTP and FP8 TP2 recipes stay under [`docs/qwen38-27/`](../qwen38-27/README.md).
Do not mix those numbers with this page.

| Route | Hardware | Serving goal | Start here |
|---|---|---|---|
| llama.cpp SYCL, community M64 GGUF | Two B70 cards, **C1** | Decode-first fused FP32 or prefill-first fused F16 | [QWEN38-FLASH-NEXT-LLAMACPP.md](QWEN38-FLASH-NEXT-LLAMACPP.md) |
| vLLM XPU / AutoRound / official BF16 | — | Not a public recipe on this host | Closed |

`c8` / `c16` / `c128` on this family are **context windows**. Every published speed cell is **C1**.
