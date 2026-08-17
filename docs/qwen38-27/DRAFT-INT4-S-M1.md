# Qwen3.8-27B — Draft INT4 (S + M1) High-Throughput Variant

Two extra runtime patches (apply after the base nightly patches **and** the
concurrency/pointer-safety fixes) that quantize the **speculative-draft** side
of MTP to INT4 g128 sym (GPTQ format), reusing the existing
`torch.ops._xpu_C.int4_gemm_w4a16` kernel.

**Lossless**: only the draft's own LM head / MTP module are quantized. The
verification (target) LM head stays BF16, and draft tokens are verified
against the target (greedy) — the emitted sequence is identical to the MTP4
baseline.

- `patches/patch_draft_lmhead_int4.py` (SHA-256: `ffae41926d5f05f4f38bb985301b5e572092441d06d6063c8820a63a39b8cefc`) — **Phase S**: draft LM head INT4 copy (2.54 GB fp16 → 0.66 GB), removes ~7.6 GB/step of DRAM reads (4 draft passes). Env `B70_DRAFT_LMHEAD_INT4=1`.
- `patches/patch_draft_mtp_int4.py` (SHA-256: `4df179c3e77fd7a248f9b9c0b60217c60caea14ebfd16b7860536fbff3b2a1e9`) — **Phase M1**: the MTP module's 5 linears to INT4 (0.85 GB → ~0.22 GB), removes ~2.6 GB/step. Env `B70_DRAFT_MTP_INT4=1`.

Both default **off** (exact baseline behavior when unset). Anchors are verified
against the legacy Qwen pinned nightly (`vllm/vllm-openai-xpu@sha256:2c427ef477da…`,
vLLM `0.26.1rc1.dev457`) and fail closed when they differ.

## Launch

```bash
docker run -d --name b70-qwen38 -p 8010:8000 --device /dev/dri \
  --group-add $(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p') \
  -v /dev/dri:/dev/dri:ro -v /path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v patches/patch_mtp_nightly.py:/patch_mtp.py:ro \
  -v patches/patch_mtp_boundary.py:/patch_boundary.py:ro \
  -v patches/patch_mtp_ptr_wrap.py:/patch_ptr_wrap.py:ro \
  -v patches/patch_gdn_split_mixed.py:/patch_gdn_split.py:ro \
  -v patches/patch_draft_lmhead_int4.py:/patch_loader_0.py:ro \
  -v patches/patch_draft_mtp_int4.py:/patch_loader_1.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 \
  --entrypoint bash vllm/vllm-openai-xpu@sha256:2c427ef477da... -lc \
  "set -e; python /patch_mtp.py; python /patch_boundary.py; python /patch_ptr_wrap.py; \
   python /patch_gdn_split.py; python /patch_loader_0.py; python /patch_loader_1.py; \
   exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131328 \
   --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 64 \
   --max-num-batched-tokens 4096 --enable-prefix-caching --language-model-only \
   --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"
```

`--device /dev/dri` + the render GID (not `--privileged`), `--max-num-batched-tokens 4096`
(+6.6% long-prefill), fp8 KV, U=0.88 (U=0.90 fills the card with MTP4) and
prefix caching are the recommended production flags.

## Results (measured 2026-08-16/17, harness C1, n=5, 230 W, B70)

Same Qwen3.8-27B GPTQ-INT4 model (gptqmodel 7.3.2 quant), legacy nightly + the
six-patch stack above. `client post-first tok/s`, median n=5:

| Cell | Baseline (2 patches, new image) | **+6 patches (legacy nightly)** |
|---|---|---|
| p512/g128 | 83.7 | **117.1** |
| p8192/g128 | 77.1 | **105.7** |

Same-stack component deltas (measured on the heavy AutoRound quant): Phase S
itself +27% (p8192/g32 71 → 90.4), S+M1 adds +7-9% more (96.5-98.4). MTP
acceptance stays 0.88-0.98; the emitted greedy sequences are identical.

Gate (legacy nightly + Qwen3.8, full six-patch stack):

| Check | Result |
|---|---|
| Quality A (45) | 45/45 |
| Sequential B (40) | 40/40 |
| Concurrent C (8) | 8/8 |
| Long context D (8K/32K) | OK |
| MTP ratio | 0.58-0.61 |
| HumanEval pass@1 (164, executed) | 92.7% |

**E2 self-reported** on a single dual-GPU test bench; independent
reproduction pending. The patches are runtime monkey-patches on a pinned
nightly — upstreaming the INT4 draft path would require a build change in
`vllm-xpu-kernels` and is left as future work.