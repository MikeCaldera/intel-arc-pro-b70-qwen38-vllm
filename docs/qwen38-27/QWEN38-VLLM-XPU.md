# Qwen3.8-27B vLLM XPU 4-Mode Recipe

![Dashboard SVG](../assets/b70-qwen38-dashboard.svg)

## Stack (latest tested vLLM XPU nightly)
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` (local tag `nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`)
- Observed vLLM `0.27.2rc1.dev77+gac7509e2b`; vllm-xpu-kernels `0.1.12.3`
- Model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (Qwen3_5ForConditionalGeneration, dense 27B, GPTQ-INT4 sym G128 desc_act=false, preserved BF16 1-layer MTP head; local dir `/qw38-gptq-out/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`)
- Patches (order): `patch_mtp_nightly.py` sha256 `f1db50bf617aacbb0a672daf172be32a98b2a73c7817ebab0c6317b22d11f36a` then `patch_mtp_boundary.py` sha256 `41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50` (both in the cookbook `patches/` dir)

## Launch (per mode; true per-mode server)
```bash
# no-spec
docker run -d --name qw38speed -p 8000:8000 --device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | sort -u | head -1) \
  -v /dev/dri:/dev/dri:ro -v /qw38-gptq-out/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v patches/patch_mtp_nightly.py:/patch_mtp.py:ro -v patches/patch_mtp_boundary.py:/patch_boundary.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc \
  "set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --language-model-only"
```
MTP1/2/4: same, but `--gpu-memory-utilization 0.88` and add `--speculative-config '{"method":"mtp","num_speculative_tokens":1|2|4}'`. Set power cap `echo 230000000 | sudo tee $B70_HWMON/power1_cap` (resolve hwmon: `/sys/class/hwmon/hwmon*/name` == `xe` and PCI 0000:0b:00.0). Restore 150 W after. Wait for `/health`.

## Results (audited, E2, C1, cache off, 230 W cap, n=5 unless noted; client monotonic timing; cold input = tokens/TTFT, decode = client post-first)

Cold input (tok/s):
| Mode | p2048/g1 | p4096/g1 | p6144/g1 | p8192/g1 |
|---|---:|---:|---:|---:|
| no-spec | 1851 | 1848 | 1809 | 1774 |
| MTP1 | 1817 | 1813 | 1776 | 1738 |
| MTP2 | 1810 | 1810 | 1770 | 1736 |
| MTP4 | 1795 | 1800 | 1767 | 1728 |

Decode p512 (tok/s): no-spec 32.9/32.9/32.8/32.7 (g32..g512); MTP1 51.1/52.0/52.1/51.4; MTP2 62.9/65.8/65.3/57.8; MTP4 76.5/**83.7**/82.9/76.4.
Decode p8192 (tok/s): no-spec 31.3/31.5/31.5/31.5; MTP1 49.0/50.0/47.2/46.0; MTP2 60.0/62.9/55.4/50.1; MTP4 76.8/**77.1**/60.4/52.1.
Control p9445/g128: 31.4/49.9/62.3/79.0. Full-context p130944/g128 (n=1): 23.2/38.9/44.4/**56.3**; p130560/g512 (n=1): 23.1/35.3/36.3/36.2.
MTP acceptance: MTP1 100.0/99.7/100.0, MTP2 99.3/97.7/96.5, MTP4 95.0/93.7/96.2 (p512/g128, p8192/g128, p130944/g128).
Power (campaign window incl warmups): 197/198/199/196 W mean; max 0.5 s interval-avg 274/276/275/274 W.

## Comparison table (Qwen3.6-27B dense, old pinned image, Run 31 — CLASSIFY as `directional_only`: different model checkpoint, build, and U)
Qwen3.6-27B (2c427ef, vLLM 0.26.1rc1.dev457, U=0.90, cache on zero-delta): MTP4 p512/g128 69.3, p8192/g128 64.1, p9445 67.25, p130944/g128 47.6, cold p8192 1654-1742 (table in /home/sergio/B70-DOCS/results/vllm-dense27-4mode-230w-20260809T163138Z-74780/tables.md).
Qwen3.8-27B (ac7509e2, vLLM 0.27.2rc1.dev77): MTP4 p512/g128 83.7, p8192/g128 77.1, p9445 79.0, p130944/g128 56.3, cold p8192 1728-1774.
Label: "best-cell cross-result / directional_only — different model, build, U, cache mode; not a model-alone claim".

## Benchmark-model skill usage section
Used `benchmark-model` Lane 3 (4-mode characterization), n=3 screens (ac7509-lane0) → n=5 full matrix, unique entropy-first prefixes, zero cache-hit delta + zero cache-query delta (runtime proof cache off), exact rendered tokens, same-shape warmups, C1 only, client monotonic timing, configured cap vs measured draw separated, claims audit before any headline. Raw evidence: `/home/sergio/B70-DOCS/results/vllm-qwen38-4mode-230w-20260815T170840Z-581426/`.

## Evidence and gate
- Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE (E2 self-reported, independent reproduction pending)
- Evidence links: raw run dir, summary.json, tables.md, claims-audit-20260815.md
- No LMX/portfolio/social numbers yet (submission pending user gate)
