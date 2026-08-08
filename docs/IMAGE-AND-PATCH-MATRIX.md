# vLLM XPU Image and Patch Matrix

## Current supported recipe

Use this exact public image:

```text
vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97
```

Observed version inside the image:

```text
vLLM v0.26.1rc1.dev457+gc810e5ee9
vllm-xpu-kernels 0.1.12
```

Use this model:

```text
llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
```

The local model directory must contain the preserved `mtp.*` tensors. The official GPTQ-INT4 checkpoint declares an MTP layer in config but does not contain those draft weights.

Apply these patches in order:

| Order | File | Purpose |
|---:|---|---|
| 1 | `patches/patch_mtp_nightly.py` | Build the preserved BF16 MTP draft outside the target GPTQ quant config |
| 2 | `patches/patch_mtp_boundary.py` | Complete exact 128K when the final MTP4 group is shorter than five tokens |

Do not apply `patch_xpu_int4_moe_v4.py` or `patch_mtp_bf16_draft.py` to this nightly. Those files target the historical vLLM 0.21 stack and use different source anchors.

## One-command launch

```bash
git pull
bash benchmarks/launch-mtp4-128k-nightly.sh \
  /path/to/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 8000
```

The launcher pins the image by digest, mounts both current patches, uses MTP4, sets `--max-model-len 131072`, and sets `--max-num-batched-tokens 8192`.

Follow startup:

```bash
sudo docker logs -f b70-mtp4-128k
```

Expected first lines include both:

```text
patched .../qwen3_5_mtp.py
patched .../gdn_attn.py
```

Check health after model load:

```bash
curl -f http://127.0.0.1:8000/health
```

## Historical image warning

```text
intel/vllm:0.21.0-xpu-int4moe
```

was a local derived image used during the 2026-08-06 research campaign. It was not published to Docker Hub. A reader cannot pull it.

`intel/vllm:0.21.0-xpu` is not a drop-in replacement. The old cookbook patches expected changes present in the local derived image, so applying them to the public base tag can fail on missing source anchors or runtime APIs.

Historical files remain in this repo to preserve the evidence trail:

- `patches/patch_xpu_int4_moe_v4.py`
- `patches/patch_mtp_bf16_draft.py`
- `benchmarks/launch-mtp-bf16draft.sh`

Do not use them for the current quick start.

## Exact 128K proof

The current two-patch path completed:

| Field | Value |
|---|---:|
| Exact prompt | 130,944 tokens |
| Requested/completed output | 128 / 128 tokens |
| Total sequence | 131,072 tokens |
| TTFT | 48.601 s |
| Client post-first rate | 96.87 tok/s |
| MTP acceptance | 72.31% |
| Prefix-cache hit delta | 0 |
| Free VRAM after load | 2,619 MiB |
| Logged KV capacity | 165,961 tokens |

`Client post-first rate` is not an engine-native timing field. It is `(completion_tokens - 1) / (request_end - first_token)`.

Full commands and real-world tables: [`REAL-WORLD-PI-BENCHMARKS.md`](REAL-WORLD-PI-BENCHMARKS.md).
