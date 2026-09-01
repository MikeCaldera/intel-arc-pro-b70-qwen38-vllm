# Qwen3.8-Flash-Next on two Arc Pro B70s

llama.cpp SYCL, one request at a time (`-np 1`). 8K / 16K / 128K on this page
are **context windows**, not concurrency.

This is a different model from [Qwen3.8-27B](../qwen38-27/README.md). The
numbers are llama.cpp engine timings, not vLLM client rates.

![Leaders](../assets/b70-qwen38-flash-next-dashboard.svg)

![Prefill and decode vs context](../assets/b70-qwen38-flash-next-context.svg)

## What we measured

Self-reported, n=5 after one discarded warmup. Sampling: temperature 1.0,
`top_p=0.95`, `top_k=20`, `ignore_eos`, exactly 128 output tokens, cache off.

| Build | Context | Prompt / output | Decode | Cold input |
|---|---|---|---:|---:|
| FP32 (`GGML_SYCL_F16=OFF`) | 8K | p512 / g128 | **23.38** tok/s (23.22–23.73) | 183.98 |
| F16 (`GGML_SYCL_F16=ON`) | 16K | actual p9096 / g128 | 20.34 tok/s | **594.49** (594.11–595.45) |

`GGML_SYCL_F16` is a compile-time flag. Build one binary per server.

Average card draw on those cells was about 97–99 W (GPU 0) and 108 W (GPU 1)
against a 195 W cap. With the 88 GiB model resident, GPU 1 package sat near
62 °C at idle.

Smoke on every measured server: `B70-FLASH-OK`, `37*19=703`, `Paris`.

Raw JSON: [results/qwen38-flash-next-dual-b70-c1/](../../results/qwen38-flash-next-dual-b70-c1/).

## Hardware and weights

- Two Intel Arc Pro B70 32 GB, CPU-attached x8 + x8, no GPU P2P.
- GGUF: [`AtomicChat/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)
  folder `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`, revision
  `142262902a46f7daed19c79d0771534c8106ad59` (33 shards, 88.03 GiB).
- Mixed quant: gate/up **IQ3_S**, down **IQ4_NL**, per-layer embeddings **Q5_1**
  on CPU. The folder name `Q4_K_M` is not a uniform Q4_K_M file.
- KV: `q8_0` K + `q4_1` V. Flash attention on. No MTP in this artifact.

## Reproduce

Need Intel oneAPI (IntelLLVM 2026.0.0), CMake, and two B70s with Level Zero.

### 1. Weights

```bash
huggingface-cli download AtomicChat/Qwen3.8-Flash-Next-GGUF \
  --revision 142262902a46f7daed19c79d0771534c8106ad59 \
  --include 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/*' \
  --local-dir "$HOME/models/qwen38-flash-next-m64"
```

First shard:

```text
$HOME/models/qwen38-flash-next-m64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf
```

### 2. llama.cpp pin

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 9723942adc518b43c4b95dc4dce6906903eb5e09
```

### 3. Patches (this cookbook, in this order)

```bash
git apply /path/to/intel-arc-pro-b70-inference-cookbook/patches/llamacpp-sycl/flashnext-arch-overlay.patch
git apply /path/to/intel-arc-pro-b70-inference-cookbook/patches/llamacpp-sycl/sycl-fused-mmvq-and-gpu-group.patch
```

SHA-256:

| File | SHA-256 |
|---|---|
| `flashnext-arch-overlay.patch` | `bf9b00b89cf132c6db95bd51f7f3398b5362986c7c116e083631ac9cec8972b1` |
| `sycl-fused-mmvq-and-gpu-group.patch` | `4fb812ee13d0df5217b2a93f3d11e9cb98afccaf55d08963d16674da4ad2c836` |

The second patch is fused IQ3_S / IQ4_NL `MUL_MAT_ID` MMVQ plus on-device
expert-ID grouping. Grouping is on by default; `GGML_SYCL_DISABLE_MMID_GPU_GROUP=1`
turns it off. The n=5 prefill median 594.49 was taken before grouping; a matched
n=3 A/B later measured 599.48 vs 594.67 (+0.81%). Decode is unchanged.

Older split files `sycl-fused-mmvq-iq3s-iq4nl.patch` and
`sycl-mmid-gpu-group-20260901.patch` conflict if both are applied. Use the
combined kernel patch.

### 4. oneAPI

```bash
source /opt/intel/oneapi/setvars.sh --force
```

### 5. Build the binary you will serve

Decode recipe (8K numbers above):

```bash
cmake -S . -B build-sycl \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=OFF \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target llama-server -j
```

Prefill recipe (16K numbers above):

```bash
cmake -S . -B build-sycl-f16 \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl-f16 --target llama-server -j
```

Measured llama-server SHA-256 (pre-grouping fused MMVQ builds):

- FP32: `f56a838b0e1a16aa8a1321ce2a217c8f3e972e961cf0aca706f76f4d6421c98b`
- F16: `2da6e9fc89d9cb68368feb1a2e7ef1045ff705acb16721755ac1f72149dd1ee3`

### 6. Runtime env

```bash
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
```

Set each card to 195 W through the `xe` hwmon `power1_cap` (restore 150 W when
finished). Confirm with `sycl-ls` that SYCL0 / SYCL1 are the two B70s.

### 7. Serve

Export the first shard:

```bash
export MODEL="$HOME/models/qwen38-flash-next-m64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
```

8K decode recipe (FP32 binary):

```bash
./build-sycl/bin/llama-server -m "$MODEL" \
  --device SYCL0,SYCL1 --tensor-split 1,1 --split-mode layer -ngl 99 \
  -ot 'per_layer_token_embd=CPU' -c 8192 -fa on -t 6 -tb 14 -np 1 \
  -b 512 --ubatch-size 512 --cache-type-k q8_0 --cache-type-v q4_1 \
  --no-warmup --no-cache-prompt --host 127.0.0.1 --port 8001
```

16K prefill recipe (F16 binary):

```bash
./build-sycl-f16/bin/llama-server -m "$MODEL" \
  --device SYCL0,SYCL1 --tensor-split 1,1 --split-mode layer -ngl 99 \
  -ot 'per_layer_token_embd=CPU' -c 16384 -fa on -t 6 -tb 14 -np 1 \
  -b 3072 --ubatch-size 3072 --cache-type-k q8_0 --cache-type-v q4_1 \
  --no-warmup --no-cache-prompt --host 127.0.0.1 --port 8001
```

128K: same flags with `-c 131072 -b 1024 --ubatch-size 1024`. Larger `-b` was
rejected (`n_gpu_layers already set by user to 99`). Do not pass `--no-mmap`
or `--mlock` on this GGUF.

`-fa on` is required: `q4_1` V refuses flash-attention off.

### 8. Smoke

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"Reply with exactly B70-FLASH-OK"}],"max_tokens":16,"temperature":0}'
```

## Context map (n=3)

Same C1 contract. 8K cannot hold a 9096-token prompt.

| Build | Context | Batch | p512 decode | p512 cold input | p9096 decode | p9096 cold input |
|---|---|---:|---:|---:|---:|---:|
| FP32 | 8K | 512 | 23.38 | 183.17 | — | — |
| FP32 | 16K | 3072 | 23.37 | 183.61 | 20.25 | 478.71 |
| FP32 | 128K | 1024 | 23.54 | 183.38 | 20.41 | 353.95 |
| F16 | 8K | 512 | 22.96 | 191.55 | — | — |
| F16 | 16K | 3072 | 23.31 | 192.64 | 20.40 | 593.98 |
| F16 | 128K | 1024 | 23.06 | 195.57 | 20.25 | 398.02 |

JSON: [`n3-context-map.json`](../../results/qwen38-flash-next-dual-b70-c1/n3-context-map.json).

## Limits

- Layer split across two cards, not vLLM tensor parallel.
- vLLM-XPU, AutoRound, and the official BF16 checkpoint are not this recipe.
- Speed here is not a quality or logit-parity result.

Catalog: [`docs/BENCHMARK-CATALOG.md`](../BENCHMARK-CATALOG.md).
Pins: [`identities.json`](../../results/qwen38-flash-next-dual-b70-c1/identities.json).
