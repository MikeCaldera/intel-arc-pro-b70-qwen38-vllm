# Full vLLM XPU Setup and Matrix Commands

These commands reproduce the 2026-08-09 phase-separated C1 matrix. They do not change the host GPU power cap. `CONFIGURED_CAP_W` records the operator-selected setting in the manifest; it does not set that value.

## 1. Clone and set paths

```bash
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git "$HOME/intel-arc-pro-b70-inference-cookbook"
cd "$HOME/intel-arc-pro-b70-inference-cookbook"
export COOKBOOK="$PWD"
export MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
export MODEL_ID='llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4'
export IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
```

If the repository already exists, enter it and update it separately. Do not run the clone command over an existing working tree.

## 2. Verify Docker and the render device

```bash
docker version
stat /dev/dri/render*
export RENDER_GID="$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')"
printf 'render_gid=%s\n' "$RENDER_GID"
```

The Docker user must be able to access the render node. Confirm the device from the pinned image:

```bash
docker run --rm \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  --entrypoint python "$IMAGE" \
  -c 'import torch; print(torch.xpu.device_count(), torch.xpu.get_device_name(0)); assert torch.xpu.device_count() == 1'
```

Expected device name: `Intel Arc Pro B70`.

## 3. Pull the immutable image and check exact packages

```bash
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

Check the versions inside that digest, not the host Python environment:

```bash
docker run --rm --entrypoint python "$IMAGE" -c '
from importlib.metadata import version
vllm = version("vllm")
kernels = version("vllm-xpu-kernels")
print("vllm=" + vllm)
print("vllm-xpu-kernels=" + kernels)
assert vllm == "0.26.1rc1.dev457+gc810e5ee9.xpu"
assert kernels == "0.1.12"
'
```

The tested versions are vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu` and `vllm-xpu-kernels 0.1.12`. PyPI `vllm-xpu-kernels 0.1.12.2` is newer but was not tested in this campaign.

## 4. Download and verify the preserved-MTP model

The host does not need the Hugging Face CLI:

```bash
mkdir -p "$MODEL_DIR"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf \
  -v "$MODEL_DIR:/model" \
  python:3.12-slim \
  sh -lc 'pip install --no-cache-dir "huggingface_hub[cli]" && hf download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4 --local-dir /model'
```

Verify the configuration, all Safetensors shards, and the preserved `mtp.*` tensors:

```bash
docker run --rm \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint python "$IMAGE" -c '
import glob, json
from safetensors import safe_open
config = json.load(open("/model/config.json"))
shards = sorted(glob.glob("/model/*.safetensors"))
assert config.get("mtp_num_hidden_layers") == 1, config.get("mtp_num_hidden_layers")
assert shards, "no safetensors shards"
mtp = 0
for path in shards:
    with safe_open(path, framework="pt", device="cpu") as shard:
        mtp += sum(key.startswith("mtp.") for key in shard.keys())
print("shards=", len(shards), "mtp_tensor_count=", mtp)
assert mtp > 0
'
sha256sum "$MODEL_DIR/config.json" "$MODEL_DIR"/*.safetensors | tee "$MODEL_DIR/SHA256SUMS.local"
```

The standard GPTQ-INT4 checkpoint declares an MTP layer but does not include the preserved draft tensors required here.

## 5. Verify patch and prompt hashes

```bash
cd "$COOKBOOK"
printf '%s  %s\n' \
  '4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14' 'patches/patch_mtp_nightly.py' \
  '41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50' 'patches/patch_mtp_boundary.py' \
  '1c4c8bba350db23ef64d166d2260a3a747d565f1ec682fde6b0e93224bc1dfb9' 'benchmarks/benchmark-system-prompt.txt' \
  '49eadcaef1b05b5ca376673c4b0be6b004e72a0fc4e48c050c781e10c90a4339' 'benchmarks/pi-system-prompt.txt' \
  | sha256sum --check
```

Patch order is fixed:

1. `patch_mtp_nightly.py` builds the preserved BF16 MTP draft outside the GPTQ target quantization config.
2. `patch_mtp_boundary.py` handles the partial final speculative group at the exact 131,072-token boundary.

Do not apply the historical vLLM 0.21 patches to this image.

## 6. Launch one tested server

Both launchers include the **tool-calling flags**
(`--enable-auto-tool-choice --tool-call-parser qwen3_coder`), required for Pi,
omp, and other agent clients that send `tool_choice: "auto"`. Without them
those clients fail with
`400: "auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`.

This one launch uses MTP2, cache on, context 131,072, scheduler 8,192, `gpu-memory-utilization=0.85`, and port 8000:

```bash
cd "$COOKBOOK"
bash benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
```

Follow startup and verify patch application and configuration:

```bash
docker logs -f b70-mtp2-cache-on
```

The log must show both patch messages and these effective fields: `max_model_len: 131072`, `max_num_batched_tokens: 8192`, `max_num_seqs: 64`, `enable_prefix_caching: True`, `enable_auto_tool_choice: True`, `tool_call_parser: qwen3_coder`, and `num_speculative_tokens: 2`.

### 6b. Dense 27B launch (same image and patches, fp8 KV)

The dense Qwen3.6-27B track (Run 29/31) uses the same pinned image and the same
two patches, plus `--kv-cache-dtype fp8` (REQUIRED: dense fp16 KV needs 9.5 GiB
at 128K and does not fit) and `gpu-memory-utilization=0.88` for MTP4 (0.90 fills
the card with MTP4 spec buffers; no-spec/MTP1/MTP2 run at 0.90).

```bash
export DENSE_DIR="$HOME/models/Qwen3.6-27B-MTP-Preserved-GPTQ-Int4"
bash benchmarks/qwen36-27/launch-dense27-128k-mode.sh "$DENSE_DIR" mtp4 on 8000
```

Follow startup and verify patch application and configuration:

```bash
docker logs -f b70-dense-mtp4-cache-on
```

The log must show both patch messages and these effective fields:
`max_model_len: 131072`, `max_num_batched_tokens: 8192`, `max_num_seqs: 64`,
`enable_prefix_caching: True`, `kv_cache_dtype: fp8`, and
`num_speculative_tokens: 4`. If `kv_cache_dtype` is not fp8, dense 128K will
fail with a "KV cache is larger than available" error — never launch dense
128K without `--kv-cache-dtype fp8`.

## 7. Check the endpoint

```bash
curl -f http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics \
  | grep -E 'vllm:prefix_cache_(hits|queries)|vllm:spec_decode_'
```

Stop the tested server before the matrix:

```bash
docker rm -f b70-mtp2-cache-on
```

## 8. Run the full phase-separated matrix

First confirm there is no other inference engine or container. The runner performs its own empty-GPU and VRAM gates.

```bash
cd "$COOKBOOK"
CONFIGURED_CAP_W=165 \
  bash benchmarks/b70-pi-prefill-decode-matrix.sh "$MODEL_DIR"
```

`CONFIGURED_CAP_W=165` only writes the operator's declared cap into evidence. The script never writes `power1_cap` or changes host power policy.

The matrix is:

- cold input: p512, p2048, p4096, p6144, p8192, and p131071, each g1;
- decode: p512 and p8192 at g32, g128, g256, and g512;
- historical control: p9445/g128;
- full-context decode: p130944/g128 and p130560/g512;
- modes: no-spec, MTP1, MTP2, and MTP4;
- one full-output same-shape warmup, then five C1 measured requests per cell;
- exact-output decode with `ignore_eos=true`;
- cache on, entropy-first unique cold prefixes, and required zero cache-hit delta.

Standard p512 through p8192 cells use `benchmark-system-prompt.txt`. The p9445 control and full-context cells use `pi-system-prompt.txt`.

![How the exact-token phase-separated campaign is measured](assets/b70-benchmark-method.svg)

The visual is generated from the same evidence contract. Agents regenerating it should use `.agentic/skills/b70-benchmark-visuals/SKILL.md` or the matching public renderer `benchmarks/render-prefill-decode-svg.py`.

## 9. Compile, audit, and render

The runner prints its result directory at completion. Set it here:

```bash
export RUN_ROOT="$COOKBOOK/results/vllm-pi-prefill-decode-matrix-<UTC>-<PID>"
python3 benchmarks/b70-compile-prefill-decode-matrix.py "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json"
python3 benchmarks/render-prefill-decode-tables.py "$RUN_ROOT/summary.json" \
  > "$RUN_ROOT/tables.md"
python3 benchmarks/render-prefill-decode-svg.py "$RUN_ROOT/summary.json" \
  --dashboard docs/assets/b70-prefill-decode-dashboard.svg \
  --method docs/assets/b70-benchmark-method.svg
```

The compiler fails closed on incomplete modes, server errors, cache-state errors, prompt mismatch, wrong token counts, and missing forced-exact replacement evidence. Do not publish directly from raw request files.

## 10. Evidence layout

```text
results/vllm-pi-prefill-decode-matrix-<UTC>-<PID>/
  manifest.txt
  package-versions.txt
  prompts/
  no-spec/
  mtp1/
  mtp2/
  mtp4/
  summary.json
  tables.md
```

Each mode retains server logs, metrics snapshots, synchronized monitor data, VRAM evidence, harness logs, raw SSE/request records, and per-cell manifests. Preserve failed and excluded cells. The original no-spec p130560/g512 early-EOS cell remains excluded; the accepted summary uses its forced-exact replacement.

Follow [Benchmark format](BENCHMARK-FORMAT.md) before copying any table.

## 11. Qwen3.8-27B champion stack (`f01e24f6`)

This family is **not** the Qwen3.6 Pi digest above. Use this image, this
checkpoint, and only the Qwen3.8 Apply list from
[IMAGE-AND-PATCH-MATRIX.md](IMAGE-AND-PATCH-MATRIX.md).

```bash
export COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
export IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
cd "$COOKBOOK"

docker pull "$IMAGE"
docker run --rm --entrypoint python "$IMAGE" -c '
from importlib.metadata import version
print(version("vllm"))
print(version("vllm-xpu-kernels"))
assert version("vllm").startswith("0.27.2rc1.dev77")
assert version("vllm-xpu-kernels") == "0.1.12.3"
'

printf '%s  %s\n' \
  '4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14' 'patches/patch_mtp_nightly.py' \
  '41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50' 'patches/patch_mtp_boundary.py' \
  '8e4a3cbe5f424f308af74ff215d0fcb8d31f63ac3f07cf359ed2269956c3fc80' 'patches/patch_gdn_mixed_split_v5.py' \
  'ffae41926d5f05f4f38bb985301b5e572092441d06d6063c8820a63a39b8cefc' 'patches/patch_draft_lmhead_int4.py' \
  '4df179c3e77fd7a248f9b9c0b60217c60caea14ebfd16b7860536fbff3b2a1e9' 'patches/patch_draft_mtp_int4.py' \
  | sha256sum --check
```

Download the preserved-MTP GPTQ artifact
`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (revision `9d189a60…`)
into `$MODEL_DIR`. `mtp.*` tensors must stay BF16.

### Default recipe — BF16 draft MTP4 (cookbook card 83.7)

Required patches only: `patch_mtp_nightly.py` then `patch_mtp_boundary.py`.

```bash
RENDER_GID="$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')"
docker rm -f qw38speed >/dev/null 2>&1 || true
docker run -d --name qw38speed -p 8000:8000 --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro -v "$MODEL_DIR:/model:ro" \
  -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131072 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --language-model-only --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
```

### Optional mixed-batch overlay (v5) — correctness, not C1 speed

Add after the two MTP patches. C1 decode is speed-flat. Mixed spec + prefill
stays alive.

```bash
# extra mounts:
#   -v "$COOKBOOK/patches/patch_gdn_mixed_split_v5.py:/patch_v5.py:ro"
# extra apply, after the two MTP patches:
#   python /patch_v5.py
```

### Optional draft-INT4 overlay (S+M1) — MTP speed, quality still gated

Runtime RTN of the **draft** LM head and five MTP linears only. Target verify
stays BF16. Set **both** env flags. Pair with v5 if mixed batches can occur.

```bash
# extra mounts (in addition to the two MTP patches + v5):
#   -v "$COOKBOOK/patches/patch_draft_lmhead_int4.py:/patch_s.py:ro"
#   -v "$COOKBOOK/patches/patch_draft_mtp_int4.py:/patch_m1.py:ro"
# extra env:
#   -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1
# extra apply:
#   python /patch_s.py; python /patch_m1.py
```

Matched same-image n=5 (not vs Run 40 83.7): p512/g128 **81.20 → 112.65**,
p8192/g128 **77.52 → 103.63**, short agentic cache-off **+32.8%**, p8192/g1
cold input flat. Accept 95.86% → 94.44%. Quality A/B (temp=0, 15 tasks):
both arms **12/15**, replay 15/15, **zero C-only regressions** — keep
optional. Prefix-on agentic (isolated C1, 816–871 MiB after load): short
**43.81 → 58.86**, 8K **48.04 → 66.99**, 16K **54.40 → 65.92**, 32K
**46.34 → 61.02**, 64K **44.83 → 56.20**. 96K thin; ~128K t1 **43.82 vs
37.48** (n=1, S+M1 slower). Do not mix cache-on and cache-off agentic in
one headline. Same-arm generation curve + isolated C1 synthetic 128K (n=5,
870 MiB after load): p512/g256 **110.76**, p8192/g512 **64.96**,
p130944/g128 **62.52** (5/5 exact 131,072 tokens). Isolated C1 only — not
a serving headline. Tables:
[DRAFT-INT4-S-M1.md](qwen38-27/DRAFT-INT4-S-M1.md).
Do **not** apply Nemotron grouped-topk / SSU, and do **not** apply the original
full-buffer GDN split.

DFlash 2 (`incoai/Qwen3.8-27B-DFlash2`) is **not** a recipe on this image.
A research overlay of open vLLM PR #52816 + v5 **did load** on 2026-08-19
(`/health` 200) but spec acceptance was **0/574**. One-shot post-first
19.18 tok/s is not a median and is slower than MTP4. Keep MTP4. Details:
[QWEN38-VLLM-XPU.md §13](qwen38-27/QWEN38-VLLM-XPU.md).
