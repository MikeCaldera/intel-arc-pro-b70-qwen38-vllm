# Full vLLM XPU Setup and Commands

This is the complete public setup used for the matched 128K cache/spec campaign. It starts from a host where Docker can access the Intel render device. It does not change the GPU power cap.

## 1. Set paths

```bash
export COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
export MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
export IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
```

Clone or update the cookbook:

```bash
if [ -d "$COOKBOOK/.git" ]; then
  git -C "$COOKBOOK" pull --ff-only
else
  git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git "$COOKBOOK"
fi
cd "$COOKBOOK"
```

## 2. Verify Docker and the Intel render device

```bash
docker version
stat /dev/dri/render*
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')
printf 'render_gid=%s\n' "$RENDER_GID"
```

The user running Docker must be able to access the render device. The launcher passes its group ID to the container with `--group-add`.

## 3. Pull the exact public image

```bash
sudo docker pull "$IMAGE"
sudo docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

Observed versions in this digest:

```text
vLLM v0.26.1rc1.dev457+gc810e5ee9
vllm-xpu-kernels 0.1.12
```

## 4. Download the preserved-MTP checkpoint

This command uses a temporary Python container, so the host does not need `huggingface-cli`:

```bash
mkdir -p "$MODEL_DIR"
sudo docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf \
  -v "$MODEL_DIR:/model" \
  python:3.12-slim \
  sh -lc 'pip install --no-cache-dir "huggingface_hub[cli]" && hf download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4 --local-dir /model'
```

Verify that the checkpoint declares one MTP layer and contains `mtp.*` tensors:

```bash
sudo docker run --rm \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint python "$IMAGE" -c '
import glob, json
from safetensors import safe_open
config = json.load(open("/model/config.json"))
print("mtp_num_hidden_layers=", config.get("mtp_num_hidden_layers"))
keys = []
for path in glob.glob("/model/*.safetensors"):
    with safe_open(path, framework="pt", device="cpu") as shard:
        keys.extend(key for key in shard.keys() if key.startswith("mtp."))
print("mtp_tensor_count=", len(keys))
assert config.get("mtp_num_hidden_layers") == 1
assert keys
'
```

The official GPTQ-INT4 checkpoint declares an MTP layer but does not carry the preserved draft tensors required by this recipe.

## 5. Verify the two current patches

```bash
cd "$COOKBOOK"
sha256sum \
  patches/patch_mtp_nightly.py \
  patches/patch_mtp_boundary.py
python3 -m py_compile \
  patches/patch_mtp_nightly.py \
  patches/patch_mtp_boundary.py
```

Patch order is fixed:

1. `patch_mtp_nightly.py` builds the preserved BF16 MTP draft outside the GPTQ target quantization config.
2. `patch_mtp_boundary.py` handles a shortened final speculative group at the exact 131,072-token boundary.

Do not add the historical vLLM 0.21 patches. They target different source files and a local image that was never published.

## 6. Launch one tested mode

The launcher contains the complete Docker command, patch mounts, patch order, environment variables, and vLLM flags.

```bash
cd "$COOKBOOK"
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
```

Follow startup and verify both patches ran:

```bash
sudo docker logs -f b70-mtp2-cache-on
```

Expected patch lines:

```text
patched .../qwen3_5_mtp.py
patched .../gdn_attn.py
```

Expected server fields include:

```text
max_model_len: 131072
max_num_batched_tokens: 8192
max_num_seqs: 64
enable_prefix_caching: True
num_speculative_tokens: 2
```

Check the endpoint:

```bash
curl -f http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics | grep -E 'prefix_cache|spec_decode'
```

Stop that server:

```bash
sudo docker rm -f b70-mtp2-cache-on
```

## 7. All eight launch commands

Each command starts one server. Stop it before starting the next command.

### No speculative decoding

```bash
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" no-spec on 8000
sudo docker rm -f b70-no-spec-cache-on

bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" no-spec off 8000
sudo docker rm -f b70-no-spec-cache-off
```

### MTP1

```bash
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp1 on 8000
sudo docker rm -f b70-mtp1-cache-on

bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp1 off 8000
sudo docker rm -f b70-mtp1-cache-off
```

### MTP2

```bash
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
sudo docker rm -f b70-mtp2-cache-on

bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 off 8000
sudo docker rm -f b70-mtp2-cache-off
```

### MTP4

```bash
bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp4 on 8000
sudo docker rm -f b70-mtp4-cache-on

bash benchmarks/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp4 off 8000
sudo docker rm -f b70-mtp4-cache-off
```

`--no-enable-prefix-caching` is required for every cache-off cell. vLLM V1 in the pinned image defaults prefix caching to on. Omitting the negative flag silently produces cache-on behavior.

## 8. Run the complete matched campaign

This command runs all eight cells and writes separate cold and resident-session evidence for each one:

```bash
cd "$COOKBOOK"
bash benchmarks/b70-pi-128k-cache-spec-matched.sh "$MODEL_DIR"
```

The campaign records:

- no-spec, MTP1, MTP2, and MTP4;
- cache enabled and explicitly disabled;
- five exact p130944/g128 cold requests per cell;
- one exact 120,000-token session preparation per cell;
- five changed resident-session g128 follow-ups per cell;
- raw SSE timestamps, endpoint token counts, cache counters, MTP counters, server logs, VRAM, and named-temperature/energy telemetry.

Results appear under:

```text
results/vllm-pi-128k-cache-spec-matched-<UTC>-<PID>/
```

## 9. Read the current measured result

```bash
python3 - <<'PY'
import json
path = "results/cache-spec-matrix-20260808-summary.json"
data = json.load(open(path))
for row in data["rows"]:
    cold = row["cold"]
    warm = row["resident"]
    print(
        row["mode"], row["cache"],
        f"cold_ttfc={cold['ttfc_median_s']:.3f}s",
        f"cold_post_first={cold['client_post_first_tps_median']:.2f}tok/s",
        f"resident_ttfc={warm['ttfc_median_s']:.3f}s",
        f"resident_e2e={warm['e2e_median_s']:.3f}s",
        f"reused={warm['reused_tokens_median']:.0f}",
    )
PY
```

The published result is E2 self-reported evidence. It has raw internal evidence and a reproducible public command path, but no independent reproduction yet.
