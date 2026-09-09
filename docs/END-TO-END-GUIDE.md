# End-to-End Guide: Qwen3.8-27B GPTQ INT4 G128 on Intel Arc Pro B70 with vLLM XPU and Open WebUI

This guide documents the complete workflow used to create, validate, publish, download, and serve a fresh GPTQ INT4 quantization of **Qwen3.8-27B** on a single **Intel Arc Pro B70 32 GB**.

It also covers exposing the model through **vLLM's OpenAI-compatible API**, connecting it to **Open WebUI**, and organizing multiple vLLM model containers on a shared Docker/Portainer network.

## Public resources

**Ready-to-download model**

https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

**Source, quantization scripts, patches, benchmarks, and deployment documentation**

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

---

## 1. Final tested configuration

| Setting | Final value |
|---|---|
| Base model | Qwen3.8-27B BF16 |
| Quantization | GPTQ INT4 |
| Group size | G128 |
| Symmetric | `true` |
| `desc_act` | `false` |
| LM head | not quantized |
| MTP target tensors | excluded from GPTQ |
| Speculative decoding | MTP4 |
| MTP draft path | unquantized |
| Context tested | 161,000 tokens |
| KV cache | FP8 |
| GPU memory utilization | 0.90 |
| Maximum sequences | 1 |
| Maximum batched tokens | 8192 |
| Prefix caching | disabled |
| Deterministic temperature | 0.0 |
| Top-p | 1.0 |
| Top-k | -1 |
| Maximum generation | 4096 |

The finished G128 checkpoint contains approximately **18.22 GiB of model weights** and appears on Hugging Face as roughly **19.6 GB** for the complete repository.

---

## 2. Why create a fresh quantization?

This project had two separate goals.

The first was raw inference performance. That produced the previously published short-context MTP4 result:

```text
Median decode: 84.65 tok/s
Mean decode:   84.49 tok/s
Prompt:        512 tokens
Generation:    128 tokens
```

The second goal was long-context, source-grounded, quality-focused production use. For that, we wanted control over the BF16 source model, GPTQ calibration data, GPTQ group size, MTP handling, long-context memory use, deterministic inference, and source-fidelity testing.

The 84.65 tok/s performance test and the ~57.75 tok/s quality-workload result are **not directly comparable** because they use different workloads and benchmark procedures.

---

## 3. Starting model

The starting point was the original BF16 checkpoint:

```text
Qwen/Qwen3.8-27B
```

Local BF16 source directory used during quantization:

```bash
~/models/Qwen3.8-27B-BF16
```

The BF16 model is roughly 51+ GB, which is too large for the desired single-B70 deployment in native form. GPTQ reduced the main model weights to roughly 18–20 GB while leaving the MTP path unquantized.

---

## 4. Freeze the calibration dataset

For a meaningful G128 vs G32 comparison, both models must be quantized from the **same calibration token stream**.

The final frozen calibration set was:

```text
128 samples
1024 tokens per sample
131,072 total calibration tokens
```

Final calibration file:

```text
quantization/calibration/c4-fixed-128x1024.json
```

SHA256:

```text
ddfc570e23458c048951501231c2ff75fa175440b120045bbeb1790bea5d2599
```

Supporting scripts:

```text
quantization/build-c4-calibration.py
quantization/build-tokenized-calibration.py
```

The calibration data was used for **GPTQ calibration only**. It was not used to train or fine-tune the model.

---

## 5. Stop vLLM before quantizing

Do not leave a large vLLM workload loaded on the B70 while attempting GPTQ quantization. A running inference container can consume enough VRAM to make the quantizer fail with apparent Hessian/XPU memory errors.

Before quantization:

```bash
docker ps
```

Stop the active model container:

```bash
docker stop <vllm-container>
```

Then verify that GPU memory has been released.

---

## 6. GPTQ quantization environment

The successful quantization environment used:

```text
GPTQModel 7.3.2
Intel XPU
4-bit GPTQ
symmetric quantization
desc_act=false
```

Important settings included:

```text
bits = 4
sym = true
desc_act = false
lm_head = false
damp_percent = 0.05
damp_auto_increment = 0.01
static_groups = false
true_sequential = true
mse = 0
act_group_aware = true
device = xpu:0
```

MTP tensors were dynamically excluded from GPTQ quantization.

Final quantizer script:

```text
quantization/quantize-qwen38-gptq.py
```

---

## 7. Build the G128 model

From the repository root:

```bash
cd ~/intel-arc-pro-b70-qwen38-vllm
```

Quantize G128:

```bash
docker run --rm -i \
  --device /dev/dri \
  -v ~/models:/models \
  -v "$PWD:/work" \
  -w /work \
  --entrypoint python \
  gptqmodel-xpu-b70:7.3.2-xpu-fallback-v2fixed \
  quantization/quantize-qwen38-gptq.py \
  --group-size 128 \
  --output /models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH
```

Output directory:

```text
~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH
```

Expected files include:

```text
chat_template.jinja
config.json
generation_config.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
processor_config.json
quant_log.csv
quantize_config.json
tokenizer.json
tokenizer_config.json
```

---

## 8. G128 vs G32

A second model was created with the same procedure except `group_size = 32`.

Approximate final sizes:

```text
G128: 18.22 GB
G32:  19.54 GB
```

### G128

At `gpu_memory_utilization = 0.90` with FP8 KV cache, vLLM reported approximately:

```text
Model loading memory: 17.38 GiB
Available KV cache:   6.26 GiB
GPU KV cache size:    166,750 tokens
```

A configured context of `161000` successfully loaded.

### G32

G32 consumed approximately 18.45 GiB for the model and left only about 5.0 GiB for KV cache. vLLM estimated roughly 6.04 GiB would be required for 161K, with an estimated maximum context around 128128. Testing was therefore performed at 128000.

G32 did not provide enough source-fidelity improvement to justify the extra memory usage.

**G128 became the preferred B70 checkpoint.**

Detailed comparison:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/main/docs/G128-G32-COMPARISON.md

---

## 9. vLLM XPU serving environment

Tested serving image:

```text
vllm-xpu-b70:26.31-test
```

Tested software stack:

```text
vLLM 0.27.2rc1.dev77+gac7509e2b
vllm-xpu-kernels 0.1.12.3
Transformers 5.15.0
Torch 2.13.0+xpu
```

Environment:

```text
B70_MTP_BF16_DRAFT=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_TARGET_DEVICE=xpu
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_AFFINITY_MASK=0
PYTORCH_ALLOC_CONF=expandable_segments:True
```

Tested MTP patches:

```text
patches/patch_mtp_nightly.py
patches/patch_mtp_boundary.py
```

---

## 10. Create a shared Docker network

For Open WebUI and multiple vLLM servers, create one common Docker bridge network:

```text
llm-shared
```

### CLI

```bash
docker network create llm-shared
```

Verify:

```bash
docker network inspect llm-shared
```

### Portainer

In Portainer:

```text
Networks
→ Add network
```

Use:

```text
Name:   llm-shared
Driver: bridge
```

A custom subnet is optional unless your deployment requires one.

---

## 11. Start vLLM on the shared network

Reference container:

```text
qwen38-quality-mtp4
```

For a fresh deployment, place it on `llm-shared` at creation time:

```bash
docker rm -f qwen38-quality-mtp4 2>/dev/null || true

docker run -d \
  --name qwen38-quality-mtp4 \
  --network llm-shared \
  --entrypoint bash \
  --device /dev/dri:/dev/dri \
  -v /dev/dri:/dev/dri \
  -v ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 11444:8000 \
  vllm-xpu-b70:26.31-test \
  -lc "python /patch_mtp.py && python /patch_boundary.py && exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 161000 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --language-model-only --enable-auto-tool-choice --tool-call-parser hermes --served-model-name qwen38 --seed 0 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"
```

Host port `11444` maps to vLLM's internal port `8000`.

Host endpoint:

```text
http://HOST:11444/v1
```

Container-to-container endpoint on `llm-shared`:

```text
http://qwen38-quality-mtp4:8000/v1
```

---

## 12. Verify vLLM

Check logs:

```bash
docker logs -f qwen38-quality-mtp4
```

Expected indicators include:

```text
Using XPUwNa16LinearKernel for AutoGPTQLinearMethod
[B70] MTP draft: forcing unquantized build
Detected MTP model
Application startup complete
```

Verify the API from the host:

```bash
curl http://127.0.0.1:11444/v1/models
```

Expected served-model name:

```text
qwen38
```

The working configuration reported `max_model_len: 161000`.

---

## 13. Publish the finished model to Hugging Face

Published model:

```text
mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Authenticate:

```bash
hf auth login
hf auth whoami
```

Create the repo:

```bash
hf repos create \
  mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

If the model directory was created by a root-running container, fix ownership:

```bash
sudo chown -R "$USER:$USER" \
  ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH
```

Copy the upstream license and add a Hugging Face `README.md` model card, then upload:

```bash
HF_REPO="mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-FRESH"

hf upload \
  "$HF_REPO" \
  "$MODEL_DIR" \
  . \
  --commit-message "Upload Qwen3.8-27B GPTQ INT4 G128 checkpoint"
```

---

## 14. Download the ready-made model

Users who do **not** want to repeat quantization can download the finished checkpoint directly.

Unlike GGUF, this is not one file. It consists of five model shards plus the model index, tokenizer, GPTQ configuration, model configuration, and chat template.

Download the **entire repository**:

```bash
hf download \
  mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --local-dir ~/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Do not download only one `.safetensors` shard.

---

## 15. Open WebUI does not load the model directly

Open WebUI is the frontend.

```text
Model files
    ↓
  vLLM
    ↓
OpenAI-compatible API
    ↓
Open WebUI
```

This is analogous to a GGUF being loaded by llama.cpp or an Ollama model being loaded by Ollama. Open WebUI consumes the API exposed by the backend.

---

## 16. Put Open WebUI on `llm-shared`

Reference Open WebUI container:

```text
openwebui
```

If necessary:

```bash
docker network connect llm-shared openwebui
docker network connect llm-shared qwen38-quality-mtp4
```

If Docker returns `endpoint with name ... already exists in network llm-shared`, the container is already attached.

Verify:

```bash
docker inspect openwebui \
  --format '{{json .NetworkSettings.Networks}}'

docker inspect qwen38-quality-mtp4 \
  --format '{{json .NetworkSettings.Networks}}'
```

Both should include `llm-shared`.

---

## 17. Test vLLM from inside Open WebUI

This is the best connectivity test:

```bash
docker exec openwebui \
  curl -s http://qwen38-quality-mtp4:8000/v1/models
```

A successful response should contain:

```text
id: qwen38
owned_by: vllm
max_model_len: 161000
```

If this works, Docker DNS and the shared network are functioning correctly.

---

## 18. Configure Open WebUI

Add an OpenAI-compatible provider in Open WebUI.

Depending on version, this is generally under:

```text
Admin Settings
→ Connections
→ OpenAI-compatible connections
```

Use:

```text
Base URL:
http://qwen38-quality-mtp4:8000/v1
```

If Open WebUI requires an API key while vLLM authentication is disabled, a placeholder such as `not-needed` is sufficient.

The model should then appear as `qwen38` in the Open WebUI model selector.

---

## 19. Why container names are better than host ports

Because both containers are on `llm-shared`, Open WebUI can access `qwen38-quality-mtp4:8000` directly.

Host port `11444` remains useful for curl tests, external API clients, benchmark scripts, development, and non-Docker clients. Open WebUI itself does not need the host port when both containers share a Docker network.

---

## 20. Running multiple vLLM models

Each vLLM container can listen internally on port `8000` because every container has its own network namespace.

Example:

```text
qwen38-quality-mtp4:8000
qwen25-14b:8000
gemma-model:8000
private-large-model:8000
```

All can share `llm-shared`.

If host ports are exposed, they must be unique:

```text
qwen38-quality-mtp4  host 11444 → container 8000
qwen25-model         host 11445 → container 8000
gemma-model          host 11446 → container 8000
```

Inside Docker, Open WebUI should still use the container names:

```text
http://qwen38-quality-mtp4:8000/v1
http://qwen25-model:8000/v1
http://gemma-model:8000/v1
```

---

## 21. Portainer layout for multiple models

A clean layout:

```text
openwebui
qwen38-quality-mtp4
qwen25-vllm
gemma-vllm
private-model-vllm
```

All connected to:

```text
llm-shared
```

Network layout:

```text
llm-shared
    |
    +-- openwebui
    +-- qwen38-quality-mtp4
    +-- qwen25-vllm
    +-- gemma-vllm
    +-- private-model-vllm
```

Open WebUI becomes the single interface for multiple independently managed inference servers.

---

## 22. Multiple B70 GPUs

The same design extends naturally to multi-B70 systems.

The single-B70 test used:

```text
ZE_AFFINITY_MASK=0
```

In a multi-GPU host, separate vLLM containers can be assigned to separate Intel GPUs:

```text
GPU 0 → qwen38-vllm
GPU 1 → model-b-vllm
GPU 2 → model-c-vllm
GPU 3 → model-d-vllm
```

All model containers can remain attached to `llm-shared`.

Open WebUI only needs the API endpoint; it does not need to know which physical GPU serves the model.

---

## 23. Do not confuse network capacity with GPU capacity

Docker can connect many model containers to `llm-shared`, but a single B70 cannot necessarily hold many large models simultaneously.

For the tested G128 configuration:

```text
model memory ≈ 17.38 GiB
KV allocation ≈ several GiB
```

For large long-context models, a practical production design is often **one large model per B70**.

---

## 24. Deterministic quality profile

Final source-grounded profile:

```text
temperature = 0
top_p = 1
top_k = -1
max_tokens = 4096
```

The 4096-token ceiling was selected for robustness. Earlier 2048-token runs occasionally truncated one case.

Two back-to-back deterministic runs completed 30/30 requests with 0 errors and 0 truncations, at approximately 57.75 tok/s aggregate generation throughput for this quality workload.

---

## 25. Source-fidelity limitation

The model should not be described as perfect.

One benchmark contained a conflicting or ambiguous quantity case involving 240 versus 238 units. The model repeatedly favored the receiving log's 238-unit value rather than leaving the conflict unresolved.

This behavior persisted across G128, G32, sampled inference, deterministic inference, and multiple prompt variants, suggesting it was not simply caused by G128 quantization.

There is also a benchmark-design ambiguity: invoice quantity and receiving-log quantity may represent different facts such as units shipped versus units received. Future benchmark versions should explicitly distinguish those fields.

---

## 26. Recommended production RAG architecture

Prompt engineering alone should not be responsible for detecting every evidence conflict.

For high-reliability RAG:

```text
Documents
    ↓
Structured extraction
    ↓
Field normalization
    ↓
Conflict detection
    ↓
Evidence classification
    ↓
Grounded context
    ↓
LLM
```

The evidence layer can classify facts as `CONSISTENT`, `CONFLICTING`, `MISSING`, or `AMBIGUOUS`.

---

## 27. GitHub vs Hugging Face

### GitHub

Use GitHub for quantization scripts, calibration, Dockerfiles, vLLM patches, benchmark prompts, benchmark runners, published results, G128/G32 comparison, deployment instructions, and performance reproduction.

Repository:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

Important documents:

```text
docs/QUANTIZATION.md
docs/G128-G32-COMPARISON.md
docs/QUALITY-TESTING.md
docs/B70-QUALITY-DEPLOYMENT.md
docs/ORIGINAL-QWEN38-B70-REPLICATION-GUIDE.md
```

### Hugging Face

Use Hugging Face for users who want the finished model without rebuilding it.

Model:

https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

---

## 28. Reproduce the stack from scratch

### Path A — use the finished model

1. Install Docker and verify Intel `/dev/dri` access.
2. Download the finished checkpoint.
3. Clone the GitHub repo.
4. Create `llm-shared` if necessary.
5. Start the B70 vLLM container using `docs/B70-QUALITY-DEPLOYMENT.md`.
6. Attach Open WebUI to `llm-shared` if necessary.
7. Test container-to-container connectivity.
8. Add `http://qwen38-quality-mtp4:8000/v1` to Open WebUI.
9. Select `qwen38`.

### Path B — rebuild the quantization

```text
Download Qwen3.8-27B BF16
        ↓
Generate/freeze calibration
        ↓
Verify calibration SHA256
        ↓
Stop GPU inference workloads
        ↓
Build GPTQ XPU image
        ↓
Quantize G128
        ↓
Verify output files/config
        ↓
Run vLLM XPU
        ↓
Test long context
        ↓
Run source-fidelity benchmark
        ↓
Optionally build G32
        ↓
Compare G128/G32
```

Use the current GitHub scripts and Dockerfiles rather than old intermediate commands from discussions or social posts.

---

## 29. Final architecture

Single-model deployment:

```text
             Hugging Face
                  |
                  v
          GPTQ G128 checkpoint
                  |
                  v
        +--------------------+
        | vLLM XPU container |
        | Qwen3.8-27B        |
        | MTP4               |
        | FP8 KV             |
        | 161K context       |
        +--------------------+
                  |
                  | :8000
                  v
              llm-shared
                  |
                  v
        +--------------------+
        |    Open WebUI      |
        +--------------------+
                  |
                  v
              Browser
```

Multi-model deployment:

```text
                 Open WebUI
                     |
                 llm-shared
        _____________|_____________
       |             |             |
       v             v             v
 qwen38-vllm    model2-vllm   model3-vllm
     :8000          :8000         :8000
       |             |             |
      B70           B70           B70
```

---

## 30. Main lessons

1. **G128 was the better B70 tradeoff.** G32 increased memory use without a meaningful quality improvement in the tested workload.
2. **Quantization and inference workloads must not compete for VRAM.** Stop vLLM before GPTQ quantization.
3. **MTP should be treated separately from the main GPTQ weights.** Keeping the MTP path unquantized worked with the final MTP4 setup.
4. **Long context is primarily a memory-allocation problem.** G128 left enough room for FP8 KV to reach the tested 161K configuration.
5. **Throughput numbers need workload context.** 84.65 tok/s and ~57.75 tok/s measure different workloads.
6. **Open WebUI only needs a compatible API endpoint.** The model storage format is a backend concern.
7. **One shared Docker network simplifies multi-model deployments.** `llm-shared` lets Open WebUI reach every vLLM server by container name.
8. **Hugging Face and GitHub serve different purposes.** Hugging Face distributes finished weights; GitHub provides reproducibility.
9. **Source-grounded systems still benefit from deterministic evidence handling.** A strong LLM should complement structured conflict detection rather than replace it.

---

## 31. Credits

Original model: **Qwen/Qwen3.8-27B**

Quantization: **GPTQModel 7.3.2**

Serving: **vLLM XPU**

This reproduction and quantization work builds substantially on the Intel Arc Pro B70 work published by **SergiioB**:

https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

Fresh G128/G32 quantization work, controlled calibration, long-context validation, source-fidelity testing, Hugging Face packaging, and deployment documentation: **Mike Caldera**

Project repository:

https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm

Model repository:

https://huggingface.co/mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

---

## 32. Final recommended configuration

```text
Qwen3.8-27B
GPTQ INT4
G128
symmetric
desc_act=false
MTP weights unquantized
MTP4
FP8 KV
161000 context
gpu_memory_utilization=0.90
max_num_seqs=1
max_num_batched_tokens=8192
prefix caching disabled
temperature=0
top_p=1
top_k=-1
max_tokens=4096
```

> **One of the best quality-focused configurations validated during this Intel Arc Pro B70 project.**

It is not claimed to be universally optimal. Future vLLM, Intel XPU, GPTQModel, kernel, or Qwen updates may improve on it.
