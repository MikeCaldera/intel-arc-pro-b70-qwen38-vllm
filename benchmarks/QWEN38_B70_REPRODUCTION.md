# Intel Arc Pro B70 — Qwen3.8-27B vLLM XPU Reproduction

This document describes the configuration, benchmark methodology, and measured results from testing Qwen3.8-27B on a single Intel Arc Pro B70 32 GB GPU with vLLM XPU.

## Reproduction overview

The reproduction process is:

1. Install a working Intel GPU driver/runtime for the Arc Pro B70.
2. Obtain the exact Qwen3.8-27B GPTQ INT4 + BF16 MTP model revision used here.
3. Build the pinned vLLM XPU Docker image with the Intel v26.31 userspace runtime.
4. Launch the vLLM server using the documented configuration.
5. Generate exact-token prompts with `b70-generate-exact-prompts.py`.
6. Run the benchmark with `b70-realworld-context-harness.py`.
7. Check `dmesg` for Xe GPU faults after the run.

## Hardware

* GPU: Intel Arc Pro B70 32 GB
* Host: Dell XPS 8940
* CPU: Intel Core i7-11700
* OS: Ubuntu 26.04 LTS
* Kernel: 7.0.0-30-generic
* GPU driver: `xe`
* PCI ID: `8086:e223`

## Model

* Model: Qwen3.8-27B
* Target weights: GPTQ INT4
* Quantization: symmetric, group size 128
* MTP draft weights: BF16
* KV cache: FP8
* Single GPU
* vLLM XPU

Local model path used during testing:

```text
/home/mikecaldera/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Replace this path with your own model location when reproducing the test.

## Model source

Model repository:

```text
SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Pinned model revision:

```text
9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e
```

For the closest reproduction, use this exact model revision rather than silently substituting a newer revision.

## Software

vLLM:

```text
0.27.2rc1.dev77+gac7509e2b.xpu
```

vLLM XPU kernels:

```text
0.1.12.3
```

Intel compute runtime used for the successful v26.31 tests:

```text
26.31.39395.13-0
```

Intel Graphics Compiler:

```text
2.40.13
```

libigdgmm:

```text
22.10.0
```

## Intel v26.31 runtime packages

The controlled v26.31 test image used these packages:

```text
intel-igc-core-2_2.40.13+22418_amd64.deb
intel-igc-opencl-2_2.40.13+22418_amd64.deb
intel-ocloc_26.31.39395.13-0_amd64.deb
intel-opencl-icd_26.31.39395.13-0_amd64.deb
libigdgmm12_22.10.0_amd64.deb
libze-intel-gpu1_26.31.39395.13-0_amd64.deb
```

The `.deb` files themselves are not included in this repository.

Obtain the matching Intel GPU userspace runtime packages from Intel's official compute-runtime package source before building the derivative image.

Verify that the package versions match those listed above.

### Updating the Intel userspace runtime

The v26.31 retest updated the Intel GPU **userspace runtime inside the derivative vLLM Docker image**.

The host continued using the Linux `xe` kernel driver. The host kernel driver was not replaced as part of this runtime comparison.

Place the six v26.31 `.deb` packages listed above in:

```text
intel-runtime-26.31/
```

Then build the derivative image:

```bash
docker build \
  -f Dockerfile.vllm-xpu-26.31 \
  -t vllm-xpu-b70:26.31-test \
  .
```

This keeps the Intel userspace-runtime change isolated inside the test container and makes the v26.27 versus v26.31 comparison easier to reproduce.

The primary controlled change in this comparison was the Intel GPU userspace runtime inside the container. The Intel Arc Pro B70 hardware and host Linux `xe` kernel driver were unchanged.

## Docker image

Pinned base image:

```text
vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
```

A derivative image was built from this pinned image with the Intel v26.31 userspace runtime packages.

Dockerfile included in this repository:

```text
Dockerfile.vllm-xpu-26.31
```

## MTP configuration

MTP2 speculative decoding:

```json
{"method":"mtp","num_speculative_tokens":2}
```

Important environment variables:

```bash
VLLM_XPU_ENABLE_XPU_GRAPH=1
PYTORCH_ALLOC_CONF=expandable_segments:True
VLLM_TARGET_DEVICE=xpu
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_AFFINITY_MASK=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
B70_MTP_BF16_DRAFT=1
```

Prefix caching was disabled during the cold-context benchmark runs.

Chunked prefill:

```text
max_num_batched_tokens=8192
```

## vLLM server configuration

The 160K MTP2 v26.31 retest used these important settings:

```text
XPU Graph enabled
max_model_len=161000
gpu_memory_utilization=0.89
max_num_seqs=1
max_num_batched_tokens=8192
kv_cache_dtype=fp8
prefix caching disabled
language-model-only
MTP2
```

Equivalent vLLM arguments include:

```text
--max-model-len 161000
--gpu-memory-utilization 0.89
--max-num-seqs 1
--max-num-batched-tokens 8192
--kv-cache-dtype fp8
--disable-prefix-caching
--language-model-only
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

The model was served as:

```text
qwen38
```

For strict reproduction, keep the model, runtime versions, patches, XPU graph setting, KV-cache type, speculative-token count, and memory configuration unchanged while comparing results.

### Exact 160K MTP2 container launch

The following command reproduces the graph-enabled v26.31 160K MTP2 server configuration used for the successful three-request retest.

Run from the repository root:

```bash
cd ~/intel-arc-pro-b70-qwen38-vllm

export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
export B70_RENDER_NODE="/dev/dri/renderD129"
export RENDER_GID="$(stat -c '%g' "$B70_RENDER_NODE")"

docker rm -f qw38context200k-mtp2-2631 2>/dev/null || true

docker run -d \
  --name qw38context200k-mtp2-2631 \
  --restart unless-stopped \
  -p 11442:8000 \
  --device /dev/dri:/dev/dri \
  -v /dev/dri:/dev/dri:ro \
  --group-add "$RENDER_GID" \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash \
  vllm-xpu-b70:26.31-test \
  -lc 'set -e
python /patch_mtp.py
python /patch_boundary.py
exec vllm serve /model \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 161000 \
  --gpu-memory-utilization 0.89 \
  --kv-cache-dtype fp8 \
  --port 8000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --served-model-name qwen38 \
  --language-model-only \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"'
```

The host port is `11442`, so the OpenAI-compatible endpoint is:

```text
http://127.0.0.1:11442/v1
```

Follow startup logs with:

```bash
docker logs -f qw38context200k-mtp2-2631
```

Do not begin the benchmark until the log reports:

```text
Application startup complete.
```

Verify the endpoint:

```bash
curl -fsS http://127.0.0.1:11442/health && echo "HEALTH OK"

curl -fsS http://127.0.0.1:11442/v1/models
```

The served model should appear as:

```text
qwen38
```

`/dev/dri/renderD129` and `ZE_AFFINITY_MASK=0` matched the validation host. Device numbering can differ on another system, so identify the B70 render node and Level Zero device before copying those values unchanged.


## Benchmark tools

The following scripts are included in this repository:

```text
benchmarks/b70-generate-exact-prompts.py
benchmarks/b70-realworld-context-harness.py
```

The exact-context harness uses:

* exact rendered prompt-token counts
* unique cold prompts for the full benchmark
* prefix caching disabled
* streaming
* `temperature=0`
* `ignore_eos`
* 128 generated tokens
* one generic warmup
* one target-shape warmup
* six measured unique prompts for full benchmark runs

For six measured repetitions, generate seven prompts:

```text
--per-target 7
```

One prompt is consumed by the target-shape warmup.

## Throughput definition

Unless otherwise noted, `tok/s` in this document refers to measured **output/decode tokens per second**.

It does not refer to prompt-prefill throughput, SSE chunks per second, or aggregate multi-request server throughput.

---

# Results

## Short-context MTP4 validated result

The original short-context validation benchmark produced:

```text
Median decode: 84.65 tok/s
Mean decode:   84.49 tok/s
Min / Max:     83.73 / 84.78 tok/s
Median TTFT:   0.3557 s
Prompt:        512 tokens
Generation:    128 tokens
Measured runs: 5 after same-shape warmup
```

This was measured on one Intel Arc Pro B70 using Qwen3.8-27B GPTQ INT4 with BF16 MTP4.

The **84.65 tok/s** figure is the median result from the original short-context `p512/g128` benchmark.

It is not directly comparable to the cold exact-context long-context harness results below and should not be interpreted as 84.65 tok/s at long context.

## MTP4 context scaling

Measured context-scaling results:

| Prompt tokens | Decode throughput |
| ------------: | ----------------: |
|           512 |       81.24 tok/s |
|         8,192 |       73.30 tok/s |
|        16,384 |       74.76 tok/s |
|        32,768 |       68.34 tok/s |
|        65,536 |       66.11 tok/s |
|       120,000 |       50.31 tok/s |

These context-scaling measurements used the exact-context benchmark methodology and should be treated separately from the original 84.65 tok/s short-context validation benchmark.

## 80K MTP2 cold-context benchmark

Configuration:

```text
Intel compute-runtime: v26.31 / 26.31.39395.13
MTP2
FP8 KV
XPU Graph enabled
80,000 exact prompt tokens
128 output tokens
prefix caching disabled
```

Startup provided enough KV capacity for the requested context.

Result:

```text
Shape warmup: PASS

Rep 1: PASS
Rep 2: PASS
Rep 3: PASS
Rep 4: PASS
Rep 5: PASS
Rep 6: PASS
```

Typical performance:

```text
Time to first token (TTFT): ~81.5 seconds
Decode: ~51.5 tok/s
MTP acceptance: ~96.6%
```

Five of the six measured runs clustered near approximately 51.5 tok/s decode.

One measured repetition had lower speculative acceptance and correspondingly lower decode throughput, but all six requests completed successfully.

## No-speculative-decoding long-context validation

The target model completed six measured runs at each of these prompt lengths without speculative decoding:

```text
160K
180K
195K
```

This demonstrated successful no-speculative-decoding inference at 160K, 180K, and 195K prompt lengths on the tested single-B70 configuration.

These tests are important because they show that the earlier long-context Xe failure was not simply caused by the target model being unable to operate at those context lengths.

## Intel compute-runtime v26.27 — 160K MTP2 failure

The original 160K MTP2 test under Intel compute-runtime v26.27 behaved as follows:

```text
Rep 1: PASS
Rep 2: PASS
Rep 3: Xe GPU failure
```

Observed kernel symptoms included:

```text
xe CCS engine reset
Timed out VLLM::EngineCor job
BCS fault
Xe coredump
Faulted Address: 0x0000d556aa4c7000
```

MTP1 at 160K completed successfully.

MTP2 and MTP4 triggered the Xe failure condition during the earlier v26.27 runtime testing.

The observed stability boundary in that campaign was therefore:

```text
160K no speculative decoding: PASS
160K MTP1:                    PASS
160K MTP2:                    Xe failure
160K MTP4:                    Xe failure
```

## Intel compute-runtime v26.31 — 160K MTP2 retest

Configuration:

```text
Intel compute-runtime: 26.31.39395.13
XPU Graph enabled
max_model_len=161000
gpu_memory_utilization=0.89
max_num_seqs=1
max_num_batched_tokens=8192
FP8 KV
MTP2
prefix caching disabled
```

Startup reported:

```text
Available KV cache memory: 5.93 GiB
GPU KV cache size: 166,854 tokens
Maximum concurrency for 161,000 tokens per request: 1.04x
Application startup complete
```

Three consecutive requests were sent using the same exact 160,000-token prompt.

Prefix caching was disabled, so each request required full prompt processing.

The earlier v26.27 harness used unique cold prompts, so this v26.31 retest closely matches the context length and MTP configuration but is not identical in prompt content.

Each request used:

```text
160,000 prompt tokens
128 completion tokens
160,128 total tokens
```

Results:

```text
Request 1: PASS
HTTP 200
Elapsed: 246.50 s

Request 2: PASS
HTTP 200
Elapsed: 245.49 s

Request 3: PASS
HTTP 200
Elapsed: 244.59 s
```

All three requests completed normally.

The Xe CCS / BCS failure seen under runtime v26.27 was **not reproduced** during this closely matched three-request 160K MTP2 retest using runtime v26.31.

This should not be interpreted as proof that the underlying issue is fixed in every possible configuration.

It means that the earlier destructive Xe failure did not reproduce during this v26.31 retest.

## Generate exact 80K prompts

From the repository root, generate seven exact 80K prompts:

```bash
docker run --rm \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  -v "$PWD:/work" \
  -v /path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  --entrypoint python \
  vllm-xpu-b70:26.31-test \
  /work/benchmarks/b70-generate-exact-prompts.py \
  --model /model \
  --output /work/benchmarks/qwen38-context-prompts-80000.json \
  --targets 80000 \
  --per-target 7
```

Replace:

```text
/path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

with the actual model directory on your system.

## Run the 80K benchmark

Example:

```bash
python3 benchmarks/b70-realworld-context-harness.py \
  --mode context \
  --prompts benchmarks/qwen38-context-prompts-80000.json \
  --outdir benchmarks/qwen38-context-80000-mtp2-2631-results \
  --model qwen38 \
  --budget 82000 \
  --reps 6 \
  --target 80000 \
  --output 128 \
  --root http://127.0.0.1:11442 \
  --ignore-eos
```

The benchmark endpoint above assumes the vLLM container is exposed on host port `11442`.

If you use a different host port, update `--root` accordingly.

## Check for Xe faults

After long-context testing, check the kernel log:

```bash
sudo dmesg -T | grep -Ei \
'engine reset|timed.?out|CCS.*timeout|BCS.*fault|coredump|faulted address|page fault|xe.*fault|xe.*reset'
```

A clean successful run should not produce the Xe CCS/BCS failure pattern documented in the v26.27 failure section.

## Patches

The tested MTP setup used:

```text
patches/patch_mtp_nightly.py
patches/patch_mtp_boundary.py
```

These files are included in this repository so the tested configuration can be inspected and reproduced.

## Important reproducibility notes

When comparing results, record at least:

* Linux distribution
* kernel version
* Intel `xe` driver status
* B70 PCI address
* negotiated PCIe generation and width
* B70 render node
* vLLM image digest
* vLLM version
* XPU kernel version
* Intel compute-runtime version
* Intel IGC version
* exact model revision
* patch versions or hashes
* vLLM command-line arguments
* relevant environment variables
* GPU power limit
* prompt-token count
* output-token count
* MTP speculative-token count
* prefix-cache setting
* XPU Graph setting
* warmup procedure
* number of measured repetitions

Changing any of these may make a result useful, but it also makes it a different benchmark configuration.

## Related issue

The original XPU/Xe failure investigation is tracked here:

[vllm-project/vllm#55425](https://github.com/vllm-project/vllm/issues/55425)

## Attribution

This work builds on the Intel Arc Pro B70 inference cookbook published by SergiioB:

[SergiioB / intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)

The benchmark extensions, Qwen3.8 testing, long-context measurements, Intel runtime comparison, and community reproduction documentation here were performed as follow-up testing on a single Intel Arc Pro B70 system.

