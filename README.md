# Intel Arc Pro B70 + Qwen3.8-27B --- vLLM XPU Replication Guide

A reproducible community guide for running **Qwen3.8-27B GPTQ INT4 with
BF16 MTP4 speculative decoding** on a single **Intel Arc Pro B70 32 GB**
GPU using vLLM XPU.

## Validated result

On the validation system, the final `p512/g128`, concurrency-1 benchmark
produced:

  Metric                                         Result
  --------------------- -------------------------------
  Median decode                         **84.65 tok/s**
  Mean decode                           **84.49 tok/s**
  Min / Max                     **83.73 / 84.78 tok/s**
  Median TTFT                              **0.3557 s**
  Prompt / generation              **512 / 128 tokens**
  Measured runs           **5 after same-shape warmup**

This closely reproduces the approximately **83.7 tok/s** BF16-draft MTP4
reference result documented by the Intel Arc Pro B70 inference cookbook.

> \[!NOTE\] This is a known-good reference, not a guarantee that every
> B70 system will produce exactly 84.65 tok/s. PCIe topology, CPU,
> kernel/driver versions, GPU power limits, thermals, and software
> revisions can affect results.

## Why this setup is useful

-   Runs a 27B-class model on one 32 GB B70.
-   GPTQ INT4 keeps model memory practical.
-   MTP4 speculative decoding substantially improves serial decode
    speed.
-   FP8 KV cache leaves useful context capacity.
-   vLLM exposes an OpenAI-compatible API.
-   The 131,072-token configured context makes the stack attractive for
    long-context and RAG workloads.
-   Local inference can keep prompts and retrieved documents on
    infrastructure you control.

## Why it is RAG-friendly

This configuration is particularly useful for **retrieval-augmented
generation (RAG)**.

A typical RAG pipeline retrieves relevant passages or records first,
inserts that evidence into the model prompt, and asks the model to
answer from the supplied context. This stack helps because:

-   **131K configured context** provides room for retrieved passages,
    instructions, conversation history, and citations.
-   **Fast prompt processing** matters because RAG often adds
    substantial evidence before generation begins.
-   **\~84.65 tok/s serial decode** keeps answer generation responsive
    after prefill.
-   **OpenAI-compatible endpoints** make integration straightforward
    with many RAG frameworks and custom applications.
-   **Local execution** can keep retrieved documents and prompts on the
    local system.
-   Applications can use compact evidence for low latency or larger
    evidence sets when a task requires them.

For reliable RAG, retrieval should remain authoritative: retrieve good
evidence first, keep it concise, instruct the model to stay grounded in
that evidence, and measure retrieval latency separately from model TTFT
and decode speed.

------------------------------------------------------------------------

# 1. Known-good software stack

  Component                Known-good value
  ------------------------ ----------------------------------------------------
  GPU                      Intel Arc Pro B70 32 GB
  Model                    `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  Model revision           `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`
  vLLM                     `0.27.2rc1.dev77+gac7509e2b.xpu`
  XPU kernels              `vllm-xpu-kernels 0.1.12.3`
  Quantization             GPTQ INT4, symmetric, group size 128
  MTP                      4 speculative tokens, BF16 draft
  KV cache                 FP8
  Context                  131072
  Max sequences            64
  Max batched tokens       8192
  Prefix caching           Disabled for this reproduced baseline
  GPU memory utilization   0.88

### Pinned container image

``` text
vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
```

### Upstream cookbook

-   https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
-   https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md
-   https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/FULL-SETUP-COMMANDS.md

------------------------------------------------------------------------

# 2. Host prerequisites

You need:

-   Linux with a working Intel `xe` driver for the Arc Pro B70.
-   Docker with permission to pass `/dev/dri` devices into containers.
-   A B70 visible through a render node such as `/dev/dri/renderD128` or
    `/dev/dri/renderD129`.
-   A healthy PCIe link.
-   Adequate GPU cooling and power delivery.
-   Enough local storage for the model, Docker image, and caches.

> \[!IMPORTANT\] Device numbering is system-specific. **Do not blindly
> copy `card2`, `renderD129`, or PCI address `04:00.0`.** Identify the
> B70 on your own host first.

------------------------------------------------------------------------

# 3. Identify your B70

Run:

``` bash
lspci -nn | grep -Ei 'VGA|Display|Intel'

ls -l /dev/dri
ls -l /dev/dri/by-path 2>/dev/null || true

for d in /sys/class/drm/card*/device; do
    echo "=== $d ==="
    readlink -f "$d"
    cat "$d/vendor" 2>/dev/null
    cat "$d/device" 2>/dev/null
done
```

Record:

1.  B70 PCI address.
2.  `/dev/dri/cardX`.
3.  `/dev/dri/renderDXXX`.
4.  Render-node group GID.

------------------------------------------------------------------------

# 4. Verify PCIe link health

Replace `04:00.0` with your B70 PCI address:

``` bash
sudo lspci -vv -s 04:00.0 | grep -E 'LnkCap|LnkSta|LnkCap2|LnkCtl2'
```

The validation machine used for this guide was operating at **PCIe Gen3
x16**.

A badly downgraded link such as x1 should be investigated before using
inference performance as a comparison.

------------------------------------------------------------------------

# 5. Get the cookbook and model

Clone the cookbook if you do not already have it:

``` bash
cd ~
git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git
cd ~/intel-arc-pro-b70-inference-cookbook
```

The model used for this reproduction is:

``` text
SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Pinned revision:

``` text
9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e
```

Expected local directory in the commands below:

``` text
$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
```

Follow the upstream cookbook's model-download procedure so the exact
revision is preserved.

------------------------------------------------------------------------

# 6. Verify required patches

This baseline uses:

``` text
patches/patch_mtp_nightly.py
SHA256: 4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14

patches/patch_mtp_boundary.py
SHA256: 41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50
```

Verify:

``` bash
cd ~/intel-arc-pro-b70-inference-cookbook

sha256sum patches/patch_mtp_nightly.py
sha256sum patches/patch_mtp_boundary.py
```

If the hashes do not match, stop and determine whether the upstream
cookbook changed. Do not silently mix a newer patch with this baseline
and call it the same reproduction.

------------------------------------------------------------------------

# 7. Docker device mapping and oneCCL/XCCL troubleshooting

Start with the normal Intel GPU device mapping:

```bash
--device /dev/dri:/dev/dri
-v /dev/dri:/dev/dri:ro
--group-add "$RENDER_GID"
```

On the validation system, oneCCL/XCCL initially failed with errors involving `ze_fd_manager` and an inability to open the DRM device directory.

If you encounter similar oneCCL/XCCL initialization errors, add this validated workaround:

```bash
--cap-add SYS_PTRACE
--security-opt seccomp=unconfined
--ipc=host
```

These extra permissions were required on the validation host, but they should be treated as a troubleshooting workaround rather than assumed to be necessary on every Intel Arc Pro B70 system.

If the server initializes correctly without them, use the simpler device mapping.


------------------------------------------------------------------------

# 8. Launch the known-good server

First set **your** B70 render node.

Example:

``` bash
export B70_RENDER_NODE="/dev/dri/renderD129"
```

Then launch:

``` bash
cd ~/intel-arc-pro-b70-inference-cookbook

export IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
export B70_RENDER_NODE="/dev/dri/renderD129"
export RENDER_GID="$(stat -c '%g' "$B70_RENDER_NODE")"

docker rm -f qw38speed 2>/dev/null || true

docker run -d \
  --name qw38speed \
  --restart unless-stopped \
  -p 11436:8000 \
  --device /dev/dri:/dev/dri \
  -v /dev/dri:/dev/dri:ro \
  --group-add "$RENDER_GID" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash \
  "$IMAGE" \
  -lc 'set -e
python /patch_mtp.py
python /patch_boundary.py
exec vllm serve /model \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --port 8000 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --served-model-name qwen38 \
  --language-model-only \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
```

> \[!WARNING\] `ZE_AFFINITY_MASK=0` selected the B70 on the validation
> host. On a system with multiple Intel GPUs, Level Zero device ordering
> may differ. Verify that device 0 is actually the B70.

### If oneCCL/XCCL fails during startup

If the server fails with `ze_fd_manager`, DRM device-directory, or similar oneCCL/XCCL errors, stop and remove the container:

```bash
docker rm -f qw38speed
```

Then rerun the same Docker command with these three additional options inserted after `--group-add "$RENDER_GID"`:

```bash
--cap-add SYS_PTRACE \
--security-opt seccomp=unconfined \
--ipc=host \
```

This workaround was required on the validation system and successfully resolved the oneCCL/XCCL startup failure.





------------------------------------------------------------------------

# 9. Verify startup

Follow the logs:

``` bash
docker logs -f qw38speed
```

Do not benchmark until you see:

``` text
Application startup complete.
```

Then, from another terminal:

``` bash
curl -fsS http://127.0.0.1:11436/health && echo "HEALTH OK"

curl -fsS http://127.0.0.1:11436/v1/models | jq
```

The model list should contain:

``` text
qwen38
```

Useful successful-startup signs include:

-   XCCL initializes without a fatal error.
-   AutoGPTQ uses the XPU path.
-   MTP drafter initialization occurs.
-   XPU graph capture succeeds.
-   Device memory is consistent with the 32 GB B70.

The validation host showed approximately **30.3 GiB** total device
memory inside vLLM.

------------------------------------------------------------------------

# 10. First inference is a warmup

The first request can trigger Triton JIT compilation for EAGLE/MTP
kernels.

Example warnings:

``` text
Triton kernel JIT compilation during inference:
eagle_prepare_next_token_padded_kernel
eagle_step_slot_mapping_metadata_kernel
eagle_prepare_inputs_padded_kernel
```

This can cause a large one-time latency spike.

**Do not use that cold request as your benchmark result.**

Repeat the same shape once before recording measurements.

------------------------------------------------------------------------

# 11. Verify MTP is actually working

After a request:

``` bash
docker logs qw38speed 2>&1 | tail -40
```

Look for `SpecDecoding metrics`.

A working MTP path should report values such as:

-   drafted tokens
-   accepted tokens
-   mean acceptance length
-   per-position acceptance rate
-   average draft acceptance rate

During validation, early requests showed mean acceptance length around
**2.7** and average draft acceptance around **42--43%**.

------------------------------------------------------------------------

# 12. Do not trust the wrong throughput numbers

Two traps were discovered during validation.

## vLLM periodic logger

For a short 128-token request, the vLLM log reported:

``` text
Avg generation throughput: 12.8 tokens/s
```

That was **not** the actual post-first-token decode rate. It was
affected by the logger's reporting interval.

## SSE chunks are not tokens

One 128-token response arrived in only **27 SSE chunks**.

Counting chunks produced a bogus result of roughly:

``` text
16.99 chunks/s
```

Multiple generated tokens can be coalesced into one streaming event.

### Correct method

Use:

``` text
(completion_tokens - 1) / (request_end - first_generated_token)
```

where `completion_tokens` comes from the API usage block and timing uses
a monotonic client clock.

------------------------------------------------------------------------

# 13. Reproduce the p512/g128 benchmark

This test:

1.  Creates an exact 512-token prompt.
2.  Runs one same-shape warmup.
3.  Runs five measured requests.
4.  Generates exactly 128 tokens.
5.  Uses concurrency 1.
6.  Reports median/mean decode and median TTFT.

``` bash
python3 - <<'PY'
import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:11436"

seed = (
    "A retrieval augmented generation system retrieves relevant source passages, "
    "constructs a grounded context, sends the evidence to a language model, and "
    "generates a concise answer based on the supplied information. "
)

text = seed

while True:
    req = urllib.request.Request(
        BASE + "/tokenize",
        data=json.dumps({"model": "qwen38", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as r:
        tok = json.load(r)

    if len(tok["tokens"]) >= 512:
        break

    text += seed

ids = tok["tokens"][:512]

req = urllib.request.Request(
    BASE + "/detokenize",
    data=json.dumps({"model": "qwen38", "tokens": ids}).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as r:
    text = json.load(r)["prompt"]


def run():
    payload = {
        "model": "qwen38",
        "prompt": text,
        "max_tokens": 128,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first = None
    end = None
    usage = None

    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()

            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                end = time.perf_counter()
                break

            obj = json.loads(data)

            if obj.get("usage"):
                usage = obj["usage"]

            choices = obj.get("choices", [])

            if choices and choices[0].get("text") and first is None:
                first = time.perf_counter()

    if end is None:
        end = time.perf_counter()

    ct = usage["completion_tokens"]
    pt = usage["prompt_tokens"]

    return {
        "prompt": pt,
        "completion": ct,
        "ttft": first - start,
        "post": end - first,
        "rate": (ct - 1) / (end - first),
    }


print("Warmup...")
w = run()
print(f"warmup: {w['rate']:.2f} tok/s  TTFT={w['ttft']:.4f}s")

rates = []
ttfts = []

for i in range(1, 6):
    r = run()
    rates.append(r["rate"])
    ttfts.append(r["ttft"])

    print(
        f"run {i}: {r['rate']:.2f} tok/s  "
        f"TTFT={r['ttft']:.4f}s  "
        f"prompt={r['prompt']} completion={r['completion']}"
    )

print()
print(f"MEDIAN DECODE: {statistics.median(rates):.2f} tok/s")
print(f"MEAN DECODE:   {statistics.mean(rates):.2f} tok/s")
print(f"MIN/MAX:       {min(rates):.2f} / {max(rates):.2f} tok/s")
print(f"MEDIAN TTFT:   {statistics.median(ttfts):.4f} s")
PY
```

------------------------------------------------------------------------

# 14. Validated benchmark result

The known-good validation run produced:

``` text
Warmup...
warmup: 83.82 tok/s  TTFT=1.1475s

run 1: 83.73 tok/s  TTFT=0.3662s  prompt=512 completion=128
run 2: 84.65 tok/s  TTFT=0.3573s  prompt=512 completion=128
run 3: 84.70 tok/s  TTFT=0.3557s  prompt=512 completion=128
run 4: 84.61 tok/s  TTFT=0.3531s  prompt=512 completion=128
run 5: 84.78 tok/s  TTFT=0.3553s  prompt=512 completion=128

MEDIAN DECODE: 84.65 tok/s
MEAN DECODE:   84.49 tok/s
MIN/MAX:       83.73 / 84.78 tok/s
MEDIAN TTFT:   0.3557 s
```

The measured spread was only about **1.05 tok/s** from minimum to
maximum.

A different B70 host does not need to land on exactly 84.65 tok/s to be
healthy. Compare the whole environment before interpreting small
differences.

------------------------------------------------------------------------

# 15. Performance context from the validation host

  ------------------------------------------------------------------------
  Stack                 Qwen3.8-27B format                 Observed decode
  --------------------- --------------------- ----------------------------
  Ollama Vulkan         Q4_K_M                              \~11--12 tok/s

  llama.cpp SYCL        Q4_K_M                                \~27.4 tok/s

  **vLLM XPU + MTP4**   **GPTQ INT4 + BF16          **84.65 tok/s median**
                        draft**               
  ------------------------------------------------------------------------

These ratios are **not universal engine benchmarks**. They are
measurements from one validation host and illustrate why the vLLM
XPU/MTP path was investigated.

------------------------------------------------------------------------

# 16. Check GPU power limits

Do not blindly copy a sysfs path or write a new power limit.

First inspect the B70's hwmon entries.

Replace `card2` with your actual B70 card:

``` bash
for h in /sys/class/drm/card2/device/hwmon/hwmon*; do
    echo "=== $h ==="
    cat "$h/name" 2>/dev/null
    cat "$h/power1_cap" 2>/dev/null
    cat "$h/power1_cap_max" 2>/dev/null
done
```

The reference campaign used a **230 W configured cap**.

Also check temperatures, clocks, and throttling while the benchmark is
running.

------------------------------------------------------------------------

# 17. Troubleshooting

  -----------------------------------------------------------------------
  Symptom                             Check
  ----------------------------------- -----------------------------------
  `ze_fd_manager` / oneCCL / XCCL     Use full `/dev/dri`, read-only DRI
  failure                             bind, render GID, `SYS_PTRACE`,
                                      `seccomp=unconfined`, and
                                      `--ipc=host`

  Wrong Intel GPU selected            Verify Level Zero affinity and
                                      check that detected memory matches
                                      a 32 GB B70

  First request is unusually slow     Check for Triton JIT compilation;
                                      repeat the same shape

  vLLM says \~12.8 tok/s              Do not use periodic logger for this
                                      short C1 benchmark

  Streaming test counts very few      SSE chunks are not tokens; use
  "tokens"                            `completion_tokens`

  MTP appears inactive                Check `SpecDecoding metrics` for
                                      drafted and accepted tokens

  Throughput is unexpectedly poor     Check PCIe width/speed, clocks,
                                      power, thermals, throttling, and
                                      software versions

  Port 11436 is occupied              Change only the host side of
                                      `-p 11436:8000`
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 18. Reproducibility checklist

When posting your own B70 result, include:

-   [ ] Linux distribution
-   [ ] Kernel version
-   [ ] Intel `xe` driver status
-   [ ] B70 PCI address
-   [ ] Negotiated PCIe generation and width
-   [ ] B70 card/render node
-   [ ] Level Zero affinity selection
-   [ ] Docker version
-   [ ] vLLM image digest
-   [ ] vLLM version
-   [ ] XPU kernel version
-   [ ] Exact model revision
-   [ ] Patch SHA256 hashes
-   [ ] vLLM command-line arguments
-   [ ] Relevant environment variables
-   [ ] GPU power cap
-   [ ] GPU clocks/temperature/throttling state
-   [ ] Prompt-token count
-   [ ] Output-token count
-   [ ] Concurrency
-   [ ] Warmup procedure
-   [ ] Number of measured repetitions
-   [ ] Timing formula
-   [ ] Median rather than only the best run

------------------------------------------------------------------------

# 19. Preserve the baseline before optimizing

Once this baseline works, save the configuration before applying newer
optimization overlays.

Newer cookbook revisions may contain changes such as:

-   draft INT4
-   MTP INT4
-   GDN mixed-split changes
-   newer vLLM images
-   different memory-utilization settings
-   different prefix-cache settings

Those may be faster, but they are **different benchmark
configurations**.

For each experiment, record a new:

-   image digest
-   model revision
-   patch set/hashes
-   launch command
-   benchmark result

That makes comparisons meaningful and makes it possible to return to the
known-good baseline.

------------------------------------------------------------------------

# 20. Quick recovery sequence

1.  Identify the B70 PCI address and render node.
2.  Verify the PCIe link.
3.  Open/clone the cookbook.
4.  Confirm the pinned model revision is present.
5.  Verify the two patch hashes.
6.  Set `B70_RENDER_NODE` correctly.
7.  Launch with the full `/dev/dri` + oneCCL Docker settings.
8.  Wait for `Application startup complete`.
9.  Check `/health`.
10. Check `/v1/models`.
11. Confirm the correct 32 GB B70 was selected.
12. Run one same-shape warmup.
13. Run five p512/g128 measured requests.
14. Compare the median with the known-good \~84 tok/s result.
15. If substantially different, investigate PCIe, GPU selection, JIT
    warmup, MTP metrics, power/thermals, and software revisions before
    changing tuning parameters.

------------------------------------------------------------------------

## Credits and upstream work

This reproduction builds on the work published in:

**SergiioB / intel-arc-pro-b70-inference-cookbook**\
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

The goal of this README is to document a successfully reproduced B70
configuration, including the Docker/oneCCL details and benchmark
pitfalls encountered while reproducing it, so other B70 owners can
validate their own systems.

If you reproduce the result on another B70, consider posting your
hardware/software details and measured median so the community can
compare configurations.
