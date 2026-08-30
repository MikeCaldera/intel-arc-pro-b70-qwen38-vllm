# Dual-B70 Multi-GPU Serving (TP2 / PP2)

Serving **one model across both Arc Pro B70 cards** with the pinned
vLLM-XPU image (`--tensor-parallel-size 2`, or `--pipeline-parallel-size 2`).
For two independent single-card servers, do **not** use this page — run the
normal single-card launch twice, once with `ZE_AFFINITY_MASK=0` and once
with `ZE_AFFINITY_MASK=1`.

Status: TP2 with the affinity patch was validated on the author's dual-B70
host (ASUS ROG STRIX X570-F, kernel 7.0.0-28). The four oneCCL threshold
variables below were then **confirmed serving on two additional dual-B70
hosts** through cookbook
[issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8)
— including one host that crashed without them even with the affinity patch
and identical driver packages. A matched re-measure on the author's host
with the variables set is pending; treat them as required, not optional.

## The failure you will hit without the fix

```text
RuntimeError: oneCCL: ze_call.cpp:28 do_call: EXCEPTION: ze error at
zeMemOpenIpcHandle, code: ZE_RESULT_ERROR_INVALID_ARGUMENT
```

It always aborts at the first `all_reduce` inside `profile_run` /
`determine_available_memory` — before the server ever comes up — regardless
of vLLM flags, `CCL_ZE_IPC_EXCHANGE` values, `--ipc=host`, or
`VLLM_XPU_ENABLE_XPU_GRAPH`. The identical error reproduces in a bare
two-process `xccl` all_reduce with no vLLM involved.

## Root cause — two independent pieces

**1. Spawn-time device visibility (the affinity patch).** A container-level
`ZE_AFFINITY_MASK=0,1` is not equivalent to per-worker masks: both worker
subprocesses then see both GPUs in one Level Zero context, and setting the
mask inside `init_device()` is already too late (Level Zero initializes when
the subprocess starts). Exposing both cards to one process also costs about
1 GiB of host RAM per 1 GiB of VRAM
([intel/compute-runtime#986](https://github.com/intel/compute-runtime/issues/986)).
`patches/patch_vllm_worker_affinity.py` injects `ZE_AFFINITY_MASK=<rank>`
into each worker's environment *before* spawn, so every rank starts with
exactly one visible device.

**2. Allreduce algorithm selection (the CCL thresholds).** Above a small
default message size, oneCCL's SYCL allreduce switches from the simple
algorithm to the large multi-kernel algorithm, which performs a Level Zero
IPC exchange of every peer rank's temp buffers — upstream source comment:
*"perform IPC exchange every time"* (oneCCL
`src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.hpp`).
`zeMemOpenIpcHandle` **is** that exchange, and the `xe` driver rejects it
with `INVALID_ARGUMENT` on desktop boards where the two cards sit on their
own CPU root ports with no shared PCIe switch and no GPU P2P. Raising the
thresholds pins every realistic TP2 message (the largest is
`max_num_batched_tokens × hidden × 2 B ≈ 40 MiB` in these recipes) on the
simple/tmp-buffer algorithms, which never open peer device memory.

This is Intel's own workaround for the identical 2× B60 failure —
[intel/llm-scaler#594](https://github.com/intel/llm-scaler/issues/594),
Intel's Wesley-Du: *"please try this as a workaround as your host platform
does not support P2P"*. The same error and the same fix class (host-staged
collectives instead of device-IPC collectives) appear in the independent
[humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) dual-B70 stack.

## The fix

### 1. Container environment (required)

```bash
-e ZE_AFFINITY_MASK=0,1 \
-e B70_WORKER_AFFINITY=1 \
-e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
```

`4294967296` (4 GiB) is Intel's value from llm-scaler#594; `1073741824`
(1 GiB) is community-confirmed to work too (every allreduce here is ≪ 1 GiB).

**Do not** add `CCL_ZE_IPC_EXCHANGE` / `CCL_ATL_TRANSPORT` / `FI_PROVIDER` /
`CCL_ATL_SHM` overrides. They only choose *how* IPC handles are exchanged,
not whether device IPC is used at all — and some of them push oneCCL back
into the failing path. Leave every one of those at its default.

### 2. Docker flags (required)

```bash
--device /dev/dri --ipc=host --cap-add SYS_PTRACE
```

`SYS_PTRACE` keeps oneCCL's cross-rank pidfd exchange working under Docker
seccomp ([uxlfoundation/oneCCL#217](https://github.com/uxlfoundation/oneCCL/issues/217)).
`--privileged` / `--security-opt seccomp=unconfined` are **not** required.

### 3. Patches

```bash
python /patches/patch_mtp_nightly.py && \
python /patches/patch_mtp_boundary.py && \
python /patches/patch_vllm_worker_affinity.py
```

Chain them (`&&` or `set -e`) and confirm each prints its success line — a
backslash-continued chain that repeats `python` on every line silently runs
only the first script. Which patches apply to which checkpoint is in
[IMAGE-AND-PATCH-MATRIX.md](IMAGE-AND-PATCH-MATRIX.md) and summarized below.

### Complete launch — Qwen3.8-27B GPTQ-Int4 MTP example

Community-confirmed working configuration (issue #8, second dual-B70 host):

```bash
export COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
export IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
RENDER_GID="$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')"

docker rm -f qw38tp2 >/dev/null 2>&1 || true
docker run -d --name qw38tp2 -p 8000:8000 \
  --device /dev/dri --ipc=host --cap-add SYS_PTRACE --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro -v "$MODEL_DIR:/model:ro" \
  -v "$COOKBOOK/patches:/patches:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0,1 \
  -e B70_WORKER_AFFINITY=1 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    python /patches/patch_mtp_nightly.py
    python /patches/patch_mtp_boundary.py
    python /patches/patch_vllm_worker_affinity.py
    exec vllm serve /model --served-model-name qwen38
      --quantization gptq --dtype float16 --kv-cache-dtype fp8
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"
      --tensor-parallel-size 2
      --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096
      --gpu-memory-utilization 0.90 --port 8000'
```

`--max-model-len 262144` is the community-confirmed value; lower it if you
do not need full-context KV. BF16 vs FP16 dtype is a measured wash on this
stack. For PP2 use `--pipeline-parallel-size 2` with
`--tensor-parallel-size 1` — the affinity patch covers both topologies.

`VLLM_XPU_ENABLE_XPU_GRAPH=0` is deliberate: XPU graph capture is refused on
multi-GPU ("only supports single-GPU execution") on every build tested
(0.21.1 / 0.27.2 / 0.28.0). Compile-only is the expected TP2 execution mode.

## Verify

Expected lines after startup:

```text
[B70_WORKER_AFFINITY_SPAWN] rank=0 mask=0
[B70_WORKER_AFFINITY_SPAWN] rank=1 mask=1
[B70_WORKER_AFFINITY] pid=... rank=0 visible=1
[B70_WORKER_AFFINITY] pid=... rank=1 visible=1
|CCL_WARN| value of CCL_ATL_TRANSPORT changed to be ofi (default:mpi)
|CCL_WARN| comm_dev_uuids is not sub-vector of node_dev_uuids, comm_dev_uuids size 2, node_dev_uuids size 1, ...
|CCL_WARN| number of result device uuids does not match number of ranks per host, result size 1, host_rank_info_vec size 2, ...
```

The `CCL_WARN` lines are **expected and benign** — they are oneCCL noticing
that each rank enumerates one device while the communicator holds two UUIDs,
which is exactly the state the per-worker masks create. Healthy runs print
them immediately before successful collectives. Then check
`curl -f http://127.0.0.1:8000/health`.

## 60-second isolation repro (no vLLM)

If anything still fails, isolate the collective from the server. Inside the
same container (`SYS_PTRACE`, `/dev/dri` mounted):

```python
# xccl_minirepro.py
import os, datetime, torch, torch.distributed as dist

rank = int(os.environ["RANK"])
torch.xpu.set_device(0)  # each process sees exactly one GPU via ZE_AFFINITY_MASK
dist.init_process_group("xccl", init_method="env://", world_size=2, rank=rank,
                        timeout=datetime.timedelta(seconds=60))
t = torch.zeros(8, device="xpu:0") + rank + 1
dist.all_reduce(t)
print(f"[mini rank={rank}] allreduce OK mean={t.float().mean().item()}", flush=True)
dist.barrier()
print(f"[mini rank={rank}] DONE", flush=True)
```

```bash
ZE_AFFINITY_MASK=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 python xccl_minirepro.py &
ZE_AFFINITY_MASK=1 RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 python xccl_minirepro.py
```

Expected: `allreduce OK mean=3.0` from both ranks.

## Which patches apply (Qwen3.8-27B on the pinned nightly)

| Patch | Status |
|---|---|
| `patch_mtp_nightly.py`, `patch_mtp_boundary.py` | Required (preserved-MTP checkpoint; boundary completes an exact final MTP group) |
| `patch_vllm_worker_affinity.py` | Required for TP2 / PP2 |
| `patch_draft_lmhead_int4.py` **then** `patch_draft_mtp_int4.py` (with `B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1`) | Optional MTP speed: matched same-image n=5 p512/g128 81.20 → 112.65, p8192/g128 77.52 → 103.63; acceptance 95.9% → 94.4% |
| `patch_gdn_mixed_split_v5.py` | Optional, concurrency only (mixed prefill + spec-decode batches); C1 speed-flat |
| `patch_fp8_w8a16.py` | Skip — reroutes block-**FP8** weights only; GPTQ-Int4 checkpoints never take this path |
| `patch_xpu_int4_moe_v4.py`, `patch_mtp_bf16_draft.py` | Never on this nightly (historical local vLLM 0.21 image) |

## Notes

- **`pcie_acs_override` is not needed.** The failing call is a user-space
  Level Zero IPC open inside oneCCL, not kernel P2P routing. Hosts serve
  with stock kernel parameters once the algorithm is forced off the device
  IPC path.
- **Topology is not the differentiator.** The author's working host has the
  same class of layout as the failing hosts: a desktop board with each B70
  on its own CPU-attached root port (`00:03.1` / `00:03.2` on X570-F),
  no shared switch, no GPU P2P. The author's host ran TP2 without the CCL
  thresholds (kernel 7.0.0-28); the confirmed-failing hosts run 7.0.0-30 and
  other kernels — the exact host delta is not pinned, and with the algorithm
  forced explicitly it stops mattering.
- **What TP2 buys.** Decode gains scale (~1.4× class on comparable stacks);
  prefill does not scale over the CPU-attached x8 links (activations bounce
  through host RAM). Both cards are occupied while TP2 runs.
- **Provenance.** Intel workaround: [intel/llm-scaler#594](https://github.com/intel/llm-scaler/issues/594)
  (Wesley-Du). Dual-B70 confirmations and the debugging trail:
  [cookbook issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8)
  (deriknel's working config, uldiseihenbergs' isolation bisect). Same-error
  convergence with host-staged collectives:
  [JP-devv/humble-b70-llm](https://github.com/JP-devv/humble-b70-llm).
