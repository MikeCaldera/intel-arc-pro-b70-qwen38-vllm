# vLLM XPU kernel source fixes (open PRs)

These are the two **source** edits in
[vllm-project/vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524).
They are **not** Python runtime patches. They change C++ / kernel-config and
need a `vllm-xpu-kernels` rebuild (or a future image that includes the PR).

The DFlash / no-spec launchers already apply the **Python** router fix
(`../patch_xpu_grouped_topk_native_v2.py` —
[vllm-project/vllm#52159](https://github.com/vllm-project/vllm/pull/52159)
plus the measured native XPU body). That is enough to *serve* Nemotron DFlash
from the public digest.

| File | What | Needed for DFlash serve? |
|---|---|---|
| `0001-zero-xe2-grouped-gemm-atomic.py` | `at::empty` → `at::zeros` on the Xe2 grouped-GEMM scheduler counter | No. Needed for temperature-0 graph-replay determinism |
| `0002-muse-paged-decode-tuple.py` | add `16,128,64,false,true,false` | **No.** Muse-Glimmer local paged-decode only |

## Apply to a kernels checkout

```bash
git clone --depth=1 https://github.com/vllm-project/vllm-xpu-kernels.git
python3 patches/vllm-xpu-kernels/0001-zero-xe2-grouped-gemm-atomic.py \
  --root vllm-xpu-kernels
python3 patches/vllm-xpu-kernels/0002-muse-paged-decode-tuple.py \
  --root vllm-xpu-kernels
# then rebuild vllm-xpu-kernels per that repo's README
```

Both scripts fail closed if the public-HEAD anchors are gone (PR already
merged, or the file moved). Do **not** treat a local Docker tag as the
reproduce default.
