# Issue and PR checks

What the bots label and ask. How to get to `ready`.

## Issues

| You included | Label |
|--------------|--------|
| Image sha256 (or `f01e24f6` / `2c427ef`) **and** a `docker run` / `vllm serve` line | `ready` |
| Missing either | `needs-info` |
| Hang (`xe coredump` / `Timedout job`) without dmesg | `needs-info` — paste ~30 lines around the timeout |
| How-to question, no sha, no command | `question` — use [Discussions](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/discussions) |
| TP2 + draft-INT4 patches | `tp2-draft-block` — those two patches are C1 only; drop them for TP>1 |

`--tensor-parallel-size` is not required on a 1× card.

Triage never closes an issue. The stale bot only expires `needs-info` after 14+7 days with no reply. `ready`, `tp2-draft-block`, and `driver-hang` are exempt.

## PRs

The PR template asks for: family, image digest, the command you ran and its last line, prefix cache on/off, TP, and what you are not claiming.

On open, a bot comments once: which `patches/*.py` files changed, whether a verify script is in the diff, whether the body has a digest and a run.

For text patches, GPU-free apply on the pinned image is enough:

```
IMAGE=vllm/vllm-openai-xpu@sha256:<pinned> bash scripts/verify-mtp-apc-fixes.sh
```

`docker run` without `--device`. Pass means the patches apply, re-apply is a no-op, and the touched files compile in both vLLM trees. That is not a live corruption measurement.
