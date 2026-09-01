# Maintainer pre-checks (issues and PRs)

How this repo decides `ready` vs `needs-info`, and what a human reply is allowed to claim.

Automation: `.github/scripts/cookbook-triage.js` (issues), `.github/scripts/pr-precheck.js` (PRs). Voice for Sergio's comments: keep them short and factual. Do not put GitHub closing keywords (`fix` / `close` / `resolve` `#N`) in commits that only change triage.

## Issues

Skip PRs. Never close from triage.

| Have | Label |
|------|--------|
| image sha256 (or `f01e24f6` / `2c427ef`) **and** `docker run` / `vllm serve` | `ready` |
| missing either | `needs-info` |
| `xe coredump` / `Timedout job` without dmesg | `needs-info` + ask for ~30 lines |
| title is a how-to, no sha, no command | `question` → Discussions |
| TP2 + draft-INT4 patches | `tp2-draft-block` — **do not close**. Runtime skip is still missing on master |

Do not require `--tensor-parallel-size` on a 1× card. Stale bot only touches `needs-info` (14 + 7 days). `ready`, `tp2-draft-block`, `driver-hang` are exempt.

This lab is 1–2 B70 and currently 16 GB host RAM. Do not claim a TP4 repro or a 27B soak from here.

## PRs

Template asks: family, image digest, command + last line, cache on/off, TP, what you are **not** claiming.

Bot comment on open (once): lists `patches/*.py`, whether a verify script is in the diff, whether the body has a digest and a run.

GPU-free bar for text patches:

```
IMAGE=vllm/vllm-openai-xpu@sha256:<pinned> bash scripts/verify-mtp-apc-fixes.sh
```

`docker run` without `--device`. Pass = apply + idempotent + `py_compile` on both trees.

`APPLY_SERGIO` is an env var **inside** that script meaning "apply this cookbook's Qwen chain first". Never ask the contributor about it. Run `APPLY_SERGIO=1` ourselves before merge if the PR ships next to that chain.

GPU-free pass is not a live corruption rate. popisec's own idle 2-row box measured 0. Do not merge on "we saw the bug here" unless we did.

## Human reply (before posting)

1. Read master / the diff. Quote a SHA or a command.
2. JSON body → `gh api --input`. Never `-f body=@file`.
3. Re-read the comment. Fail if it starts with `@/`.
4. PATCH in place to edit wording. Do not stack a second comment.
5. Do not merge unless Sergio said merge.
