#!/usr/bin/env python3

import json
import statistics
from pathlib import Path

EXPECTED = {
    512: 81.2390,
    8192: 73.3014,
    16384: 74.7602,
    32768: 68.3402,
    65536: 66.1055,
    120000: 50.3087,
}

BASE = Path(__file__).resolve().parent

all_ok = True

print()
print("Intel Arc Pro B70 — Qwen3.8-27B — MTP4 Context Scaling")
print("=" * 74)
print(
    f"{'Prompt':>10}  {'Reps':>4}  {'MTP pos':>13}  "
    f"{'Median tok/s':>12}  {'Cache':>7}  {'Status':>8}"
)
print("-" * 74)

for target, expected_median in EXPECTED.items():
    path = BASE / f"context-{target}-results.json"

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        reps = data
    elif isinstance(data, dict):
        reps = (
            data.get("results")
            or data.get("reps")
            or data.get("records")
            or data.get("runs")
            or []
        )
    else:
        reps = []

    rates = []
    positions = set()
    prompt_tokens = set()
    completion_tokens = set()
    cache_hits = []

    for r in reps:
        if not isinstance(r, dict):
            continue

        rate = r.get("client_post_first_tps")
        if isinstance(rate, (int, float)):
            rates.append(rate)

        if isinstance(r.get("prompt_tokens"), int):
            prompt_tokens.add(r["prompt_tokens"])

        if isinstance(r.get("completion_tokens"), int):
            completion_tokens.add(r["completion_tokens"])

        if "prefix_cache_hits_delta" in r:
            cache_hits.append(r.get("prefix_cache_hits_delta"))

        per_pos = r.get("mtp_accepted_per_position_delta", {})
        if isinstance(per_pos, dict):
            for key in per_pos:
                if 'position="' in key:
                    try:
                        positions.add(
                            int(key.split('position="', 1)[1].split('"', 1)[0])
                        )
                    except Exception:
                        pass

    median = statistics.median(rates) if rates else float("nan")

    mtp4_ok = positions == {0, 1, 2, 3}
    prompt_ok = prompt_tokens == {target}
    output_ok = completion_tokens == {128}
    cache_ok = bool(cache_hits) and all(x == 0.0 for x in cache_hits)
    median_ok = abs(median - expected_median) < 0.0001

    ok = mtp4_ok and prompt_ok and output_ok and cache_ok and median_ok
    all_ok &= ok

    print(
        f"{target:>10,}  "
        f"{len(reps):>4}  "
        f"{str(sorted(positions)):>13}  "
        f"{median:>12.4f}  "
        f"{'cold' if cache_ok else 'CHECK':>7}  "
        f"{'PASS' if ok else 'FAIL':>8}"
    )

print("-" * 74)

if all_ok:
    print("VERDICT: ALL SIX DATASETS CONFIRMED MTP4")
    print("         Exact prompt lengths, 128 output tokens, zero prefix-cache hits.")
else:
    print("VERDICT: VERIFICATION FAILED — review data before publishing.")
    raise SystemExit(1)

print()
