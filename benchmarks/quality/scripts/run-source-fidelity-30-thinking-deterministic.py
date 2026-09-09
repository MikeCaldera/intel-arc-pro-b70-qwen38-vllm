#!/usr/bin/env python3

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e

    elapsed = time.perf_counter() - start
    return json.loads(raw), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["mtp2", "mtp3", "mtp4"])
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--prompts",
        default="prompts/desktop-source-fidelity-30-v1.json",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_dir = Path("results") / args.config
    output_dir.mkdir(parents=True, exist_ok=True)

    with prompts_path.open("r", encoding="utf-8") as f:
        tests = json.load(f)

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    results = []

    print(f"Configuration : {args.config}")
    print(f"Endpoint      : {url}")
    print(f"Tests         : {len(tests)}")
    print()

    for number, test in enumerate(tests, start=1):
        payload = {
            "model": "qwen38",
            "messages": [
                {
                    "role": "user",
                    "content": test["prompt"],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": True
            },
        }

        print(
            f"[{number:02d}/{len(tests):02d}] "
            f"{test['id']} ({test['category']}) ..."
        )

        try:
            response, latency = post_json(url, payload)

            choice = response["choices"][0]
            message = choice["message"]
            usage = response.get("usage") or {}

            raw_content = message.get("content") or ""
            native_reasoning = message.get("reasoning")

            # Qwen3.8/vLLM may either:
            #   1. return reasoning separately in message["reasoning"], or
            #   2. embed reasoning in content followed by </think>.
            #
            # Store only the final answer in "response" and preserve
            # the thinking text separately in "reasoning".
            if native_reasoning:
                final_content = raw_content.strip()
                reasoning_content = native_reasoning
            elif "</think>" in raw_content:
                reasoning_part, final_part = raw_content.split("</think>", 1)
                reasoning_content = reasoning_part.strip()

                if reasoning_content.startswith("<think>"):
                    reasoning_content = reasoning_content[len("<think>"):].lstrip()

                final_content = final_part.strip()
            else:
                reasoning_content = None
                final_content = raw_content.strip()

            result = {
                "id": test["id"],
                "category": test["category"],
                "configuration": args.config,
                "prompt": test["prompt"],
                "expected": test["expected"],
                "response": final_content,
                "reasoning": reasoning_content,
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "latency_seconds": round(latency, 6),
                "system_fingerprint": response.get("system_fingerprint"),
                "raw_response": response,
                "error": None,
            }

            print(
                f"       finish={result['finish_reason']} "
                f"tokens={result['completion_tokens']} "
                f"latency={result['latency_seconds']:.3f}s"
            )

        except Exception as exc:
            result = {
                "id": test["id"],
                "category": test["category"],
                "configuration": args.config,
                "prompt": test["prompt"],
                "expected": test["expected"],
                "response": None,
                "reasoning": None,
                "finish_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_seconds": None,
                "system_fingerprint": None,
                "raw_response": None,
                "error": str(exc),
            }

            print(f"       ERROR: {exc}")

        results.append(result)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output = {
        "benchmark": "desktop-source-fidelity-30-thinking-v1",
        "configuration": args.config,
        "endpoint": url,
        "created_utc": timestamp,
        "settings": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "enable_thinking": True,
        },
        "results": results,
    }

    out_file = output_dir / f"desktop-source-fidelity-30-thinking-v1-{args.config}-{timestamp}.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved: {out_file}")

    errors = sum(1 for x in results if x["error"])
    truncated = sum(1 for x in results if x["finish_reason"] == "length")

    print(f"Errors       : {errors}")
    print(f"Truncations  : {truncated}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
