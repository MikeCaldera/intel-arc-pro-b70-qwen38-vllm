#!/usr/bin/env python3
"""Run every GPU-free repository gate through one stable command."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKS = [
    ("system-map", [sys.executable, "scripts/validate-system-map.py"]),
    ("markdown-links", [sys.executable, "scripts/check-markdown-links.py"]),
    ("generated-catalog", [sys.executable, "scripts/render-benchmark-catalog.py", "--check"]),
    ("python-syntax", [sys.executable, "-m", "compileall", "-q", "scripts", "tests"]),
    ("unit-tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
    ("triage-lib", ["node", "--test", "tests/test_triage_lib.js"]),
]


def run_checks() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, command in CHECKS:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append({
            "name": name,
            "command": " ".join(command),
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        })
        if completed.returncode != 0:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    results = run_checks()

    if args.format == "json":
        print(json.dumps({"ok": all(r["ok"] for r in results), "checks": results}, indent=2))
    else:
        for result in results:
            marker = "PASS" if result["ok"] else "FAIL"
            print(f"[{marker}] {result['name']}: {result['command']}")
            details = result["stdout"] or result["stderr"]
            if details:
                print(details)
    return 0 if len(results) == len(CHECKS) and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
