#!/usr/bin/env python3
"""Validate repository-relative links in tracked Markdown files."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def repository_markdown() -> list[pathlib.Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def validate_links(files: list[pathlib.Path]) -> list[str]:
    errors: list[str] = []
    for source in files:
        for line_number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
            for raw_target in LINK_RE.findall(line):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                path_text = unquote(target.split("#", 1)[0])
                if not path_text or "<" in path_text or ">" in path_text:
                    continue
                resolved = (source.parent / path_text).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    errors.append(f"{source.relative_to(ROOT)}:{line_number}: link escapes repository: {target}")
                    continue
                if not resolved.exists():
                    errors.append(f"{source.relative_to(ROOT)}:{line_number}: missing link target: {target}")
    return errors


def main() -> int:
    files = repository_markdown()
    errors = validate_links(files)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: Markdown links ({len(files)} repository files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
