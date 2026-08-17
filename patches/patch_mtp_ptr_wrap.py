#!/usr/bin/env python3
"""Patch vLLM mamba_utils to handle high GPU addresses on Intel XPU.

On the B70, torch.xpu data_ptr() returns addresses in the upper half of the
64-bit space (>= 2**63). Storing such a value into an int64 tensor raises
"ValueError: Overflow when unpacking long long". The XPU GDN/mamba kernels
read these int64 slots and reinterpret the bits as pointers, so wrapping to a
signed int64 (same bit pattern) is safe and correct.
"""
from __future__ import annotations

from pathlib import Path


def wrap_ptr(value: int) -> int:
    """Wrap an unsigned 64-bit pointer into signed int64 (preserving bits)."""
    value &= 0xFFFFFFFFFFFFFFFF
    if value >= 0x8000000000000000:
        value -= 0x10000000000000000
    return value


PATCHES = [
    # initialize_from_forward_context: state base address
    (
        "self.state_base_addrs[idx] = state.data_ptr()",
        "self.state_base_addrs[idx] = wrap_ptr(state.data_ptr())",
    ),
    # block table pointers
    (
        "self.block_table_ptrs[i] = bt.data_ptr()",
        "self.block_table_ptrs[i] = wrap_ptr(bt.data_ptr())",
    ),
    # do_mamba_copy_block: per-request dest pointers
    (
        "dst_ptrs_np[offset] = state[dest_block_id].data_ptr()",
        "dst_ptrs_np[offset] = wrap_ptr(state[dest_block_id].data_ptr())",
    ),
]

MARKER = "B70_PTR_WRAP"


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    # Inject helper after the import block
    if MARKER not in text:
        if "\ndef wrap_ptr" not in text:
            # find a good insertion point: after the last top-level import
            lines = text.splitlines()
            last_import = 0
            for i, ln in enumerate(lines):
                if ln.startswith(("import ", "from ")):
                    last_import = i
            helper = (
                f"\n\ndef {MARKER}_wrap_ptr(value: int) -> int:\n"
                f'    """Wrap unsigned 64-bit pointer into signed int64."""\n'
                "    value &= 0xFFFFFFFFFFFFFFFF\n"
                "    if value >= 0x8000000000000000:\n"
                "        value -= 0x10000000000000000\n"
                "    return value\n"
            )
            lines.insert(last_import + 1, helper)
            text = "\n".join(lines)
    for old, new in PATCHES:
        if old in text:
            text = text.replace(old, new.replace("wrap_ptr", MARKER + "_wrap_ptr"), 1)
    return text


def main() -> None:
    import vllm
    import os

    vllm_dir = os.path.dirname(vllm.__file__)
    path = Path(vllm_dir) / "v1" / "worker" / "mamba_utils.py"
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
