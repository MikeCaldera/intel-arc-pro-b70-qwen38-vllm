#!/usr/bin/env python3
"""B70: split mixed spec-decode + non-spec (prefill) gdn_attention calls on XPU.

SYMPTOM (measured 14-ago-2026, vLLM 0.26.1rc1.dev457 XPU, MTP4 + Fase S+M1):
Under concurrency, a prefill request scheduled in the same step as running
spec-decode requests crashes the engine:

    RuntimeError: causal_conv1d does not support spec-decode and non-spec
    (prefill + decode) tokens in the same invocation; the spec path and the
    non-spec path are mutually exclusive

The guard lives in the compiled `gdn_attention` op (`_xpu_C`). Single-request
(C1) serving never mixes tokens, so the crash only appears with 2+ requests
in flight (the harness always ran 1 request at a time -> never caught).

ROOT CAUSE:
`_gdn_attention_core_xpu_impl` (vllm/_xpu_ops.py) forwards the full mixed
batch to `torch.ops._xpu_C.gdn_attention(...)` in one call. The compiled
kernel supports each group alone (pure spec decode steps and pure prefill
steps both work) but refuses to process both in one invocation.

FIX:
When `num_spec_decodes > 0` AND `num_prefills > 0` (or `num_decodes > 0`),
split the invocation into two calls:
  1. spec-only  : spec_token_indx / spec_state_indices / spec_query_start_loc
                  / num_accepted_tokens, other groups nulled.
  2. non-spec   : non_spec_token_indx / non_spec_state_indices /
                  non_spec_query_start_loc / has_initial_state, spec nulled.
The two groups index disjoint token/state positions (absolute token_indx into
the full buffers), so the calls compose: outputs land in `core_attn_out` at
each group's positions, and conv/ssm state updates touch disjoint cache lines.

`num_actual_tokens` is set to each group's token count (== its token_indx
length) in the split calls, mirroring how pure steps pass it.

SCOPE / GATING:
- Env-gated: `B70_SPLIT_MIXED_GDN` (default "1" when VLLM_TARGET_DEVICE=xpu,
  set "0" to keep upstream behavior / the crash). Non-XPU: no-op.
- The condition is a Python-level check on metadata ints; the op is opaque to
  torch.compile and runs eagerly, so no graph re-specialization concerns.
- Idempotent: re-running is a no-op; anchors are asserted.
"""
from __future__ import annotations

import os


def patch_text(text: str) -> str:
    if "B70_SPLIT_MIXED_GDN" in text:
        return text

    old = """    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )"""

    new = """    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    def _b70_gdn_call(
        num_prefills_,
        num_decodes_,
        num_spec_decodes_,
        has_initial_state_,
        non_spec_query_start_loc_,
        non_spec_token_indx_,
        non_spec_state_indices_tensor_,
        spec_query_start_loc_,
        spec_token_indx_,
        spec_state_indices_tensor_,
        num_accepted_tokens_,
        num_actual_tokens_,
    ):
        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            conv_state=self.kv_cache[0],
            ssm_state=self.kv_cache[1],
            conv_weights=conv_weights,
            conv_bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_prefills=num_prefills_,  # type: ignore[attr-defined]
            num_decodes=num_decodes_,  # type: ignore[attr-defined]
            num_spec_decodes=num_spec_decodes_,  # type: ignore[attr-defined]
            has_initial_state=has_initial_state_,  # type: ignore[attr-defined]
            non_spec_query_start_loc=non_spec_query_start_loc_,  # type: ignore[attr-defined]
            non_spec_token_indx=non_spec_token_indx_,  # type: ignore[attr-defined]
            non_spec_state_indices_tensor=non_spec_state_indices_tensor_,  # type: ignore[attr-defined]
            spec_query_start_loc=spec_query_start_loc_,  # type: ignore[attr-defined]
            spec_token_indx=spec_token_indx_,  # type: ignore[attr-defined]
            spec_state_indices_tensor=spec_state_indices_tensor_,
            num_accepted_tokens=num_accepted_tokens_,  # type: ignore[attr-defined]
            num_actual_tokens=num_actual_tokens_,  # type: ignore[attr-defined]
            tp_size=self.tp_size,
            reorder_input=not self.gqa_interleaved_layout,
        )

    if (
        _b70_split_mixed_gdn()
        and num_spec_decodes > 0
        and (num_prefills > 0 or num_decodes > 0)
    ):
        # Mixed spec-decode + non-spec batch: the compiled kernel rejects it.
        # Split into two calls on disjoint token/state indices.
        if spec_token_indx is not None:
            _b70_gdn_call(
                num_prefills_=0,
                num_decodes_=0,
                num_spec_decodes_=num_spec_decodes,
                has_initial_state_=None,
                non_spec_query_start_loc_=None,
                non_spec_token_indx_=None,
                non_spec_state_indices_tensor_=None,
                spec_query_start_loc_=spec_query_start_loc,
                spec_token_indx_=spec_token_indx,
                spec_state_indices_tensor_=spec_state_indices_tensor,
                num_accepted_tokens_=num_accepted_tokens,
                num_actual_tokens_=spec_token_indx.size(0),
            )
        if non_spec_token_indx is not None:
            _b70_gdn_call(
                num_prefills_=num_prefills,
                num_decodes_=0,
                num_spec_decodes_=0,
                has_initial_state_=has_initial_state,
                non_spec_query_start_loc_=non_spec_query_start_loc,
                non_spec_token_indx_=non_spec_token_indx,
                non_spec_state_indices_tensor_=non_spec_state_indices_tensor,
                spec_query_start_loc_=None,
                spec_token_indx_=None,
                spec_state_indices_tensor_=None,
                num_accepted_tokens_=None,
                num_actual_tokens_=non_spec_token_indx.size(0),
            )
    else:
        _b70_gdn_call(
            num_prefills_=num_prefills,
            num_decodes_=num_decodes,
            num_spec_decodes_=num_spec_decodes,
            has_initial_state_=has_initial_state,
            non_spec_query_start_loc_=non_spec_query_start_loc,
            non_spec_token_indx_=non_spec_token_indx,
            non_spec_state_indices_tensor_=non_spec_state_indices_tensor,
            spec_query_start_loc_=spec_query_start_loc,
            spec_token_indx_=spec_token_indx,
            spec_state_indices_tensor_=spec_state_indices_tensor,
            num_accepted_tokens_=num_accepted_tokens,
            num_actual_tokens_=num_actual_tokens,
        )"""

    if old not in text:
        raise RuntimeError("anchor not found: _gdn_attention_core_xpu_impl call")
    text = text.replace(old, new, 1)

    # module-level helper, inserted right before the first top-level function
    # (idempotent via guard). Uses a local `import os` since _xpu_ops.py may
    # not import `os` at module level.
    helper = """
def _b70_split_mixed_gdn() -> bool:
    import os
    if os.environ.get("VLLM_TARGET_DEVICE") != "xpu":
        return False
    v = os.environ.get("B70_SPLIT_MIXED_GDN", "1")
    return v not in ("0", "false", "False", "off")
"""
    if "def _b70_split_mixed_gdn" not in text:
        # Insert at module level, before the first top-level function.
        idx = text.find("\ndef ")
        if idx == -1:
            raise RuntimeError("no insert point for _b70_split_mixed_gdn")
        text = text[:idx] + helper + "\n" + text[idx:]
    return text


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
    )
    t = open(p).read()
    open(p, "w").write(patch_text(t))
    print("patched", p)
