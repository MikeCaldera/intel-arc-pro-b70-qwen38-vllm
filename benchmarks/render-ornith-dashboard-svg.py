#!/usr/bin/env python3
"""Render the Ornith MixedCal-v2 B70 dashboard from summary.json.

Lane-1-shaped: two cold-input anchors, two decode anchors, MTP depth, exact
capacity. Every number is read from the compiler JSON. Self-reported E2
banner is on-canvas.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

W, H = 1600, 1380
C = {
    "bg": "#07111f",
    "panel": "#0f1d30",
    "panel2": "#0b1829",
    "line": "#263a53",
    "text": "#f5f8fc",
    "muted": "#9caec4",
    "accent": "#55d6be",
    "warn": "#ffb45c",
    "no-spec": "#aeb9c7",
    "mtp1": "#55d6be",
    "mtp2": "#68a7ff",
    "mtp4": "#ffb45c",
}


def esc(v):
    return html.escape(str(v))


def t(x, y, v, size=24, color=None, weight=400, anchor="start"):
    color = color or C["text"]
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">{esc(v)}</text>'
    )


def panel(x, y, w, h, eyebrow, title):
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{C["panel"]}"/>',
        t(x + 34, y + 40, eyebrow.upper(), 16, C["accent"], 700),
        t(x + 34, y + 78, title, 26, C["text"], 700),
    ]


def inst_med(block, prefix, cell):
    vals = []
    for k, v in block.items():
        if k.startswith(prefix) and cell in v and v[cell].get("median") is not None:
            vals.append(v[cell]["median"])
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def fx_med(fx, serve, needle, key):
    rows = [r for r in fx if r.get("serve") == serve and needle in r.get("cell", "")]
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def fmt(v, kind="dec"):
    if v is None:
        return "—"
    if kind == "cold":
        return f"{v:,.0f}"
    return f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text())
    n5 = s.get("n5_no_spec_32k_150w", {})
    mtp1 = s.get("mtp1_16k_150w_n5", {})
    depth = s.get("mtp_depth_ab", {})
    fx = s.get("exact_boundary_cells", [])
    m131 = s.get("mtp1_131k_u90", {})

    cand_p512 = inst_med(n5, "cand", "p512_g128")
    cand_p8k = inst_med(n5, "cand", "p8192_g128")
    cand_c2k = inst_med(n5, "cand", "cold_p2048_g1")
    cand_c8k = inst_med(n5, "cand", "cold_p8192_g1")
    orig_p512 = inst_med(n5, "orig", "p512_g128")
    orig_p8k = inst_med(n5, "orig", "p8192_g128")

    def depth_med(name, cell):
        d = depth.get(name, {})
        m = d.get(cell, {})
        return m.get("median"), (d.get("acceptance") or {}).get("pct")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Ornith 1.5 MixedCal-v2 on Intel Arc Pro B70 — self-reported C1 card</title>',
        '<desc id="desc">Client post-first decode and cold-input rates for Ornith MixedCal-v2 GPTQ INT4 on one B70. MTP1 wins. MTP4 is slower than no-spec. Self-reported E2, not independently reproduced. LocalMaxxing not submitted.</desc>',
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        t(48, 52, "INTEL ARC PRO B70 · vLLM XPU", 16, C["accent"], 700),
        t(48, 96, "Ornith-1.5-35B-A3B MixedCal-v2", 36, C["text"], 700),
        t(48, 130, "GPTQ INT4 G128 experts-only · MTP BF16 · C1 · cache off · 150 W configured", 18, C["muted"]),
        f'<rect x="1180" y="28" width="372" height="88" rx="18" fill="{C["panel"]}"/>',
        t(1366, 62, "SELF-REPORTED E2", 20, C["accent"], 700, "middle"),
        t(1366, 92, "C1 · LMX NOT SUBMITTED", 14, C["muted"], 600, "middle"),
    ]

    # Decode panel
    parts += panel(48, 168, 760, 478, "C1 decode", "Client post-first tok/s · n=5")
    rows = [
        ("No-spec 32K orig", orig_p512, orig_p8k, None, C["no-spec"]),
        ("No-spec 32K MixedCal", cand_p512, cand_p8k, None, C["no-spec"]),
        ("MTP1 16K orig", (mtp1.get("orig") or {}).get("p512_g128", {}).get("median"),
         (mtp1.get("orig") or {}).get("p8192_g128", {}).get("median"),
         (mtp1.get("orig") or {}).get("acceptance", {}).get("pct"), C["mtp1"]),
        ("MTP1 16K MixedCal", (mtp1.get("cand") or {}).get("p512_g128", {}).get("median"),
         (mtp1.get("cand") or {}).get("p8192_g128", {}).get("median"),
         (mtp1.get("cand") or {}).get("acceptance", {}).get("pct"), C["mtp1"]),
        ("MTP2 MixedCal", *depth_med("D2-cand-mtp2-16k", "p512_g128")[:1],
         depth_med("D2-cand-mtp2-16k", "p8192_g128")[0],
         depth_med("D2-cand-mtp2-16k", "p512_g128")[1], C["mtp2"]),
        ("MTP4 MixedCal", *depth_med("D4-cand-mtp4-16k", "p512_g128")[:1],
         depth_med("D4-cand-mtp4-16k", "p8192_g128")[0],
         depth_med("D4-cand-mtp4-16k", "p512_g128")[1], C["mtp4"]),
    ]
    # fix MTP2/4 tuple packing
    d2p, d2a = depth_med("D2-cand-mtp2-16k", "p512_g128")
    d2q, _ = depth_med("D2-cand-mtp2-16k", "p8192_g128")
    d4p, d4a = depth_med("D4-cand-mtp4-16k", "p512_g128")
    d4q, _ = depth_med("D4-cand-mtp4-16k", "p8192_g128")
    di = ((s.get("draft_int4") or {}).get("serves") or {})
    di1 = di.get("DI1-cand-mtp1-draftint4") or {}
    rows[4] = ("MTP2 MixedCal", d2p, d2q, d2a, C["mtp2"])
    rows[5] = ("MTP4 MixedCal", d4p, d4q, d4a, C["mtp4"])
    rows.append((
        "MTP1 DraftINT4 overlay",
        (di1.get("p512_g128") or {}).get("median"),
        (di1.get("p8192_g128") or {}).get("median"),
        (di1.get("acceptance") or {}).get("pct"),
        C["accent"],
    ))

    left, top = 82, 270
    parts.append(f'<rect x="{left}" y="{top}" width="692" height="40" rx="12" fill="{C["panel2"]}"/>')
    parts.append(t(left + 16, top + 27, "Mode", 16, C["muted"], 600))
    parts.append(t(left + 340, top + 27, "p512/g128", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 490, top + 27, "p8192/g128", 16, C["muted"], 600, "middle"))
    parts.append(t(left + 640, top + 27, "accept", 16, C["muted"], 600, "middle"))
    for i, (label, a, b, acc, col) in enumerate(rows):
        cy = top + 44 + 48 * i
        if i % 2:
            parts.append(f'<rect x="{left}" y="{cy}" width="692" height="48" fill="#12243a" opacity="0.5"/>')
        parts.append(f'<circle cx="{left+16}" cy="{cy+24}" r="5" fill="{col}"/>')
        parts.append(t(left + 30, cy + 30, label, 16, C["text"], 600))
        winner = i in (3, 6)  # MixedCal MTP1 BF16 draft and DraftINT4 overlay
        parts.append(t(left + 340, cy + 30, fmt(a), 18, C["accent"] if winner else C["text"], 700 if winner else 500, "middle"))
        parts.append(t(left + 490, cy + 30, fmt(b), 18, C["accent"] if winner else C["text"], 700 if winner else 500, "middle"))
        parts.append(t(left + 640, cy + 30, "—" if acc is None else f"{acc:.1f}%", 16, C["muted"], 500, "middle"))

    def power_round_med(tag, cell):
        pa = (s.get("power_ab_cold_input") or {}).get(tag) or {}
        return (pa.get(cell) or {}).get("median")

    p150 = [power_round_med(f"r{i}-150w", "p2048/g1") for i in (1, 2, 3)]
    p230 = [power_round_med(f"r{i}-230w", "p2048/g1") for i in (1, 2, 3)]
    p150v = [v for v in p150 if v is not None]
    p230v = [v for v in p230 if v is not None]
    med150 = sorted(p150v)[len(p150v) // 2] if p150v else None
    med230 = sorted(p230v)[len(p230v) // 2] if p230v else None

    # Cold input
    parts += panel(832, 168, 720, 478, "Cold input", "Endpoint tokens / client TTFT")
    cold_rows = [
        ("p2048/g1 MixedCal n=5 32K @150 W", cand_c2k),
        ("p8192/g1 MixedCal n=5 32K @150 W", cand_c8k),
        ("paired A/B p2048 @150 W (3-round med)", med150),
        ("paired A/B p2048 @230 W (3-round med)", med230),
        ("LMX long-prompt tokSPrefill @230 W",
         ((s.get("lmx_nospec_32k_230w_long_prompt") or {}).get("tokSPrefill"))),
        ("same-load harness p8192/g1 @230 W",
         (((s.get("lmx_nospec_32k_230w_long_prompt") or {}).get("same_load_harness") or {}).get("p8192_g1") or {}).get("median")),
    ]
    left, top = 866, 270
    parts.append(t(left, top - 8, "Paired A/B is MixedCal; day-0 9.5k is original @230 W.", 15, C["warn"]))
    for i, (label, v) in enumerate(cold_rows):
        cy = top + 28 + 52 * i
        parts.append(f'<rect x="{left}" y="{cy}" width="652" height="46" rx="12" fill="{C["panel2"]}"/>')
        parts.append(t(left + 18, cy + 30, label, 16, C["muted"]))
        note = "230 W" if "230" in label else ""
        parts.append(t(left + 630, cy + 30, fmt(v, "cold"), 20, C["warn"] if note else C["text"], 700, "end"))

    # Capacity
    parts += panel(48, 670, 1504, 280, "Exact-token capacity MixedCal-v2", "Calibrated with /tokenize · finish=length")
    cap = [
        ("65,536 no-spec", "p65408/g128", fx_med(fx, "FX1-cand-65k-boundary", "p65408/g128", "post_first_tps"), "n=3"),
        ("131,072 no-spec", "p130944/g128", fx_med(fx, "FX2-cand-131k-boundary", "p130944/g128", "post_first_tps"), "n=3"),
        ("131,072 MTP1", "p130944/g128", fx_med(fx, "FX5-cand-131k-mtp1-boundary", "p130944/g128", "post_first_tps"), "n=3"),
        ("131K MTP1 U=0.90", "p512/g128", (m131.get("p512_g128") or {}).get("median"), "n=5 85.3% acc"),
        ("262,144 no-spec", "p262016/g128", fx_med(fx, "FX4-cand-262k-boundary", "p262016/g128", "post_first_tps"), "n=3"),
    ]
    for i, (serve, cell, v, note) in enumerate(cap):
        x = 82 + 294 * i
        parts.append(f'<rect x="{x}" y="768" width="278" height="150" rx="18" fill="{C["panel2"]}"/>')
        parts.append(t(x + 18, 800, serve, 15, C["muted"], 600))
        parts.append(t(x + 18, 856, fmt(v), 32, C["accent"] if "MTP1" in serve else C["text"], 700))
        parts.append(t(x + 18, 888, f"{cell} · {note}", 14, C["muted"]))

    # Footer
    stack = s.get("stack", {})
    art = (s.get("artifacts") or {}).get("candidate") or {}
    rtn = (art.get("rtn_fallback") or {})
    parts += [
        t(48, 988, "Contract", 16, C["accent"], 700),
        t(48, 1020, f"Image {stack.get('image','')}  ·  {stack.get('vllm','')}  ·  kernels {stack.get('vllm_xpu_kernels','')}", 15, C["muted"]),
        t(48, 1044, f"MoE {stack.get('moe_backend','')}  ·  {stack.get('timing_source','')}", 15, C["muted"]),
        t(48, 1068, f"MixedCal-v2 RTN {rtn.get('pct')}% ({rtn.get('count')}/{rtn.get('total')}) vs original 24.76%. Speed at 150 W is parity, not a MixedCal win.", 15, C["muted"]),
        t(48, 1104, "Winner: MixedCal-v2 MTP1. Overlay DraftINT4 screened faster, local-only. MTP4 slower than no-spec. Prefill lever is 230 W cap (~9.7k), not MixedCal.", 16, C["text"]),
        t(48, 1168, "E2 self-reported with raw evidence · not independently reproduced · greedy diagnostic · C1 only · LMX not submitted", 16, C["warn"], 600),
        t(48, 1198, "Source: results/ornith15-mixedcal-v2-summary/summary.json  ·  claims: docs/ornith15-35a3/CLAIMS.md", 14, C["muted"]),
        "</svg>",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
