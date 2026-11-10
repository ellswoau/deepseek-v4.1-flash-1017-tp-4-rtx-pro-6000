#!/usr/bin/env python3
"""Build (1) the after-fix performance CSV and (2) a before/after comparison CSV.

Inputs (sglang.benchmark.serving logs produced by p5_sweep.sh / p8_sweep.sh):
  --sybil   Sybil build sweep (engine cap 32; rows C<=16 used)
  --nothp   v0.5.23 sweep before the Engram huge-page fix (C=1, 8 + cold 32K)
  --thp     v0.5.23 sweep after the fix (C=1..16 + cold 32K)
"""
import argparse, csv, re

def parse(path):
    rows, cur = [], None
    for line in open(path):
        m = re.match(r"#+ (steady decode|agent-like|cold prefill 32K)(?: C=(\d+))?", line)
        if m:
            if cur: rows.append(cur)
            kind = {"steady decode": "steady", "agent-like": "agent", "cold prefill 32K": "cold32k"}[m.group(1)]
            cur = {"kind": kind, "C": int(m.group(2) or 1)}
            continue
        if cur is None or ":" not in line: continue
        k, v = (s.strip() for s in line.split(":", 1))
        try: f = float(v)
        except ValueError: continue
        key = {"Successful requests": "ok", "Benchmark duration (s)": "duration_s", "Output token throughput (tok/s)": "total_tok_s",
               "Peak output token throughput (tok/s)": "peak_tok_s", "Accept length": "accept_len", "Mean TTFT (ms)": "mean_ttft_ms",
               "Median TTFT (ms)": "median_ttft_ms", "P99 TTFT (ms)": "p99_ttft_ms", "Median TPOT (ms)": "tpot_ms",
               "Input token throughput (tok/s)": "input_tok_s"}.get(k)
        if key: cur[key] = f
    if cur: rows.append(cur)
    # cold-prefill block in the no-THP log was appended without a C header; label it
    return rows

WORK = {"steady": ("random 1024 in / 2048 out, ignore_eos", 1024, 2048),
        "agent": ("shared ~30K system prompt + 2K unique question", 32000, 512),
        "cold32k": ("3 unique random 32K prompts, 16 out", 32768, 16)}
NAME = {"steady": "Steady decode sweep", "agent": "Agent-shaped sweep", "cold32k": "Cold 32K prefill"}

def per_stream(r): return round(1000 / r["tpot_ms"], 1) if r.get("tpot_ms") else None
def s(v, nd=2): return "" if v is None else round(v, nd)
def ms(r, k): return "" if r.get(k) is None else round(r[k] / 1000, 3)

ap = argparse.ArgumentParser()
for a in ("sybil", "nothp", "thp", "out_after", "out_compare"): ap.add_argument("--" + a, required=True)
a = ap.parse_args()
sy = {(r["kind"], r["C"]): r for r in parse(a.sybil) if r["C"] <= 16}
no = {(r["kind"], r["C"]): r for r in parse(a.nothp)}
th = {(r["kind"], r["C"]): r for r in parse(a.thp)}

# 1. after-fix CSV (same column style as ds41-performance-tests-2026-11-10.csv)
cols = ["Test", "Engine", "Engine cap", "Concurrency", "Input tokens", "Output tokens", "Requests OK", "Requests total",
        "Prefill tok/s", "Median TTFT s", "P99 TTFT s", "Per-stream decode tok/s", "Total decode tok/s", "Peak output tok/s", "DSpark accept length"]
with open(a.out_after, "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for (kind, C), r in sorted(th.items(), key=lambda x: (["steady", "agent", "cold32k"].index(x[0][0]), x[0][1])):
        _, inp, out = WORK[kind]
        w.writerow([NAME[kind], "SGLang v0.5.23 + Sybil SM120 port + HiCache + Engram THP", 16, C, inp, out,
                    int(r.get("ok", 0)), 3 if kind == "cold32k" else 4 * C, s(r.get("input_tok_s"), 1), ms(r, "median_ttft_ms"),
                    ms(r, "p99_ttft_ms"), s(per_stream(r), 1), s(r.get("total_tok_s"), 1), s(r.get("peak_tok_s"), 1), s(r.get("accept_len"))])

# 2. before/after comparison
def pct(new, old): return "" if not new or not old else f"{(new - old) / old * 100:+.1f}%"
ccols = ["Test", "Concurrency", "Metric",
         "Sybil build (prod before)", "v0.5.23 no THP (before fix)", "v0.5.23 + THP (after fix)",
         "After vs no-THP", "After vs Sybil"]
metrics = [("Total output tok/s", lambda r: r.get("total_tok_s"), True),
           ("Per-stream decode tok/s", per_stream, True),
           ("Median TTFT s", lambda r: r.get("median_ttft_ms") and r["median_ttft_ms"] / 1000, False),
           ("P99 TTFT s", lambda r: r.get("p99_ttft_ms") and r["p99_ttft_ms"] / 1000, False),
           ("DSpark accept length", lambda r: r.get("accept_len"), True)]
with open(a.out_compare, "w", newline="") as f:
    w = csv.writer(f); w.writerow(ccols)
    for kind in ("steady", "agent"):
        for C in (1, 2, 4, 8, 12, 16):
            for label, fn, _ in metrics:
                vs, vn, vt = (fn(d[(kind, C)]) if (kind, C) in d else None for d in (sy, no, th))
                if vt is None and vs is None: continue
                w.writerow([NAME[kind], C, label, s(vs, 3), s(vn, 3), s(vt, 3), pct(vt, vn), pct(vt, vs)])
    # cold prefill: Sybil value comes from its own matrix.py (32K C=1 TTFT 4.1 s), different client
    t, n = th.get(("cold32k", 1), {}), no.get(("cold32k", 1), {})
    for label, k in (("Median TTFT s", "median_ttft_ms"), ("Mean TTFT s", "mean_ttft_ms")):
        vn = n.get(k) and n[k] / 1000; vt = t.get(k) and t[k] / 1000
        w.writerow([NAME["cold32k"], 1, label, "4.1 (matrix.py, different client)" if label == "Median TTFT s" else "",
                    s(vn, 3), s(vt, 3), pct(vt, vn), ""])
print("wrote", a.out_after, "and", a.out_compare)
