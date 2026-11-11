#!/usr/bin/env python3
"""DSpark block-size benchmark with real-text prompts (random tokens distort draft acceptance).
Per concurrency C: 2*C requests (min 4) drawn round-robin from a mixed prompt set, model default
sampling, up to 1024 output tokens, thinking off. Reports per-stream decode tok/s, total output
tok/s and TTFT; acceptance length is parsed from the engine log for the same time window.
Usage: dspark_bench.py <label> [C list]   (prints one JSON line per C, appends to $DS41_OUT/dspark.jsonl)"""
import asyncio, json, os, re, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from openai import AsyncOpenAI

LABEL = sys.argv[1]
CS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4,8,16").split(",")]
KEY = Path("/opt/ds41/v0523/state/api-key").read_text().strip()
OUT = Path(os.environ.get("DS41_OUT", "/opt/ds41/evidence/phase8/dspark")); OUT.mkdir(parents=True, exist_ok=True)
DOC = Path("/opt/ds41/tests/chat.py").read_text()[:6000]
PROMPTS = [
    ("prose", "Write a detailed maintenance guide for a fleet of Class 8 tractors covering air brakes, DPF/SCR aftertreatment, cooling, electrical and tires, ending with a weekly inspection checklist."),
    ("code", "Write a complete, well-commented Python module that implements an LRU cache with TTL expiry, thread safety, and statistics, plus pytest unit tests for it."),
    ("json", "Generate a JSON array of 25 realistic heavy-duty truck parts. Each object needs part_number, description, category, unit_cost, supplier, weight_lbs and compatible_models (array). Output only JSON."),
    ("summary", "Explain in detail what this Python script does, section by section, then list five improvements:\n\n" + DOC),
    ("sql", "Design a PostgreSQL schema for a multi-branch parts distributor (branches, parts, inventory, customers, orders, order lines, transfers). Give full CREATE TABLE statements with constraints and indexes, then five example reporting queries."),
    ("email", "Draft a professional customer-facing announcement explaining a new online parts-ordering portal: features, how to sign up, FAQ with eight questions and answers, and support contacts."),
    ("bash", "Write a robust bash script that backs up a set of PostgreSQL databases, compresses and encrypts the dumps, rotates old backups by age, logs every step, and exits non-zero on failure. Explain each section after the script."),
    ("analysis", "Compare leasing versus buying for a 40-truck regional fleet over seven years. Walk through the cost components, tax considerations, maintenance, residual value and risk, with a worked numeric example."),
]

def log_window_accept(since_iso):
    out = subprocess.run(["docker", "logs", "v0523-ds41b-1", "--since", since_iso], capture_output=True, text=True)
    lines = (out.stdout + out.stderr).splitlines()
    acc = []
    for l in lines:
        if "TP0 EP0] Decode batch" not in l: continue
        m = re.search(r"#running-req: (\d+).*accept len: ([\d.]+)", l)
        if m: acc.append((int(m.group(1)), float(m.group(2))))
    if not acc: return None, None
    w = sum(r for r, _ in acc)
    return round(statistics.mean(a for _, a in acc), 2), round(sum(r * a for r, a in acc) / w, 2)

async def one(c, i):
    kind, p = PROMPTS[i % len(PROMPTS)]
    t0 = time.perf_counter(); first = None; n = 0
    s = await c.chat.completions.create(model="deepseek-v4.1-flash", stream=True, stream_options={"include_usage": True},
          max_tokens=1024, messages=[{"role": "user", "content": p}],
          extra_body={"chat_template_kwargs": {"thinking": False}})
    async for ch in s:
        if ch.usage: n = ch.usage.completion_tokens
        if ch.choices and ch.choices[0].delta.content and first is None: first = time.perf_counter()
    end = time.perf_counter()
    return dict(kind=kind, tokens=n, ttft=first - t0, decode=(n - 1) / (end - first) if first and n > 1 else None, end=end)

async def run_c(C):
    c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=900)
    await one(c, 0)  # warm-up
    since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    N = max(4, 2 * C); sem = asyncio.Semaphore(C); t0 = time.perf_counter()
    async def guarded(i):
        async with sem: return await one(c, i)
    res = await asyncio.gather(*[guarded(i) for i in range(N)])
    dur = time.perf_counter() - t0
    acc_mean, acc_w = log_window_accept(since)
    d = [r["decode"] for r in res if r["decode"]]
    row = dict(label=LABEL, C=C, requests=N, out_tokens=sum(r["tokens"] for r in res),
               decode_per_stream_median=round(statistics.median(d), 1), decode_per_stream_min=round(min(d), 1),
               total_out_tok_s=round(sum(r["tokens"] for r in res) / dur, 1),
               ttft_median_s=round(statistics.median(r["ttft"] for r in res), 2),
               accept_len_mean=acc_mean, accept_len_reqweighted=acc_w,
               by_kind={k: round(statistics.median([r["decode"] for r in res if r["kind"] == k and r["decode"]]), 1)
                        for k in sorted({r["kind"] for r in res})})
    print(json.dumps(row), flush=True)
    with open(OUT / "dspark.jsonl", "a") as f: f.write(json.dumps(row) + "\n")

for C in CS: asyncio.run(run_c(C))
