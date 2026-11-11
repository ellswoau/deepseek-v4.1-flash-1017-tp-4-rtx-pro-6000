#!/usr/bin/env python3
"""Mixed long + short decode: does one very long request slow everyone else down?
For each size: prime the long prompt once (cold), then resend it cached with ignore_eos (2,500 out). As soon as its first
token arrives, start 8 short streams (~2K prompt, 600 out, ignore_eos). Report short-stream per-stream decode tok/s,
their total, and the long request decode tok/s. Also an 8-short-streams-alone baseline. Usage: mixed_long.py <label> [sizes]"""
import asyncio, json, os, random, sys, time
from pathlib import Path
from openai import AsyncOpenAI
from tokenizers import Tokenizer
LABEL = sys.argv[1]; SIZES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "250000,1000000").split(",")]
OUT = Path("/opt/ds41/evidence/phase8/fix5"); OUT.mkdir(parents=True, exist_ok=True)
KEY = Path("/opt/ds41/v0523/state/api-key").read_text().strip()
TOK = Tokenizer.from_file("/models/DeepSeek-V4.1-Flash-1017/tokenizer.json")
src = open("/opt/ds41/tests/phase7.py").read(); exec(src[src.index("UNITS ="):src.index("def ask(")])
c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
JL = open(OUT / "mixed_long.jsonl", "a", buffering=1)
async def stream(msgs, max_tokens, first_evt=None):
    t0 = time.perf_counter(); first = None; n = 0
    s = await c.chat.completions.create(model="deepseek-v4.1-flash", messages=msgs, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, extra_body={"ignore_eos": True, "chat_template_kwargs": {"thinking": False}})
    async for ch in s:
        if ch.usage: n = ch.usage.completion_tokens
        if ch.choices and ch.choices[0].delta.content and first is None:
            first = time.perf_counter()
            if first_evt: first_evt.set()
    end = time.perf_counter()
    return dict(ttft=round(first - t0, 2), decode=round((n - 1) / (end - first), 1), out=n, end=end, first=first)
SHORT = [[{"role": "user", "content": build(2000, 5000 + i, n_needles=0)[0] + "\nWrite a long, detailed maintenance report about these records."}] for i in range(8)]
async def shorts():
    rs = await asyncio.gather(*[stream(m, 600) for m in SHORT]); return rs
async def main():
    base = await shorts()
    r = dict(label=LABEL, case="shorts_alone", per_stream=sorted(x["decode"] for x in base)[4], total=round(sum(x["out"] for x in base) / (max(x["end"] for x in base) - min(x["first"] for x in base)), 1))
    print(r, flush=True); JL.write(json.dumps(r) + "\n")
    for size in SIZES:
        doc, nd = build(size, 777 + size // 1000)
        msgs = [{"role": "system", "content": "You are a careful records analyst."}, {"role": "user", "content": doc + "\nList every vault code, then write a very long analysis of the records."}]
        p = await stream(msgs, 8)  # prime (cold prefill) so the timed run is a cached resend
        evt = asyncio.Event(); lt = asyncio.create_task(stream(msgs, 2500, evt)); await evt.wait()
        sh = await shorts(); lg = await lt
        r = dict(label=LABEL, case=f"shorts_during_{size//1000}K_decode", prime_ttft=p["ttft"], long_ttft=lg["ttft"], long_decode=lg["decode"],
                 per_stream=sorted(x["decode"] for x in sh)[4], per_stream_min=min(x["decode"] for x in sh),
                 total=round(sum(x["out"] for x in sh) / (max(x["end"] for x in sh) - min(x["first"] for x in sh)), 1), short_ttft_med=sorted(x["ttft"] for x in sh)[4])
        print(r, flush=True); JL.write(json.dumps(r) + "\n")
asyncio.run(main()); print("MIXED_DONE", flush=True)
