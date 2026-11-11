#!/usr/bin/env python3
"""Growing-context agent test: N agents start at the same moment; each runs 4 calls on one conversation whose prompt grows
~90K -> ~120K -> ~150K -> ~200K tokens (each call = previous turns + a new block of records + a question), like an agent that
keeps reading. Every block hides vault-code needles; each answer must list ALL codes seen so far, then a ~150-word summary.
Usage: grow4.py [agents=4] [label]. Output: $OUT (default /opt/ds41/evidence/phase8/grow4)/grow4.jsonl + console table."""
import asyncio, json, os, sys, time, random
from pathlib import Path
from openai import AsyncOpenAI
from tokenizers import Tokenizer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; LABEL = sys.argv[2] if len(sys.argv) > 2 else f"{N}x"
OUT = Path(os.environ.get("OUT", "/opt/ds41/evidence/phase8/grow4")); OUT.mkdir(parents=True, exist_ok=True)
KEY = Path("/opt/ds41/v0523/state/api-key").read_text().strip()
TOK = Tokenizer.from_file("/models/DeepSeek-V4.1-Flash-1017/tokenizer.json")
src = open("/opt/ds41/tests/phase7.py").read(); exec(src[src.index("UNITS ="):src.index("def ask(")])
BLOCKS = [(90_000, 4), (30_000, 1), (30_000, 1), (50_000, 1)]  # cumulative ~90K / 120K / 150K / 200K
Q = ("Somewhere in all the maintenance records in this conversation are notes about vault access codes. First list every vault "
     "name with its exact access code, one per line, formatted as `name: code`. Then, after a blank line, write a summary of about "
     "150 words of the most frequent part failures in the newest block of records.")
JL = open(OUT / "grow4.jsonl", "a", buffering=1)

def prep(a):
    base = int(time.time()) % 50000 + a * 1000; docs = []; s = base
    for size, nn in BLOCKS:
        d, nd = build(size, s, n_needles=nn); docs.append((d, nd)); s += nn  # consecutive seeds -> distinct vault names
    return docs

async def agent(a, docs, t_start):
    c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
    msgs = [{"role": "system", "content": "You are a careful fleet-records analyst agent."}]; seen = []; rows = []
    await asyncio.sleep(max(0, t_start - time.time()))
    for k, (doc, nd) in enumerate(docs, 1):
        seen += nd; head = "Maintenance records:\n" if k == 1 else "Additional maintenance records just retrieved:\n"
        msgs.append({"role": "user", "content": head + doc + "\n" + Q})
        t0 = time.perf_counter(); first = None; out = []; usage = None
        s = await c.chat.completions.create(model="deepseek-v4.1-flash", messages=msgs, max_tokens=600, stream=True,
                                            stream_options={"include_usage": True}, extra_body={"chat_template_kwargs": {"thinking": False}})
        async for ch in s:
            if ch.usage: usage = ch.usage
            if ch.choices and ch.choices[0].delta.content:
                if first is None: first = time.perf_counter()
                out.append(ch.choices[0].delta.content)
        end = time.perf_counter(); txt = "".join(out); n = usage.completion_tokens if usage else 0
        found = sum(code.lower() in txt.lower() for _, code in seen)
        r = dict(label=LABEL, agent=a, call=k, prompt_tokens=usage.prompt_tokens if usage else None, out_tokens=n,
                 ttft=round(first - t0, 2) if first else None, decode=round((n - 1) / (end - first), 1) if first and n > 8 else None,
                 total_s=round(end - t0, 2), needles=f"{found}/{len(seen)}", ok=found == len(seen), started=round(t0, 1), codes=[c for _, c in seen], text=txt)
        JL.write(json.dumps(r) + "\n"); rows.append(r)
        print(f"  agent {a} call {k}: prompt {r['prompt_tokens']:>7,}  TTFT {r['ttft']:>6}s  decode {r['decode']} tok/s  "
              f"out {n}  total {r['total_s']}s  needles {r['needles']}", flush=True)
        msgs.append({"role": "assistant", "content": txt})
    return rows

async def main():
    print(f"################ grow4 [{LABEL}]: {N} agent(s) at once, calls ~90K/120K/150K/200K  {time.strftime('%T')}", flush=True)
    docs = await asyncio.gather(*[asyncio.to_thread(prep, a) for a in range(N)])
    t_start = time.time() + 1; T0 = time.perf_counter()
    res = await asyncio.gather(*[agent(a, docs[a], t_start) for a in range(N)])
    wall = time.perf_counter() - T0; rows = [r for rs in res for r in rs]
    print(f"  per call (median over agents):")
    for k in range(1, 5):
        rs = sorted((r for r in rows if r["call"] == k), key=lambda r: r["ttft"] or 0)
        m = rs[len(rs) // 2]
        print(f"    call {k}: prompt ~{m['prompt_tokens']:,}  TTFT median {m['ttft']}s (min {rs[0]['ttft']}, max {rs[-1]['ttft']})  "
              f"decode median {sorted(r['decode'] or 0 for r in rs)[len(rs)//2]} tok/s  needles ok {sum(r['ok'] for r in rs)}/{len(rs)}")
    print(f"  wall time all agents done: {wall:.1f}s; all needles ok: {all(r['ok'] for r in rows)}  GROW4_DONE", flush=True)
asyncio.run(main())
