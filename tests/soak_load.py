#!/usr/bin/env python3
"""24 h soak load for the production engine (127.0.0.1:8010).

Traffic mix (all through the OpenAI-compatible API, the way agents call it):
  * agent sessions: one of 6 shared ~20K-token system prompts + 3-8 user turns, answers appended to the history
    (exercises prefix cache and HiCache). Some turns carry a tool definition and must produce a tool call;
    some are arithmetic with a checkable answer.
  * long-document worker: every ~8 min a fresh 60K-300K token document with 4 vault-code needles, asked twice
    (cold, then cached); both answers must contain all 4 codes.
  * huge-document worker: first at +30 min, then every 2 h, alternating ~800K and ~900K tokens, cold then cached,
    same needle check (needs --context-length 1048576).
Active agent workers follow a 4-hour sine between 2 and 12 so the engine sees ramps, peaks and quiet periods.
Every request is written to requests.jsonl; a summary line is printed every 15 minutes.
Usage: soak_load.py [hours] (default 24). Output dir: $SOAK_OUT (default /opt/ds41/evidence/soak)."""
import asyncio, json, math, os, random, statistics, sys, time
from pathlib import Path
from openai import AsyncOpenAI
from tokenizers import Tokenizer

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
OUT = Path(os.environ.get("SOAK_OUT", "/opt/ds41/evidence/soak")); OUT.mkdir(parents=True, exist_ok=True)
KEY = Path("/opt/ds41/v0523/state/api-key").read_text().strip()
TOK = Tokenizer.from_file("/models/DeepSeek-V4.1-Flash-1017/tokenizer.json")
src = open("/opt/ds41/tests/phase7.py").read()
exec(src[src.index("UNITS ="):src.index("def ask(")])  # entry(), build(), QUESTION
MODEL = "deepseek-v4.1-flash"; MAX_AGENTS = 12; T0 = time.time(); END = T0 + HOURS * 3600
REQ = open(OUT / "requests.jsonl", "a", buffering=1)
STATS = []  # rows since last summary

SYSTEMS = []
for k in range(6):
    doc, _ = build(20000, 900 + k, n_needles=0)
    SYSTEMS.append(f"You are fleet-ops assistant #{k}. Reference maintenance records follow; use them when relevant.\n\n" + doc)

TOOLS = [{"type": "function", "function": {"name": "check_inventory", "description": "Look up on-hand quantity of a part at a branch.",
          "parameters": {"type": "object", "properties": {"part_number": {"type": "string"}, "branch": {"type": "string"}},
                         "required": ["part_number", "branch"]}}}]
TOPICS = ["air brake adjustment", "DPF regeneration", "SCR/DEF faults", "cooling system flush", "alternator testing",
          "tire wear patterns", "fifth-wheel maintenance", "clutch adjustment", "PM scheduling", "warranty claims"]
BRANCHES = ["North", "South", "East", "West", "Central"]

def active_agents(t):
    return round(2 + 10 * (0.5 + 0.5 * math.sin(2 * math.pi * (t - T0) / (4 * 3600) - math.pi / 2)))

def record(**row):
    row["ts"] = round(time.time(), 1); REQ.write(json.dumps(row) + "\n"); STATS.append(row)

async def call(c, kind, msgs, max_tokens, tools=None, check=None):
    t0 = time.perf_counter(); first = None; text = []; calls = []; usage = None
    try:
        s = await c.chat.completions.create(model=MODEL, messages=msgs, stream=True, stream_options={"include_usage": True},
                                            max_tokens=max_tokens, tools=tools,
                                            extra_body={"chat_template_kwargs": {"thinking": False}})
        async for ch in s:
            if ch.usage: usage = ch.usage
            if not ch.choices: continue
            d = ch.choices[0].delta
            if (d.content or d.tool_calls) and first is None: first = time.perf_counter()
            if d.content: text.append(d.content)
            if d.tool_calls: calls.extend(tc.function.name for tc in d.tool_calls if tc.function and tc.function.name)
        end = time.perf_counter(); out = "".join(text)
        n = usage.completion_tokens if usage else 0
        ok = check(out, calls) if check else bool(out.strip() or calls)
        record(kind=kind, ok=ok, prompt_tokens=usage.prompt_tokens if usage else None, out_tokens=n,
               ttft=round(first - t0, 3) if first else None,
               decode=round((n - 1) / (end - first), 1) if first and n > 8 and end > first else None)
        return out
    except Exception as e:
        record(kind=kind, ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
        await asyncio.sleep(5); return None

async def agent(i):
    c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=1800)
    rng = random.Random(i * 7919 + int(T0))
    while time.time() < END:
        if i >= active_agents(time.time()):
            await asyncio.sleep(20); continue
        msgs = [{"role": "system", "content": rng.choice(SYSTEMS)}]
        for turn in range(rng.randint(3, 8)):
            if time.time() >= END: break
            r = rng.random()
            if r < 0.15:
                part = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"; br = rng.choice(BRANCHES)
                msgs.append({"role": "user", "content": f"How many of part {part} are on hand at the {br} branch?"})
                await call(c, "tool", msgs, 300, TOOLS, check=lambda o, k: "check_inventory" in k)
                msgs.pop(); continue
            if r < 0.25:
                a, b = rng.randint(100, 999), rng.randint(100, 999)
                msgs.append({"role": "user", "content": f"Quick check: what is {a} * {b}? Reply with just the number."})
                out = await call(c, "math", msgs, 50, check=lambda o, k, v=a * b: str(v) in o.replace(",", ""))
            else:
                msgs.append({"role": "user", "content": f"Turn {turn}: give a practical, detailed explanation of {rng.choice(TOPICS)} "
                                                        f"for a technician, referencing any relevant records above."})
                out = await call(c, "agent", msgs, rng.choice([300, 500, 800]))
            if out is None: break
            msgs.append({"role": "assistant", "content": out})
            await asyncio.sleep(rng.uniform(1, 8))

async def longdoc():
    c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
    rng = random.Random(int(T0)); n = 0
    while time.time() < END:
        size = rng.choice([60_000, 100_000, 150_000, 200_000, 300_000]); n += 1
        doc, needles = await asyncio.to_thread(build, size, int(time.time()) % 100000)
        msgs = [{"role": "system", "content": "You are a careful records analyst."}, {"role": "user", "content": doc + "\n" + QUESTION}]
        chk = lambda o, k, nd=needles: all(code.lower() in o.lower() for _, code in nd)
        for kind in ("long_cold", "long_cached"):
            await call(c, kind, msgs, 200, check=chk)
        await asyncio.sleep(rng.uniform(360, 600))

async def huge():
    c = AsyncOpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
    await asyncio.sleep(1800); n = 0
    while time.time() < END - 1800:
        size = (800_000, 900_000)[n % 2]; n += 1
        doc, needles = await asyncio.to_thread(build, size, int(time.time()) % 100000)
        msgs = [{"role": "system", "content": "You are a careful records analyst."}, {"role": "user", "content": doc + "\n" + QUESTION}]
        chk = lambda o, k, nd=needles: all(code.lower() in o.lower() for _, code in nd)
        for kind in ("huge_cold", "huge_cached"):
            await call(c, kind, msgs, 200, check=chk)
        print(f"[{time.strftime('%H:%M')}] huge doc #{n} (~{size // 1000}K) done: " +
              ", ".join(f"{r['kind']} ok={r['ok']} ttft={r.get('ttft')} prompt={r.get('prompt_tokens')}" for r in STATS if r["kind"].startswith("huge"))[-400:], flush=True)
        await asyncio.sleep(7200)

def pct(v, p):
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(p * len(v)))], 2) if v else None

async def summarizer():
    while time.time() < END + 60:
        await asyncio.sleep(900)
        rows = STATS[:]; STATS.clear()
        by = {}
        for r in rows: by.setdefault(r["kind"], []).append(r)
        parts = []
        for k in ("agent", "tool", "math", "long_cold", "long_cached", "huge_cold", "huge_cached"):
            rs = by.get(k, [])
            if not rs: continue
            bad = sum(not r["ok"] for r in rs)
            parts.append(f"{k} {len(rs)}" + (f" FAIL {bad}" if bad else "") +
                         f" ttft p50 {pct([r.get('ttft') for r in rs], .5)} p95 {pct([r.get('ttft') for r in rs], .95)}" +
                         (f" dec {pct([r.get('decode') for r in rs], .5)}" if k == "agent" else ""))
        print(f"[{time.strftime('%H:%M')} +{(time.time() - T0) / 3600:.1f}h agents={active_agents(time.time())}] " + " | ".join(parts), flush=True)

async def main():
    print(f"soak start {time.strftime('%F %T')} for {HOURS} h -> {OUT}", flush=True)
    await asyncio.gather(summarizer(), longdoc(), huge(), *[agent(i) for i in range(MAX_AGENTS)])
    print(f"SOAK_LOAD_DONE {time.strftime('%F %T')}", flush=True)

asyncio.run(main())
