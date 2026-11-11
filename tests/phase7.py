#!/usr/bin/env python3
"""Phase 7: multi-needle long-context test at ~250K / 500K / 750K / 1M tokens.

For each size: unique prefix, 4 needles ("vault codes") at depths 10/40/70/95%, realistic log filler sized with
the model's own tokenizer. Run alone, then a multi-turn cache test on the 1M conversation (+10K new tokens and one
more needle), then all sizes again with 8 background streams (measures their slowdown).
Key read from file; results -> $DS41_OUT/phase7_results.json.
"""
import json, os, random, threading, time, uuid
from pathlib import Path
from openai import OpenAI
from tokenizers import Tokenizer

KEY = Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip()
OUT = Path(os.environ.get("DS41_OUT", "results/phase7")); OUT.mkdir(parents=True, exist_ok=True)
TOK = Tokenizer.from_file(os.environ.get("DS41_TOKENIZER", os.environ.get("DS41_TOKENIZER", "/models/DeepSeek-V4.1-Flash-1017/tokenizer.json")))
SIZES = [int(x) for x in os.environ.get("SIZES", "250000,500000,750000,1000000").split(",")]
c = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
M = "deepseek-v4.1-flash"; NOTHINK = {"chat_template_kwargs": {"thinking": False}}

UNITS = [f"T-{n:03d}" for n in range(100, 400)]
TECHS = "Alvarez Brooks Chen Dubois Eriksen Fischer Garcia Haddad Ivanova Jensen Kowalski Larsen Moreau Nakamura Okafor Petrov Quinn Rossi Silva Tanaka".split()
PARTS = ("air dryer cartridge|slack adjuster|brake chamber|wheel seal|kingpin bushing|fifth-wheel jaw|DEF pump|"
         "EGR cooler|turbo actuator|alternator|starter solenoid|leaf spring|torque rod|U-joint|clutch release bearing|"
         "fan clutch|water pump|radiator hose|fuel filter|DPF").split("|")
FINDINGS = ["showed uneven wear on the inboard side", "was within spec but noted for the next PM", "leaked under load test",
            "had a cracked housing", "failed the pressure test at 90 psi", "passed after torque correction",
            "was replaced under warranty", "showed corrosion at the mounting flange", "measured 2 mm below minimum",
            "had a loose connector that was re-terminated"]
ACTIONS = ["Ordered replacement from the regional warehouse.", "Customer approved the quote by phone.",
           "Parts on backorder; unit returned to service with restriction.", "Core credit applied to the invoice.",
           "Rechecked after a 20-mile road test; no further issue.", "Escalated to the shop foreman for review.",
           "Logged for fleet trend analysis.", "Scheduled follow-up at next service interval."]
VAULTS = ["cobalt heron", "amber falcon", "silver otter", "crimson lynx", "jade badger", "violet marten", "ochre ibis"]

def entry(rng, i):
    return (f"Record {i:06d} | {rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2026 | unit {rng.choice(UNITS)} | tech {rng.choice(TECHS)}: "
            f"the {rng.choice(PARTS)} {rng.choice(FINDINGS)}; odometer {rng.randint(40_000, 900_000):,} mi. {rng.choice(ACTIONS)}\n")

def build(target_tokens, seed, depths=(0.10, 0.40, 0.70, 0.95), n_needles=4):
    rng = random.Random(seed)
    needles = [(VAULTS[(seed + k) % len(VAULTS)], f"{rng.choice('QXZKV')}{rng.choice('RWMT')}-{rng.randint(10000, 99999)}")
               for k in range(n_needles)]
    parts, total, i = [], 0, 0
    while total < target_tokens:
        e = entry(rng, i); parts.append(e); total += len(TOK.encode(e, add_special_tokens=False).ids); i += 1
    for d, (name, code) in zip(depths, needles):
        parts.insert(int(len(parts) * d), f"Vault note: the access code for the {name} vault is {code}.\n")
    return "".join(parts), needles

QUESTION = ("Somewhere in the maintenance records above are notes about vault access codes. List every vault name with "
            "its exact access code, one per line, formatted as `name: code`. Output nothing else.")

def ask(messages, max_tokens=4096):
    t0 = time.perf_counter(); first = None; text = []; usage = None
    for ch in c.chat.completions.create(model=M, messages=messages, stream=True, stream_options={"include_usage": True},
                                        max_tokens=max_tokens, temperature=0.0, extra_body=NOTHINK):
        if ch.usage: usage = ch.usage
        if ch.choices and ch.choices[0].delta.content:
            if first is None: first = time.perf_counter()
            text.append(ch.choices[0].delta.content)
    end = time.perf_counter(); out = "".join(text)
    n_out = usage.completion_tokens if usage else None
    return dict(prompt_tokens=usage.prompt_tokens if usage else None, completion_tokens=n_out,
                ttft_s=round((first or end) - t0, 2), total_s=round(end - t0, 2),
                decode_tok_s=round(n_out / (end - first), 1) if first and n_out and end > first else None, answer=out)

def score(answer, needles):
    a = answer.lower(); found = [code for name, code in needles if code.lower() in a]
    return len(found), len(needles)

# ---------- background load (8 streams) ----------
bg_events, bg_stop = [], threading.Event()
def bg_worker(k):
    rng = random.Random(1000 + k)
    while not bg_stop.is_set():
        body = "".join(entry(rng, i) for i in range(70))  # ~2K tokens, unique per call
        try:
            s = c.chat.completions.create(model=M, stream=True, max_tokens=512, temperature=1.0, extra_body=NOTHINK,
                                          messages=[{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{body}\nSummarise the main failure trends in 5 bullet points."}])
            for ch in s:
                if ch.choices and ch.choices[0].delta.content: bg_events.append(time.time())
                if bg_stop.is_set(): break
        except Exception as e:
            bg_events.append(("err", time.time(), str(e)[:100]))
def bg_rate(t0, t1):
    n = sum(1 for e in bg_events if isinstance(e, float) and t0 <= e < t1)
    return round(n / max(t1 - t0, 1e-6), 1)  # stream chunks/s ~ tokens/s (one token per chunk, DSpark may batch)

results = {"sizes": SIZES, "alone": [], "cache": None, "with_bg": []}
def run_size(size, mode, seed):
    doc, needles = build(size, seed)
    msgs = [{"role": "system", "content": f"[{uuid.uuid4().hex}] You are a careful records analyst."},
            {"role": "user", "content": doc + "\n" + QUESTION}]
    ts = time.time(); r = ask(msgs); te = time.time()
    f, n = score(r["answer"], needles)
    row = dict(target=size, mode=mode, found=f, needles=n, start=ts, end=te, **{k: v for k, v in r.items() if k != "answer"},
               answer=r["answer"][:400], expected=[f"{a}: {b}" for a, b in needles])
    if mode == "with_bg":
        first_tok = ts + r["ttft_s"]
        row["bg_tok_s_baseline"] = bg_rate(ts - 40, ts)
        row["bg_tok_s_during_prefill"] = bg_rate(ts + 5, first_tok)
        row["bg_tok_s_during_decode"] = bg_rate(first_tok, te)
    print(f"[{mode}] ~{size/1000:.0f}K: prompt {r['prompt_tokens']:,} tok, TTFT {r['ttft_s']}s, decode {r['decode_tok_s']} tok/s, "
          f"needles {f}/{n}" + (f", bg tok/s base {row['bg_tok_s_baseline']} / prefill {row['bg_tok_s_during_prefill']} / decode {row['bg_tok_s_during_decode']}" if mode == "with_bg" else ""), flush=True)
    if f < n: print("   answer:", r["answer"][:300].replace("\n", " | "), " expected:", row["expected"], flush=True)
    return row, msgs, r["answer"]

print("=== alone", flush=True)
last = None
for i, size in enumerate(SIZES):
    row, msgs, ans = run_size(size, "alone", 100 + i); results["alone"].append(row); last = (size, msgs, ans)
    (OUT / "phase7_results.json").write_text(json.dumps(results, indent=1))

print("=== cache: resend the largest conversation + ~10K new tokens", flush=True)
size, msgs, ans = last
extra, extra_needles = build(10_000, 777, depths=(0.5,), n_needles=1)
msgs2 = msgs + [{"role": "assistant", "content": ans},
                {"role": "user", "content": "Additional records:\n" + extra + "\n" + QUESTION + " Include vaults from both sets of records."}]
ts = time.time(); r = ask(msgs2); f, n = score(r["answer"], extra_needles)
results["cache"] = dict(prompt_tokens=r["prompt_tokens"], ttft_s=r["ttft_s"], decode_tok_s=r["decode_tok_s"], start=ts,
                        new_needle_found=f, answer=r["answer"][:400])
print(f"[cache] prompt {r['prompt_tokens']:,} tok, TTFT {r['ttft_s']}s, new needle found {f}/{n}", flush=True)
(OUT / "phase7_results.json").write_text(json.dumps(results, indent=1))

print("=== with 8 background streams", flush=True)
threads = [threading.Thread(target=bg_worker, args=(k,), daemon=True) for k in range(8)]
for t in threads: t.start()
time.sleep(60)
for i, size in enumerate(SIZES):
    row, _, _ = run_size(size, "with_bg", 200 + i); results["with_bg"].append(row)
    (OUT / "phase7_results.json").write_text(json.dumps(results, indent=1))
    time.sleep(20)
bg_stop.set(); time.sleep(5)
results["bg_errors"] = [e for e in bg_events if isinstance(e, tuple)][:20]
(OUT / "phase7_results.json").write_text(json.dumps(results, indent=1))
print("PHASE7_DONE", flush=True)
