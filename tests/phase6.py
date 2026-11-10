#!/usr/bin/env python3
"""Phase 6: multi-turn prefix-cache test (~60K context) and HiCache reload after GPU-cache eviction.

Test A (per thinking mode): turn 1 cold (~60K), turns 2-3 resend full history incl. the model's previous
answer + ~2K new tokens. Server-side cached-token counts are read afterwards from the engine log
(lines tagged by the per-test nonce timestamps). Pass: >=95% of the previous prefix cached and TTFT <= 2 s.

Test B: flood the GPU KV pool (4.2M tokens) with unique prompts, then resend the conversation. HiCache should
reload the prefix from host RAM (TTFT ~1-2 s) instead of a cold re-prefill (~8 s).
"""
import concurrent.futures as cf, json, os, random, sys, time, uuid
from pathlib import Path
from openai import OpenAI

BASE = os.environ.get("DS41_BASE", "http://127.0.0.1:8010/v1")
KEY = Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip()
OUT = Path(os.environ.get("DS41_OUT", "results/phase6")); OUT.mkdir(parents=True, exist_ok=True)
FLOOD_TOKENS = int(os.environ.get("FLOOD_TOKENS", "4700000"))
c = OpenAI(base_url=BASE, api_key=KEY, timeout=1800)
M = "deepseek-v4.1-flash"
rng = random.Random(41)

WORDS = ("axle bearing brake chamber caliper rotor drum shoe lining slack adjuster valve compressor dryer "
         "governor reservoir hose fitting coupling gladhand tractor trailer kingpin fifth wheel spring shackle "
         "bushing torque rod differential pinion ring gear carrier yoke u-joint driveshaft transmission clutch "
         "flywheel starter alternator battery cable harness sensor module injector turbo intercooler radiator "
         "thermostat coolant filter element gasket seal hub oil grease warranty invoice order quote shipment "
         "carrier freight pallet backorder vendor branch counter account terms core credit return inspection").split()

def para(n_words, tag):
    return f"[{tag}] " + " ".join(rng.choice(WORDS) for _ in range(n_words)) + "."

def doc(n_paras, words_per, prefix):
    return "\n".join(para(words_per, f"{prefix}-{i:04d}") for i in range(n_paras))

def ask(messages, think, max_tokens=200):
    kw = dict(reasoning_effort="high") if think else dict(extra_body={"chat_template_kwargs": {"thinking": False}})
    t0 = time.perf_counter(); first = None; text = []; usage = None
    s = c.chat.completions.create(model=M, messages=messages, stream=True, stream_options={"include_usage": True},
                                  max_tokens=max_tokens, temperature=1.0, top_p=1.0, **kw)
    for ch in s:
        if ch.usage: usage = ch.usage
        if ch.choices:
            d = ch.choices[0].delta
            r = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning_content")
            if (d.content or r) and first is None: first = time.perf_counter()
            if d.content: text.append(d.content)
    return dict(ttft_s=round((first or time.perf_counter()) - t0, 3), total_s=round(time.perf_counter() - t0, 3),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None, answer="".join(text))

def conversation(think, label):
    nonce = uuid.uuid4().hex[:12]
    sysmsg = {"role": "system", "content": f"[{nonce}] You are an operations analyst. Answer briefly and cite record tags. "
              + doc(8, 120, "POLICY")}
    msgs = [sysmsg, {"role": "user", "content": "Here is the parts and order log to analyse:\n" + doc(330, 120, "LOG")
                     + "\n\nQuestion 1: list three record tags that mention both 'kingpin' and 'warranty'."}]
    turns = []
    for t in range(1, 4):
        ts = time.time(); r = ask(msgs, think); r.update(turn=t, ts=ts, label=label, nonce=nonce)
        turns.append(r)
        print(f"  {label} turn {t}: prompt {r['prompt_tokens']:,} tok, TTFT {r['ttft_s']:.2f}s, out {r['completion_tokens']}", flush=True)
        msgs = msgs + [{"role": "assistant", "content": r["answer"]},
                       {"role": "user", "content": "More records:\n" + doc(11, 120, f"ADD{t}") +
                        f"\n\nQuestion {t + 1}: which of the new records mention 'compressor'? Tags only."}]
    return msgs, turns

results = {"started": time.time()}
print("=== Test A: multi-turn prefix cache", flush=True)
msgs_off, results["A_thinking_off"] = conversation(False, "think-off")
msgs_on, results["A_thinking_on"] = conversation(True, "think-on")

print(f"=== Test B: flood ~{FLOOD_TOKENS:,} unique tokens to evict GPU cache, then resend", flush=True)
probe = msgs_off[:-1] + [{"role": "user", "content": msgs_off[-1]["content"] + "\nAlso: summarise in one line."}]
# warm reference: same history resent before the flood (should be a GPU hit)
r = ask(probe, False); r.update(label="B-pre-flood", ts=time.time()); results["B_pre"] = r
print(f"  pre-flood resend: prompt {r['prompt_tokens']:,}, TTFT {r['ttft_s']:.2f}s", flush=True)
per = 110_000; n = FLOOD_TOKENS // per + 1
def flood(i):
    body = f"[flood-{uuid.uuid4().hex}] " + doc(per // 155, 120, f"F{i}")
    t0 = time.perf_counter()
    c.chat.completions.create(model=M, messages=[{"role": "user", "content": body + "\nReply OK."}], max_tokens=1,
                              extra_body={"chat_template_kwargs": {"thinking": False}})
    return time.perf_counter() - t0
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(4) as ex: durs = list(ex.map(flood, range(n)))
results["B_flood"] = dict(requests=n, approx_tokens=n * per, seconds=round(time.perf_counter() - t0, 1))
print(f"  flood: {n} requests, ~{n*per:,} tokens, {results['B_flood']['seconds']}s", flush=True)
r = ask(probe, False); r.update(label="B-post-flood", ts=time.time()); results["B_post"] = r
print(f"  post-flood resend: prompt {r['prompt_tokens']:,}, TTFT {r['ttft_s']:.2f}s", flush=True)
# cold reference: brand-new 60K conversation of the same size
cold = [msgs_off[0] | {"content": "[" + uuid.uuid4().hex + "] " + msgs_off[0]["content"]}] + probe[1:]
r = ask(cold, False); r.update(label="B-cold-reference", ts=time.time()); results["B_cold"] = r
print(f"  cold reference (same size, new prefix): prompt {r['prompt_tokens']:,}, TTFT {r['ttft_s']:.2f}s", flush=True)

for k in ("A_thinking_off", "A_thinking_on"):
    for t in results[k]: t.pop("answer", None)
for k in ("B_pre", "B_post", "B_cold"): results[k].pop("answer", None)
(OUT / "phase6_results.json").write_text(json.dumps(results, indent=1))
print("PHASE6_DONE", flush=True)
