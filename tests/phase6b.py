#!/usr/bin/env python3
"""Phase 6b: HiCache diagnostic. Build a ~52K conversation, then floods of increasing size
(each well under the 4.2M GPU full-KV pool except the last), resending the conversation after each.
Shows the flood size at which the prefix stops being reusable (SWA tier churn vs full-KV eviction)."""
import concurrent.futures as cf, json, os, random, time, uuid
from pathlib import Path
from openai import OpenAI

KEY = Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip()
OUT = Path(os.environ.get("DS41_OUT", "results/phase6"))
c = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=1800)
M = "deepseek-v4.1-flash"; rng = random.Random(7)
W = "axle bearing brake chamber caliper rotor drum valve compressor dryer hose fitting tractor trailer kingpin spring bushing pinion yoke clutch starter battery sensor injector turbo radiator filter gasket seal hub warranty invoice order quote freight pallet vendor branch account core credit".split()
doc = lambda n, tag: "\n".join(f"[{tag}-{i}] " + " ".join(rng.choice(W) for _ in range(120)) + "." for i in range(n))
NOOP = {"chat_template_kwargs": {"thinking": False}}

def ttft(messages):
    t0 = time.perf_counter()
    for ch in c.chat.completions.create(model=M, messages=messages, stream=True, max_tokens=8, extra_body=NOOP):
        if ch.choices and ch.choices[0].delta.content: return round(time.perf_counter() - t0, 2)
    return round(time.perf_counter() - t0, 2)

conv = [{"role": "system", "content": f"[{uuid.uuid4().hex}] analyst"},
        {"role": "user", "content": doc(330, "LOG") + "\nSummarise in one line."}]
res = {"cold": ttft(conv)}; print("cold build:", res["cold"], "s", flush=True)
res["warm"] = ttft(conv); print("warm resend:", res["warm"], "s", flush=True)
total = 0
for step_tokens in (50_000, 200_000, 600_000, 1_500_000, 3_000_000):
    per = 50_000; n = max(1, step_tokens // per)
    def f(i):
        c.chat.completions.create(model=M, max_tokens=1, extra_body=NOOP,
            messages=[{"role": "user", "content": f"[{uuid.uuid4().hex}] " + doc(per // 155, f"F{i}") + "\nOK"}])
    with cf.ThreadPoolExecutor(4) as ex: list(ex.map(f, range(n)))
    total += n * per
    t = ttft(conv); res[f"after_{total}"] = t
    print(f"after cumulative flood ~{total:,} tokens: resend TTFT {t} s", flush=True)
(OUT / "phase6b_results.json").write_text(json.dumps(res, indent=1)); print("PHASE6B_DONE", flush=True)
