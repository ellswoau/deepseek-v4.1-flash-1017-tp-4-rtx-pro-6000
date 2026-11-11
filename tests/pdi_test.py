#!/usr/bin/env python3
"""Prefill/decode interleave test: 8 background chat streams + one cold ~250K-token prompt.

Measures real background tokens/s (tokenizer-counted stream text) in three windows: baseline (before the long
request), during its prefill (send -> first token), and after it finishes. Also the long request's TTFT, decode
rate and needle recall. Usage: pdi_test.py <label>   (results appended to $DS41_OUT/pdi_results.jsonl)
"""
import json, os, random, sys, threading, time, uuid
from pathlib import Path
from openai import OpenAI
from tokenizers import Tokenizer

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
KEY = Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip()
OUT = Path(os.environ.get("DS41_OUT", "results/pdi")); OUT.mkdir(parents=True, exist_ok=True)
TOK = Tokenizer.from_file(os.environ.get("DS41_TOKENIZER", "/models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))
SIZE = int(os.environ.get("LONG_TOKENS", "250000"))
c = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key=KEY, timeout=3600)
M = "deepseek-v4.1-flash"; NOTHINK = {"chat_template_kwargs": {"thinking": False}}
# reuse the filler/needle builder from phase7.py (UNITS ... QUESTION), without running that script
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase7.py")).read()
exec(src[src.index("UNITS ="):src.index("def ask(")])  # defines entry(), build(), QUESTION

events, stop = [], threading.Event()  # (time, n_tokens, request_id)
reqs = []  # dict(sent, first, id)
def worker(k):
    rng = random.Random(5000 + k)
    while not stop.is_set():
        body = "".join(entry(rng, i) for i in range(70))
        rid = uuid.uuid4().hex; rec = dict(id=rid, sent=time.time(), first=None); reqs.append(rec)
        try:
            for ch in c.chat.completions.create(model=M, stream=True, max_tokens=512, temperature=1.0, extra_body=NOTHINK,
                    messages=[{"role": "user", "content": f"[{rid}]\n{body}\nSummarise the main failure trends in 5 bullet points."}]):
                if ch.choices and ch.choices[0].delta.content:
                    now = time.time()
                    if rec["first"] is None: rec["first"] = now
                    events.append((now, len(TOK.encode(ch.choices[0].delta.content, add_special_tokens=False).ids), rid))
                if stop.is_set(): break
        except Exception as e:
            print("bg error:", str(e)[:120], flush=True)
def rate(t0, t1): return round(sum(n for t, n, _ in events if t0 <= t < t1) / max(t1 - t0, 1e-6), 1)
def inflight_rate(t0, t1):
    """tok/s over [t0,t1) from requests that had already started streaming before t0 (answers in progress)."""
    started = {r["id"] for r in reqs if r["first"] and r["first"] < t0}
    toks = [(t, n) for t, n, rid in events if rid in started and t0 <= t < t1]
    if not toks: return 0.0
    end = min(t1, max(t for t, _ in toks) + 0.5)  # until those answers finished
    return round(sum(n for _, n in toks) / max(end - t0, 1e-6), 1)
def new_req_ttfts(t0, t1):
    """TTFT of background requests sent during [t0,t1)."""
    return sorted(round(r["first"] - r["sent"], 1) for r in reqs if t0 <= r["sent"] < t1 and r["first"])

ths = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(8)]
for t in ths: t.start()
time.sleep(45)
doc, needles = build(SIZE, int(time.time()) % 100000)
msgs = [{"role": "system", "content": f"[{uuid.uuid4().hex}] You are a careful records analyst."},
        {"role": "user", "content": doc + "\n" + QUESTION}]
ts = time.time(); t0 = time.perf_counter(); first = None; text = []; usage = None
for ch in c.chat.completions.create(model=M, messages=msgs, stream=True, stream_options={"include_usage": True},
                                    max_tokens=1024, temperature=0.0, extra_body=NOTHINK):
    if ch.usage: usage = ch.usage
    if ch.choices and ch.choices[0].delta.content:
        if first is None: first = time.perf_counter(); t_first = time.time()
        text.append(ch.choices[0].delta.content)
te = time.time(); time.sleep(30); stop.set(); time.sleep(3)
ans = "".join(text).lower(); found = sum(code.lower() in ans for _, code in needles)
row = dict(label=LABEL, prompt_tokens=usage.prompt_tokens if usage else None, ttft_s=round(first - t0, 2),
           long_decode_tok_s=round(usage.completion_tokens / (te - t_first), 1) if usage and te > t_first else None,
           needles=f"{found}/{len(needles)}", bg_tok_s_baseline=rate(ts - 40, ts), bg_tok_s_during_prefill=rate(ts + 2, t_first),
           bg_tok_s_after=rate(te, te + 30))
row["bg_retained_during_prefill_pct"] = round(100 * row["bg_tok_s_during_prefill"] / row["bg_tok_s_baseline"], 1) if row["bg_tok_s_baseline"] else None
row["inflight_tok_s_during_prefill"] = inflight_rate(ts + 1, t_first)
row["inflight_tok_s_baseline"] = inflight_rate(ts - 30, ts - 15)
tt = new_req_ttfts(ts + 1, t_first)
row["new_bg_requests_during_prefill"] = len(tt)
row["new_bg_ttft_median_s"] = tt[len(tt) // 2] if tt else None
row["new_bg_ttft_max_s"] = tt[-1] if tt else None
row["baseline_bg_ttft_median_s"] = (lambda x: x[len(x) // 2] if x else None)(new_req_ttfts(ts - 40, ts - 5))
print(json.dumps(row), flush=True)
with open(OUT / "pdi_results.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
