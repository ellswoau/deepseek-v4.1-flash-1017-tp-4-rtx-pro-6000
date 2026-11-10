#!/usr/bin/env python3
"""Single-stream steady decode probe: 2048 output tokens, temperature 1.0 vs 0.0, two runs each."""
import json, os, time
from pathlib import Path
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key=Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip(), timeout=600)
P = ("Write a detailed, multi-section maintenance guide for a fleet of Class 8 tractors: air brakes, "
     "aftertreatment (DPF, SCR, DEF), cooling, electrical, tires, and a weekly inspection checklist. Be thorough.")
out = []
for temp in (1.0, 0.0):
    for run in (1, 2):
        t0 = time.perf_counter(); first = None; n = 0
        s = c.chat.completions.create(model="deepseek-v4.1-flash", stream=True, stream_options={"include_usage": True},
              max_tokens=2048, temperature=temp, top_p=1.0, messages=[{"role": "user", "content": P}],
              extra_body={"ignore_eos": True})
        for ch in s:
            if ch.usage: n = ch.usage.completion_tokens
            if ch.choices and ch.choices[0].delta.content and first is None: first = time.perf_counter()
        end = time.perf_counter()
        r = dict(temperature=temp, run=run, tokens=n, ttft_s=round(first - t0, 2), decode_tok_s=round(n / (end - first), 1))
        out.append(r); print(r, flush=True)
(Path(os.environ.get("DS41_OUT", ".")) / "decode_probe.json").write_text(json.dumps(out, indent=1))
