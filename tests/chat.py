#!/usr/bin/env python3
"""Minimal streaming terminal chat for the local DeepSeek-V4.1-Flash endpoint.

Usage:  chat.py [--think] [--base http://127.0.0.1:8010/v1] [--key-file PATH]
Commands inside the chat:  /think on|off   /reset   /quit
The API key is read from a file (mode 600) and never printed.
"""
import argparse, os, sys, time
from pathlib import Path
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get("DS41_BASE", "http://127.0.0.1:8010/v1"))
ap.add_argument("--key-file", default=os.environ.get("DS41_KEY_FILE", "state/api-key"))
ap.add_argument("--think", action="store_true", help="enable reasoning (reasoning_effort=high)")
ap.add_argument("--system", default="You are a helpful assistant.")
a = ap.parse_args()

client = OpenAI(base_url=a.base, api_key=Path(a.key_file).read_text().strip(), timeout=900)
history = [{"role": "system", "content": a.system}]
think = a.think
print(f"DeepSeek-V4.1-Flash @ {a.base}  (thinking {'on' if think else 'off'}; /think on|off, /reset, /quit)\n")
while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not user: continue
    if user in ("/quit", "/exit"): break
    if user == "/reset": history = history[:1]; print("(history cleared)\n"); continue
    if user.startswith("/think"): think = user.endswith("on"); print(f"(thinking {'on' if think else 'off'})\n"); continue
    history.append({"role": "user", "content": user})
    kw = dict(reasoning_effort="high") if think else {}
    t0 = time.perf_counter(); first = None; out = []; usage = None; in_reason = False
    try:
        stream = client.chat.completions.create(model="deepseek-v4.1-flash", messages=history, stream=True,
                                                stream_options={"include_usage": True}, temperature=1.0, top_p=1.0, **kw)
        print("ds41> ", end="", flush=True)
        for ch in stream:
            if ch.usage: usage = ch.usage
            if not ch.choices: continue
            d = ch.choices[0].delta
            r = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning_content")
            if r:
                if not in_reason: print("\033[2m[thinking] ", end=""); in_reason = True
                print(r, end="", flush=True)
            if d.content:
                if in_reason: print("\033[0m\n", end=""); in_reason = False
                if first is None: first = time.perf_counter()
                out.append(d.content); print(d.content, end="", flush=True)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    except Exception as e:
        print(f"\n[error] {e}"); history.pop(); continue
    if in_reason: print("\033[0m", end="")
    dt = time.perf_counter() - t0
    if usage:
        tps = usage.completion_tokens / (time.perf_counter() - (first or t0)) if first else 0
        print(f"\n\033[2m[{usage.prompt_tokens} in / {usage.completion_tokens} out, {dt:.1f}s, ~{tps:.0f} tok/s]\033[0m\n")
    else:
        print("\n")
    history.append({"role": "assistant", "content": "".join(out)})
