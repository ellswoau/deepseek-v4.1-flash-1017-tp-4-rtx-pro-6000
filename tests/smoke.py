#!/usr/bin/env python3
"""DS41 Phase 4 smoke tests against the local OpenAI-compatible endpoint.

Covers: streamed chat (TTFT, tok/s), thinking via reasoning_effort, tool-call round trips under a
large (~16K token) agent system prompt (target 8/8), JSON-schema output, and one image input.
Key is read from a mode-600 file; it is never printed.
"""
import base64, io, json, os, sys, time
from pathlib import Path
from openai import OpenAI

BASE = "http://127.0.0.1:8010/v1"
KEY = Path(os.environ.get("DS41_KEY_FILE", "state/api-key")).read_text().strip()
MODEL = "deepseek-v4.1-flash"
SAMPLING = dict(temperature=1.0, top_p=1.0)          # model-card defaults
client = OpenAI(base_url=BASE, api_key=KEY, timeout=600)
results = {}

def section(name): print(f"\n=== {name}", flush=True)

# ---------------------------------------------------------------- 1. streamed chat
section("1. streamed chat")
t0 = time.perf_counter(); ttft = None; text = []; usage = None
stream = client.chat.completions.create(model=MODEL, stream=True, stream_options={"include_usage": True},
    max_tokens=600, messages=[{"role": "user", "content":
    "Explain in about 300 words why heavy-duty truck brake chambers fail, and how a parts counter "
    "employee should help a customer pick a replacement."}], **SAMPLING)
for ch in stream:
    if ch.usage: usage = ch.usage
    if ch.choices and ch.choices[0].delta.content:
        if ttft is None: ttft = time.perf_counter() - t0
        text.append(ch.choices[0].delta.content)
total = time.perf_counter() - t0
out_toks = usage.completion_tokens if usage else None
tps = out_toks / (total - ttft) if out_toks and ttft else None
print(f"TTFT {ttft:.2f}s  output {out_toks} tok  decode {tps:.1f} tok/s  total {total:.1f}s")
print("excerpt:", "".join(text)[:200].replace("\n", " "), "...")
results["stream"] = dict(ttft_s=round(ttft, 3), completion_tokens=out_toks, decode_tok_s=round(tps, 1), ok=bool(text))

# ---------------------------------------------------------------- 2. thinking
section("2. thinking via reasoning_effort")
r = client.chat.completions.create(model=MODEL, max_tokens=4096, reasoning_effort="high",
    messages=[{"role": "user", "content": "A truck uses 6.2 mpg. Diesel is $3.89/gal. A route is 1,340 miles "
              "round trip, run 3 times a week for 52 weeks. What is the annual fuel cost? Give the final number."}], **SAMPLING)
m = r.choices[0].message
reasoning = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content")
expected = 1340 * 3 * 52 / 6.2 * 3.89
ok = bool(reasoning) and any(s in (m.content or "").replace(",", "") for s in (f"{expected:.0f}", f"{expected:.2f}", f"{expected - 1:.0f}", f"{expected + 1:.0f}"))
print(f"reasoning chars {len(reasoning or '')}, answer: {(m.content or '').strip()[-160:]!r}")
print(f"expected ≈ {expected:,.2f}  -> {'PASS' if ok else 'CHECK'}  (tokens {r.usage.completion_tokens})")
results["thinking"] = dict(reasoning_chars=len(reasoning or ""), ok=ok)

# ---------------------------------------------------------------- 3. tool calls under a big system prompt
section("3. tool calls, large agent system prompt")
rules = []
for i in range(1, 181):
    rules.append(f"Rule {i}. When handling request category {i % 17} at branch {i % 23}, confirm the customer account, "
                 f"check open orders older than {i % 9 + 1} days, never promise delivery before the carrier cut-off, "
                 f"record the interaction in the ticket system with tag OPS-{i:03d}, and escalate to a branch manager if the "
                 f"order value exceeds ${(i * 137) % 9000 + 1000}. Parts with core charges must show the core amount separately. "
                 f"Hazmat items (batteries, refrigerant, brake fluid) require the hazmat flag on shipment.")
SYSTEM = ("You are Ops-Agent, an internal operations agent for a parts distributor. "
          "Use the provided tools whenever the user asks for data or an action; never invent inventory numbers, "
          "ticket IDs or customer data. Call exactly one tool when one tool is enough.\n\n" + "\n".join(rules))
TOOLS = [
  {"type": "function", "function": {"name": "check_inventory", "description": "Get on-hand quantity of a part number at a branch.",
    "parameters": {"type": "object", "properties": {"part_number": {"type": "string"}, "branch": {"type": "string"}},
                   "required": ["part_number", "branch"], "additionalProperties": False}}},
  {"type": "function", "function": {"name": "lookup_customer", "description": "Find a customer account by name or account number.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
  {"type": "function", "function": {"name": "create_ticket", "description": "Open an internal support ticket.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
                   "required": ["summary", "priority"], "additionalProperties": False}}},
]
CASES = [
  ("How many 3200-1123 brake chambers do we have at the Northside branch?", "check_inventory", {"part_number": "3200-1123"}),
  ("Find the customer account for Example County Road Commission.", "lookup_customer", {}),
  ("Open a high priority ticket: the Eastside branch label printer is down.", "create_ticket", {"priority": "high"}),
  ("Is part ABS-77801 in stock in Westside?", "check_inventory", {"part_number": "ABS-77801"}),
  ("Look up account number 104882.", "lookup_customer", {}),
  ("Create a low priority ticket to reorder shelf labels for aisle 12.", "create_ticket", {"priority": "low"}),
  ("Check stock of FLT-5501 air filters at the Southside branch.", "check_inventory", {"part_number": "FLT-5501"}),
  ("Ticket please, normal priority: customer reports wrong core charge on invoice 55120.", "create_ticket", {"priority": "normal"}),
]
FAKE = {"check_inventory": {"on_hand": 14, "bin": "B-07"}, "lookup_customer": {"account": "104882", "name": "Example County Road Commission", "terms": "Net 30"},
        "create_ticket": {"ticket_id": "OPS-10001"}}
passed = 0; prompt_toks = []; cached = []; ttfts = []
for q, want, args_must in CASES:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, max_tokens=512, **SAMPLING)
    ttfts.append(time.perf_counter() - t0)
    prompt_toks.append(r.usage.prompt_tokens)
    det = getattr(r.usage, "prompt_tokens_details", None)
    cached.append(getattr(det, "cached_tokens", None) if det else None)
    calls = r.choices[0].message.tool_calls or []
    good = len(calls) == 1 and calls[0].function.name == want
    if good:
        a = json.loads(calls[0].function.arguments)
        good = all(str(a.get(k, "")).upper() == v.upper() for k, v in args_must.items())
        # round trip: feed the tool result back and require a final natural-language answer
        msgs += [r.choices[0].message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps(FAKE[want])}]
        r2 = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, max_tokens=512, **SAMPLING)
        final = r2.choices[0].message.content or ""
        good = good and len(final.strip()) > 0 and not r2.choices[0].message.tool_calls
    passed += good
    print(f"  {'PASS' if good else 'FAIL'}  {want:16s} {q[:60]!r}  -> {[ (c.function.name, c.function.arguments) for c in calls ]}")
print(f"tool round trips: {passed}/8   system prompt ≈ {prompt_toks[0]:,} tokens; "
      f"cached tokens per call: {cached}; full-call latency s: {[round(t, 1) for t in ttfts]}")
results["tools"] = dict(passed=passed, of=8, prompt_tokens=prompt_toks[0], cached_tokens=cached)

# ---------------------------------------------------------------- 4. JSON schema
section("4. JSON-schema output")
schema = {"type": "object", "additionalProperties": False, "required": ["part_number", "description", "core_charge", "hazmat"],
          "properties": {"part_number": {"type": "string"}, "description": {"type": "string"},
                         "core_charge": {"type": "number"}, "hazmat": {"type": "boolean"}}}
r = client.chat.completions.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content":
    "Extract: 'Group 31 AGM battery, part BAT-31AGM, $22 core.'"}],
    response_format={"type": "json_schema", "json_schema": {"name": "part", "strict": True, "schema": schema}}, **SAMPLING)
obj = json.loads(r.choices[0].message.content)
ok = set(obj) == set(schema["required"]) and obj["part_number"].upper() == "BAT-31AGM" and float(obj["core_charge"]) == 22 and obj["hazmat"] is True
print(obj, "->", "PASS" if ok else "CHECK")
results["json_schema"] = dict(obj=obj, ok=ok)

# ---------------------------------------------------------------- 5. one image
section("5. single image")
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (900, 400), "white"); d = ImageDraw.Draw(img)
d.polygon([(80, 330), (230, 70), (380, 330)], fill="green"); d.rectangle((520, 90, 780, 330), fill="orange")
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
except Exception: font = ImageFont.load_default()
d.text((560, 20), "ABC-42", fill="black", font=font)
buf = io.BytesIO(); img.save(buf, format="PNG")
r = client.chat.completions.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": [
    {"type": "text", "text": "Name each colored shape left to right, and read the text in the image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}], **SAMPLING)
t = (r.choices[0].message.content or "").lower()
ok = all(w in t for w in ("green", "triangle", "orange")) and ("abc-42" in t or "abc 42" in t)
print(t[:300].replace("\n", " "), "->", "PASS" if ok else "CHECK")
results["image"] = dict(ok=ok, prompt_tokens=r.usage.prompt_tokens)

print("\n=== SUMMARY"); print(json.dumps(results, indent=1))
(Path(os.environ.get("DS41_OUT", ".")) / "smoke_results.json").write_text(json.dumps(results, indent=1))
