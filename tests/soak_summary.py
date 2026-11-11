#!/usr/bin/env python3
"""Summarize a soak run: per-kind request stats + monitor stats. Writes soak-summary.csv and soak-hourly.csv in the soak dir."""
import csv, json, sys, statistics, collections
from pathlib import Path
D = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/ds41/evidence/soak")
rows = [json.loads(l) for l in open(D / "requests.jsonl")]
def pct(v, p):
    v = sorted(x for x in v if x is not None); return round(v[min(len(v) - 1, int(p * len(v)))], 2) if v else ""
t0 = min(r["ts"] for r in rows); by = collections.defaultdict(list)
for r in rows: by[r["kind"]].append(r)
with open(D / "soak-summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["kind", "requests", "failed", "fail_pct", "exceptions", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "ttft_max_s", "decode_p50_tok_s", "prompt_tokens_p50", "prompt_tokens_max"])
    for k in ("agent", "tool", "math", "long_cold", "long_cached", "huge_cold", "huge_cached"):
        rs = by[k]; t = [r.get("ttft") for r in rs]; bad = sum(not r["ok"] for r in rs)
        w.writerow([k, len(rs), bad, round(100 * bad / len(rs), 2) if rs else "", sum("error" in r for r in rs), pct(t, .5), pct(t, .95), pct(t, .99), pct(t, 1),
                    pct([r.get("decode") for r in rs], .5), pct([r.get("prompt_tokens") for r in rs], .5), pct([r.get("prompt_tokens") for r in rs], 1)])
    w.writerow(["ALL", len(rows), sum(not r["ok"] for r in rows), round(100 * sum(not r["ok"] for r in rows) / len(rows), 2), sum("error" in r for r in rows)] + [""] * 7)
mon = list(csv.DictReader(open(D / "monitor.csv")))
hours = collections.defaultdict(lambda: {"req": [], "mon": []})
for r in rows: hours[int((r["ts"] - t0) // 1800)]["req"].append(r)
import datetime
m0 = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc)
for m in mon:
    ts = datetime.datetime.strptime(m["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    h = int((ts - m0).total_seconds() // 1800)
    if h >= 0: hours[h]["mon"].append(m)
with open(D / "soak-hourly.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["half_hour", "requests", "failed", "agent_ttft_p50", "agent_ttft_p95", "agent_decode_p50", "running_max", "gen_tps_mean", "cache_hit_mean", "hicache_host_used_max", "gpu_temp_max", "gpu_power_sum_mean", "host_avail_gb_min"])
    for h in sorted(hours):
        rq = hours[h]["req"]; ag = [r for r in rq if r["kind"] == "agent"]; mm = hours[h]["mon"]
        fl = lambda c: [float(x[c]) for x in mm if x[c] not in ("", None)]
        w.writerow([h / 2, len(rq), sum(not r["ok"] for r in rq), pct([r.get("ttft") for r in ag], .5), pct([r.get("ttft") for r in ag], .95), pct([r.get("decode") for r in ag], .5),
                    max(fl("running"), default=""), round(statistics.mean(fl("gen_tps")), 1) if fl("gen_tps") else "", round(statistics.mean(fl("cache_hit")), 3) if fl("cache_hit") else "",
                    max(fl("hicache_host_used"), default=""), max(fl("gpu_temp_max"), default=""), round(statistics.mean(fl("gpu_power_sum"))) if fl("gpu_power_sum") else "", min(fl("host_avail_gb"), default="")])
dur = (max(r["ts"] for r in rows) - t0) / 3600
print(f"duration {dur:.2f} h, monitor rows {len(mon)}")
