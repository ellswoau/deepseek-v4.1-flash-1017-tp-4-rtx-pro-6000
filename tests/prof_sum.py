#!/usr/bin/env python3
"""Sum GPU kernel time by kernel name in a torch-profiler trace (rank 0). Usage: prof_sum.py trace.json.gz [top]"""
import gzip, json, sys, collections
t = json.load(gzip.open(sys.argv[1])); top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
ev = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
d = collections.Counter(); n = collections.Counter()
for e in ev: k = e["name"][:110]; d[k] += e["dur"]; n[k] += 1
tot = sum(d.values()); span = (max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)) / 1e3
print(f"kernels {len(ev)}  gpu busy {tot/1e3:.1f} ms  span {span:.1f} ms")
for k, v in d.most_common(top): print(f"{v/1e3:9.2f} ms {100*v/tot:5.1f}%  n={n[k]:6d}  {k}")
