#!/usr/bin/env python3
"""Which physical memory region backs the engine's big host buffers? (root, read-only)
Samples every 64 MiB of each large mapping of scheduler TP0 and classifies the physical address against the
boundary of the 4-way interleaved range (0x8080000000 on this host)."""
import os, re, struct, subprocess, sys
BOUNDARY = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0x8080000000
pid = subprocess.check_output(["pgrep", "-f", "sglang::scheduler_TP0"], text=True).split()[0]
maps = open(f"/proc/{pid}/maps").read().splitlines()
pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
STEP = 64 << 20
tot = {"8-way (low)": 0, "4-way (high)": 0, "not resident": 0}
rows = []
for line in maps:
    m = re.match(r"([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \S+\s*(.*)", line)
    lo, hi, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3) or "[anon]"
    size = hi - lo
    if size < (1 << 30): continue
    c = {"8-way (low)": 0, "4-way (high)": 0, "not resident": 0}
    for va in range(lo, hi, STEP):
        e = struct.unpack("Q", os.pread(pm, 8, (va // 4096) * 8))[0]
        if not (e >> 63) & 1: c["not resident"] += 1; continue
        pa = (e & ((1 << 55) - 1)) * 4096
        c["8-way (low)" if pa < BOUNDARY else "4-way (high)"] += 1
    for k in c: tot[k] += c[k] * STEP
    rows.append((size, name[:48], c))
for size, name, c in sorted(rows, reverse=True)[:12]:
    n = sum(c.values()) or 1
    print(f"{size/2**30:7.1f} GiB  {name:48s}  low {100*c['8-way (low)']/n:5.1f}%  high {100*c['4-way (high)']/n:5.1f}%  nonres {100*c['not resident']/n:5.1f}%")
print("TOTAL (mappings >= 1 GiB): " + ", ".join(f"{k} {v/2**30:.0f} GiB" for k, v in tot.items()))
