# Results (2026-11-10)

All runs were on one machine: 4 × RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W, Gen5 x16 each, no NVLink), Xeon 654 (18 cores), 512 GB DDR5-6400 (4 × 128 GB, 4 channels), Ubuntu 26.04.1, driver 595.99.02. Model: DeepSeek-V4.1-Flash-1017. Engine: TP4/EP4, FP8 KV, DSpark block 5, context 409,600, memory fraction 0.92, KV pool 4,199,936 tokens.

| File | Contents |
|---|---|
| [sybil-build-performance-2026-11-10.csv](sybil-build-performance-2026-11-10.csv) | Sybil build: Sybil's `matrix.py` (with Sybil's published reference numbers) and a C=1–32 sweep |
| [sglang-v0523-performance-2026-11-10.csv](sglang-v0523-performance-2026-11-10.csv) | This build (v0.5.23 + fixes + HiCache + Engram THP): C=1–16, cold 32K, three C=1 repeats |
| [before-after-2026-11-10.csv](before-after-2026-11-10.csv) | Per C and metric: Sybil, v0.5.23 before the THP fix, v0.5.23 after |
| [raw/](raw/) | Benchmark client output and hardware tests (NCCL, P2P, topology, link speed under load) |

## 1. Sybil build: our hardware against Sybil's published matrix

This uses Sybil's `matrix.py`: an LRU-cache code prompt padded with filler, `/generate`, temperature 0, decode measured over output tokens 129–641, engine C8. Every one of the 16 waves succeeded.

| Input | C | Prefill tok/s (ours / Sybil) | TTFT s | Decode/stream (ours / Sybil) | Total decode (ours / Sybil) |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 2,516 / 1,070 | 0.2 | 173 / 152 | 173 / 152 |
| 2,048 | 8 | 7,676 / 1,169 | 1.5 | 109 / – | 847 / – |
| 8,192 | 1 | 7,953 / 7,458 | 1.0 | 197 / 158 | 182 / 158 |
| 8,192 | 8 | 7,995 / 3,167 | 4.9 | 103 / – | 843 / – |
| 32,768 | 1 | 8,000 / 7,817 | 4.1 | 189 / 187 | 190 / 187 |
| 32,768 | 8 | 8,020 / 5,947 | 18.6 | 108 / – | 837 / – |

## 2. Sweep: Sybil build against this build

Workloads (`benchmarks/sweep.sh`, `sglang.benchmark.serving --backend sglang`):
- *Steady:* random 1,024 in / 2,048 out, `ignore_eos`.
- *Agent:* `generated-shared-prefix`, a 30K shared system prompt plus a 2K unique question, 512 out.

Each point is one wave of 4 × C requests, and every request succeeded. Per-stream speed is 1000 / median TPOT.

| | C | Sybil total tok/s | v0.5.23 no THP | **v0.5.23 + THP** | per-stream (THP) | p99 TTFT s (Sybil → THP) |
|---|---:|---:|---:|---:|---:|---:|
| Steady | 1 | 191 | 253 | **215** (repeats 214/224/240) | 225 | – → 0.16 |
| | 2 | 334 | | **390** | 233 | – → 0.17 |
| | 4 | 551 | | **543** | 154 | – → 0.24 |
| | 8 | 703 | 799 | **827** | 122 | – → 0.71 |
| | 12 | 858 | | **958** | 92 | – → 0.72 |
| | 16 | 1,014 | | **1,147** | 87 | – → 0.73 |
| Agent | 1 | 160 | 192 | **190** | 224 | 0.50 → 0.37 |
| | 2 | 305 | | **329** | 180 | 0.86 → 0.40 |
| | 4 | 437 | | **493** | 141 | 0.80 → 0.64 |
| | 8 | 689 | 599 | **699** | 105 | 1.60 → 1.36 |
| | 12 | 796 | | **710**² | 70 | 1.96 → 1.54 |
| | 16 | 778 | | **901** | 64 | 2.69 → 2.03 |

DSpark accept length was 4.6–4.9 throughout.

² This point looks like a single-run outlier (C=8 gave 699 and C=16 gave 901).

On the Sybil build with an engine cap of 32, steady throughput kept scaling to 1,379 tok/s at C=32. Agent-shaped throughput levelled off at about 800 tok/s from C=12, with p99 TTFT at 7.3 s for C=24 and 9.9 s for C=32. That is why C16 is the recommended cap.

## 3. Notes

- A forced-length prose probe (`ignore_eos` past the natural end) gives about 80 tok/s single-stream, because DSpark acceptance collapses on filler. Do not use it to judge speed.
- With `--tool-call-parser deepseekv41`, `/v1/chat/completions` uses SGLang's native V4.1 encoder. The checkpoint's `chat_template.jinja` is not used.
- Smoke results on both builds: 8/8 tool round trips with an 18,314-token system prompt (warm calls 0.3–0.5 s through the prefix cache), strict JSON schema, thinking via `reasoning_effort`, and single-image vision.

## 4. Host-memory baseline before the DIMM change (4 of 8 channels)

The tested host ran 4 × 128 GB DDR5-6400, populating 4 of the Xeon's 8 memory channels. This baseline was recorded before moving to 8 channels. Raw output is in [raw/dimm-4ch-before/](raw/dimm-4ch-before/).

| Measurement | 4 channels |
|---|---:|
| STREAM Triad / Copy, 36 threads, 4 GiB arrays ([benchmarks/stream.c](../benchmarks/stream.c)) | 129 / 115 GB/s (theoretical 204.8) |
| STREAM Triad, 18 threads | 121 GB/s |
| NCCL all_reduce busbw, default (SHM transport), 8M–1G | 20.4 GB/s |
| NCCL alltoall busbw, default (SHM) | 20.3 GB/s (algbw 27.5) |
| NCCL all_reduce / alltoall busbw, `NCCL_P2P_LEVEL=SYS` | 24.9 / 2.0 GB/s |
| Raw P2P per pair, unidirectional / bidirectional | 38–42 / 73–76 GB/s |
| Raw P2P, all 12 directed pairs concurrently | 79–86 GB/s total |

Expectation: decode is GPU-bound, and Engram lookups are latency-bound (a few KB per token), so decode should change by less than about 5%. NCCL's default SHM collectives stage every byte through host DRAM. At an alltoall algorithm bandwidth of about 27.5 GB/s per GPU, they already move roughly as much traffic as STREAM can sustain. Prefill-heavy and high-concurrency agent work (EP all-to-all) is therefore where 8 channels could show up.
