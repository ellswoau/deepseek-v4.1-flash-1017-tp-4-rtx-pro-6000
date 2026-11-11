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

## 5. Prefix cache and HiCache (Phase 6)

**Multi-turn prefix cache: works.** The test conversation was 52–56K tokens. Each turn resent the full history, including the model's previous answer, plus about 1.8K new tokens. Thinking was tested both off and on (`reasoning_effort=high`).

| | Turn 1 (cold) | Turn 2 | Turn 3 | Cached tokens on resend (engine log) |
|---|---:|---:|---:|---|
| thinking off | 9.3 s³ | 0.27 s | 0.26 s | 52,224 / 54,016 |
| thinking on | 5.3 s | 0.26 s | 0.29 s | 52,224 / 53,760 / 55,808 |

³ includes first-request warm-up after start. A cold 52K prefill is otherwise about 5 s.

The whole previous prefix is reused (256-token pages). There is no cap around 11.5K tokens (sglang#43738).

**HiCache reload after GPU eviction: does not work for V4.1 on v0.5.23 in this configuration.** The test sent 4.73M tokens of unique traffic, then resent the conversation. TTFT was 5.95 s, identical to a cold prefill of the same size, and the engine logged no host load-back.

HiCache does write KV to host RAM: 4.9M of 16.8M host tokens were used and none were dropped. The problem is the size of the host pool for sliding-window state. `_deepseek_v4_num_host_pages` sets it to device SWA pages × `hicache_ratio` (here 186 × 4 = 744 pages). A hybrid-SWA prefix cannot be matched without its window state.

A stepwise test showed that a 48K conversation stays reusable through 850K tokens of other traffic (0.15 s TTFT), but not through 2.35M, which is well before the 4.2M-token GPU pool fills.

## 6. HiCache SWA host-pool patch (fix 4)

`DS41_DSV4_SWA_HOST_MULT` multiplies only the V4 SWA host pool. Measured with the Phase 6 scripts (`tests/phase6.py`, `tests/phase6b.py`):

| Config | Host pools per GPU | Multi-turn (Test A) | Active 48K conversation, resent between floods totalling 5.35M tokens | Idle 57K conversation after one 4.73M-token flood |
|---|---:|---|---|---|
| stock: ratio 4, 744 SWA pages | 32 GB | 0.26–0.29 s | lost by 2.35M (4.96 s) | cold (5.95 s) |
| **ratio 2 + SWA ×11 (4,092 pages)** (default) | **40 GB** | 0.27–0.29 s | **kept: 0.15 s, 48,128 tokens cached** | cold (5.96 s) |
| ratio 4 + SWA ×5.5 (4,092 pages) | 54 GB | 0.26–0.28 s | — | cold (5.93 s) |
| **ratio 4 + SWA ×11 (8,184 pages)**, needs about 768 GB host RAM | **80 GB** | 0.27–0.29 s | — | **0.17 s** (cold 6.03 s): 57,344 tokens reloaded from host in 0.03 s |

Cache-mode HiCache backs up about 0.45 SWA tokens per token of traffic, one window per prefill chunk. For an *idle* conversation to survive F tokens of other traffic, the SWA host tier needs about 0.45 × F tokens, at about 25 KB per token per GPU.

For F = 4.7M that is about 53 GB per GPU of SWA alone. **On a 768 GB host (ratio 4 + ×11, about 320 GB of pinned pools beside the 189 GiB Engram tables, about 226 GB still free), the idle conversation was reloaded from host RAM in 0.17 s instead of re-prefilled in 6 s.** On a 512 GB host, use ratio 2 + ×11. *Active* conversations survive anyway, because each resend refreshes their place in the cache.

Note: every v0.5.23 start logs `Metadata glue-graph capture failed ... falling back to eager metadata prep` once per rank, because DSpark draft metadata calls `seq_lens.tolist()` during capture. All v0.5.23 numbers here include that fallback.

## 7. Host memory: 4 channels to 8 channels (mixed DIMMs)

The upgrade went from 4 × 128 GB DDR5-6400 to 4 × 128 GB DDR5-6400 plus 4 × 64 GB DDR5-5600, 768 GB in total. All DIMMs train at **5600 MT/s**.

The BIOS creates **one NUMA node** with two interleave sets:
- 512 GB **8-way**: 64 GB from every DIMM.
- 256 GB **4-way**: the upper half of the 128 GB DIMMs.

Linux sees a single node and tier, and hands out the high (4-way) range first. [benchmarks/regionbw.c](../benchmarks/regionbw.c) measures each range using 1 GiB pages sorted by physical address. [benchmarks/whereis.py](../benchmarks/whereis.py) shows which range backs the engine's buffers.

| Measurement | 4 ch (before) | 768 GB mixed (after) |
|---|---:|---:|
| STREAM Triad, 8-way range | — | **226.5 GB/s** |
| STREAM Triad, 4-way range | — | 128.1 GB/s |
| STREAM Triad, ordinary allocation | 128.9 GB/s | 123.4 GB/s (lands in the 4-way range) |
| NCCL all_reduce / alltoall busbw (SHM) | 20.4 / 20.3 GB/s | 20.0 / 20.2 GB/s |
| Steady decode C=1 / 8 / 16, total tok/s | 215 / 827 / 1,147 | 227 / 822 / 1,173 |
| Agent-shaped C=1 / 8 / 16, total tok/s | 190 / 699 / 901 | 186 / 695 / 914 |
| Cold 32K prefill TTFT | 3.19 s | 3.07 s |

The Engram host tables ended up 100% in the 8-way range, and the HiCache pools mostly in the 4-way range. **Serving throughput did not change** beyond run-to-run noise (about ±8%), so it is bound by the GPUs and PCIe, not by host memory bandwidth. Doubling the channels added capacity and CPU-side bandwidth, not tokens per second. NCCL's SHM collectives did not get faster either.

Full rows: [ram-upgrade-before-after-2026-11-11.csv](ram-upgrade-before-after-2026-11-11.csv).

## 8. Long context, 1,048,576 tokens (Phase 7)

Setup: tokenizer-exact prompts of maintenance-log filler with 4 "vault code" needles at 10/40/70/95% depth (`tests/phase7.py`), temperature 0. Engine before FP8 tuning, chunk 2048. Raw output: `raw/phase7/`.

| Prompt tokens | Cold TTFT alone | Cold TTFT with 8 bg streams | Needles (alone / bg) |
|---:|---:|---:|---|
| 250,167 | 31.3 s | 27.3 s | 4/4, 4/4 |
| 500,178 | 56.9 s | 57.7 s | 4/4, 4/4 |
| 750,184 | 91.7 s | 92.3 s | 4/4, 4/4 |
| 1,000,160 | 129.9 s | 130.1 s | 4/4, 4/4 |

Cached resend of the 1M conversation with ~10K new tokens and a fifth needle: **2.89 s** TTFT, new needle found. No OOM. With interval 0, the background streams emitted nothing during every long prefill.

## 9. Prefill/decode interleaving (`--prefill-decode-interval`)

Test: 8 background streams plus one cold 250K prompt (`tests/pdi_test.py`). Raw output: `raw/phase7/pdi_results.jsonl`.

| Setting | 250K TTFT | In-flight answers, tok/s before → during | New requests during the long prefill |
|---|---:|---|---|
| interval 0 | 27.2 s | 194 → 0 | blocked |
| interval 8 | 32.4 s | 123 → 172 | first token after ~31.5 s |
| interval 8 + shortest-prefill-first | 33.5 s | 185 → 148 | ~33.5 s |
| mixed-chunk / interval 2 / interval 4 | 29.6 / 32.3 / 33.4 s | (total bg retained 9–15%) | — |

Interval 8 keeps answers that are already streaming alive. New requests still queue behind an in-progress chunked prefill; no scheduling policy changes that. The cost is lower prefill throughput when several cold prompts arrive at once (§10).

## 10. FP8 block-GEMM tuning (`tuned_configs/`)

Kernel microbenchmark for N=1536, K=5120 (`tests/fp8bench.py`), default config → tuned:

| M | 1 | 16 | 128 | 2048 |
|---|---:|---:|---:|---:|
| µs | 89.9 → 20.2 | 37.4 → 29.7 | 44.0 → 36.9 | 349.8 → 175.8 |

End to end with Sybil's `matrix.py` (1,024 output, 409,600 context, C16 cap). Full table: [build-comparison-2026-11-11.csv](build-comparison-2026-11-11.csv).

| | Sybil build | Untuned, interval 8 | **Tuned, interval 0** | **Tuned, interval 8** |
|---|---:|---:|---:|---:|
| Decode per stream, C=1, 2K–32K | 178–197 | 209 | 266–323 | 242–291 |
| Total decode, C=8, 2K | 847 | 993 | 1,054 | 1,060 |
| Cold prefill, C=1, 8K / 32K | 7,953 / 8,000 | 9,308 / 9,322 | 11,849 / 11,934 | 10,833 / 10,971 |
| Prefill, C=4, 8K–32K | ~8,030 | ~5,800 | ~12,000 | ~6,900 |
| Cold 32K TTFT, C=1 | 4.1 s | — | 2.7 s | 3.0 s |

## 11. Prefill chunk size (`--chunked-prefill-size`)
Setup: tuned FP8 image, interval 8, 1,048,576 context (to stress memory), restart per size. Cold ~1M prompt with 4 needles; the 250K background-streaming test from §9; Sybil's `matrix.py` at 8K/32K, C=1/4, 1,024 output. Raw output: `raw/chunk-size/`.

| Chunk | GPU free after load | Cold 1M TTFT | 32K C=1 prefill | 8K C=4 prefill | 32K C=4 prefill / TTFT | In-flight streams during 250K prefill (baseline ~200 tok/s) | New-request wait | 1M needles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 5.61 GB | 128.1 s | 10.9K | 6.9K | 6.6K / 11.1 s | 212 tok/s (100%) | 27.5 s | 4/4 |
| **4096** | 5.33 GB | 113.1 s | 12.2K | 9.3K | 8.6K / 8.5 s | 136 tok/s (~68%) | 23.7 s | 4/4 |
| 8192 | 4.96 GB | 99.0 s | 13.3K | 11.4K | 10.9K / 7.0 s | 48 tok/s (~24%) | 21.7 s | 4/4 |

No OOM or engine restart at any size. Single-stream decode is unchanged (~290 tok/s). Each chunk occupies the GPUs for longer, so interleaved decode rounds come less often; the interval cannot compensate because it only adds decode rounds between chunks. New requests still wait for the whole in-progress prefill at every size. The default here is **4096**: −12% cold 1M TTFT and about +30% concurrent prefill, while streams in flight keep about two thirds of their speed.
