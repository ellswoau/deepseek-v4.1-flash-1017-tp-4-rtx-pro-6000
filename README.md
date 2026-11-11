# DeepSeek-V4.1-Flash on 4 × RTX PRO 6000 (SM120) with stock SGLang v0.5.23

This repo serves **DeepSeek-V4.1-Flash-1017** on one workstation with **four 96 GB RTX PRO 6000 Blackwell Max-Q GPUs** (SM120, PCIe only, no NVLink). It uses the **stock SGLang v0.5.23 image** plus three small SM120 runtime fixes. Engram tables live in locked, huge-page host RAM through SGLang's **native Engram host table**, and **HiCache** is on, with a larger sliding-window host pool.

It builds on the [Sybil 4 × RTX PRO 6000 build](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000), which pins an earlier SGLang development image. Here, Sybil's SM120 fixes are ported to the v0.5.23 release. Its Engram row-store adapter is replaced by the upstream host table, and v0.5.23's new SM120 startup failure (sglang#44445) is fixed.

> **Status:** experimental and measured on one machine. Throughput, prefix caching, tools, JSON and single-image vision have passed. Long-context (1M) acceptance and a 24 h soak are pending.

## 1. Results at a glance

These numbers use the same hardware, the same weights, the same `sglang.benchmark.serving` workloads, engine settings C16 (or C32 for the Sybil column), 409,600 context and DSpark block 5. Full tables are in [results/RESULTS.md](results/RESULTS.md).

| Workload | C | Sybil build | **This build** | Change |
|---|---:|---:|---:|---:|
| Steady decode (1K in / 2K out), total tok/s | 1 | 191 | **215–240** | +12–26% |
| | 8 | 703 | **827** | +18% |
| | 16 | 1,014 | **1,147** | +13% |
| Agent-shaped (30K shared prompt + 2K turn, 512 out), total tok/s | 1 | 160 | **190** | +19% |
| | 8 | 689 | **699** | +1% |
| | 16 | 778 | **901** | +16% |
| Agent p99 time-to-first-token | 16 | 2.69 s | **2.03 s** | −25% |
| Cold 32K-token prefill TTFT | 1 | 4.1 s¹ | **~3.2 s** | |

¹ measured with Sybil's own `benchmarks/matrix.py`.

The Engram huge-page setting matters. Without it, agent-shaped throughput at C=8 was **599 tok/s, 13% below Sybil**. With it, throughput was **699 tok/s**. See [host prep](host/README.md).

## 2. What is different from stock SGLang v0.5.23

| # | Change | Why | License |
|---|---|---|---|
| 1 | `runtime/flash_mla_sm120.py`: also split V4.1's ratio-1/2 **extra** KV source (256/128-token pages) into 64-token pages for the FlashInfer SM120 sparse-MLA prefill, using its own scratch buffer. Ported from Sybil. The 13-line diff is in [runtime/flash_mla_sm120.patch](runtime/flash_mla_sm120.patch). | Without it, FlashInfer rejects `extra_page_block_size=128` and prefill falls back to Triton: about 294 tok/s at 1M (sglang#44445) against about 8K tok/s here. | Apache-2.0 (SGLang) |
| 2 | `adapter/sitecustomize.py`: set `PagedIndexerMetadata.force_deep_gemm_metadata` for compress ratio 1/2 on SM120. Ported from Sybil. | V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel. SM120 needs DeepGEMM's split planner metadata, and upstream never sets it. | MIT |
| 3 | `adapter/sitecustomize.py`: guard `make_candidate_indexer`. When DeepGEMM paged *sparse* MQA logits are unavailable, it returns the dense-blocks backend, built exactly as the function's own early-return path does. New for v0.5.23. | v0.5.23 enables the sparse-table candidate indexer when `is_sm100_or_newer()` (capability ≥ 10, which includes SM120). It only enables the DeepGEMM kernel on SM100, so startup raises *"the candidate indexer needs DeepGEMM's paged sparse MQA logits"*. Unlike the `== 10` patch in sglang#44445, this keeps DeepGEMM's dense FP4 indexer on SM120. | MIT |
| 4 | `adapter/sitecustomize.py` (opt-in, `DS41_DSV4_SWA_HOST_MULT`, default 11): multiply only the HiCache **SWA host pool** page count. New for v0.5.23. | v0.5.23 sizes the V4 SWA host pool as device SWA pages × `--hicache-ratio` (186 × 2 here). Cache-mode HiCache backs up one sliding window per inserted node, which means every prefill chunk, so the pool churns after a few hundred thousand tokens and prefixes whose window state is gone cannot be matched. With ×11 (4,092 pages, about 26 GB per GPU), an active 48K conversation stays reusable through 5.35M tokens of other traffic. Unpatched, it was lost by 2.35M. See results §6. | MIT |
| 5 | `tuned_configs/`: Triton block-FP8 GEMM configs for RTX PRO 6000 Blackwell Max-Q, 11 shapes, `BLOCK_SIZE_K=32`. | SGLang ships none for this GPU ("Using default W8A8 Block FP8 kernel config"). The stock tuner can't tune block [32,32] (its search space has only 64/128 K-tiles; K-tile must equal the 32-element scale group). Kernel 1.2–4.5× faster; single-stream decode +40–55%, cold prefill +28% (results §10). | — |

Nothing else is changed. Sybil's Engram row-store adapter is **not** used. v0.5.23's `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` (shared layout, `cudaHostRegister`, the GPU reads rows directly) replaces it, and the method Sybil hooked no longer exists.

Fixes 2–4 load through an import hook. They are active only when `DS41_SM120_INDEXER_FIX=1`, which the Dockerfile sets, and each startup logs `DS41 SM120: ... dense-blocks candidate indexer (use_deep_gemm_prefill=True)` on every rank.

## 3. Requirements

- 4 × RTX PRO 6000 Blackwell (96 GB, SM120). Tested on the 300 W Max-Q edition.
- An NVIDIA open driver ≥ 580 (tested on 595.99.02), Docker with Compose v2, and the NVIDIA Container Toolkit.
- **About 400 GB host RAM:** 189 GiB of locked Engram tables, about 162 GB of HiCache host pools (`HICACHE_RATIO=2` plus SWA ×11, about 40 GB per GPU), plus headroom. Tested with 512 GB DDR5-6400.
- About 520 GB of local NVMe for the checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash-1017`).
- **Host prep (required for full speed):** transparent huge pages for shmem set to `advise`. See [host/README.md](host/README.md).

## 4. Start

```bash
git clone <this repo> && cd <this repo>
sudo install -m 644 host/ds41-thp-shmem.conf /etc/tmpfiles.d/ && sudo systemd-tmpfiles --create /etc/tmpfiles.d/ds41-thp-shmem.conf
mkdir -p state
MODEL_DIRECTORY=/models/DeepSeek-V4.1-Flash-1017 docker compose up --build -d
docker compose logs -f   # first start ~7 min (weights, JIT, graphs); later starts ~3 min
```

A key is generated into `state/api-key` (mode 600) on first start. You can supply your own with `API_KEY` in `.env`. The API binds to `127.0.0.1:8010` (set `BIND_ADDRESS` to change it).

```bash
KEY=$(cat state/api-key)
curl -s http://127.0.0.1:8010/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 19 + 23?"}]}'
python3 tests/chat.py            # interactive streaming chat (pip install openai)
python3 tests/smoke.py           # stream, thinking, 8 tool round trips @18K prompt, JSON schema, image
```

On hosts with more than four GPUs, pin the engine with [compose.pin-gpus.example.yaml](compose.pin-gpus.example.yaml).

## 5. Configuration

| Variable | Default | Notes |
|---|---:|---|
| `MAX_RUNNING_REQUESTS` | 16 | Also the decode CUDA-graph batch size. Agent-shaped throughput plateaus past about 16, and p99 TTFT grows quickly. |
| `CONTEXT_LENGTH` | 409600 | The model limit is 1,048,576. 1M acceptance is pending. |
| `MEMORY_FRACTION` | 0.92 | About 7 GB free per GPU after graphs at C16. |
| `MAX_TOTAL_TOKENS` | 4200000 | FP8 KV pool, 4,199,936 tokens allocated. |
| `CHUNKED_PREFILL_SIZE` | 4096 | Tokens per prefill chunk. Larger = faster cold prefill but in-flight streams slow down more during a big load; 2048 keeps streaming at full speed, 8192 is for batch use (results §11). |
| `HICACHE_RATIO` | 2 | Main-KV host pool = 2 × GPU pool (8.4M tokens, about 14 GB per GPU). |
| `DS41_DSV4_SWA_HOST_MULT` | 11 | SWA host pool multiplier (fix 4). 4,092 pages, about 26 GB per GPU. Set it to 1 to disable. |
| | | **With ≥ 768 GB host RAM**, set `HICACHE_RATIO=4` (keep ×11): 8,184 SWA pages, about 80 GB of pools per GPU. Idle 57K conversations then reload from host in 0.17 s after 4.7M tokens of other traffic (results §6). |
| `PREFILL_DECODE_INTERVAL` | 8 | Decode rounds between prefill chunks of a long prompt, so streaming answers keep flowing during a big cold load. Costs ~19% TTFT for that load; 0 = maximum prefill throughput (results §9). |
| `DSPARK_BLOCK_SIZE` | 5 | DSpark draft tokens per step (verify window = block + 1). 5 is the size the bundled draft head was trained with; 3, 4 and 6 were within noise or slower (results §12). |
| `EXTRA_ARGS` | — | Appended to `sglang.launch_server`. |

Fixed: TP4/EP4, `dsv4` attention, `flashinfer_mxfp4` MoE, FP8 KV, `--min-free-slots-delay 1`, decoder SWA bounded replay, `deepseekv41` tool parser, `deepseek-v41` reasoning parser, write-through HiCache, `--enable-metrics` (Prometheus at `/metrics`).

## 6. Hardware notes from this machine

- **PCIe:** each GPU sits on its own CPU root port and all four train at Gen5 x16 under load. Raw P2P copy runs at 38–42 GB/s one way and 74–76 GB/s both ways per pair. With all 12 directed pairs active at once, the total is **79–90 GB/s (about 20 GB/s per GPU)**, a CPU-mesh ceiling.
- **NCCL:** by default it picks the SHM transport (about 17–20 GB/s bus bandwidth). `NCCL_P2P_LEVEL=SYS` raises all-reduce to about 25 GB/s but **collapses all-to-all to about 2 GB/s**, so don't set it for EP without measuring. See `results/raw/hw_*`.
- **Memory:** the tested host had 4 × 128 GB DDR5-6400, so only 4 of 8 memory channels were populated.
- **Tuning configs:** "Using default W8A8 Block FP8 kernel config" appears because SGLang has no tuned configs for the Max-Q device name. It is harmless.

## 7. Reproduce the benchmarks

```bash
docker cp benchmarks/sweep.sh <container>:/tmp/ && docker exec <container> bash /tmp/sweep.sh   # C=1..16 + cold 32K
python3 benchmarks/build_csv.py --sybil <log> --nothp <log> --thp <log> --out_after a.csv --out_compare b.csv
nvcc -O2 -o p2pbw benchmarks/p2pbw.cu && ./p2pbw                                            # raw P2P matrix
```

Each point is one wave of 4 × C requests, so treat differences under about 8% as noise. Repeated single-stream runs on this build gave 214, 224 and 240 tok/s.

## 8. Provenance

| Component | Source |
|---|---|
| Model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) rev `08c8657f02291e11b6f6b63e0e16014f8e748518`. It differs from Sybil's pin `1bae0e10` only by an added `chat_template.jinja`, which is unused because the `deepseekv41` tool parser selects the native encoder. Weights are not redistributed. |
| SGLang | `lmsysorg/sglang:v0.5.23-cu130` @ `sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f` (sglang 0.5.23, sglang-kernel 0.4.10, flashinfer 0.7.2, sgl-deep-gemm 0.2.3.post2, torch 2.14.1+cu130) |
| Ported fixes | [sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) @ `ce69105c` |
| Related | [sglang#44445](https://github.com/sgl-project/sglang/issues/44445) (4 × SM120 report and patches) |

## 9. License

The scripts, configuration and `adapter/sitecustomize.py` are MIT ([LICENSE](LICENSE)). The indexer-metadata fix is derived from Sybil's MIT code ([LICENSE.sybil](LICENSE.sybil)). `runtime/flash_mla_sm120.py` is derived from SGLang and stays under Apache-2.0 ([LICENSE.sglang](LICENSE.sglang)). See [NOTICE](NOTICE).
