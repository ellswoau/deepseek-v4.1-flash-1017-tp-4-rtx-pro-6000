#!/bin/bash
# Phase 8 sweep (runs INSIDE the engine container). Same workloads as p5_sweep.sh so results compare 1:1.
# Key comes from /state/api-key via env, never on a command line.
set -u
export OPENAI_API_KEY="$(cat /state/api-key)"
OUT=${OUT:-/state/phase8}; mkdir -p $OUT
B="python3 -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:8010 --model deepseek-v4.1-flash --tokenizer /models/DeepSeek-V4.1-Flash --seed 1 --disable-tqdm"
for C in ${CONCURRENCIES:-1 2 4 8 12 16}; do
  echo "################ steady decode C=$C (random 1024 in / 2048 out, ignore_eos)"
  $B --dataset-name random --random-input-len 1024 --random-output-len 2048 --random-range-ratio 1.0 \
     --num-prompts $((4*C)) --max-concurrency $C --warmup-requests 1 --output-file $OUT/steady.jsonl 2>&1 \
     | grep -E "Successful|Benchmark duration|Output token throughput|Peak output|Mean TTFT|Median TTFT|P99 TTFT|Median TPOT|Median ITL|Accept length"
  echo "################ agent-like C=$C (shared ~30K system prompt + 2K question, 512 out)"
  $B --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group $((4*C)) \
     --gsp-system-prompt-len 30000 --gsp-question-len 2000 --gsp-output-len 512 \
     --max-concurrency $C --warmup-requests 1 --output-file $OUT/agent.jsonl 2>&1 \
     | grep -E "Successful|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Median TPOT|Accept length"
done
echo "################ cold prefill 32K C=1 (3 unique prompts, 16 out)"
$B --seed 7 --dataset-name random --random-input-len 32768 --random-output-len 16 --random-range-ratio 1.0 \
   --num-prompts 3 --max-concurrency 1 --output-file $OUT/cold32k.jsonl 2>&1 \
   | grep -E "Successful|Input token throughput|Mean TTFT|Median TTFT"
echo SWEEP_DONE
