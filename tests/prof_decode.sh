#!/bin/bash
# Profile ~30 decode steps under agent-shaped load (C=8, shared 30K prompt + 2K question, 1024 out). Runs INSIDE the engine container.
# Usage: prof_decode.sh <tag>   -> traces in /state/prof-<tag>/
set -u; T=$1; export OPENAI_API_KEY="$(cat /state/api-key)"; D=/state/prof-$T; rm -rf $D; mkdir -p $D
python3 -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:8010 --model deepseek-v4.1-flash \
  --tokenizer /models/DeepSeek-V4.1-Flash --seed 1 --disable-tqdm --dataset-name generated-shared-prefix --gsp-num-groups 1 \
  --gsp-prompts-per-group 24 --gsp-system-prompt-len 30000 --gsp-question-len 2000 --gsp-output-len 1024 --max-concurrency 8 \
  --warmup-requests 1 --output-file $D/bench.jsonl > $D/bench.log 2>&1 &
BP=$!; sleep 30
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" http://127.0.0.1:8010/start_profile \
  -d "{\"output_dir\": \"$D\", \"num_steps\": 30, \"activities\": [\"GPU\"], \"profile_prefix\": \"$T\"}"; echo
wait $BP; grep -E "Output token throughput|Median TPOT|Accept" $D/bench.log; ls -la $D | head
