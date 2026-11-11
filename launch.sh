#!/bin/bash
# Starts SGLang for DeepSeek-V4.1-Flash on 4 x SM120. Creates /state/api-key (mode 600) on first run.
# The key is passed to SGLang but never printed.
set -euo pipefail
umask 077
if [ -n "${API_KEY:-}" ]; then printf '%s' "$API_KEY" > /state/api-key
elif [ ! -s /state/api-key ]; then python3 -c 'import secrets;print(secrets.token_urlsafe(32),end="")' > /state/api-key; fi
KEY=$(cat /state/api-key)

shmem=$(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null || true)
case "$shmem" in *"[never]"*|*"[deny]"*)
  echo "WARNING: transparent_hugepage/shmem_enabled is '$shmem'. Engram host-table lookups will be ~10x slower." >&2
  echo "         See host/README.md (set it to 'advise')." >&2;; esac

exec python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash --served-model-name deepseek-v4.1-flash \
  --trust-remote-code --load-format safetensors \
  --host 0.0.0.0 --port 8010 --api-key "$KEY" \
  --tp-size 4 --ep-size 4 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "${CONTEXT_LENGTH:-409600}" \
  --mem-fraction-static "${MEMORY_FRACTION:-0.92}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-4200000}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-16}" \
  --cuda-graph-max-bs-decode "${MAX_RUNNING_REQUESTS:-16}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}" \
  --min-free-slots-delay 1 --random-seed 0 \
  --prefill-decode-interval "${PREFILL_DECODE_INTERVAL:-8}" \
  --speculative-algorithm DSPARK --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE:-5}" \
  --enable-decoder-swa-bounded-replay \
  --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 \
  --enable-hicache --hicache-ratio "${HICACHE_RATIO:-2}" --hicache-write-policy write_through \
  --enable-metrics --watchdog-timeout 3600 \
  ${EXTRA_ARGS:-}
