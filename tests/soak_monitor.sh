#!/bin/bash
# Soak monitor: one CSV row per minute (engine, GPUs, host memory, cache, kernel errors) for $1 hours (default 24.2).
# Prints a short line every 5 minutes. Output: $SOAK_OUT/monitor.csv (default /opt/ds41/evidence/soak).
set -u
H=${1:-24.2}; O=${SOAK_OUT:-/opt/ds41/evidence/soak}; mkdir -p $O; F=$O/monitor.csv
KEY=$(cat /opt/ds41/v0523/state/api-key); END=$(( $(date +%s) + ${H%.*}*3600 + 720 ))
START=$(date '+%F %T')
edac() { cat /sys/devices/system/edac/mc/mc*/$1_count 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
[ -s $F ] || echo "time,health,state,restarts,running,queue,token_usage,swa_usage,cache_hit,gen_tps,accept_len,hicache_host_used,hicache_dropped,gpu_temp_max,gpu_power_sum,gpu_mem_max_mib,gpu_sm_clk_min,throttle_any,host_avail_gb,md0,kernel_err_new,edac_ce,edac_ue" > $F
n=0
while [ $(date +%s) -lt $END ]; do
  h=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8010/health)
  st=$(docker inspect -f '{{.State.Status}},{{.RestartCount}}' v0523-ds41b-1 2>/dev/null || echo missing,na)
  M=$(curl -s -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8010/metrics | grep -v '^#')
  g() { echo "$M" | grep -E "^sglang:$1\{" | grep -E 'tp_rank="0"' | head -1 | awk '{print $2}'; }
  drop=$(echo "$M" | grep -E '^sglang:hicache_dropped_tokens_total' | awk '{s+=$2} END{print s+0}')
  G=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,clocks.sm,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader,nounits)
  gt=$(echo "$G" | awk -F', ' '{if($1>m)m=$1} END{print m}'); gp=$(echo "$G" | awk -F', ' '{s+=$2} END{printf "%.0f",s}')
  gm=$(echo "$G" | awk -F', ' '{if($3>m)m=$3} END{print m}'); gc=$(echo "$G" | awk -F', ' 'NR==1||$4<m{m=$4} END{print m}')
  thr=$(echo "$G" | awk -F', ' '{for(i=5;i<=8;i++) if($i=="Active"){c++;break}} END{print c+0}')
  av=$(awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo)
  md=$(grep -A1 '^md0' /proc/mdstat | grep -oE '\[[U_]+\]' | head -1)
  ke=$(journalctl -k -q --no-pager --since "$START" 2>/dev/null | grep -ciE "xid|nvrm.*error|edac|mce:|hardware error|i/o error|md/raid")
  echo "$(date -u +%FT%TZ),$h,$st,$(g num_running_reqs),$(g num_queue_reqs),$(g token_usage),$(g swa_token_usage),$(g cache_hit_rate),$(g gen_throughput),$(g spec_accept_length),$(g hicache_host_used_tokens),$drop,$gt,$gp,$gm,$gc,$thr,$av,$md,$ke,$(edac ce),$(edac ue)" >> $F
  if [ $((n % 5)) = 0 ] || [ "$h" != 200 ]; then
    tail -1 $F | awk -F, '{printf "[mon %s] health %s %s restarts=%s run=%s q=%s kv=%s swa=%s hit=%s gen=%.0f acc=%s host_kv=%s drop=%s | GPU max %sC %sW sum, mem %s MiB, clk min %s, throttle %s | host avail %s GB | md0 %s | new kernel err %s edac ce/ue %s/%s\n", substr($1,12,5),$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23}'
  fi
  n=$((n + 1)); sleep 60
done
echo "SOAK_MONITOR_DONE $(date -u +%T)"
