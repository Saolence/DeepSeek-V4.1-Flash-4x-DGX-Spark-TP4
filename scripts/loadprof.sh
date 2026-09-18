#!/usr/bin/env bash
# Profile the weight-load phase of a boot on the head: py-spy dumps of the TP0 scheduler every
# 10 s, NVMe read rate and MemFree/Cached every 5 s. Run on the head after `./start-tp4.sh serve`.
#   bash scripts/loadprof.sh ~/loadprof-$(date +%Y%m%d-%H%M)
# Summarise the disk log with:
#   sed 's/readMB=//' disk.log | awk '{if(prev!=""){printf "%s +%.2f GB/s\n", $1, ($2-prev)/5000}; prev=$2}'
set -u
OUT=${1:-~/loadprof}; mkdir -p "$OUT"
L=${LOG:-logs-tp4/dsv41.log}
until docker ps --format '{{.Names}}' | grep -q '^dsv41-head$'; do sleep 2; done
until grep -aq "Load weight begin" "$L" 2>/dev/null; do sleep 2; done
( while true; do echo "$(date +%T) $(awk '$3=="nvme0n1"{printf "readMB=%.0f", $6*512/1e6}' /proc/diskstats) $(grep -E 'MemFree|MemAvailable|^Cached' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; sleep 5; done ) > "$OUT/disk.log" 2>&1 &
DP=$!
until grep -aq "Engine startup timings" "$L" 2>/dev/null; do
  PID=$(docker exec dsv41-head sh -c "ps -eo pid,args | grep scheduler_TP0 | grep -v grep | awk '{print \$1}'" 2>/dev/null | head -1)
  T=$(date +%H%M%S)
  if [ -n "$PID" ]; then
    timeout 20 docker exec dsv41-head py-spy dump --pid "$PID" > "$OUT/dump-$T.txt" 2>&1
    docker exec dsv41-head sh -c "top -b -n1 -H -p $PID | sed -n 7,16p" > "$OUT/top-$T.txt" 2>&1
  fi
  sleep 10
done
kill $DP 2>/dev/null
grep -aE "Load weight|fast load|Engine startup timings|max_total_num_tokens" "$L" | cut -c1-220 > "$OUT/timeline.txt"
echo "profile in $OUT"
