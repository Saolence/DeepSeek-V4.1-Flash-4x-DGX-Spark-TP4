#!/usr/bin/env bash
# Maintenance-window runbook, 2026-09-16: two staged changes, measured one at a time.
#   A) MAX_RUNNING_REQUESTS 8 -> 16           (decode aggregate at c>8)
#   B) sglang#39187 backport (DSV41_INDEXER_CHUNKED=1) + CHUNKED_PREFILL_SIZE 1024 -> 4096
# Run ON Spark_01 as knapcio:  bash window.sh <phase>
# Phases: preflight | build | bootA | benchA | bootB | benchB | rollback
# Every phase is idempotent and prints what it did; read the numbers before the next one.
set -euo pipefail
cd ~/dsv41-mia
DASH=http://127.0.0.1:5555/api/sparks/spark-01/llm
TS=$(date +%Y%m%d-%H%M)
OUT=~/dsv41-window-20260916; mkdir -p "$OUT"

log(){ printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_ready(){  # until "server is fired up" in this boot or the head dies
  local since; since=$(date -u +%Y-%m-%dT%H:%M:%S)
  for i in $(seq 1 60); do
    docker ps --format '{{.Names}}' | grep -q '^dsv41-head$' || { echo "HEAD DIED"; docker logs --tail 40 dsv41-head; return 1; }
    docker logs --since "$since" dsv41-head 2>&1 | grep -aq 'server is fired up' && break
    sleep 15
  done
  docker logs --since "$since" dsv41-head 2>&1 | grep -aE 'max_total_num_tokens=|gamma=|INDEXER|indexer chunked|max_running_requests' | cut -c1-200
}

bench(){  # bench <type> <concurrencies> -> prints per/agg per level
  curl -s -X POST "$DASH/bench" -H 'Content-Type: application/json' \
    -d "{\"port\":8888,\"concurrencies\":[$2],\"maxTokens\":256,\"promptType\":\"$1\"}" >/dev/null
  for i in $(seq 1 90); do sleep 4; curl -s "$DASH/bench" | python3 -c 'import json,sys; sys.exit(1 if json.load(sys.stdin)["active"] else 0)' && break; done
  curl -s "$DASH/bench" | python3 -c 'import json,sys; d=json.load(sys.stdin)["last"]; [print(d["config"]["promptType"], "c%d"%r["concurrency"], "per", r["meanDecodeTps"], "agg", r["aggregateDecodeTps"], "ttft", r["meanTtftMs"], "err", r.get("error")) for r in d["results"]]' | tee -a "$OUT/bench-$TS.txt"
}

prefill_bench(){
  curl -s -X POST "$DASH/prefill-bench" -H 'Content-Type: application/json' \
    -d '{"port":8888,"contextSizes":[4096,16384,32768,65536,131072,262144]}' >/dev/null
  for i in $(seq 1 120); do sleep 10; curl -s "$DASH/prefill-bench" | python3 -c 'import json,sys; sys.exit(1 if json.load(sys.stdin)["active"] else 0)' && break; done
  curl -s "$DASH/prefill-bench" | python3 -c 'import json,sys; d=json.load(sys.stdin)["last"]; [print("prefill", r["promptTokens"], "tok/s", r["prefillTps"], "ttft_ms", r["ttftMs"], "err", r.get("error")) for r in d["results"]]' | tee -a "$OUT/prefill-$TS.txt"
}

greedy_probe(){  # greedy_probe <label>: 60k-token deterministic prompt, saves the text for equivalence
  python3 - "$1" "$OUT" <<'PY'
import json,sys,urllib.request,hashlib,time
label,out=sys.argv[1],sys.argv[2]
words=("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon").split()
body=" ".join(words[(i*7+i//13)%len(words)]+str(i%97) for i in range(45000))  # ~60k tokens, no repetition loop
msg=[{"role":"user","content":"Here is a long token stream:\n"+body+"\n\nSummarise in 5 sentences what pattern the stream follows, then list the 10th, 100th and 1000th items."}]
req={"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":200,"chat_template_kwargs":{"thinking":False},"messages":msg}
t=time.time()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",data=json.dumps(req).encode(),headers={"Content-Type":"application/json"}),timeout=1800)
d=json.load(r); txt=d["choices"][0]["message"]["content"]; el=time.time()-t
open(f"{out}/greedy-{label}.txt","w").write(txt)
print(label,"prompt_tokens",d["usage"]["prompt_tokens"],"wall_s",round(el,1),"sha",hashlib.sha256(txt.encode()).hexdigest()[:16])
PY
}

case "${1:-}" in
preflight)
  log "engine traffic in the last 5 min (must be ~0 before stopping):"
  docker logs --since 5m dsv41-head 2>&1 | grep -ac 'POST /v1/chat' || true
  log "h3 / other GPU holders on the head (must be none):"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
  pgrep -af 'vllm serve' | grep -v pgrep || true
  log "staged env:"; grep -nE '^MAX_RUNNING_REQUESTS|^CHUNKED_PREFILL_SIZE|^EXTRA_CONTAINER_ENV' .env.tp4
  ;;
build)
  log "stopping engine and building the image with the new adapters + in-image tests"
  ./start-tp4.sh stop || true
  docker tag dsv41-4x-spark:local dsv41-4x-spark:pre-window-$TS
  for h in 10.100.96.1 10.100.96.3 10.100.96.4; do ssh -i ~/.ssh/id_ed25519_nvsync_cluster_assistant -o StrictHostKeyChecking=no knapcio@$h "docker tag dsv41-4x-spark:local dsv41-4x-spark:pre-window-$TS"; done
  ./start-tp4.sh build 2>&1 | tee "$OUT/build-$TS.log" | grep -E 'info|\[\+\]|passed|Error|error' | tail -20
  ;;
bootA)
  log "boot A: MAX_RUNNING_REQUESTS=16, indexer chunked OFF, chunk 1024 (today's config otherwise)"
  sed -i 's/^MAX_RUNNING_REQUESTS=.*/MAX_RUNNING_REQUESTS=16/' .env.tp4
  sed -i 's/^CHUNKED_PREFILL_SIZE=.*/CHUNKED_PREFILL_SIZE=1024/' .env.tp4
  sed -i 's/ DSV41_INDEXER_CHUNKED=1//' .env.tp4
  ./start-tp4.sh stop || true
  nohup ./start-tp4.sh serve > "$OUT/serve-A-$TS.log" 2>&1 &
  sleep 30; wait_ready
  ;;
benchA)
  log "bench A (idle engine!): decode sweeps + greedy probe A"
  bench prose "1,2,4,8,16"; bench code "1,8,16"; bench structured "1"
  greedy_probe A | tee -a "$OUT/greedy-$TS.txt"
  log "compare with 2026-09-16 clean: prose 51.4/77.4/105.3/153.3, code 97.1/-/361.6(c8), structured 104.1"
  ;;
bootB)
  log "boot B: + DSV41_INDEXER_CHUNKED=1, CHUNKED_PREFILL_SIZE=4096"
  grep -q 'DSV41_INDEXER_CHUNKED=1' .env.tp4 || sed -i 's/^EXTRA_CONTAINER_ENV="\(.*\)"$/EXTRA_CONTAINER_ENV="\1 DSV41_INDEXER_CHUNKED=1"/' .env.tp4
  sed -i 's/^CHUNKED_PREFILL_SIZE=.*/CHUNKED_PREFILL_SIZE=4096/' .env.tp4
  grep -nE '^CHUNKED_PREFILL_SIZE|^EXTRA_CONTAINER_ENV' .env.tp4
  ./start-tp4.sh stop || true
  nohup ./start-tp4.sh serve > "$OUT/serve-B-$TS.log" 2>&1 &
  sleep 30; wait_ready
  docker logs dsv41-head 2>&1 | grep -a 'indexer chunked' | tail -1 || echo "!! adapter did not arm"
  ;;
benchB)
  log "bench B: greedy probe B (must match A byte for byte), prefill sweep, head MemAvailable during the 262k prefill"
  greedy_probe B | tee -a "$OUT/greedy-$TS.txt"
  cmp "$OUT/greedy-A.txt" "$OUT/greedy-B.txt" && echo "GREEDY EQUIVALENT" || echo "!! GREEDY DIFFERS (expected only top-k tie order on repeated text; inspect)"
  ( for i in $(seq 1 100); do awk '/MemAvailable/{print strftime("%H:%M:%S"), $2/1048576 " GiB"}' /proc/meminfo; sleep 5; done ) > "$OUT/memavail-B-$TS.txt" &
  prefill_bench
  kill %1 2>/dev/null || true
  sort -k2 -n "$OUT/memavail-B-$TS.txt" | head -1 | sed 's/^/head MemAvailable low-water: /'
  bench prose "1,8"; bench code "1,8"
  log "compare prefill with today: 4k 3390, 16k 3866, 32k 3877, 64k 3718, 128k 3211 tok/s (chunk 1024 + adaptive)"
  ;;
rollback)
  log "rollback to today's serving config (MRR 8, chunk 1024, indexer chunked off); image stays (adapter is gated off)"
  sed -i 's/^MAX_RUNNING_REQUESTS=.*/MAX_RUNNING_REQUESTS=8/; s/^CHUNKED_PREFILL_SIZE=.*/CHUNKED_PREFILL_SIZE=1024/; s/ DSV41_INDEXER_CHUNKED=1//' .env.tp4
  ./start-tp4.sh stop || true; nohup ./start-tp4.sh serve > "$OUT/serve-rollback-$TS.log" 2>&1 & sleep 30; wait_ready
  ;;
*) echo "usage: $0 preflight|build|bootA|benchA|bootB|benchB|rollback"; exit 2;;
esac
