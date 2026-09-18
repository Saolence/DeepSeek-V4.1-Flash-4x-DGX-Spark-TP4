# Upstream watch (2026-09-18)

What the pieces we run on are pinned to, why, and what would let us move. Checked against
sgl-project/sglang, DeepGEMM, b12x, HuggingFace and the other public Spark recipes on 2026-09-18.

## Pinned, and why the pin stays

| Piece | We run | Upstream now | Why we stay |
|---|---|---|---|
| SGLang `dsv4.1` branch | `f80c91a4b` (2026-09-16) | head `fc3107f0d`, 176 commits later, not merged to main, no tag; umbrella PR #38798 says the branch is "being refactored and is unstable" | every head after #39671 (2026-09-16 22:15) raises `the candidate indexer needs DeepGEMM's paged sparse MQA logits` on SM121: the torch fallback was removed and the sparse kernel exists only for SM100 (DeepGEMM #432). The fix is #39914 (restores `candidate_torch.py`, base main, CI failing, untested on SM121) |
| sglang#39187 (bounded indexer prefill) | backported in `adapter/indexer_chunked_v3.py` | still OPEN and conflicting; successors #40015 / #39095 also open | nothing merged supersedes it |
| sglang#39704 (mHC medium batches) | not used | OPEN, conflicting | ±2 % here (c ≤ 16) |
| `sglang-kernel` / `sgl-deep-gemm` | 0.4.7 / 0.2.0 | latest (2026-09-13 / 09-14) | nothing newer |
| `lmsysorg/sglang:dev-dsv41` image | base for `Dockerfile` | last push 2026-09-11 (commit da64c5cbb, 684 commits behind our pin) | older than what we build |
| RoCEnante (b12x, rhys101 SG17) | pinned overlay | b12x #313 (graph-replay wedge) still open and unanswered; #383 fixed startup prep on 4x GB10 | `SGLANG_ROCE_ALLREDUCE=0` stays the escape hatch |
| Prefill TP split (rhys101 SG18) | `adapter/spark_prefill_dense.py` | rhys101 has no commits since 2026-09-14, no SG19 | nothing to port |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40` | unchanged since 2026-09-10; no V4.2, no new draft head anywhere on HF | nothing to re-download |
| DSpark block size | 5 | Mia's default is 3; two independent fleets (Mia #20, #21) measured k=3 −16 to −26 % on code/structured | keep 5 |

## Seen upstream, same idea as ours

- sglang#39365 (open draft): read only the shards holding the DSpark head. Same as the draft
  half of `adapter/fast_load.py`.
- sglang#39666 (merged to main): Engram host table with `posix_fadvise(DONTNEED)` of the checkpoint
  page cache after load, the same reason `fast_load` drops it (the KV pool is sized from
  `MemAvailable`).
- sglang#39482 (merged to main 2026-09-18): SM121 added to DeepGEMM's UE8M0 scale selection.
  Only matters on a DeepGEMM MoE path; ours is FlashInfer CUTLASS / b12x.
- vLLM #57432 (merged): DSpark's non-causal draft window let padded keys into the softmax
  (+20 % acceptance on GSM8K after the fix). Not our bug: SGLang's SM120 sparse kernel pads
  indices with −1 and masks them (`flash_mla_sm120.py`, `invalid_mask` → `-inf`).

## Leads measured here on 2026-09-18

- **`--sleep-on-idle`: adopted.** Head scheduler idle CPU 47 % → 14 %, workers 5 %; first
  response after 20 s idle 0.18–0.21 s either way; decode unchanged.
- **Engram gap (the "prestage" lead).** Live profile of 40 decode steps at c1 (rank 0): step
  48.5 ms, GPU busy 93.9 %, and 2.2–2.4 ms/step of the 3 ms idle sits before the two
  `_engram_gather_kernel` launches on real Polish prose (24–64 % row-cache hits); 1.0 ms/step on
  sparkDash's prose (96 % hits). Not the callback mechanism: a host node inside a replayed graph
  costs ~2 µs and the pinned H2D copy 26 µs (measured). It is NVMe miss latency: an O_DIRECT
  8 KB `pread` on the packed shard is ~200 µs median (p99 280–660 µs), each rank reads its own
  misses one after another inside the host callback, and the all-reduce waits for the slowest
  rank. Tried: probe the cache first and send ≥2 misses to the I/O pool — no change (2.38 vs
  2.44 ms), because at c1 a rank owns ~1.5 of the 6 rows per lookup, so a second miss per rank
  is rare. The only fix left is fetching rows *before* the graph runs, which needs the hash ids
  ahead of time: a CPU replica of the Engram hasher (n-gram over the request's tokens, bit-exact
  against the GPU one) driven from the scheduler between steps. Upside ≈ 2.4 ms/step (~5 %) at
  c1 on real text, ~0 on sparkDash prose; not built.
- **Markov W2 top-k pruning (vLLM #56694).** The bf16 vocab-sized GEMMs (LM head + Markov W2)
  are 2.8 ms/step in the same profile, so the pruning can recover at most ~1–1.5 ms/step
  (2–3 %), and it changes the draft distribution on prose, our weakest column. Not built.
- **vLLM #57432** (padded keys in the DSpark softmax): not applicable, see above.
- Real-text reference for the prose column: a 700-token English essay ("history of Krakow",
  temperature 0, warm row cache, engine otherwise idle) decodes at 44–46 tok/s at c1, against
  57 tok/s on sparkDash's prose filler; the same essay in Polish 36–40 tok/s. Acceptance, not
  Engram, is the difference. Any run with another request in flight (`running-req: 2` in the
  log) reads 20–40 % lower, so check the log before quoting a number.

## Leads not yet tried here

- LuZ-0.1.7 (luxingcom, 4x Spark TP4 ring): fused ratio-1 decode (RMSNorm + RoPE + FP4 quant +
  FlashMLA write), fused ratio-2 pair pooling, FP4 storage for ratio-1 latents. Their own
  OFF/ON table moves code 100.3 → 99.8, aggregate c1–c12 by +1 %, prose 33.3 → 36.6 tok/s
  (from a base below our real-text 44–46); the kernels are not bit-exact (fp32-ulp deltas by
  their own note) and are grafted onto a different SGLang commit + FlashInfer 0.6.18. On our
  budget the fusable tiny kernels are ~3.8 ms/step and the hc kernels 3.6 ms, so the ceiling
  here is a few percent for a large, version-bound port. Not planned.
- nktlabs Engram prestage (vLLM patch): the same idea as the CPU-hasher prefetch above.

## Watch-outs from other fleets

- Mia #23: host buddy-allocator fragmentation after the weight load on GB10 (`NV_ERR_NO_MEMORY`
  with free memory, SSH dead, ICMP alive, power cycle). Reproduced on a 4x TP4/EP2 SGLang fleet at
  512k context; same signature as our 2026-09-18 Spark_02 wedge.
- Mia #14: the September OTA's `nvidia-spark-grub-kho` (`kho=off`) caps ConnectX-7 transmit at
  12.7 Gb/s on 6.17.0-1032. Not present on our four nodes as of 2026-09-18; check
  `/proc/cmdline` after any OTA.
- Mia #21: sparkDash's prefill filler is one repeated token, so the Engram row cache inflates
  that column; random-text rows are the ones to quote.
