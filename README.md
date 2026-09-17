<h1 align="center">DeepSeek-V4.1-Flash on 4x DGX Spark — tuned TP4 profile</h1>

<p align="center">A measured TP4 serving profile for <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">deepseek-ai/DeepSeek-V4.1-Flash</a> (native weights, SGLang, DSpark) on four NVIDIA DGX Spark (GB10) nodes, built on top of the MiaAI-Lab recipe.</p>

## Credits

This repository is a downstream profile of **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)** by Mia (MiaAI-Lab). The launcher (`start.sh`, `start-tp4.sh`, `boot.py`), the Engram NVMe row store, the MXFP8 b12x routing, the memory model in `docs/chunked-prefill-memory.md`, the thinking alias, the output cap, the loop abort and the overall deployment design are her work, kept here with full git history and under the same licence. The original README is preserved as [`docs/README-upstream.md`](docs/README-upstream.md); read it first for the fleet setup (NFS share, Engram packing, fabric, 3-node profile).

Other work this profile builds on:

- **kpham-sgl**, [sgl-project/sglang#39187](https://github.com/sgl-project/sglang/pull/39187): the bounded dense-indexer prefill transient, backported here as `adapter/indexer_chunked*.py`.
- **BBuf** and the SGLang `dsv4.1` branch contributors ([#39370](https://github.com/sgl-project/sglang/pull/39370), [#39646](https://github.com/sgl-project/sglang/pull/39646), [#39648](https://github.com/sgl-project/sglang/pull/39648), [#39653](https://github.com/sgl-project/sglang/pull/39653)): the decode kernel work in the optional `Dockerfile.canary` image.
- **hushengkai**, for independently reproducing the EP2 / Engram cache / shared-expert padding changes on a second 4x GB10 fleet.
- **rhys101**, [DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8): the SG17 SGLang overlay that routes small tensor-parallel all-reduces to RoCEnante (reused with a TP4 adaptation in `Dockerfile.canary-roce`) and the SG18 native prefill TP split (`adapter/spark_prefill_dense.py`, combined with the indexer backport in `adapter/indexer_chunked_v3.py`).
- **local-inference-lab / Luke Alonso and Jason (original-el8)**, [b12x](https://github.com/local-inference-lab/b12x): RoCEnante, the one-shot RDMA all-reduce (`runtime/b12x`, Apache-2.0, frozen at the SG17 revision).
- **MiaAI-Lab/sparkDash**, the benchmark used for every number below.

## What this profile changes

Relative to the upstream TP4 example, all of it in `.env.tp4.example` plus gated adapters:

| Setting | Upstream | Here | Why (measured) |
|---|---|---|---|
| `EP_SIZE` | 4 | **2** | Two expert groups instead of four halve the per-layer straggler wait: NCCL time per step 16.5 → 10.2 ms, MoE GEMM unchanged |
| `DSV41_CACHE_GIB` / `WAYS` | 0 / 4 | **4 / 16** | Engram rows do repeat (bigram/trigram heads): 67–76 % hit rate, 4x fewer NVMe reads; 16 ways are free |
| `--min-free-slots-delay 1` | on | on | Without it the admission delayer never fills the last slot |
| `--enable-deepseek-v4-fp4-indexer` | off | **on** | FP4 DSA indexer kernel path |
| `DSPARK_BLOCK_SIZE` | 3 | **5** | k=3 is a 3-node prose result; on TP4 k=5 wins on code by ~10 % and ties on prose |
| `MAX_RUNNING_REQUESTS` | 8 | **16** | CUDA graphs to bs 16 cost ~5.6 GB and add a c16 tier (+64 % aggregate over c8) |
| `CHUNKED_PREFILL_SIZE` | 1024 | **4096** | Safe only together with the indexer backport below |
| `DSV41_SHARED_PAD_K=1` | – | **on** | Pads the shared expert's K 576 → 640 so it stops falling off the b12x MXFP8 kernel: −0.9 ms/step, bit-identical |
| `DSV41_INDEXER_CHUNKED=1` | – | **on** | sglang#39187: indexer logits scored in ≤ 2 GiB row chunks, tail-only candidate masks; 262k cold prefill keeps ≥ 7 GiB free on the head |
| prefill TP split (`SPARK_PREFILL_TP_SPLIT=1`, canary images) | – | **on** | SG18: the dense prefill indexer's query rows are partitioned across the four ranks from 32k context; each rank scores a quarter, the top-k and candidate block ids are all-gathered as ints. sparkDash prefill 128k 3499 → 4364, 262k 2701 → 3893 |
| `--enable-cache-report` | off | **on** | `usage.prompt_tokens_details.cached_tokens` on every response (also in streaming `usage`), so clients can see prefix-cache hits |

Everything else (memory fraction 0.80, 8M-token KV pin, 1M context, NFS/Engram layout, the OpenAI serving fixes) is upstream's.

## Measured

sparkDash decode bench, 256 new tokens, temperature 0, thinking off, idle fleet, no foreign traffic (checked against the engine's `#running-req` log). Four DGX Spark, TP4/EP2, driver 580.x, `lmsysorg/sglang:dev-dsv41` base. Both images were rebuilt from a fresh clone of this repository on 2026-09-17 and re-measured (base: prose c1 52.7 / c8 170, code c1 100.5; canary: prose c1 55.2 / c8 177, code c1 100.8). Full record with every intermediate step: [`docs/window-20260916.md`](docs/window-20260916.md), raw outputs in [`docs/results/window-20260916/`](docs/results/window-20260916/).

### Prose decode, aggregate tok/s (per stream in brackets)

| Profile | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| upstream TP4 example (from its README) | 45.4 | 72.9 | 103.1 (26.7) | 114.1 (23.2) | 134.2 (22.0) |
| this profile, `Dockerfile` (base image) | 51.6 | 76.7 | 109.3 (28.5) | 160.9 (20.8) | 248.8 (16.7) |
| this profile, `Dockerfile.canary` (upstream dsv4.1 branch) | 55.4 | 80.9 | 118.9 (30.7) | 178.2 (24.0) | 277.5 (18.6) |
| this profile, `Dockerfile.canary-roce` (branch + RoCEnante, 512 KiB route) | 56.9 | 82.4 | 120.4 (32.0) | 180.0 (24.1) | 275.6 (18.3) |
| this profile, `Dockerfile.canary-roce`, 2 MiB route (`SGLANG_ROCE_MAX_SIZE=2097152`) | **56.2** | – | – | **181.5 (24.6)** | **288.4 (18.7)** |

### Code and structured decode, aggregate tok/s

| Profile | code c1 | code c8 | code c16 | structured c1 |
|---|---:|---:|---:|---:|
| this profile, base image | 96.7 | 446.7 | 595.3 | 104.5 |
| this profile, canary image | 100.4 | 513.3 | 838.6 | 108.0 |
| this profile, canary + RoCEnante (512 KiB route) | 103.1 | 509.6 | **867.6** | 116.9 |
| this profile, canary + RoCEnante (2 MiB route) | **106.6** | – | 851.8 | **120.0** |
| this profile, base image + RoCEnante | 104.0 | 450.3 | 770.0 | 112.1 |

### Prefill, cold, tok/s by prompt length

| Profile | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| upstream example (chunk 1024) | 3350 | 3782 | 3768 | 3531 | 3251 | – |
| this profile, base image (chunk 4096 + indexer backport) | 3532 | 4006 | 4038 | 3917 | 3230 | 2724 |
| this profile, canary image | 3174 | 3982 | 4180 | 4010 | 3499 | 2701 |
| this profile, canary-roce + prefill TP split (v3) | **4070** | **4513** | **4554** | **4375** | **4364** | **3893** |

**Caveat on the prefill table:** sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache (`DSV41_CACHE_GIB=4`) inflates those numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)). The same canary engine on random-word text, cold, one request per size, `prompt_tokens / TTFT`:

| random text | 11.8k | 23.8k | 47.3k | 94.3k | 188.7k |
|---|---:|---:|---:|---:|---:|
| canary, tok/s | 3330 | 3666 | 3347 | 3187 | 2657 |
| canary-roce + prefill TP split (v3), tok/s | 2879 | 3612 | 3776 | 4006 | 3092 |

Use these rows for real prompts; the sparkDash column overstates by 9–20 % at 16k–128k. The v3 row's 12k value is a single cold request right after boot (the split does not engage below 32k).

Long-context checks: needle retrieval PASS at 131k, 262k and **985k** tokens on the canary image (985k cold prefill 732 s, head `MemAvailable` low-water 6.6 GiB) and on canary-roce + prefill TP split (131k 25.6 s, 262k 58 s, **985k 585 s**, low-water 5.0 GiB). The split without the chunked scoring (SG18 as published, base image) reached 503 s at 985k but left only 1.9 GiB on the head, which is why v3 keeps the 2 GiB logits budget inside each rank's partition.

## Quick start

Identical to upstream; only the example file differs.

```bash
cp .env.tp4.example .env.tp4          # fill in HEAD_IP / WORKER_* / MODEL_DIR / fabric as in docs/README-upstream.md
./start-tp4.sh doctor
./start-tp4.sh build                  # bakes the adapters and runs the in-image tests
./start-tp4.sh share && ./start-tp4.sh pack   # first time only, see upstream README
./start-tp4.sh serve                  # ./start-tp4.sh stop | status | logs | smoke
```

The boot log must show these lines, otherwise the profile is not active:

```
DSV41 shared-expert padding K: ... (5120, 576) -> (5120, 640)
DSV41 indexer chunked (sglang#39187 backport) ARMED: ...
Initialized DSpark draft runner. ... gamma=5, verify_num_draft_tokens=6
max_total_num_tokens=..., chunked_prefill_size=4096, ... max_running_requests=16
```

### Optional: the upstream `dsv4.1` branch image

`Dockerfile.canary` keeps the same base image but swaps the SGLang python tree for the upstream `dsv4.1` branch at a pinned commit (default `f80c91a4b`, 2026-09-16) and installs the kernel packages that branch pins (`sglang-kernel 0.4.7`, `sgl-deep-gemm 0.2.0`, aarch64 wheels from PyPI). It carries the branch's mHC / metadata / communication kernel work and is the faster of the two profiles in every column above. It is pinned, not tracked: refreshing it means a new tarball and a re-check that every adapter still finds its hook (the branch is refactoring file layout at the time of writing).

```bash
scripts/fetch-sglang-canary.sh                           # stages runtime/sglang-canary/python (~75 MB)
docker build -f Dockerfile.canary -t dsv41-4x-spark:canary .   # on the head and on every worker
# .env.tp4:
#   IMAGE=dsv41-4x-spark:canary
#   EXTRA_CONTAINER_ENV="DSV41_INDEXER_CHUNKED=1 SGLANG_DSPARK_FOLDED_SAMPLING=2"
./start-tp4.sh serve
```

`SGLANG_DSPARK_FOLDED_SAMPLING=2` matters: the branch folds only the greedy draft proposal into the CUDA graph by default, and sampled requests (temperature > 0, i.e. normal chat) would take the eager path. With it forced, sampled decode runs ~5 % slower than greedy on this image (it was equal on the base image); without it, ~9 % slower.

### Optional: RoCEnante for the tensor-parallel all-reduces

`Dockerfile.canary-roce` adds the SG17 SGLang overlay from rhys101's eight-Spark work on top of the canary image: every tensor-parallel SUM all-reduce of at most 512 KiB (bf16/fp32) goes through b12x's one-shot RDMA all-reduce over both RoCE rails instead of NCCL, inside the CUDA graphs, with a transport health check at every result boundary (a stalled transfer fails the step instead of hanging the rank). The overlay was written for TP8; `runtime/roce_tp4_adapt.py` relaxes it to TP4/TP8 and to one or two rails. The RDMA proxy is plain C over libibverbs, compiled on first use inside the container (`B12X_ROCE_CACHE_DIR`).

```bash
scripts/fetch-sglang-canary.sh
docker build -f Dockerfile.canary-roce -t dsv41-4x-spark:canary-roce .    # on every node
# .env.tp4:
#   IMAGE=dsv41-4x-spark:canary-roce
#   EXTRA_CONTAINER_ENV="DSV41_INDEXER_CHUNKED=1 SGLANG_DSPARK_FOLDED_SAMPLING=2 SGLANG_ROCE_ALLREDUCE=1 SGLANG_ROCE_MAX_SIZE=2097152 B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 B12X_ROCE_CACHE_DIR=/state/b12x-roce B12X_COMPILE_CACHE_DIR=/state/b12x-compile"
./start-tp4.sh serve
```

`B12X_ROCE_HCA` lists the RDMA devices to stripe across (both rails of the switched fabric here; their port-1 GID at `NCCL_IB_GID_INDEX` must be populated). The boot log must show `RoCEnante ready: world=4 hcas=...` and later `ROCE_TP8_ROUTE ... bytes=491520`. Measured on this fleet: +3 % prose c1, +8 % structured c1, +3.5 % code c16 over the canary image; needle PASS at 131k and 262k; a soak of three 16-stream code/prose waves concurrent with a 262k cold prefill completed with zero transport errors. `SGLANG_ROCE_MAX_SIZE` defaults to the overlay's 512 KiB; the 16-request decode step's all-reduce is 983 KB, so 2 MiB (b12x's own default) routes it too: c16 aggregate +4.5 %, code c1 +3 %, structured +3 %, and sampled decode becomes equal to greedy. The cost is a new transport in the decode path: b12x reports one open issue where a rank wedged under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)); the result-boundary health check in the overlay is the mitigation, and NCCL is one env change away (`SGLANG_ROCE_ALLREDUCE=0`).

### Optional: prefill TP split (with either canary image)

`adapter/spark_prefill_dense.py` is rhys101's SG18 helper with its topology check relaxed from eight ranks to four or eight; `adapter/indexer_chunked_v3.py` calls it from inside the #39187 path when a prefill chunk has at least `SPARK_PREFILL_TP_MIN_ROWS` rows (1024) and the context is at least `SPARK_PREFILL_TP_MIN_CONTEXT` (32768). Each rank scores only its slice of the query rows, in row chunks of at most 2 GiB of fp32 logits, and publishes only the tail rows of the candidate masks; the top-k and block ids travel as an int all-gather (no floating-point collective). The bitwise CPU test covers the split at world sizes 1, 2 and 4, with and without tail-only publishing, down to one row per chunk. Enable with

```
EXTRA_CONTAINER_ENV="... SPARK_PREFILL_TP_SPLIT=1 SPARK_PREFILL_TP_MIN_CONTEXT=32768 SPARK_PREFILL_TP_MIN_ROWS=1024"
```

and look for `DSV41 prefill TP split (v3) rank=0 ... end=1024` in the boot log. Decode is unaffected (the draft runner keeps the stock path). `runtime/flash_mla_sm120.canary.py` also carries SG18's scratch zero-initialisation (masked candidates gather slot 0; keeping the scratch finite avoids a NaN through a zero probability).

## Adapters added here

All adapters are import hooks in `adapter/sitecustomize.py`, gated by an environment variable, off unless the variable is set, and each refuses to boot if the engine symbol it patches has drifted.

| File | Gate | What |
|---|---|---|
| `adapter/shared_pad_k.py` | `DSV41_SHARED_PAD_K` | Shared-expert `down_proj` K padded 576 → 640 with re-blocked scales, so the shape stays on the b12x MXFP8 kernel instead of the CUTLASS fallback |
| `adapter/indexer_chunked.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 adapted to the `dev-dsv41` image backend (`self.candidate_masks`) |
| `adapter/indexer_chunked_v2.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 verbatim, for the `candidate_metadata` backend of the `dsv4.1` branch (kept for reference) |
| `adapter/indexer_chunked_v3.py` + `spark_prefill_dense.py` | `DSV41_INDEXER_CHUNKED`, `SPARK_PREFILL_TP_SPLIT` | v2 plus the SG18 prefill TP split inside the chunked path; `sitecustomize` picks v1 or v3 from the stock source |

Tests: `tests/test_indexer_chunked.py` and `tests/test_indexer_chunked_v2.py` lift the stock function out of the engine's source, drive it and the backport with deterministic fake kernels, and require bitwise-equal `page_indices`, `raw_indices` and candidate masks over six scenarios (ragged batches, empty requests, full and tail-only mask publishing, mask consumption, one row per chunk up to a single chunk). Both run inside the image build (`Dockerfile` runs v1, `Dockerfile.canary` runs v2), together with upstream's thinking-alias, output-cap and loop-abort tests.

## Measurement notes

- Bench with sparkDash's decode and prefill benches, never with a hand-rolled loop; the first two points after a boot read low (cold Engram row cache).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic landed in the window; the tables above were taken with none.
- Greedy text equality is not a usable correctness gate on this stack: identical cold prompts of ~70k tokens produce different greedy continuations run to run. Correctness of the indexer backport rests on the bitwise CPU tests, upstream's in-forward `page_indices` comparison, the needle tests and the benches.
- `scripts/window-20260916.sh` is the runbook that produced the tables (preflight, build, two boots, benches, rollback).

## Rollback

Upstream's profile is one env change away: `MAX_RUNNING_REQUESTS=8`, `CHUNKED_PREFILL_SIZE=1024`, `EXTRA_CONTAINER_ENV=""` (and `EP_SIZE=4`, `DSV41_CACHE_GIB=0` if you want the exact upstream example). The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The canary image's 4k-token prefill is ~6 % slower than the base image; everything from 16k up is faster.
- Sampled decode on the canary image is ~5 % behind greedy (see above).
- Single-stream prose speed is bounded by DSpark acceptance (~2–3 accepted tokens per step on prose against ~6 on code); no configuration changes that.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).
