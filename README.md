<h1 align="center">DeepSeek-V4.1-Flash on 4x DGX Spark — tuned TP4 profile</h1>

<p align="center">A measured TP4 serving profile for <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">deepseek-ai/DeepSeek-V4.1-Flash</a> (native weights, SGLang, DSpark) on four NVIDIA DGX Spark (GB10) nodes, built on top of the MiaAI-Lab recipe.</p>

## Credits

This repository is a downstream profile of **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)** by Mia (MiaAI-Lab). The launcher (`start.sh`, `start-tp4.sh`, `boot.py`), the Engram NVMe row store, the MXFP8 b12x routing, the memory model in `docs/chunked-prefill-memory.md`, the thinking alias, the output cap, the loop abort and the overall deployment design are her work, kept here with full git history and under the same licence. The original README is preserved as [`docs/README-upstream.md`](docs/README-upstream.md); read it first for the fleet setup (NFS share, Engram packing, fabric, 3-node profile).

Other work this profile builds on:

- **kpham-sgl**, [sgl-project/sglang#39187](https://github.com/sgl-project/sglang/pull/39187): the bounded dense-indexer prefill transient, backported here as `adapter/indexer_chunked*.py`.
- **BBuf** and the SGLang `dsv4.1` branch contributors ([#39370](https://github.com/sgl-project/sglang/pull/39370), [#39646](https://github.com/sgl-project/sglang/pull/39646), [#39648](https://github.com/sgl-project/sglang/pull/39648), [#39653](https://github.com/sgl-project/sglang/pull/39653)): the decode kernel work in the optional `Dockerfile.canary` image.
- **hushengkai**, for independently reproducing the EP2 / Engram cache / shared-expert padding changes on a second 4x GB10 fleet.
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

Everything else (memory fraction 0.80, 8M-token KV pin, 1M context, NFS/Engram layout, the OpenAI serving fixes) is upstream's.

## Measured

sparkDash decode bench, 256 new tokens, temperature 0, thinking off, idle fleet, no foreign traffic (checked against the engine's `#running-req` log). Four DGX Spark, TP4/EP2, driver 580.x, `lmsysorg/sglang:dev-dsv41` base. Full record with every intermediate step: [`docs/window-20260916.md`](docs/window-20260916.md), raw outputs in [`docs/results/window-20260916/`](docs/results/window-20260916/).

### Prose decode, aggregate tok/s (per stream in brackets)

| Profile | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| upstream TP4 example (from its README) | 45.4 | 72.9 | 103.1 (26.7) | 114.1 (23.2) | 134.2 (22.0) |
| this profile, `Dockerfile` (base image) | 51.6 | 76.7 | 109.3 (28.5) | 160.9 (20.8) | 248.8 (16.7) |
| this profile, `Dockerfile.canary` (upstream dsv4.1 branch) | **55.4** | **80.9** | **118.9 (30.7)** | **178.2 (24.0)** | **277.5 (18.6)** |

### Code and structured decode, aggregate tok/s

| Profile | code c1 | code c8 | code c16 | structured c1 |
|---|---:|---:|---:|---:|
| this profile, base image | 96.7 | 446.7 | 595.3 | 104.5 |
| this profile, canary image | **100.4** | **513.3** | **838.6** | **108.0** |

### Prefill, cold, tok/s by prompt length

| Profile | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| upstream example (chunk 1024) | 3350 | 3782 | 3768 | 3531 | 3251 | – |
| this profile, base image (chunk 4096 + indexer backport) | 3532 | 4006 | 4038 | 3917 | 3230 | 2724 |
| this profile, canary image | 3174 | 3982 | 4180 | 4010 | 3499 | 2701 |

Long-context checks on the canary image: needle retrieval PASS at 131k and 262k tokens; head `MemAvailable` low-water 7.0 GiB during the 262k cold prefill (7.7 GiB on the base image).

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

## Adapters added here

All adapters are import hooks in `adapter/sitecustomize.py`, gated by an environment variable, off unless the variable is set, and each refuses to boot if the engine symbol it patches has drifted.

| File | Gate | What |
|---|---|---|
| `adapter/shared_pad_k.py` | `DSV41_SHARED_PAD_K` | Shared-expert `down_proj` K padded 576 → 640 with re-blocked scales, so the shape stays on the b12x MXFP8 kernel instead of the CUTLASS fallback |
| `adapter/indexer_chunked.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 adapted to the `dev-dsv41` image backend (`self.candidate_masks`) |
| `adapter/indexer_chunked_v2.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 verbatim, for the `candidate_metadata` backend of the `dsv4.1` branch; `sitecustomize` picks v1 or v2 from the stock source |

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
- 1M-token prompts were not re-tested with the 4096 chunk; upstream reports a 9.6 GB indexer transient at 1M with a 16k chunk on GB300.
- Sampled decode on the canary image is ~5 % behind greedy (see above).
- Single-stream prose speed is bounded by DSpark acceptance (~2–3 accepted tokens per step on prose against ~6 on code); no configuration changes that.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).
