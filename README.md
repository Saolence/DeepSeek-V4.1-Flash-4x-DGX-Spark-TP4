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

## What runs in production (2026-09-17)

One image, one env file. Everything in the tables below labelled **production** is this stack:

| Layer | Setting | Status | Why |
|---|---|---|---|
| Image | `Dockerfile.canary-roce` = upstream `dsv4.1` branch at `f80c91a4b` + RoCEnante overlay + all adapters | **on** | fastest decode of the three images (branch kernels + RDMA all-reduce) |
| Slots | `MAX_RUNNING_REQUESTS=16` | on | adds the c16 tier; c1–c8 unchanged |
| Experts | `EP_SIZE=2`, `--enable-deepseek-v4-fp4-indexer` | on | straggler wait halved; kernel path |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16` | on | 67–76 % row-cache hits on real text |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | on | k=5 wins on code, ties on prose; forced fold keeps sampled decode equal to greedy on the branch |
| Prefill | `CHUNKED_PREFILL_SIZE=4096` + `DSV41_INDEXER_CHUNKED=1` (v3) + `SPARK_PREFILL_TP_SPLIT=1` | on | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks; 985k prompt leaves 5 GiB on the head |
| Shared expert | `DSV41_SHARED_PAD_K=1` | on | keeps the K=576 shape on the b12x kernel, −0.9 ms/step, bit-identical |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` | on | TP SUM all-reduces up to 2 MiB over RDMA on both rails; 2 MiB covers the 16-slot step (983 KB) |
| NCCL | `IB_HCA=rocep1s0f0,roceP2p1s0f0` | on | neutral within noise, kept for the remaining collectives |
| Fabric | switched RoCE, tree reachable | on | every default assumes a switch; a switchless ring sets `NCCL_SWITCHLESS_RING_ONLY=1` instead (see below) |
| Serving | `--enable-cache-report`, `--min-free-slots-delay 1`, `DSV41_MAX_NEW_TOKENS`, loop abort, thinking alias | on | cached-token usage for clients; the rest is upstream's |
| Weight loading | `DSV41_FAST_LOAD=1` (+ `--model-loader-extra-config {"num_threads":1}`) | **on** | engine start 343 s → 111–124 s, bytes identical, decode/prefill/needle unchanged; costs 3–13 % of the KV pool (6.71–7.27 M vs 7.47–7.82 M tokens on the same image), the one trade-off in this table ([docs/fast-load.md](docs/fast-load.md)) |
| Rust image processor | `SGLANG_RUST_BUILD_MODE=never` | off | the branch's `cargo` probe can hang the head before the HTTP server starts; PIL path is used |
| Adaptive chunk sizer | `DSV41_ADAPTIVE_CHUNK` | off | superseded by the bounded indexer; it would only shrink chunks needlessly |
| DSpark SPS table / ragged verify | `DSPARK_SPS_TABLE` | off (file absent) | crashes the Engram path on this model; verify-all schedule stays |
| NVFP4 checkpoint (`nvidia/DeepSeek-V4.1-Flash-NVFP4`) | – | not used | routed experts only, no bandwidth saved on GB10, +16 GiB, DSpark unvalidated |

Rollback to any earlier point is an env change: `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch.

## What this profile changes

Relative to the upstream TP4 example, all of it in `.env.tp4.example` plus gated adapters:

| Setting | Upstream | Here | Why (measured) |
|---|---|---|---|
| `EP_SIZE` | 4 | **2** | Two expert groups instead of four halve the per-layer straggler wait: NCCL time per step 16.5 → 10.2 ms, MoE GEMM unchanged |
| `DSV41_CACHE_GIB`/`WAYS` | 0/4 | **4/16** | Engram rows do repeat (bigram/trigram heads): 67–76 % hit rate, 4x fewer NVMe reads; 16 ways are free |
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

sparkDash decode bench, 256 new tokens, temperature 0, thinking off, idle fleet, no foreign traffic (checked against the engine's `#running-req` log). Four DGX Spark, TP4/EP2, driver 580.x, `lmsysorg/sglang:dev-dsv41` base. Run-to-run spread between boots of the same configuration is about ±2 % on c1, so differences inside that band are noise. Both images were rebuilt from a fresh clone of this repository on 2026-09-17 and re-measured (base: prose c1 52.7 / c8 170, code c1 100.5; canary: prose c1 55.2 / c8 177, code c1 100.8). Full record with every intermediate step: [`docs/window-20260916.md`](docs/window-20260916.md), raw outputs in [`docs/results/window-20260916/`](docs/results/window-20260916/).

### Prose decode, aggregate tok/s (per stream in brackets)

| Profile | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| upstream TP4 example (from its README) | 45.4 | 72.9 | 103.1 (26.7) | 114.1 (23.2) | 134.2 (22.0) |
| this profile, `Dockerfile` (base image) | 51.6 | 76.7 | 109.3 (28.5) | 160.9 (20.8) | 248.8 (16.7) |
| this profile, `Dockerfile.canary` (upstream dsv4.1 branch) | 55.4 | 80.9 | 118.9 (30.7) | 178.2 (24.0) | 277.5 (18.6) |
| **production** (`Dockerfile.canary-roce`, 2 MiB route, prefill TP split, both rails, fast load; 2026-09-18) | **57.5** | **79.2** | **121.6 (32.3)** | **183.4 (24.9)** | **285.0 (18.8)** |

### Code and structured decode, aggregate tok/s

| Profile | code c1 | code c8 | code c16 | structured c1 |
|---|---:|---:|---:|---:|
| this profile, base image | 96.7 | 446.7 | 595.3 | 104.5 |
| this profile, canary image | 100.4 | 513.3 | 838.6 | 108.0 |
| **production** (see above) | **107.7** | **539.2** | **863.6** | **116.7** |

### Prefill, cold, tok/s by prompt length

| Profile | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| upstream example (chunk 1024) | 3350 | 3782 | 3768 | 3531 | 3251 | – |
| this profile, base image (chunk 4096 + indexer backport) | 3532 | 4006 | 4038 | 3917 | 3230 | 2724 |
| this profile, canary image | 3174 | 3982 | 4180 | 4010 | 3499 | 2701 |
| **production** (canary-roce + prefill TP split + fast load) | **2587** | **4641** | **4601** | **4647** | **4194** | **3910** |

**Caveat on the prefill table:** sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache (`DSV41_CACHE_GIB=4`) inflates those numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)). The same canary engine on random-word text, cold, one request per size, `prompt_tokens / TTFT`:

| random text | 11.8k | 23.8k | 47.3k | 94.3k | 188.7k |
|---|---:|---:|---:|---:|---:|
| canary, tok/s | 3330 | 3666 | 3347 | 3187 | 2657 |
| production, tok/s | 2879 | 3612 | 3776 | 4006 | 3092 |

Use these rows for real prompts; the sparkDash column overstates by 9–20 % at 16k–128k. The v3 row's 12k value is a single cold request right after boot (the split does not engage below 32k).

Long-context checks: needle retrieval PASS at 131k, 262k and **985k** tokens on the canary image (985k cold prefill 732 s, head `MemAvailable` low-water 6.6 GiB) and on production (131k 25.6 s, 262k 58 s, **985k 585 s**, low-water 5.0 GiB). The split without the chunked scoring (SG18 as published, base image) reached 503 s at 985k but left only 1.9 GiB on the head, which is why v3 keeps the 2 GiB logits budget inside each rank's partition.

### Boot time

| | stock loader | fast load (`DSV41_FAST_LOAD=1`) |
|---|---:|---:|
| target `load_weight` (rank 0 / 1 / 2 / 3) | 225–246 / 95 / 246 / 114 s | 71–74 / 83 / 73 / 71 s |
| draft `load_weight` | 38–50 s | 3–8 s |
| engine start to ready (`scheduler_e2e`) | 343–354 s | 124–129 s |
| `max_total_num_tokens` (KV pool, same image, same night) | 7.47–7.82 M | 7.27 M and 6.71 M on two boots (pinned buffers); 6.2–6.8 M with the earlier mmap buffers |

Same image, gate on versus off, 2026-09-18. Decode, prefill and the needle test are unchanged. The KV pool is 3–13 % smaller (it also varies more from boot to boot): SGLang sizes it from the head's `MemAvailable` right after the loads, and ~0.8 GB less is available then with the fast loader (with pageable buffers it was 1.5 GB, traced to driver staging memory; the remainder shows only as mapped file pages of the scheduler process). Flip `DSV41_FAST_LOAD=0` if the last 0.5–1 M tokens of pool matter more than 220 s per boot. Profile, dead ends and raw snapshots: [docs/fast-load.md](docs/fast-load.md).

## Quality gate

Speed changes here are meant to be lossless: same weights, every draft token verified by the target, backports bitwise-equal to the stock path in the CPU tests. Because RoCEnante sums in a different order than NCCL and the branch ships different mHC kernels, the numerics are not identical, so the profile is also scored. `scripts/qeval.py` runs 75 auto-scored tasks (code executed against hidden asserts, JSON schema-checked, numeric answers matched, format constraints enforced, prose checked for degeneration; no LLM judge), one request at a time, temperature 0, and compares two runs pairwise with McNemar's exact test.

| Run (2026-09-17, same day, same fleet) | pass | broke | fixed | p |
|---|---:|---:|---:|---:|
| base image, same env (`dsv41-4x-spark:local`, chunk 4096, indexer backport) | 71/75 | – | – | – |
| **production** (branch + RoCEnante + prefill TP split) | **72/75** | 0 | 1 (`json_count`) | 1.000 |
| reference: upstream example, 2026-09-11 | 71/75 | | | |

The three tasks that fail on every stack (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the upstream example. Raw results: [`docs/results/quality-20260917/`](docs/results/quality-20260917/). Run it from a worker, not from the head (it executes model-generated Python).

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
#   BUILD_DOCKERFILE=Dockerfile.canary   # or let `./start-tp4.sh build` build it everywhere
#   EXTRA_CONTAINER_ENV="DSV41_INDEXER_CHUNKED=1 SGLANG_DSPARK_FOLDED_SAMPLING=2"
./start-tp4.sh serve
```

`SGLANG_RUST_BUILD_MODE=never` is required on the branch images: the dsv4.1 tree probes a Rust toolchain to build its image preprocessor, and that `cargo --version` call can hang before the HTTP server starts (workers come up, `/health` never answers). `never` keeps the PIL image path.

`SGLANG_DSPARK_FOLDED_SAMPLING=2` matters: the branch folds only the greedy draft proposal into the CUDA graph by default, and sampled requests (temperature > 0, i.e. normal chat) would take the eager path. With it forced, sampled decode runs ~5 % slower than greedy on this image (it was equal on the base image); without it, ~9 % slower.

`./start-tp4.sh build` compiles `BUILD_DOCKERFILE` (default `Dockerfile`, and it has to stay inside the repository so the rsync that stages the workers sees it) and tags the result `$IMAGE`; `BUILD_ARGS` carries anything else the recipe needs, e.g. `--build-arg PIP_INDEX=https://<mirror>/simple` on a network without pypi.org.

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

`B12X_ROCE_HCA` lists the RDMA devices to stripe across (both rails of the switched fabric here; their port-1 GID at `NCCL_IB_GID_INDEX` must be populated). The boot log must show `RoCEnante ready: world=4 hcas=...` and later `ROCE_TP8_ROUTE ... bytes=491520`. Measured on this fleet (512 KiB route, same boot as the canary row): +3 % prose c1, +8 % structured c1, +3.5 % code c16 over the canary image; needle PASS at 131k and 262k; a soak of three 16-stream code/prose waves concurrent with a 262k cold prefill completed with zero transport errors. `SGLANG_ROCE_MAX_SIZE` defaults to the overlay's 512 KiB; the 16-request decode step's all-reduce is 983 KB, so 2 MiB (b12x's own default) routes it too: c16 aggregate +4.5 %, code c1 +3 %, structured +3 %, and sampled decode becomes equal to greedy. The cost is a new transport in the decode path: b12x reports one open issue where a rank wedged under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)); the result-boundary health check in the overlay is the mitigation, and NCCL is one env change away (`SGLANG_ROCE_ALLREDUCE=0`).

### Optional: switchless ring (no RoCE switch)

Every default above assumes a switched fabric. If the four Sparks are cabled as a **ring**
(a-b-c-d-a, one DAC per adjacency, no switch) the stack does not boot on those defaults:
NCCL builds a tree as well as the ring, the tree wants a direct path between opposite nodes
(rank0 ↔ rank2) which a four-node ring does not have, and RoCE queue pairs do not follow IP
routing, so the tree never connects and `ncclCommInitRank` dies with
`NCCL error: unhandled system error`. No counter and no `/health` ever come up.

`NCCL_SWITCHLESS_RING_ONLY=1` fixes it. It is off by default and every other deployment is
unchanged when it is off — the switch only decides whether the ring environment and the
overlay mount are injected:

```ini
NCCL_SWITCHLESS_RING_ONLY=1
NCCL_ALGO=Ring
NCCL_P2P_LEVEL=SYS
```

The switch then injects `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`,
`NCCL_SKIP_TREE_CONNECT=1`, `NCCL_IB_SUBNET_PREFIX_LEN=24`, `NCCL_MIN_NCHANNELS=4` and
`NCCL_P2P_LEVEL=SYS` into the head **and every worker**, and mounts the patched library
**over** the image's pip NCCL (`NCCL_PIP_SO`) rather than on `LD_LIBRARY_PATH` — two visible
NCCL runtimes make DeepEP's `check_nccl_so()` abort before NCCL is initialised.
`NCCL_OVERLAY_PIP` defaults to following the switch and can be enabled on its own.

It needs a **patched NCCL** in `NCCL_HOST_DIR` (FujitsuPolycom/sparkring's
`switchless-cycle` / `skip-tree-pat` patches) on every node, and `NFS_SHARE=0` with a
per-node checkpoint, because a ring has no fabric-wide NFS path. `NFS_SHARE=0` is a
pre-existing switch that did not work — `cmd_share` always stood the exporter up, so
`serve` re-shared and replaced the local volumes. It is a real no-op now, and `serve`
refuses a worker whose `dsv41-weights` volume is still NFS-backed from an earlier
`NFS_SHARE=1` run, which would otherwise read over NFS with the probe passing. `./start.sh doctor` validates the configuration and every
rank's HCA/GID before any container is replaced, and `serve` treats a failure as fatal:

```
[+] switchless ring: config OK (NNODES=4 TP=4 EP=2, IB_HCA=rocep1s0f0,rocep1s0f1)
[+] switchless ring: head preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.2 preflight OK (RoCEv2 GID index 3)
```

`EP_SIZE` stays free here (`1 <= EP_SIZE <= TP_SIZE`); only `NNODES == TP_SIZE == 4` is
required, because the ring spans the tensor-parallel group. Expect ring bandwidth, not
switched: opposite ranks talk through a transit node, so the bisection is one link, not two.
Cabling, addressing, the `NFS_SHARE=0` migration, pitfalls and the full benchmark panel
(prefill 1k-64k, decode prose and code at 1-8 streams, with the sparkDash filler caveat)
are in [`docs/switchless-ring.md`](docs/switchless-ring.md).

Before listing more than two devices in `IB_HCA`, read the same document's
["Devices past the second are never advertised"](docs/switchless-ring.md#devices-past-the-second-are-never-advertised):
NCCL accepts the extra devices, publishes listener GIDs for only the first two, and
reports nothing — so a four-device board serves on half of it until the dual-PCI-domain
patch and its flags are in place. `doctor` warns, and the port counters are the proof. The ring configuration came from
[MiaAI-Lab#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3) /
[#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19), with the NCCL
patch from [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).

The decode table in [`docs/switchless-ring.md`](docs/switchless-ring.md) (prose c1 60.5, code c1 107.5 at 400 output tokens) is the same stack measured on the author's ring; re-run here with a 400-token window the switched production profile gives prose c1 58–60 and code c1 107.4, i.e. the ring neither adds nor costs decode speed. Its prefill column is lower, as the ring's single-link bisection predicts.

### Optional: prefill TP split (with either canary image)

`adapter/spark_prefill_dense.py` is rhys101's SG18 helper with its topology check relaxed from eight ranks to four or eight; `adapter/indexer_chunked_v3.py` calls it from inside the #39187 path when a prefill chunk has at least `SPARK_PREFILL_TP_MIN_ROWS` rows (1024) and the context is at least `SPARK_PREFILL_TP_MIN_CONTEXT` (32768). Each rank scores only its slice of the query rows, in row chunks of at most 2 GiB of fp32 logits, and publishes only the tail rows of the candidate masks; the top-k and block ids travel as an int all-gather (no floating-point collective). The bitwise CPU test covers the split at world sizes 1, 2 and 4, with and without tail-only publishing, down to one row per chunk. Enable with

```
EXTRA_CONTAINER_ENV="... SPARK_PREFILL_TP_SPLIT=1 SPARK_PREFILL_TP_MIN_CONTEXT=32768 SPARK_PREFILL_TP_MIN_ROWS=1024"
```

and look for `DSV41 prefill TP split (v3) rank=0 ... end=1024` in the boot log. Decode is unaffected (the draft runner keeps the stock path). The helper refuses thresholds below 32768 tokens / 1024 rows at boot (`ValueError`), and a sweep with that floor relaxed to 16k gained nothing outside the ±5–10 % run-to-run spread of short prefills, so 32768 stays. `runtime/flash_mla_sm120.canary.py` also carries SG18's scratch zero-initialisation (masked candidates gather slot 0; keeping the scratch finite avoids a NaN through a zero probability).

### Tested and not adopted (2026-09-17)

- **sglang#39704** (mHC/metadata overhead for medium batches) applied onto the pinned branch: every column within ±2 % of production on this fleet (its gain is at 32–64 concurrent requests on GB300). Kept out.
- **Newer `dsv4.1` heads (from 2026-09-16 22:15, #39671)** drop the torch candidate indexer and gate the DeepGEMM one on SM100; on SM121 DeepGEMM then rejects the 256-token KV pages (`block_kv == 64`). The pin stays at `f80c91a4b` until upstream has an SM12x candidate path again.
- `CHUNKED_PREFILL_SIZE=8192`, split threshold 16k, NVFP4 experts, `DSV41_CACHE_GIB` above 4, k≠5, NCCL channel/algorithm tuning: measured, no gain or worse.

## Adapters added here

All adapters are import hooks in `adapter/sitecustomize.py`, gated by an environment variable, off unless the variable is set, and each refuses to boot if the engine symbol it patches has drifted.

| File | Gate | What |
|---|---|---|
| `adapter/shared_pad_k.py` | `DSV41_SHARED_PAD_K` | Shared-expert `down_proj` K padded 576 → 640 with re-blocked scales, so the shape stays on the b12x MXFP8 kernel instead of the CUTLASS fallback |
| `adapter/indexer_chunked.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 adapted to the `dev-dsv41` image backend (`self.candidate_masks`) |
| `adapter/indexer_chunked_v2.py` | `DSV41_INDEXER_CHUNKED` | sglang#39187 verbatim, for the `candidate_metadata` backend of the `dsv4.1` branch (kept for reference) |
| `adapter/indexer_chunked_v3.py` + `spark_prefill_dense.py` | `DSV41_INDEXER_CHUNKED`, `SPARK_PREFILL_TP_SPLIT` | v2 plus the SG18 prefill TP split inside the chunked path; `sitecustomize` picks v1 or v3 from the stock source |
| `adapter/fast_load.py` | `DSV41_FAST_LOAD` | Checkpoint tensors this rank will copy (owned experts, no Engram tables) read eagerly by a 16-thread `pread` pool into pinned host memory and returned from `safe_open`; the model's async copies paced to a byte budget so the reads stay just ahead; the DSpark draft load opens only the `mtp.*` shards. Loader-only: the model still does every narrow and copy |

Tests: `tests/test_indexer_chunked.py` and `tests/test_indexer_chunked_v2.py` lift the stock function out of the engine's source, drive it and the backport with deterministic fake kernels, and require bitwise-equal `page_indices`, `raw_indices` and candidate masks over six scenarios (ragged batches, empty requests, full and tail-only mask publishing, mask consumption, one row per chunk up to a single chunk). Both run inside the image build (`Dockerfile` runs v1, `Dockerfile.canary` runs v2), together with upstream's thinking-alias, output-cap and loop-abort tests. `tests/test_fast_load_pacing.py` (all images) checks the copy pacing; the in-image checkpoint test for the eager reads (bitwise equality against the stock `safe_open` for every dtype, draft shard filtering, memory release) needs the weights mounted and is described in `docs/fast-load.md`.

## Measurement notes

- Bench with sparkDash's decode and prefill benches, never with a hand-rolled loop; the first two points after a boot read low (cold Engram row cache).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic landed in the window; the tables above were taken with none.
- Greedy text equality is not a usable correctness gate on this stack: identical cold prompts of ~70k tokens produce different greedy continuations run to run. Correctness of the indexer backport rests on the bitwise CPU tests, upstream's in-forward `page_indices` comparison, the needle tests and the benches.
- `scripts/window-20260916.sh` is the runbook that produced the tables (preflight, build, two boots, benches, rollback).
- The production rows were re-measured on 2026-09-18 on the fast-load boot (`docs/results/fastload-20260918/prodbench-fastload-20260918.txt`): two warm-up prose c1 runs discarded, then one run per cell; prose c1 is the median of three runs (57.5, 59.2, 52.4), and the 4k prefill cell is a single cold point that read 2.4–3.6k across the night's boots.

## Rollback

Upstream's profile is one env change away: `MAX_RUNNING_REQUESTS=8`, `CHUNKED_PREFILL_SIZE=1024`, `EXTRA_CONTAINER_ENV=""` (and `EP_SIZE=4`, `DSV41_CACHE_GIB=0` if you want the exact upstream example). The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The canary image's 4k-token prefill is ~6 % slower than the base image; everything from 16k up is faster.
- Sampled decode on the canary image is ~5 % behind greedy (see above).
- Single-stream prose speed is bounded by DSpark acceptance (~2–3 accepted tokens per step on prose against ~6 on code); no configuration changes that.

## Open items

- **KV pool with the fast loader.** With `DSV41_FAST_LOAD=1` the pool comes out 3–13 % smaller than with the stock loader and varies more between boots (6.71–7.27 M vs 7.47–7.82 M tokens on the same image). Pinned buffers removed most of the gap; the last ~0.8 GB of `MemAvailable` shows up only as `Mapped` file pages of the scheduler process, with the CUDA allocator, anonymous memory, slab and page tables identical. Not chased further; the snapshots to start from are in `docs/results/fastload-20260918/` (`control-boot-observe*.txt`, `fastload-verify-v13-pinned.txt`). `DSV41_FAST_LOAD=0` restores the full pool at the cost of ~220 s per boot.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).
