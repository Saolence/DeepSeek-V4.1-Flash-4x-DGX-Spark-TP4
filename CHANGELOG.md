# Changelog

Newest first. Each entry says what changed in the production stack and what was measured; the
raw results live under `docs/results/`.

## 2026-09-18

- **Engram prefetch on a side stream (`adapter/engram_prefetch.py`, `DSV41_ENGRAM_PREFETCH=1`, on).**
  The NVMe misses of the Engram row lookup no longer stall the graph before each gather.
  Same-image A/B with identical prompts: step 51.7 → 49.5 ms, GPU idle 3.0 → 0.85 ms/step,
  English essay c1 42.6 → 45.1 tok/s, sparkDash prose c1 57–59 → 60–61, prose c4 120 → 124,
  structured 116 → 121; rows bit-identical in check mode.
- `--sleep-on-idle` on (RustyAiLab's suggestion in Mia #16): idle head scheduler CPU 47 % → 14 %,
  workers 5 %; first response after idle and decode unchanged.
- `docs/upstream-watch.md`: re-checked every pinned piece (SGLang dsv4.1 head, #39187/#39704,
  kernels, image, b12x, checkpoint); the SM121 candidate-indexer blocker still holds, pins stay.
- **Faster boot: `adapter/fast_load.py`, gate `DSV41_FAST_LOAD=1`, on in production.** This
  rank's checkpoint tensors are read by a 16-thread `pread` pool into pinned host memory instead
  of being page-faulted through the loader's mmap at 0.5 GB/s; the model's async copies are paced
  to a byte budget; the DSpark draft load opens the 3 shards that hold `mtp.*` instead of 48.
  Engine start 343–354 s → 111–129 s, values bitwise identical, decode/prefill/needle unchanged.
  Trade-off: the KV pool is 3–13 % smaller (6.71–7.27 M vs 7.47–7.82 M tokens) and varies more
  between boots. Profile, dead ends and memory accounting: `docs/fast-load.md`.
- Production decode/prefill rows in the README re-measured on the fast-load boot.
- Merged from Saolence: #5 (worker containers mount the same NCCL as the head, worker preflight
  actually runs), #6 (anchored rsync excludes in `build`, `BUILD_DOCKERFILE`, `BUILD_ARGS`,
  `PIP_INDEX` mirror knob), #2 (second-plane addressing for the ring, docs). `PIP_INDEX` also
  added to `Dockerfile.canary-roce`. Noted that torch loads the pip NCCL through its RPATH, so
  only `NCCL_OVERLAY_PIP=1` replaces it.
- New: `scripts/loadprof.sh` (py-spy + diskstats sampler for the load phase),
  `tests/test_fast_load_pacing.py` (in every image build), `tests/test_fast_load_checkpoint.py`.

## 2026-09-17

- **Production image = `Dockerfile.canary-roce`**: upstream `dsv4.1` branch at `f80c91a4b` +
  RoCEnante overlay (rhys101's SG17, adapted to TP4, both rails, 2 MiB route) + all adapters.
  Prose c1 57.0 / c16 290.7, code c1 107.0 / c16 860.8, structured 115.4.
- **Prefill TP split** (rhys101's SG18) combined with the sglang#39187 backport in
  `adapter/indexer_chunked_v3.py`: sparkDash 128k 4364, 262k 3893; 985k needle in 585 s with
  5 GiB head low-water.
- Quality gate: 75-task paired evaluation, production 72/75 vs base 71/75 (p = 1.000);
  `scripts/qeval.py`.
- `--enable-cache-report` on; `SGLANG_RUST_BUILD_MODE=never` (the branch's cargo probe can hang
  the head before the HTTP server starts).
- Tested and not adopted: sglang#39704 on the pinned branch (±2 %), newer `dsv4.1` heads on
  SM12x (candidate indexer needs the DeepGEMM paged path), chunk 8192, split threshold 16k.
- Merged from Saolence: #1, switchless four-Spark ring as an opt-in (`NCCL_SWITCHLESS_RING_ONLY`,
  `NFS_SHARE=0`), 44 offline tests; the ring decode table is a same-stack reference, not a
  speedup.
- Repository published: fork of MiaAI-Lab's recipe with history, tuned TP4 profile (EP2, Engram
  row cache, 16 slots, chunk 4096 + sglang#39187 backport, shared-expert K pad, optional canary
  image); verified from a fresh clone including the 985k-token needle.

## 2026-09-16

- Maintenance window (`scripts/window-20260916.sh`, `docs/window-20260916.md`):
  `MAX_RUNNING_REQUESTS` 8 → 16 (adds the c16 tier), sglang#39187 backport with
  `CHUNKED_PREFILL_SIZE` 1024 → 4096 and the adaptive chunk sizer off, upstream `dsv4.1`
  canary image with folded sampling forced (`SGLANG_DSPARK_FOLDED_SAMPLING=2`).

## Before the fork

- MiaAI-Lab's `DeepSeek-v4.1-Flash-DGX-Sparks` up to 2026-09-15 (#18: thinking alias, output
  cap, loop abort, DSpark k=3). See `docs/README-upstream.md` and the Credits section.
