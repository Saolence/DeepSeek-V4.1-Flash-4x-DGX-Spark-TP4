# Changelog

Newest first. Each entry says what changed in the production stack and what was measured; the
raw results live under `docs/results/`.

## 2026-09-23 (evening)

- **`adapter/verify_cap.py`, `DSV41_VERIFY_CAP=conf:0.1`, on.** Adaptive verify length without changing
  the verify layout: dead rows reuse the anchor row's experts (in-graph Triton remap), acceptance is
  capped through the engine's cutoff, the draft confidence head (force-built in static mode) picks the
  length as a stopping rule. sparkDash prose c1 66.0 -> 69.7, prose c4 125.4 -> 133.5, sampled
  thinking traffic +5 %, code/structured flat; greedy outputs identical; toy-LM chi-square test in the
  image build.
- **`adapter/autotune_keep.py`, `DSV41_AUTOTUNE_KEEP=1`, on.** The FlashInfer MoE autotune cache is kept
  across boots under EP (sglang#40320); 0 re-tunes from the second boot, was 26.
- `adapter/block_verify.py`: `live=` for shortened blocks.
- Fixed verify caps (1..4) measured and rejected: prose -8..-15 %, code up to -40 %.
- **`DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, on.** wo_a at 9-192 verify/draft rows from the fp8
  twin (verify step c4 82.1 -> 78.3 ms, c8 120.0 -> 118.1 ms), and the bf16 copy released (722 MB per
  rank) with bit-identical prefill through per-call dequantization.
- `verify_cap`: live lengths broadcast from rank 0 each step (the confidence head reads all-reduced
  activations, which may differ in the last bits between ranks); no measurable cost.

## 2026-09-23

- **`adapter/wo_a_w8.py`, `DSV41_WO_A_W8=1`, on.** The verify/draft `wo_a` (2–8 rows) reads the
  checkpoint's fp8 bytes instead of the bf16 copy made at load: exact fp8 twins for 43 of 43
  layers, same Triton tiling with the bf16 tile rebuilt in registers. 43 layers at M=6
  3.43 → 2.20 ms, decode step at c1 52.9 → 51.7 ms (bench of 12 × 800-token answers, warm).
  MXFP8-epilogue path bitwise identical; plain bf16 path differs in fp32 accumulation order only.
  Quality gate 72/75 twice against 73/75: primary code+reason+math 53/55 both, json 15/15; the one
  flip is `prose_p2` coming in at 88 words against a 100-word minimum.
- **`adapter/draft_tau.py`, `DSV41_DRAFT_TAU=0.8`, on.** Draft proposal temperature for sampled
  requests; exact by construction. +1.2 % accepted tokens per step at T=1 / top_p=0.95 offline
  (3.242 → 3.280 on held-out traffic); 0.7–0.8 is the optimum.
- **`adapter/draft_head_fp8.py`, `DSV41_DRAFT_HEAD_FP8=1`, on.** The draft's block logits come from an
  fp8 copy of the shared LM head (target head untouched): 1405 → 721 us, live step 51.7 → 50.9 ms.
- **`adapter/block_verify.py`, `DSV41_BLOCK_VERIFY=1`, on.** Block verification (Sun et al., ICLR 2025)
  for sampled rows. Lossless (chi-square test on a toy LM in the image build). Offline on
  engine-exact p and q: +1.8 % tokens/step on prose (2.705 -> 2.754), +2.5 % on coding with thinking
  (2.927 -> 2.999); live, the engine accepted 1.766 drafts/step on steps where token verification
  expects 1.705.
- **`adapter/folded_result_fence.py`, `DSV41_FOLDED_FENCE=1`, on.** Folded (all-greedy) verify results are
  cloned before the overlapped D2H copy (sglang#40919); sampled batches were never exposed.
- `adapter/draft_capture.py` schema 4: also the committed tokens, the drafted tokens and the draft's
  top-64 logits with its full-vocab normaliser, so verification rules can be evaluated offline.
- sparkDash after all of the above: prose c1 65.3 (was 61.0), code c1 113.8, structured 124.7.
- `sitecustomize`: `dspark_verify` was missing from the hooked modules, so `DSV41_DRAFT_CAPTURE`
  never installed; added together with `dspark_draft_sampler`.
- Tested and not adopted: draft fine-tune on own traffic, multi-pass drafting, expert-sharing
  routing, n-gram lookup, relaxed acceptance, split-K MXFP8 and hyper-connection kernel variants
  (numbers in the README).

## 2026-09-18

- Tested and not adopted: Markov W2 unsharded (+0.6 ms/step), fast-load knobs 12 GB / 2 threads
  (−2 s to ready), parallel Engram misses in the row store (no change).
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
- Production decode/prefill rows in the README re-measured on the final boot (fast load + Engram
  prefetch): prose c1 61.0 / c16 292.9, code c1 113.3 / c16 882.4, structured 124.1.
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
