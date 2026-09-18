# Fast weight loading (2026-09-18)

`adapter/fast_load.py`, gate `DSV41_FAST_LOAD=1`. Engine start on the production stack went
from 356 s to 125 s; the values that reach the model are bitwise the same. Status on
2026-09-18: the gate ships **off**. The version that ran on the fleet (anonymous mmap buffers)
leaves the KV pool 10-14 % smaller than the stock loader; the cause is understood and the fix
(pinned buffers) is in the file and unit-tested, but the fleet boot that confirms it did not
happen that night (a side test took a worker down, see the last section).

## Where the 285 s went

Profiled with `py-spy dump` every 10 s and `/proc/diskstats` every 5 s during production boots:

- The "Multi-thread loading shards" bar (~45 s) is not the read. safetensors' `get_tensor`
  returns mmap-backed tensors; the bytes move when the model's 24-thread copy pool runs
  `expert_data.copy_(loaded_weight)` in FusedMoE (`_load_w13` / `_load_w2`), and the model
  never waits for those futures until the enumeration is done.
- Those copies page-fault the mmap in 4-128 KB pieces (`read_ahead_kb` = `max_sectors_kb` =
  128 on the Sparks). Per rank ~130 GB at 0.45-0.6 GB/s on the two ranks with `moe_tp_rank`
  0 and ~1.5-2 GB/s on the other two: 230-290 s versus 95-130 s, every boot, same hardware.
  The same NVMe delivers 5.8 GB/s to 16 `pread` threads over the same byte ranges and
  10.5 GB/s O_DIRECT sequential.
- The checkpoint is 510 GB: routed experts 289 GB (384 experts, so 144 GB per EP rank),
  Engram tables 203 GB (never copied; the row store serves them from NVMe and the loader only
  checks their shape), everything else 10.5 GB, and the DSpark draft (`mtp.*`) 7.9 GB.
- The draft load re-opened all 48 shards to consume 2401 `mtp.*` tensors that live in three
  of them: 44-63 s per rank.

## What was tried and why it is what it is

| step | result |
|---|---|
| `posix_fadvise(WILLNEED)` on this rank's ranges when a shard is opened | no gain: the kernel caps it at the readahead window (128 KB per call) |
| explicit threaded `pread` warming of the page cache at shard open | reads ran at 4-5 GB/s for 40 s, then the copies still faulted at 0.5 GB/s: the enumeration ran 45 s ahead of the copies, and on GB10 the GPU weights are system memory, so the page cache is squeezed while the weights land and warmed pages are evicted before use |
| the above plus pacing the copies (`maybe_executor_submit` wrapped with a budget) | ranks 1/3: 108-130 s -> 62-65 s; ranks 0/2 unchanged: their faults are the slow kind whatever is in the cache |
| eager reads into host memory (`torch.empty` + `preadv`), returned from `get_tensor` | load 66 s on every rank, but ~11 GB stayed resident afterwards (glibc keeps freed multi-MB chunks once its dynamic mmap threshold has grown) and the KV pool, sized from the head's free memory, shrank 5x |
| eager reads into anonymous `mmap` buffers (`torch.frombuffer`), pool shut down and `malloc_trim` after each load | load 73 s, the process itself back to baseline, but the KV pool still 10-14 % smaller than a same-image control boot: SGLang sizes it from `MemAvailable` (integrated GPU), and ~1.5 GB had left `MemAvailable` without showing up in any process or kernel counter. A standalone test reproduced it: after a burst of concurrent host-to-device copies from *pageable* memory the driver keeps a staging pool (~0.4 GB per 3 GB burst); pinned sources leave nothing behind |
| eager reads into pinned host memory (torch's caching host allocator), everything released before the engine measures memory | **current code; standalone copy test shows pinned sources leave nothing behind; the confirming fleet boot is still owed** |
| upstream `--weight-loader-prefetch-checkpoints` + `--weight-loader-disable-mmap` | OOM-killed the head (disable-mmap stages whole 10 GB shards in RAM, nine in flight) |

## How it works

1. `safetensors.safe_open` is wrapped (as used by `weight_utils`). For a shard, the names this
   rank will copy are every tensor except the routed experts another EP rank owns and the Engram
   tables; owned experts are read whole even though MoE-TP copies half of w1/w3 (FusedMoE's
   narrowing has too many branches to mirror safely; the extra read is ~50 GB at NVMe speed).
   Those tensors are read by a 16-thread pool into pinned host buffers (torch's caching host
   allocator, so the copy into the parameter is a plain DMA; anonymous mmaps when CUDA is not
   available); `get_tensor` returns them, everything else stays the stock mmap tensor. The EP rank comes from the runtime
   context, then `parallel_state`, then `DSV41_FAST_LOAD_EP_SIZE`; `n_routed_experts` from
   `config.json` (nested under `text_config` for V4.1) or `DSV41_FAST_LOAD_N_EXPERTS`. If the
   layout is unknown only the non-expert tensors are read eagerly.
2. `maybe_executor_submit` in `deepseek_v4` is wrapped with a byte budget
   (`DSV41_FAST_LOAD_INFLIGHT_GB`, default 6): the enumeration, and with it the loader's window
   and the eager reads, stays just ahead of the copies. Host memory in flight is bounded by that
   budget plus the loader's window (`--model-loader-extra-config {"num_threads":1}` = 2 shards).
3. The DSpark draft's `load_weights` is marked; during it shards without `mtp.*` are handed back
   as empty handles and the three real ones are read eagerly.
4. When the target's `load_weights` returns and again after the draft's, before the engine
   measures its memory: the reader pool is shut down, the header cache dropped, `gc.collect`,
   `malloc_trim`, `torch.cuda.empty_cache`, the pinned blocks handed back to the driver
   (`torch._C._host_emptyCache`) and every shard's page cache dropped (`POSIX_FADV_DONTNEED`).
   The log line reports RSS, `MemAvailable` and the CUDA allocator before and after.

## Measured (production image + fast load, 2026-09-18)

| | before | after |
|---|---:|---:|
| target `load_weight`, rank 0 / 1 / 2 / 3 | 228 / 95 / 246 / 114 s | 74 / 83 / 73 / 71 s |
| draft `load_weight` | 38-50 s | 3-6 s |
| `Engine startup timings: load_weight` | 276-278 s | 77-89 s |
| `scheduler_e2e` (start to ready) | 346-354 s | 125-129 s |
| `max_total_num_tokens` (same image, gate off vs on, same night) | 7.47-7.82M | 6.2-6.8M (mmap-buffer version; pinned version not yet booted) |
| decode prose c1 / c4 agg, code c1, structured c1 | 57.0 / 118.6 / 107 / 115 | 56.8 / 120.8 / 92-107 / 99-111 |
| prefill 4k / 32k / 128k | 4070 / 4554 / 4364 | 2747-3227 / 4209-4232 / 4474-4494 |
| needle 131k | PASS | PASS |

Decode and prefill numbers are single sparkDash runs right after each boot and sit inside the
run-to-run spread of the production table (the first point after a boot reads low). Raw outputs:
`docs/results/fastload-20260918/`.

## Correctness

- `tests/test_fast_load_pacing.py` (runs in every image build): the paced submit never exceeds
  the budget, completes every copy, leaves the synchronous path untouched and leaks no permit on
  an exception.
- In-image checkpoint test (needs the weights mounted, not part of the build): driving the real
  `buffered_multi_thread_safetensors_weights_iterator` in draft phase over all 48 shards yields
  exactly the 2401 `mtp.*` tensors, 45 shards skipped; every eager tensor of a target shard and
  of the draft is bitwise equal (`view(uint8)`) to the stock `safe_open` tensor across all five
  dtypes present (`float8_e4m3fn`, `float8_e8m0fnu`, `int8`, `bfloat16`, `float32`); RSS returns to
  baseline after a shard's tensors are released.
- The model's own loader still does every narrow and copy; the adapter only changes where the
  source bytes are resident.

## Why the KV pool shrank, and the incident

SGLang sizes the KV pool from `psutil.virtual_memory().available` on integrated GPUs
(`get_available_gpu_memory`, "these devices use sysmem as device mem"). With the mmap-buffer
loader, `MemAvailable` at that moment was 1.4-1.7 GB lower than with the stock loader on the same
image, while the process (RSS, anonymous maps), the CUDA caching allocator (`memory_reserved`
identical to the byte) and the kernel counters (slab, page tables, unevictable) all matched. What
did not show anywhere is driver memory: a standalone test on a worker copying 3 GB from pageable
host memory with 24 threads leaves ~0.4 GB missing from `MemAvailable` after the buffers are
freed, a second burst adds nothing, and copies from pinned memory leave nothing. So the eager
buffers are now pinned (torch's caching host allocator, returned with `_host_emptyCache` before
the engine measures). The raw snapshots are in `docs/results/fastload-20260918/`
(`control-boot-observe*.txt`, `control-late.txt` for the stock loader; `fastload-verify-v*.txt`
for the fast-load boots).

The checkpoint test that validates bitwise equality held the whole draft (7.9 GB) in memory; run
with pinned buffers on a *serving* worker it pushed the node into a state where SSH stopped
answering and the fleet went down. It now compares tensor by tensor and holds nothing, and it
must only be run with the fleet stopped. That is the reason the pinned-buffer fleet boot is still
owed.
