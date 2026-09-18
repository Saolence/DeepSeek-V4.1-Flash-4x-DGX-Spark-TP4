"""Faster checkpoint loading without changing the bytes that reach the model.

Measured on the 4x Spark TP4/EP2 fleet (2026-09-17): the target load_weight took 230-290 s while
every rank copied only ~130 GB of the 510 GB checkpoint. The stock loader hands the model
mmap-backed safetensors tensors, and the 24-thread copy pool page-faults them in 4-128 KB pieces
(0.5 GB/s on the ranks with moe_tp_rank 0, ~2 GB/s on the others) while the same NVMe delivers
5-6 GB/s to threaded ``pread``. Warming the page cache ahead of the copies does not survive on
GB10: GPU memory is system memory, so the cache is squeezed while the weights land and the
warmed pages are evicted before the copy touches them. ``posix_fadvise(WILLNEED)`` is capped at
the 128 KB readahead window anyway.

What this does, gated by ``DSV41_FAST_LOAD=1``:

1. When the loader opens a shard, the tensors *this rank* will copy are read eagerly into
   pinned host memory (torch's caching host allocator; anonymous mmaps without CUDA) by a
   16-thread ``pread`` pool (every tensor except the routed experts
   another EP rank owns and the Engram tables, which the row store serves straight from NVMe).
   ``get_tensor`` for those names returns the resident tensor; everything else stays the stock
   mmap tensor. Same file offsets, same dtype, same shape: the values are identical by
   construction and a wrong ownership guess only costs I/O.
2. The model's ``load_weights`` submits every copy to a thread pool and never waits, so the
   shard enumeration would run far ahead and keep whole shards resident. ``maybe_executor_submit``
   is wrapped with a byte budget (``DSV41_FAST_LOAD_INFLIGHT_GB``, default 6): the enumeration,
   and with it the loader's window and the eager reads, stays just ahead of the copies. Host
   memory in flight is bounded by that budget plus the loader window (``--model-loader-extra-config
   {"num_threads":1}`` = 2 shards, ~3.4 GB each on TP4/EP2).
3. The DSpark draft is loaded from the same 48 shards but consumes only ``mtp.*`` tensors (see
   ``_remap_dspark_weight_name``). During that load, shards without any ``mtp.*`` key are handed
   back empty so the loader does not open and enumerate 45 files for nothing.

Everything is released before the KV pool is sized, so the head's memory budget is unchanged.
"""
import concurrent.futures
import json
import logging
import os
import struct
import threading

logger = logging.getLogger(__name__)

_DRAFT_PREFIX = "mtp."
_TARGET_SKIP_PREFIXES = (_DRAFT_PREFIX,)
_ENGRAM_TABLE = ".engram.embed."
_CHUNK = 8 << 20
_state = {"phase": "target", "armed": False, "warned": False, "ep_warned": False, "layout_logged": False,
          "bytes": 0, "files": 0, "skipped": 0}
_lock = threading.Lock()
_header_cache: dict = {}
_pool = None

_DTYPES = {
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu",
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64", "BOOL": "bool",
}


def enabled() -> bool:
    return os.environ.get("DSV41_FAST_LOAD", "0").strip() in ("1", "on", "true")


def observing() -> bool:
    """``DSV41_FAST_LOAD=observe``: stock loader, but the same memory snapshot after each load."""
    return enabled() or os.environ.get("DSV41_FAST_LOAD", "0").strip() == "observe"


def _parse_header(path):
    with _lock:
        hit = _header_cache.get(path)
    if hit is not None:
        return hit
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    entry = (8 + n, header)
    with _lock:
        _header_cache[path] = entry
    return entry


def _ep_info():
    """(ep_rank, ep_size) of this process, or None when unknown."""
    try:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        return int(par.moe_ep_rank), int(par.moe_ep_size)
    except Exception as exc:
        try:
            from sglang.srt.distributed import parallel_state as ps

            return ps.get_moe_expert_parallel_rank(), ps.get_moe_expert_parallel_world_size()
        except Exception as exc2:
            # Last resort: SGLang assigns moe_ep_rank = tp_rank // (tp_size // ep_size).
            try:
                ep_size = int(os.environ["DSV41_FAST_LOAD_EP_SIZE"])
                from sglang.srt.distributed import parallel_state as ps

                tp_rank, tp_size = ps.get_tensor_model_parallel_rank(), ps.get_tensor_model_parallel_world_size()
                return tp_rank // (tp_size // ep_size), ep_size
            except Exception as exc3:
                if not _state["ep_warned"]:
                    _state["ep_warned"] = True
                    logger.warning("DSV41 fast load: EP layout unknown (%r / %r / %r); eager reads for non-expert tensors only",
                                   exc, exc2, exc3)
                return None


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _n_routed_experts(path):
    """Routed expert count from config.json next to the shards (nested under text_config for V4.1)."""
    cfg = os.path.join(os.path.dirname(path), "config.json")
    try:
        with open(cfg) as f:
            hit = _find_key(json.load(f), "n_routed_experts")
        if hit is not None:
            return int(hit)
    except Exception:
        pass
    env = os.environ.get("DSV41_FAST_LOAD_N_EXPERTS")
    return int(env) if env else None


def _expert_id(name):
    if ".experts." not in name:
        return None
    try:
        return int(name.split(".experts.", 1)[1].split(".", 1)[0])
    except ValueError:
        return None


def needed_names(path, phase, ep=None, n_routed=None):
    """Tensor names this rank will copy out of ``path`` in ``phase``."""
    _, header = _parse_header(path)
    if phase == "draft":
        return [k for k in header if k.startswith(_DRAFT_PREFIX)]
    if ep is None or n_routed is None or n_routed % ep[1] != 0:
        return [k for k in header if not k.startswith(_TARGET_SKIP_PREFIXES) and _ENGRAM_TABLE not in k
                and _expert_id(k) is None]
    per_rank = n_routed // ep[1]
    lo, hi = ep[0] * per_rank, (ep[0] + 1) * per_rank
    out = []
    for k in header:
        if k.startswith(_TARGET_SKIP_PREFIXES) or _ENGRAM_TABLE in k:
            continue
        eid = _expert_id(k)
        if eid is None or lo <= eid < hi:
            out.append(k)
    return out


def needed_ranges(path, phase, ep=None, n_routed=None):
    """Merged absolute byte ranges of ``needed_names`` (for tests and accounting)."""
    base, header = _parse_header(path)
    ranges = sorted((base + a, base + b) for k in needed_names(path, phase, ep, n_routed)
                    for a, b in [header[k]["data_offsets"]] if b > a)
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _executor():
    global _pool
    if _pool is None:
        _pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("DSV41_FAST_LOAD_THREADS", "16")), thread_name_prefix="dsv41-read")
    return _pool


def _read_into(fd, mv, off):
    got = 0
    n = len(mv)
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at {off + got}")
        got += r


def _read_tensor(fd, base, info):
    """Read one safetensors entry into a fresh contiguous torch tensor.

    The buffer is an anonymous private mmap (like safetensors' own tensors), not a malloc
    allocation: glibc keeps freed multi-MB chunks in its arenas once the dynamic mmap threshold
    has grown, and a first version of this loader left ~11 GB resident on the head after the
    load, which cost 5x of the KV pool (the pool is sized from the head's free memory).
    Unmapped as soon as the tensor is released.
    """
    import mmap

    import torch

    dtype = getattr(torch, _DTYPES[info["dtype"]])
    a, b = info["data_offsets"]
    n = b - a
    if n == 0:
        return torch.empty(info["shape"], dtype=dtype)
    pinned = _pinned_ok()
    if pinned:
        # Pinned source: the H2D copy is a plain DMA. Pageable sources cost more than the copy:
        # the driver keeps a staging pool behind after bursts of concurrent pageable copies
        # (measured ~0.4 GB per 3 GB burst on a worker, ~1.5 GB after a full load on the head),
        # and that memory is gone from MemAvailable when the KV pool is sized. Blocks come from
        # torch's caching host allocator and are returned to the driver by _release_all.
        flat = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        mv = memoryview(flat.numpy())
    else:
        mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mv = memoryview(mm)
    # Sequential in this thread (a nested pool submit would starve the pool); the pool's
    # parallelism comes from the ~1200 tensors of a shard being read concurrently.
    for c in range(0, n, _CHUNK):
        _read_into(fd, mv[c:c + _CHUNK], base + a + c)
    mv.release()
    if not pinned:
        flat = torch.frombuffer(mm, dtype=torch.uint8)
    return flat.view(dtype).reshape(info["shape"])


def _pinned_ok():
    hit = _state.get("pinned")
    if hit is None:
        try:
            import torch

            hit = bool(torch.cuda.is_available()) and os.environ.get("DSV41_FAST_LOAD_PINNED", "1") == "1"
            if hit:
                torch.empty(1, dtype=torch.uint8, pin_memory=True)
        except Exception:
            hit = False
        _state["pinned"] = hit
        logger.warning("DSV41 fast load: eager buffers are %s", "pinned host memory" if hit else "anonymous mmaps")
    return hit


class _EmptyShard:
    """What the loader sees for a shard the current load cannot use."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return []

    def offset_keys(self):
        return []

    def metadata(self):
        return {}

    def get_tensor(self, name):
        raise KeyError(name)


class _EagerShard:
    """Stock safe_open handle whose needed tensors are read eagerly by the thread pool."""

    def __init__(self, inner, path, names):
        self._inner = inner
        self._path = path
        base, header = _parse_header(path)
        self._fd = os.open(path, os.O_RDONLY)
        ex = _executor()
        self._futures = {k: ex.submit(_read_tensor, self._fd, base, header[k]) for k in names}
        total = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in names)
        with _lock:
            _state["bytes"] += total
            _state["files"] += 1

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            for f in self._futures.values():
                f.cancel()
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            os.close(self._fd)
        finally:
            return self._inner.__exit__(*exc)

    def keys(self):
        return self._inner.keys()

    def offset_keys(self):
        return self._inner.offset_keys()

    def metadata(self):
        return self._inner.metadata()

    def get_slice(self, name):
        return self._inner.get_slice(name)

    def get_tensor(self, name):
        fut = self._futures.pop(name, None)
        if fut is None:
            return self._inner.get_tensor(name)
        try:
            return fut.result()
        except Exception as exc:
            if not _state["warned"]:
                _state["warned"] = True
                logger.warning("DSV41 fast load: eager read of %s failed (%r); falling back to mmap", name, exc)
            return self._inner.get_tensor(name)


def _wrap_open(orig, filename, args, kwargs):
    phase = _state["phase"]
    if not (isinstance(filename, (str, os.PathLike)) and str(filename).endswith(".safetensors")):
        return orig(filename, *args, **kwargs)
    path = os.fspath(filename)
    ep = _ep_info() if phase == "target" else None
    n_routed = _n_routed_experts(path) if ep else None
    if phase == "target" and not _state["layout_logged"]:
        _state["layout_logged"] = True
        logger.warning("DSV41 fast load: target layout ep=%s n_routed_experts=%s first shard %s", ep, n_routed, path)
    names = needed_names(path, phase, ep, n_routed)
    if phase == "draft" and not names:
        with _lock:
            _state["skipped"] += 1
        return _EmptyShard()
    inner = orig(filename, *args, **kwargs)
    if not names or kwargs.get("device", "cpu") not in ("cpu", None) or (len(args) > 1 and args[1] != "cpu"):
        return inner
    return _EagerShard(inner, path, names)


def install_weight_utils(module):
    """Wrap ``safetensors.safe_open`` as used by ``weight_utils``."""
    if not enabled() or _state["armed"]:
        return
    import safetensors

    orig = safetensors.safe_open

    def safe_open(filename, *args, **kwargs):
        try:
            return _wrap_open(orig, filename, args, kwargs)
        except Exception as exc:  # never let the fast path break the load
            if not _state["warned"]:
                _state["warned"] = True
                logger.warning("DSV41 fast load: disabled after error on %s: %r", filename, exc)
            return orig(filename, *args, **kwargs)

    safetensors.safe_open = safe_open
    if getattr(module, "safetensors", None) is not None:
        module.safetensors.safe_open = safe_open
    _state["armed"] = True
    logger.warning("DSV41 fast load ARMED: eager threaded reads of this rank's tensors; draft load opens only mtp shards")


def _malloc_trim():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _release_all():
    """After a load: drop the reader pool, header cache and malloc arenas; report what is still mapped."""
    global _pool
    import gc
    import mmap

    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
    with _lock:
        paths = list(_header_cache)
        _header_cache.clear()
    gc.collect()
    live = [o for o in gc.get_objects() if isinstance(o, mmap.mmap) and not o.closed]
    _malloc_trim()
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass
    # The KV pool is sized from what CUDA reports free, and on GB10 that is the kernel's
    # MemFree: page cache counts as used. Drop every shard's cached pages now (the stock
    # handles of the last window are closed by this point).
    for d in {os.path.dirname(p) for p in paths}:
        try:
            for name in os.listdir(d):
                if name.endswith(".safetensors"):
                    fd = os.open(os.path.join(d, name), os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
        except OSError:
            pass
    return len(live), sum(len(o) for o in live), _rss_mb()


def _meminfo_gb(key):
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(key):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return -1.0


def _snapshot():
    out = {k: round(_meminfo_gb(k), 2) for k in ("MemFree", "MemAvailable", "Cached", "Shmem", "AnonPages", "Mapped",
                                                   "Unevictable", "Mlocked", "Slab", "SUnreclaim", "KReclaimable",
                                                   "PageTables", "VmallocUsed", "Percpu", "HugePages_Total")}
    try:
        for line in open("/proc/self/smaps_rollup"):
            if line.startswith(("Rss", "Anonymous", "Shared_Clean", "Private_Clean")):
                out["self_" + line.split(":")[0]] = round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        pass
    try:
        import torch

        out["cuda_free"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
        out["cuda_reserved"] = round(torch.cuda.memory_reserved() / 2**30, 2)
        out["cuda_allocated"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        st = torch.cuda.memory_stats()
        out["cuda_inactive_split"] = round(st.get("inactive_split_bytes.all.current", 0) / 2**30, 2)
    except Exception:
        pass
    try:  # biggest mappings of this process (file or anonymous), MB
        big = []
        cur = None
        for line in open("/proc/self/smaps"):
            if line[0] in "0123456789abcdef" and "-" in line.split()[0]:
                parts = line.split()
                cur = parts[5] if len(parts) > 5 else "[anon]"
            elif line.startswith("Rss:"):
                kb = int(line.split()[1])
                if kb >= 64 * 1024:
                    big.append((kb // 1024, cur))
        big.sort(reverse=True)
        out["big_maps_mb"] = big[:8]
    except Exception:
        pass
    return out


def _log_phase(phase):
    with _lock:
        b, f, s = _state["bytes"], _state["files"], _state["skipped"]
        _state["bytes"] = _state["files"] = _state["skipped"] = 0
    before = _snapshot()
    if enabled():
        live, live_bytes, rss = _release_all()
        logger.warning("DSV41 fast load: phase=%s read %.1f GB eagerly over %d shards, %d shards skipped; "
                       "after release: %d anonymous maps alive (%.2f GB), RSS %d MB",
                       phase, b / 1e9, f, s, live, live_bytes / 1e9, rss)
    logger.warning("DSV41 fast load: phase=%s memory before release %s | after %s", phase, before, _snapshot())


def _tensor_bytes(func_args):
    for a in func_args:
        try:
            import torch

            if isinstance(a, torch.Tensor) and not isinstance(a, torch.nn.Parameter):
                return a.numel() * a.element_size()
        except Exception:
            pass
    return 0


def install_deepseek_v4(module):
    """Pace the model's async weight copies so the eager reads stay just ahead of consumption."""
    if not enabled():
        return
    orig = module.maybe_executor_submit
    budget = int(float(os.environ.get("DSV41_FAST_LOAD_INFLIGHT_GB", "6")) * 2**30)
    cv = threading.Condition()
    inflight = [0]

    def paced_submit(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
        if not use_async:
            return orig(executor=executor, futures=futures, use_async=use_async, func=func,
                        func_args=func_args, func_kwargs=func_kwargs)
        size = max(1, _tensor_bytes(func_args))
        with cv:
            while inflight[0] > 0 and inflight[0] + size > budget:
                cv.wait(timeout=1.0)
            inflight[0] += size
        before = len(futures)

        def release(_f=None):
            with cv:
                inflight[0] -= size
                cv.notify_all()

        try:
            orig(executor=executor, futures=futures, use_async=use_async, func=func,
                 func_args=func_args, func_kwargs=func_kwargs)
        except BaseException:
            release()
            raise
        if len(futures) > before:
            futures[-1].add_done_callback(release)
        else:
            release()

    module.maybe_executor_submit = paced_submit
    logger.warning("DSV41 fast load: weight copies paced to %.1f GB in flight", budget / 2**30)
    _wrap_target_load(module)


def _wrap_target_load(module):
    """Release everything the moment the target's copies are done, before the engine measures
    its memory: SGLang derives the KV pool from what is free after the weights landed."""
    cls = getattr(module, "DeepseekV4ForCausalLM", None)
    if cls is None or getattr(cls.load_weights, "_dsv41_fast_load", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        try:
            return orig(self, weights, *args, **kwargs)
        finally:
            if _state["phase"] == "target":
                _log_phase("target")

    load_weights._dsv41_fast_load = True
    cls.load_weights = load_weights


def _ps_top(n=6):
    try:
        rows = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                rss = 0
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS"):
                        rss = int(line.split()[1]) // 1024
                        break
                comm = open(f"/proc/{pid}/comm").read().strip()
                rows.append((rss, pid, comm))
            except Exception:
                pass
        rows.sort(reverse=True)
        return rows[:n]
    except Exception:
        return []


def _schedule_late_snapshots():
    """Diagnostics: how the host memory settles after the loads (other processes still starting)."""
    import time

    def run():
        for delay in (30, 60, 120, 240):
            time.sleep(delay if delay == 30 else delay - prev[0])
            prev[0] = delay
            snap = _snapshot()
            snap.pop("big_maps_mb", None)
            logger.warning("DSV41 fast load: +%ds after draft load: %s | top rss MB %s", delay, snap, _ps_top())

    prev = [0]
    threading.Thread(target=run, name="dsv41-late-snap", daemon=True).start()


def install_dspark(module):
    """Mark the DSpark draft load so shard filtering and mtp-only eager reads apply."""
    if not observing():
        return
    cls = module.DeepseekV4ForCausalLMDSpark
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        _state["phase"] = "draft"
        try:
            return orig(self, weights, *args, **kwargs)
        finally:
            _log_phase("draft")
            _state["phase"] = "target"
            if os.environ.get("DSV41_FAST_LOAD_DEBUG") == "1":
                _schedule_late_snapshots()

    cls.load_weights = load_weights
    if not enabled():
        # observe mode: the target hook is otherwise installed by install_deepseek_v4
        import sys

        mod = sys.modules.get("sglang.srt.models.deepseek_v4")
        if mod is not None:
            _wrap_target_load(mod)


if __name__ == "__main__":  # self-test: python3 fast_load.py <shard> <ep_rank> <ep_size>
    import sys, time

    path, ep = sys.argv[1], (int(sys.argv[2]), int(sys.argv[3]))
    names = needed_names(path, "target", ep, _n_routed_experts(path) or 384)
    base, header = _parse_header(path)
    total = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in names)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    t = time.time()
    futs = [_executor().submit(_read_tensor, fd, base, header[k]) for k in names]
    tensors = [f.result() for f in futs]
    dt = time.time() - t
    print(f"{os.path.basename(path)}: read {total/1e9:.2f} GB in {len(names)} tensors in {dt:.1f}s = {total/1e9/dt:.2f} GB/s")
