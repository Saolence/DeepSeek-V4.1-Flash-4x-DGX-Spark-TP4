# Switchless ring: four Sparks, no RoCE switch

`NCCL_SWITCHLESS_RING_ONLY=1` boots this stack on four DGX Sparks cabled as a ring
(a-b-c-d-a, one DAC per adjacency per rail, no switch in between). It is **off by
default and changes nothing when off**: the switch only decides whether the ring
NCCL environment and the overlay mount are injected.

## Why the defaults do not boot on a ring

```
NCCL INFO Trees [0] 2/-1/-1->0->-1
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
...
RuntimeError: NCCL error: unhandled system error
    at ncclTransportTreeConnect -> ncclCommInitRank
```

NCCL builds a **tree** in addition to the ring. In a four-node ring the tree wants a
direct fabric path between opposite nodes (rank0 ↔ rank2), and there is none: the
diagonal exists at the IP layer through a transit node, but RoCE queue-pair setup
does not follow IP routing. The tree therefore never connects and the boot dies
inside `ncclCommInitRank`. The NCCL counters and `/health` never come up.

The fix is two environment variables plus a patched NCCL:

| | |
|---|---|
| `NCCL_ALGO=Ring` | no tree, ring only |
| `NCCL_SKIP_TREE_CONNECT=1` | belt and braces: the patched build refuses the tree connect outright |
| patched NCCL | `sparkring`'s `switchless-cycle` / `skip-tree-pat` patches (FujitsuPolycom/sparkring) |

## Cabling

Four DACs, no switch and no diagonal links:

```text
  rank0                         rank1
  CX7-0 -------- cable 1 ------- CX7-1
  CX7-1                         CX7-0
    |                             |
  cable 4                       cable 2
    |                             |
  CX7-0                         CX7-1
  CX7-1 -------- cable 3 ------- CX7-0
  rank3                         rank2
```

```text
cable 1: rank0 CX7-0 <-> rank1 CX7-1
cable 2: rank1 CX7-0 <-> rank2 CX7-1
cable 3: rank2 CX7-0 <-> rank3 CX7-1
cable 4: rank3 CX7-0 <-> rank0 CX7-1
```

Use a separate `/24` per cable, or one `/24` with `NCCL_IB_SUBNET_PREFIX_LEN=24` and
`NCCL_IB_SUBNET_AWARE_ROUTING=1` (the default this switch relies on). Keep the
management LAN on a different interface (`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`)
— the ring carries the data plane only.

## Setup

1. Build/install the patched NCCL into `NCCL_HOST_DIR` (default `$HOME/nccl-2.30.7`)
   on **every** node. The library must contain the `SWITCHLESS_RING_ONLY` marker;
   the preflight rejects a stock build, so a wrong library fails before any
   container is replaced rather than at NCCL init.

2. Put a local checkpoint on every node and set `NFS_SHARE=0`. A ring has no
   fabric-wide NFS path, so the tested configuration keeps its own copy of the
   weights on every Spark — see [Local weights](#local-weights).

3. Uncomment the ring block in `.env.tp4`:

   ```ini
   NCCL_SWITCHLESS_RING_ONLY=1
   NCCL_ALGO=Ring
   NCCL_P2P_LEVEL=SYS
   ```

4. Validate, then serve:

   ```bash
   ./start.sh doctor
   ./start-tp4.sh serve
   ```

## Local weights

`NFS_SHARE=1` (the default) exports the head's checkpoint and gives every worker an
NFS-backed docker volume. On a ring that is the wrong shape: the share would ride the
same four DACs the ring needs for collectives, opposite nodes would reach the head
through a transit node, and a dead exporter takes the whole serve down. `NFS_SHARE=0`
keeps `$NFS_VOLUME` (default `dsv41-weights`) local on each node instead.

It is a pre-existing switch in this repo, extended here to cover the whole profile.

### What it turns off

With `NFS_SHARE=0` none of the NFS apparatus is touched:

* no exporter container on the head, no `nfs_publish_model`, no `/etc/exports` rewrite;
* no per-worker docker volume creation, so no `nfs-common` on the workers and no client
  ACL to get wrong — the ACL is the usual reason a worker cannot see the checkpoint;
* `NFS_SERVER_IP`, `NFS_SERVER_IPS`, `WORKER_FABRIC_IPS` and the `NFS_SERVER_IP_<n>`
  legacy variables become unused. Nothing reads them, so leaving them set is harmless;
* `$COMMON_MODEL` (the head's symlink to `MODEL_DIR`) is still created, but nothing
  consumes it on a worker any more — see below.

`cmd_share` returns immediately under `NFS_SHARE=0`. It stays idempotent, so `serve` can
keep calling it unconditionally without knowing which profile it is in.

### What it checks, and when

`serve` validates **before** it replaces containers — a half-provisioned node otherwise
fails twenty minutes into a load, with four containers already down:

* the head: a readable, non-empty `config.json` plus at least `EXPECTED_SHARDS` readable,
  non-empty shards (`local_model_has_weights`);
* every worker: the same, inside its own `$NFS_VOLUME` (`nfs_worker_has_model`).

`doctor` runs the same per-worker check in **both** profiles, so a missing node shows up
in the pre-flight rather than in `serve`. It reports the weights source once:

```
[+] weights: NFS_SHARE=0 — head reads $HOME/NewModels/DeepSeek-V4.1-Flash; workers use local volume dsv41-weights
```

### What it fixed

* **`status` read the wrong path.** It tested `$COMMON_MODEL/config.json` on each worker,
  but `$COMMON_MODEL` is a **head-side symlink to `MODEL_DIR`** and never exists on a
  worker, so healthy workers reported `weights:MISSING` in this profile. It now inspects
  the worker's own volume and prints `weights:OK (dsv41-weights)`.
* **The probe could create what it was checking.** `docker run -v $NFS_VOLUME:/m` creates
  an empty volume on a typo, and the old probe pulled a mutable `alpine:latest` to do it.
  It now inspects the volume first, then runs the serving image already required by
  `serve` with `--pull=never --network none`, needs neither network nor GPUs, and checks
  the shard count instead of only `config.json`.

### Migrating from `NFS_SHARE=1`

A leftover NFS-backed volume **keeps the same name** and still contains `config.json`, so
the existence probe passes and the worker quietly keeps reading over NFS — the exact
thing this profile exists to avoid, and it fails confusingly the moment the exporter is
gone. `nfs_unmount_workers()` exists but is not wired to a command, so nothing removes it
for you. `serve` therefore refuses it outright:

```
$ grep -q '"type":"nfs"' <<< "$(docker volume inspect -f '{{json .Options}}' dsv41-weights)"
$ ssh spark2 docker volume rm dsv41-weights
[+] spark2: weights:OK (dsv41-weights)      # after copying the checkpoint locally
```

`Driver` cannot tell the two apart — a plain volume and an NFS volume both report
`local`. The mount options can: a plain volume reports `null`, an NFS one reports
`{"type":"nfs",...}`.

Provisioning is then a plain copy per node, for example from the head:

```bash
rsync -a --info=progress2 $MODEL_DIR/ spark2:$MODEL_DIR/
./start.sh doctor          # one weights line per worker
./start-tp4.sh serve
```

Two consequences worth planning for: the checkpoint costs **4× the disk** (476 GiB per
node here), and a model update has to be pushed to all four nodes before the next
`serve` — a node left behind fails the content check rather than serving stale weights.

## What the switch does

With `NCCL_SWITCHLESS_RING_ONLY=1`:

* injects `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`,
  `NCCL_SKIP_TREE_CONNECT=1`, `NCCL_IB_SUBNET_PREFIX_LEN=24`,
  `NCCL_MIN_NCHANNELS=4` and `NCCL_P2P_LEVEL=SYS` into the head **and every worker**
  (each value still overridable from the env file);
* mounts the patched library **over** the image's pip NCCL
  (`NCCL_PIP_SO`, default `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`)
  instead of putting it on `LD_LIBRARY_PATH`. This matters if you use DeepEP: two
  visible NCCL runtimes make `check_nccl_so()` abort before NCCL is initialised;
* validates the configuration and every rank's HCA/GID in `doctor`, and fatally in
  `serve` — the check sits before any container is replaced.

`NCCL_OVERLAY_PIP` defaults to following the switch, so the overlay can also be
enabled on a switched fabric on its own (`NCCL_OVERLAY_PIP=1`, switch off).

## Expected logs

```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
```

`doctor` prints one line per rank:

```
[+] switchless ring: config OK (NNODES=4 TP=4 EP=2, IB_HCA=rocep1s0f0,rocep1s0f1)
[+] switchless ring: head preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.2 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.3 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.4 preflight OK (RoCEv2 GID index 3)
```

## Pitfalls

* **`EP_SIZE` is free, `TP_SIZE` is not.** The ring needs `NNODES == TP_SIZE == 4`
  (the ring spans the tensor-parallel group). `EP_SIZE` only decides how the MoE
  all-to-all is grouped, so `1 <= EP_SIZE <= TP_SIZE` is accepted and `EP_SIZE=2`
  is a common choice on this fabric.
* **A wrong GID index is the usual failure.** All ports listed in `IB_HCA` must
  share one nonzero IPv4-mapped RoCE v2 GID; the preflight finds it or validates
  your `NCCL_IB_GID_INDEX` override. Management IPs need not appear in the GID table.
* **Do not also set `NCCL_SWITCHLESS_RING_ONLY=1` with a switched fabric.** The ring
  skips the tree, which is a performance loss when the tree is reachable.
* **A ring is not a non-blocking fabric.** Opposite ranks talk through a transit
  node, so a four-node ring's bisection bandwidth is one link, not two. Expect the
  decode numbers below rather than the switched ones.

## Measured

Four GB10 Sparks in a ring (`a-b-c-d-a`, no switch), TP4 / EP2, 1M context,
DSpark k=5, local weights (`NFS_SHARE=0`), canary image, this switch on. sparkDash's
benchmark panel, one engine, no other load.

Boot, first time:

```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
parallel: nnodes=4 TP=4 EP=2
```

All four ranks healthy, `doctor: ready`, `/health` 200, `--enable-cache-report` live
(cold request `prompt_tokens_details: None`, warm `{'cached_tokens': 1024}`).

### Prefill, cold

| context | 1k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|
| prompt tokens | 1,041 | 4,116 | 8,213 | 16,405 | 32,793 | 65,555 |
| TTFT | 406 ms | 1.11 s | 2.01 s | 3.89 s | 7.67 s | 15.43 s |
| tok/s | 2563 | 3712 | 4081 | 4218 | 4274 | 4249 |

**Caveat, the same one as the README's prefill table:** sparkDash's prefill filler is one
repeated token, so every filler token hits the same Engram row and the row cache
(`DSV41_CACHE_GIB`) inflates the 16k-128k column by roughly 9-20 % (reported by
koldfrontier in MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks#21). Treat the shape as real and
the absolute numbers as an upper bound; a cold single request with a 53,613-token natural
prompt took **17 s** on the same boot, which is the same order as the 32k row above.

### Decode, 400 output tokens

Prose:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 60.5 | 82.9 | 116.4 | 186.1 |
| per stream | 60.5 | 41.4 | 30.3 | 24.0 |
| TTFT | 182 ms | 209 ms | 263 ms | 286 ms |

Code:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 107.5 | 189.6 | 326.4 | 521.6 |
| per stream | 107.5 | 94.8 | 81.6 | 65.2 |
| TTFT | 257 ms | 313 ms | 381 ms | 533 ms |

Decode is where a ring is the right trade: aggregate scales close to linearly through
eight streams (prose 60.5 → 186.1, code 107.5 → 521.6) while per-stream decay stays
gentle, which is what the 16-slot decoder and DSpark are for. Prefill is where the
bisection shows: ranks on opposite sides of the ring talk through a transit node, so a
four-node ring's bisection is one link, not two. That is the price of having no switch,
not a way to beat one.

### Reference

The ring configuration was originally contributed as
[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks #3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)
by [@Saolence](https://github.com/Saolence), carried forward with an all-rank preflight,
the overlay-mount strategy and regression fixtures in
[#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19). The NCCL
transport patch itself is from
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
