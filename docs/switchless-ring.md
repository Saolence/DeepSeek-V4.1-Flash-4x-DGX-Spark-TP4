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
same four DACs the ring needs for collectives, and opposite nodes would reach the head
through a transit node. `NFS_SHARE=0` keeps `$NFS_VOLUME` local on each node instead.
It is a pre-existing switch in this repo, extended here to cover the whole profile:

* `cmd_share` returns immediately — no exporter, no worker volumes, no symlink. It is
  idempotent, so `serve` can keep calling it unconditionally.
* `serve` validates **before** it replaces containers: a readable, non-empty
  `config.json` plus at least `EXPECTED_SHARDS` readable, non-empty shards on the head,
  and the same on every worker. A half-provisioned node otherwise fails twenty minutes
  into a load, with four containers already down.
* `status` reads each worker's own volume. It used to test `$COMMON_MODEL/config.json`,
  which is a **head-side symlink to `MODEL_DIR`** and never exists on a worker, so
  healthy workers reported `weights:MISSING` in this profile.
* `doctor` checks every worker's volume in both profiles, so a missing node shows up in
  the pre-flight rather than in `serve`.

The worker probe was also hardened for a fabric that may have no Docker Hub access:
it inspects the volume instead of `-v` (which silently **creates** an empty volume on a
typo), runs the serving image already required by `serve` with `--pull=never
--network none` instead of pulling `alpine`, and checks the shard count, not just
`config.json`.

Provisioning is a plain copy per node, for example from the head:

```bash
rsync -a --info=progress2 $MODEL_DIR/ spark2:$MODEL_DIR/
./start.sh doctor          # prints weights:OK / weights:MISSING per worker
./start-tp4.sh serve
```

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

On a four-node GB10 ring (TP4/EP2, 1M context, DSpark k=5, local weights,
`NFS_SHARE=0`), first boot with this switch on:

* `NCCL INFO Connected all rings` with the tree and PAT transports disabled,
  all four ranks healthy, `doctor: ready`;
* cold prefill of a 53,613-token prompt: **17 s**, head `MemAvailable` 14.3 GiB after.

The ring configuration was originally contributed as
[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks #3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)
by [@Saolence](https://github.com/Saolence), carried forward with an all-rank
preflight, the overlay-mount strategy and regression fixtures in
[#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19).
The NCCL transport patch itself is from
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
