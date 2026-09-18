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

### Names on the host

"CX7-0" and "CX7-1" are the diagram's names, not the OS's. Each card exposes two
functions, and both the interface and the RDMA device carry the slot and port:

| Diagram | Linux interface | RDMA device (`IB_HCA`) |
|---|---|---|
| first card, port 0 | `enp1s0f0np0` | `rocep1s0f0` |
| first card, port 1 | `enp1s0f1np1` | `rocep1s0f1` |
| second card, port 0 | `enP2p1s0f0np0` | `roceP2p1s0f0` |
| second card, port 1 | `enP2p1s0f1np1` | `roceP2p1s0f1` |

`IB_HCA` takes the RDMA device names, not the interfaces. On a node you have not seen
before, these two commands identify them:

```bash
ls /sys/class/infiniband/                                     # the RDMA devices
for d in /sys/class/infiniband/*; do echo -n "$d -> "; ls "$d/device/net"; done   # device -> interface
```

### Addressing

One `/24` per cable with `.10` and `.11` at the two ends is what this fleet runs, and it
is the shape `NCCL_IB_SUBNET_AWARE_ROUTING=1` with `NCCL_IB_SUBNET_PREFIX_LEN=24`
expects: a device is matched to the subnet of the peer it is talking to, and a rank only
needs routes for the two cables it is not attached to. One flat `/24` for all four cables
also works -- then set the prefix length to cover all eight ends and skip the routes.

| cable | between | subnet | rank0 | rank1 | rank2 | rank3 |
|---|---|---|---|---|---|---|
| 1 | rank0-rank1 | `10.9.0.0/24` | `10.9.0.10` | `10.9.0.11` | | |
| 2 | rank1-rank2 | `10.9.1.0/24` | | `10.9.1.10` | `10.9.1.11` | |
| 3 | rank2-rank3 | `10.9.2.0/24` | | | `10.9.2.10` | `10.9.2.11` |
| 4 | rank3-rank0 | `10.9.3.0/24` | `10.9.3.11` | | | `10.9.3.10` |

Rank0's two ports as an example; every rank mirrors it with its own neighbours:

```ini
# enp1s0f0np0 -- the cable-1 end, next hop rank1
addresses: [10.9.0.10/24]
routes: to 10.9.1.0/24 via 10.9.0.11
# enp1s0f1np1 -- the cable-4 end, next hop rank3
addresses: [10.9.3.11/24]
routes: to 10.9.2.0/24 via 10.9.3.10
```

MTU 9000 on every ring port. The subnets are placeholders -- one block per cable, with a
prefix length that matches the one NCCL is given. A second card, if you cable it, gets a
second block of four subnets and the same geometry, as in
[Devices past the second are never advertised](#devices-past-the-second-are-never-advertised).

## The patched NCCL

The switch needs a library carrying `sparkring`'s switchless-cycle change. Nothing in
this tree ships it, the image's own NCCL is a stock build, and the preflight refuses a
library without the marker -- so this is the one artefact the deployment has to bring.

| | |
|---|---|
| Where | `NCCL_HOST_DIR`, default `$HOME/nccl-2.30.7`, on **every** node: the head mounts it read-only into its container, and so does each worker |
| File name | `libnccl.so.2.30.7` or `libnccl.so.2`. Any other name is ignored without a warning, and the container silently falls back to the image's NCCL |
| Marker | `grep -qa SWITCHLESS_RING_ONLY libnccl.so.2.30.7` must succeed |
| GID | the port-1 RoCE v2 GID at `NCCL_IB_GID_INDEX` has to be populated and identical on every rank |

**Route A: the published native runtime.** `sparkring` ships a qualified aarch64 build;
this is the one this fleet runs, hashes included:

| | |
|---|---|
| Release | `native-runtime-sm121-aa8fa11831af`, asset `native-runtime-files-20260908.tar` |
| tar sha256 | `aa8fa11831afaa4539e0b74442fd1ea25dd8cf49ad5edb6fd95e7f05fdf5ce86` |
| `libnccl.so.2.30.7` sha256 | `768a450b5eb84bf3d1191795350e43c96de75aeba4783ec314d47672fe6e1fc6` |

```bash
mkdir -p ~/nccl-2.30.7 && cd ~/nccl-2.30.7
sha256sum native-runtime-files-20260908.tar     # compare against the tar hash above
tar -xf native-runtime-files-20260908.tar      # carries the qualified libnccl.so.2.30.7
ln -sf libnccl.so.2.30.7 libnccl.so.2
grep -qa SWITCHLESS_RING_ONLY libnccl.so.2.30.7 && echo 'marker ok'
```

**Route B: build it.** Base `NVIDIA/nccl` v2.30.7-1 (`73cf112295c33aee2b895f329f592f2a9b4b0f97`)
with `nccl-2.30.7-dual-pci-domain.patch` from
[sparkring](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/DUAL_PCI_DOMAIN.md),
applied **alone** -- it already contains the switchless-cycle change, and the two must
not be layered. Build the aarch64/SM121 target the way that document describes, then
install the result as `libnccl.so.2.30.7` plus the `libnccl.so.2` symlink.

**Checking it landed.** `doctor` runs the preflight on the head and on every worker
before any container is replaced:

```text
[+] switchless ring: config OK (NNODES=4 TP=4 EP=2, IB_HCA=rocep1s0f0,rocep1s0f1)
[+] switchless ring: head preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.2 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.3 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.4 preflight OK (RoCEv2 GID index 3)
```

One caveat before customising it: the head honours `NCCL_HOST_DIR`, but `serve`'s worker
heredoc still looks for `$HOME/nccl-2.30.7` on the worker and passes it through
`LD_LIBRARY_PATH` instead of overlaying the pip library the way the head does
(`start.sh:750-754` against `:79`). Keeping the default path on every node is what makes
the two agree; a worker that finds nothing there starts on the image's NCCL with no
warning. `doctor`'s worker preflight is not affected -- it uses `NCCL_WORKER_DIR`
(`files/nccl.sh:228`), so it can pass while `serve` takes the other path.

## Setup

1. Install the patched NCCL from [The patched NCCL](#the-patched-nccl) into
   `NCCL_HOST_DIR` (default `$HOME/nccl-2.30.7`) on **every** node. The preflight
   rejects a stock build, so a wrong library fails before any container is replaced
   rather than at NCCL init.

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

## Devices past the second are never advertised

`NCCL_IB_HCA` accepts any number of devices, and NCCL says nothing when it cannot
use them. The switchless-cycle patch publishes at most **two** listener GIDs per
rank — `gidSlot < 2` in `net_ib/connect.cc`, in both the original and the patched
loop. Devices after the second are therefore absent from the handle every peer
receives, no peer can match their subnet, and their ports carry zero bytes. The
ring still forms and serves; it just runs on the first two devices.

The symptom is easy to miss because the channel plan looks right:

```
NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE \
                   [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE [RO]
NCCL INFO Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/2
```

Both lines name device 2, and it still moves nothing. What actually happens is a
silent collapse onto the first device, visible only in the routing log:

```
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 2 with dev 0
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 3 with dev 0
```

`doctor` now says so instead of leaving it to be discovered by counters:

```
warning: IB_HCA lists 4 devices but listener GID publication is capped at 2
warning: only rocep1s0f0 and rocep1s0f1 can be selected by a peer; the rest stay at zero
warning: set NCCL_IB_EXTENDED_IPV4_GIDS=1 with a dual-PCI-domain NCCL build
```

### Raising the cap

A four-Spark board exposes its ConnectX-7 functions through **two PCI root
domains** (`0000:` and `0002:` here), which NCCL discovers as four separate
devices:

```
[0] rocep1s0f0    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[1] rocep1s0f1    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[2] roceP2p1s0f0  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
[3] roceP2p1s0f1  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
```

FujitsuPolycom/sparkring's cumulative
[`nccl-2.30.7-dual-pci-domain.patch`](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/DUAL_PCI_DOMAIN.md)
raises the bound to four behind a flag, and adds a fallback that substitutes a
device **within the same PCI root** rather than collapsing across domains. Apply
it **alone** — it already contains the switchless-cycle changes, so it must not be
layered over them.

```ini
NCCL_IB_EXTENDED_IPV4_GIDS=1     # publish up to four IPv4-mapped listener GIDs
NCCL_IB_PRESERVE_PCI_DOMAIN=1    # substitute within the selected PCI root
NCCL_IB_ROUTE_DIAGNOSTICS=1      # one record per final QP: which device it landed on
NCCL_IB_QPS_PER_CONNECTION=1
```

Set `IB_HCA` to all four devices and the ring uses both planes. Every value has to
reach every rank, head and workers alike.

The flags are read at NCCL init, so the effect is visible before any request:

```
NCCL INFO NET/IB ListenerRouting format=ipv4-v1 advertised=4 observed=4
```

`advertised=2` means the cap is still in force. The routing records then stop
collapsing: `overriding dev 3 with dev 2` stays inside the second PCI root instead
of reaching for `dev 0`.

### Channel count

`NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` decide how many channels share the
devices. Four channels over four devices gives one channel per device, which is
the mapping that reaches all of them:

```ini
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
```

Eight channels over four devices still round-robins 0,1,2,3,0,1,2,3, so it is not
wrong, but a four-versus-eight comparison on this workload found no serving
benefit and 0.14 GiB more head-node shared memory
([sparkring#193](https://github.com/FujitsuPolycom/sparkring/issues/193)).

### What it is worth

Measured here on four Sparks, TP4 / EP2, DSpark k=5, one 64k prefill plus 16
concurrent streams, IB port counters before and after:

| port | PCI root | before | after |
|---|---|---:|---:|
| `rocep1s0f0` | 0000 | 65.45 GB | 32.20 GB |
| `rocep1s0f1` | 0000 | 65.45 GB | 32.19 GB |
| `roceP2p1s0f0` | 0002 | **0.00 GB** | **31.80 GB** |
| `roceP2p1s0f1` | 0002 | **0.00 GB** | **31.80 GB** |

Half the traffic moves to the second plane. The total is unchanged — this spreads
the same collectives over twice the ports, it does not make them smaller. The
ported case is bounded by what the ring was waiting on, not by cable bandwidth:
the ports ran at roughly 5 % of line rate under this load, so expect a low
single-digit prefill gain and no decode change, matching the
[contributor measurement](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/transport/nccl-dual-domain-deepseek.md)
of +5.43–6.82 % prefill for this exact model and runtime. Verify with counters and
the routing records rather than trusting the channel plan.

### Diagnosing it

`ListenerRouting` and the routing records are logged at the `NET` level. With
`NCCL_DEBUG_SUBSYS=INIT,ENV` they never appear and the collapse is invisible:

```ini
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV,NET    # add NET while validating; drop it afterwards
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
