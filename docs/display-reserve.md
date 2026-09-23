# Engram row cache in the GB10 display reservation (optional)

GB10 firmware sets aside roughly 2 GiB of the unified memory for display scanout. The CUDA
allocator and `MemAvailable` never see it, so on a headless Spark it is idle. With the NVIDIA DRM
driver in modeset mode, a DRM "dumb" buffer of up to 2032 MiB can be created on `/dev/dri/card0`
and is carved from that reservation. The idea comes from coolbho3k's two-Spark DeepSeek recipe
(and MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks#234, which uses it for KV); the implementation
here is separate and much smaller.

Here it backs the row cache of one Engram layer (`adapter/row_store.cpp`,
`row_store_next_cache_on_drm`). The cache is only ever touched by the CPU (rows are staged to the
GPU through pinned buffers), so no CUDA registration is needed. That layer's cache then no longer
grows into ordinary memory as it fills: about 1.8 GiB of runtime headroom per node, which matters
on the head node (its free memory also bounds the KV pool). The cache is anonymous memory that
fills lazily, so the KV pool computed at boot does not change.

Measured on this fleet (driver 580.173.02): with the pool backing the layer-1 cache, verify step
40.5 ms (prose) / 46.7 ms (code) against 40.9 / 47.2 without, Engram hit rate unchanged (72.4 %
on both layers). CPU reads from the mapping are uncached (~1 GB/s against 28 GB/s), writes run at
full speed; a decode step touches a few hundred 264-byte rows, so it does not show.

Fresh clone of this repository with the option on: greedy outputs byte-identical to the ordinary cache
(three runs), sparkDash prose c1 69.6, code c1 115.3; qeval 71 and 72 of 75 (the one extra miss,
`reason_r12`, passed on the repeat and in every earlier run).

## Host setup (every node, once)

DGX OS ships `/etc/modprobe.d/zz-nvidia-drm-override.conf` with `modeset=0`, which disables dumb
buffers, and boots into the desktop, which competes for the same region.

```bash
sudo cp /etc/modprobe.d/zz-nvidia-drm-override.conf /root/zz-nvidia-drm-override.conf.bak
echo 'options nvidia-drm modeset=1 fbdev=0' | sudo tee /etc/modprobe.d/zz-nvidia-drm-override.conf
sudo update-initramfs -u -k all
sudo systemctl set-default multi-user.target      # headless: no gdm / display-manager
sudo reboot
# check: Y and N
sudo cat /sys/module/nvidia_drm/parameters/modeset /sys/module/nvidia_drm/parameters/fbdev
```

Revert: restore the backup, `update-initramfs -u -k all`, `systemctl set-default graphical.target`,
reboot. Going headless alone raised this fleet's first KV pool after the reboot to 7.48 M tokens
(5.9-6.4 M on the desktop target the same day).

## Enable

```
EXTRA_CONTAINER_ENV="... DSV41_ENGRAM_DRM_NODE=/dev/dri/card0"
# optional: DSV41_ENGRAM_DRM_MIB=1792 (multiple of 16, at most 2032), DSV41_ENGRAM_DRM_LAYER=<engram layer id>
```

The containers already run `--privileged`, so the device node is visible. Look for
`Engram: row cache of ... bytes on the display reservation (/dev/dri/card0)` in the boot log. If the
node is missing or the buffer cannot be created, the store falls back to ordinary memory and logs why.
Qualified on driver 580.173.02 only; the technique is reported not to work on 595.84.
