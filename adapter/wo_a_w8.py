"""wo_a from its fp8 checkpoint bytes instead of the bf16 copy, same Triton tiling. Gated on DSV41_WO_A_W8.

The checkpoint ships wo_a as e4m3 with 32x32 ue8m0 block scales. The engine dequantizes it to bf16
at load (its fp8 absorb path needs 128x128 blocks) and the verify/draft small-batch kernel
`_wo_a_partial` then streams 16.8 MB per layer and rank. This adapter keeps an fp8 twin of every
wo_a (e4m3 values plus one exponent per 32x32 block), built after load and kept only when it
reconstructs the bf16 weight exactly, and runs a copy of `_wo_a_partial` that loads the e4m3 tile
and its exponent and rebuilds the same bf16 tile in registers before the same `tl.dot`.

Half the weight bytes for the same operand values. Measured on GB10 (TP4 shape, 43 layers, M=6):
3.43 -> 2.20 ms; the MXFP8-epilogue path (what production runs) is bitwise identical to the stock
kernel; the plain bf16 path differs only in the MMA's fp32 accumulation order (1 bf16 ulp on
0.016 % of outputs). Only 2 <= M <= 8 rows take the new kernel (verify at bs=1, the draft block);
every other shape and any layer whose twin is not exact stays on the stock path.
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("DSV41_WO_A_W8", "0").strip() not in ("0", "", "off", "false")
_TWINS = {}          # bf16 weight data_ptr -> (e4m3 [2,1024,4096], exponent uint8 [2,32,128])


@triton.jit
def _wo_a_partial_w8(X, W8, S, P, M: tl.constexpr, SX: tl.constexpr):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        w8 = tl.load(W8 + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        e = tl.load(S + (group * 32 + n[None, :] // 32) * 128 + offsets[:, None] // 32)
        w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


def make_twin(w: torch.Tensor):
    """bf16 [2, 1024, 4096] -> (e4m3, exponent) or None if the twin does not reproduce w exactly."""
    g, r, d = w.shape
    wf = w.float().view(g, r // 32, 32, d // 32, 32)
    amax = wf.abs().amax(dim=(2, 4)).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(e)[:, :, None, :, None]
    q = (wf / scale).to(torch.float8_e4m3fn)
    back = (q.float() * scale).to(torch.bfloat16).view(g, r, d)
    if not torch.equal(back, w):
        return None
    return q.view(g, r, d).contiguous(), (e + 127).to(torch.uint8).contiguous()


def _partial_w8(x, twin, m):
    w8, s = twin
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial_w8[(16, 2, 8)](x, w8, s, partial, m, x.stride(0), num_warps=4, num_stages=3)
    return partial


def _patch_kernels(dsv4_module, kernel_module):
    for name in ("wo_a_bf16_small_batch", "wo_a_bf16_small_batch_mxfp8", "_wo_a_reduce", "_quantize_partial"):
        if not hasattr(kernel_module, name):
            raise RuntimeError(f"DSV41_WO_A_W8: {kernel_module.__name__}.{name} is gone; engine drifted")
    orig_small = kernel_module.wo_a_bf16_small_batch
    orig_mx = kernel_module.wo_a_bf16_small_batch_mxfp8
    reduce_k = kernel_module._wo_a_reduce
    quantize_partial = kernel_module._quantize_partial

    def small(x, weight):
        twin = _TWINS.get(weight.data_ptr())
        if twin is None:
            return orig_small(x, weight)
        m = x.shape[0]
        result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
        reduce_k[(triton.cdiv(m * 2048, 256),)](_partial_w8(x, twin, m), result, m * 2048, num_warps=4)
        return result

    def small_mx(x, weight):
        twin = _TWINS.get(weight.data_ptr())
        if twin is None:
            return orig_mx(x, weight)
        return quantize_partial(_partial_w8(x, twin, x.shape[0]))

    kernel_module.wo_a_bf16_small_batch = small
    kernel_module.wo_a_bf16_small_batch_mxfp8 = small_mx
    # deepseek_v4 imported both names into its own namespace
    if getattr(dsv4_module, "wo_a_bf16_small_batch", None) is not orig_small or \
            getattr(dsv4_module, "wo_a_bf16_small_batch_mxfp8", None) is not orig_mx:
        raise RuntimeError("DSV41_WO_A_W8: deepseek_v4 no longer imports the wo_a small-batch kernels by name")
    dsv4_module.wo_a_bf16_small_batch = small
    dsv4_module.wo_a_bf16_small_batch_mxfp8 = small_mx


def build_twins(model: torch.nn.Module) -> tuple[int, int]:
    made = kept = 0
    for mod in model.modules():
        w = getattr(getattr(mod, "wo_a", None), "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.numel() != 2 * 1024 * 4096:
            continue
        w3 = w.data.view(2, 1024, 4096)
        twin = make_twin(w3)
        if twin is None:
            kept += 1
            continue
        _TWINS[w3.data_ptr()] = twin
        made += 1
    print(f"[wo_a_w8] fp8 twins for {made} wo_a weights, {kept} left on bf16", flush=True)
    return made, kept


def _wrap_load(cls):
    if getattr(cls, "_dsv41_wo_a_w8", False):
        return
    cls._dsv41_wo_a_w8 = True
    orig = cls.load_weights

    def load_weights(self, *a, **kw):
        out = orig(self, *a, **kw)
        build_twins(self)
        return out

    cls.load_weights = load_weights


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4 (target model; also owns the wo_a dispatch the draft uses)."""
    if not ENABLED:
        return
    import importlib
    _patch_kernels(dsv4_module, importlib.import_module("sglang.kernels.ops.attention.dsv4.wo_a_bf16"))
    _wrap_load(dsv4_module.DeepseekV4ForCausalLM)


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: twins for the draft's three wo_a as well."""
    if ENABLED:
        _wrap_load(dspark_module.DeepseekV4ForCausalLMDSpark)
