"""CPU checks for adapter/wo_a_w8.py: twin exactness, rejection of non-representable weights, and
the kernel patch (including the drift guard). The GPU kernel itself is checked on the Spark."""
import os
import types

os.environ["DSV41_WO_A_W8"] = "1"
import torch  # noqa: E402

import wo_a_w8  # noqa: E402


def checkpoint_like(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(2, 1024, 4096, generator=g) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(-14, -6, (2, 32, 128), generator=g).float()
    return (q.float().view(2, 32, 32, 128, 32) * torch.exp2(e)[:, :, None, :, None]).view(2, 1024, 4096).to(torch.bfloat16)


def test_twin_is_exact():
    w = checkpoint_like()
    q, s = wo_a_w8.make_twin(w)
    back = (q.float().view(2, 32, 32, 128, 32) * torch.exp2(s.float() - 127)[:, :, None, :, None]).view(2, 1024, 4096)
    assert torch.equal(back.to(torch.bfloat16), w)


def test_non_representable_is_rejected():
    w = torch.randn(2, 1024, 4096).to(torch.bfloat16)      # bf16 mantissas e4m3 cannot hold
    assert wo_a_w8.make_twin(w) is None


def test_patch_and_drift_guard():
    def small(x, w):
        return "stock"

    def small_mx(x, w):
        return "stock-mx"

    kern = types.SimpleNamespace(wo_a_bf16_small_batch=small, wo_a_bf16_small_batch_mxfp8=small_mx,
                                 _wo_a_reduce=None, _quantize_partial=None, __name__="k")
    model = types.SimpleNamespace(wo_a_bf16_small_batch=small, wo_a_bf16_small_batch_mxfp8=small_mx)
    wo_a_w8._patch_kernels(model, kern)
    assert model.wo_a_bf16_small_batch is kern.wo_a_bf16_small_batch is not small
    # a weight without a twin still goes to the stock kernel
    assert model.wo_a_bf16_small_batch(torch.zeros(6, 2, 4096), torch.zeros(1)) == "stock"
    drifted = types.SimpleNamespace(wo_a_bf16_small_batch=small, _wo_a_reduce=None, _quantize_partial=None, __name__="k")
    try:
        wo_a_w8._patch_kernels(types.SimpleNamespace(), drifted)
    except RuntimeError:
        return
    raise AssertionError("drifted kernel module was accepted")


if __name__ == "__main__":
    test_twin_is_exact()
    test_non_representable_is_rejected()
    test_patch_and_drift_guard()
    print("test_wo_a_w8: ok")
