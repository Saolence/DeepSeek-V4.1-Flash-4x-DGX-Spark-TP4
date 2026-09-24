"""CPU check for adapter/draft_head_fp8.py: the fp8 copy reconstructs the head within fp8 rounding,
and the patch refuses a drifted draft class. The Triton kernel itself is checked on the Spark."""
import os
import types

os.environ["DSV41_DRAFT_HEAD_FP8"] = "1"
import torch  # noqa: E402

import draft_head_fp8 as dh  # noqa: E402


def main():
    w = (torch.randn(1000, 256) * 0.02).to(torch.bfloat16)          # 1000 rows: not a multiple of 32
    q, e = dh.make_twin(w)
    assert q.shape == (1000, 256) and e.shape == (32, 8)
    back = q.float().view(1000, 8, 32) * torch.exp2(e.float() - 127).repeat_interleave(32, 0)[:1000].unsqueeze(-1)
    rel = ((back.view(1000, 256) - w.float()).norm() / w.float().norm()).item()
    assert rel < 0.05, rel
    try:
        dh.install(types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=type("X", (), {})))
    except RuntimeError:
        print(f"test_draft_head_fp8: ok (rel err {rel:.3f})")
        return
    raise AssertionError("drifted class accepted")


if __name__ == "__main__":
    main()
