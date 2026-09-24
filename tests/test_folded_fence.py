"""CPU check for adapter/folded_result_fence.py with stand-ins that have the engine's interface."""
import os
import types
from dataclasses import dataclass

os.environ["DSV41_FOLDED_FENCE"] = "1"
import torch  # noqa: E402

import folded_result_fence as ff  # noqa: E402


@dataclass
class AcceptOuts:
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    out_tokens: torch.Tensor


BUF = {f: torch.arange(6) for f in ff.FIELDS}


class Executor:
    def accept_and_finalize(self, *, folded_accept, **kw):
        return AcceptOuts(**BUF) if folded_accept else AcceptOuts(**{f: torch.zeros(6) for f in ff.FIELDS})


def main():
    ff.install(types.SimpleNamespace(TargetVerifyExecutor=Executor, AcceptOuts=AcceptOuts))
    out = Executor().accept_and_finalize(folded_accept=True)
    for f in ff.FIELDS:
        assert torch.equal(getattr(out, f), BUF[f]) and getattr(out, f).data_ptr() != BUF[f].data_ptr()
    BUF["out_tokens"].fill_(-1)                   # the next replay overwrites the persistent buffer
    assert out.out_tokens.tolist() == list(range(6))
    eager = Executor().accept_and_finalize(folded_accept=False)
    assert eager.out_tokens.sum() == 0
    try:
        ff.install(types.SimpleNamespace())
    except RuntimeError:
        print("test_folded_fence: ok")
        return
    raise AssertionError("drifted module accepted")


if __name__ == "__main__":
    main()
