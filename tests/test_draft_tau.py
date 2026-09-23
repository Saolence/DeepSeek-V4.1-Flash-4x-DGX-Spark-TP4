"""CPU check for adapter/draft_tau.py against a stand-in sampler with the engine's interface."""
import os
import types

os.environ["DSV41_DRAFT_TAU"] = "0.8"
import torch  # noqa: E402

import draft_tau  # noqa: E402


class FakeSampler:
    def __init__(self, folded=True):
        self.folded_sampling = folded
        self.temperatures = torch.ones(4) if folded else None

    def stage_sampling_params(self, *, bs, sampling_info):
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            return
        self.temperatures[:bs].copy_(sampling_info.temperatures[:bs])


def main():
    mod = types.SimpleNamespace(DsparkDraftSampler=FakeSampler)
    draft_tau.install(mod)
    s = FakeSampler()
    info = types.SimpleNamespace(temperatures=torch.tensor([1.0, 0.7, 1e-5, 1.0]))
    s.stage_sampling_params(bs=3, sampling_info=info)
    assert torch.allclose(s.temperatures, torch.tensor([0.8, 0.56, 0.8e-5, 1.0]))
    s.stage_sampling_params(bs=2, sampling_info=None)        # all-greedy batch: untouched
    assert torch.equal(s.temperatures[:2], torch.ones(2))
    try:
        draft_tau.install(types.SimpleNamespace())
    except RuntimeError:
        print("test_draft_tau: ok")
        return
    raise AssertionError("missing sampler class was accepted")


if __name__ == "__main__":
    main()
