"""CPU checks for adapter/verify_cap.py and block_verify's live= path.

1. Exactness: on a toy language model (V=5, gamma=3), block verification with a per-step verify
   length chosen by a stopping rule (include draft j+1 iff prod_{i<=j} q_i(X_i) >= THR, i.e. it
   never looks at the draft it drops) still reproduces the target's joint distribution (chi-square).
2. The dead-row map sends rows past each request's live length to that request's anchor row.
3. The patches refuse drifted engine classes.
"""
import itertools
import math
import os
import types

os.environ["DSV41_VERIFY_CAP"] = "3"
import torch  # noqa: E402

import block_verify as bv  # noqa: E402
import verify_cap as vc  # noqa: E402

V, G = 5, 3
N = int(os.environ.get("VC_TEST_N", "150000"))
THR = 0.35
gen = torch.Generator().manual_seed(1)
table_p, table_q = {}, {}


def p_of(prefix):
    key = tuple(prefix)
    if key not in table_p:
        table_p[key] = torch.softmax(torch.randn(V, generator=gen) * 1.5, -1)
    return table_p[key]


def q_of(prefix):
    key = tuple(prefix)
    if key not in table_q:
        table_q[key] = torch.softmax(torch.log(p_of(prefix)) + 0.8 * torch.randn(V, generator=gen), -1)
    return table_q[key]


def run(capped):
    out = torch.zeros(N, 3, dtype=torch.long)
    produced = torch.zeros(N, dtype=torch.long)
    seqs = [[] for _ in range(N)]
    acc = steps = 0
    while (produced < 3).any():
        act = [i for i in range(N) if produced[i] < 3]
        bs = len(act)
        X = torch.zeros(bs, G, dtype=torch.long)
        TP = torch.zeros(bs, G + 1, V)
        DP = torch.zeros(bs, G, V)
        for r, i in enumerate(act):
            pre = list(seqs[i])
            for j in range(G):
                DP[r, j], TP[r, j] = q_of(pre), p_of(pre)
                X[r, j] = int(torch.multinomial(DP[r, j], 1, generator=gen))
                pre.append(int(X[r, j]))
            TP[r, G] = p_of(pre)
        live = None
        if capped:
            cum = torch.cumprod(DP.gather(-1, X.unsqueeze(-1)).squeeze(-1), 1)
            live = 1 + (cum[:, :G - 1] >= THR).long().cumprod(1).sum(1)
        tau, corr = bv.block_accept_probs(X, TP, DP, live=live)
        acc += int(tau.sum())
        steps += bs
        for r, i in enumerate(act):
            for t in X[r, :int(tau[r])].tolist() + [int(corr[r])]:
                seqs[i].append(t)
                if produced[i] < 3:
                    out[i, produced[i]] = t
                    produced[i] += 1
    return out, acc / steps


def chi2_z(out):
    counts = torch.zeros(V, V, V)
    for a, b, c in out.tolist():
        counts[a, b, c] += 1
    stat, dof = 0.0, 0
    for a, b, c in itertools.product(range(V), repeat=3):
        e = N * float(p_of([])[a] * p_of([a])[b] * p_of([a, b])[c])
        if e >= 5:
            stat += (counts[a, b, c] - e) ** 2 / e
            dof += 1
    dof -= 1
    return (stat - dof) / math.sqrt(2 * dof)


def main():
    full, acc_full = run(False)
    cut, acc_cut = run(True)
    z_full, z_cut = chi2_z(full), chi2_z(cut)
    print(f"block: {acc_full:.4f} drafts/step (z={z_full:+.2f}); stopping-rule cut: {acc_cut:.4f} (z={z_cut:+.2f})")
    assert abs(z_full) < 4.5 and abs(z_cut) < 4.5, "output distribution differs from the target"
    assert acc_cut < acc_full, "the cut did not shorten any block"

    vc._state["live"] = torch.tensor([6, 2, 4] + [6] * (vc.MAX_BS - 3))
    vc._state["src"] = None
    src = vc._src_rows(18, "cpu").tolist()
    assert src[:6] == list(range(6))
    assert src[6:12] == [6, 7, 6, 6, 6, 6]
    assert src[12:18] == [12, 13, 14, 15, 12, 12]
    w, i = torch.rand(18, 6), torch.randint(0, 384, (18, 6))
    vc._state["src"] = None
    kernel, vc._remap_kernel = vc._remap_kernel, None      # CPU: the index_select path
    w2, i2 = vc.remap_dead_rows(w, i)
    vc._remap_kernel = kernel
    assert torch.equal(i2[8], i[6]) and torch.equal(w2[17], w[12]) and torch.equal(i2[13], i[13])

    for fn, arg in ((vc.install_verify, types.SimpleNamespace()), (vc.install_model, types.SimpleNamespace())):
        try:
            fn(arg)
        except AttributeError:
            continue
        raise AssertionError(f"{fn.__name__} accepted a drifted module")
    print("test_verify_cap: ok")


if __name__ == "__main__":
    main()
